from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioChunk:
    index: int
    path: Path
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg nicht gefunden. Auf RunPod installieren: apt-get update && apt-get install -y ffmpeg"
        )
    return exe


def probe_duration_sec(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe nicht gefunden (Teil von ffmpeg).")
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe fehlgeschlagen: {(proc.stderr or proc.stdout)[:300]}")
    return float((proc.stdout or "").strip())


def convert_to_wav_16k_mono(src: Path, dst: Path) -> Path:
    """Einheitliches WAV für Qwen-Omni-Utils."""
    ffmpeg = _require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg Konvertierung fehlgeschlagen: {(proc.stderr or '')[:500]}")
    return dst


def prepare_master_wav(audio_path: Path, work_dir: Path) -> Path:
    """MP3/M4A/… → 16 kHz mono WAV im Arbeitsordner."""
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(str(audio_path))
    master = work_dir / "master_16k_mono.wav"
    if audio_path.suffix.lower() == ".wav":
        # trotzdem normalisieren (Kanäle/Sample-Rate)
        return convert_to_wav_16k_mono(audio_path, master)
    return convert_to_wav_16k_mono(audio_path, master)


def slice_chunks(
    master_wav: Path,
    work_dir: Path,
    *,
    chunk_sec: float,
    overlap_sec: float,
) -> list[AudioChunk]:
    """Schneidet die Master-WAV in überlappende Clips."""
    if chunk_sec <= 0:
        raise ValueError("chunk_sec muss > 0 sein")
    overlap_sec = max(0.0, min(overlap_sec, chunk_sec * 0.5))
    step = max(0.1, chunk_sec - overlap_sec)
    total = probe_duration_sec(master_wav)
    ffmpeg = _require_ffmpeg()
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    out: list[AudioChunk] = []
    start = 0.0
    idx = 0
    while start < total - 0.05:
        end = min(total, start + chunk_sec)
        dur = end - start
        if dur < 0.25:
            break
        chunk_path = chunks_dir / f"chunk_{idx:04d}.wav"
        proc = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{dur:.3f}",
                "-i",
                str(master_wav),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(chunk_path),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Chunk-Schnitt fehlgeschlagen (idx={idx}): {(proc.stderr or '')[:400]}"
            )
        out.append(
            AudioChunk(index=idx, path=chunk_path, start_sec=start, end_sec=end)
        )
        idx += 1
        if end >= total - 0.05:
            break
        start += step
    return out


def make_work_dir(prefix: str = "qwen_omni_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))
