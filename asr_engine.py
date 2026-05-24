from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List


class ASRError(RuntimeError):
    pass


@dataclass
class WhisperSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass
class WhisperASR:
    """Schnelle lokale ASR pro Chunk (GPU), analog RunPod-Setup ohne HF-Omni."""

    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = "de"
    _model: Any = field(default=None, repr=False)

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ASRError(
                "faster-whisper fehlt. Install: pip install faster-whisper"
            ) from e
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, wav_path: Path) -> tuple[str, List[WhisperSegment]]:
        self.load()
        assert self._model is not None
        lang = self.language
        if lang in ("", "auto", "none"):
            lang = None
        segments_iter, _info = self._model.transcribe(
            str(wav_path.resolve()),
            language=lang,
            vad_filter=True,
            beam_size=5,
            word_timestamps=False,
        )
        out: List[WhisperSegment] = []
        lines: List[str] = []
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text:
                continue
            out.append(
                WhisperSegment(
                    start_sec=float(seg.start),
                    end_sec=float(seg.end),
                    text=text,
                )
            )
            lines.append(f"[{seg.start:.2f}s–{seg.end:.2f}s] {text}")
        transcript = "\n".join(lines) if lines else "(keine Sprache erkannt)"
        return transcript, out


def build_asr_from_env() -> WhisperASR:
    return WhisperASR(
        model_size=(os.getenv("WHISPER_MODEL") or "large-v3").strip(),
        device=(os.getenv("WHISPER_DEVICE") or "cuda").strip(),
        compute_type=(os.getenv("WHISPER_COMPUTE_TYPE") or "float16").strip(),
        language=(os.getenv("WHISPER_LANGUAGE") or "de").strip() or None,
    )
