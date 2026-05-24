#!/usr/bin/env python3
"""RunPod / lokal: Qwen-Omni Audio-Transkription mit Timecodes & Dialog-Metadaten."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from audio_chunks import (
    AudioChunk,
    make_work_dir,
    prepare_master_wav,
    slice_chunks,
)
from export_results import write_outputs
from json_util import parse_json_response
from omni_model import OmniModelError, build_engine_from_env
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def _env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _progress_iter(chunks: List[AudioChunk], *, total: int, enabled: bool):
    if not enabled or total <= 0:
        yield from chunks
        return
    try:
        from tqdm import tqdm  # type: ignore

        yield from tqdm(
            chunks,
            total=total,
            unit="chunk",
            desc="Transkribiere",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
    except ImportError:
        yield from chunks


def _progress_chunk_done(
    chunk: AudioChunk,
    *,
    total: int,
    n_utterances: int,
    overlap: str,
    elapsed_s: float,
    enabled: bool,
) -> None:
    if enabled:
        try:
            from tqdm import tqdm  # type: ignore

            tqdm.write(
                f"      ✓ Chunk {chunk.index + 1}/{total} "
                f"[{chunk.start_sec:.1f}–{chunk.end_sec:.1f}s] "
                f"→ {n_utterances} Äußerung(en), overlap={overlap}, {elapsed_s:.1f}s"
            )
            return
        except ImportError:
            pass
    print(
        f"      Chunk {chunk.index + 1}/{total} "
        f"[{chunk.start_sec:.1f}–{chunk.end_sec:.1f}s] "
        f"→ {n_utterances} Äußerung(en), overlap={overlap}, "
        f"{elapsed_s:.1f}s",
        flush=True,
    )


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parent
    for p in (root / ".env", root.parent / ".env"):
        if not p.is_file():
            continue
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(p, override=False)
        except ImportError:
            for raw in p.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        break


def _parse_chunk_json(raw: str, chunk: AudioChunk) -> Dict[str, Any]:
    try:
        data = parse_json_response(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {
            "chunk_index": chunk.index,
            "chunk_start_sec": chunk.start_sec,
            "chunk_end_sec": chunk.end_sec,
            "audio_quality": {
                "overlap_speech": "unknown",
                "hard_to_understand": True,
                "notes": "Modell-Antwort war kein gültiges JSON",
            },
            "utterances": [],
            "parse_error": True,
            "raw_text": (raw or "")[:4000],
        }
    data.setdefault("chunk_index", chunk.index)
    data.setdefault("chunk_start_sec", chunk.start_sec)
    data.setdefault("chunk_end_sec", chunk.end_sec)
    return data


def run(
    *,
    audio: Path,
    out_dir: Path,
    chunk_sec: float,
    overlap_sec: float,
    keep_work: bool,
) -> int:
    audio = audio.resolve()
    out_dir = out_dir.resolve()
    if not audio.is_file():
        print(f"Audio nicht gefunden: {audio}", file=sys.stderr)
        return 2

    work_dir = make_work_dir()
    try:
        print(f"[1/4] Konvertiere Audio → 16 kHz mono WAV …")
        master = prepare_master_wav(audio, work_dir)
        chunks = slice_chunks(
            master, work_dir, chunk_sec=chunk_sec, overlap_sec=overlap_sec
        )
        if not chunks:
            print("Keine Audio-Chunks erzeugt (Datei zu kurz?).", file=sys.stderr)
            return 3
        print(f"      {len(chunks)} Chunk(s), je ~{chunk_sec}s (Overlap {overlap_sec}s)")

        print(f"[2/4] Lade Modell …")
        engine = build_engine_from_env()
        print(f"      {engine.model_id} (Familie: {engine._family})")
        engine.load()
        print(f"      Modell bereit.")

        chunk_results: List[Dict[str, Any]] = []
        raw_responses: List[Dict[str, Any]] = []
        total = len(chunks)
        use_tqdm = _env_bool("OMNI_PROGRESS", True)

        print(f"[3/4] Transkribiere {total} Chunk(s) …")
        chunk_iter = _progress_iter(chunks, total=total, enabled=use_tqdm)
        for chunk in chunk_iter:
            t0 = time.time()
            user_prompt = USER_PROMPT_TEMPLATE.format(
                source_name=audio.name,
                chunk_index=chunk.index + 1,
                chunk_total=total,
                chunk_start_sec=chunk.start_sec,
                chunk_end_sec=chunk.end_sec,
                chunk_duration_sec=chunk.duration_sec,
            )
            raw = engine.transcribe_clip(
                wav_path=chunk.path,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            elapsed = time.time() - t0
            parsed = _parse_chunk_json(raw, chunk)
            chunk_results.append(parsed)
            raw_responses.append(
                {
                    "chunk_index": chunk.index,
                    "elapsed_s": round(elapsed, 2),
                    "raw_text": raw,
                }
            )
            n_ut = len(parsed.get("utterances") or [])
            aq = parsed.get("audio_quality") or {}
            _progress_chunk_done(
                chunk,
                total=total,
                n_utterances=n_ut,
                overlap=str(aq.get("overlap_speech", "?")),
                elapsed_s=elapsed,
                enabled=use_tqdm,
            )

        print(f"[4/4] Export …")
        paths = write_outputs(
            out_dir=out_dir,
            source_audio=audio,
            model_id=engine.model_id,
            chunk_results=chunk_results,
            raw_responses=raw_responses,
        )
        print(f"      JSON:     {paths['json']}")
        print(f"      Excel:    {paths['xlsx']}")
        print(f"      SRT:      {paths['srt']}")
        print(f"      Dialog:   {paths['dialogue']}")
        return 0
    except OmniModelError as e:
        print(f"Modell-Fehler: {e}", file=sys.stderr)
        return 4
    except Exception as e:
        print(f"Fehler: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 5
    finally:
        if keep_work:
            print(f"Arbeitsordner behalten: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(
        description="Qwen-Omni Transkription (RunPod): Timecodes, Überlappung, Sprecherwechsel."
    )
    ap.add_argument(
        "--audio",
        "-a",
        required=True,
        help="Pfad zur Audio-Datei (wav, mp3, m4a, …)",
    )
    ap.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Ordner für JSON / Excel / SRT (Standard: ./output)",
    )
    ap.add_argument(
        "--chunk-sec",
        type=float,
        default=float(os.getenv("OMNI_CHUNK_SEC", "30")),
        help="Chunk-Länge in Sekunden (Standard: 30)",
    )
    ap.add_argument(
        "--overlap-sec",
        type=float,
        default=float(os.getenv("OMNI_OVERLAP_SEC", "2")),
        help="Überlappung zwischen Chunks (Standard: 2)",
    )
    ap.add_argument(
        "--keep-work",
        action="store_true",
        help="Temporäre WAV-Chunks nicht löschen",
    )
    args = ap.parse_args()
    raise SystemExit(
        run(
            audio=Path(args.audio),
            out_dir=Path(args.output_dir),
            chunk_sec=args.chunk_sec,
            overlap_sec=args.overlap_sec,
            keep_work=args.keep_work,
        )
    )


if __name__ == "__main__":
    main()
