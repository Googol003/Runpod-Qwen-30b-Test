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
    estimate_chunk_count,
    make_work_dir,
    prepare_master_wav,
    probe_duration_sec,
    slice_chunks,
)
from export_results import write_outputs
from json_util import parse_json_response
from gpu_memory import clear_gpu_on_start, setup_cuda_allocator
from omni_model import (
    OmniModelError,
    apply_runpod_defaults,
    build_engine_from_env,
    load_plan_lines,
)
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

            extra = f", overlap={overlap}" if overlap else ""
            tqdm.write(
                f"      ✓ Chunk {chunk.index + 1}/{total} "
                f"[{chunk.start_sec:.1f}–{chunk.end_sec:.1f}s] "
                f"→ {n_utterances} Äußerung(en){extra}, {elapsed_s:.1f}s"
            )
            return
        except ImportError:
            pass
    print(
        f"      Chunk {chunk.index + 1}/{total} "
        f"[{chunk.start_sec:.1f}–{chunk.end_sec:.1f}s] "
        f"→ {n_utterances} Äußerung(en), {elapsed_s:.1f}s",
        flush=True,
    )


def max_new_tokens_for_chunk(duration_sec: float) -> int:
    """
    Antwortlänge pro 30s-Clip begrenzen (nicht 2048 für wenig Sprache).
    OMNI_MAX_NEW_TOKENS leer/auto → skaliert mit Clip-Dauer, Deckel 1024.
  """
    raw = (os.getenv("OMNI_MAX_NEW_TOKENS") or "").strip().lower()
    if raw and raw not in ("auto", ""):
        return max(128, int(raw))
    auto = int(duration_sec * 10) + 120
    return max(256, min(512, auto))


def _maybe_empty_cuda_cache() -> None:
    if not _env_bool("OMNI_EMPTY_CUDA", False):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parent
    for p in (root / ".env", root.parent / ".env"):
        if not p.is_file():
            continue
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(p, override=True)
        except ImportError:
            for raw in p.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k:
                    os.environ[k] = v
        break


def _bootstrap_env() -> None:
    setup_cuda_allocator()
    _load_dotenv()
    apply_runpod_defaults()


def _parse_chunk_json(raw: str, chunk: AudioChunk) -> Dict[str, Any]:
    try:
        data = parse_json_response(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {
            "chunk_index": chunk.index,
            "chunk_start_sec": chunk.start_sec,
            "chunk_end_sec": chunk.end_sec,
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
    max_new_tokens_cli: int | None = None,
) -> int:
    audio = audio.resolve()
    out_dir = out_dir.resolve()
    if not audio.is_file():
        print(f"Audio nicht gefunden: {audio}", file=sys.stderr)
        return 2

    print("[0] GPU-Speicher freigeben …")
    for line in clear_gpu_on_start():
        print(f"      {line}")

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
        audio_dur = probe_duration_sec(master)
        step = max(0.1, chunk_sec - max(0.0, min(overlap_sec, chunk_sec * 0.5)))
        expect = estimate_chunk_count(
            audio_dur, chunk_sec=chunk_sec, overlap_sec=overlap_sec
        )
        print(
            f"      Audio: {audio_dur / 60:.1f} min ({audio_dur:.0f} s) "
            f"→ {len(chunks)} Chunk(s) à ~{chunk_sec}s "
            f"(Schritt {step:.0f}s, Overlap {overlap_sec}s)"
        )
        if expect != len(chunks):
            print(f"      (erwartet laut Dauer: {expect} Chunks)")
        print(
            "      Pro Chunk nur ~30 s WAV ans Modell — nie die komplette Datei auf einmal."
        )
        if len(chunks) > 15:
            print(
                f"      Gesamt: {len(chunks)} Modell-Läufe nacheinander. "
                "Bei 30B + CPU-Offload kann das Stunden dauern — "
                "für 27 min eher Qwen2.5-Omni-7B auf GPU (OMNI_MODEL_ID)."
            )

        print(f"[2/4] Lade Modell … (unabhängig von Audiolänge — nur Gewichte in VRAM)")
        engine = build_engine_from_env()
        for line in load_plan_lines(engine.model_id):
            print(f"      {line}")
        engine.load()
        print(f"      Modell bereit.")
        for line in engine.device_report_lines():
            print(f"      {line}")

        chunk_results: List[Dict[str, Any]] = []
        raw_responses: List[Dict[str, Any]] = []
        total = len(chunks)
        use_tqdm = _env_bool("OMNI_PROGRESS", True)

        print(f"[3/4] Transkribiere {total} Chunk(s) …")
        chunk_iter = _progress_iter(chunks, total=total, enabled=use_tqdm)
        for chunk in chunk_iter:
            t0 = time.time()
            print(
                f"      → Chunk {chunk.index + 1}/{total} "
                f"[{chunk.start_sec:.1f}–{chunk.end_sec:.1f}s] Inferenz startet …",
                flush=True,
            )
            user_prompt = USER_PROMPT_TEMPLATE.format(
                source_name=audio.name,
                chunk_index=chunk.index + 1,
                chunk_total=total,
                chunk_start_sec=chunk.start_sec,
                chunk_end_sec=chunk.end_sec,
                chunk_duration_sec=chunk.duration_sec,
            )
            tok_limit = (
                max(128, max_new_tokens_cli)
                if max_new_tokens_cli is not None
                else max_new_tokens_for_chunk(chunk.duration_sec)
            )
            raw = engine.transcribe_clip(
                wav_path=chunk.path,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_new_tokens=tok_limit,
            )
            _maybe_empty_cuda_cache()
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
            _progress_chunk_done(
                chunk,
                total=total,
                n_utterances=n_ut,
                overlap="",
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
    _bootstrap_env()
    ap = argparse.ArgumentParser(
        description="Qwen-Omni Transkription (RunPod): Timecodes, Überlappung, Sprecherwechsel."
    )
    ap.add_argument(
        "--audio",
        "-a",
        default=None,
        help="Pfad zur Audio-Datei (wav, mp3, m4a, …); bei --check nicht nötig",
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
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max. generierte Tokens pro Chunk (sonst auto ~384–1024)",
    )
    ap.add_argument(
        "--keep-work",
        action="store_true",
        help="Temporäre WAV-Chunks nicht löschen",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Nur Modell-Konfiguration prüfen (optional Gewichte laden mit --check-load)",
    )
    ap.add_argument(
        "--check-load",
        action="store_true",
        help="Mit --check: Modell wirklich laden und GPU/Quantisierung verifizieren",
    )
    args = ap.parse_args()
    if args.check:
        print("=== Modell-Check (Hugging Face Qwen-Omni) ===")
        for line in clear_gpu_on_start():
            print(f"  {line}")
        engine = build_engine_from_env()
        for line in load_plan_lines(engine.model_id):
            print(f"  {line}")
        if args.check_load:
            print("\nLade Gewichte …")
            engine.load()
            for line in engine.device_report_lines():
                print(f"  {line}")
        else:
            print("\nTipp: --check-load lädt das Modell und prüft 4-bit / GPU-Offload.")
        raise SystemExit(0)
    if not args.audio:
        ap.error("--audio/-a ist erforderlich (außer mit --check).")
    if args.max_new_tokens is not None:
        os.environ["OMNI_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    raise SystemExit(
        run(
            audio=Path(args.audio),
            out_dir=Path(args.output_dir),
            chunk_sec=args.chunk_sec,
            overlap_sec=args.overlap_sec,
            keep_work=args.keep_work,
            max_new_tokens_cli=args.max_new_tokens,
        )
    )


if __name__ == "__main__":
    main()
