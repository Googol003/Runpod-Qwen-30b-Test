#!/usr/bin/env python3
"""RunPod / lokal: Transkription mit Whisper-Chunks + Ollama (wie runpod_excel_corrector)."""

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
from asr_engine import ASRError, WhisperASR, build_asr_from_env
from export_results import write_outputs
from llm_clients import (
    LLMError,
    OllamaClient,
    build_client_from_env,
    parse_json_response,
)
from prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_OLLAMA,
    USER_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE_OLLAMA,
)


def _env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _backend() -> str:
    return (os.getenv("TRANSCRIBE_BACKEND") or "ollama").strip().lower()


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


def num_predict_for_chunk(duration_sec: float) -> int:
    raw = (os.getenv("OMNI_MAX_NEW_TOKENS") or os.getenv("LLM_NUM_PREDICT") or "").strip().lower()
    if raw and raw not in ("auto", ""):
        return max(128, int(raw))
    auto = int(duration_sec * 18) + 180
    return max(384, min(1024, auto))


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


def _call_ollama(
    client: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    num_predict: int,
    temperature: float,
) -> str:
    kwargs: Dict[str, Any] = {}
    num_ctx = int(os.getenv("LLM_NUM_CTX", "8192") or "8192")
    if isinstance(client, OllamaClient):
        kwargs["options"] = {"num_ctx": num_ctx, "num_predict": num_predict}
    return client.chat(system_prompt, user_prompt, temperature=temperature, **kwargs)


def _run_ollama(
    *,
    audio: Path,
    out_dir: Path,
    chunk_sec: float,
    overlap_sec: float,
    keep_work: bool,
    max_new_tokens_cli: int | None,
) -> int:
    work_dir = make_work_dir()
    try:
        print("[1/4] Konvertiere Audio → 16 kHz mono WAV …")
        master = prepare_master_wav(audio, work_dir)
        chunks = slice_chunks(
            master, work_dir, chunk_sec=chunk_sec, overlap_sec=overlap_sec
        )
        if not chunks:
            print("Keine Audio-Chunks erzeugt (Datei zu kurz?).", file=sys.stderr)
            return 3

        audio_dur = probe_duration_sec(master)
        step = max(0.1, chunk_sec - max(0.0, min(overlap_sec, chunk_sec * 0.5)))
        print(
            f"      Audio: {audio_dur / 60:.1f} min ({audio_dur:.0f} s) "
            f"→ {len(chunks)} Chunk(s) à ~{chunk_sec}s "
            f"(Schritt {step:.0f}s, Overlap {overlap_sec}s)"
        )
        print("      Pro Chunk: Whisper (ASR) → Ollama (JSON/Sprecher)")

        print("[2/4] Lade Whisper + Ollama …")
        asr = build_asr_from_env()
        asr.load()
        print(f"      Whisper: {asr.model_size} ({asr.device}, {asr.compute_type})")

        client = build_client_from_env()
        provider = type(client).__name__
        model = getattr(client, "model", "?")
        base_url = getattr(client, "base_url", "?")
        print(f"      LLM: {provider} {base_url} model={model}")
        if hasattr(client, "warmup"):
            t0w = time.time()
            client.warmup()
            print(f"      Warmup: {time.time() - t0w:.1f}s")

        temperature = float(os.getenv("LLM_TEMPERATURE", "0.05") or "0.05")
        chunk_results: List[Dict[str, Any]] = []
        raw_responses: List[Dict[str, Any]] = []
        total = len(chunks)
        use_tqdm = _env_bool("OMNI_PROGRESS", True)

        print(f"[3/4] Transkribiere {total} Chunk(s) …")
        for chunk in _progress_iter(chunks, total=total, enabled=use_tqdm):
            t0 = time.time()
            print(
                f"      → Chunk {chunk.index + 1}/{total} "
                f"[{chunk.start_sec:.1f}–{chunk.end_sec:.1f}s] …",
                flush=True,
            )
            whisper_text, _segs = asr.transcribe(chunk.path)
            user_prompt = USER_PROMPT_TEMPLATE_OLLAMA.format(
                source_name=audio.name,
                chunk_index=chunk.index + 1,
                chunk_total=total,
                chunk_start_sec=chunk.start_sec,
                chunk_end_sec=chunk.end_sec,
                chunk_duration_sec=chunk.duration_sec,
                whisper_transcript=whisper_text,
            )
            n_predict = (
                max(128, max_new_tokens_cli)
                if max_new_tokens_cli is not None
                else num_predict_for_chunk(chunk.duration_sec)
            )
            raw = _call_ollama(
                client,
                system_prompt=SYSTEM_PROMPT_OLLAMA,
                user_prompt=user_prompt,
                num_predict=n_predict,
                temperature=temperature,
            )
            elapsed = time.time() - t0
            parsed = _parse_chunk_json(raw, chunk)
            chunk_results.append(parsed)
            raw_responses.append(
                {
                    "chunk_index": chunk.index,
                    "elapsed_s": round(elapsed, 2),
                    "whisper_transcript": whisper_text[:8000],
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

        print("[4/4] Export …")
        model_label = f"whisper:{asr.model_size}+ollama:{model}"
        paths = write_outputs(
            out_dir=out_dir,
            source_audio=audio,
            model_id=model_label,
            chunk_results=chunk_results,
            raw_responses=raw_responses,
        )
        print(f"      JSON:     {paths['json']}")
        print(f"      Excel:    {paths['xlsx']}")
        print(f"      SRT:      {paths['srt']}")
        print(f"      Dialog:   {paths['dialogue']}")
        return 0
    except (LLMError, ASRError) as e:
        print(f"Fehler: {e}", file=sys.stderr)
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


def _run_hf(
    *,
    audio: Path,
    out_dir: Path,
    chunk_sec: float,
    overlap_sec: float,
    keep_work: bool,
    max_new_tokens_cli: int | None,
) -> int:
    from omni_model import OmniModelError, build_engine_from_env

    work_dir = make_work_dir()
    try:
        print("[1/4] Konvertiere Audio → 16 kHz mono WAV …")
        master = prepare_master_wav(audio, work_dir)
        chunks = slice_chunks(
            master, work_dir, chunk_sec=chunk_sec, overlap_sec=overlap_sec
        )
        if not chunks:
            print("Keine Audio-Chunks erzeugt (Datei zu kurz?).", file=sys.stderr)
            return 3
        audio_dur = probe_duration_sec(master)
        print(
            f"      Audio: {audio_dur / 60:.1f} min → {len(chunks)} Chunk(s) "
            f"(Backend: Hugging Face Qwen-Omni)"
        )

        print("[2/4] Lade Qwen-Omni (HF) …")
        engine = build_engine_from_env()
        print(f"      {engine.model_id} (Familie: {engine._family})")
        engine.load()
        for line in engine.device_report_lines():
            print(f"      {line}")

        chunk_results: List[Dict[str, Any]] = []
        raw_responses: List[Dict[str, Any]] = []
        total = len(chunks)
        use_tqdm = _env_bool("OMNI_PROGRESS", True)

        print(f"[3/4] Transkribiere {total} Chunk(s) …")
        for chunk in _progress_iter(chunks, total=total, enabled=use_tqdm):
            t0 = time.time()
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
                else num_predict_for_chunk(chunk.duration_sec)
            )
            raw = engine.transcribe_clip(
                wav_path=chunk.path,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_new_tokens=tok_limit,
            )
            elapsed = time.time() - t0
            parsed = _parse_chunk_json(raw, chunk)
            chunk_results.append(parsed)
            raw_responses.append(
                {"chunk_index": chunk.index, "elapsed_s": round(elapsed, 2), "raw_text": raw}
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

        print("[4/4] Export …")
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
    finally:
        if keep_work:
            print(f"Arbeitsordner behalten: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


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

    backend = _backend()
    if backend in ("hf", "huggingface", "omni"):
        return _run_hf(
            audio=audio,
            out_dir=out_dir,
            chunk_sec=chunk_sec,
            overlap_sec=overlap_sec,
            keep_work=keep_work,
            max_new_tokens_cli=max_new_tokens_cli,
        )
    if backend in ("ollama", "default"):
        return _run_ollama(
            audio=audio,
            out_dir=out_dir,
            chunk_sec=chunk_sec,
            overlap_sec=overlap_sec,
            keep_work=keep_work,
            max_new_tokens_cli=max_new_tokens_cli,
        )
    print(f"Unbekannter TRANSCRIBE_BACKEND: {backend}", file=sys.stderr)
    return 6


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(
        description="Transkription RunPod: Whisper-Chunks + Ollama (Standard) oder HF Qwen-Omni."
    )
    ap.add_argument("--audio", "-a", required=True, help="Audio-Datei (wav, mp3, …)")
    ap.add_argument("--output-dir", "-o", default="output", help="Ausgabeordner")
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
        help="Ollama num_predict / HF max_new_tokens pro Chunk",
    )
    ap.add_argument(
        "--backend",
        choices=("ollama", "hf"),
        default=None,
        help="ollama (Standard) oder hf (Qwen-Omni via transformers)",
    )
    ap.add_argument("--keep-work", action="store_true", help="Temp-Chunks behalten")
    args = ap.parse_args()
    if args.backend:
        os.environ["TRANSCRIBE_BACKEND"] = args.backend
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
