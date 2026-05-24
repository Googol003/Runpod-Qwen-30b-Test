from __future__ import annotations

"""
Qwen3-Omni laden — 4-bit exakt wie:
https://gist.github.com/phhusson/4bc8851935ff1caafd3a7f7ceec34335
(RTX 3090 24GB, funktioniert)

Dein Download Qwen3-Omni-30B-A3B-Instruct (BF16 auf Disk) ist RICHTIG.
4-bit entsteht beim from_pretrained via bitsandbytes — kein separates „4bit-Modell“ nötig.
"""

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar


class OmniModelError(RuntimeError):
    pass


_T = TypeVar("_T")


def _env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _detect_family(model_id: str) -> str:
    mid = (model_id or "").lower()
    if "qwen3" in mid or "qwen-3" in mid:
        return "qwen3"
    return "qwen25"


def _transformers_version_tuple() -> tuple[int, int, int] | None:
    try:
        import importlib.metadata as md

        raw = md.version("transformers")
        parts = []
        for p in raw.split(".")[:3]:
            parts.append(int("".join(c for c in p if c.isdigit()) or "0"))
        while len(parts) < 3:
            parts.append(0)
        return parts[0], parts[1], parts[2]
    except Exception:
        return None


def _transformers_compat_lines() -> list[str]:
    ver = _transformers_version_tuple()
    if ver is None:
        return ["transformers: Version unbekannt"]
    if ver[0] >= 5:
        return [
            f"FEHLER: transformers {ver[0]}.{ver[1]}.{ver[2]} — 4-bit MoE bricht ab v5",
            "Symptom: ~31 GB VRAM bei 57 % Laden (unquantisiert auf GPU).",
            "Fix: pip install 'transformers>=4.51.0,<5.0.0'",
            "Zusätzlich: HF_DEACTIVATE_ASYNC_LOAD=1 (in .env)",
        ]
    return [f"transformers: {ver[0]}.{ver[1]}.{ver[2]} OK (<5)"]


def _prepare_load_env() -> None:
    """Workaround Transformers 5 async load + allocator (Gist + HF #43032 / #44387)."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")


def _gpu_total_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    return None


def _resolve_model_source(model_id: str) -> tuple[str, bool]:
    raw = (model_id or "").strip()
    if not raw:
        raise OmniModelError("OMNI_MODEL_ID ist leer")
    p = Path(raw).expanduser()
    if p.exists() and p.is_dir():
        if not (p / "config.json").is_file():
            raise OmniModelError(
                f"Kein config.json in {p}\n"
                "Richtig: huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct "
                f"--local-dir {p}"
            )
        return str(p.resolve()), True
    if raw.startswith("/") or raw.startswith("."):
        raise OmniModelError(f"Ordner fehlt: {p}")
    return raw, False


def _check_download(model_source: str, local: bool) -> list[str]:
    lines = [f"Quelle: {model_source} ({'lokal' if local else 'Hub'})"]
    if not local:
        lines.append("Hinweis: Lokal laden spart RAM (OMNI_MODEL_ID=/workspace/models/...)")
        return lines
    p = Path(model_source)
    cfg = p / "config.json"
    if cfg.is_file():
        try:
            import json

            meta = json.loads(cfg.read_text(encoding="utf-8"))
            arch = meta.get("architectures") or meta.get("model_type") or "?"
            lines.append(f"config.json OK — architecture: {arch}")
        except Exception as e:
            lines.append(f"config.json lesen: {e}")
    st = list(p.glob("*.safetensors")) + list(p.glob("model-*-of-*.safetensors"))
    lines.append(f"Gewichte: {len(st)} safetensors-Shard(s) gefunden")
    if len(st) == 0:
        lines.append("FEHLER: Keine .safetensors — Download unvollständig!")
    return lines


def load_plan_lines(model_id: str) -> list[str]:
    load_4 = _env_bool("OMNI_LOAD_IN_4BIT", True)
    no_talker = not _env_bool("OMNI_ENABLE_AUDIO_OUTPUT", False)
    flash = _env_bool("OMNI_FLASH_ATTN", False)
    lines = [
        f"Modell: {model_id}",
        f"Lademodus: Gist-4bit (bitsandbytes), load_in_4bit={load_4}",
        f"device_map: cuda (wie Gist)",
        f"dtype: auto (wie Gist)",
        f"enable_audio_output: {not no_talker} (False = Talker nicht laden, ~2GB weniger)",
        f"attn_implementation: {'flash_attention_2' if flash else 'aus'}",
        "Erwartung nach Laden: ~15–20 GB VRAM (4-bit), nicht ~31 GB",
    ]
    gb = _gpu_total_gb()
    if gb:
        lines.append(f"GPU: {gb:.1f} GB")
    lines.extend(_transformers_compat_lines())
    if os.getenv("HF_DEACTIVATE_ASYNC_LOAD", "").strip() in ("1", "true", "yes"):
        lines.append("HF_DEACTIVATE_ASYNC_LOAD: 1")
    else:
        lines.append("WARNUNG: HF_DEACTIVATE_ASYNC_LOAD nicht gesetzt (OOM-Risiko ab transformers 5)")
    try:
        import bitsandbytes  # noqa: F401

        lines.append("bitsandbytes: OK")
    except ImportError:
        lines.append("FEHLER: pip install -U bitsandbytes")
    try:
        src, local = _resolve_model_source(model_id)
        lines.extend(_check_download(src, local))
    except OmniModelError as e:
        lines.append(str(e))
    return lines


def _load_qwen3_gist(model_source: str, *, local_only: bool) -> Any:
    """
    1:1 aus phhusson Gist (RTX 3090 + 4bit).
    Zusatz: enable_audio_output=False für Transkription (offizielle HF-Doku).
    """
    import torch
    from transformers import (  # type: ignore
        BitsAndBytesConfig,
        Qwen3OmniMoeForConditionalGeneration,
    )

    _prepare_load_env()
    ver = _transformers_version_tuple()
    if ver is not None and ver[0] >= 5:
        raise OmniModelError(
            "\n".join(
                [
                    "transformers >= 5.0 — 4-bit für Qwen3-Omni-MoE funktioniert so nicht.",
                    "Lade zuerst unquantisiert auf die GPU (~31 GB) → OOM bei 57 %.",
                    "pip install 'transformers>=4.51.0,<5.0.0'",
                    "export HF_DEACTIVATE_ASYNC_LOAD=1",
                ]
            )
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    kwargs: dict[str, Any] = {
        "dtype": "auto",
        "device_map": "cuda",
        "quantization_config": quant_config,
        "low_cpu_mem_usage": True,
    }
    if local_only:
        kwargs["local_files_only"] = True
    if _env_bool("OMNI_FLASH_ATTN", False):
        kwargs["attn_implementation"] = "flash_attention_2"
    if not _env_bool("OMNI_ENABLE_AUDIO_OUTPUT", False):
        kwargs["enable_audio_output"] = False

    print("      [Gist] from_pretrained(…, 4bit, device_map=cuda, dtype=auto)", flush=True)
    return Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_source, **kwargs)


def _load_qwen25(model_source: str, *, local_only: bool, flash: bool) -> Any:
    import torch
    from transformers import (  # type: ignore
        BitsAndBytesConfig,
        Qwen2_5OmniForConditionalGeneration,
    )

    if not _env_bool("OMNI_LOAD_IN_4BIT", False):
        kw: dict[str, Any] = {"device_map": "auto", "torch_dtype": "auto"}
        if local_only:
            kw["local_files_only"] = True
        if flash:
            kw["attn_implementation"] = "flash_attention_2"
        return Qwen2_5OmniForConditionalGeneration.from_pretrained(model_source, **kw)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    kw = {
        "device_map": "cuda",
        "quantization_config": quant_config,
        "low_cpu_mem_usage": True,
    }
    if local_only:
        kw["local_files_only"] = True
    if flash:
        kw["attn_implementation"] = "flash_attention_2"
    if not _env_bool("OMNI_ENABLE_AUDIO_OUTPUT", False):
        kw["enable_audio_output"] = False
    return Qwen2_5OmniForConditionalGeneration.from_pretrained(model_source, **kw)


@dataclass
class OmniEngine:
    model_id: str
    flash_attn: bool = False
    max_new_tokens: int = 1024

    def __post_init__(self) -> None:
        self._family = _detect_family(self.model_id)
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        if self._model is not None:
            return

        from gpu_memory import free_gpu_memory

        for line in free_gpu_memory():
            print(f"      {line}")

        model_source, local_only = _resolve_model_source(self.model_id)
        flash = self._flash_attn

        try:
            if self._family == "qwen3":
                from transformers import Qwen3OmniMoeProcessor  # type: ignore

                self._model = _load_qwen3_gist(model_source, local_only=local_only)
                self._processor = Qwen3OmniMoeProcessor.from_pretrained(
                    model_source, local_files_only=local_only
                )
            else:
                from transformers import Qwen2_5OmniProcessor  # type: ignore

                self._model = _load_qwen25(
                    model_source, local_only=local_only, flash=flash
                )
                self._processor = Qwen2_5OmniProcessor.from_pretrained(
                    model_source, local_files_only=local_only
                )
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                raise OmniModelError(
                    "\n".join(
                        [
                            "CUDA OOM bei ~57 % / ~31 GB — typisch transformers 5.x + 4-bit MoE.",
                            "Gewichte werden VOR der Quantisierung als FP16 auf die GPU kopiert.",
                            "Fix (RunPod):",
                            "  pip install 'transformers>=4.51.0,<5.0.0'",
                            "  export HF_DEACTIVATE_ASYNC_LOAD=1",
                            "  Neuer Python-Prozess (VRAM nach OOM leeren)",
                            "Optional (Gist): flash-attn + OMNI_FLASH_ATTN=1",
                            *load_plan_lines(self.model_id),
                        ]
                    )
                ) from e
            raise

        if not getattr(self._model, "is_loaded_in_4bit", False):
            raise OmniModelError(
                "Modell ist NICHT 4-bit geladen (is_loaded_in_4bit=False).\n"
                "Download ist OK — aber bitsandbytes-GPTQ-Pfad schlägt fehl.\n"
                "pip install -U bitsandbytes"
            )

        free_gpu_memory()

    def device_report_lines(self) -> list[str]:
        if self._model is None:
            return ["Nicht geladen."]
        lines: list[str] = []
        if getattr(self._model, "is_loaded_in_4bit", False):
            lines.append("OK: 4-bit (bitsandbytes)")
        else:
            lines.append("FEHLER: nicht 4-bit")
        try:
            import torch

            if torch.cuda.is_available():
                a = torch.cuda.memory_allocated(0) / (1024**3)
                t = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                lines.append(f"VRAM: {a:.1f} / {t:.1f} GB")
                if a > 26:
                    lines.append(
                        "WARNUNG: >26 GB — wirkt unquantisiert; Flash-Attn kann helfen."
                    )
        except ImportError:
            pass
        return lines

    @property
    def _flash_attn(self) -> bool:
        return self.flash_attn and _env_bool("OMNI_FLASH_ATTN", False)

    def transcribe_clip(
        self,
        *,
        wav_path: Path,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        self.load()
        from qwen_omni_utils import process_mm_info  # type: ignore

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(wav_path.resolve())},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        use_audio_in_video = False
        processor = self._processor
        model = self._model

        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=use_audio_in_video
        )
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        dev = model.device if hasattr(model, "device") else next(model.parameters()).device
        inputs = inputs.to(dev)
        if hasattr(model, "dtype"):
            inputs = inputs.to(model.dtype)
        input_len = inputs["input_ids"].shape[1]
        limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens

        import torch

        def _gen():
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=limit,
                    use_audio_in_video=use_audio_in_video,
                    return_audio=False,
                    do_sample=False,
                )
            return out

        out = _run_with_heartbeat(_gen, label=wav_path.name)
        if hasattr(out, "sequences"):
            gen_ids = out.sequences[:, input_len:]
        else:
            gen_ids = out[:, input_len:]
        decoded = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (decoded[0] or "").strip() if decoded else ""


def _run_with_heartbeat(fn: Callable[[], _T], *, label: str) -> _T:
    raw = (os.getenv("OMNI_HEARTBEAT_SEC") or "30").strip()
    try:
        interval = max(0.0, float(raw))
    except ValueError:
        interval = 30.0
    if interval <= 0:
        return fn()
    stop = threading.Event()

    def _beat() -> None:
        t0 = time.time()
        while not stop.wait(interval):
            print(f"      … {label} ({time.time() - t0:.0f}s)", flush=True)

    th = threading.Thread(target=_beat, daemon=True)
    th.start()
    try:
        return fn()
    finally:
        stop.set()
        th.join(timeout=1.0)


def apply_runpod_defaults() -> None:
    os.environ.setdefault("OMNI_LOAD_IN_4BIT", "1")
    os.environ.setdefault("OMNI_ENABLE_AUDIO_OUTPUT", "0")
    os.environ.setdefault("OMNI_FLASH_ATTN", "0")
    _prepare_load_env()


def _engine_max_new_tokens_default() -> int:
    raw = (os.getenv("OMNI_MAX_NEW_TOKENS") or "").strip().lower()
    if not raw or raw == "auto":
        return 1024
    return max(128, int(raw))


def build_engine_from_env() -> OmniEngine:
    model_id = os.getenv("OMNI_MODEL_ID") or "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    return OmniEngine(
        model_id=model_id.strip(),
        flash_attn=_env_bool("OMNI_FLASH_ATTN", False),
        max_new_tokens=_engine_max_new_tokens_default(),
    )
