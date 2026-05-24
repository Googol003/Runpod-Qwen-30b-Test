from __future__ import annotations

import os
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar


class OmniModelError(RuntimeError):
    pass


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


def _build_load_kwargs(*, local_only: bool, family: str) -> dict[str, Any]:
    """
    device_map=auto legt bei zu wenig VRAM still Teile auf CPU.
    30B-A3B auf 32 GB: OMNI_LOAD_IN_4BIT=1 und OMNI_NO_CPU_OFFLOAD=1.
    """
    import torch

    kwargs: dict[str, Any] = {}
    if local_only:
        kwargs["local_files_only"] = True

    load_4 = _env_bool("OMNI_LOAD_IN_4BIT", False)
    load_8 = _env_bool("OMNI_LOAD_IN_8BIT", False)
    no_cpu = _env_bool("OMNI_NO_CPU_OFFLOAD", False)
    device_map = (os.getenv("OMNI_DEVICE_MAP") or "").strip()

    if load_4 and load_8:
        raise OmniModelError("Nur OMNI_LOAD_IN_4BIT oder OMNI_LOAD_IN_8BIT setzen, nicht beides.")

    if load_4 or load_8:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except ImportError as e:
            raise OmniModelError(
                "Quantisierung braucht bitsandbytes: pip install bitsandbytes"
            ) from e
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_4,
            load_in_8bit=load_8,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = device_map or "cuda:0"
        return kwargs

    if no_cpu:
        if not torch.cuda.is_available():
            raise OmniModelError("OMNI_NO_CPU_OFFLOAD=1, aber keine CUDA-GPU sichtbar.")
        idx = 0
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        reserve_gb = float(os.getenv("OMNI_GPU_RESERVE_GB", "3") or "3")
        cap_gb = max(1, int(total_gb - reserve_gb))
        kwargs["device_map"] = device_map or "auto"
        kwargs["max_memory"] = {idx: f"{cap_gb}GiB", "cpu": "0GiB"}
        return kwargs

    kwargs["device_map"] = device_map or "auto"
    return kwargs


def _assert_no_cpu_offload_if_requested(model: Any) -> None:
    if not _env_bool("OMNI_NO_CPU_OFFLOAD", False):
        return
    cpuish = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        d = str(p.device).lower()
        if "cpu" in d or "meta" in d:
            cpuish += p.numel()
    if total and cpuish > total * 0.01:
        raise OmniModelError(
            f"OMNI_NO_CPU_OFFLOAD=1, aber {100.0 * cpuish / total:.1f}% der Parameter "
            "liegen noch auf CPU/Meta. Unquantisiert braucht 30B-A3B ~60 GB (BF16) — "
            "auf 32 GB RTX 5090: export OMNI_LOAD_IN_4BIT=1 (pip install bitsandbytes)."
        )


def _resolve_model_source(model_id: str) -> tuple[str, bool]:
    """
    Hugging-Face-Repo-ID (namespace/name) oder lokaler Ordner mit config.json.
    """
    raw = (model_id or "").strip()
    if not raw:
        raise OmniModelError("OMNI_MODEL_ID ist leer")
    p = Path(raw).expanduser()
    if p.exists() and p.is_dir():
        if not (p / "config.json").is_file():
            raise OmniModelError(
                f"Lokaler Modellordner unvollständig (kein config.json): {p}\n"
                "Erneut laden: huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct "
                f"--local-dir {p}"
            )
        return str(p.resolve()), True
    if raw.startswith("/") or raw.startswith("."):
        raise OmniModelError(
            f"Modellordner existiert nicht: {p}\n"
            "Download: huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct "
            f"--local-dir {p}"
        )
    return raw, False


def load_plan_lines(model_id: str) -> list[str]:
    """Konfiguration vor dem Laden (ohne Gewichte zu laden)."""
    mid = model_id.lower()
    family = _detect_family(model_id)
    load_4 = _env_bool("OMNI_LOAD_IN_4BIT", False)
    load_8 = _env_bool("OMNI_LOAD_IN_8BIT", False)
    no_cpu = _env_bool("OMNI_NO_CPU_OFFLOAD", False)
    flash = _env_bool("OMNI_FLASH_ATTN", True)
    dm_env = (os.getenv("OMNI_DEVICE_MAP") or "").strip()
    if load_4 or load_8:
        device_map = dm_env or "cuda:0"
    elif no_cpu:
        device_map = dm_env or "auto (max_memory, cpu=0)"
    else:
        device_map = dm_env or "auto"

    if load_4:
        quant = "4-bit NF4 (bitsandbytes)"
    elif load_8:
        quant = "8-bit (bitsandbytes)"
    else:
        quant = "keine — dtype/torch_dtype=auto (typ. BF16/FP16, ~60 GB bei 30B)"

    lines = [
        f"Backend: Hugging Face transformers (Qwen-Omni, Familie: {family})",
        f"OMNI_MODEL_ID: {model_id}",
        f"Quantisierung: {quant}",
        f"device_map: {device_map}",
        f"OMNI_NO_CPU_OFFLOAD: {'ja' if no_cpu else 'nein'}",
        f"OMNI_FLASH_ATTN: {'ja' if flash else 'nein'}",
    ]

    try:
        import torch

        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            lines.append(f"CUDA: {torch.cuda.get_device_name(0)} ({gb:.1f} GB)")
        else:
            lines.append("CUDA: nicht verfügbar")
    except ImportError:
        lines.append("torch: nicht installiert")

    if load_4 or load_8:
        try:
            import bitsandbytes  # noqa: F401

            lines.append("bitsandbytes: installiert")
        except ImportError:
            lines.append("bitsandbytes: FEHLT — pip install bitsandbytes")

    if ("30b" in mid or "qwen3" in mid) and not load_4 and not load_8:
        lines.append(
            "HINWEIS: 30B ohne 4-bit auf ≤32 GB GPU → CPU-Offload (sehr langsam). "
            "Setze OMNI_LOAD_IN_4BIT=1"
        )

    try:
        src, local = _resolve_model_source(model_id)
        lines.append(f"Quelle: {src} ({'lokal' if local else 'Hugging Face Hub'})")
    except OmniModelError as e:
        lines.append(f"Quelle: FEHLER — {e}")

    return lines


@dataclass
class OmniEngine:
    """Lädt Qwen2.5-Omni oder Qwen3-Omni einmal und transkribiert Audio-Clips."""

    model_id: str
    flash_attn: bool = True
    max_new_tokens: int = 2048

    def __post_init__(self) -> None:
        self._family = _detect_family(self.model_id)
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as e:
            raise OmniModelError("torch nicht installiert") from e

        attn = "flash_attention_2" if self._flash_attn else None
        model_source, local_only = _resolve_model_source(self.model_id)
        load_kw = _build_load_kwargs(local_only=local_only, family=self._family)
        if attn and not load_kw.get("quantization_config"):
            load_kw["attn_implementation"] = attn

        if self._family == "qwen3":
            try:
                from transformers import (  # type: ignore
                    Qwen3OmniMoeForConditionalGeneration,
                    Qwen3OmniMoeProcessor,
                )
            except ImportError as e:
                raise OmniModelError(
                    "Qwen3-Omni braucht eine aktuelle transformers-Version "
                    "(siehe README / pip install transformers -U)."
                ) from e
            if not load_kw.get("quantization_config"):
                load_kw["dtype"] = "auto"
            self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_source, **load_kw
            )
            if hasattr(self._model, "disable_talker"):
                self._model.disable_talker()
            self._processor = Qwen3OmniMoeProcessor.from_pretrained(
                model_source, local_files_only=local_only
            )
        else:
            try:
                from transformers import (  # type: ignore
                    Qwen2_5OmniForConditionalGeneration,
                    Qwen2_5OmniProcessor,
                )
            except ImportError as e:
                raise OmniModelError(
                    "Qwen2.5-Omni braucht transformers mit Qwen2_5Omni "
                    "(siehe README)."
                ) from e
            if not load_kw.get("quantization_config"):
                load_kw["torch_dtype"] = "auto"
            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_source, **load_kw
            )
            self._processor = Qwen2_5OmniProcessor.from_pretrained(
                model_source, local_files_only=local_only
            )

        _assert_no_cpu_offload_if_requested(self._model)
        _ = torch  # noqa: F841

    def device_report_lines(self) -> list[str]:
        """Nach dem Laden: GPU/CPU-Aufteilung."""
        if self._model is None:
            return ["Modell nicht geladen."]
        try:
            import torch
        except ImportError:
            return ["torch nicht verfügbar."]

        lines: list[str] = []
        is_4bit = getattr(self._model, "is_loaded_in_4bit", False)
        is_8bit = getattr(self._model, "is_loaded_in_8bit", False)
        if is_4bit:
            lines.append("Geladen als: 4-bit quantisiert (bitsandbytes)")
        elif is_8bit:
            lines.append("Geladen als: 8-bit quantisiert (bitsandbytes)")
        else:
            lines.append("Geladen als: unquantisiert (dtype auto)")

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            alloc = torch.cuda.memory_allocated(0) / (1024**3)
            lines.append(f"GPU: {name} ({gb:.1f} GB gesamt, {alloc:.1f} GB belegt)")
        else:
            lines.append("Keine CUDA-GPU sichtbar — Inferenz nur auf CPU.")

        by_dev: Counter[str] = Counter()
        for p in self._model.parameters():
            by_dev[str(p.device)] += p.numel()
        total = sum(by_dev.values()) or 1
        for dev, n in by_dev.most_common():
            lines.append(f"  Parameter auf {dev}: {100.0 * n / total:.1f}%")

        hf_map = getattr(self._model, "hf_device_map", None)
        if isinstance(hf_map, dict) and hf_map:
            layer_devs = Counter(str(v) for v in hf_map.values())
            lines.append(f"  Layer-Verteilung: {dict(layer_devs)}")

        cpuish = sum(
            n for d, n in by_dev.items() if "cpu" in d.lower() or "meta" in d.lower()
        )
        if cpuish > total * 0.05:
            lines.append(
                "  WARNUNG: Teile auf CPU/Meta (Offload). Bei 30B auf 32 GB: "
                "OMNI_LOAD_IN_4BIT=1 + OMNI_NO_CPU_OFFLOAD=1 + pip install bitsandbytes"
            )
        elif "qwen3" in self.model_id.lower() or "30b" in self.model_id.lower():
            lines.append("  Modell vollständig auf GPU (kein CPU-Offload erkannt).")
        return lines

    @property
    def _flash_attn(self) -> bool:
        return self.flash_attn and _env_bool("OMNI_FLASH_ATTN", True)

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
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
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
        input_device = _input_device(model)
        inputs = inputs.to(input_device)
        if hasattr(model, "dtype"):
            inputs = inputs.to(model.dtype)
        input_len = inputs["input_ids"].shape[1]

        token_limit = (
            self.max_new_tokens if max_new_tokens is None else max_new_tokens
        )
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": token_limit,
            "use_audio_in_video": use_audio_in_video,
            "return_audio": False,
            "do_sample": False,
        }
        label = f"Chunk-Inferenz ({wav_path.name}, max_new={token_limit})"

        import torch

        def _generate():
            with torch.inference_mode():
                return model.generate(**inputs, **gen_kwargs)

        if self._family == "qwen3":
            out = _run_with_heartbeat(_generate, label=label)
            if hasattr(out, "sequences"):
                gen_ids = out.sequences[:, input_len:]
            else:
                gen_ids = out[:, input_len:]
        else:
            out = _run_with_heartbeat(_generate, label=label)
            gen_ids = out[:, input_len:]

        decoded = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if not decoded:
            return ""
        return (decoded[0] or "").strip()


_T = TypeVar("_T")


def _input_device(model: Any):
    import torch

    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _heartbeat_interval_sec() -> float:
    raw = (os.getenv("OMNI_HEARTBEAT_SEC") or "30").strip()
    try:
        v = float(raw)
    except ValueError:
        v = 30.0
    return max(0.0, v)


def _run_with_heartbeat(fn: Callable[[], _T], *, label: str) -> _T:
    interval = _heartbeat_interval_sec()
    if interval <= 0:
        return fn()

    stop = threading.Event()

    def _beat() -> None:
        t0 = time.time()
        while not stop.wait(interval):
            elapsed = time.time() - t0
            print(f"      … {label} läuft ({elapsed:.0f}s) — prüfe GPU: nvidia-smi", flush=True)

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop.set()
        thread.join(timeout=1.0)


def apply_runpod_defaults() -> None:
    """Falls .env fehlt: 30B auf ≤40 GB GPU → 4-bit + kein CPU-Offload."""
    if _env_bool("OMNI_LOAD_IN_4BIT", False) or _env_bool("OMNI_LOAD_IN_8BIT", False):
        return
    model_id = (
        os.getenv("OMNI_MODEL_ID") or "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    ).lower()
    if "30b" not in model_id and "qwen3" not in model_id:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gb > 40:
            return
    except ImportError:
        return
    os.environ.setdefault("OMNI_LOAD_IN_4BIT", "1")
    os.environ.setdefault("OMNI_NO_CPU_OFFLOAD", "1")
    os.environ.setdefault("OMNI_FLASH_ATTN", "0")
    os.environ.setdefault("OMNI_DEVICE_MAP", "cuda:0")


def _engine_max_new_tokens_default() -> int:
    """Engine-Fallback; pro Chunk überschreibt transcribe_omni bei OMNI_MAX_NEW_TOKENS=auto."""
    raw = (os.getenv("OMNI_MAX_NEW_TOKENS") or "").strip().lower()
    if not raw or raw == "auto":
        return 1024
    return max(128, int(raw))


def build_engine_from_env() -> OmniEngine:
    model_id = (
        os.getenv("OMNI_MODEL_ID")
        or "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )
    flash = _env_bool("OMNI_FLASH_ATTN", True)
    return OmniEngine(
        model_id=model_id.strip(),
        flash_attn=flash,
        max_new_tokens=_engine_max_new_tokens_default(),
    )
