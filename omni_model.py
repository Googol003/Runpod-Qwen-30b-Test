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


def _gpu_total_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    return None


def _require_bitsandbytes() -> None:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as e:
        raise OmniModelError(
            "4-bit braucht bitsandbytes:\n  pip install -U bitsandbytes"
        ) from e


def _bnb_config(*, load_4: bool, load_8: bool) -> Any:
    import torch
    from transformers import BitsAndBytesConfig  # type: ignore

    return BitsAndBytesConfig(
        load_in_4bit=load_4,
        load_in_8bit=load_8,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _gpu_cap_gib() -> str:
    """VRAM-Obergrenze beim Laden (4-bit final ~15 GB, Peak darf nicht 31 GB sein)."""
    raw = (os.getenv("OMNI_GPU_CAP_GB") or "").strip()
    if raw:
        return f"{max(12, int(float(raw)))}GiB"
    gb = _gpu_total_gb()
    if gb is None:
        return "22GiB"
    reserve = float(os.getenv("OMNI_GPU_RESERVE_GB", "8") or "8")
    return f"{max(12, int(gb - reserve))}GiB"


def _build_load_kwargs(*, local_only: bool, family: str) -> dict[str, Any]:
    """
    4-bit 30B ≈ 15 GB — passt auf 32 GB.
    OOM entsteht, wenn beim Laden FP16-Buffers auf die GPU wandern (cuda + dtype=auto).
    Standard: device_map=auto + max_memory → quantisieren ohne 31-GB-Lade-Peak.
    """
    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if local_only:
        kwargs["local_files_only"] = True

    load_4 = _env_bool("OMNI_LOAD_IN_4BIT", False)
    load_8 = _env_bool("OMNI_LOAD_IN_8BIT", False)

    if load_4 and load_8:
        raise OmniModelError("Nur OMNI_LOAD_IN_4BIT oder OMNI_LOAD_IN_8BIT, nicht beides.")

    if load_4 or load_8:
        _require_bitsandbytes()
        kwargs["quantization_config"] = _bnb_config(load_4=load_4, load_8=load_8)
        strategy = (os.getenv("OMNI_LOAD_STRATEGY") or "staged").strip().lower()
        cpu_mem = (os.getenv("OMNI_CPU_MEMORY") or "128GiB").strip()
        if strategy == "cuda":
            kwargs["device_map"] = "cuda"
        else:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {0: _gpu_cap_gib(), "cpu": cpu_mem}
        # Kein dtype/torch_dtype bei 4-bit — sonst doppelter Speicher beim Laden
        return kwargs

    no_cpu = _env_bool("OMNI_NO_CPU_OFFLOAD", False)
    device_map = (os.getenv("OMNI_DEVICE_MAP") or "auto").strip() or "auto"
    if no_cpu:
        if not torch.cuda.is_available():
            raise OmniModelError("OMNI_NO_CPU_OFFLOAD=1, aber keine CUDA-GPU.")
        idx = 0
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        reserve_gb = float(os.getenv("OMNI_GPU_RESERVE_GB", "3") or "3")
        cap_gb = max(1, int(total_gb - reserve_gb))
        kwargs["device_map"] = device_map
        kwargs["max_memory"] = {idx: f"{cap_gb}GiB", "cpu": "0GiB"}
        return kwargs

    kwargs["device_map"] = device_map
    return kwargs


def _assert_quantized_if_requested(model: Any) -> None:
    if not _env_bool("OMNI_LOAD_IN_4BIT", False):
        return
    if getattr(model, "is_loaded_in_4bit", False):
        return
    raise OmniModelError(
        "OMNI_LOAD_IN_4BIT=1, aber das Modell ist NICHT 4-bit (is_loaded_in_4bit=False).\n"
        "Meist: Ladevorgang ohne echte 4-bit-Quantisierung (dtype=auto + cuda).\n"
        "  export OMNI_LOAD_STRATEGY=staged OMNI_LOAD_IN_4BIT=1"
    )


def _resolve_model_source(model_id: str) -> tuple[str, bool]:
    raw = (model_id or "").strip()
    if not raw:
        raise OmniModelError("OMNI_MODEL_ID ist leer")
    p = Path(raw).expanduser()
    if p.exists() and p.is_dir():
        if not (p / "config.json").is_file():
            raise OmniModelError(
                f"Lokaler Modellordner unvollständig (kein config.json): {p}"
            )
        return str(p.resolve()), True
    if raw.startswith("/") or raw.startswith("."):
        raise OmniModelError(f"Modellordner existiert nicht: {p}")
    return raw, False


def load_plan_lines(model_id: str) -> list[str]:
    mid = model_id.lower()
    family = _detect_family(model_id)
    load_4 = _env_bool("OMNI_LOAD_IN_4BIT", False)
    load_8 = _env_bool("OMNI_LOAD_IN_8BIT", False)
    no_audio = not _env_bool("OMNI_ENABLE_AUDIO_OUTPUT", False)
    flash = _env_bool("OMNI_FLASH_ATTN", True)
    strategy = (os.getenv("OMNI_LOAD_STRATEGY") or "staged").strip()
    if load_4:
        quant = f"4-bit NF4 (~15 GB Gewichte, passt auf 32 GB)"
        dm = f"{strategy} → GPU max {_gpu_cap_gib()}"
    elif load_8:
        quant = "8-bit (bitsandbytes)"
        dm = (os.getenv("OMNI_DEVICE_MAP") or "auto").strip() or "auto"
    else:
        quant = "KEIN 4-bit (~60 GB BF16) — auf 5090 32 GB → OOM"
        dm = (os.getenv("OMNI_DEVICE_MAP") or "auto").strip() or "auto"

    lines = [
        f"Modell: {model_id} (Familie: {family})",
        f"Quantisierung: {quant}",
        f"device_map: {dm}",
        f"Talker aus (nur Text): {'ja' if no_audio else 'nein'}",
        f"Flash Attention: {'ja' if flash else 'nein'}",
    ]
    gb = _gpu_total_gb()
    if gb is not None:
        lines.append(f"GPU: {gb:.1f} GB")
    if load_4:
        try:
            _require_bitsandbytes()
            lines.append("bitsandbytes: OK")
        except OmniModelError as e:
            lines.append(str(e))
    if ("30b" in mid or "qwen3" in mid) and not load_4:
        lines.append("FEHLER: 30B ohne 4-bit ≈ 60 GB — nicht auf 32 GB laden.")
    if load_4:
        lines.append(
            "Hinweis: OOM bei 57% Ladebalken = Lade-Peak (FP16), nicht finale 4-bit-Größe."
        )
    try:
        src, local = _resolve_model_source(model_id)
        lines.append(f"Quelle: {src} ({'lokal' if local else 'Hub'})")
    except OmniModelError as e:
        lines.append(str(e))
    return lines


def oom_diagnosis_lines(model_id: str) -> list[str]:
    return [
        "CUDA OOM beim Laden (Lade-Peak, nicht finale Modellgröße).",
        "4-bit 30B ≈ 15 GB — sollte auf 32 GB passen.",
        *load_plan_lines(model_id),
        "Fix: git pull && cp .env.example .env",
        "  export OMNI_LOAD_STRATEGY=staged OMNI_LOAD_IN_4BIT=1",
        "  python transcribe_omni.py --check --check-load",
    ]


def _from_pretrained_4bit(
    model_cls: Any,
    model_source: str,
    *,
    base_kw: dict[str, Any],
) -> Any:
    """staged zuerst (vermeidet FP16-Lade-Peak), optional cuda als Fallback."""
    from gpu_memory import free_gpu_memory

    load_4 = _env_bool("OMNI_LOAD_IN_4BIT", False)
    if not load_4:
        return model_cls.from_pretrained(model_source, **base_kw)

    strategy = (os.getenv("OMNI_LOAD_STRATEGY") or "staged").strip().lower()
    cpu_mem = (os.getenv("OMNI_CPU_MEMORY") or "128GiB").strip()
    cap = _gpu_cap_gib()
    attempts: list[tuple[str, dict[str, Any]]] = []
    staged = {
        **base_kw,
        "device_map": "auto",
        "max_memory": {0: cap, "cpu": cpu_mem},
    }
    cuda_kw = {k: v for k, v in base_kw.items() if k != "max_memory"}
    cuda_kw["device_map"] = "cuda"
    if strategy == "cuda":
        attempts = [("cuda", cuda_kw), ("staged", staged)]
    else:
        attempts = [("staged", staged), ("cuda", cuda_kw)]

    last_err: Exception | None = None
    for name, kw in attempts:
        print(f"      Lade 4-bit ({name}, GPU-Limit {cap}) …", flush=True)
        free_gpu_memory()
        try:
            return model_cls.from_pretrained(model_source, **kw)
        except Exception as e:
            last_err = e
            if "out of memory" not in str(e).lower():
                raise
            print(f"      {name}: OOM — nächster Versuch …", flush=True)
    raise OmniModelError("\n".join(oom_diagnosis_lines(""))) from last_err


@dataclass
class OmniEngine:
    model_id: str
    flash_attn: bool = True
    max_new_tokens: int = 1024

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

        from gpu_memory import free_gpu_memory

        for line in free_gpu_memory():
            print(f"      {line}")

        if _env_bool("OMNI_LOAD_IN_4BIT", False):
            _require_bitsandbytes()

        model_source, local_only = _resolve_model_source(self.model_id)
        load_kw = _build_load_kwargs(local_only=local_only, family=self._family)

        attn = "flash_attention_2" if self._flash_attn else None
        if attn and not load_kw.get("quantization_config"):
            load_kw["attn_implementation"] = attn

        if not _env_bool("OMNI_ENABLE_AUDIO_OUTPUT", False):
            load_kw["enable_audio_output"] = False

        if self._family == "qwen3":
            from transformers import (  # type: ignore
                Qwen3OmniMoeForConditionalGeneration,
                Qwen3OmniMoeProcessor,
            )

            if not load_kw.get("quantization_config"):
                load_kw["dtype"] = "auto"
            self._model = _from_pretrained_4bit(
                Qwen3OmniMoeForConditionalGeneration,
                model_source,
                base_kw=load_kw,
            )
            self._processor = Qwen3OmniMoeProcessor.from_pretrained(
                model_source, local_files_only=local_only
            )
        else:
            from transformers import (  # type: ignore
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )

            if not load_kw.get("quantization_config"):
                load_kw["torch_dtype"] = "auto"
            self._model = _from_pretrained_4bit(
                Qwen2_5OmniForConditionalGeneration,
                model_source,
                base_kw=load_kw,
            )
            self._processor = Qwen2_5OmniProcessor.from_pretrained(
                model_source, local_files_only=local_only
            )

        _assert_quantized_if_requested(self._model)
        free_gpu_memory()

    def device_report_lines(self) -> list[str]:
        if self._model is None:
            return ["Modell nicht geladen."]
        try:
            import torch
        except ImportError:
            return ["torch nicht verfügbar."]

        lines: list[str] = []
        if getattr(self._model, "is_loaded_in_4bit", False):
            lines.append("OK: 4-bit quantisiert geladen")
        elif getattr(self._model, "is_loaded_in_8bit", False):
            lines.append("OK: 8-bit quantisiert geladen")
        else:
            lines.append("WARNUNG: unquantisiert geladen")

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            lines.append(f"VRAM: {alloc:.1f} / {total:.1f} GB")
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
        input_device = _input_device(model)
        inputs = inputs.to(input_device)
        if hasattr(model, "dtype"):
            inputs = inputs.to(model.dtype)
        input_len = inputs["input_ids"].shape[1]

        token_limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": token_limit,
            "use_audio_in_video": use_audio_in_video,
            "return_audio": False,
            "do_sample": False,
        }
        label = f"Chunk ({wav_path.name}, max_new={token_limit})"

        import torch

        def _generate():
            with torch.inference_mode():
                return model.generate(**inputs, **gen_kwargs)

        out = _run_with_heartbeat(_generate, label=label)
        if hasattr(out, "sequences"):
            gen_ids = out.sequences[:, input_len:]
        else:
            gen_ids = out[:, input_len:]

        decoded = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (decoded[0] or "").strip() if decoded else ""


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
        return max(0.0, float(raw))
    except ValueError:
        return 30.0


def _run_with_heartbeat(fn: Callable[[], _T], *, label: str) -> _T:
    interval = _heartbeat_interval_sec()
    if interval <= 0:
        return fn()
    stop = threading.Event()

    def _beat() -> None:
        t0 = time.time()
        while not stop.wait(interval):
            print(f"      … {label} ({time.time() - t0:.0f}s)", flush=True)

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop.set()
        thread.join(timeout=1.0)


def apply_runpod_defaults() -> None:
    """32–48 GB GPU + Qwen3/30B: erzwinge 4-bit-Profil (5090)."""
    model_id = (
        os.getenv("OMNI_MODEL_ID") or "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    ).lower()
    if "30b" not in model_id and "qwen3" not in model_id:
        return
    gb = _gpu_total_gb()
    if gb is None or gb > 48:
        return
    os.environ["OMNI_LOAD_IN_4BIT"] = "1"
    os.environ["OMNI_ENABLE_AUDIO_OUTPUT"] = "0"
    os.environ["OMNI_FLASH_ATTN"] = "0"
    os.environ["OMNI_LOAD_STRATEGY"] = "staged"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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
