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


def _resolve_model_source(model_id: str) -> tuple[str, bool]:
    """
    Hugging-Face-Repo-ID (namespace/name) oder lokaler Ordner mit config.json.
    Fehlt der Ordner, wirft HF sonst „Repo id must be …“ für absolute Pfade.
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
        kwargs: dict[str, Any] = {
            "device_map": "auto",
        }
        if local_only:
            kwargs["local_files_only"] = True
        if attn:
            kwargs["attn_implementation"] = attn

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
            kwargs["dtype"] = "auto"
            self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_source, **kwargs
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
            kwargs["torch_dtype"] = "auto"
            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_source, **kwargs
            )
            self._processor = Qwen2_5OmniProcessor.from_pretrained(
                model_source, local_files_only=local_only
            )

        _ = torch  # noqa: F841 — nur Import-Check

    def device_report_lines(self) -> list[str]:
        """Kurzbericht: welche Anteile des Modells auf GPU vs. CPU liegen."""
        if self._model is None:
            return ["Modell nicht geladen."]
        try:
            import torch
        except ImportError:
            return ["torch nicht verfügbar."]

        lines: list[str] = []
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
                "  Hinweis: Teile des Modells liegen auf CPU (Offload). "
                "Inferenz nutzt die GPU, ist aber deutlich langsamer — erster Chunk kann "
                "viele Minuten dauern; tqdm bleibt bis dahin bei 0%."
            )
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

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "use_audio_in_video": use_audio_in_video,
            "return_audio": False,
        }
        label = f"Chunk-Inferenz ({wav_path.name})"
        if self._family == "qwen3":
            out = _run_with_heartbeat(
                lambda: model.generate(**inputs, **gen_kwargs),
                label=label,
            )
            if hasattr(out, "sequences"):
                gen_ids = out.sequences[:, input_len:]
            else:
                gen_ids = out[:, input_len:]
        else:
            out = _run_with_heartbeat(
                lambda: model.generate(**inputs, **gen_kwargs),
                label=label,
            )
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


def build_engine_from_env() -> OmniEngine:
    model_id = (
        os.getenv("OMNI_MODEL_ID")
        or os.getenv("LLM_MODEL")
        or "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )
    max_new = int(os.getenv("OMNI_MAX_NEW_TOKENS", "2048") or "2048")
    flash = _env_bool("OMNI_FLASH_ATTN", True)
    return OmniEngine(model_id=model_id.strip(), flash_attn=flash, max_new_tokens=max_new)
