from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


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
        kwargs: dict[str, Any] = {
            "device_map": "auto",
        }
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
                self.model_id, **kwargs
            )
            if hasattr(self._model, "disable_talker"):
                self._model.disable_talker()
            self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id)
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
                self.model_id, **kwargs
            )
            self._processor = Qwen2_5OmniProcessor.from_pretrained(self.model_id)

        _ = torch  # noqa: F841 — nur Import-Check

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
        inputs = inputs.to(model.device).to(model.dtype)
        input_len = inputs["input_ids"].shape[1]

        if self._family == "qwen3":
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                use_audio_in_video=use_audio_in_video,
                return_audio=False,
            )
            if hasattr(out, "sequences"):
                gen_ids = out.sequences[:, input_len:]
            else:
                gen_ids = out[:, input_len:]
        else:
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                use_audio_in_video=use_audio_in_video,
                return_audio=False,
            )
            gen_ids = out[:, input_len:]

        decoded = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if not decoded:
            return ""
        return (decoded[0] or "").strip()


def build_engine_from_env() -> OmniEngine:
    model_id = (
        os.getenv("OMNI_MODEL_ID")
        or os.getenv("LLM_MODEL")
        or "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )
    max_new = int(os.getenv("OMNI_MAX_NEW_TOKENS", "2048") or "2048")
    flash = _env_bool("OMNI_FLASH_ATTN", True)
    return OmniEngine(model_id=model_id.strip(), flash_attn=flash, max_new_tokens=max_new)
