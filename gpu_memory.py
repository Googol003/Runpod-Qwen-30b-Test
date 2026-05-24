"""GPU-Speicher vor dem Modell-Laden freigeben und VRAM-Status anzeigen."""

from __future__ import annotations

import gc
import os
import subprocess
from typing import List


def _env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def setup_cuda_allocator() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def free_gpu_memory() -> List[str]:
    """
    PyTorch-Cache leeren. Beendet keine anderen Prozesse (Jupyter/Ollama).
    """
    setup_cuda_allocator()
    lines: List[str] = []

    try:
        import torch
    except ImportError:
        lines.append("torch nicht installiert — kein GPU-Clear möglich.")
        return lines

    if not torch.cuda.is_available():
        lines.append("Keine CUDA-GPU — Clear übersprungen.")
        return lines

    idx = 0
    before_alloc = torch.cuda.memory_allocated(idx) / (1024**3)
    before_reserved = torch.cuda.memory_reserved(idx) / (1024**3)

    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats(idx)

    after_alloc = torch.cuda.memory_allocated(idx) / (1024**3)
    after_reserved = torch.cuda.memory_reserved(idx) / (1024**3)
    lines.append(
        f"PyTorch VRAM: {before_alloc:.2f}→{after_alloc:.2f} GB belegt, "
        f"reserviert {before_reserved:.2f}→{after_reserved:.2f} GB"
    )
    lines.extend(_nvidia_smi_lines())
    return lines


def _nvidia_smi_lines() -> List[str]:
    lines: List[str] = []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            used, total = (x.strip() for x in proc.stdout.strip().split(",")[:2])
            lines.append(f"nvidia-smi gesamt: {used} / {total} MiB belegt")
            used_mib = int(float(used))
            if used_mib > 8000:
                lines.append(
                    "  Hinweis: VRAM schon belegt (anderer Prozess?). "
                    "Stoppe Jupyter/Ollama/alte Python-Läufe: nvidia-smi"
                )
        proc2 = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc2.returncode == 0 and proc2.stdout.strip():
            for row in proc2.stdout.strip().splitlines()[:5]:
                lines.append(f"  GPU-Prozess: {row.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return lines


def clear_gpu_on_start() -> List[str]:
    if not _env_bool("OMNI_CLEAR_GPU_ON_START", True):
        return ["GPU-Clear beim Start: aus (OMNI_CLEAR_GPU_ON_START=0)"]
    return free_gpu_memory()
