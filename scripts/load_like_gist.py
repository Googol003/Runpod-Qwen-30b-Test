#!/usr/bin/env python3
"""Minimaltest: exakt Gist-Laden (ohne transcribe_omni)."""
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = root / ".env"
if env.is_file():
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

from omni_model import build_engine_from_env, load_plan_lines  # noqa: E402

if __name__ == "__main__":
    e = build_engine_from_env()
    for line in load_plan_lines(e.model_id):
        print(line)
    print("\nLade …")
    e.load()
    for line in e.device_report_lines():
        print(line)
