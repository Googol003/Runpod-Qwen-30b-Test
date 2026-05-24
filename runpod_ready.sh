#!/usr/bin/env bash
set -euo pipefail
AUDIO="${1:-}"
[[ -z "${AUDIO}" ]] && { echo "Usage: bash runpod_ready.sh audio.mp3"; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
[[ -d .venv ]] || python -m venv .venv
source .venv/bin/activate
apt-get update -qq && apt-get install -y -qq ffmpeg 2>/dev/null || true
pip install -q -U pip bitsandbytes transformers accelerate qwen-omni-utils -r requirements.txt
[[ -f .env ]] || cp .env.example .env
set -a && source .env && set +a
python transcribe_omni.py --check --check-load
python transcribe_omni.py --audio "${AUDIO}" --output-dir "${OUTPUT_DIR:-output}"
