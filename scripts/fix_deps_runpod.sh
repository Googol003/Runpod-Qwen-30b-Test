#!/usr/bin/env bash
# RunPod: transformers 5.x → 4-bit MoE lädt ~31 GB FP16 → OOM. Downgrade + HF-Flag.
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -U pip
pip install 'transformers>=4.51.0,<5.0.0' -U bitsandbytes accelerate
python -c "import transformers; print('transformers', transformers.__version__)"
export HF_DEACTIVATE_ASYNC_LOAD=1
echo "OK. Dann: python scripts/load_like_gist.py"
