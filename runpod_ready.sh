#!/usr/bin/env bash
# RunPod Pod: Abhängigkeiten + ffmpeg
set -euo pipefail
cd "$(dirname "$0")"

apt-get update -qq
apt-get install -y -qq ffmpeg

python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional (empfohlen für 30B):
# pip install -U flash-attn --no-build-isolation

echo "OK. Beispiel:"
echo '  export OMNI_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct'
echo '  python transcribe_omni.py --audio TestAudio.mp3 --output-dir output'
