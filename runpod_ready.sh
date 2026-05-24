#!/usr/bin/env bash
set -euo pipefail

# RunPod: HF Qwen-Omni + optional 4-bit
#
# Usage:
#   bash runpod_ready.sh "/workspace/TestAudio.mp3"

AUDIO="${1:-}"
if [[ -z "${AUDIO}" ]]; then
  echo "ERROR: Bitte Audio-Pfad angeben."
  echo "Example: bash runpod_ready.sh \"/workspace/TestAudio.mp3\""
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

log "cwd=${SCRIPT_DIR}"
if [[ ! -f "${AUDIO}" ]]; then
  echo "ERROR: Audio nicht gefunden: ${AUDIO}"
  exit 2
fi

if [[ -d ".venv" ]]; then
  log "Reusing venv .venv"
else
  log "Creating venv .venv"
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

apt-get update -qq && apt-get install -y -qq ffmpeg 2>/dev/null || true

REQ_HASH_FILE=".deps.sha256"
NEW_HASH="$(python - <<'PY'
import hashlib
from pathlib import Path
p = Path("requirements.txt")
print(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "")
PY
)"
OLD_HASH=""
[[ -f "${REQ_HASH_FILE}" ]] && OLD_HASH="$(cat "${REQ_HASH_FILE}" || true)"
if [[ "${NEW_HASH}" != "${OLD_HASH}" ]]; then
  log "pip install -r requirements.txt"
  python -m pip install -U pip
  pip install -r requirements.txt
  echo "${NEW_HASH}" > "${REQ_HASH_FILE}"
fi

if [[ ! -f .env ]]; then
  log "Keine .env — kopiere .env.example"
  cp .env.example .env
fi

log "OMNI_MODEL_ID=${OMNI_MODEL_ID}"
log "OMNI_LOAD_IN_4BIT=${OMNI_LOAD_IN_4BIT} OMNI_NO_CPU_OFFLOAD=${OMNI_NO_CPU_OFFLOAD}"

log "Modell-Check …"
python transcribe_omni.py --check --check-load || exit 4

OUT_DIR="${OUTPUT_DIR:-output}"
mkdir -p "${OUT_DIR}"
log "Transkription starten …"
python transcribe_omni.py --audio "${AUDIO}" --output-dir "${OUT_DIR}"
log "Done."
