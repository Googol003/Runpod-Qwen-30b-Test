#!/usr/bin/env bash
set -euo pipefail

# RunPod: Whisper-Chunks + Ollama (wie runpod_excel_corrector)
#
# Usage:
#   bash runpod_ready.sh "/workspace/TestAudio.mp3"
#
# Optional:
#   LLM_MODEL=gemma4:26b WHISPER_MODEL=large-v3 bash runpod_ready.sh audio.mp3

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

export TRANSCRIBE_BACKEND="${TRANSCRIBE_BACKEND:-ollama}"
export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
export LLM_MODEL="${LLM_MODEL:-gemma4:26b}"
export WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
export WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
export OMNI_CHUNK_SEC="${OMNI_CHUNK_SEC:-30}"
export OMNI_OVERLAP_SEC="${OMNI_OVERLAP_SEC:-2}"

log "TRANSCRIBE_BACKEND=${TRANSCRIBE_BACKEND}"
log "LLM_MODEL=${LLM_MODEL} OLLAMA_BASE_URL=${OLLAMA_BASE_URL}"
log "WHISPER_MODEL=${WHISPER_MODEL}"

if ! curl -fsS --max-time 15 "${OLLAMA_BASE_URL}/api/tags" >/dev/null; then
  log "ERROR: Ollama nicht erreichbar unter ${OLLAMA_BASE_URL}"
  log "Start: ollama serve"
  exit 3
fi

if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "${LLM_MODEL}"; then
  log "Ollama-Modell vorhanden: ${LLM_MODEL}"
else
  log "Pull: ${LLM_MODEL}"
  ollama pull "${LLM_MODEL}"
fi

OUT_DIR="${OUTPUT_DIR:-output}"
mkdir -p "${OUT_DIR}"
log "Transkription starten …"
python transcribe_omni.py --audio "${AUDIO}" --output-dir "${OUT_DIR}"
log "Done."
