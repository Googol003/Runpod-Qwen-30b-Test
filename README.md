# Runpod Qwen 30b test

**Transkription mit 30‑Sekunden-Chunks** — Standard wie `runpod_excel_corrector`:

1. **Whisper** (`faster-whisper`) transkribiert jeden Chunk auf der GPU (schnell, effizient).
2. **Ollama** strukturiert das Ergebnis als JSON (Sprecher, Timecodes im Clip, Überlappung).

Ollama kann **kein** Qwen-Omni-Audio direkt — deshalb diese Zweistufen-Pipeline.

## RunPod Setup

```bash
# Terminal 1
ollama serve

# Terminal 2
cd /workspace/Runpod-Qwen-30b-Test
cp .env.example .env
chmod +x runpod_ready.sh
bash runpod_ready.sh "/workspace/TestAudio.mp3"
```

Oder manuell:

```bash
export LLM_MODEL=gemma4:26b
export WHISPER_MODEL=large-v3   # oder medium für weniger VRAM
python transcribe_omni.py --audio TestAudio.mp3 --output-dir output
```

## Umgebungsvariablen (wie Excel Corrector)

| Variable | Standard | Bedeutung |
|----------|----------|-----------|
| `LLM_PROVIDER` | `ollama` | `ollama` oder OpenAI-compat |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama-Server |
| `LLM_MODEL` | `gemma4:26b` | Ollama-Modell für JSON |
| `LLM_NUM_CTX` | `8192` | Kontext |
| `LLM_TIMEOUT_S` | `180` | HTTP-Timeout |
| `WHISPER_MODEL` | `large-v3` | ASR-Modell |
| `WHISPER_DEVICE` | `cuda` | GPU für Whisper |
| `OMNI_CHUNK_SEC` | `30` | Chunk-Länge |
| `OMNI_OVERLAP_SEC` | `2` | Überlappung |

## Ausgabe

| Datei | Inhalt |
|--------|--------|
| `transcription_result.json` | Chunks + Äußerungen |
| `transcription_result.xlsx` | Excel |
| `transcription_result.srt` | Untertitel |
| `transcription_dialogue.txt` | Dialog-Protokoll |

## Optional: Hugging Face Qwen-Omni

Nur wenn du das volle Omni-Modell willst (langsam auf 32 GB):

```bash
pip install torch transformers qwen-omni-utils accelerate soundfile
export TRANSCRIBE_BACKEND=hf
export OMNI_MODEL_ID=Qwen/Qwen2.5-Omni-7B
export OMNI_FLASH_ATTN=0
python transcribe_omni.py --audio TestAudio.mp3 -o output --backend hf
```

## 27‑Minuten-Datei

~58 Chunks à 30 s → 58× (Whisper + Ollama). Mit `large-v3` + `gemma4:26b` deutlich schneller als 30B HF-Omni mit CPU-Offload.
