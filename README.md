# Qwen-Omni Transkription (RunPod / RTX 5090)

## 32 GB: genau diese Schritte

```bash
cd /Runpod-Qwen-30b-Test
git pull
pip install -U bitsandbytes transformers accelerate qwen-omni-utils

cp .env.example .env

# Neuer Prozess — nach OOM alten Python beenden
python transcribe_omni.py --check --check-load
```

Erfolg = `OK: 4-bit quantisiert geladen` und `VRAM: ~18–24 / 31 GB`.

```bash
python transcribe_omni.py --audio TestAudio.mp3 --output-dir output
```

## Wichtig

- **30B auf 32 GB nur mit 4-bit** (`OMNI_LOAD_IN_4BIT=1`, `OMNI_DEVICE_MAP=cuda`)
- **`device_map=auto` vermeiden** — lädt oft unquantisiert bis OOM
- **Talker aus** (`OMNI_ENABLE_AUDIO_OUTPUT=0`) — nur Transkript
- **Audiolänge** (27 min) verursacht **kein** OOM beim Laden

## Fallback

```bash
export OMNI_MODEL_ID=/workspace/models/Qwen2.5-Omni-7B
export OMNI_LOAD_IN_4BIT=0
python transcribe_omni.py --audio audio.mp3 -o output
```
