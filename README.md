# Qwen-Omni auf RunPod (RTX 5090 32 GB)

**30B in 4-bit ≈ 15 GB** — das passt auf 32 GB.  
OOM bei „Loading weights 57%“ = **Lade-Peak** (kurz FP16 auf GPU), nicht die finale Modellgröße.

## Start

```bash
git pull
pip install -U bitsandbytes transformers accelerate qwen-omni-utils
cp .env.example .env

python transcribe_omni.py --check --check-load
python transcribe_omni.py --audio TestAudio.mp3 --output-dir output
```

Erfolg: `OK: 4-bit quantisiert` und `VRAM: ~15–22 / 31 GB`.

## `.env` (wichtig)

| Variable | Wert | Warum |
|----------|------|--------|
| `OMNI_LOAD_IN_4BIT` | `1` | ~15 GB statt ~60 GB |
| `OMNI_LOAD_STRATEGY` | `staged` | kein 31-GB-Lade-Peak |
| `OMNI_ENABLE_AUDIO_OUTPUT` | `0` | Talker nicht laden |

**Nicht** `device_map=cuda` + `dtype=auto` beim Laden — das füllt die GPU mit FP16-Zwischenständen.

## Audiolänge

27 min → viele 30-s-Chunks **nach** dem Laden. Verursacht **kein** OOM in Schritt 2.
