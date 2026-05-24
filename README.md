# Qwen3-Omni 30B auf RunPod (5090 32GB)

## Download — hast du richtig

`Qwen3-Omni-30B-A3B-Instruct` auf Disk = **normal** (BF16/FP16 Gewichte).  
**4-bit** kommt von **bitsandbytes beim Laden** — kein falsches Modell.

## Laden — exakt wie Gist (RTX 3090 24GB, bewiesen)

```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
from_pretrained(..., dtype="auto", device_map="cuda", quantization_config=...)
enable_audio_output=False   # nur Text / Transkript
```

## Befehle

```bash
git pull
pip install -U bitsandbytes transformers accelerate qwen-omni-utils
cp .env.example .env

python scripts/load_like_gist.py
# oder:
python transcribe_omni.py --check --check-load
```

**Erfolg:** `OK: 4-bit` und `VRAM: ~15–20 / 31 GB`  
**Fehler:** `VRAM: ~31 GB` beim Laden → 4-bit greift nicht → siehe Flash-Attn unten.

## Wenn OOM bei ~57% / 31 GB VRAM

Das ist **kein** „Modell zu groß“, sondern **Laden ohne echte 4-bit-Quantisierung**.

1. Neuer Terminal-Tab (VRAM leer)
2. `pip install -U flash-attn --no-build-isolation` (offizielle Qwen-Empfehlung, spart VRAM)
3. In `.env`: `OMNI_FLASH_ATTN=1`
4. Nochmal `python scripts/load_like_gist.py`

## Transkription

```bash
python transcribe_omni.py --audio TestAudio.mp3 --output-dir output
```
