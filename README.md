# Qwen3-Omni 30B auf RunPod (5090 32GB)

## Download — richtig

`Qwen3-Omni-30B-A3B-Instruct` auf Disk = BF16-Gewichte (~60 GB).  
**4-bit** erzeugt **bitsandbytes beim Laden** — kein separates 4bit-Modell.

## OOM bei ~57 % / ~31 GB VRAM — echte Ursache

Dein Log (`core_model_loading.py` → `tensor.to(device=cuda)`) passt zu **Transformers ≥ 5.0**:

Gewichte werden **unquantisiert auf die GPU** kopiert, **danach** soll 4-bit greifen — auf 32 GB reicht das nicht.

| Symptom | Bedeutung |
|--------|-----------|
| ~31 GB bei 57 % | FP16-Ladepeak, **kein** 4-bit |
| 1407 Shards, Talker aus | Download/Lade-Pfad OK |
| 20 MiB OOM am Ende | GPU schon voll |

**Nicht** Audio-Länge, **nicht** falscher Download.

## Fix auf RunPod (zuerst)

```bash
git pull
bash scripts/fix_deps_runpod.sh
cp .env.example .env   # enthält HF_DEACTIVATE_ASYNC_LOAD=1

python scripts/load_like_gist.py
```

**Erfolg:** `OK: 4-bit` und `VRAM: ~15–20 / 31 GB`

Manuell:

```bash
pip install 'transformers>=4.51.0,<5.0.0'
export HF_DEACTIVATE_ASYNC_LOAD=1
python -c "import transformers; print(transformers.__version__)"   # muss 4.x sein
```

**Nicht:** `pip install -U transformers` (zieht 5.x nach).

## Laden (Gist, RTX 3090 24GB)

```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
from_pretrained(..., dtype="auto", device_map="cuda", quantization_config=...)
enable_audio_output=False
```

Gist nutzt zusätzlich `flash_attention_2` — optional:

```bash
pip install -U flash-attn --no-build-isolation
# .env: OMNI_FLASH_ATTN=1
```

## Transkription

```bash
python transcribe_omni.py --audio TestAudio.mp3 --output-dir output
```

## Referenzen

- [HF transformers #43032](https://github.com/huggingface/transformers/issues/43032) — 4-bit OOM, GPU vor Quantisierung
- [HF transformers #44387](https://github.com/huggingface/transformers/issues/44387) — `HF_DEACTIVATE_ASYNC_LOAD=1`
- [Gist phhusson](https://gist.github.com/phhusson/4bc8851935ff1caafd3a7f7ceec34335)
