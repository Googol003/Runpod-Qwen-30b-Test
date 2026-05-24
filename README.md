# Runpod Qwen 30b test

**Qwen-Omni** über **Hugging Face** (`transcribe_omni.py`): Audio in **30‑Sekunden-Chunks**, JSON mit Sprecher/Timecodes.

> **Nicht Ollama** — Ollama kann Qwen-Omni-Audio nicht. Das Hauptprojekt nutzt Ollama nur für **Text-Korrektur** (`llm_module.py`).

## Modell-Check (vor dem Lauf)

```bash
cp .env.example .env   # anpassen
python transcribe_omni.py --check
python transcribe_omni.py --check --check-load   # lädt Gewichte, prüft 4-bit + GPU
```

## 32 GB GPU + 30B (RTX 5090 o.ä.)

**OOM beim Laden?** Das liegt **nicht** an langer Audio (27 min), sondern am Modell in VRAM.

| Phase | Was passiert |
|--------|----------------|
| Schritt 1 | ffmpeg schneidet WAV auf **CPU/Disk** |
| Schritt 2 | **Modell-Gewichte** laden → hier OOM wenn >32 GB Peak |
| Schritt 3 | Pro Chunk ~30 s ans Modell (erst nach erfolgreichem Laden) |

30B-Instruct inkl. Talker ist auf 32 GB knapp. Empfohlen:

```bash
pip install bitsandbytes
cp .env.example .env
# .env: OMNI_LOAD_IN_4BIT=1, OMNI_ENABLE_AUDIO_OUTPUT=0
python transcribe_omni.py --check --check-load
```

Wenn weiter OOM: **Qwen2.5-Omni-7B** (passt sicher) oder größere GPU.

## 7B (schneller, ohne 4-bit)

```bash
export OMNI_MODEL_ID=Qwen/Qwen2.5-Omni-7B
export OMNI_FLASH_ATTN=0
unset OMNI_LOAD_IN_4BIT
python transcribe_omni.py --audio TestAudio.mp3 -o output
```

## Umgebungsvariablen

| Variable | Standard | Bedeutung |
|----------|----------|-----------|
| `OMNI_MODEL_ID` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | HF-Id oder lokaler Ordner |
| `OMNI_LOAD_IN_4BIT` | `0` | `1` = 4-bit NF4 (30B auf 32 GB) |
| `OMNI_NO_CPU_OFFLOAD` | `0` | `1` = Fehler wenn Teile auf CPU |
| `OMNI_FLASH_ATTN` | `1` | `0` auf RunPod ohne flash-attn |
| `OMNI_CHUNK_SEC` | `30` | Chunk-Länge |
| `OMNI_OVERLAP_SEC` | `2` | Überlappung |

## Ausgabe

| Datei | Inhalt |
|--------|--------|
| `transcription_result.json` | Chunks + Äußerungen |
| `transcription_result.xlsx` | Excel |
| `transcription_result.srt` | Untertitel |
| `transcription_dialogue.txt` | Dialog-Protokoll |
