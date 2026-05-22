# Runpod Qwen 30b test

Testprojekt für **Qwen-Omni** auf RunPod: Audio transkribieren, **Timecodes** (über Chunk-Offsets + geschätzte Segmentzeiten), Hinweise bei **viel Durcheinander**, und pro Äußerung ob vermutlich **Sprecherwechsel** oder **gleicher Sprecher** folgt.

Analog zum Excel-Korrektur-Skript (`runpod_excel_corrector`): ein CLI, `.env`, `runpod_ready.sh`.

## Wichtig: Timecodes & Omni

- **Qwen-Omni** ist kein Whisper — es liefert **keine** garantierten Wort-für-Wort-Timestamps.
- Dieses Skript schneidet die Datei mit **ffmpeg** in Clips (Standard **30 s**), kennt den **globalen Start** jedes Clips und lässt das Modell **relative** `start_sec` / `end_sec` pro Äußerung schätzen.
- Ergebnis: brauchbare **SRT/Excel-Timecodes** für Tests, aber nicht broadcast-genau.

### Welches Qwen-Omni-Modell?

| Modell (Hugging Face) | Wofür |
|------------------------|--------|
| **`Qwen/Qwen3-Omni-30B-A3B-Instruct`** | **Empfohlen** für Transkription + Dialog (Sprecher, Überlappung, JSON) |
| `Qwen/Qwen3-Omni-30B-A3B-Captioner` | Nur detaillierte **Audio-Beschreibung**, weniger Dialog-Struktur |
| `Qwen/Qwen3-Omni-30B-A3B-Thinking` | Mit „Denk“-Kette, langsamer – nicht nötig für ASR-Tests |
| `Qwen/Qwen2.5-Omni-7B` | Kleiner GPU-Test, weniger VRAM |

Setze in `.env`: `OMNI_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct`

Offizielle Doku: [Qwen3-Omni auf GitHub](https://github.com/QwenLM/Qwen3-Omni) · [HF Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)

## RunPod Setup

```bash
cd runpod_qwen_30b_test
chmod +x runpod_ready.sh
./runpod_ready.sh

cp .env.example .env
# ggf. HF_TOKEN setzen
export OMNI_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct
```

Falls `transformers` das Modell noch nicht kennt, laut [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) die empfohlene Version installieren, z. B.:

```bash
pip install -U "transformers>=4.51.0" qwen-omni-utils
```

## Test-Audio im Repo

`TestAudio.mp3` liegt im Projektordner (Beispiel für RunPod):

```bash
python transcribe_omni.py \
  --audio "TestAudio.mp3" \
  --output-dir "output"
```

## Ausführen (beliebige Datei)

```bash
python transcribe_omni.py \
  --audio "/path/to/dialog.wav" \
  --output-dir "/workspace/out"
```

Ausgabe:

| Datei | Inhalt |
|--------|--------|
| `transcription_result.json` | Chunks + flache `utterances` |
| `transcription_result.xlsx` | Sheet `utterances` + `summary` |
| `transcription_result.srt` | Untertitel; bei Wechsel: `[[ SPRECHERWECHSEL → S2 ]]` |
| `transcription_dialogue.txt` | Lesbares Dialog-Protokoll mit `=== SPRECHERWECHSEL ===` pro Satz |

### Sprecher & Wechsel (pro Satz)

- **`speaker`**: S1, S2, … (vom Modell geschätzt, **kein** pyannote)
- **`speaker_change`**: `true`/`false` — ob dieser Satz mit neuer Stimme beginnt
- **`speaker_change_mark`**: z. B. `SPRECHERWECHSEL (S1 → S2)` in Excel

**Ehrlich:** Omni trennt Stimmen nur so gut wie das Modell sie im Audio unterscheidet; bei viel Überlappung oft ungenau.

### Spalten (Excel)

- `start_tc` / `end_tc` — globale Timecodes  
- `text` — Transkript  
- `speaker` — S1, S2, … (vom Modell im Clip)  
- `boundary_after` — `same_speaker` \| `speaker_change` \| `unknown`  
- `overlap_speech` — `none` \| `mild` \| `heavy` (Chunk)  
- `hard_to_understand` — bool  
- `chunk_notes` — Kurznotiz zum Audioqualität

## Optionen

```bash
python transcribe_omni.py -h

# Längere Clips (mehr Kontext, mehr VRAM):
OMNI_CHUNK_SEC=45 python transcribe_omni.py -a audio.mp3 -o out

# Temp-Chunks behalten:
python transcribe_omni.py -a audio.mp3 -o out --keep-work
```

## VRAM

30B MoE braucht eine **große** RunPod-GPU (z. B. A100 80GB). Bei OOM: kleineres Modell oder kürzere `OMNI_CHUNK_SEC`.
