from __future__ import annotations

SYSTEM_PROMPT = """Du transkribierst Audio und markierst Sprecherwechsel.
Antwort: **nur** gültiges JSON (kein Markdown, kein Kommentar, keine Metadaten außer dem Schema).

- Wörtliche Transkription (Deutsch, sofern gesprochen).
- **Eine Äußerung = ein Satz** (oder kurzer Turn).
- ``speaker``: S1, S2, … — gleiche Stimme = gleiches Label im Clip.
- ``speaker_change_at_start``: **true** nur wenn diese Zeile mit einer **anderen** Stimme beginnt als die vorherige; erste Zeile im Clip: **false**.
- Zeiten ``start_sec`` / ``end_sec`` relativ zum Clip (0 = Clip-Start).
- Keine Qualitätsbewertung, keine Notizen, kein ``boundary_after``.
"""

USER_PROMPT_TEMPLATE = """Transkribiere diesen Audio-Ausschnitt.

Datei: {source_name} — Ausschnitt {chunk_index}/{chunk_total}, Sek. {chunk_start_sec:.2f}–{chunk_end_sec:.2f} (Dauer {chunk_duration_sec:.2f} s).

Nur dieses JSON:

{{
  "chunk_index": {chunk_index},
  "chunk_start_sec": {chunk_start_sec},
  "chunk_end_sec": {chunk_end_sec},
  "utterances": [
    {{
      "start_sec": 0.0,
      "end_sec": 0.0,
      "text": "transkribierter Satz",
      "speaker": "S1",
      "speaker_change_at_start": false
    }}
  ]
}}

Regeln:
- Leere ``utterances`` nur wenn keine Sprache hörbar.
- Zeiten innerhalb [0, {chunk_duration_sec:.2f}].
- Nur transkribieren + Sprecher + ``speaker_change_at_start`` — sonst nichts.
"""

# Ollama-Pfad: Whisper-Transkript + LLM (wie runpod_excel_corrector)
SYSTEM_PROMPT_OLLAMA = """Du bist ein präziser Dialog-Analyse-Assistent.
Du erhältst ein automatisches Transkript (Whisper) eines kurzen Audio-Ausschnitts (~30 s).
Liefere **nur** ein gültiges JSON-Objekt (kein Markdown, kein Kommentar).

Wichtig:
- Nutze den Whisper-Text als Basis; korrigiere nur offensichtliche ASR-Fehler, erfinde nichts.
- **Eine Äußerung = ein Satz** (oder ein kurzer Turn).
- **Sprecher schätzen** (S1, S2, …) aus Kontext/Turn-Wechsel — kein echtes Diarisierungs-Modell.
- **Zeitangaben** relativ zum Clip (0 = Clip-Start); orientiere dich an den Whisper-Zeilen [start–end].
- Kennzeichne **überlappendes** Sprechen ehrlich in ``audio_quality``.
"""

USER_PROMPT_TEMPLATE_OLLAMA = """Strukturiere diesen Audio-Ausschnitt als JSON.

Kontext (gesamte Datei):
- Dateiname: {source_name}
- Ausschnitt {chunk_index}/{chunk_total}: Sekunden {chunk_start_sec:.2f} – {chunk_end_sec:.2f} (Dauer {chunk_duration_sec:.2f} s)

Automatisches Transkript (Whisper, Zeiten relativ zum Clip):
{whisper_transcript}

Antworte **ausschließlich** mit diesem JSON-Schema:

{{
  "chunk_index": {chunk_index},
  "chunk_start_sec": {chunk_start_sec},
  "chunk_end_sec": {chunk_end_sec},
  "audio_quality": {{
    "overlap_speech": "none|mild|heavy",
    "hard_to_understand": false,
    "notes": "kurz auf Deutsch"
  }},
  "utterances": [
    {{
      "start_sec": 0.0,
      "end_sec": 0.0,
      "text": "transkribierter Satz",
      "speaker": "S1",
      "speaker_change_at_start": false,
      "boundary_after": "same_speaker|speaker_change|unknown"
    }}
  ]
}}

Regeln für ``utterances``:
- Leere Liste nur wenn Whisper „keine Sprache“ meldet.
- ``start_sec`` / ``end_sec`` innerhalb [0, {chunk_duration_sec:.2f}].
- ``speaker``: S1, S2, … — gleiche vermutete Person = gleiches Label im Clip.
- ``speaker_change_at_start``: true wenn neue Stimme vs. vorherige Äußerung; erste Zeile immer false.
- ``boundary_after``: speaker_change | same_speaker | unknown.
- Crosstalk: mehrere Zeilen + ``overlap_speech`` = heavy.
"""
