from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from speaker_markers import annotate_speaker_changes, format_dialogue_txt


def _sec_to_tc(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def flatten_utterances(
    chunk_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Chunk-JSON → flache Zeilen mit globalen Timecodes."""
    rows: List[Dict[str, Any]] = []
    for cr in chunk_results:
        chunk_start = float(cr.get("chunk_start_sec", 0.0))
        chunk_end = float(cr.get("chunk_end_sec", chunk_start))
        for u in cr.get("utterances") or []:
            if not isinstance(u, dict):
                continue
            rel_start = float(u.get("start_sec", 0.0))
            rel_end = float(u.get("end_sec", rel_start))
            g_start = chunk_start + rel_start
            g_end = chunk_start + rel_end
            if g_end > chunk_end + 0.05:
                g_end = chunk_end
            text = str(u.get("text", "")).strip()
            if not text:
                continue
            rows.append(
                {
                    "start_sec": round(g_start, 3),
                    "end_sec": round(g_end, 3),
                    "start_tc": _sec_to_tc(g_start),
                    "end_tc": _sec_to_tc(g_end),
                    "text": text,
                    "speaker": str(u.get("speaker", "")),
                    "speaker_change_at_start": u.get("speaker_change_at_start"),
                    "chunk_index": int(cr.get("chunk_index", 0)),
                }
            )
    return annotate_speaker_changes(rows)


def write_outputs(
    *,
    out_dir: Path,
    source_audio: Path,
    model_id: str,
    chunk_results: List[Dict[str, Any]],
    raw_responses: List[Dict[str, Any]],
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = flatten_utterances(chunk_results)

    payload = {
        "source_audio": str(source_audio),
        "model_id": model_id,
        "timecode_note": (
            "Zeitstempel stammen aus Chunk-Offsets plus vom Modell geschätzte "
            "relative Zeiten im Clip — nicht wie Whisper word-level."
        ),
        "speaker_note": (
            "Sprecher S1/S2/… und speaker_change kommen vom Modell (geschätzt), "
            "nicht von professioneller Diarisierung. Spalte speaker_change_mark "
            "pro Satz bei erkanntem Wechsel."
        ),
        "chunks": chunk_results,
        "utterances": rows,
        "raw_model_responses": raw_responses,
    }
    json_path = out_dir / "transcription_result.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    xlsx_path = out_dir / "transcription_result.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="utterances", index=False)
        summary = {
            "key": ["source_audio", "model_id", "utterance_count", "chunk_count"],
            "value": [
                str(source_audio),
                model_id,
                len(rows),
                len(chunk_results),
            ],
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)

    srt_path = out_dir / "transcription_result.srt"
    lines: List[str] = []
    for i, r in enumerate(rows, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_tc(r['start_sec'])} --> {_srt_tc(r['end_sec'])}")
        sp = r.get("speaker") or "?"
        if r.get("speaker_change"):
            prefix = f"[[ SPRECHERWECHSEL → {sp} ]] "
        else:
            prefix = f"[{sp}] "
        lines.append(prefix + str(r["text"]))
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")

    dialogue_path = out_dir / "transcription_dialogue.txt"
    dialogue_path.write_text(format_dialogue_txt(rows), encoding="utf-8")

    return {
        "json": json_path,
        "xlsx": xlsx_path,
        "srt": srt_path,
        "dialogue": dialogue_path,
    }


def _srt_tc(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
