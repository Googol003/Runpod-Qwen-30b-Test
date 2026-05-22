from __future__ import annotations

from typing import Any, Dict, List


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "ja"):
        return True
    if s in ("0", "false", "no", "nein"):
        return False
    return None


def annotate_speaker_changes(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pro Satz: speaker_change + Markierung für Excel/SRT (auch über Chunk-Grenzen)."""
    if not rows:
        return rows
    sorted_rows = sorted(rows, key=lambda r: (float(r.get("start_sec", 0)), int(r.get("chunk_index", 0))))
    prev_speaker = ""
    for i, r in enumerate(sorted_rows):
        sp = str(r.get("speaker", "") or "").strip()
        boundary_prev = (
            str(sorted_rows[i - 1].get("boundary_after", "")).strip().lower()
            if i > 0
            else ""
        )
        model_flag = _as_bool(r.get("speaker_change_at_start"))

        change = False
        if model_flag is True:
            change = True
        elif model_flag is False:
            change = False
        elif i == 0:
            change = False
        elif boundary_prev == "speaker_change":
            change = True
        elif prev_speaker and sp and sp != prev_speaker:
            change = True

        prev_label = prev_speaker or "—"
        r["speaker_change"] = change
        if change:
            r["speaker_change_mark"] = f"SPRECHERWECHSEL ({prev_label} → {sp or '?'})"
        else:
            r["speaker_change_mark"] = ""
        if sp:
            prev_speaker = sp
    return sorted_rows


def format_dialogue_txt(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for r in rows:
        mark = str(r.get("speaker_change_mark", "") or "").strip()
        sp = str(r.get("speaker", "") or "?")
        tc = str(r.get("start_tc", ""))
        text = str(r.get("text", "")).strip()
        if mark:
            lines.append(f"=== {mark} [{tc}] ===")
        lines.append(f"[{sp}] {text}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
