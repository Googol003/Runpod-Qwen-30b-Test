from __future__ import annotations

import json
from typing import Any, Dict


def parse_json_response(text: str) -> Dict[str, Any]:
    """Robust gegen ```json```-Fences und Text um das JSON herum."""
    t = (text or "").strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        try:
            t2 = json.loads(t)
            if isinstance(t2, str) and t2.strip():
                t = t2.strip()
        except Exception:
            pass
    if t.startswith("```"):
        t = t.strip("`")
        if "\n" in t:
            t = t.split("\n", 1)[1].strip()
    if not t.startswith("{"):
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start : end + 1]
    if '""' in t and '"items"' not in t:
        t = t.replace('""', '"')
    return json.loads(t)
