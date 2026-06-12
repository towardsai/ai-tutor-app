"""Shared helpers for the eval runner/grader/report (no app imports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

BATTERY_TYPES = ("singleturn", "sessions", "personas", "replay")
# Identifying key per battery type (see data/eval/README.md schemas).
_TYPE_KEYS = {
    "case_id": "singleturn",
    "session_id": "sessions",
    "persona_id": "personas",
    "replay_id": "replay",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def detect_battery_type(records: list[dict[str, Any]]) -> str:
    if not records:
        raise ValueError("Battery file is empty.")
    for key, battery_type in _TYPE_KEYS.items():
        if key in records[0]:
            return battery_type
    raise ValueError(
        f"Unrecognized battery schema; expected one of {sorted(_TYPE_KEYS)} "
        f"in the first record, got {sorted(records[0])}"
    )


def normalize_url(url: str | None) -> str:
    """Comparable form for lesson-URL matching (recall ground truth).

    Battery `lesson_url`s point at the discussion on the lesson page
    (`.../<lesson>/discussions/<post_id>`); retrieval matches carry the bare
    lesson URL, so the discussion suffix is stripped before comparing.
    """
    if not url:
        return ""
    url = url.strip().lower().split("#", 1)[0].split("?", 1)[0]
    url = url.split("/discussions/", 1)[0]
    return url.rstrip("/")


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; None on empty input."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[rank]
