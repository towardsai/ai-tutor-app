"""Shared helpers for the eval runner/grader/report (no app imports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

BATTERY_TYPES = ("singleturn", "sessions", "personas", "replay")

# Context-stats keys that each mean "a compaction mechanism fired this turn".
# Mirrors app.telemetry.COMPACTION_SIGNAL_NAMES (plus the two original
# checkpoint-detected markers). The harness deliberately never imports app code
# so saved bundles re-grade forever; keep the two lists in sync by hand when a
# new mechanism is added.
COMPACTION_SIGNAL_KEYS = (
    "summary_messages",  # SummarizationMiddleware (checkpoint marker)
    "cleared_tool_outputs",  # ContextEditingMiddleware (checkpoint marker)
    "dropped_messages",  # sliding_window / history retrieval (turn signal)
    "truncated_tool_outputs",  # observation_truncation (turn signal)
    "compressed_messages",  # prompt_compression (turn signal)
    "history_retrievals",  # incontext_history_retrieval (turn signal)
)
# context_reset / selective_retention gate via summary_messages (they are
# SummarizationMiddleware prompt variants), so they need no dedicated signal.
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


def compaction_active(stats: dict[str, Any] | None) -> bool:
    """True if any known compaction mechanism fired this turn.

    Takes a bundle's merged ``context_stats`` dict and recognizes both the
    checkpoint-detected markers (summary / cleared tool outputs) and the
    per-call-view middleware signals, so a sliding-window or truncation arm is
    not mis-read as "never compacted" when the probe gate checks it.
    """
    stats = stats or {}
    return any((stats.get(key) or 0) for key in COMPACTION_SIGNAL_KEYS)


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; None on empty input."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[rank]


# Batteries superseded after a validity audit found probe-invalidating defects.
# The frozen files stay on disk (completed runs regrade against them), but new
# runs must target the repaired successor instead.
DEPRECATED_BATTERIES: dict[str, str] = {
    "battery_sessions_v2.jsonl": (
        "the 2026-07-15 audits found its recycled v1 filler injects first-person "
        "persona/update/pivot claims that collide with planted facts in 4 of 6 "
        "sessions (evals.md, harness corrections); use the repaired "
        "battery_sessions_v2_1.jsonl"
    ),
}


def ensure_battery_not_deprecated(path: str | Path, *, override: bool) -> None:
    """Refuse to start a run on a battery superseded for validity defects."""
    reason = DEPRECATED_BATTERIES.get(Path(path).name)
    if reason and not override:
        raise SystemExit(
            f"{Path(path).name} is deprecated for new runs: {reason}. "
            "Pass --allow-deprecated-battery only to reproduce a historical run."
        )
