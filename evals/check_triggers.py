"""Verify compression triggers fired where the session battery assumes.

  uv run -m evals.check_triggers --runs runs/a4_sessions_prod
  uv run -m evals.check_triggers --runs runs/a4_s03_fullhist --expect-none

A memory eval where compaction never activated measures nothing (see
data/eval/README.md), so this is the gate before any bake-off:
- default: every probe turn must have compaction active (summary_messages or
  cleared_tool_outputs > 0 in context_stats); exit 1 otherwise.
- --expect-none (for full_history): NO turn may show compaction; exit 1 if
  the baseline ever compressed — that would invalidate the comparison.

Also prints per-session cost/token totals for Part B/C budget projection.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import load_jsonl


def session_table(bundles: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict]]:
    sessions: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for bundle in bundles:
        if bundle.get("turn_index") is None:
            continue
        sessions[(bundle["unit_id"], bundle["trial"])].append(bundle)
    for rows in sessions.values():
        rows.sort(key=lambda b: b["turn_index"])
    return sessions


def probe_indices(battery_path: str, session_id: str) -> list[int]:
    for record in load_jsonl(battery_path):
        if record.get("session_id") == session_id:
            return [probe["turn_index"] for probe in record.get("probes", [])]
    return []


def compaction_active(bundle: dict[str, Any]) -> bool:
    stats = bundle.get("context_stats") or {}
    return bool(
        (stats.get("summary_messages") or 0) or (stats.get("cleared_tool_outputs") or 0)
    )


def check_run(run_dir: Path, expect_none: bool) -> bool:
    bundles = load_jsonl(run_dir / "bundles.jsonl")
    ok = True
    for (session_id, trial), rows in sorted(session_table(bundles).items()):
        preset = rows[0]["preset"]
        probes = set(probe_indices(rows[0]["battery_path"], session_id))
        first_summary = next(
            (
                b["turn_index"]
                for b in rows
                if (b.get("context_stats") or {}).get("summary_messages")
            ),
            None,
        )
        first_clear = next(
            (
                b["turn_index"]
                for b in rows
                if (b.get("context_stats") or {}).get("cleared_tool_outputs")
            ),
            None,
        )
        tokens = sum(
            (b.get("context_stats") or {}).get("input_tokens") or 0 for b in rows
        )
        cost = sum(
            (b.get("context_stats") or {}).get("est_cost_usd") or 0 for b in rows
        )
        errors = [b["turn_index"] for b in rows if b.get("error")]
        bad_probes = []
        compressed_turns = [b["turn_index"] for b in rows if compaction_active(b)]
        if expect_none:
            if compressed_turns:
                ok = False
                verdict = f"FAIL compaction at turns {compressed_turns} (expected none)"
            else:
                verdict = "OK no compaction"
        else:
            bad_probes = [
                b["turn_index"]
                for b in rows
                if b["turn_index"] in probes and not compaction_active(b)
            ]
            if bad_probes:
                ok = False
                verdict = f"FAIL probes without compaction: {bad_probes}"
            elif not probes:
                verdict = "OK (no probes found?)"
            else:
                verdict = "OK all probes under compaction"
        if errors:
            ok = False
            verdict += f" | ERRORS at turns {errors}"
        print(
            f"{session_id} t{trial} [{preset}] turns={len(rows)} "
            f"first_summary@{first_summary} first_clear@{first_clear} "
            f"input_tok={tokens:,} est_cost=${cost:.2f} -> {verdict}"
        )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument(
        "--expect-none",
        action="store_true",
        help="Assert NO compaction anywhere (full_history baseline).",
    )
    args = parser.parse_args()
    ok = all(check_run(Path(r), args.expect_none) for r in args.runs)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
