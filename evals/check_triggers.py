"""Verify memory compaction happened before session probe turns.

  uv run -m evals.check_triggers --runs runs/a4_sessions_prod
  uv run -m evals.check_triggers --runs runs/a4_s03_fullhist --expect-none

Compaction means the app summarized older messages or cleared old tool output.
A memory eval where compaction never activated does not test memory trimming,
so this is the gate before comparing presets:
- default: every graded probe turn must show compaction in context_stats
  (summary_messages or cleared_tool_outputs > 0); exit 1 otherwise.
- --expect-none (for full_history): no turn may show compaction; exit 1 if
  the baseline ever compressed, because that would invalidate the comparison.

Also prints per-session cost/token totals for Part B/C budget projection.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import compaction_active as _compaction_active
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
    """Any compaction mechanism (summary/clear or a per-call-view signal) fired."""
    return _compaction_active(bundle.get("context_stats"))


def check_run(
    run_dir: Path,
    expect_none: bool,
    *,
    min_compactions: int = 0,
    max_compactions: int | None = None,
    min_summary_input: int = 0,
    expected_trigger_tokens: int = 0,
    first_pre_tokens_min: int = 0,
    first_pre_tokens_max: int = 0,
) -> bool:
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
        compaction_events = [
            (b["turn_index"], event)
            for b in rows
            for event in ((b.get("context_stats") or {}).get("compaction_events") or [])
        ]
        signaled_compactions = sum(
            int((b.get("context_stats") or {}).get("compactions_this_turn") or 0)
            for b in rows
        )
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
        event_problems: list[str] = []
        if signaled_compactions != len(compaction_events):
            event_problems.append(
                "compaction signal/event mismatch "
                f"({signaled_compactions} != {len(compaction_events)})"
            )
        if len(compaction_events) < min_compactions:
            event_problems.append(
                f"only {len(compaction_events)} compactions; need {min_compactions}"
            )
        if max_compactions is not None and len(compaction_events) > max_compactions:
            event_problems.append(
                f"{len(compaction_events)} compactions; max {max_compactions}"
            )
        if min_summary_input:
            too_small = [
                turn
                for turn, event in compaction_events
                if int(event.get("summary_input_tokens_approx") or 0)
                < min_summary_input
            ]
            if too_small:
                event_problems.append(
                    f"summary input < {min_summary_input:,} at turns {too_small}"
                )
        if expected_trigger_tokens:
            wrong_trigger = [
                (turn, event.get("configured_trigger_tokens"))
                for turn, event in compaction_events
                if int(event.get("configured_trigger_tokens") or 0)
                != expected_trigger_tokens
            ]
            if wrong_trigger:
                event_problems.append(
                    f"configured trigger != {expected_trigger_tokens:,}: "
                    f"{wrong_trigger}"
                )
        if compaction_events and (first_pre_tokens_min or first_pre_tokens_max):
            first_event = compaction_events[0][1]
            first_tokens = max(
                int(first_event.get("pre_compaction_tokens_approx") or 0),
                int(first_event.get("trigger_reported_tokens") or 0),
            )
            if first_pre_tokens_min and first_tokens < first_pre_tokens_min:
                event_problems.append(
                    f"first compaction at {first_tokens:,} < {first_pre_tokens_min:,}"
                )
            if first_pre_tokens_max and first_tokens > first_pre_tokens_max:
                event_problems.append(
                    f"first compaction at {first_tokens:,} > {first_pre_tokens_max:,}"
                )
        if event_problems:
            ok = False
            verdict += " | EVENT FAIL " + "; ".join(event_problems)
        if errors:
            ok = False
            verdict += f" | ERRORS at turns {errors}"
        print(
            f"{session_id} t{trial} [{preset}] turns={len(rows)} "
            f"first_summary@{first_summary} first_clear@{first_clear} "
            f"compactions={len(compaction_events)} "
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
    parser.add_argument("--min-compactions", type=int, default=0)
    parser.add_argument("--max-compactions", type=int)
    parser.add_argument("--min-summary-input", type=int, default=0)
    parser.add_argument("--expected-trigger-tokens", type=int, default=0)
    parser.add_argument("--first-pre-tokens-min", type=int, default=0)
    parser.add_argument("--first-pre-tokens-max", type=int, default=0)
    args = parser.parse_args()
    # Check every run before aggregating: all() over a bare generator would
    # short-circuit on the first failing run and silently skip the later runs'
    # checks and per-session diagnostics (the exit code would be right, but the
    # output would hide where else the gate failed).
    results = [
        check_run(
            Path(r),
            args.expect_none,
            min_compactions=args.min_compactions,
            max_compactions=args.max_compactions,
            min_summary_input=args.min_summary_input,
            expected_trigger_tokens=args.expected_trigger_tokens,
            first_pre_tokens_min=args.first_pre_tokens_min,
            first_pre_tokens_max=args.first_pre_tokens_max,
        )
        for r in args.runs
    ]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
