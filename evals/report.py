"""Build a side-by-side report from one or more graded runs.

  uv run -m evals.report --runs runs/bake1_singleturn_prod \
      runs/bake1_singleturn_full_history --out runs/bake1_report

Each run directory is one battery run with one model and one memory preset.
Runs that share a battery type are shown side by side, with presets as
columns. The command also emits tokens_by_turn.csv, a per-turn session token
curve, and tokens_by_turn.png when matplotlib is available.

Reads grades_merged.jsonl when present (human grades included), else
grades_auto.jsonl; quality rows (key-point coverage, probe accuracy, behavior
accuracy) appear only once human grades are merged.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .common import load_jsonl, percentile


def run_label(run_dir: Path, preset: str) -> str:
    """Column label = preset, disambiguated by flag-based arms that share a
    preset (e.g. retrieval-budget / kb-off both run on preset=prod)."""
    label = preset
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("disable_kb"):
            label += "+kb_off"
        if cfg.get("retrieval_budget"):
            label += f"+rb{int(cfg['retrieval_budget']) // 1000}k"
        if cfg.get("retriever") in ("graphrag", "classical"):
            label += f"+{cfg['retriever']}"
    return label


def load_run(run_dir: Path) -> dict[str, Any]:
    grades_path = run_dir / "grades_merged.jsonl"
    if not grades_path.exists():
        grades_path = run_dir / "grades_auto.jsonl"
    grades = load_jsonl(grades_path)
    if not grades:
        raise SystemExit(f"No grades in {run_dir}; run evals.grade first.")
    # Grade source for the column header: judge / human / mixed (None if no
    # merged grades). Stops the report from labeling judge grades "(human)".
    sources = {g.get("grade_source") for g in grades if g.get("grade_source")}
    grade_source = (
        sources.pop() if len(sources) == 1 else ("mixed" if sources else None)
    )
    return {
        "dir": run_dir,
        "label": run_label(run_dir, grades[0]["preset"]),
        "battery_type": grades[0]["battery_type"],
        "model": grades[0]["model"],
        "grades": grades,
        "bundles": load_jsonl(run_dir / "bundles.jsonl"),
        "merged": grades_path.name == "grades_merged.jsonl",
        "grade_source": grade_source,
    }


def rate(values: list[Any]) -> str:
    """pass-rate over boolean-like values, ignoring None."""
    decided = [v for v in values if v is not None]
    if not decided:
        return "—"
    return f"{sum(1 for v in decided if v) / len(decided):.0%} (n={len(decided)})"


def fmt_ms(values: list[Any]) -> str:
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return "—"
    return f"{percentile(nums, 50):.0f} / {percentile(nums, 95):.0f}"


def fmt_mean(values: list[Any], spec: str = "{:.0f}") -> str:
    nums = [v for v in values if isinstance(v, (int, float))]
    return spec.format(mean(nums)) if nums else "—"


def col(grades: list[dict[str, Any]], key: str) -> list[Any]:
    return [g.get(key) for g in grades]


def _cache_hit_rate(grades: list[dict[str, Any]]) -> str:
    hits = sum(int(g.get("cache_read_tokens") or 0) for g in grades)
    inputs = sum(int(g.get("input_tokens") or 0) for g in grades)
    return f"{hits / inputs:.1%}" if inputs else "—"


def _trajectory_sum_mean(grades: list[dict[str, Any]], key: str) -> str:
    totals: dict[tuple[str, int], float] = defaultdict(float)
    for grade in grades:
        value = grade.get(key)
        if isinstance(value, (int, float)):
            totals[(grade["unit_id"], grade["trial"])] += float(value)
    return f"{mean(totals.values()):.4f}" if totals else "—"


def _trajectory_difference_mean(
    grades: list[dict[str, Any]], positive_key: str, negative_key: str
) -> str:
    totals: dict[tuple[str, int], float] = defaultdict(float)
    for grade in grades:
        key = (grade["unit_id"], grade["trial"])
        totals[key] += float(grade.get(positive_key) or 0)
        totals[key] -= float(grade.get(negative_key) or 0)
    return f"{mean(totals.values()):.0f}" if totals else "—"


def metric_rows(battery_type: str, runs: list[dict[str, Any]]) -> list[list[str]]:
    """One row per metric, one column per run (preset)."""
    rows: list[list[str]] = []

    def add(name: str, fn) -> None:
        rows.append([name] + [fn(run["grades"]) for run in runs])

    def src_tag(source_key: str) -> str:
        """Per-metric provenance tag across the shown runs: [judge]/[human]/[mixed].

        Lets a single report mix human-graded rows (e.g. Part B key-points,
        session probes) with judge-graded rows (the new holistic/faithfulness)
        and label each honestly, even when a run's overall grade_source is mixed.
        """
        seen = {
            g.get(source_key)
            for run in runs
            for g in run["grades"]
            if g.get(source_key)
        }
        if not seen:
            return ""
        return f" [{seen.pop()}]" if len(seen) == 1 else " [mixed]"

    add("cases run", lambda g: str(len({x["unit_id"] for x in g})))
    add("errors", lambda g: str(sum(1 for x in g if x.get("error"))))
    # --- Work & cost: run-time-INDEPENDENT, the unconfounded headline ----------
    # These depend only on the request/response, not on when the arm ran, so
    # they (not the latency rows below) are the basis for any efficiency claim.
    # The compaction re-work that drives latency shows up here as extra
    # calls/tokens (F9/F27), without the time-of-run noise.
    add("llm calls/turn", lambda g: fmt_mean(col(g, "llm_calls"), "{:.1f}"))
    add("tool calls/turn", lambda g: fmt_mean(col(g, "tool_call_count"), "{:.1f}"))
    add(
        "input tok/turn (billed, all calls)", lambda g: fmt_mean(col(g, "input_tokens"))
    )
    add(
        "context tokens/turn (window size)",
        lambda g: fmt_mean(col(g, "context_tokens_approx")),
    )
    add("output tok/turn", lambda g: fmt_mean(col(g, "output_tokens")))
    add("cache hit ratio (all input)", _cache_hit_rate)
    add(
        "est cost/turn $",
        lambda g: fmt_mean(col(g, "est_cost_usd"), "{:.4f}"),
    )
    add(
        "model cost/trajectory $ (mean)",
        lambda g: _trajectory_sum_mean(g, "est_cost_usd"),
    )
    add(
        "summarization cost/trajectory $ (mean)",
        lambda g: _trajectory_sum_mean(g, "summarization_cost_usd"),
    )
    add(
        "compactions/trajectory (mean)",
        lambda g: _trajectory_sum_mean(g, "compactions_this_turn"),
    )
    add(
        "tool outputs capped/trajectory (mean)",
        lambda g: _trajectory_sum_mean(g, "tool_outputs_capped"),
    )
    add(
        "tool-output bytes removed/trajectory (mean)",
        lambda g: _trajectory_difference_mean(
            g, "tool_output_original_bytes", "tool_output_retained_bytes"
        ),
    )
    add(
        "max model request context tokens",
        lambda g: (
            str(max(int(x.get("max_request_context_tokens_approx") or 0) for x in g))
            if g
            else "—"
        ),
    )
    # --- Latency: TIME-OF-RUN CONFOUNDED (sequential arms meet different API
    # load) -> supporting, not headline; `total - ttft` does NOT de-confound (it
    # is only the answer-streaming tail). See the runtime note + evals.md F27.
    # time-to-first-token (first reasoning/tool/text token) is only present on
    # runs recorded after that telemetry was added; older runs show "—".
    if any(
        g.get("time_to_first_token_ms") is not None
        for run in runs
        for g in run["grades"]
    ):
        add(
            "time to first token ms p50/p95 [confounded]",
            lambda g: fmt_ms(col(g, "time_to_first_token_ms")),
        )
    add(
        "time to first text ms p50/p95 [confounded]",
        lambda g: fmt_ms(col(g, "ttft_ms")),
    )
    add("turn ms p50/p95 [confounded]", lambda g: fmt_ms(col(g, "total_ms")))
    if any(
        g.get("history_embedding_texts") is not None
        for run in runs
        for g in run["grades"]
    ):
        add(
            "history embed inputs/turn",
            lambda g: fmt_mean(col(g, "history_embedding_texts"), "{:.1f}"),
        )
        add(
            "history embed chars/turn",
            lambda g: fmt_mean(col(g, "history_embedding_chars"), "{:.0f}"),
        )

    if battery_type == "singleturn":
        add(
            "grounding tool called (corpus)",
            lambda g: rate(
                [
                    x.get("called_retrieval")
                    for x in g
                    if x.get("expected_behavior") == "answer_from_corpus"
                ]
            ),
        )
        add(
            "recall@shown source (retrieval only)",
            lambda g: rate(col(g, "recall_source")),
        )
        add(
            "recall@shown lesson (retrieval only)",
            lambda g: rate(col(g, "recall_lesson")),
        )
        add(
            "right lesson rank (MRR)",
            lambda g: fmt_mean([x.get("mrr_lesson") for x in g], "{:.2f}"),
        )
        add(
            "recall source (any tool: retrieval+KB)",
            lambda g: rate(col(g, "recall_anytool_source")),
        )
        add(
            "citation present",
            lambda g: rate(
                [
                    x.get("has_citation")
                    for x in g
                    if x.get("expected_behavior") == "answer_from_corpus"
                ]
            ),
        )
        add(
            "cited correct source (answer, corpus)",
            lambda g: rate(
                [
                    x.get("cited_correct_source")
                    for x in g
                    if x.get("expected_behavior") == "answer_from_corpus"
                ]
            ),
        )
        add(
            "cited correct lesson (answer, corpus)",
            lambda g: rate(
                [
                    x.get("cited_correct_lesson")
                    for x in g
                    if x.get("expected_behavior") == "answer_from_corpus"
                ]
            ),
        )
        add(
            "behavior proxy from code checks",
            lambda g: rate(col(g, "behavior_heuristic")),
        )
        add(
            f"behavior (graded){src_tag('behavior_source')}",
            lambda g: rate(col(g, "behavior_pass")),
        )
        add(
            f"key-point coverage (graded){src_tag('key_points_source')}",
            lambda g: _kp_coverage(g),
        )
        add(
            f"holistic: staff-approval (graded){src_tag('holistic_source')}",
            lambda g: rate(col(g, "holistic_pass")),
        )
        # NOTE: faithfulness-to-evidence is intentionally not reported. On bundles
        # recorded under the old 6k tool-output cap, KB-browse evidence is
        # truncated, so the judge can't see what the agent grounded on and the
        # score tracks capture-completeness, not grounding (evals.md F23/F24
        # class). grade.py only emits it once evidence is captured in full; re-add
        # the row after raising TOOL_OUTPUT_MAX_CHARS and re-recording a run.

    if battery_type == "sessions":
        add("cumulative input tok (last turn, mean)", _final_cumulative)
        add(
            "compaction active at probes",
            lambda g: rate(
                [x.get("compaction_active") for x in g if x.get("is_probe")]
            ),
        )
        add(
            f"probe accuracy (graded){src_tag('probe_source')}",
            lambda g: rate([x.get("probe_pass") for x in g if x.get("is_probe")]),
        )
        # Answer quality across all session turns (not just probes): does holistic
        # quality / faithfulness degrade under compaction over a long session?
        add(
            f"holistic: staff-approval (graded){src_tag('holistic_source')}",
            lambda g: rate(col(g, "holistic_pass")),
        )
        for probe_type in sorted(
            {
                x.get("probe_type")
                for run in runs
                for x in run["grades"]
                if x.get("probe_type")
            }
        ):
            rows.append(
                [f"  └ {probe_type}"]
                + [
                    rate(
                        [
                            x.get("probe_pass")
                            for x in run["grades"]
                            if x.get("probe_type") == probe_type
                        ]
                    )
                    for run in runs
                ]
            )

    if battery_type == "personas":
        add("personalization pass (auto)", lambda g: rate(col(g, "auto_pass")))
        add(
            "anti-pattern failures",
            lambda g: str(sum(1 for x in g if x.get("anti_pattern_hits"))),
        )
        add(
            f"holistic: staff-approval (graded){src_tag('holistic_source')}",
            lambda g: rate(col(g, "holistic_pass")),
        )

    if battery_type == "replay":
        add(
            f"replay reply pass (graded){src_tag('replay_source')}",
            lambda g: rate(col(g, "replay_pass")),
        )
    return rows


def _kp_coverage(grades: list[dict[str, Any]]) -> str:
    passed = sum(g.get("key_points_passed", 0) for g in grades)
    total = sum(g.get("key_points_total", 0) for g in grades)
    return f"{passed / total:.0%} ({passed}/{total})" if total else "—"


def _final_cumulative(grades: list[dict[str, Any]]) -> str:
    finals: dict[tuple[str, int], int] = {}
    for g in grades:
        tokens = g.get("input_tokens")
        if g.get("turn_index") is None or not isinstance(tokens, (int, float)):
            continue
        key = (g["unit_id"], g["trial"])
        finals[key] = finals.get(key, 0) + int(tokens)
    return fmt_mean(list(finals.values()))


def write_token_curves(runs: list[dict[str, Any]], out_dir: Path) -> None:
    session_runs = [r for r in runs if r["battery_type"] == "sessions"]
    if not session_runs:
        return
    rows = []
    for run in session_runs:
        cumulative: dict[tuple[str, int], int] = defaultdict(int)
        for g in sorted(
            run["grades"],
            key=lambda x: (x["unit_id"], x["trial"], x["turn_index"] or 0),
        ):
            if g.get("turn_index") is None:
                continue
            key = (g["unit_id"], g["trial"])
            cumulative[key] += int(g.get("input_tokens") or 0)
            rows.append(
                {
                    "preset": run["label"],
                    "session_id": g["unit_id"],
                    "trial": g["trial"],
                    "turn_index": g["turn_index"],
                    "turn_input_tokens": g.get("input_tokens") or 0,
                    "cumulative_input_tokens": cumulative[key],
                }
            )
    path = out_dir / "tokens_by_turn.csv"
    with open(path, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        _plot_token_curves(rows, out_dir)
    except ImportError:
        print("matplotlib not installed; wrote CSV only.")


def _plot_token_curves(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_preset: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_preset[row["preset"]][row["turn_index"]].append(
            row["cumulative_input_tokens"]
        )
    fig, ax = plt.subplots(figsize=(8, 5))
    for preset, by_turn in sorted(by_preset.items()):
        turns = sorted(by_turn)
        ax.plot(turns, [mean(by_turn[t]) for t in turns], marker="o", label=preset)
    ax.set_xlabel("turn")
    ax.set_ylabel("cumulative input tokens (mean across sessions)")
    ax.set_title("Input tokens accumulated by memory preset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tokens_by_turn.png", dpi=150)


def write_cost_curves(runs: list[dict[str, Any]], out_dir: Path) -> None:
    """Write cached + uncached + output cumulative model-cost trajectories."""
    session_runs = [run for run in runs if run["battery_type"] == "sessions"]
    if not session_runs:
        return
    rows: list[dict[str, Any]] = []
    for run in session_runs:
        cumulative: dict[tuple[str, int], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for bundle in sorted(
            run["bundles"],
            key=lambda item: (
                item["unit_id"],
                item["trial"],
                item.get("turn_index") or 0,
            ),
        ):
            if bundle.get("turn_index") is None:
                continue
            stats = bundle.get("context_stats") or {}
            cost = stats.get("cost_breakdown") or {}
            components_available = bool(cost)
            key = (bundle["unit_id"], bundle["trial"])
            values = {
                "cache_read_input_usd": float(cost.get("cache_read_input_usd") or 0),
                "cache_miss_input_usd": float(cost.get("cache_miss_input_usd") or 0),
                "cache_creation_input_usd": float(
                    cost.get("cache_creation_input_usd") or 0
                ),
                "output_usd": float(cost.get("output_usd") or 0),
                "total_usd": float(
                    cost.get("total_usd")
                    if cost.get("total_usd") is not None
                    else stats.get("est_cost_usd") or 0
                ),
                "summarization_usd": float(stats.get("summarization_cost_usd") or 0),
            }
            for name, value in values.items():
                cumulative[key][name] += value
            rows.append(
                {
                    "preset": run["label"],
                    "cost_components_available": components_available,
                    "session_id": bundle["unit_id"],
                    "trial": bundle["trial"],
                    "turn_index": bundle["turn_index"],
                    **{f"turn_{name}": value for name, value in values.items()},
                    **{f"cumulative_{name}": cumulative[key][name] for name in values},
                    "input_tokens": stats.get("input_tokens") or 0,
                    "cache_read_tokens": stats.get("cache_read_tokens") or 0,
                    "cache_miss_tokens": stats.get("cache_miss_tokens") or 0,
                    "active_context_tokens_approx": stats.get("context_tokens_approx")
                    or 0,
                    "max_request_context_tokens_approx": stats.get(
                        "max_request_context_tokens_approx"
                    )
                    or 0,
                    "compactions_this_turn": stats.get("compactions_this_turn") or 0,
                    "tool_outputs_capped": stats.get("tool_outputs_capped") or 0,
                    "tool_output_original_bytes": stats.get(
                        "tool_output_original_bytes"
                    )
                    or 0,
                    "tool_output_retained_bytes": stats.get(
                        "tool_output_retained_bytes"
                    )
                    or 0,
                }
            )
    if not rows:
        return
    path = out_dir / "trajectory_cost_by_turn.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        _plot_cost_curves(rows, out_dir)
    except ImportError:
        print("matplotlib not installed; wrote cost CSV only.")


def _plot_cost_curves(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_preset: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_preset[row["preset"]][row["turn_index"]].append(row["cumulative_total_usd"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for preset, by_turn in sorted(by_preset.items()):
        turns = sorted(by_turn)
        ax.plot(turns, [mean(by_turn[t]) for t in turns], marker="o", label=preset)
    ax.set_xlabel("turn")
    ax.set_ylabel("cumulative model cost, USD (mean)")
    ax.set_title("Cached + cache-miss + output cost by trajectory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_cost_by_turn.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="", help="Report dir (default: first run)")
    args = parser.parse_args()
    runs = [load_run(Path(r)) for r in args.runs]
    out_dir = Path(args.out or runs[0]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Eval report", ""]
    models = {run["model"] for run in runs}
    lines.append(
        f"Model(s): {', '.join(sorted(models))}. Runs missing human grades show "
        "— for quality rows (fill handgrade_sheet.csv, re-run evals.grade "
        "with --handgrades)."
    )
    lines.append(
        "\n**Runtime read:** treat **llm calls/turn, tool calls/turn, and "
        "tokens/turn** as the efficiency headline — they are run-time-independent. "
        "The **latency rows are marked `[confounded]`**: when arms run "
        "sequentially their seconds reflect API load at that moment, not just the "
        "arm (evals.md F27). `total - ttft` does not fix this (it is only the "
        "answer-streaming tail). Run arms interleaved or repeat trials before "
        "quoting latency."
    )
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_type[run["battery_type"]].append(run)
    for battery_type, type_runs in by_type.items():
        lines.append(f"\n## {battery_type}\n")
        header = ["metric"] + [
            run["label"]
            + ("" if run["merged"] else " (auto only)")
            + (
                f" [{run['grade_source']}]"
                if run["merged"] and run.get("grade_source")
                else ""
            )
            for run in type_runs
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in metric_rows(battery_type, type_runs):
            lines.append("| " + " | ".join(row) + " |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    write_token_curves(runs, out_dir)
    write_cost_curves(runs, out_dir)
    print(f"Report written to {out_dir}/report.md")


if __name__ == "__main__":
    main()
