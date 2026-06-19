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


def metric_rows(battery_type: str, runs: list[dict[str, Any]]) -> list[list[str]]:
    """One row per metric, one column per run (preset)."""
    rows: list[list[str]] = []

    def add(name: str, fn) -> None:
        rows.append([name] + [fn(run["grades"]) for run in runs])

    add("cases run", lambda g: str(len({x["unit_id"] for x in g})))
    add("errors", lambda g: str(sum(1 for x in g if x.get("error"))))
    add("time to first text ms p50/p95", lambda g: fmt_ms(col(g, "ttft_ms")))
    add("turn ms p50/p95", lambda g: fmt_ms(col(g, "total_ms")))
    add(
        "input tok/turn (billed, all calls)", lambda g: fmt_mean(col(g, "input_tokens"))
    )
    add(
        "context tokens/turn (window size)",
        lambda g: fmt_mean(col(g, "context_tokens_approx")),
    )
    add("output tok/turn", lambda g: fmt_mean(col(g, "output_tokens")))
    add(
        "est cost/turn $",
        lambda g: fmt_mean(col(g, "est_cost_usd"), "{:.4f}"),
    )
    add("llm calls/turn", lambda g: fmt_mean(col(g, "llm_calls"), "{:.1f}"))
    add("tool calls/turn", lambda g: fmt_mean(col(g, "tool_call_count"), "{:.1f}"))
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
        add("behavior (graded)", lambda g: rate(col(g, "behavior_pass")))
        add("key-point coverage (graded)", lambda g: _kp_coverage(g))

    if battery_type == "sessions":
        add("cumulative input tok (last turn, mean)", _final_cumulative)
        add(
            "compaction active at probes",
            lambda g: rate(
                [x.get("compaction_active") for x in g if x.get("is_probe")]
            ),
        )
        add(
            "probe accuracy (graded)",
            lambda g: rate([x.get("probe_pass") for x in g if x.get("is_probe")]),
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

    if battery_type == "replay":
        add("replay reply pass (human grade)", lambda g: rate(col(g, "replay_pass")))
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
    print(f"Report written to {out_dir}/report.md")


if __name__ == "__main__":
    main()
