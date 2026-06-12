"""Build the comparison report from one or more graded runs.

  uv run -m evals.report --runs runs/bake1_singleturn_prod \
      runs/bake1_singleturn_full_history --out runs/bake1_report

Each run dir = one battery x preset x model. Runs sharing a battery_type are
shown side by side (columns = presets) — the bake-off table. Also emits
tokens_by_turn.csv (cumulative input tokens per session turn per preset: the
signature plot) and tokens_by_turn.png when matplotlib is available.

Reads grades_merged.jsonl when present (hand grades included), else
grades_auto.jsonl; quality rows (key-point coverage, probe accuracy, behavior
accuracy) appear only once hand grades are merged.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .common import load_jsonl, percentile


def load_run(run_dir: Path) -> dict[str, Any]:
    grades_path = run_dir / "grades_merged.jsonl"
    if not grades_path.exists():
        grades_path = run_dir / "grades_auto.jsonl"
    grades = load_jsonl(grades_path)
    if not grades:
        raise SystemExit(f"No grades in {run_dir}; run evals.grade first.")
    return {
        "dir": run_dir,
        "label": f"{grades[0]['preset']}",
        "battery_type": grades[0]["battery_type"],
        "model": grades[0]["model"],
        "grades": grades,
        "merged": grades_path.name == "grades_merged.jsonl",
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
    add("ttft ms p50/p95", lambda g: fmt_ms(col(g, "ttft_ms")))
    add("turn ms p50/p95", lambda g: fmt_ms(col(g, "total_ms")))
    add("input tok/turn", lambda g: fmt_mean(col(g, "input_tokens")))
    add("output tok/turn", lambda g: fmt_mean(col(g, "output_tokens")))
    add(
        "est cost/turn $",
        lambda g: fmt_mean(col(g, "est_cost_usd"), "{:.4f}"),
    )
    add("llm calls/turn", lambda g: fmt_mean(col(g, "llm_calls"), "{:.1f}"))
    add("tool calls/turn", lambda g: fmt_mean(col(g, "tool_call_count"), "{:.1f}"))

    if battery_type == "singleturn":
        add(
            "retrieval called (corpus)",
            lambda g: rate(
                [
                    x.get("called_retrieval")
                    for x in g
                    if x.get("expected_behavior") == "answer_from_corpus"
                ]
            ),
        )
        add("recall@shown source", lambda g: rate(col(g, "recall_source")))
        add("recall@shown lesson", lambda g: rate(col(g, "recall_lesson")))
        add(
            "MRR lesson", lambda g: fmt_mean([x.get("mrr_lesson") for x in g], "{:.2f}")
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
        add("behavior heuristic", lambda g: rate(col(g, "behavior_heuristic")))
        add("behavior (hand)", lambda g: rate(col(g, "behavior_pass")))
        add("key-point coverage (hand)", lambda g: _kp_coverage(g))

    if battery_type == "sessions":
        add("cumulative input tok (last turn, mean)", _final_cumulative)
        add(
            "compaction active at probes",
            lambda g: rate(
                [x.get("compaction_active") for x in g if x.get("is_probe")]
            ),
        )
        add(
            "probe accuracy (hand)",
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
        add("replay reply pass (hand)", lambda g: rate(col(g, "replay_pass")))
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
    ax.set_title("Context cost per turn by memory preset")
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
        f"Model(s): {', '.join(sorted(models))}. Runs missing hand grades show "
        "— for quality rows (fill handgrade_sheet.csv, re-run evals.grade "
        "with --handgrades)."
    )
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_type[run["battery_type"]].append(run)
    for battery_type, type_runs in by_type.items():
        lines.append(f"\n## {battery_type}\n")
        header = ["metric"] + [
            run["label"] + ("" if run["merged"] else " (auto only)")
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
