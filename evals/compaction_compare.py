"""Cross-model comparison for the SLM knowledge-compaction study.

Aggregates several ``evals.knowledge_compaction`` run dirs (one per model) into a
single table: every model x every compaction method, with all metrics --
judge pass rate (quality), context/input/output tokens, latency p50/p95, ctx
overflow, and $/turn. This is the "which SLM + which compaction method" matrix
for models that run on a small GPU (e.g. an M1 Pro Mac).

  uv run -m evals.compaction_compare \
      data/compaction_slm_llama3.1-8b \
      data/compaction_slm_qwen2.5-7b \
      data/compaction_slm_qwen3-8b \
      --out runs/slm_compaction_compare
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean

from .common import load_jsonl, percentile

logger = logging.getLogger("evals.compaction_compare")


def _model_label(run_dir: Path) -> str:
    meta = run_dir / "meta.json"
    if meta.exists():
        data = json.loads(meta.read_text())
        label = str(data.get("model", run_dir.name))
        if data.get("reasoning_effort") == "none":
            label += " (no-think)"
        return label
    return run_dir.name


def _rows_for(run_dir: Path) -> list[dict]:
    return load_jsonl(run_dir / "bundles.jsonl")


def compare(run_dirs: list[str], out_dir: Path) -> str:
    lines = [
        "# SLM knowledge-compaction: cross-model comparison",
        "",
        "Same lesson, same questions, retrieval/compaction held identical per "
        "method; only the model under test changes. Judge is a fixed "
        "large-context model (it reads the full lesson as ground truth). Local "
        "models cost $0, so read tokens + latency + ctx overflow as the cost.",
        "",
        "| model | method | judge pass | ctx tok | in tok | out tok | "
        "latency p50/p95 s | ctx overflow | $/turn |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    per_model_best: list[str] = []
    for rd in run_dirs:
        run_dir = Path(rd)
        rows = _rows_for(run_dir)
        if not rows:
            logger.warning("no bundles in %s; skipping", run_dir)
            continue
        model = _model_label(run_dir)
        by_strategy: dict[str, list[dict]] = {}
        for r in rows:
            by_strategy.setdefault(r["strategy"], []).append(r)
        order = sorted(
            by_strategy,
            key=lambda s: -sum(1 for r in by_strategy[s] if r.get("judge_pass"))
            / len(by_strategy[s]),
        )
        for s in order:
            rs = by_strategy[s]
            passed = sum(1 for r in rs if r.get("judge_pass"))
            lat = [r["latency_s"] for r in rs if r.get("latency_s")]
            overflow = sum(1 for r in rs if r.get("ctx_overflow"))
            ov_cell = f"{overflow}/{len(rs)}" if overflow else "-"
            lines.append(
                f"| {model} | {s} | {passed}/{len(rs)} ({passed / len(rs):.0%}) | "
                f"{mean(r['context_tokens'] for r in rs):.0f} | "
                f"{mean(r['input_tokens'] for r in rs):.0f} | "
                f"{mean(r['output_tokens'] for r in rs):.0f} | "
                f"{percentile(lat, 50):.1f}/{percentile(lat, 95):.1f} | {ov_cell} | "
                f"${mean(r['cost_usd'] for r in rs):.4f} |"
            )
        # Best method = highest pass rate, ties broken by lower latency.
        best = max(
            by_strategy,
            key=lambda s: (
                sum(1 for r in by_strategy[s] if r.get("judge_pass")) / len(by_strategy[s]),
                -percentile([r["latency_s"] for r in by_strategy[s]], 50),
            ),
        )
        bp = sum(1 for r in by_strategy[best] if r.get("judge_pass")) / len(by_strategy[best])
        per_model_best.append(f"- **{model}**: best method `{best}` ({bp:.0%} pass)")

    lines += ["", "## Best compaction method per model", "", *per_model_best, ""]
    out = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.md").write_text(out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="knowledge_compaction out dirs")
    parser.add_argument("--out", default="runs/slm_compaction_compare")
    args = parser.parse_args()
    print(compare(args.run_dirs, Path(args.out)))


if __name__ == "__main__":
    main()
