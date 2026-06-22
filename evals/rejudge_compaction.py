"""Re-grade saved knowledge-compaction bundles with the current judge.

The answers under test are already saved per run, so when the judge changes
(prompt, token budget, parsing) we can re-grade offline -- on the fixed
large-context judge model -- without re-running the (slow, local) models. Useful
after the judge-truncation fix: the original run counted any verdict whose JSON
was cut off at max_tokens as a fail, understating pass rates.

  uv run --env-file .env -m evals.rejudge_compaction \
      data/compaction_slm_llama3.1-8b data/compaction_slm_qwen2.5-7b \
      data/compaction_slm_qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import knowledge_compaction as kc
from .knowledge_compaction import LESSON_PATH, Row, judge, load_lesson, report

logger = logging.getLogger("evals.rejudge_compaction")


def _set_report_model(run_dir: Path) -> None:
    """Point report()'s model header at the model that produced this run (from
    meta.json), not the default; the judge stays on JUDGE_CFG regardless."""
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    kc.CFG = kc.replace(
        kc.PROVIDERS["ollama"],
        model=meta.get("model", kc.CFG.model),
        num_ctx=meta.get("num_ctx"),
    )


def rejudge_dir(run_dir: Path, lesson: str) -> tuple[int, int]:
    _set_report_model(run_dir)
    bundle = run_dir / "bundles.jsonl"
    rows_raw = [
        json.loads(line) for line in bundle.read_text().splitlines() if line.strip()
    ]
    flipped = 0
    rows: list[Row] = []
    for d in rows_raw:
        before = bool(d.get("judge_pass"))
        jp, jr = judge(d["question"], d.get("answer") or "", lesson)
        if jp != before:
            flipped += 1
        d["judge_pass"], d["judge_reason"] = jp, jr
        rows.append(Row(**d))
    bundle.write_text(
        "\n".join(json.dumps(r.__dict__, ensure_ascii=False) for r in rows) + "\n"
    )
    report(rows, run_dir)  # rewrite report.md with the corrected grades
    return flipped, len(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--lesson", default=LESSON_PATH)
    args = parser.parse_args()
    lesson = load_lesson(args.lesson)
    for rd in args.run_dirs:
        run_dir = Path(rd)
        flipped, n = rejudge_dir(run_dir, lesson)
        logger.info("%s: re-graded %d rows, %d verdicts changed", run_dir, n, flipped)


if __name__ == "__main__":
    main()
