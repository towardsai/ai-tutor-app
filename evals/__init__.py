"""Eval harness: run batteries against the tutor, grade bundles, report.

Pipeline (see evals.md):
  uv run -m evals.run_battery  -> runs/<exp>/bundles.jsonl   (talks to the app)
  uv run -m evals.grade        -> grades_auto.jsonl + handgrade_sheet.csv (pure)
  uv run -m evals.report       -> report.md + tokens_by_turn.csv (pure)

Only run_battery imports app code; grade/report operate on JSON alone so they
can re-grade old bundles forever without a running system.
"""
