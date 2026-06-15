"""Eval harness for repeatable tutor tests.

Terms:
- battery: a JSONL file of related test questions or conversations.
- bundle: the saved JSON evidence for one tutor turn: input, answer, tools,
  sources, timing, token usage, and errors.
- grade: a metric row computed later from bundles.

Pipeline (see evals.md):
  uv run -m evals.run_battery  -> runs/<exp>/bundles.jsonl   (talks to the app)
  uv run -m evals.grade        -> grades_auto.jsonl + handgrade_sheet.csv (pure)
  uv run -m evals.report       -> report.md + tokens_by_turn.csv (pure)

Only run_battery imports app code; grade/report operate on JSON alone so they
can re-grade old bundles forever without a running system.
"""
