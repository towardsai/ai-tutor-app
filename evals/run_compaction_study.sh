#!/usr/bin/env bash
# Run the compaction study: the same long-context session under each memory
# preset, on Gemini 2.5 Flash, with NO tools (answer from retained context only).
# Then judge + report with: uv run --env-file .env -m evals.compaction_study report --runs 'runs/compaction_*'
#
#   uv run --env-file .env -m evals.compaction_study build --questions 15
#   bash evals/run_compaction_study.sh
set -u

BATTERY="${BATTERY:-data/compaction/longctx_session.jsonl}"
MODEL="${MODEL:-google-genai:gemini-2.5-flash}"
PRESETS="${PRESETS:-full_history prod summarization_only sliding_window prompt_compression selective_retention incontext_history_retrieval aggressive}"

for p in $PRESETS; do
  echo "=== preset: $p ==="
  uv run --env-file .env -m evals.run_battery --battery "$BATTERY" --preset "$p" \
    --model "$MODEL" --no-tools --concurrency 1 --out "runs/compaction_$p"
done
echo "=== ALL PRESETS DONE ==="
