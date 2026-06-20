#!/usr/bin/env bash
# Axis-A SLM study: the memory-compaction methods (how to compact a GROWING
# conversation history) on a small local model forced to compact. Companion to
# the Axis-B doc-compaction study (evals/run_slm_compaction.sh).
#
# Setup that makes it a real test:
#   - The long-context session battery: turn 0 loads the ~37.5k-token lesson,
#     turns 1..N ask questions, retrieval OFF (--no-tools) so the agent must
#     answer from whatever the memory preset KEPT in context.
#   - A small window (a num_ctx=32768 Ollama model variant), so full_history
#     overflows and every compaction preset is forced to fire.
#   - Each preset runs the real app middlewares via run_battery, then
#     compaction_study judges each answer vs the lesson (Gemini) and reports.
#
# Prereqs: `ollama serve`; build the 32k variant once, e.g.
#   printf 'FROM qwen2.5:7b-instruct\nPARAMETER num_ctx 32768\n' | ollama create qwen2.5-7b-ctx32k -f /dev/stdin
# Build the battery once: uv run --env-file .env -m evals.compaction_study build --questions 15
#
#   MODEL=ollama:qwen2.5-7b-ctx32k TAG=qwen2.5-7b bash evals/run_slm_axis_a.sh
set -u
cd "$(dirname "$0")/.."

# Local experiment: skip LangSmith upload (quota noise; we use context_stats).
export LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false

BATTERY="${BATTERY:-data/compaction/longctx_session.jsonl}"
MODEL="${MODEL:-ollama:qwen2.5-7b-ctx32k}"   # a num_ctx Ollama variant
TAG="${TAG:-qwen2.5-7b}"                       # short label for out dirs
# The Axis-A memory-compaction methods (full_history is the keep-all baseline).
PRESETS="${PRESETS:-full_history sliding_window summarization_only prompt_compression selective_retention incontext_history_retrieval delta_summarization hierarchical_summarization}"

for p in $PRESETS; do
  out="runs/axisa_${TAG}_${p}"
  echo "===== preset: $p -> $out ($(date)) ====="
  uv run --env-file .env -m evals.run_battery --battery "$BATTERY" --preset "$p" \
    --model "$MODEL" --no-tools --concurrency 1 --out "$out" > "${out}.log" 2>&1
  echo "  exit=$? : $p done ($(date))"
done

echo "===== report ($(date)) ====="
# Family A only (these ARE the in-context presets); skip the Family-B retrieval join.
uv run --env-file .env -m evals.compaction_study report \
  --runs "runs/axisa_${TAG}_*" --family-b "" --out "runs/axisa_${TAG}_report"
echo "===== DONE ($(date)) ====="
