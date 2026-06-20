#!/usr/bin/env bash
# SLM knowledge-compaction study: run the same long-lesson question set under
# every compaction method, on each small local model in turn (Ollama), then
# build the cross-model comparison. The point: on a small-context model with no
# caching, "shove it all" (full_context) overflows the window -- so the study
# ranks which compaction method actually wins on a cheap model that fits a Mac.
#
# All metrics are captured per model x method: judge pass (a large-context model
# grades vs the full lesson), context/input/output tokens, latency, ctx overflow,
# and the retrieved context for rag/graphrag. Local models cost $0.
#
# Prereqs: `ollama serve` running; models pulled (ollama pull <model>); the
# Gemini judge key in .env. Reuses the questions from an existing run for
# cross-model comparability.
#
#   bash evals/run_slm_compaction.sh
set -u
cd "$(dirname "$0")/.."

QUESTIONS_FILE="${QUESTIONS_FILE:-data/compaction/questions.jsonl}"
N_QUESTIONS="${N_QUESTIONS:-15}"
NUM_CTX="${NUM_CTX:-32768}"
# "litellm-model-id|out-dir|extra-flags" per model under test.
MODELS=(
  "ollama_chat/llama3.1:8b|data/compaction_slm_llama3.1-8b|"
  "ollama_chat/qwen2.5:7b-instruct|data/compaction_slm_qwen2.5-7b|"
  "ollama_chat/qwen3:8b|data/compaction_slm_qwen3-8b|--reasoning-effort none"
)

OUT_DIRS=()
for spec in "${MODELS[@]}"; do
  IFS='|' read -r model out extra <<<"$spec"
  echo "===== MODEL: $model -> $out ($(date)) ====="
  uv run --env-file .env -m evals.knowledge_compaction \
    --provider ollama --model "$model" --num-ctx "$NUM_CTX" \
    --out "$out" --questions-file "$QUESTIONS_FILE" --questions "$N_QUESTIONS" $extra \
    > "${out}.log" 2>&1
  echo "  exit=$? : $model done ($(date))"
  OUT_DIRS+=("$out")
done

echo "===== cross-model comparison ====="
uv run -m evals.compaction_compare "${OUT_DIRS[@]}" --out runs/slm_compaction_compare
echo "===== DONE ($(date)) ====="
