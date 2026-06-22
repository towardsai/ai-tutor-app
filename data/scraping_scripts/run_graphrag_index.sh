#!/usr/bin/env bash
# Build the GraphRAG index (experiment: GraphRAG vs classical RAG).
#
# graphrag aborts the whole extract_graph stage if a single Gemini call fails
# after its retries (transient 503 / broken pipe happen under load), but it
# CACHES every successful call -- so re-running resumes and only retries what's
# missing, at no extra cost. This wrapper loops the index until entities.parquet
# exists (or MAX_ATTEMPTS is hit).
#
# Prereqs: a valid GEMINI_API_KEY (indexing LLM, 2.5 Flash) and OPENAI_API_KEY
# (embeddings) in .env. Cohere is NOT needed to index. The graph DB is
# local-only (data/graphrag/output) and never uploaded to the HF bundle.
#
# Usage:
#   data/scraping_scripts/graphrag_prep_input.py --sources full_stack_ai_engineering
#   bash data/scraping_scripts/run_graphrag_index.sh
set -u

ROOT="${GRAPHRAG_ROOT:-data/graphrag}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
OUT="$ROOT/output/entities.parquet"

for i in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== graphrag index attempt $i/$MAX_ATTEMPTS ==="
  uv run --env-file .env graphrag index --root "$ROOT"
  if [ -f "$OUT" ]; then
    echo "SUCCESS: $OUT present after attempt $i"
    exit 0
  fi
  echo "attempt $i did not produce entities.parquet; retrying (cache resumes)"
  sleep 5
done

echo "FAILED: no entities.parquet after $MAX_ATTEMPTS attempts" >&2
exit 1
