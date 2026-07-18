# Manual End-to-End LangSmith Runbook

This runbook verifies the live chatbot path end to end and then looks up the
matching LangSmith trace/run from the CLI.

Use it when changing agent tools, KB browsing, citation handling, streaming API
parts, or model/provider settings.

## Prerequisites

Run commands from the repository root.

Required local artifacts and environment:

- `.env` contains `COHERE_API_KEY`
- `.env` contains `DEEPSEEK_API_KEY`
- `.env` contains `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `.env` contains `LANGSMITH_API_KEY`
- `.env` has `LANGSMITH_TRACING=true`
- `.env` has `LANGSMITH_PROJECT=ai-tutor-app`
- `data/chroma-db-all_sources/` exists
- `data/kb/wiki/index.md` exists
- `curl`, `jq`, and the `langsmith` CLI are available through `uv run`

Quick check:

```bash
uv run dotenv -f .env run -- python - <<'PY'
import os
for key in [
    "COHERE_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
]:
    value = os.getenv(key)
    print(f"{key}={'set' if value else 'missing'}")
PY
```

## Fast Automated Smoke Tests

These tests run a live model-backed request and check the frontend/API stream
shape.

API stream contract:

```bash
uv run dotenv -f .env run -- env RUN_LIVE_API_E2E=1 \
  pytest tests/test_api.py::test_live_api_stream_exposes_frontend_parts -q
```

## Manual API Test

Start the FastAPI app in one terminal:

```bash
uv run dotenv -f .env run -- env \
  AI_TUTOR_API_HOST=127.0.0.1 \
  AI_TUTOR_API_PORT=8000 \
  python -m app.api
```

From another terminal, confirm the server is ready:

```bash
curl -s http://127.0.0.1:8000/healthz
```

Create a request payload. This prompt intentionally asks for both structured
retrieval and KB shell browsing so both citation paths are exercised.

```bash
cat >/tmp/ai_tutor_e2e_payload.json <<'JSON'
{
  "messages": [
    {
      "role": "user",
      "parts": [
        {
          "type": "text",
          "text": "Use run_kb_command to answer: can you tell me about codex?"
        }
      ]
    }
  ],
  "sourceKeys": [
    "agentic_ai_engineering",
    "langchain",
    "langgraph",
    "peft",
    "transformers"
  ],
  "enabledTools": [],
  "model": "deepseek:deepseek-v4-flash",
  "includeReasoning": true,
  "threadId": ""
}
JSON
```

On `enabledTools`: it is an explicit allowlist of the toggle tools for the turn.
`[]` (as above) disables web search and URL reading, so this run exercises only
the corpus retrieval + KB shell citation paths. Two things to know:

- **`url_context` defaults to off in the UI** (`active: False` in `_tool_catalog`,
  `app/api.py`), so a browser turn only sends it when the user opts in. Enabling
  it flips `keep_unresolved_sources=True` for that turn, which surfaces cited
  http(s) URLs that match no evidence as low-trust **Web** chips (`group="web"`).
  With it off, such URLs are dropped from the source cards (they still render as
  plain links in the answer prose).
- **Omitting `enabledTools` entirely is not the same as `[]`.** A direct API
  caller that drops the field falls through to a server default that enables
  *every* toggle tool regardless of the UI `active` flag (including
  `url_context`). Send an explicit list to control this; see `build_chat_request`
  in `app/api.py`.

Send the request and save the server-sent event stream:

```bash
curl -sN --max-time 240 http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d @/tmp/ai_tutor_e2e_payload.json \
  -o /tmp/ai_tutor_e2e_api.sse
```

Convert the stream to JSONL parts:

```bash
grep '^data: ' /tmp/ai_tutor_e2e_api.sse \
  | sed 's/^data: //' \
  | grep -v '^\[DONE\]$' \
  > /tmp/ai_tutor_e2e_api.parts.jsonl
```

Inspect what the frontend received:

```bash
jq -r '.type' /tmp/ai_tutor_e2e_api.parts.jsonl | sort | uniq -c
```

Inspect tool calls:

```bash
jq -r '
  select(.type == "tool-input-available" or .type == "tool-input-start")
  | [.type, .toolName, ((.input // .inputText // "") | tostring | gsub("\n"; " ") | .[0:220])]
  | @tsv
' /tmp/ai_tutor_e2e_api.parts.jsonl
```

Inspect final answer text:

```bash
jq -r 'select(.type == "text-delta") | .delta' \
  /tmp/ai_tutor_e2e_api.parts.jsonl \
  | tr -d '\n'
echo
```

Inspect source cards:

```bash
jq -r '
  select(.type == "data-source")
  | [.data.title, .data.url, .data.sourceKey, (.data.score // "")]
  | @tsv
' /tmp/ai_tutor_e2e_api.parts.jsonl
```

Expected result:

- The stream includes `tool-input-*`, `tool-output-available`, `text-delta`,
  `source-url`, `source-document`, `data-source`, and `finish` parts.
- Tool calls include `retrieve_tutor_context` and `run_kb_command`.
- Every `retrieve_tutor_context` tool run has a nested
  `Hybrid Retrieval Pipeline` span. Expanding it in the waterfall shows
  separate Cohere embed, Chroma, dense hydration, BM25, RRF, Cohere rerank,
  and token-budget runs.
- The answer has inline citations, not only a final sources list.
- `data-source` parts include the source cards the frontend will render.

## Find The LangSmith Trace

List recent traces in the project:

```bash
uv run dotenv -f .env run -- langsmith trace list \
  --project ai-tutor-app \
  --limit 10 \
  --full \
  --format json \
  -o /tmp/ai_tutor_langsmith_traces.json
```

Print the likely match. Look for the newest trace whose input contains the
manual prompt.

```bash
jq -r '
  .[]
  | [
      (.trace_id // .id),
      (.status // ""),
      (.duration_ms // .latency_ms // .latency // ""),
      ((.inputs.query // .inputs.input // .inputs.messages[-1].content // .inputs // "") | tostring | gsub("\n"; " ") | .[0:220])
    ]
  | @tsv
' /tmp/ai_tutor_langsmith_traces.json
```

Save the trace ID:

```bash
TRACE_ID="paste-trace-id-here"
```

Fetch the full trace:

```bash
uv run dotenv -f .env run -- langsmith trace get "$TRACE_ID" \
  --project ai-tutor-app \
  --full \
  --format json \
  -o "/tmp/ai_tutor_trace_${TRACE_ID}.json"
```

Fetch the corresponding run. For the root run, the trace ID is usually also
the run ID. If this fails, open the trace JSON and use the root run ID shown
there.

```bash
uv run dotenv -f .env run -- langsmith run get "$TRACE_ID" \
  --project ai-tutor-app \
  --full \
  --format json \
  -o "/tmp/ai_tutor_run_${TRACE_ID}.json"
```

## Inspect Trace/Run Details

Overall status and latency:

```bash
jq '{
  id: (.trace_id // .id),
  name,
  status,
  duration_ms: (.duration_ms // .latency_ms // .latency),
  run_count: (.run_count // ((.runs // .child_runs // []) | length))
}' "/tmp/ai_tutor_trace_${TRACE_ID}.json"
```

Tool calls:

```bash
jq -r '
  .. | objects
  | select((.name? // "") | test("retrieve_tutor_context|run_kb_command"))
  | [
      (.name // ""),
      (.status // ""),
      (.duration_ms // .latency_ms // .latency // ""),
      ((.inputs.command // .inputs.query // .inputs.input // "") | tostring | gsub("\n"; " ") | .[0:220])
    ]
  | @tsv
' "/tmp/ai_tutor_trace_${TRACE_ID}.json"
```

Hybrid retrieval latency breakdown:

```bash
jq -r '
  .runs[]
  | select(.name == "Hybrid Retrieval Pipeline")
  | (.custom_metadata.retrieval_timing
      // .extra.metadata.retrieval_timing
      // {}) as $timing
  | [
      (.inputs.query // ""),
      ($timing.status // ""),
      ($timing.filter_mode // ""),
      ($timing.total_ms // ""),
      ($timing.stage_ms.embed_ms // ""),
      ($timing.stage_ms.chroma_ms // ""),
      ($timing.stage_ms.dense_hydration_ms // ""),
      ($timing.stage_ms.bm25_ms // ""),
      ($timing.stage_ms.fusion_ms // ""),
      ($timing.stage_ms.rerank_ms // ""),
      ($timing.stage_ms.token_budget_ms // "")
    ]
  | @tsv
' "/tmp/ai_tutor_trace_${TRACE_ID}.json"
```

The columns are query, status, filter mode, total, embed, Chroma, dense-result
hydration, BM25, RRF, rerank, and token-budget latency, all in milliseconds.
The backend logs the same values as one `retrieval_timing` line without logging
the raw student query. A complete source selection should report
`filter_mode=all_sources_omitted`; real source subsets should report
`single_source` or `source_subset`.

In the LangSmith waterfall, expand `retrieve_tutor_context`, then
`Hybrid Retrieval Pipeline`, to see the same stages as individually timed child
runs. Their names are `Cohere Embed`, `Chroma Vector Search`,
`Dense Result Hydration`, `BM25 Search`, `RRF Fusion`, `Cohere Rerank`, and
`Token Budget`.

LLM calls:

```bash
jq -r '
  .. | objects
  | select((.run_type? // "") | test("llm|chat_model"))
  | [
      (.name // ""),
      (.status // ""),
      (.duration_ms // .latency_ms // .latency // ""),
      (.prompt_tokens // .usage.prompt_tokens // .outputs.llm_output.token_usage.prompt_tokens // ""),
      (.completion_tokens // .usage.completion_tokens // .outputs.llm_output.token_usage.completion_tokens // "")
    ]
  | @tsv
' "/tmp/ai_tutor_trace_${TRACE_ID}.json"
```

Search the raw trace for the KB shell command sequence:

```bash
rg -n "run_kb_command|retrieve_tutor_context|sed -n|rg -n|cat wiki" \
  "/tmp/ai_tutor_trace_${TRACE_ID}.json"
```

For latency debugging, compare these numbers:

- Root trace duration.
- Count of `run_kb_command` calls.
- Total shell/tool duration.
- The `Hybrid Retrieval Pipeline` stage breakdown for every retrieval call.
- Count of LLM/chat model calls.
- Longest LLM call duration.
- Whether the trace ended in `success`, `error`, or cancellation.

If `Hybrid Retrieval Pipeline` is fast but total trace duration is large, the
bottleneck is model inference or repeated model turns. If retrieval is slow,
its child runs and metadata identify whether the time was spent in Cohere
embedding, Chroma, BM25/RRF, Cohere reranking, or token budgeting.
