# GraphRAG vs classical RAG (experiment)

A single, scoped comparison of the production hybrid retriever against a true
Microsoft GraphRAG index, on the same eval battery and the same chat model, to
get a defensible answer to "does GraphRAG beat our classical RAG for the tutor?"

GraphRAG was previously **dropped** from the Part C plan ("low information given
the findings", `evals.md`). This revisits it deliberately, with a fair head-to-head
rather than on faith. It lives entirely on branch `experiment/graphrag-vs-rag`
and is fully opt-in: the default retriever and all production behavior are
unchanged.

## What varies, what's held constant

The **only** variable is the retrieval backend behind `retrieve_tutor_context`:

- **classical** — the production hybrid `LocalChromaRetriever` (Cohere `embed-v4`
  dense + BM25 -> RRF -> Cohere rerank -> token budget).
- **graphrag** — `GraphRAGRetriever` (`app/graph_rag.py`): local-search-style
  context assembly over a prebuilt GraphRAG index. It embeds the query (OpenAI
  `text-embedding-3-small`, same model used to embed entities at index time),
  finds the nearest entities in the entity LanceDB table, gathers the text units
  those entities were extracted from (mapped back to the real `source`/`url` via
  each text unit's `document_id` -> `corpus_manifest`), adds the top community
  reports as context, then applies the **same Cohere rerank + token budget** as
  classical so the comparison is fair.

Held constant: the **chat model is Gemini 3.5 Flash** (the agent that writes the
answer, `run_battery` default), the system prompt, the battery, the token budget,
and source scoping. The GraphRAG retriever is a **context provider only** -- it
never runs GraphRAG's own LLM answer-synthesis, so the agent's 3.5 Flash is the
sole generation model in both arms.

Two models, two roles:
- **Build the graph DB with Gemini 2.5 Flash** (cheap offline indexing).
- **Run the eval with Gemini 3.5 Flash** (same baseline as every other eval run).

## Scope (and why)

A full-corpus GraphRAG index (3,197 docs / 8.3M tokens) measured out to roughly
**$2,000** on 2.5 Flash -- GraphRAG cost is dominated by community-report
generation and scales with the graph, not linearly with a cheap-per-token rate.
To get one non-negligible comparison within a ~$60 budget, the index is **scoped
to `full_stack_ai_engineering`** -- the source behind **41 of the 60** single-turn
cases (27 of them `answer_from_corpus`, where retrieval recall is the live
question). Both arms run on those 41 cases, both restricted to that source
(`--scope-sources`), so it is a clean same-corpus comparison.

Measured course-content indexing rate: **~$88 / 1M corpus tokens** (vs ~$255 for
dense API-reference docs). full_stack = 486k tokens -> **~$43** to index.

## Build the graph DB

Local-only; never uploaded to the HF vector-db bundle (the `data/graphrag/`
generated dirs are gitignored). Indexing needs `GEMINI_API_KEY` (2.5 Flash) and
`OPENAI_API_KEY` (embeddings); Cohere is **not** needed to index.

```bash
# 1. corpus -> graphrag jsonl input (id = doc_id, so text units map back to the manifest)
uv run -m data.scraping_scripts.graphrag_prep_input --sources full_stack_ai_engineering

# 2. build (loops until entities.parquet exists; cache makes re-runs free)
bash data/scraping_scripts/run_graphrag_index.sh
```

Config: `data/graphrag/settings.yaml` (tracked). Notable choices:
- **Gemini 2.5 Flash via the OpenAI-compatible endpoint**
  (`model_provider: openai`, `api_base: .../v1beta/openai`). The native litellm
  gemini transport drops connections (`Server disconnected` / 503) under
  sustained extraction and aborts the whole `extract_graph` stage even at
  concurrency 1; the OpenAI-client path is stable.
- Domain entity types (library/framework/model/concept/method/...), not the
  default organization/person/geo/event.
- `concurrent_requests: 6`, hard retries, and the `run_graphrag_index.sh`
  retry-loop: graphrag aborts the stage if one call exhausts retries, but caches
  every success, so re-running resumes.

<!-- INDEX-STATS: filled from the real full_stack index -->

## Run the comparison

Blocked only by Cohere being available (both arms use Cohere embed/rerank; the
classical vector store is Cohere-embedded so this cannot be swapped without
changing the baseline). Both arms, same 41 cases, Gemini 3.5 Flash, scoped to
the source:

```bash
IDS=$(cat data/graphrag/fullstack_case_ids.txt)
# classical RAG arm
uv run -m evals.run_battery --battery data/eval/battery_singleturn_v1.jsonl \
    --preset prod --retriever classical --ids $IDS --scope-sources \
    --out runs/grag_classical
# GraphRAG arm
uv run -m evals.run_battery --battery data/eval/battery_singleturn_v1.jsonl \
    --preset prod --retriever graphrag --ids $IDS --scope-sources \
    --out runs/grag_graphrag
# grade (auto retrieval/citation metrics, no judge needed) + report side by side
uv run -m evals.grade --run runs/grag_classical
uv run -m evals.grade --run runs/grag_graphrag
uv run -m evals.report --runs runs/grag_classical runs/grag_graphrag --out runs/grag_report
```

Metrics (already in `evals/grade.py`, free to recompute from saved bundles):
`recall_source` / `recall_lesson` / `mrr_lesson` (did retrieval surface the gold
source/lesson), `recall_anytool_source`, `cited_correct_source/lesson` (did the
answer cite it), plus cost / latency / tokens / tool-calls. Community-report
matches carry a synthetic `graphrag_community` source with no url, so they add
context without ever falsely scoring as a recall/citation hit.

## Results

<!-- RESULTS: filled after the comparison run (pending Cohere billing cap reset). -->
_Pending the comparison run (blocked on the Cohere billing cap)._
