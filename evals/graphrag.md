# GraphRAG vs classical RAG (experiment)

A single, scoped comparison of the production hybrid retriever against a true
Microsoft GraphRAG index, on the same eval battery and the same chat model, to
get a defensible answer to "does GraphRAG beat our classical RAG for the tutor?"

GraphRAG was previously **dropped** from the Part C plan ("low information given
the findings", `evals.md`). This revisits it deliberately, with a fair head-to-head
rather than on faith. It was developed on branch `experiment/graphrag-vs-rag` (since merged — the retriever, `graphrag` extra, and runner flags are on `main`)
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
answer; both arms pass `--model google-genai:gemini-3.5-flash` explicitly --
`run_battery`'s no-flag default is the app default model, since moved to
DeepSeek V4 Flash), the system prompt, the battery, the token budget,
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

## Setup (optional extra)

`graphrag` (and its `lancedb`/`litellm`/`pandas` deps) is an **eval-only optional
extra** — it is imported lazily and only by the opt-in `graphrag` retriever, never
on the production path, so it is kept out of the default install to keep the prod
image lean. Install it to build/run this experiment:

```bash
uv sync --extra graphrag
```

A default `uv sync` (prod, CI) omits it; `tests/test_graph_rag.py` skips itself
when the extra is absent.

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

**Built index (full_stack_ai_engineering):** 90 docs -> 493 text units, **15,059
entities**, 32,748 relationships, 3,371 communities + community reports. Cost
**$44.96** on Gemini 2.5 Flash (in 34.5M / out 13.8M tokens), in budget. The
retriever needs only the `entity_description` LanceDB table at query time.

### Reuse the prebuilt index (skip the ~$45 rebuild)

The index is published to the **same private HF dataset** as the rest of the data
(`towardsai-tutors/ai-tutor-vector-db`, under `graphrag/output/`). It is **not**
pulled by the runtime cold-start (`ensure_local_vector_db` ignores `graphrag/**`),
so prod Spaces stay lean; pull it explicitly with the same `HF_TOKEN` to run the
eval:

```bash
uv run python -c "from huggingface_hub import snapshot_download as d; d(repo_id='towardsai-tutors/ai-tutor-vector-db', repo_type='dataset', allow_patterns=['graphrag/**'], local_dir='data')"
```

Re-publish after rebuilding: `uv run -m data.scraping_scripts.upload_dbs_to_hf --graphrag` (prune-free; never touches the production bundle).

## Run the comparison

Blocked only by Cohere being available (both arms use Cohere embed/rerank; the
classical vector store is Cohere-embedded so this cannot be swapped without
changing the baseline). Both arms, same 41 cases, Gemini 3.5 Flash, scoped to
the source:

```bash
# fullstack_case_ids.txt is a gitignored local artifact: it is the 41 case_ids from
# battery_singleturn_v1.jsonl whose source_key == full_stack_ai_engineering.
IDS=$(cat data/graphrag/fullstack_case_ids.txt)
# classical RAG arm
uv run -m evals.run_battery --battery data/eval/battery_singleturn_v1.jsonl \
    --preset prod --retriever classical --ids $IDS --scope-sources --disable-kb \
    --model google-genai:gemini-3.5-flash \
    --out runs/grag_classical
# GraphRAG arm
uv run -m evals.run_battery --battery data/eval/battery_singleturn_v1.jsonl \
    --preset prod --retriever graphrag --ids $IDS --scope-sources --disable-kb \
    --model google-genai:gemini-3.5-flash \
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

Run 2026-06-19: 41 full_stack single-turn cases, Gemini 3.5 Flash, `--disable-kb`
(forces `retrieve_tutor_context` so the retriever is the only variable),
`--scope-sources`, 1 trial, 0 errors per arm. Auto metrics only (key-point /
behavior human grades pending). Bundles: `runs/grag_classical`, `runs/grag_graphrag`;
report: `runs/grag_report/report.md`.

| metric | classical RAG | GraphRAG |
|---|---|---|
| recall@shown source | 100% (n=41) | 100% (n=41) |
| recall@shown lesson | 76% (n=41) | 76% (n=41) |
| right-lesson rank (MRR) | **0.70** | 0.65 |
| cited-correct source (answer) | 100% (n=27) | 100% (n=27) |
| cited-correct lesson (answer) | 85% (n=27) | 85% (n=27) |
| behavior proxy (code check) | 89% (n=37) | 92% (n=37) |
| input tokens / turn | **110k** | 178k |
| est cost / turn | **$0.147** | $0.212 |
| TTFT p50 / p95 | **18.0s** / 31.8s | 18.8s / 39.2s |
| turn ms p50 / p95 | **24.1s** / 40.9s | 26.9s / 44.4s |
| llm calls / turn | 3.5 | 3.6 |

**Finding (F-GraphRAG): on single-turn course-content questions, GraphRAG does
not beat classical hybrid RAG — it ties on grounding accuracy and costs more.**
Both surface and cite the right source 100% of the time and tie on lesson recall
(76%) and cited-correct lesson (85%); classical is actually slightly better at
*ranking* the right lesson (MRR 0.70 vs 0.65). GraphRAG pulls community-report
context, so it spends **+61% input tokens and +44% $/turn** and is a bit slower,
with no accuracy payoff. The +3pt behavior proxy is a coarse n=37 code check, not
a real quality signal. This empirically supports the earlier decision to drop
GraphRAG, rather than dropping it on faith.

Caveats: scoped to one source (full_stack), single-turn, n=41 (27 corpus-answer),
1 trial, auto metrics only. `recall@source` is saturated by `--scope-sources`
(retrieval is restricted to the gold source), so `recall@lesson` / MRR /
`cited_correct_lesson` are the discriminating numbers. GraphRAG's theoretical
edge is multi-hop / cross-document synthesis, which this single-hop battery barely
tests -- so this says GraphRAG doesn't help on typical single-hop course Q&A, not
that it never helps. A fair test of its strength would need a multi-hop probe set
(`_v2`), which is out of scope for this one comparison.
