# Compaction study: compaction methods vs keeping everything in context

A **standalone** experiment (branch `experiment/context-compaction`) answering the
workshop's core question: when a long context is established and then queried,
should you **keep it all** (and lean on caching), **compact it** (summarize /
trim / sliding window / selective / delta / hierarchical), or **not keep it and
retrieve per question** (RAG / GraphRAG)? Filling the context window is where the
tokens and dollars go, so the goal is to find the method that keeps answer
quality at the lowest cost.

> **Standalone fleet — do not cross-compare.** Every arm here runs on **Gemini
> 2.5 Flash**. The Part B/C memory-preset results and the GraphRAG-vs-RAG study
> ran on **Gemini 3.5 Flash** — a different model with different pricing and
> caching. These numbers are comparable only to each other.

## Setup

- **Lesson:** the corpus's largest course lesson, `master_ai_for_work`
  "Case Study: Pulling Together Our LLM Uses" (**37.5k tokens**).
- **Questions:** 15 generated from the lesson (specific-detail + cross-section
  synthesis). We never write reference answers (repo rule).
- **Two families, same questions, same model:**
  - **A — keep/compact in context:** the lesson is loaded in turn 0 of a session;
    turns 1..15 ask the questions with **no tools**, so the agent must answer from
    whatever the memory preset *retained*. Real app middlewares (`full_history`,
    `prod`, `summarization_only`, `sliding_window`, `prompt_compression`,
    `selective_retention`, `incontext_history_retrieval`, `aggressive`, plus the
    two new arms below). This isolates the compaction method.
  - **B — retrieve per question:** the lesson is **not** held; each question
    retrieves from the lesson (`rag` = chunk+embed+rerank; `graphrag` = a
    per-lesson GraphRAG index). Stateless / fresh context each question.
- **Grading:** an LLM judge (Gemini 2.5 Flash, large context) reads the **full
  lesson** as ground truth and marks each answer correct/supported or not.
- **New arms built for this study** (`app/memory_presets.py`,
  `app/chat_service.py`): **delta_summarization** (one running summary updated
  each trigger with only what changed) and **hierarchical_summarization** (a
  map-reduce middleware: summarize chunks, then summarize the summaries; chunk
  summaries cached so a static lesson is mapped once and only the reduce re-runs).
- **Left out by design** (different axes; prior findings): the
  skills / progressive-disclosure / lazy-prompt-loading family (evals.md measured
  the instructions block at ~458 tokens ≈ 2% of the prompt) and multi-agent /
  sub-agent orchestration (dropped as costly/low-info). The useful kernel of
  "sub-agent with clean context per question" is the Family-B retrieve arms.

## Results (Gemini 2.5 Flash, 15 questions, 1 trial)

| arm | family | judge pass | mean in tok/turn | total $ | latency p50 s |
|---|---|---|---|---|---|
| **rag** | B retrieve | **9/15 (60%)** | **3,200** | **$0.020** | 2.3 |
| incontext_history_retrieval | A in-context | 9/15 (60%) | 41,682 | $0.097 | 6.3 |
| graphrag | B retrieve | 8/15 (53%) | 8,960 | $0.045 | 1.9 |
| summarization_only | A in-context | 8/15 (53%) | 27,344 | $0.089 | 4.8 |
| prompt_compression | A in-context | 8/15 (53%) | 40,919 | $0.134 | 4.3 |
| full_history (keep all) | A in-context | 8/15 (53%) | 42,935 | $0.134 | 4.8 |
| delta_summarization | A in-context | 7/15 (47%) | 27,033 | $0.107 | 5.9 |
| selective_retention | A in-context | 6/15 (40%) | 27,121 | $0.105 | 5.9 |
| prod | A in-context | 6/15 (40%) | 27,406 | $0.103 | 4.6 |
| aggressive | A in-context | 5/15 (33%) | 11,131 | $0.075 | 6.5 |
| sliding_window | A in-context | 5/15 (33%) | 18,453 | $0.058 | 5.9 |
| hierarchical_summarization | A in-context | 5/15 (33%) | 30,688 | $0.605 | 41.8 |

(Full report: `runs/compaction_report/report.md`; bundles in gitignored
`runs/compaction_*` and `data/compaction/`.)

## Findings

- **F-C1 — Retrieval beats both hoarding and compacting.** `rag` is the cheapest
  arm ($0.020, 3.2k tok/turn) **and** tied for the best quality (60%). For a long
  document queried question-by-question, fetch the relevant chunk; don't hold or
  summarize the whole thing.
- **F-C2 — "Keep everything" is the priciest in-context arm and not even the best
  quality.** `full_history` costs $0.134 / 43k tok per turn for 53%. (Note: this
  is a **no-tools** setting, so the F9 "full_history is cheapest" result does NOT
  hold here — F9's cost win came from avoiding re-retrieval, and there is nothing
  to re-retrieve here. With no re-retrieval penalty, compaction *does* cut
  tokens/cost — but at a quality cost, see F-C3.)
- **F-C3 — Compaction cuts cost but loses quality, and saves less than you'd
  hope.** The summarization-family arms land at ~27k tok / ~$0.09–0.11 (≈35% fewer
  tokens than full_history) but drop to 40–53% quality. `sliding_window` /
  `aggressive` are cheapest-in-context but worst (33%) — they evict the lesson and
  cannot recover it.
- **F-C4 — The best in-context method is selective retrieval *over* the history.**
  `incontext_history_retrieval` ties `rag` on quality (60%) but at **13× the
  tokens** of `rag`, because the lesson sits in one big turn-0 block it keeps
  re-including. The two winners surface the *relevant* part; everything else
  hoards or lossily compresses.
- **F-C5 — Naive hierarchical summarization is the worst trade.** `hierarchical`
  is the most expensive ($0.605) and slowest (41.8s p50) — its per-turn
  map-reduce dominates — for the *lowest* quality tier (33%). Hierarchical's value
  is handling content too large for one summarization call; on a single lesson it
  is all cost and no benefit.
- **F-C6 — GraphRAG vs RAG (again, on this lesson):** `graphrag` (53%, $0.045)
  does not beat plain `rag` (60%, $0.020) here either — consistent with the
  GraphRAG-vs-RAG study: the corpus's topic separation is already good, so the
  graph adds cost without accuracy.

**Takeaway for the talk:** filling the context *is* the cost. Hoarding (keep-all)
is expensive and not best; compaction buys modest token savings at a real quality
cost; the win is **retrieving the relevant slice per question**. "Just shove it
all in" loses on both axes once the document is long.

**Caveats:** n=15, 1 session, 1 trial — quality percentages are coarse rankings,
not precise rates. Judge is Gemini 2.5 Flash (same family as the model under
test; it grades against the full lesson). Hierarchical's $/latency include its
per-turn summarization calls.

## Companion: local small-model arm

`knowledge_compaction.py` can also drive a **local SLM** (Ollama) with a small
context window, where the 37.5k-token lesson does not fit and "shove it all"
becomes physically impossible — so compaction stops being optional. That arm is
a **separate experiment** (different model; its own fleet, never compared to the
Gemini numbers above) and is developed in the SLM PR, not here.

## Run it

```bash
uv run --env-file .env -m evals.compaction_study build --questions 15
bash evals/run_compaction_study.sh        # Family A presets, 2.5 Flash, no tools
uv run --env-file .env -m evals.knowledge_compaction --questions 15 \
    --strategies rag graphrag --out data/compaction      # Family B
uv run --env-file .env -m evals.compaction_study report --runs 'runs/compaction_*'
```
