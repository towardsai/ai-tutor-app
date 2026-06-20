# Knowledge compaction on small local models (SLMs) — experiments

Two scoped experiments on **how to survive a long lesson in a small context
window**, run on cheap **local** models (Ollama) instead of a large cloud model,
across three SLMs that run on a 16 GB Mac. It lives entirely on branch
`experiment/slm-compaction` and is fully opt-in: the default chat model and all
production behavior are unchanged.

## Why this exists (the gap it fills)

The compaction results on Gemini are anticlimactic (F2/F9/F15): under implicit
prompt caching, keeping the **full history** is cheapest *and* best, so
compaction looks unnecessary. That conclusion only holds when context is cheap
and effectively unbounded. The interesting regime is the opposite one a workshop
audience actually hits on their own hardware: **a small model with a small
context window and no caching**, where the ~37.7k-token lesson physically does
not fit. There, "shove it all" is not an option, so the question stops being
*whether* to compact and becomes **which method wins** — at what quality,
latency, and token cost.

Two complementary axes, same lesson / questions / models / 32k window:

- **Experiment 1 — Axis B (how to *fetch/compact a document* into context):**
  one giant document, compared as full-context vs trim/summary/hierarchical vs
  RAG/GraphRAG vs selective skeleton. Stateless per question.
- **Experiment 2 — Axis A (how to *compact a growing conversation history*):**
  the lesson is loaded turn 0 of a session, then questions accumulate, and each
  real app **memory preset** (full_history / sliding_window / summarization /
  selective_retention / ...) manages the growing context through the middleware
  stack, retrieval off.

The judge, models, and the eviction mechanism (below) are shared. Per-model
reports: `runs/slm_compaction_compare/compare.md` (Axis B) and
`runs/axisa_<model>_report/report.md` (Axis A).

---

# Experiment 1 — Axis B: how to fit a document into context

## What varies, what's held constant

The variables are **(a) the model under test** and **(b) the compaction method**.
Everything else is fixed: the same lesson (the corpus's largest course lesson,
~37.7k tokens), the same 15-question set (mixed specific-detail and
cross-section synthesis, reused verbatim across all models for comparability),
the same 32k context window (`--num-ctx 32768`), retrieval/budget constants, and
the same judge.

**Compaction methods** (`evals/knowledge_compaction.py`):

| method | what it puts in context |
|---|---|
| `full_context` | the whole lesson ("shove it all") — overflows the 32k window |
| `trim` | head+tail truncation to a 4k-token budget |
| `summary` | one single-pass LLM summary of the lesson |
| `hierarchical_summary` | map-reduce: summarize chunks, then summarize the summaries |
| `rag` | chunk + embed (Cohere) + retrieve top-k for the question |
| `graphrag` | a true GraphRAG retriever over a per-lesson index |
| `selective` | structural skeleton (headings + lead sentences) to budget |

The compaction *builders* (summary/hierarchical/trim/selective) run on the
**model under test** — realistically the same cheap model you would deploy, which
also surfaces an honest limit: a single-pass `summary` of a doc bigger than the
window is itself truncated at build time.

**Judge (the measuring instrument, never a compared arm).** Quality is graded by
an LLM judge that reads the **full lesson** as ground truth and checks whether the
answer is correct and supported (we never write reference answers — repo rule).
The judge stays on a **large-context model (Gemini 2.5 Flash)** because the 37.7k
lesson cannot fit an SLM, and a model must never grade itself. The comparison is
therefore **SLM-vs-SLM across methods**; Gemini is only the ruler.

## Models and hardware

Run on a **MacBook Pro, Apple M1 Pro, 16 GB**, one model at a time via
`evals/run_slm_compaction.sh` (≈7h total), then aggregated with
`evals/compaction_compare.py`.

- `llama3.1:8b` — the already-installed baseline.
- `qwen2.5:7b-instruct` — stronger QA/extraction at the same ~8B footprint.
- `qwen3:8b` — newest; **thinking disabled** (`--reasoning-effort none` → Ollama
  `think=False`) to keep the comparison cheap and fast.

14B is the realistic 16 GB ceiling (≈2× slower, memory pressure) and was not run;
`gemma2:9b` is disqualified (8k context can't play the 32k experiment fairly).

## Results

15 questions × 7 methods × 3 models. Local models cost $0, so the cost axes are
**tokens, latency, and context overflow**. Full matrix:
`runs/slm_compaction_compare/compare.md`; per-model: `data/compaction_slm_*/report.md`.

| model | method | judge pass | ctx tok | in tok | out tok | latency p50/p95 s | ctx overflow |
|---|---|---|---|---|---|---|---|
| llama3.1:8b | graphrag | 13/15 (87%) | 8243 | 8324 | 138 | 50.8/65.2 | – |
| llama3.1:8b | full_context | 12/15 (80%) | 37770 | 32767 | 151 | 296.3/306.4 | 15/15 |
| llama3.1:8b | rag | 12/15 (80%) | 2939 | 3019 | 112 | 19.6/28.6 | – |
| llama3.1:8b | trim | 8/15 (53%) | 4007 | 4087 | 83 | 11.6/25.9 | – |
| llama3.1:8b | selective | 7/15 (47%) | 4006 | 4086 | 79 | 9.2/14.5 | – |
| llama3.1:8b | hierarchical_summary | 4/15 (27%) | 517 | 598 | 66 | 4.2/6.4 | – |
| llama3.1:8b | summary | 0/15 (0%) | 446 | 527 | 58 | 4.1/5.6 | – |
| qwen2.5:7b-instruct | rag | 15/15 (100%) | 2939 | 3085 | 131 | 20.1/27.4 | – |
| qwen2.5:7b-instruct | graphrag | 15/15 (100%) | 8243 | 8664 | 192 | 52.2/62.1 | – |
| qwen2.5:7b-instruct | full_context | 10/15 (67%) | 37770 | 32767 | 332 | 265.6/276.1 | 15/15 |
| qwen2.5:7b-instruct | trim | 7/15 (47%) | 4007 | 4214 | 132 | 9.5/23.7 | – |
| qwen2.5:7b-instruct | selective | 7/15 (47%) | 4006 | 4211 | 124 | 4.4/17.0 | – |
| qwen2.5:7b-instruct | summary | 5/15 (33%) | 664 | 743 | 62 | 2.7/7.4 | – |
| qwen2.5:7b-instruct | hierarchical_summary | 3/15 (20%) | 1064 | 1161 | 46 | 2.4/6.3 | – |
| qwen3:8b (no-think) | rag | 15/15 (100%) | 2939 | 3093 | 198 | 25.9/32.4 | – |
| qwen3:8b (no-think) | graphrag | 15/15 (100%) | 8243 | 8672 | 228 | 58.7/77.9 | – |
| qwen3:8b (no-think) | full_context | 11/15 (73%) | 37770 | 32767 | 515 | 345.5/353.6 | 15/15 |
| qwen3:8b (no-think) | hierarchical_summary | 11/15 (73%) | 1696 | 1796 | 169 | 10.4/21.6 | – |
| qwen3:8b (no-think) | selective | 11/15 (73%) | 4006 | 4219 | 163 | 13.0/29.3 | – |
| qwen3:8b (no-think) | trim | 10/15 (67%) | 4007 | 4222 | 173 | 26.1/37.8 | – |
| qwen3:8b (no-think) | summary | 10/15 (67%) | 943 | 1042 | 144 | 8.7/19.5 | – |

### Findings

1. **RAG is the winning compaction method on every SLM.** `rag` is top or tied-top
   on all three models (qwen2.5 100%, qwen3 100%, llama 80%), at ~2.9k context
   tokens and ~20-26s/answer.

2. **`full_context` is dominated, not competitive.** It matches or trails `rag` on
   quality (67-80% vs 80-100%) while being **~12-15× slower** (266-346s vs ~20s)
   and **truncated on 15/15 turns** (37.7k → 32,767 tokens). On an SLM "shove it
   all" buys you *worse* answers, far higher latency, and silent information loss.
   This is the exact inverse of the cached-Gemini result, and it is the cost story
   the workshop needs: a cheap model + RAG **matches or beats** stuffing the whole
   document, at ~1/13th the context and a fraction of the latency.

3. **GraphRAG never wins outright — same verdict as the cloud experiment.** It ties
   `rag` on the qwens (100%) and edges it on llama (87% vs 80%), but at **~2.8× the
   context tokens (8.2k vs 2.9k) and ~2.5× the latency (~50-78s vs ~20-26s)**.
   No quality gain justifies the cost (cf. `evals_graphrag.md`).

4. **Summarization compaction degrades hard, worst on the smallest model.**
   `summary` and `hierarchical_summary` are the bottom methods (llama `summary`
   **0%**; qwen2.5 20-33%; qwen3 67-73%). Aggressive lossy compaction throws away
   the specific facts the questions probe; a weaker model both summarizes worse and
   recovers less. `trim`/`selective` (structural truncation) land in the middle
   (47-73%).

5. **Model choice matters as much as method.** The Qwens beat llama: both hit
   **100% on retrieval**, and **qwen3 (no-think) is the most robust across methods**
   (every method 67-100%), while qwen2.5 is excellent on retrieval but weak on
   summarization (20-33%), and llama is the most polarized (strong on
   retrieval/full, 0% on summary). Validates the recommendation to prefer
   Qwen2.5-7B/Qwen3-8B over llama3.1:8b for this workload.

### Qualitative (judge reasons + retrieval results)

- **Truncation is real, not theoretical.** On a question about which AI model the
  case study used, qwen2.5 `full_context` answered *"no specific AI model was
  mentioned"* — wrong; the intro names it, but it was lost to the 32k truncation /
  lost-in-the-middle. The `rag` arm retrieved the exact passage and answered
  correctly. (`retrieved_context` is saved per row for rag/graphrag.)
- **Summary drops the askable facts.** llama `summary` failed a question whose
  answer ("Global AI Summit 2025") is stated verbatim several times in the lesson
  but did not survive the single-pass summary the 8B model produced.

---

# Experiment 2 — Axis A: how to compact a growing conversation history

## Setup

The lesson is loaded in **turn 0** of a session; turns 1-15 ask the same 15
questions; **retrieval is off** so the agent answers only from whatever the
**memory preset retained**. Each preset runs through the real app middlewares
(`evals/compaction_study.py` → `run_battery` → the production agent), on a
`num_ctx=32768` Ollama model variant so the oversized lesson forces compaction.
Quality is judged on Gemini against the full lesson, same as Axis B.

### The eviction mechanism (important, and a finding in itself)

On a small local model, "keep everything" cannot even be *truncated-and-kept*.
The lesson (~37.7k tokens) is one giant turn-0 message that exceeds the 32k
window. At turn 0 Ollama keeps a truncated 32,767-token slice of it; but from
turn 1 on, **Ollama's server-side context fitting evicts the whole oversized old
message** to make room for the newer turns, so `full_history` answers turns 1-15
with the lesson *gone* (it sees ~1-5k tokens of accumulated Q&A, not the lesson).
On Gemini's large window this never happens (`full_history` held ~43k/turn). So
on an SLM, keep-everything is not a baseline you can lean on — the runtime
silently drops the content, and a method that **summarizes the lesson into a
small message the middleware injects before Ollama** is the only way to retain
its gist. (Verified by inspecting the checkpoint: the full lesson is in stored
state, but the model receives ~300-1.4k tokens.)

## Results (per model; judge pass = correct+supported, n=15, 1 trial)

| preset | qwen2.5-7b | llama3.1-8b | qwen3-8b (no-think) |
|---|---|---|---|
| full_history (keep all) | 27% | 7% | 67% |
| sliding_window | 13% | 13% | 40% |
| summarization_only | 20% | 0% | **73%** |
| prompt_compression | **60%** | 13% | 60% |
| selective_retention | 47% | 13% | 60% |
| incontext_history_retrieval | 33% | 13% | 60% |
| delta_summarization | 47% | 0% | **73%** |
| hierarchical_summarization | 33% | **40%** | 67% |
| **best method** | prompt_compression | hierarchical | summarization/delta |

Local models cost $0; the cost axes are tokens and latency. Notable: hierarchical
is by far the most expensive (it retains a ~25-29k-token layered summary and runs
a ~30-call map-reduce → p50 latency 233s on qwen2.5, 333s on qwen3, vs ~5-80s for
the others), for no quality lead except on the weakest model.

### Findings

1. **No single compaction method wins across models.** `prompt_compression`
   leads on qwen2.5, `hierarchical_summarization` on llama3.1, and
   `summarization_only`/`delta_summarization` on qwen3. Method effectiveness is
   model-dependent, so there is no universal "best compaction" to recommend.
2. **`full_history` never tops the table on an SLM** (27% / 7% / 67%) — it is the
   evicted-lesson case above, and on the two weaker models it is near the bottom.
   This **inverts** the cached-Gemini result (F9/F15) where full history wins.
3. **`sliding_window` is reliably among the worst** (13% / 13% / 40%): pure
   recency dropping evicts the lesson turn outright, with no summary to fall back
   on.
4. **Model capability dominates the compaction method.** qwen3 (40-73% across
   *every* method) >> qwen2.5 (13-60%) >> llama3.1 (0-40%). Choosing a better
   small model buys more than choosing the best compaction strategy — qwen3 even
   answers `full_history` at 67% *despite* the lesson being evicted (it leans on
   accumulated Q&A and prior answers).
5. **Summarization-family needs a capable model.** `summarization_only` /
   `delta_summarization` are top on qwen3 (73%) but bottom on llama3.1 (0%): a
   weak model writes a lossy summary and then can't recover the facts from it.

## Methodology note — judge-truncation fix (important)

The first pass under-reported quality badly: the judge emitted
`{"pass":…, "reason":"…"}` and a long reason hit `max_tokens`, truncating the
JSON, so **~64% of verdicts were unparseable and silently counted as fails**. The
directly-measured axes (tokens, latency, overflow) were unaffected, but the
quality ranking was noise. Fixed by capping the judge's reason to 25 words,
raising the token budget, and recovering the leading `pass` boolean from truncated
JSON (the boolean is emitted first). The SLM answers were all saved, so we
**re-graded offline** with `evals/rejudge_compaction.py` (no SLM re-run) → **0
unparseable verdicts**. The table above is post-fix.

## Caveats (both experiments)

- **Coarse n.** 15 questions/method, 1 trial. Read the tables as rankings, not
  precise rates; saturated (100%) and small-difference cells are not separable.
- **LLM-judge variance.** Re-judging the same Axis-A answers moved individual
  cells by ~1-3/15 (e.g. delta 27%↔47%) even at temperature 0, so only sizeable
  gaps are meaningful — not 1-2 question differences.
- **One lesson, one domain.** A single case-study lesson; absolute numbers and the
  gaps may move on denser or more numeric content.
- **Axis-A eviction is runtime+model-specific.** The `full_history` numbers
  reflect Ollama's message-eviction behavior on a 32k window; a different serving
  layer (or per-message truncation) could keep a truncated lesson instead.
- **Qwen3 thinking off** (Axis B `--reasoning-effort none`; Axis A via Ollama
  `/v1`, which returns final answers without reasoning) to keep cost/latency
  honest.
- **Strict judge.** Gemini grades "correct AND supported by the lesson."

## Reproduce

```bash
# prereqs: `ollama serve`; ollama pull llama3.1:8b qwen2.5:7b-instruct qwen3:8b
# GEMINI_API_KEY in .env (judge + question gen); COHERE_API_KEY (rag embeds)

# Experiment 1 (Axis B): 3 models x 7 methods x 15 Q (~7h)
bash evals/run_slm_compaction.sh
uv run --env-file .env -m evals.rejudge_compaction data/compaction_slm_*   # re-grade if judge changes
uv run -m evals.compaction_compare data/compaction_slm_* --out runs/slm_compaction_compare

# Experiment 2 (Axis A): one num_ctx=32768 Ollama variant per model, then:
printf 'FROM qwen2.5:7b-instruct\nPARAMETER num_ctx 32768\n' | ollama create qwen2.5-7b-ctx32k -f /dev/stdin
uv run --env-file .env -m evals.compaction_study build --questions 15    # build the session battery once
MODEL=ollama:qwen2.5-7b-ctx32k TAG=qwen2.5-7b bash evals/run_slm_axis_a.sh
```

Outputs live under `data/compaction_slm_*` and `runs/axisa_*` (gitignored — no
eval data in git).
