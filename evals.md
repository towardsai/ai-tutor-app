# Evaluating the AI tutor

We compare memory and context-management techniques on the tutor with measured numbers — answer quality, memory retention, retrieval accuracy, tokens, cost, latency — for the June 2026 workshop, and keep the same harness as the ongoing quality program afterwards. Research sources and design rationale: `evals_background.md`.

## What we evaluate, and what varies

The system under test is the production tutor (`app/`): a LangGraph agent over the course corpus with hybrid retrieval (`retrieve_tutor_context`), a sandboxed KB shell (`run_kb_command`), and per-thread conversation memory. One variable changes between experiment arms — the **memory preset** (`app/memory_presets.py`):

| Preset | Configuration |
|---|---|
| `full_history` | no compaction (baseline: quality ceiling, token worst case) |
| `prod` | production settings: summarization @30k tokens keep 20 msgs + tool-output clearing @5k keep 5 |
| `summarization_only` / `editing_only` | each compaction technique isolated |
| `aggressive` | summarize @8k keep 8, clear @2k keep 2 |
| `profile_memory` | prod + long-term memory: stored student profile injected into the system prompt, updated after each turn (`StudentProfileMiddleware`) |

Held constant per experiment: model (`gemini-3.5-flash`), system prompt, retrieval configuration, source selection, web tools off, temperature. Presets are selectable per request (`ChatRequest.memory_preset`, API `memoryPreset`) and are part of the agent cache key.

## What we measure

| Layer | Metrics | Source |
|---|---|---|
| Ops | TTFT, turn latency, input/output tokens, cache-read/creation tokens, est. cost, LLM calls, compaction-trigger firings | `context_stats` event each turn (`app/telemetry.py`) — self-contained, no LangSmith dependency |
| Trajectory | tool calls per turn by tool, redundant calls, KB-budget use | tool-call events (code) |
| Retrieval | recall@shown / MRR of the case's true `source_key` + `lesson_url` in retrieval results | ground truth carried by every discussion-derived case (code) |
| Behavior | called retrieval on corpus questions, citations present and resolving, redirect/feedback routing | code assertions + heuristics; authoritative number from hand grades |
| Quality | key-point coverage, behavior correctness, probe pass/fail | **hand-graded binary** (blinded workbook); an LLM judge only after validation against those labels (TPR/TNR, target >90%) |
| Memory | probe accuracy by type (fact recall, preference, anaphora, fact-update, behavior-under-pressure), personalization pass rate | session probes + persona checks |

Cost is reported both as tokens and as cached dollars — they rank presets differently (finding F2).

## The dataset

Source: `data/academy_discussion_eval.jsonl` — 151 real student posts from the academy discussion boards with real staff answers, annotated by gemini-3.5-flash. All eval inputs are real or hand-authored; **we never generate reference answers** — ground truth is staff replies (distilled to `key_points`) or facts we wrote ourselves.

**Review.** All 62 gold + 30 not-time-bound usable annotations (92) were re-reviewed against the live KB with file/line evidence. 60 kept, 32 excluded. Main corrections: the annotator badly under-flagged staleness (course content was updated after many posts — ~14 exclusions had premises no longer true in today's corpus), 5 near-duplicates, 1 fabricated URL, 1 misaligned question/answer pair, and key points rewritten to claims that are atomic, still true, and binary-checkable. Colab-notebook dependency was checked per case: **1 of 92** truly required the notebook (excluded); lessons embed the cells everywhere else. Full audit trail: `data/eval/review_log_v1.md` (11 judgment-call cases flagged for human review).

**Batteries** (`data/eval/`, schemas and glossary in its `README.md`; files are gitignored — real student text):

| Battery | Contents | Tests |
|---|---|---|
| `battery_singleturn_v1` | 60 reviewed real questions (37 corpus / 10 redirect / 7 general / 6 feedback; 49 gold) | one-shot answer quality + behavior routing; presets should roughly tie here |
| `battery_sessions_v1` | 32 authored study sessions, 337 turns, 113 probes (5 hand-written, 27 generated and spot-checked; middle turns are verbatim real questions) | where memory methods separate: facts planted early, context inflated past compression triggers, probes late. Probe types: fact_recall (55), preference_compliance (27), anaphora (12), anaphora_consistency (9), **fact_update** (7 — the student changes a fact mid-session; answering with the stale value fails), behavior_routing (3) |
| `battery_personas_v1` | 10 authored student profiles × 4 questions each, unanswerable correctly without the profile; self-grading regex/anti-pattern checks | long-term profile memory |
| `replay_n1_v1` | 30 prefixes of real multi-turn threads, graded against the real next staff reply | multi-turn behavior on fully real data (secondary) |

## How it runs

```bash
uv run -m evals.run_battery --battery data/eval/battery_singleturn_v1.jsonl --preset prod --trials 2 --out runs/exp
uv run -m evals.grade  --run runs/exp                      # code checks + handgrade_sheet.csv
uv run -m evals.check_triggers --runs runs/exp             # gate: compaction fired where probes assume
uv run -m evals.report --runs runs/expA runs/expB          # side-by-side tables + token curves
uv run -m evals.handgrade_workbook build|merge ...         # blinded human-grading workbook
```

Every turn persists a JSON trace bundle, so grading and reporting re-run offline forever without touching the API. Runs are resume-safe (re-run the same command after any interruption), have a 10-minute per-turn timeout, and keep LangSmith off by default. Needs `GEMINI_API_KEY` + `COHERE_API_KEY`; the full 4-preset bake-off below cost $73.

**Setup for collaborators.** Code and docs are in git; the datasets and run results contain real student text and live only in the private HF dataset (`towardsai-tutors/ai-tutor-data` — git is force-pushed to the public prod Space on deploys, so student data never enters git). With an `HF_TOKEN` that can read it:

```bash
uv run python -c "from huggingface_hub import snapshot_download as d; d(repo_id='towardsai-tutors/ai-tutor-data', repo_type='dataset', allow_patterns=['eval/**','eval_runs/**'], local_dir='.')"
mv eval/* data/eval/ && mv eval_runs/part_b/* runs/   # restore working paths
```

## Results — Part B bake-off, 2026-06-12

4 presets × 2 trials over the full single-turn battery, full persona battery, and 3 sessions (s01 13-turn, s02 agentic, s08 fact-update). 1,232 turns, zero API errors. Tables: `runs/b_report/report.md`; token curves: `tokens_by_turn.csv`. **Auto-graded only — quality and probe-accuracy columns land after the hand-grading pass.**

| | full_history | prod | profile_memory | aggressive |
|---|---|---|---|---|
| personalization pass (personas, n=70) | 63% | 67% | **94%** | 56% |
| sessions: est. cost/turn | **$0.034** | $0.051 | $0.047 | $0.066 |
| sessions: TTFT p50 | **17s** | 21s | 22s | 43s |
| sessions: tool calls/turn | **2.8** | 3.5 | 3.9 | 8.3 |
| sessions: cumulative input tokens | 2.9M | 1.7M | 1.6M | 1.75M |
| single-turn: behavior heuristic (n=106) | **88%** | 86% | 86% | 78% |
| single-turn: retrieval recall@shown source | 51% | 50% | 49% | 47% |

Row definitions:
- **personalization pass** — % of persona question-runs whose answer passed every authored check (expected regex matched, e.g. `conda` for the conda persona) with no anti-pattern hit (e.g. bash `export` for a Windows persona). n=70: 40 questions × 2 trials, minus 10 runs whose checks need human judgment.
- **est. cost/turn** — mean estimated $ per conversation turn: token counts × `MODEL_PRICING`, cached input billed at the cache-read discount (our table, not the invoice).
- **TTFT p50** — median seconds from user message to first answer text, including all tool calls and internal LLM rounds before the answer starts (perceived wait).
- **tool calls/turn** — mean retrieval + KB-command invocations per turn; here it's the re-work signal (compaction presets re-search for evidence their compressed history lost).
- **cumulative input tokens** — all input tokens billed across an entire session (every turn, every internal call), mean over the 6 session-runs per preset.
- **behavior heuristic** — % of single-turn case-runs where a programmatic proxy confirms the right *kind* of response (corpus → called retrieval/KB; support issue → answer points to support; feedback → acknowledges). n=106 of 120: `answer_general` has no heuristic. Proxy only — authoritative behavior accuracy comes from the hand grades.
- **retrieval recall@shown source** — % of case-runs where the reranked retrieval results the agent saw contained at least one chunk from the correct course. Stricter `recall@lesson` (right lesson page, 30–36%) and MRR are in the full report.

### Observations

Numbered findings; each states what it was tested on. Convention: entries are never edited, only superseded.

- **F1 — Retrieval payloads dominate input tokens, not conversation history.** Each retrieval call may return up to `DEFAULT_CONTEXT_TOKEN_BUDGET = 100_000` tokens; turns average ~200k input. (All runs; high confidence.) → retrieval budget became a Part C variant dimension.
- **F2 — Compaction saves tokens but not necessarily dollars.** Summarization rewrites the prompt prefix and invalidates Gemini's implicit cache: on the same session, `full_history` had 86.8% of input billed at the ~4x cache discount; `aggressive` used 44% fewer tokens yet cost 68% more. (s03 × 3 presets, then confirmed n=24 session-runs; provider-specific — Anthropic caching is explicit.)
- **F3 — Context editing never fires on this workload.** `ClearToolUsesEdit` excludes retrieval results — where the tokens are; 0 clears across all session runs. → `editing_only` dropped from Part B; "clear retrieval results too" variant queued for Part C.
- **F4 — Aggressive compaction degrades even single-turn behavior.** 18.0 vs 9.6 LLM calls/turn, 57s vs 39s median, behavior heuristic −10pts vs full_history (60 cases × 2 trials). Mid-turn compaction churn changes agent behavior, it doesn't just trim.
- **F5 — Gemini reasoning tokens are ~90%+ of output even with reasoning display off.** Billed as output; dominates latency; recorded per model in `usage_by_model`.
- **F6 — Long context costs latency, not correctness.** Zero API errors in 1,018+ turns at any size (max 6.06M tokens across one turn's calls; largest single context 274k). Median TTFT scales 22s → 76s from <100k to >800k input tokens/turn. Compaction's value here is responsiveness and spend, not keeping the model functional.
- **F7 — ~0.8% of turns produce no answer text** despite tool calls and billed reasoning tokens; preset- and size-independent. Candidate golden-case assertion ("answer non-empty").
- **F8 — Profile memory wins on quality AND cost.** 94% personalization vs 56–67% without, while the cheapest persona preset ($0.049 vs $0.081–0.095/turn, 7.6 vs 10–11.2 tool calls): the stored profile saves the agent from re-searching for user context. (40 questions × 2 trials × 4 presets, auto checks; LLM-check rows pending hand grades.)
- **F9 — Full history is cheapest AND fastest up to 13 turns; compaction causes re-work.** Sessions: $0.034/turn, TTFT 17s, 2.8 tool calls for `full_history` vs $0.051–0.066, 21–43s, 3.5–8.3 elsewhere — despite ~2x the tokens (F2's cache mechanism plus a second one: raw history lets the agent re-use earlier retrieval evidence; summaries force re-retrieval). (3 sessions × 4 presets × 2 trials, n=24, zero errors.) The conventional pitch inverts: under modern prompt caching, the naive baseline wins short-to-medium sessions; compaction must justify itself on quality (pending hand grades) and long horizons. Where the crossover actually sits is a Part C question.

Harness corrections (bugs in our measurement, not findings): the overnight 06-12 stall was machine sleep hanging API streams (all four pipelines stopped the same minute; fixed with the per-turn timeout); battery lesson-URLs carried a `/discussions/` suffix that silently zeroed recall@lesson until normalized (caught by the A3 smoke, re-graded from bundles without re-running).

Talk-outline correction: the planned "skills / just-in-time KB instructions" change (Change 4) was never built, and its premise is off — the KB instructions block measures **458 tokens** (not ~4k) inside a 1,655-token system prompt; max saving ~2% of a turn, mostly cache-discounted. Recommended reframe: demo progressive disclosure on tool *outputs* (F1), where our numbers actually are.

## Remaining work

**Omar**
- [ ] Grade `runs/b_report/workbook.csv` — 284 blinded rows, priority-ordered (96 session probes ≈ 25 min is the critical tier). Don't open `workbook_keymap.csv` until done. Then: `uv run -m evals.handgrade_workbook merge --workbook <filled.csv>` and re-run grade/report for the final table.
- [ ] Audit `data/eval/review_log_v1.md` (the 11 flagged judgment calls first).
- [ ] Verify the `gemini-3.5-flash` entry in `app/telemetry.py:MODEL_PRICING` against the current price sheet.

**Workshop**
- [ ] Replace placeholder slide numbers with measured ones; pick 2–3 failure traces from the graded probes for the failure→fix→number beats (a `fact_update` failure under summarization is the target demo).
- [ ] Token-vs-turn plot: CSV is ready; add matplotlib (or plot elsewhere) for the PNG.
- [ ] Next.js meter component over the `data-context-stats` SSE part, if the live demo needs it.
- [ ] Skills section: reframe per the 458-token measurement above.

**Part C (post-workshop, ~$300–400, after the judge is validated)**
- [ ] Variants: `summarization_only`, `aggressive`, clear-retrieval-results editing, retrieval budget 100k/30k/10k, 1–2 SOTA methods (retrieval-over-history, structured note-taking, LangMem-style consolidation), optionally lazy KB instructions.
- [ ] Full batteries (60 single-turn + 32 sessions + 30 replay) × 3 trials; paired stats (McNemar/bootstrap), pass^3 consistency; failure-taxonomy diffs per variant.

**Product quality track (post-workshop)**
Golden cases as a CI gate (5–10 critical-path cases, deterministic assertions, incl. the F7 non-empty check) → error-analysis cycles (expert reads 50–100 traces in a small viewer, binary + first-failure note, stop at saturation) → failure taxonomy → code assertions for recurring modes → LLM judges built from the hand-grade critiques and validated to >90% TPR/TNR on held-out labels before any judge-graded number is reported → nightly battery + weekly trace sampling. Full methodology behind each step: `evals_background.md`.

## Files

- `evals.md` — this file: the what, the data, the results, the queue.
- `evals_background.md` — research sources (Hamel, howtoeval, OpenAI macro-evals) and design rationale.
- `data/eval/README.md` — battery schemas + glossary of terms; `review_log_v1.md` — dataset audit trail.
- `evals/` — the harness code; `app/memory_presets.py`, `app/telemetry.py` — the app-side hooks.
- `runs/b_report/` — generated: final tables, token curves, the blinded grading workbook.
