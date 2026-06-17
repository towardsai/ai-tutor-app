# Evaluating the AI tutor

This document explains how we test the AI tutor with repeatable conversations and measured results. The main question is: when the tutor has a long chat history, which memory strategy gives the best answers without wasting too many tokens, dollars, or seconds?

The eval setup compares answer quality, memory retention, retrieval accuracy, tokens, cost, and latency for the June 2026 workshop. The same harness also becomes the ongoing quality program afterwards. Research sources and design rationale live in `evals_background.md`.

Plain-English map:
- A **battery** is a dataset of test conversations or test questions.
- A **run** is one battery executed with one model and one memory setting.
- A **trace bundle** is the saved JSON record for one tutor turn: the question, answer, tool calls, sources, token usage, timing, and any error.
- A **grade** is the pass/fail or metric row computed later from saved trace bundles.
- A **memory preset** is a named setting that controls how much chat history the tutor keeps, summarizes, or stores as a student profile.
- An **axis** is one independent dimension the experiment varies. Part B varied a single axis (the memory preset). Part C uses two: **A — memory & context management** (how conversation history is kept or compacted) and **B — retrieval & tool outputs** (how retrieved docs and tool results are sized, cleared, or whether the KB tool exists). A **single-axis** run changes one variant on one axis and holds everything else at the production baseline, so any difference is attributable to that one change; a **cross-product** tests every Axis-A × Axis-B combination at once (far more runs, far bigger bill).

## What we evaluate, and what varies

The system under test is the production tutor (`app/`): the same agent, retrieval tools, prompts, and telemetry used by the app. The tutor can answer from the course/docs corpus through `retrieve_tutor_context`, browse the local knowledge base through `run_kb_command`, and keep per-thread conversation memory.

One thing changes between experiment arms: the **memory preset** (`app/memory_presets.py`). In this document, **compaction** means reducing the prompt by summarizing older messages or clearing old tool outputs so the model sees less raw history.

| Preset | Configuration |
|---|---|
| `full_history` | Keeps the whole visible conversation. This is the baseline for "what if we do nothing clever?" It may use the most tokens. |
| `prod` | Current production behavior. It summarizes once the prompt is large and clears some old tool outputs. |
| `summarization_only` / `editing_only` | Test just one compaction technique at a time, so we can tell which one helped or hurt. |
| `aggressive` | Compacts much earlier and keeps fewer recent messages. This tests the high-pressure version of memory trimming. |
| `profile_memory` | Uses `prod` compaction plus a stored student profile, such as level, goal, operating system, and preferences. The profile is inserted into future turns. |

Everything else is held constant within an experiment: model (`gemini-3.5-flash`), system prompt, retrieval configuration, selected sources, web tools off, and temperature. Presets are selectable per request (`ChatRequest.memory_preset`, API `memoryPreset`) and are part of the agent cache key.

## What we measure

| Layer | Metrics | Source |
|---|---|---|
| Runtime and cost | Time to first answer text, total turn time, input/output tokens, cached tokens, estimated dollars, number of model calls, and whether compaction fired. | The `context_stats` event emitted at the end of each turn (`app/telemetry.py`). This does not depend on LangSmith. |
| Tool behavior | Which tools the tutor called, how often, and whether it re-searched for information it had already seen. | Tool-call events saved in each trace bundle. |
| Retrieval | Whether the retrieved chunks included the correct course or lesson. `recall@shown` means "was the right source among the results shown to the model?" `MRR` rewards putting the right lesson earlier in the shown results. | Ground truth stored on each real discussion case: `source_key` and `lesson_url`. |
| Response behavior | Whether the tutor did the correct kind of thing: answer from course content, answer generally, redirect a platform/support issue, or acknowledge feedback. | Cheap code checks for early signals; final behavior accuracy comes from human grades. |
| Answer quality | Whether the answer covered the required key points and whether graded memory probes passed. | Human pass/fail grading in a blinded workbook. LLM judges are used only after they are validated against human labels. |
| Memory | Whether the tutor remembered or updated facts from earlier in the session, followed preferences, resolved references like "that thing from earlier," and used stored student profiles. | Session probes and persona checks. |

Cost is reported both as raw tokens and as estimated dollars after cached-token discounts. These can rank presets differently (finding F2).

## The dataset

Source: `data/academy_discussion_eval.jsonl` — 151 real student posts from the academy discussion boards with real staff answers, annotated by gemini-3.5-flash. All eval inputs are real or hand-authored; **we never generate reference answers** — ground truth is staff replies (distilled to `key_points`) or facts we wrote ourselves.

**Review.** All 62 gold + 30 not-time-bound usable annotations (92) were re-reviewed against the live KB with file/line evidence. 60 kept, 32 excluded. Main corrections: the annotator badly under-flagged staleness (course content was updated after many posts — ~14 exclusions had premises no longer true in today's corpus), 5 near-duplicates, 1 fabricated URL, 1 misaligned question/answer pair, and key points rewritten to claims that are atomic, still true, and binary-checkable. Colab-notebook dependency was checked per case: **1 of 92** truly required the notebook (excluded); lessons embed the cells everywhere else. Full audit trail: `data/eval/review_log_v1.md` (11 judgment-call cases flagged for human review).

**Batteries** (`data/eval/`, schemas and glossary in its `README.md`; files are gitignored because they contain real student text):

| Battery | Contents | Tests |
|---|---|---|
| `battery_singleturn_v1` | 60 reviewed real questions asked as one-off chats. They include course-content questions, support/platform issues, general AI/programming questions, and course feedback. | Basic answer quality and correct response type. Memory presets should score about the same here; big differences suggest a bug. |
| `battery_sessions_v1` | 32 multi-turn study sessions. Early turns give the tutor facts about the student, middle turns make the chat long, and later turns check whether important facts survived. | Memory under pressure. A **probe** is one of the later turns that gets graded. Probe examples: remember a fact, respect a preference, understand "that thing from earlier," or use a newer fact after the student changed their mind. |
| `battery_personas_v1` | 10 stored student profiles with 4 questions each. The questions are written so the best answer needs the profile. | Long-term profile memory: does a stored profile actually improve personalization across fresh chats? |
| `replay_n1_v1` | 30 tests made from real multi-turn discussion threads. For each test, we give the tutor the conversation up to just before a real staff reply, ask it to write the next reply, and compare it with what staff actually wrote. | Multi-turn behavior on fully real data. This is secondary because it needs human or validated judge grading. |

## How it runs

```bash
uv run -m evals.run_battery --battery data/eval/battery_singleturn_v1.jsonl --preset prod --trials 2 --out runs/exp
uv run -m evals.grade  --run runs/exp                      # code checks + handgrade_sheet.csv
uv run -m evals.check_triggers --runs runs/exp             # gate: compaction fired where probes assume
uv run -m evals.report --runs runs/expA runs/expB          # side-by-side tables + token curves
uv run -m evals.handgrade_workbook build|merge ...         # blinded human-grading workbook
```

Command roles:
- `run_battery` talks to the tutor and saves one JSON trace bundle per turn.
- `grade` reads saved bundles and writes automatic grades plus a CSV for human judgments.
- `check_triggers` verifies that session probes really happened after compaction. A memory test is not useful if memory trimming never occurred.
- `report` builds side-by-side tables and token curves from graded runs.
- `handgrade_workbook` builds and merges a blinded grading workbook, so the grader does not know which preset produced an answer.

Every turn persists a JSON trace bundle, so grading and reporting can re-run offline without touching the API. Runs are resume-safe: re-run the same command after an interruption and completed units are skipped. Each turn has a 10-minute timeout, and LangSmith is off by default. Running the tutor needs `GEMINI_API_KEY` + `COHERE_API_KEY`; the full 4-preset comparison run below cost **≈$330** at corrected pricing (the pre-2026-06-13 price table under-priced `gemini-3.5-flash` ~4.4×; the often-quoted "$73" is a pre-correction figure — see Harness corrections).

**Setup for collaborators.** Code and docs are in git; the datasets and run results contain real student text and live only in the private HF dataset (`towardsai-tutors/ai-tutor-data` — git is force-pushed to the public prod Space on deploys, so student data never enters git). With an `HF_TOKEN` that can read it:

```bash
export HF_TOKEN=hf_...    # token with READ access; the inline cmd does NOT load .env (or: uv run huggingface-cli login)
mkdir -p data/eval runs  # runs/ is gitignored, so a fresh clone has no runs/ dir yet
uv run python -c "from huggingface_hub import snapshot_download as d; d(repo_id='towardsai-tutors/ai-tutor-data', repo_type='dataset', allow_patterns=['eval/**','eval_runs/**'], ignore_patterns=['eval/README.md'], local_dir='.')"
mv eval/* data/eval/ && mv eval_runs/part_*/* runs/   # restore working paths (part_b, part_c, ...)
```

## Results — Part B comparison run, 2026-06-12

4 presets × 2 trials over the full single-turn battery, full persona battery, and 3 sessions (s01 13-turn, s02 agentic, s08 fact-update). 1,232 turns, zero API errors. Tables: `runs/b_report/report.md`; token curves: `tokens_by_turn.csv`. **Auto-graded only** means the current table includes metrics computed by code. Answer quality and session probe accuracy appear after the human grading workbook is filled and merged.

| | full_history | prod | profile_memory | aggressive |
|---|---|---|---|---|
| personalization pass (personas, n=70) | 63% | 67% | **94%** | 56% |
| sessions: est. cost/turn (corrected pricing) | **$0.112** | $0.237 | $0.218 | $0.296 |
| sessions: median time to first answer text | **17s** | 21s | 22s | 43s |
| sessions: tool calls/turn | **2.8** | 3.5 | 3.9 | 8.3 |
| sessions: cumulative input tokens | 2.9M | 1.7M | 1.6M | 1.75M |
| single-turn: behavior proxy from code checks (n=106) | **88%** | 86% | 86% | 78% |
| single-turn: retrieval recall@shown source | 51% | 50% | 49% | 47% |

Row definitions:
- **personalization pass** — % of persona question-runs whose answer passed every authored check (expected regex matched, e.g. `conda` for the conda persona) with no anti-pattern hit (e.g. bash `export` for a Windows persona). n=70: 40 questions × 2 trials, minus 10 runs whose checks need human judgment.
- **est. cost/turn** — mean estimated $ per conversation turn: token counts × `MODEL_PRICING`, cached input billed at the cache-read discount (our table, not the invoice).
- **median time to first answer text** — median seconds from user message to first visible answer text. It includes tool calls and internal model rounds before the visible answer starts, so it approximates the user's perceived wait.
- **tool calls/turn** — mean retrieval + KB-command invocations per turn; here it's the re-work signal (compaction presets re-search for evidence their compressed history lost).
- **cumulative input tokens** — all input tokens billed across an entire session (every turn, every internal call), mean over the 6 session-runs per preset.
- **behavior proxy from code checks** — % of single-turn case-runs where a cheap code check confirms the right *kind* of response (course question → called retrieval/KB; support issue → points to support; feedback → acknowledges). n=106 of 120: `answer_general` has no proxy check. This is only an early signal; authoritative behavior accuracy comes from human grades.
- **retrieval recall@shown source** — % of case-runs where the reranked retrieval results the agent saw contained at least one chunk from the correct course. Stricter `recall@lesson` (right lesson page, 30–36%) and the right-lesson ranking score (MRR) are in the full report.

### Observations

Numbered findings; each states what it was tested on. Convention: entries are never edited, only superseded.

- **F1 — Retrieval payloads dominate input tokens, not conversation history.** Each retrieval call may return up to `DEFAULT_CONTEXT_TOKEN_BUDGET = 100_000` tokens; turns average ~200k input. (All runs; high confidence.) → retrieval budget became a Part C variant dimension.
- **F2 — Compaction saves tokens but not necessarily dollars.** Summarization rewrites the prompt prefix and invalidates Gemini's implicit cache: on the same session, `full_history` had 86.8% of input billed at the ~4x cache discount; `aggressive` used 44% fewer tokens yet cost 68% more. (s03 × 3 presets, then confirmed n=24 session-runs; provider-specific — Anthropic caching is explicit.)
- **F3 — Clearing old tool-output messages never fires on this workload.** `ClearToolUsesEdit` excludes retrieval results — where the tokens are; 0 clears across all session runs. → `editing_only` dropped from Part B; "clear retrieval results too" variant queued for Part C.
- **F4 — Aggressive compaction degrades even single-turn behavior.** 18.0 vs 9.6 LLM calls/turn, 57s vs 39s median, behavior proxy −10pts vs full_history (60 cases × 2 trials). Mid-turn compaction churn changes agent behavior, it doesn't just trim.
- **F5 — Gemini reasoning tokens are ~90%+ of output even with reasoning display off.** Billed as output; dominates latency; recorded per model in `usage_by_model`.
- **F6 — Long context costs latency, not correctness.** Zero API errors in 1,018+ turns at any size (max 6.06M tokens across one turn's calls; largest single context 274k). Median time to first answer text scales 22s → 76s from <100k to >800k input tokens/turn. Compaction's value here is responsiveness and spend, not keeping the model functional.
- **F7 — ~0.8% of turns produce no answer text** despite tool calls and billed reasoning tokens; preset- and size-independent. Candidate golden-case assertion ("answer non-empty").
- **F8 — Profile memory wins on quality AND cost.** 94% personalization vs 56–67% without, while the cheapest persona preset ($0.049 vs $0.081–0.095/turn, 7.6 vs 10–11.2 tool calls): the stored profile saves the agent from re-searching for user context. (40 questions × 2 trials × 4 presets, auto checks; LLM-check rows pending human grades.)
- **F9 — Full history is cheapest AND fastest up to 13 turns; compaction causes re-work.** Sessions: $0.034/turn, time to first answer text 17s, 2.8 tool calls for `full_history` vs $0.051–0.066, 21–43s, 3.5–8.3 elsewhere — despite ~2x the tokens (F2's cache mechanism plus a second one: raw history lets the agent re-use earlier retrieval evidence; summaries force re-retrieval). (3 sessions × 4 presets × 2 trials, n=24, zero errors.) The conventional pitch inverts: under modern prompt caching, the naive baseline wins short-to-medium sessions; compaction must justify itself on quality (pending human grades) and long horizons. Where the crossover actually sits is a Part C question.
- **F10 — Compaction degrades within-session memory, and what it drops is *old* facts.** Session-probe accuracy: `full_history` **92%** vs `prod` 38% / `profile_memory` 38% / `aggressive` 42% (n=24/preset). The collapse is entirely turn-0 material — `fact_recall` 100% → 17–25%, `preference_compliance` 100% → 0% — while `fact_update` stays **100% across all presets** (the mid-session update sits inside the kept-recent window; summarization evicts the early planted facts, not the recent ones). With F9 (full_history also cheapest and fastest here), the naive baseline wins cost, speed, AND memory on ≤13-turn sessions — the workshop headline. (3 sessions × 4 presets × 2 trials; **provisional — LLM-graded under 4 reviewer rubric policies, pending human review of the 96 probes**.)
- **F11 — Profile memory helps personalization, not working memory.** `profile_memory` reached 92% persona personalization (vs 54–66% without) yet scored **38% on session probes — identical to `prod`**: the long-term student-profile store improves fresh-chat answers while the live thread is still summarized exactly as in `prod`. Long-term store ≠ in-session retention; they are independent subsystems and a preset can win one while tying the other. (Personas n=80 auto; sessions n=24, provisional.) Refines F8.
- **F12 — Aggressive compaction is dominated on every axis.** Beyond F4's cost/latency blowup (18 LLM calls/turn, 57s median), it is also worst on quality: key-point coverage 36% vs 72% (`full_history`), single-turn behavior 67% vs 83–92%, session probes 42% vs 92%. No measured metric favors it. (Single-turn quality n thin at 12–25; sessions n=24; provisional.)
- **F13 — Session-probe grades are human-confirmed (zero overrides), so F10–F12's memory numbers are no longer provisional.** 2026-06-15 Omar reviewed all 96 session probes — the 83 high-confidence LLM verdicts by sign-off and the 13 low-confidence ones individually — and agreed with every grade. The provisional LLM grading therefore stands as human ground truth, confirming the 92%-vs-38% probe-accuracy result (F10) and the probe components of F11/F12. Caveat: only the session-probe tier was human-reviewed; the non-probe quality rows (persona, key_point, behavior — feeding key-point coverage and single-turn behavior in F8/F12) remain LLM-graded. The judge has **not** yet been validated against these labels (next step), so judge-graded Part C numbers stay gated.

### Part C screen (2026-06-15)

11 arms (prod + full_history anchors + 9 variants), each on a fixed subset (24 stratified single-turn + 3 sessions s01/s02/s08), 1 trial, ~660 turns, 0 errors, ≈$166 at correct pricing. Graded by the validated subagent judge. Full table: `runs/c_report/report.md`. Probe-accuracy n=12/arm (1 trial) — treat percentages as coarse rankings, not precise rates; promotion (×3 trials, full batteries) would firm them up.

- **F14 — The LLM judge is validated; Part C quality columns are reportable.** A blind subagent re-grade of the 96 human-confirmed probes reproduced them at 98% agreement / TPR 100% / TNR 96% (clears the >90% gate; this measures grader *reproducibility*, since the labels were sign-offs, not blind-from-scratch). The validated grader is the subagent workflow (same per-item-type rubric as `evals/judge.py`), run on subscription at no API cost; all Part C arms are graded that way. Supersedes F13's "judge not yet validated" caveat. (`runs/b_report/judge_val/`.)
- **F15 — The screen reproduces F9/F10 on independent arms.** `full_history` is cheapest on sessions ($0.10/turn) AND best memory (100% probe); every compaction arm is both pricier and weaker. `prompt_compression` also reaches 100% memory — it shrinks message text but drops no facts — yet saves little (highest tokens), so "don't drop content" preserves memory but is not a cost win. (Screen; n thin.)
- **F16 — `incontext_history_retrieval` works but is short-session-neutral.** 83% probe accuracy (best after full_history): retrieving the relevant old turns restores what summarization loses. But on ≤13-turn sessions it can't drop much, so it costs ≈ full_history (~$0.20/turn) plus embed overhead; its cost payoff needs LONG sessions (a `_v2` battery). The principled cost answer to F9, pending a long-horizon test; separate from the higher-priority v2 contradiction-precision question.
- **F17 — Turning the KB off inverts retrieval recall (Axis B tradeoff).** `kb_off` forces `retrieve_tutor_context` instead of browsing → single-turn recall@shown jumps 50%→96%, but it is the priciest session arm ($0.31/turn) and worst memory (33%). KB browsing is cheaper and better for memory yet actively *lowers* top-k retrieval recall (the agent browses instead of retrieving the labeled source). Not a clear win either way.
- **F18 — The 100k retrieval budget is over-provisioned.** `retrieval_budget_30k` matches prod's recall@shown (50%) at a third of the budget → tokens can be cut with no recall loss on this subset (direct confirmation of F1). The 10k rung is untested.
- **F19 — `observation_truncation` backfires (Axis B negative).** Head/tail-truncating tool outputs in the model's view makes the agent re-call tools to recover what was cut: 22.8 LLM calls/turn, 430s p95 latency (single-turn), memory only 33% — the same churn pathology as `aggressive` (F4).
- **F20 — Axis-A summary/trim arms do not rescue memory.** `context_reset` (17% probe, 56% key-points — dominated on every axis), `selective_retention` (25%), and `sliding_window` (42%) all stay well below `full_history`/`prompt_compression` and none beats `prod` (58%): dropping or summarizing early turns loses the planted facts the probes test (consistent with F10).

- **F21 (2026-06-16) — Supersedes F11 in part: `profile_memory` was dormant on the sessions battery, so its 38% is `prod`, not a test of the profile.** `evals/run_battery.run_session` passes no `student_id`, and `StudentProfileMiddleware` injection plus the post-turn write-back both no-op without one — so on sessions `profile_memory` reduces exactly to `prod` and could not differ. The 38% is real but comes from the live conversation thread + prod's lossy summary (the student states the facts in-thread; summarization keeps the recent ones and a compressed summary of the rest), **not** from any profile — which is why it is ~38%, not 0. F11's reading that this shows "long-term store ≠ in-session retention, independent subsystems" is therefore unsupported: the store was never engaged in-session. Whether a store that captures in-session facts verbatim and re-injects them recovers recall under compaction is **open** — the `collection_memory` v1 test (sessions get a `student_id` + an append-atomic-facts write-back; goalposts: compaction 17–25% vs `full_history` 100%). F11's personas number (94%) stands; personas do set a `student_id`.

- **F22 (2026-06-16) — Activating `profile_memory` on sessions rescues much of the old-fact loss, but does not replace full history.** Experiment A reran the three Part-C sessions with `run_session` passing a per-session/per-trial `student_id`, so the profile store was actually read/written and injected through the system prompt on every turn. With compaction active at 100% of probes, probe accuracy rose to **75% (9/12)** vs same-screen `prod` **58% (7/12)**, while `full_history` stayed **100% (12/12)**. The gain was concentrated exactly where F10/F21 predicted: `fact_recall` improved to **83% (5/6)** vs `prod` **33% (2/6)**, and `preference_compliance` to **100% (1/1)** vs `prod` 0%. It still missed two reference/consistency probes, and was not a cost win on this short screen ($0.265/turn vs `prod` $0.229 and `full_history` $0.103). Interpretation: injecting stored facts outside summarized history can recover many facts lost by compaction, but the current "5 durable lines" profile write-back is an incomplete working-memory store. Next test: `collection_memory` / verbatim atomic facts to determine whether the remaining gap is extraction/storage loss vs answer synthesis. **Caveat (thin n):** the only robust signal is `fact_recall` (5/6 vs 2/6) and the 9/12-vs-7/12 aggregate — the `preference_compliance`/`fact_update` cells are n=1 anecdotes; and `profile_memory` actually *regressed* on anaphora / anaphora_consistency (50% vs prod's 100%, n=2 each), possibly profile injection distracting from reference resolution, possibly noise. (Codex handgrade at Omar's request; one embeddings row was borderline-pass, strict fail would make this **67% (8/12)** without changing the direction.)

- **F23 (2026-06-17) — F17's recall inversion was a measurement artifact: `recall@shown` is blind to KB grounding, and a KB-fair metric erases the gap.** `recall@shown`/`recall@lesson`/MRR count only `retrieve_tutor_context` matches, so when `prod` grounds by *browsing* the KB instead of retrieving, the labeled source is invisible to the metric — which is exactly `kb_off`'s only structural difference, so the 50%→96% "inversion" was partly definitional. Re-grading the same Part C single-turn bundles (no re-run; `runs/kbfair_report/`) with two tool-agnostic measures: **recall source (any tool: retrieval+KB)** = `prod` **100%** vs `kb_off` 96% (n=24), and **cited-correct source/lesson** (does the *answer* cite the labeled source/lesson, resolved from both tools via `kb_manifest`) = **100%/100% for both** (n=14 corpus). Verified the any-tool credit is real, not a regex hit: in all **12** prod cases where retrieval missed, the agent had browsed that exact gold source via `run_kb_command`. So turning the KB off does **not** improve grounding — both configs find and cite the right source. `kb_off`'s genuine, non-artifact wins remain latency (19s vs 38s p50 TTFT), efficiency (3.6 vs 10.1 LLM calls/turn), and a slight single-turn quality edge (behavior 88% vs 79%, key-points 82% vs 78%) — all from doing one retrieval call instead of many browse rounds, not from better recall. **Supersedes F17's recall reading; F17's cost/memory points stand.** Caveats: n=24 screen subset, 1 trial; `cited_correct` saturates at 100% here (coarse at this n), so any-tool recall is the discriminating fair metric; bundle KB output is capped at 6000 chars (could undercount KB hits, which only biases *against* prod — the true gap can be smaller, not larger). New auto metrics in `evals/grade.py`: `recall_anytool_source`, `cited_correct_source/lesson` (free to recompute on any saved bundle).

**Screen winners → promotion candidates:** `incontext_history_retrieval` (needs the long-session test), `retrieval_budget_30k` and `clear_retrieval_kb` (cheap Axis-B wins; clear_retrieval_kb is cheaper than prod with similar memory + better recall/key-points), anchored by `full_history`. **Drop:** `context_reset`, `observation_truncation`, `selective_retention`.

**Part C — what remains (beyond the workshop).** The screen answered the core question; everything below *hardens* or *extends* it and is **gated on a spend or data decision from Omar** — none is required for the workshop.

1. **Promotion (rigor, not new findings).** Re-run the 2-3 winners (+ 1-2 combos) on full batteries × 3 trials with paired stats, to turn the screen's coarse n=12 rankings into confident numbers. **~$1,300-1,800 at corrected pricing** (the old "$300-400" was at pre-correction prices; × the 4.4 fix — estimate precisely before running, since the per-turn cost is now ~$0.25). Detail in "Part C — rigorous" below.
2. **The v2 contradiction test — the highest-value new dataset.** Moderate sessions are enough: plant A, update to A′, then probe after prod compaction has evicted A from the kept-recent window. This tests the one correctness regime where `full_history` itself might lose (it sees both A and A′ and must choose the current one). If only one v2 slice is built, build this first.
3. **The `incontext_history_retrieval` long-session cost test — principled but less product-critical.** The screen showed `incontext` works (83%, F16) but couldn't show a cost benefit on ≤13-turn sessions. Whether retrieve-old-turns beats `full_history` *once sessions get genuinely long* needs a token-calibrated long-session v2 tier; useful for the workshop argument, but gated by whether that regime matters in real tutor telemetry.
4. **Conditional builds.** Build `temporal_graph_memory` only if `full_history` fails contradictions, `delta_summarization` only if the long-horizon cost tier shows room to beat full history, and `entity_memory` only if multi-project probes show fact bleed. `collection_memory` is a separate cheap v1/product follow-up for verbatim fact storage (F22 tested the current profile write-back, not that). `hierarchical_summarization` and `sleeptime_consolidation` are deferred. The step-by-step v2 plan is in `evals_part_c_plan.md` → "Battery v2".

**V2 execution note (2026-06-16).** A private/gitignored 6-session `battery_sessions_v2.jsonl` was built and partially run. The full v2 screen is too slow under the lower-tier Gemini key: `full_history` at concurrency 3 hit the 3M input-tokens/minute quota, and concurrency 1 works but makes the full 4-arm screen a multi-hour job. Current local artifacts (`runs/e2_v2_partial_report/report.md`) are directional, not final: `prod` completed all 6 sessions with 0 errors, all probes under compaction, **14% probe accuracy (1/7)** and contradiction **0/3**; `full_history` completed 2 Tier-1 contradiction sessions with 0 errors, no compaction, contradiction **2/2**. Priority is now **Tier 1 only**: finish the missing `full_history` contradiction session, then run Tier-1 `profile_memory` and optionally Tier-1 `incontext_history_retrieval`.

Harness corrections (bugs in our measurement, not findings): the overnight 06-12 stall was machine sleep hanging API streams (all four pipelines stopped the same minute; fixed with the per-turn timeout); battery lesson-URLs carried a `/discussions/` suffix that silently zeroed recall@lesson until normalized (caught by the A3 smoke, re-graded from bundles without re-running).

- **Pricing correction (2026-06-13).** `MODEL_PRICING` under-priced `gemini-3.5-flash` before this date (~$0.30/$2.50 per MTok vs the correct **$1.50 input / $9.00 output / $0.15 cache-read**, verified against Google's price sheet). Every dollar figure generated earlier — the Part B table and the inline costs in findings **F2/F8/F9/F12** — is **~4.4× too low**; token counts are unaffected. Part B bundles were re-costed from their saved token counts (2026-06-15) and `runs/b_report/report.md` regenerated at correct pricing: **relative rankings are unchanged** (F9 still has `full_history` cheapest, $0.11/turn sessions), only absolute dollars move. Real Part A+B spend ≈ **$338**, not ~$73. Use the regenerated report for absolute costs; finding dollars above are pre-correction.
- **Stable `sheet_row_id` (2026-06-15).** `evals.grade` built `sheet_row_id` with builtin `hash()`, which is salted per process, so regenerating a `handgrade_sheet.csv` produced ids that no longer matched the frozen workbook keymap — silently emptying `handgrade_workbook merge`. Switched to a `hashlib.md5` hash; the Part B re-merge was rebuilt via the deterministic `run_id` to recover the human grades.

Talk-outline correction: the planned "skills / just-in-time KB instructions" change (Change 4) was never built, and its premise is off — the KB instructions block measures **458 tokens** (not ~4k) inside a 1,655-token system prompt; max saving ~2% of a turn, mostly cache-discounted. Recommended reframe: demo progressive disclosure on tool *outputs* (F1), where our numbers actually are.

## Remaining work

**Omar**
- [x] Grade the session probes (the critical tier). 2026-06-15: all 96 reviewed — the 83 high-confidence LLM verdicts by sign-off, the 13 low-confidence ones individually — with **zero overrides**, so the provisional LLM grades are now human-confirmed ground truth (see F13). The non-probe rows (persona / key_point / behavior in `review_other.md`) were not part of this pass and remain LLM-graded; bless them too if those columns need a human label.
- [ ] Audit `data/eval/review_log_v1.md` (the 11 flagged judgment calls first).
- [x] Verify and update the `gemini-3.5-flash` entry in `app/telemetry.py:MODEL_PRICING` against the current price sheet.

**Workshop**
- [ ] Replace placeholder slide numbers with measured ones; pick 2–3 failure traces from the graded probes for the failure→fix→number beats (a `fact_update` failure under summarization is the target demo).
- [ ] Token-vs-turn plot: CSV is ready; add matplotlib (or plot elsewhere) for the PNG.
- [ ] Next.js meter component over the `data-context-stats` SSE part, if the live demo needs it.
- [ ] Skills section: reframe per the 458-token measurement above.

**Part C — widen the comparison (workshop).** Part B compared four presets; Part C adds the strategies from the context-engineering syllabus along the **two axes** defined in the plain-English map above — **A: memory & context management** and **B: retrieval & tool outputs**.

_Build status (2026-06-15)._ The Phase-0 foundation and the Part C screen are implemented, tested, adversarially reviewed, run, and judge-graded. Built: Axis B `clear_retrieval_kb`, `kb_off`, `observation_truncation`, `retrieval_budget`; Axis A `sliding_window`, `prompt_compression`, `selective_retention`, `context_reset`, and the `incontext_history_retrieval` subsystem. Deferred / conditional: `delta_summarization` only after a long-horizon v2 tier shows a real cost regime; `hierarchical_summarization` dropped unless extreme-length sessions matter; `collection_memory` is an optional v1/product follow-up for verbatim fact storage; `entity_memory` waits for multi-project fact-bleed probes; `sleeptime_consolidation` waits for a multi-thread battery and runner hook. Build/run/grade status lives in `evals_part_c_plan.md`. Each new strategy is run as a *single-axis* change: take the `prod` baseline, alter one variant on one axis, leave everything else fixed, and compare. This isolates each strategy's effect and keeps the run count additive (sum of variants); we do *not* test the full cross-product (every Axis-A × Axis-B combination), which would multiply the matrix and the bill. Only the 1–2 most promising *combinations* are tested afterward, once the single-axis winners are known. Ordering follows Part B's findings: the tokens live in tool outputs, not chat history (F1, F3), and compaction can cost more dollars than it saves (F2, F9) — so the cheap retrieval/tool-output variants run first, and the history-compaction variants run to *confirm that inversion* rather than on faith.

_Gate (hard dependency)._ No judge-graded number is reportable until the LLM judge is validated, and the judge is validated only against the Part B hand-grades. The order is forced: grade `workbook.csv` → build the judge (strong Anthropic model, binary verdicts) → measure agreement → use it only at >90% true-positive AND true-negative on held-out labels. Until then Part C reports auto-gradable rows only (cost, latency, tokens, tool calls, retrieval recall, persona auto-pass).

_Gate status (2026-06-15): CLEARED._ A blind subagent re-grade of the 96 human-confirmed probes hit **98% agreement, TPR 100%, TNR 96%** (PASS; details in `runs/b_report/judge_val/validation_summary.md`). Caveat: this measures grader *reproducibility* (the confirmed labels were sign-offs of an earlier LLM grading, not blind-from-scratch), so it shows the automated grader stably reproduces the blessed grades, not judge-vs-independent-human. The **validated grader is the subagent workflow** (same per-item-type rubric as `evals/judge.py`, run on subscription not API) — Part C grading should use that same path so the deployed grader equals the validated one.

_Per-arm harness work._ Every new mechanism needs its own `context_stats` signal, or `evals.grade` / `evals.check_triggers` will not see it fire — they currently detect compaction only via `summary_messages` / `cleared_tool_outputs`, so a sliding-window or truncation arm would read as "never compacted" and the probe gate would false-fail. Cost/token/report code is generic and unchanged.

_Run discipline._ Single-axis vs `prod`; cheap screen on the single-turn battery + 3 sessions; drop the losers; promote only the 2–3 winners (plus 1–2 promising combinations, e.g. `profile_memory` + clear-retrieval) to full batteries × 3 trials. New probe types (contradiction, fact-update, long-horizon recall) need a frozen `_v2` battery — never edit v1 in place.

Axis A — **memory & context management** (preset-shaped = a `MemoryConfig` flag + middleware; subsystem = a store + tools, *not* a preset):

| Variant | Shape | Tests / expectation |
|---|---|---|
| sliding window / trimming (keep recent N) | preset | cheapest trim; risks dropping planted facts the probes need |
| delta summarization (running summary, new facts only) | preset | watch cache invalidation (F2) |
| hierarchical summarization | preset | expected to matter only on the longest sessions |
| context reset seeded with summary | preset | prefix rewrite → cache confound (F2) |
| prompt compression | preset | prefix rewrite → cache confound (F2) |
| selective retention (keep constraints/decisions/state) | preset (summary prompt) | quality-preserving compaction |
| in-context retrieval over chat history | subsystem | the principled answer to F9 (retrieve old turns vs carry/summarize all history) |
| collection memory (many small fact docs) | subsystem | + an explicit memory-write/merge tool |
| entity memory (per student / course / project) | subsystem | profiles keyed by entity |
| sleep-time consolidation | subsystem + runner hook | background cleanup between sessions |

Axis B — **retrieval & tool outputs** (where F1 says the tokens actually are):

| Variant | Shape | Tests / expectation |
|---|---|---|
| observation truncation (head/tail tool outputs, incl. KB) | preset | trims the dominant token source |
| clear retrieval + KB results too | preset | fixes F3 (current clearing excludes them, so it never fires) |
| retrieval budget sweep 100k / 30k / 10k | config knob | tests F1 directly; `recall@shown` shows when a tighter budget drops the right source |
| KB on/off | tool toggle (+ drop the KB prompt block) | "agentic browse vs top-k RAG", and **does the KB improve answers?** KB is the dominant grounding tool — ≈89% of Part B turns, ~7.7 KB vs ~0.9 retrieval calls/turn (measured over 1,088 turns), and Part B never ran without it. Hypothesis: turning it off may *raise* cost/latency as the agent falls back to 100k-token retrieval payloads. |

**Stretch (heavier, likely post-workshop):** temporal-graph memory — defensible because the `fact_update` probe is a built-in scorecard; sub-agent isolation — reframed as a tool-output-token play (keep noisy `run_kb_command` output out of the main context; ties to F1).

**Dropped (low information given the findings):** the skills / lazy-prompt-loading family (the KB-instructions block is 458 tokens, so its only measurable win folds into observation truncation), procedural memory, GraphRAG, and multi-agent / parallel-research agents.

**Part C — rigorous (post-workshop, ~$1,300–1,800 at corrected pricing; the original ~$300–400 was at the pre-2026-06-13 under-price — see Harness corrections).** The workshop winners, re-run on full batteries (60 single-turn + 32 sessions + 30 replay) × 3 trials, with paired statistical tests, consistency across repeats, and failure-taxonomy diffs per variant; plus an Anthropic (Haiku) re-run of the prefix-rewriting arms to test whether the cost ranking flips under explicit caching (F2). Builds deferred to here: temporal-graph memory, sub-agent isolation, and any `_v2` battery for contradiction/temporal probes.

**Product quality track (post-workshop)**
Golden cases as a CI gate (5–10 critical-path cases with deterministic assertions, including the F7 non-empty check) → error-analysis cycles (an expert reads 50–100 traces in a small viewer, records pass/fail plus the first failure reason, and stops when new failure types stop appearing) → failure taxonomy → code assertions for recurring modes → LLM judges built from the hand-grade critiques and validated to >90% true-positive and true-negative rates on held-out human labels before any judge-graded number is reported → nightly battery + weekly trace sampling. Full methodology behind each step: `evals_background.md`.

## Files

- `evals.md` — this file: the what, the data, the results, the queue.
- `evals_part_c_plan.md` — Part C execution plan: orchestration (workflow vs direct), the variant catalog with exact wiring + telemetry signals, the subset run matrix, and cost.
- `evals_background.md` — research sources (Hamel, howtoeval, OpenAI macro-evals) and design rationale.
- `data/eval/README.md` — battery schemas + glossary of terms; `review_log_v1.md` — dataset audit trail.
- `evals/` — the harness code; `app/memory_presets.py`, `app/telemetry.py` — the app-side hooks.
- `runs/b_report/` — generated: final tables, token curves, the blinded grading workbook.
