# Part C — execution plan

The **how** for Part C. The **what** (scope, the two axes, the variant catalog, the findings that motivate them) lives in `evals.md` — see "What we evaluate, and what varies" for the arms/axes and the "Part C screen" results for the findings. This doc is the build/run/grade plan: orchestration, wiring, schedule, cost, and status. Read `evals.md` first.

## Decisions locked (2026-06-14)

- **Scope: everything, including the subsystem memory methods** (collection/entity memory, in-context retrieval over chat history, sleep-time consolidation), not just the cheap preset variants.
- **Budget: subset-screen first.** Run every variant on a curated subset (~$50-150), promote only the 2-3 winners (plus 1-2 combinations) to full batteries x 3 trials. Spend is checkpointed before any batch run.
- **Orchestration: hybrid.** Dynamic workflows for the embarrassingly-parallel phases (runs, judge-grading, isolated subsystem builds); direct/coherent code for the shared-file foundation. See the table below.

## The gate (hard dependency)

No **quality** column (key-points, behavior, memory probes) is reportable until the LLM judge is validated against held-out **human** labels. The judge harness exists (`evals/judge.py`); validating it needs Omar's grades on the calibration set — the same 96 session probes + slice that also confirm F10. Until then, Part C reports only the auto-gradable columns (cost, latency, tokens, tool-calls, retrieval recall, persona regex), which need nothing from anyone.

**Gate CLEARED 2026-06-15.** The 96 probes are human-confirmed (zero overrides), and a blind subagent re-grade reproduced them at **98% agreement / TPR 100% / TNR 96%** (PASS; `runs/b_report/judge_val/validation_summary.md`). Decision (Omar): grade via **subagent workflow on subscription**, NOT `evals/judge.py` on the API — so the validated grader and the deployed grader are the same subscription path (judge.py stays as the rubric reference + an API fallback). Caveat: reproducibility, not independent-human (labels were sign-offs). Quality columns are now reportable once the screen runs exist.

```
Omar grades calibration set  ->  evals.judge validate  ->  (>=90% TP/TN?)  ->  judge grades all Part C variants
                                                                  |
                                                              if no: refine judge prompt, or hand-grade a per-variant subset
```

## Orchestration model — workflow vs direct

| Work | Tool | Why |
|---|---|---|
| Foundation: `MemoryConfig` fields, middleware classes, telemetry signals, gate fix, `disable_kb` flag, judge module | **Direct (me)** | All edit the same files (`build_agent_middleware`, `MEMORY_PRESETS`, `telemetry.py`, the gate). Parallel agents here only create merge conflicts. One coherent, tested pass. |
| Preset-variant implementations (once the pattern exists) | **Direct + targeted workflow** | After Phase 0 makes variants pluggable (each = a small self-contained middleware + one preset entry), low-contention; small ones by hand, the rest drafted against the pattern. |
| Subsystem builds (collection/entity memory, in-context retrieval, sleep-time) | **Workflow, worktree-isolated** | Bigger, independent; each agent in its own git worktree, integrated one at a time. |
| Variant battery runs (every variant x subset) | **Workflow** | Independent `run_battery` invocations, no shared-file writes. Textbook parallel fan-out. |
| Judge grading (every variant's `handgrade_sheet`) | **Workflow** | Same shape as the 284-row workbook grading already done. |

## Phases

### Phase 0 — Foundations (direct, ~no API cost)
- [x] **Judge harness** (`evals/judge.py`): `run` auto-fills `handgrade_sheet.csv` -> `judge_filled.csv` (Anthropic `claude-opus-4-8`, structured output, blinded, per-item-type rubrics); `validate` reports TP/TN vs human labels, gates at >=90%. Slots into `run_battery -> grade -> judge run -> grade --handgrades -> report`. Lint-clean, smoke-tested.
- [x] **Variant foundation**: `MemoryConfig` extended (`summary_prompt`, `sliding_window_keep`, `truncate_tool_outputs`/`truncate_*_chars`, `compress_prompt`, plus the earlier `clear_excludes_retrieval`); three `AgentMiddleware` subclasses (`SlidingWindowMiddleware`, `ObservationTruncationMiddleware`, `PromptCompressionMiddleware`) wired into `build_agent_middleware`. See "Foundation pattern" below.
- [x] **Telemetry signals**: added a turn-scoped signal registry to `app/telemetry.py` (`reset_turn_signals`/`record_turn_signal`/`pop_turn_signals`, keyed by `message_id`), merged into the `context_stats` event in `stream_chat`. This is the general fix for the gotcha: per-call-view middlewares (sliding window, truncation, compression) never reach the checkpoint, so `context_window_stats` can't see them; they tally here instead. `COMPACTION_SIGNAL_NAMES` is the canonical list.
- [x] **Gate fix**: generalized `compaction_active` into `evals/common.py` (mirrors the app's `COMPACTION_SIGNAL_NAMES` as `COMPACTION_SIGNAL_KEYS` — the harness must not import app), used by `evals/grade.py` and `evals/check_triggers.py`. A unit test guards the two lists from drifting. Sliding-window/truncation arms now register as "compacted" so the probe gate doesn't false-fail.
- [x] **Retrieval-budget knob**: per-request `retrieval_budget` threaded `ChatRequest -> AppContext -> retrieve_tutor_context -> LocalChromaRetriever.search(token_budget=) -> _apply_token_budget`; `run_battery --retrieval-budget`. `None`/`0` keeps the default 100k (prod byte-identical).
- [x] **`disable_kb` flag**: added to `ChatRequest`, threaded through `build_agent`, `effective_tool_names`, `build_system_prompt`, and both call sites; `run_battery --disable-kb`. KB tool + prompt sections removed, retrieval intact (verified, tests green).

### Phase 1 — Judge validation (needs Omar's labels) [GATE — CLEARED 2026-06-15, see "The gate" above]
- [x] Omar graded the calibration set: all 96 session probes, zero overrides → F13.
- [x] Validated via a blind **subagent-workflow** re-grade (not `evals.judge.py`-on-API): 98% agreement / TPR 100% / TNR 96% (PASS). That subagent workflow is the deployed grader; `evals.judge.py` stays as the rubric reference + API fallback.

### Phase 2 — Axis B: retrieval & tool outputs (cheap, highest-info, mostly auto-graded)
Variants: `retrieval_budget_100k/30k/10k`, `clear_retrieval_kb`, `observation_truncation`, `kb_off`. Each: implement + signal, screen vs `prod`. Headline metrics (recall@shown, cost, latency, tokens, tool-calls) are auto — results land even before the judge.

### Phase 3 — Axis A: memory & context management (preset-shaped)
Screened preset-shaped variants: `sliding_window`, `context_reset`, `prompt_compression`, `selective_retention`. Report **$ and tokens both** (F2 cache confound); watch the re-work signal (tool-calls/turn). Conditional follow-up: `delta_summarization` only after the long-horizon v2 tier shows a real cost regime; `hierarchical_summarization` is dropped unless a later extreme-length stress test justifies it.

### Phase 4 — Axis A subsystems (conditional, worktree-isolated workflow)
Built: `incontext_history_retrieval`. Future subsystem builds are data-driven: `temporal_graph_memory` only after contradiction probes show `full_history` choosing stale facts; `entity_memory` only after multi-project probes show fact bleed; `collection_memory` is an optional cheap v1/product follow-up for verbatim fact storage; `sleeptime_consolidation` stays deferred because it needs a multi-thread battery plus a between-session runner hook.

### Phase 5 — Promote + report
Winners (2-3) + 1-2 combinations (e.g. `profile_memory` + `clear_retrieval_kb`) -> full batteries x 3 trials. Optional Anthropic (Haiku) re-run of the prefix-rewriting arms to test the F2 cost-flip under explicit caching. Final tables + new findings (F13+).

## Foundation pattern (worked examples, learned 2026-06-14)

How a variant is actually wired, so a fresh session can replicate it.

**Middleware hook API.** Custom middlewares subclass `AgentMiddleware` and override `wrap_model_call(self, request, handler)` (+ async `awrap_model_call`): mutate the request via `request.override(messages=…, system_message=…, model_settings=…)`, then `return handler(request)`. Templates in `chat_service.py`: `SourcePreferenceMiddleware` (:803), `StudentProfileMiddleware` (:852). The built-in *state-rewriting* compaction (`SummarizationMiddleware`, `ContextEditingMiddleware`) is assembled in `build_agent_middleware` (:894) from `MemoryConfig` flags.

**Telemetry-signal gotcha (now solved by the turn-signal registry).** `context_window_stats` (`app/telemetry.py`) runs over the **checkpointed** messages (called near `chat_service.py:1526`) and detects markers: `lc_source: summarization` and the `CLEARED_TOOL_OUTPUT_PLACEHOLDER`. A middleware that only reduces the *per-call* view via `wrap_model_call` leaves the checkpoint unchanged, so it would emit no signal. The general fix is in place: a per-turn signal registry in `app/telemetry.py` (`reset_turn_signals(message_id)` at turn start, `record_turn_signal(message_id, name, n)` from the middleware, `pop_turn_signals(message_id)` merged into the `context_stats` event). A new per-call-view mechanism just calls `record_turn_signal` with a name, adds that name to `COMPACTION_SIGNAL_NAMES` (app) **and** `COMPACTION_SIGNAL_KEYS` (`evals/common.py`, kept in sync by a unit test), and the gate + report pick it up. A module-level dict + lock (not a `ContextVar`) because LangChain may run sync `wrap_model_call` in a worker thread.

**Worked example 1 — `clear_retrieval_kb` (built, verified).** One `MemoryConfig` field `clear_excludes_retrieval` (default True = prod). `build_agent_middleware` sets `ClearToolUsesEdit(exclude_tools=("retrieve_tutor_context",) if clear_excludes_retrieval else ())`. Reuses the `cleared_tool_outputs` signal → zero telemetry/gate work. The F3 fix in ~5 lines. Run: `--preset clear_retrieval_kb`.

**Worked example 2 — `kb_off` / `disable_kb` (built, verified).** A `ChatRequest.disable_kb` flag threaded through `build_agent` (tool list + cache key), `effective_tool_names`, `build_system_prompt` (drops `KB_TOOL_LINES`, `KB_USAGE_SECTION`, and the `data/kb/AGENTS.md` block), and both call sites. Run: `run_battery --disable-kb` (orthogonal to `--preset`). Known caveat: 3 inert `run_kb_command` citation lines remain in `ANSWERING_RULES` — the tool is gone, so they cannot enable browsing.

**Worked example 3 — per-call-view middlewares (`sliding_window`, `observation_truncation`, `prompt_compression`; built, tested).** Each subclasses `AgentMiddleware`, overrides `wrap_model_call`/`awrap_model_call`, mutates `request.messages` via `request.override(messages=…)`, and calls `record_turn_signal(_turn_id_for(request), <name>, n)`. `_turn_id_for` reads `request.runtime.context.kb_session_id` (= the turn's `message_id`). `SlidingWindowMiddleware` advances its cut to the next user message so a kept window never orphans a tool result. `ObservationTruncationMiddleware`/`PromptCompressionMiddleware` use `message.model_copy(update={"content": …})` and only touch `str` content (multimodal list content is skipped). Selected by `MemoryConfig` flags (`sliding_window_keep`, `truncate_tool_outputs`, `compress_prompt`) in `build_agent_middleware`.

**Worked example 4 — summary-prompt variants (`selective_retention`, `context_reset`; built, tested).** `MemoryConfig.summary_prompt` (a `{messages}`-templated string in `app/memory_presets.py`) is forwarded to `SummarizationMiddleware(summary_prompt=…)` when set. Reuses the existing `summary_messages` checkpoint signal — no telemetry/gate work. `context_reset` also lowers the trigger (15k) and keep (4) for an aggressive reset.

**Run interface.** Presets: `run_battery --preset <name>` (now incl. `observation_truncation`, `sliding_window`, `prompt_compression`, `selective_retention`, `context_reset`, `clear_retrieval_kb`). KB ablation: `--disable-kb`. Retrieval-budget sweep: `--retrieval-budget 30000` (per-request path built; `0` = default 100k).

## Variant catalog (implementation spec)

`shape`: **preset** = `MemoryConfig` flag + middleware; **knob** = config value; **toggle** = tool/flag; **subsystem** = store + tools.

| Variant | Axis | Shape | Wiring | `context_stats` signal | Tests / finding |
|---|---|---|---|---|---|
| `sliding_window` | A | preset | keep last N messages middleware | `dropped_messages` | F9/F10 — recency-only memory |
| `delta_summarization` | A | preset | running summary, new-only | reuse `summary_messages` + `summary_mode` | F2 cache confound |
| `hierarchical_summarization` | A | preset | summarize chunks then summaries | `summary_levels` | long-session compaction |
| `context_reset` | A | preset | fresh state seeded w/ summary | reuse `summary_messages` (summary-prompt variant) | F2; prefix rewrite |
| `prompt_compression` | A | preset | rewrite history fewer tokens | `compressed_messages`, `chars_saved` | F2 |
| `selective_retention` | A | preset | summary prompt keeps constraints/decisions | reuse `summary_messages` | quality-preserving compaction |
| `incontext_history_retrieval` | A | subsystem | index past turns, retrieve relevant | `history_retrievals` | F9 — the principled answer |
| `collection_memory` | A | subsystem | many small fact docs + write/merge tool | `memory_writes/reads` | structured note-taking |
| `entity_memory` | A | subsystem | per student/course/project profiles | `entity_reads` | structured note-taking |
| `sleeptime_consolidation` | A | subsystem + runner hook | background cleanup between sessions | `consolidations` | LangMem-style |
| `observation_truncation` | B | preset | head/tail tool outputs incl. KB | `truncated_tool_outputs`, `chars_saved` | F1 — tokens are in tool outputs |
| `clear_retrieval_kb` | B | preset | extend `ClearToolUsesEdit` to retrieval + KB | `cleared_tool_outputs` (now fires) | F3 — clearing excluded them |
| `retrieval_budget_{100k,30k,10k}` | B | knob | `DEFAULT_CONTEXT_TOKEN_BUDGET` | (token counts already captured) + `retrieval_budget` | F1 — direct |
| `kb_off` | B | toggle | `disable_kb` flag + drop KB prompt block | `kb_enabled` | does the KB improve answers? (89%/9:1 usage) |

Stretch (post-workshop, evals.md): `temporal_graph_memory` (scorecard = `fact_update` probe), `subagent_isolation` (reframed as a tool-output-token play).

## Run matrix (subset screen)

- **Baseline:** `prod`. Every variant is a **single-axis** change vs `prod` (not a cross-product).
- **Subset:** stratified single-turn subset (~24 cases: weighted to `answer_from_corpus`) + the 3 Part-B sessions (s01/s02/s08) + a persona subset, **1-2 trials**.
- **Promote:** screen -> drop losers (most history-compaction arms are predicted to lose on $; confirm the F2/F9 inversion) -> full batteries x 3 trials for winners + combos.
- **Gate per session arm:** `evals.check_triggers` must confirm the variant's mechanism actually fired before probes (the generalized signal), or the memory result is meaningless.

## Cost model (rough)

**NOTE: estimates below were made at the pre-2026-06-13 under-price (~4.4x too low). Corrected per-turn cost is ~$0.25 (Gemini); use that for any new run.**

| Item | Estimate (corrected pricing) |
|---|---|
| Subset screen, 11 arms | **actual: ~$166** (660 turns × ~$0.25; the old "$50-150" was at the buggy price) |
| Judge grading per variant | **$0** — done via subagent workflow on subscription, not the API |
| Full-battery promotion (2-3 winners + combos x 3 trials) | **~$1,300-1,800** (4-6 arms × full single-turn+sessions × 3 trials; the old "$300-400" rigorous figure × the 4.4 pricing fix; replay/personas add more — scope precisely first) |
| Anthropic cache-flip re-run (optional) | small, prefix-rewriters only |

Spend is checkpointed: nothing batch-runs without an explicit go. Runs are **local** (no push), keys (`GEMINI`/`COHERE`/`ANTHROPIC`) are in `.env`.

## Battery v2 — where `full_history` can actually lose

**The principle that scopes v2.** Every memory mechanism reduces to "get the right facts into the context window," and the model usually uses whatever is there (`full_history` ~92–100% recall; `profile_memory` 94% on personas when its store holds the fact). `full_history` is therefore the recall baseline, but not an absolute ceiling: long contexts can still create attention failures, and raw history can contain conflicting facts. A strategy is worth a v2 build only if it wins in one of two places where `full_history` is plausibly weak:

1. **Precision, when history contains a contradiction.** With fact A planted early and A′ later, `full_history` has *both* in context. Does it answer with A′? This can happen at realistic tutor-session lengths, so it is the higher-value product test.
2. **Cost, on genuinely long sessions.** `full_history` carries everything; once history is large enough, even at the prompt-cache discount it can become expensive (F2/F9). This is the principled/workshop answer to F16, but less product-critical unless telemetry shows real sessions often get that long.

If only one new dataset gets built, build the contradiction test first. Long-horizon cost is intellectually clean and useful for the workshop story; contradiction precision is more likely to matter in real tutor use.

**What we want to test — two questions.**

- **Q2 first — precision under contradiction.** When `full_history` holds both A and A′, does it use A′? If yes, no temporal memory is needed (itself a finding). If it *fails*, that is the first crack in the baseline, and a memory that explicitly supersedes A→A′ (`temporal_graph_memory`) could win on correctness. This does **not** require 30–60 turns: it only needs enough turns/tokens for prod to evict A past the kept-recent window before the probe, roughly 15–25 turns to start, then calibrated by `context_stats`.
- **Q1 second — cost at long horizon (the open F16 question).** Once sessions are long enough that `full_history` is genuinely expensive, can `incontext_history_retrieval` (retrieve the relevant old turns) or `delta_summarization` (append-only running summary) **match its recall at lower cost**? `incontext` already works but was short-session-neutral (F16); this is the test that could finally beat the baseline. Report chat cost, latency, and Cohere embedding overhead explicitly, because the current in-context middleware embeds older turn-blocks at run time.

**Why only these two (and what we are NOT building).** The rest of the backlog targets regimes where `full_history` already wins, needs a rarer session shape, or can be tested more cheaply elsewhere:

- `hierarchical_summarization` — needs *extreme* length to beat a single-level summary; v2's first-pass regimes will not get there. **Dropped unless telemetry or a later stress test justifies extreme sessions.**
- `collection_memory` — F22 tested the engaged profile store, not a verbatim atomic-fact store. So `collection_memory` remains a valid cheap v1/product follow-up if we want to test storage quality directly, but it is **not a v2 blocker**: v2's long-cost question is covered by `incontext`/`delta`, and its conflict question is covered by Q2/temporal memory.
- `sleeptime_consolidation` + multi-session — heaviest build (a between-session runner hook + consolidation logic + multi-thread data), and its value over simply *persisting* the store across sessions is marginal. **Deferred.**
- `entity_memory` — **test-first**: author a few multi-project sessions and check whether `full_history` *bleeds* project A's facts into project B's answer. Build the per-project store only if it does (it probably holds two projects straight on its own).

**The battery to add — one file, tagged tiers.** `battery_sessions_v2.jsonl`: not every session has to be long. Keep one file for operational simplicity, but tag the session/probe tiers so Q2 can run cheaply without dragging every contradiction into a very-long context.

- **Tier 1: moderate contradiction sessions (~15–25 turns to start).** Plant A in turn 0, change it to A′ mid-session, probe after A has left prod's kept-recent window. This is Q2, the highest-value first build.
- **Tier 2: long-horizon cost sessions (~30–60+ turns, token-calibrated).** Real corpus questions as filler (reuse `post_id`s/lessons from v1) so the history grows past the point where `full_history`'s cost bites and `incontext`/`delta` have something to drop. This is Q1. They can also carry contradictions, but contradictions do not require this length.
- **Tier 3: multi-project sessions (small, test-first).** One student, project X and project Y; a probe about X fails if it uses Y's constraint.

Same authoring model as v1 — real filler + authored planted facts + a binary `check_note` (we never generate reference answers); model the shape on `data/eval/sessions_generated_*.jsonl`. New `probe_type` values, each graded binary:

- `contradiction` — uses old A = fail (the far-back `fact_update`).
- `longhorizon_recall` — a turn-0 fact probed after many turns and several compactions.
- `entity_isolation` (Tier 3) — uses the wrong project's fact = fail.

(Dropped from the earlier plan: `cross_session_recall` and the separate multi-thread battery — those were for `sleeptime_consolidation`, now deferred.)

**How — build → test → build-only-what-the-data-demands.**

1. **Freeze + data discipline.** v2 is a new file; never touch v1. Gitignored; lives only in the private `ai-tutor-data` HF dataset under `eval/`.
2. **Author Tier 1 first** (moderate contradiction sessions), reusing real corpus filler; model the shape on `data/eval/sessions_generated_*.jsonl` and the schema in `data/eval/README.md`. Add Tier 2 long-horizon sessions only if we decide Q1 is worth the spend/workshop story.
3. **Calibrate by tokens and observed context state, not turn count.** For Q2, confirm prod summarization/compaction fired and A is no longer in the kept-recent window before the contradiction probe; if A is still visible to prod, the probe measures nothing. For Q1, smoke-run a few lengths (for example 30/45/60 turns) and confirm `full_history` cost actually diverges before running a screen.
4. **Wire grading** — add `contradiction` / `longhorizon_recall` / `entity_isolation` rubric entries to `evals/judge.py` (`RUBRICS`) and the human rubric.
5. **Test `full_history` + `prod` + `incontext` + active `profile_memory` first.** This answers the cheap questions and tells us what, if anything, to build: does `full_history` fail contradictions (Q2), does `profile_memory` recover them through the store, and does `incontext` match `full_history`'s recall at lower all-in cost once sessions are actually long (Q1)?
6. **Build only what the data demands**, one at a time, single-axis vs prod (each new mechanism needs its own `context_stats` signal mirrored in `evals/common.py` or the probe gate will not see it fire):
   - `temporal_graph_memory` — **only if** `full_history` fails Q2's contradictions.
   - `delta_summarization` — **only if** Q1 looks promising, as the second cost answer (append-only summary middleware; the F2 counter-hypothesis).
   - `entity_memory` — **only if** `full_history` bleeds on the multi-project probes.
   - `collection_memory` — optional separate v1/product follow-up if we want to isolate verbatim fact storage vs the current 5-line profile write-back.
7. **Screen, then promote.** Start with a tiny Tier-1 screen if only Q2 is in scope; expand to ~8–12 mixed-tier sessions if Q1 is also in scope. Drop losers; promote winners to the full v2 battery × 3 trials, judge-graded — same discipline as the Part C screen.

**Execution note (2026-06-16).** Built a private/gitignored `battery_sessions_v2.jsonl` with 6 sessions / 158 turns: 3 Tier-1 contradiction sessions (22 turns), 2 Tier-2 long-horizon sessions (36 turns), and 1 Tier-3 entity session (20 turns, 2 probes). The full 6-session screen is too slow under the lower-tier Gemini key: `full_history` at concurrency 3 repeatedly hit the Gemini paid-tier input-token/minute limit (3M input tokens/minute), and concurrency 1 works but turns the full 4-arm screen into a multi-hour run. Narrowing is the right move. Current completed local artifacts:

- `runs/e2_v2_prod`: all 6 sessions, 158 turns, 0 errors, all probes under compaction, estimated chat cost **$50.47**. Handgraded partial result: **14% probe accuracy (1/7)**; contradiction **0/3**, long-horizon recall **0/2**, entity isolation **1/2**.
- `runs/e2_v2_full_history`: 2 Tier-1 sessions, 44 turns, 0 errors, no compaction, estimated chat cost **$4.51**. Handgraded partial result: contradiction **2/2**.

Interpretation: enough for a directional signal, not a final finding. The clean next run is **Tier 1 only**: finish the missing `full_history` contradiction session, then run Tier-1 `profile_memory` and optionally Tier-1 `incontext_history_retrieval`. That gives the useful Q2 answer with 3 probes/arm and avoids buying the long-horizon cost story before we know it matters.

## Status / next actions

- **Done + verified (173 tests pass, ruff clean, middleware-assembly smoke ok; adversarially reviewed by a 7-agent workflow, all findings fixed):** orchestration decided; `evals/judge.py` (judge harness); the Phase-0 foundation (turn-signal registry with per-turn-max + oldest-eviction, generalized gate, per-request retrieval budget); and the variant layer:
  - Axis B: `clear_retrieval_kb` (F3 fix), `kb_off`/`disable_kb`, `observation_truncation`, `retrieval_budget` (`--retrieval-budget`).
  - Axis A (preset-shaped): `sliding_window`, `prompt_compression`, `selective_retention`, `context_reset`.
  - Axis A (subsystem): `incontext_history_retrieval` (turn-block retrieval over chat history; offline-tested with a stub embedder; embeds via Cohere at run time).
  - Tests: `tests/test_memory_variants.py` (+ turn-signal tests in `tests/test_telemetry.py`).
- **Deferred / conditional:** `delta_summarization` needs a faithful append-only summary middleware and only becomes worth building after the long-horizon screen shows a real cost regime; `hierarchical_summarization` is dropped unless we later test extreme sessions; `collection_memory` is an optional v1/product follow-up for verbatim fact storage (F22 tested the current profile write-back, not that); `entity_memory` waits for multi-project probes to show bleed; `sleeptime_consolidation` remains deferred because it needs multi-thread data plus a between-session runner hook.
- **SUBSET SCREEN DONE (2026-06-15).** All 11 arms run (~660 turns, 0 errors, ≈$166 at correct pricing), graded by the validated subagent judge, reported to `runs/c_report/report.md`. Findings F14–F24 recorded in `evals.md` (F21–F24 added later: `profile_memory`-on-sessions rerun (D), KB-fair recall regrade, and the `cleared_tool_outputs`-blind-signal correction). Headlines: F9/F10 reproduced (`full_history` cheapest + best memory); `incontext_history_retrieval` 83% but short-session-neutral (needs long sessions); `kb_off` inverts recall@shown 50→96% but priciest + worst memory; `retrieval_budget_30k` matches 100k recall; `observation_truncation` backfires; `context_reset`/`selective_retention`/`sliding_window` dominated. **Workshop-level Part C is complete.** Winners→promotion: `incontext_history_retrieval` (long-session test), `retrieval_budget_30k`, `clear_retrieval_kb`, anchor `full_history`.
- **Beyond workshop (optional, gated — none needed for the workshop):** (1) **promotion** — 2-3 winners (+1-2 combos) × full batteries × 3 trials, hardens the rankings (doesn't answer anything new): **~$1,300-1,800 at corrected pricing**; (2) **Tier-1 contradiction v2** — now the priority path, because the full v2 screen is quota/time-heavy and Tier 1 already shows the strongest signal; (3) **the `incontext` long-session cost test** — principled but less product-critical unless real telemetry shows long sessions are common; (4) conditional builds (`temporal_graph_memory`, `delta_summarization`, `entity_memory`, optional `collection_memory`) only after the relevant baseline failure appears.
- **Omar (open, non-blocking):** decide the `13167988` v2 reword; sign off the review-log audit; push `a564460` + the Part C/cost-fix working-tree changes; greenlight promotion spend / author `_v2` only if going beyond the workshop.
