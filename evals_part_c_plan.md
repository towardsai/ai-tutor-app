# Part C — execution plan

The **how** for Part C. The **what** (scope, the two axes, the variant catalog, the findings that motivate them) lives in `evals.md` under "Part C — widen the comparison". This doc is the build/run/grade plan: orchestration, wiring, schedule, cost, and status. Read `evals.md` first.

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

### Phase 1 — Judge validation (needs Omar's labels) [GATE]
- [ ] Omar grades the calibration set (`runs/b_report/review_session_probes.md` + `review_other.md`, folded via the workbook).
- [ ] `evals.judge run` over the same rows -> `evals.judge validate` -> ship only at >=90% TP/TN. If short, refine the rubric or fall back to per-variant hand-grading.

### Phase 2 — Axis B: retrieval & tool outputs (cheap, highest-info, mostly auto-graded)
Variants: `retrieval_budget_100k/30k/10k`, `clear_retrieval_kb`, `observation_truncation`, `kb_off`. Each: implement + signal, screen vs `prod`. Headline metrics (recall@shown, cost, latency, tokens, tool-calls) are auto — results land even before the judge.

### Phase 3 — Axis A: memory & context management (preset-shaped)
Variants: `sliding_window`, `delta_summarization`, `hierarchical_summarization`, `context_reset`, `prompt_compression`, `selective_retention`. Report **$ and tokens both** (F2 cache confound); watch the re-work signal (tool-calls/turn).

### Phase 4 — Axis A subsystems (heavy builds, worktree-isolated workflow)
Variants: `incontext_history_retrieval`, `collection_memory`, `entity_memory`, `sleeptime_consolidation`. Each = a store + tools + retrieval/consolidation path; sleep-time also needs a between-session runner hook in `run_battery`. Some need **battery v2** (below).

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

## Battery v2 (for the subsystem/temporal probes)

New probe types (contradiction resolution, temporal facts, longer-horizon recall) need a `_v2` battery — `v1` is frozen and **never edited in place** (`data/eval/README.md` rule). v2 is its own data sub-project: author cases, re-run assembly, bump the version. **Eval data never enters git** (real student text; `main` force-pushes to the public prod Space) — share via the private HF dataset only.

## Status / next actions

- **Done + verified (173 tests pass, ruff clean, middleware-assembly smoke ok; adversarially reviewed by a 7-agent workflow, all findings fixed):** orchestration decided; `evals/judge.py` (judge harness); the Phase-0 foundation (turn-signal registry with per-turn-max + oldest-eviction, generalized gate, per-request retrieval budget); and the variant layer:
  - Axis B: `clear_retrieval_kb` (F3 fix), `kb_off`/`disable_kb`, `observation_truncation`, `retrieval_budget` (`--retrieval-budget`).
  - Axis A (preset-shaped): `sliding_window`, `prompt_compression`, `selective_retention`, `context_reset`.
  - Axis A (subsystem): `incontext_history_retrieval` (turn-block retrieval over chat history; offline-tested with a stub embedder; embeds via Cohere at run time).
  - Tests: `tests/test_memory_variants.py` (+ turn-signal tests in `tests/test_telemetry.py`).
- **Deferred (need custom state-rewriting machinery or v2 data):** `delta_summarization` and `hierarchical_summarization` (faithful versions need append-only / multi-level summary middleware, not just a prompt swap — the F2 counter-hypothesis is delta's whole point, so don't ship a mislabeled re-summarization); the 3 remaining Axis-A **subsystems** (`collection_memory`, `entity_memory`, `sleeptime_consolidation`), which want `_v2` probe types (contradiction / multi-entity / multi-session-consolidation) plus, for sleep-time, a between-session `run_battery` hook — Omar's data-authoring task. (Per the critic: `entity_memory` collapses to the existing `profile_memory` preset on v1, and `collection_memory` has no v1 probe that rewards explicit fact storage.)
- **SUBSET SCREEN DONE (2026-06-15).** All 11 arms run (~660 turns, 0 errors, ≈$166 at correct pricing), graded by the validated subagent judge, reported to `runs/c_report/report.md`. Findings F14–F20 recorded in `evals.md`. Headlines: F9/F10 reproduced (`full_history` cheapest + best memory); `incontext_history_retrieval` 83% but short-session-neutral (needs long sessions); `kb_off` inverts recall@shown 50→96% but priciest + worst memory; `retrieval_budget_30k` matches 100k recall; `observation_truncation` backfires; `context_reset`/`selective_retention`/`sliding_window` dominated. **Workshop-level Part C is complete.** Winners→promotion: `incontext_history_retrieval` (long-session test), `retrieval_budget_30k`, `clear_retrieval_kb`, anchor `full_history`.
- **Beyond workshop (optional, gated — none needed for the workshop):** (1) **promotion** — 2-3 winners (+1-2 combos) × full batteries × 3 trials, hardens the rankings (doesn't answer anything new): **~$1,300-1,800 at corrected pricing**; (2) the **`incontext` long-session test** — the one open question (does retrieve-old-turns beat full_history once sessions get long?), needs a `_v2` long-session battery; (3) the **deferred builds** (delta/hierarchical + collection/entity/sleeptime) — also need `_v2` data. All three need Omar's spend or data.
- **Omar (open, non-blocking):** decide the `13167988` v2 reword; sign off the review-log audit; push `a564460` + the Part C/cost-fix working-tree changes; greenlight promotion spend / author `_v2` only if going beyond the workshop.
