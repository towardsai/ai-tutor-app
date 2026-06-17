# everything context engineering — annotated with what we tested

> **This is a companion to `everything context engineering.md`.** The original file is unchanged.
> Every original bullet is reproduced verbatim; under each one we add **what we did about it in the
> eval program** (see `evals.md` for the full study). Finding tags like **F9** point to the
> reference list at the bottom of this file, so it is self-contained.
>
> **Legend**
> - ✅ **TESTED** — run as an experiment arm, with the finding.
> - ◐ **PARTIAL / PROXY** — not its own arm, but a nearby mechanism touches it; flagged as analogy, not a clean test.
> - 🔧 **INFRA** — always-on component of the system under test (exercised in every run, not isolated as its own arm).
> - ⛔ **NOT TESTED** — with the reason. Two sub-cases, distinguished in the text: **deferred / conditional** (we *would* test it once a specific baseline failure appears) vs **dropped** (we decided it is not worth testing at all).
> - 📖 **DEFINITION** — a concept, not a testable method.
> - ⚠️ **PLACEHOLDER / DEBUNKED** — a planned demo number that must be replaced or was disproven.
>
> **On "dropped" items:** the four entries marked *dropped at the planning stage* (procedural memory, GraphRAG, parallel research agents, multi-agent systems) all trace to a single line in `evals.md` — *"Dropped (low information given the findings)"*. **None has a dedicated finding;** "the findings" there is collective reasoning, not an F-number. That reasoning is reconstructed under each entry below.

---

## Context Engineering

- **Context engineering:** Designing what the model sees at each step: instructions, history, memory, retrieved docs, tool outputs, and skills.
  - 📖 Definition. The whole eval program *is* context engineering; nothing to isolate.
- **Context window:** The fixed token budget available to the model in one call.
  - 📖 Definition.
- **Context management:** The practical work of keeping the useful context and removing or externalizing the rest.
  - 📖 Definition. This is the experiment's subject; the **memory presets** are the management strategies under test.
- **Context rot:** Long contexts can reduce quality, increase latency/cost, and bury important details.
  - ✅ **Tested, and partly contradicted.** On quality: **F6** — long context costs *latency, not correctness* (zero API errors in 1,018+ turns; max 6.06M tokens across one turn's calls; TTFT scaled 22s→76s from <100k to >800k input tokens/turn). Quality degradation appeared only when we *actively dropped* context via compaction (**F10**), not from length itself.

Useful links:

- [Claude compaction docs](https://platform.claude.com/docs/en/build-with-claude/compaction)
- [OpenAI compaction guide](https://developers.openai.com/api/docs/guides/compaction)
- [Factory: evaluating compression](https://factory.ai/news/evaluating-compression)
- [Intro to context management and compaction](https://youtu.be/PQglg4N_jxo)

---

## Context Compaction

- **Compaction:** Umbrella term for reducing context while preserving what still matters.
  - ✅ **Tested extensively** (this is the core of the study). Headline: under modern prompt caching it often *loses* — **F2** (saves tokens, can cost more dollars), **F9** (full history is cheaper short/medium), **F10** (degrades memory).
- **Trimming:** Drop older messages when they are no longer relevant.
  - ✅ **Tested** via `sliding_window`. See below.
- **Sliding window:** Keep the most recent N turns, sometimes with overlap for continuity.
  - ✅ **TESTED** (`sliding_window`, keep last 12, cut on a user-turn boundary). **F20**: 42% probe accuracy, dominated — it drops the early planted facts the probes test (**F10**). Cheap but memory-lossy.
- **Observation truncation:** Cut huge tool outputs to head/tail or important excerpts before adding them to history.
  - ✅ **TESTED** (`observation_truncation`). **F19**: *backfires* — the agent re-calls tools to recover what was cut (22.8 LLM calls/turn, 430s p95 latency, memory only 33%). Same churn pathology as `aggressive` (**F4**).
- **Tool-result clearing:** Replace old tool outputs with a placeholder while keeping the tool-call record.
  - ✅ **TESTED** (prod's `ClearToolUsesEdit` = `editing_only`; plus `clear_retrieval_kb`). **F3**: prod's clearing *never fires* on this workload because it excludes retrieval results, which is where the tokens are. The `clear_retrieval_kb` fix (clear retrieval+KB too) is a **cheap Axis-B winner** — cheaper than prod, similar memory, better recall/key-points; promoted.
- **Selective retention:** Keep constraints, decisions, open tasks, and state; drop exploration and duplicates.
  - ✅ **TESTED** (`selective_retention`, a tutoring-aware summary prompt told to keep facts/preferences/decisions). **F20**: 25% probe accuracy, still dominated — a better summary prompt does *not* rescue memory, because it still summarizes away the early planted facts.
- **Summarization:** Replace older history with a shorter lossy summary.
  - ✅ **TESTED** (core of `prod` and `aggressive`; Part B + C). **F2** (rewriting the prefix invalidates Gemini's implicit cache → tokens down, dollars up), **F10** (degrades memory, loses *old* facts), **F9** (full history is cheaper anyway ≤13 turns). `summarization_only` was defined but folded into `prod` (since editing never fires per F3, prod ≈ summarization-only here).
- **Hierarchical summarization:** Summarize chunks, then summarize the summaries for long-running sessions.
  - ⛔ **NOT TESTED. Dropped** unless extreme-length sessions matter: it only beats single-level summary at lengths v2's regimes won't reach.
- **Delta summarization:** Update a running summary only with what changed since the last turn.
  - ⛔ **NOT TESTED. Deferred (conditional).** It's the F2 counter-hypothesis (append-only, no prefix rewrite), but only worth building *after* a long-horizon cost tier (**Q1**, defined below — *can a cheaper strategy match `full_history`'s recall at lower cost once sessions get genuinely long?*) shows there's room to beat `full_history`.
- **Context reset:** Start a fresh conversation state seeded with the compacted summary.
  - ✅ **TESTED** (`context_reset`, minimal snapshot + aggressive 15k trigger / keep 4). **F20**: worst of the Axis-A summary arms (17% probe, 56% key-points), dominated on every axis. Prefix rewrite → shares the F2 cache penalty.
- **Prompt compression:** Rewrite prompts/history into fewer tokens while trying to preserve meaning.
  - ✅ **TESTED** (`prompt_compression`, a deterministic model-free stand-in — *not* real LLMLingua). **F15**: reaches 100% memory because it drops no facts, but saves little (highest tokens) — "don't drop content" preserves memory but is **not a cost win**.
- **Offload to files or memory:** Persist details externally and keep only a pointer in context.
  - ✅ **TESTED** in two forms: the `profile_memory` long-term store (offload facts; **F8/F22**) and the KB file-browse `run_kb_command` (always on; **F17**). The "keep only a pointer" idea is what `incontext_history_retrieval` does for history.
- **In-context retrieval:** Pull only the relevant memory/docs back into the prompt when needed.
  - ✅ **TESTED** (`incontext_history_retrieval`, keep 2 recent turn-blocks + retrieve top-3 old ones). **F16**: works (83% probes, best after full_history) but **short-session-neutral** — its cost payoff needs genuinely long sessions (the open **Q1** test, defined below — *can retrieving old turns match `full_history`'s recall at lower all-in cost once sessions get genuinely long?*).

Helpful explainers:

- [Claude Code compaction explained](https://okhlopkov.com/claude-code-compaction-explained/)
- [Context compaction in Codex, Claude Code, and OpenCode](https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode)

---

## Memory

- **Working memory:** Current-session context: recent turns, scratchpad, tool results, intermediate state.
  - 🔧 The substrate every session run tests; the probes *are* working-memory checks. **F10** is the working-memory-loss finding.
- **Semantic memory:** Durable facts about the user, company, project, preferences, goals, or entities.
  - ✅ **TESTED** via `profile_memory` (durable user facts). See profile memory below.
- **Episodic memory:** Past events or examples retrieved as "what happened before."
  - ◐ **PARTIAL / PROXY — by analogy only.** No dedicated episodic store was built; the closest arm is `incontext_history_retrieval`, which RAGs over prior *conversation turns* (F16). That is "retrieve what happened before" in a loose sense, but it was not framed or graded as an episodic-memory test.
- **Procedural memory:** Learned rules, workflows, habits, or skills the agent uses to improve how it works.
  - ⛔ **NOT TESTED. Dropped at the planning stage** (`evals.md` drop list) — *not* tied to any specific finding; no arm was ever run. Why: (1) its only measurable form in this app is loading rules/instructions into the prompt, i.e. the **skills family**, which the 458-token measurement already showed is a ~2% effect (mostly cache-discounted); and (2) **no probe type scores it** — the batteries grade facts and references (`fact_recall`, `preference_compliance`, `anaphora`, `anaphora_consistency`, `fact_update`, `behavior_routing`), and none of them measures "did the agent learn a better workflow over time," so the harness has no instrument to grade it even if we built it.
- **Profile memory:** One compact document representing current known facts about a user/entity.
  - ✅ **TESTED heavily** (`profile_memory`; Part B + the F22 "active" rerun). **F8** (94% personalization, cheapest on personas), **F11** (helps personalization, not working memory), **F21** (it was *dormant* on the sessions battery — the runner passed no `student_id`, so its 38% was just `prod`: a measurement bug), **F22** (once activated it rescued much old-fact loss: 75% vs prod 58%, `fact_recall` 83% vs 33% — but not a cost win on short sessions and doesn't replace full history).
- **Collection memory:** Many small fact documents that can be searched, merged, or updated.
  - ⛔ **NOT TESTED. Deferred (optional v1/product follow-up).** F22 tested the current 5-line profile write-back, *not* verbatim atomic-fact storage; `collection_memory` would isolate that, but it is not a v2 blocker.
- **Entity memory:** Profiles for multiple people, companies, repos, projects, or objects.
  - ⛔ **NOT TESTED. Conditional (test-first).** Build the per-entity store only if a multi-project (Tier-3 v2) probe shows `full_history` *bleeds* project A's facts into project B's answer — it probably holds two projects straight on its own.
- **Memory operations:** Add, update, merge, consolidate, delete, and resolve contradictions.
  - ◐ **PARTIAL / PROXY.** Only *add/update* is exercised, via `profile_memory`'s post-turn write-back (F22); merge/consolidate/delete were not built, and **resolve-contradictions is precisely the open v2 question** (the `contradiction` probe, candidate for `temporal_graph_memory`). Not a standalone arm.
- **Sleep-time consolidation:** Background process that cleans and compresses memories outside the live turn.
  - ⛔ **NOT TESTED. Deferred (heaviest build).** Needs a multi-thread battery + a between-session runner hook, and its value over simply *persisting* the store is marginal.
- **Karpathy-style memory system:** Human-readable project/profile/instruction files that agents read and update as durable context.
  - ✅ **This is the KB** (`data/kb/` wiki — explicitly implements Karpathy's "LLM wiki"). Always on; tested its *presence* via `kb_off`. **F17** (turning it off raises `recall@shown` 50%→96% but is the priciest, worst-memory session arm) + KB-is-dominant-tool (≈89% of Part B turns, ~7.7 KB vs ~0.9 retrieval calls/turn). **But see F23:** that recall jump was a *measurement artifact* — `recall@shown` only counts `retrieve_tutor_context` matches, so prod's KB *browsing* was invisible to it. On a KB-fair metric, prod finds/cites the right source just as well (100% any-tool vs 96%); kb_off's real wins are latency and efficiency, not recall.

---

## Retrieval

- **Keyword search:** Exact or fuzzy text lookup; cheap and strong when terms are known.
  - 🔧 **INFRA, always on** (BM25 leg of hybrid retrieval).
- **Vector DBs:** Embedding-based semantic search for "similar meaning" retrieval.
  - 🔧 **INFRA, always on** (Cohere `embed-v4` over ChromaDB).
- **Hybrid search:** Combines keyword + vector search for better recall.
  - 🔧 **INFRA, always on** (dense + BM25 → Reciprocal Rank Fusion). Measured every run via recall@shown / recall@lesson / MRR, not varied as its own arm.
- **Reranking:** Reorders retrieved chunks by relevance before giving them to the model.
  - 🔧 **INFRA, always on** (Cohere rerank).
- **RAG:** Retrieve external knowledge, then inject it into the model context.
  - ✅ **TESTED** (the `retrieve_tutor_context` tool). **F1** (retrieval payloads dominate input tokens, not chat history) and **F18** (the 100k budget is over-provisioned: `retrieval_budget_30k` matches prod's recall at a third the budget).
- **RAG-as-memory:** Store memories as retrievable documents; simple baseline but weak with changing facts.
  - ✅ **TESTED** via `incontext_history_retrieval` (turns stored as retrievable docs). The "weak with changing facts" caveat is exactly the unresolved v2 contradiction question.
- **GraphRAG:** Uses a knowledge graph to retrieve entities, relationships, and community summaries.
  - ⛔ **NOT TESTED. Dropped at the planning stage** (`evals.md` drop list) — *not* tied to a specific finding. Why: it is a heavyweight retrieval *architecture* (build and maintain a knowledge graph over the corpus), but the study's variable is context *management* (what to do with history and tool outputs), not the base retriever. The retrieval questions that *are* in scope get answered far more cheaply — the `retrieval_budget` sweep (F18), the `kb_off` toggle (F17), and the reranker already in place — so a graph build is out of scope for this experiment and too heavy to justify for the workshop.
- **Temporal graph memory:** Tracks facts over time, useful when facts change or contradict each other.
  - ⛔ **NOT TESTED. Conditional (stretch).** Scorecard = the `fact_update`/`contradiction` probe. Only build it if `full_history` *fails* the v2 contradiction test (**Q2**, defined below — *when history holds both old fact A and updated A′, does the model answer with the current A′?*) — and the partial v2 run so far suggests `full_history` handles contradictions (2/2) while `prod` fails (0/3), so the baseline may not need it.
- **In-context retrieval:** Retrieve only what fits the current task, then place it directly into the prompt.
  - ✅ **TESTED** (`incontext_history_retrieval`; same as the Compaction section). **F16**.

---

## Skills

- **Skills:** Reusable instructions, workflows, tools, or examples loaded only when relevant.
  - ⛔ **NOT TESTED. Dropped; premise debunked.** The planned just-in-time KB-instructions change (talk Change 4) was never built: the KB-instructions block measures **458 tokens** inside a 1,655-token system prompt (not the assumed ~4k), so max saving ~2% of a turn, mostly cache-discounted. Reframed: demo progressive disclosure on tool *outputs* (F1).
- **Progressive disclosure:** Keep skills small and load detailed instructions only when needed.
  - ⛔ **NOT TESTED as designed.** Reframed onto tool outputs (observation_truncation / clear_retrieval_kb territory), where the tokens actually are.
- **Lazy prompt loading:** Avoid putting the whole knowledge base/system manual into every call.
  - ⛔ **NOT TESTED.** Same 458-token reason as Skills.
- **Skill registry:** A table of contents the agent can search to decide which skill to load.
  - ⛔ **NOT TESTED. Dropped** — part of the skills family, same 458-token reason: the loadable instruction surface in this app is too small for a registry / lazy loading to move the numbers.
- **Small focused skills:** Easier to retrieve, update, and compose than one giant skill file.
  - ⛔ **NOT TESTED. Dropped** — part of the skills family (458-token reason).

---

## Tools

- **Tool calls:** Let the model act on external systems: search, files, code, browser, DBs, APIs.
  - ✅ **TESTED** (retrieve + `run_kb_command`). Tool-calls/turn is a core metric and the **re-work signal** — compaction arms re-search for evidence their compressed history lost (**F9/F19**).
- **Memory write tool:** Lets the agent explicitly save durable facts or summaries.
  - ◐ **PARTIAL / PROXY.** `profile_memory` writes durable facts via an *automatic* post-turn write-back (which did run, F22), but the agent never calls an **explicit** write tool. An agent-invoked write/merge tool is the (unbuilt) `collection_memory`.
- **Context editing middleware:** Automatically clears, trims, or rewrites context before the next call.
  - ✅ **TESTED** (`ContextEditingMiddleware` / `ClearToolUsesEdit` = prod). **F3**.
- **Checkpointer:** Stores conversation state inside a session.
  - 🔧 **INFRA, always on** (`InMemorySaver` keyed by `thread_id`).
- **Store:** Persists long-term memory across sessions.
  - 🔧 **INFRA** for the profile-memory family (the long-term store `profile_memory` reads/writes).
- **Token/cost/latency meter:** Makes context growth visible so compaction can be demonstrated.
  - ✅ **BUILT and central.** The `context_stats` telemetry event (`app/telemetry.py`) — Layer-1 telemetry, the measurement spine of *every* finding and the live UI meter.

---

## Agents

- **Single-agent workflow:** One strong agent with good context, tools, skills, and memory.
  - 🔧 This is the tutor; never varied.
- **Sub-agent isolation:** Give a side task to another agent so its exploration does not pollute main context.
  - ⛔ **NOT TESTED. Stretch/post-workshop.** Reframed as a tool-output-token play (keep noisy `run_kb_command` output out of the main context — ties to F1), but not built as an arm.
- **Parallel research agents:** Useful for broad read/research tasks where branches can run independently.
  - ⛔ **NOT TESTED. Dropped at the planning stage** (`evals.md` drop list) — *not* tied to a finding. Why: the study varies *single-agent* context management; parallelism addresses task decomposition, not the memory/cost/latency tradeoffs under test, and the "context-first agents" thesis is to improve context rather than add agents.
- **Multi-agent systems:** Multiple agents coordinating; powerful but costly and harder for shared-state writing tasks.
  - ⛔ **NOT TESTED. Dropped at the planning stage** (`evals.md` drop list) — *not* tied to a finding. Same reason as parallel research agents: out of scope for a single-agent context-management study. The one multi-agent-adjacent idea kept (as a stretch) is **sub-agent isolation**, reframed as a tool-output-token play — keep noisy `run_kb_command` output out of the main context (ties to F1).
- **Context-first agents:** Agents become better less by adding more agents, and more by improving context, memory, tools, and workflows.
  - 📖 A thesis, not an arm — and arguably the whole result *supports* it (the wins came from better context handling, not more agents).

---

## Demo / Case Study

- **AI Tutor repo:** [https://github.com/towardsai/ai-tutor-app](https://github.com/towardsai/ai-tutor-app)
  - 📖 Reference.
- **AI Tutor HF Space:** [https://huggingface.co/spaces/towardsai-tutors/ai-tutor](https://huggingface.co/spaces/towardsai-tutors/ai-tutor)
  - 📖 Reference.
- **Demo idea:** Show failure first, then fix it with compaction, memory, and lazy skills.
  - ✅ Adopted as methodology ("rehearsal = error analysis lite"; pick failure traces for failure→fix→number beats). Caveat: the "fix it with compaction" framing is inverted by F9 — compaction often makes things *worse*, so the demo should show the tradeoff.
- **Compaction demo:** Context grows from ~4k to ~45k tokens, then drops to ~8k after compaction.
  - ⚠️ **PLACEHOLDER numbers**, flagged for replacement with measured ones. The real story is the inversion (F9): full history is often cheaper, so the beat should show compaction's *tradeoff*, not just a token drop.
- **Memory demo:** New session recall goes from 0/5 facts to 5/5 facts with profile memory.
  - ⚠️ **PLACEHOLDER.** Real numbers: personas **94%** personalization (F8); sessions were **dormant** (F21), and the *activated* rerun got **75% vs 58%** (F22) — recovers many facts but not all.
- **Skills demo:** System prompt drops from ~4k tokens to ~800 tokens with lazy loading.
  - ⛔ **DEBUNKED** by the 458-token measurement (the KB-instructions block is 458 tokens inside a 1,655-token system prompt). Cut or reframe this beat.

---

## Findings reference (F1–F22)

The findings from `evals.md`, condensed so this file stands alone. Convention there: entries are never edited, only superseded. **Dollar note:** figures in F2/F8/F9/F12 are *pre-correction* and ~4.4× too low; `MODEL_PRICING` under-priced `gemini-3.5-flash` before 2026-06-13 (correct: $1.50 in / $9.00 out / $0.15 cache-read per MTok). Token counts and relative rankings are unaffected; corrected per-turn cost is ~$0.25 (Gemini).

**Part B (4 presets × 2 trials, 1,232 turns, 0 errors):**

- **F1 — Retrieval payloads dominate input tokens, not conversation history.** Each retrieval call may return up to `DEFAULT_CONTEXT_TOKEN_BUDGET = 100_000` tokens; turns average ~200k input. → retrieval budget became a Part C dimension. (All runs; high confidence.)
- **F2 — Compaction saves tokens but not necessarily dollars.** Summarization rewrites the prompt prefix and invalidates Gemini's implicit cache: `full_history` had 86.8% of input billed at the ~4× cache discount; `aggressive` used 44% fewer tokens yet cost 68% more. (n=24 session-runs; provider-specific — Anthropic caching is explicit.)
- **F3 — Clearing old tool-output messages never fires on this workload.** `ClearToolUsesEdit` excludes retrieval results (where the tokens are); 0 clears across all session runs. → `editing_only` dropped from Part B; "clear retrieval too" queued for Part C (became `clear_retrieval_kb`).
- **F4 — Aggressive compaction degrades even single-turn behavior.** 18.0 vs 9.6 LLM calls/turn, 57s vs 39s median, behavior proxy −10 pts vs `full_history`. Mid-turn compaction churn changes behavior, not just trims. (60 cases × 2 trials.)
- **F5 — Gemini reasoning tokens are ~90%+ of output even with reasoning display off.** Billed as output; dominates latency; recorded per model in `usage_by_model`.
- **F6 — Long context costs latency, not correctness.** Zero API errors in 1,018+ turns at any size (max 6.06M tokens across one turn's calls; largest single context 274k). Median TTFT scales 22s → 76s from <100k to >800k input tokens/turn.
- **F7 — ~0.8% of turns produce no answer text** despite tool calls and billed reasoning tokens; preset- and size-independent. Candidate golden-case assertion ("answer non-empty").
- **F8 — Profile memory wins on quality AND cost.** 94% personalization vs 56–67% without, while the cheapest persona preset ($0.049 vs $0.081–0.095/turn, 7.6 vs 10–11.2 tool calls): the stored profile saves re-searching for user context. (40 questions × 2 trials × 4 presets, auto checks.)
- **F9 — Full history is cheapest AND fastest up to 13 turns; compaction causes re-work.** Sessions: $0.034/turn (pre-correction), TTFT 17s, 2.8 tool calls for `full_history` vs $0.051–0.066, 21–43s, 3.5–8.3 elsewhere — despite ~2× the tokens (F2's cache plus raw history letting the agent re-use earlier retrieval evidence; summaries force re-retrieval). The conventional pitch inverts under modern caching. (n=24, 0 errors.)
- **F10 — Compaction degrades within-session memory, and what it drops is *old* facts.** Session-probe accuracy: `full_history` **92%** vs `prod`/`profile_memory` 38% / `aggressive` 42%. The collapse is entirely turn-0 material — `fact_recall` 100% → 17–25%, `preference_compliance` 100% → 0% — while `fact_update` stays **100%** (the recent update sits in the kept-recent window). (n=24/preset.)
- **F11 — Profile memory helps personalization, not working memory.** 92% persona personalization yet **38% on session probes — identical to `prod`**. (Refined by F21: profile was dormant on sessions.)
- **F12 — Aggressive compaction is dominated on every axis.** key-point coverage 36% vs 72% (`full_history`), single-turn behavior 67% vs 83–92%, session probes 42% vs 92%. No metric favors it.
- **F13 — Session-probe grades are human-confirmed (zero overrides).** 2026-06-15 Omar reviewed all 96 session probes (83 high-confidence LLM verdicts by sign-off + 13 low-confidence individually), agreeing with every grade → F10–F12 memory numbers are no longer provisional. (Non-probe quality rows remain LLM-graded.)

**Part C (screen: 11 arms, ~660 turns, 0 errors, ≈$166; judge-graded; probe n=12/arm — coarse rankings):**

- **F14 — The LLM judge is validated; Part C quality columns are reportable.** A blind subagent re-grade of the 96 human-confirmed probes reproduced them at **98% agreement / TPR 100% / TNR 96%** (clears the >90% gate; measures grader *reproducibility*, since labels were sign-offs). Validated grader = the subagent workflow, run on subscription at no API cost.
- **F15 — The screen reproduces F9/F10 on independent arms.** `full_history` cheapest on sessions ($0.10/turn) AND best memory (100% probe); every compaction arm pricier and weaker. `prompt_compression` also hits 100% memory (drops no facts) yet saves little (highest tokens) — "don't drop content" preserves memory but isn't a cost win.
- **F16 — `incontext_history_retrieval` works but is short-session-neutral.** 83% probe accuracy (best after `full_history`): retrieving relevant old turns restores what summarization loses. But on ≤13-turn sessions it can't drop much, so it costs ≈ `full_history` (~$0.20/turn) plus embed overhead; its cost payoff needs LONG sessions (a `_v2` battery).
- **F17 — Turning the KB off inverts retrieval recall (Axis B tradeoff).** `kb_off` forces `retrieve_tutor_context` → single-turn recall@shown jumps 50%→96%, but it is the priciest session arm ($0.31/turn) and worst memory (33%). KB browsing is cheaper and better for memory yet *lowers* top-k recall. **(Recall reading superseded by F23 — see below.)**
- **F18 — The 100k retrieval budget is over-provisioned.** `retrieval_budget_30k` matches prod's recall@shown (50%) at a third of the budget → tokens cut with no recall loss on this subset (direct confirmation of F1). The 10k rung is untested.
- **F19 — `observation_truncation` backfires (Axis B negative).** Head/tail-truncating tool outputs makes the agent re-call tools to recover what was cut: 22.8 LLM calls/turn, 430s p95 latency (single-turn), memory only 33% — the `aggressive` churn pathology (F4).
- **F20 — Axis-A summary/trim arms do not rescue memory.** `context_reset` (17% probe, 56% key-points — dominated everywhere), `selective_retention` (25%), and `sliding_window` (42%) all stay well below `full_history`/`prompt_compression` and none beats `prod` (58%): dropping/summarizing early turns loses the planted facts (consistent with F10).
- **F21 (2026-06-16) — Supersedes F11 in part: `profile_memory` was dormant on the sessions battery, so its 38% is `prod`, not a test of the profile.** `run_session` passes no `student_id`, and both profile injection and write-back no-op without one → on sessions `profile_memory` reduces exactly to `prod`. The 38% is real but comes from the live thread + prod's lossy summary, not from any profile. F11's "independent subsystems" reading is unsupported (the store was never engaged in-session); F11's personas number (94%) stands (personas do set a `student_id`).
- **F22 (2026-06-16) — Activating `profile_memory` on sessions rescues much of the old-fact loss, but does not replace full history.** Reran the 3 Part-C sessions with a per-session `student_id` so the store was read/written/injected every turn. With compaction active at 100% of probes, probe accuracy rose to **75% (9/12)** vs same-screen `prod` **58% (7/12)**; `full_history` stayed **100%**. Concentrated where F10/F21 predicted: `fact_recall` **83% (5/6)** vs `prod` 33% (2/6); `preference_compliance` 100% (1/1) vs 0%. Not a cost win on this short screen ($0.265/turn vs `prod` $0.229, `full_history` $0.103). Thin-n caveat: robust signal is `fact_recall` + the 9/12-vs-7/12 aggregate; it regressed on anaphora (50% vs 100%, n=2).

- **F23 (2026-06-17) — F17's recall inversion was a measurement artifact.** `recall@shown` counts only `retrieve_tutor_context` matches, so prod's KB *browsing* was invisible to it. Re-grading the same Part C bundles (free, no re-run; `runs/kbfair_report/`) with KB-fair metrics: **recall source any-tool (retrieval+KB)** = prod **100%** vs kb_off 96% (n=24); **cited-correct source/lesson** (answer cites the labeled doc, resolved from both tools) = **100%/100% both** (n=14). Verified real: in all 12 prod retrieval-misses, the agent had browsed that exact gold source. So KB-off does **not** improve grounding — its real wins are latency (19s vs 38s p50) and efficiency (3.6 vs 10.1 LLM calls/turn). Supersedes F17's recall reading; F17's cost/memory points stand. New auto metrics: `recall_anytool_source`, `cited_correct_source/lesson` in `evals/grade.py`.

**Screen winners → promotion candidates:** `incontext_history_retrieval` (needs the long-session test), `retrieval_budget_30k` and `clear_retrieval_kb` (cheap Axis-B wins), anchored by `full_history`. **Drop:** `context_reset`, `observation_truncation`, `selective_retention`.

---

## Open v2 questions (Q1 / Q2)

`full_history` won Part B and the Part C screen, so the unfinished v2 work (`battery_sessions_v2.jsonl`) exists to find the **two regimes where the baseline might actually lose**. Both are scoped in `evals_part_c_plan.md` → "Battery v2." Heads-up on the labels: they read "Q1 / Q2" but **Q2 is the higher priority and is built first** — the numbering is from the doc, not the order of work. Every annotation above that says "Q1" or "the v2 contradiction test" points here.

- **Q2 — precision under contradiction (built first, highest-value, more product-critical).** Plant fact A early, update it to A′ mid-session, then probe *after* prod's compaction has evicted A from the kept-recent window. The twist that makes this the one place `full_history` is genuinely at risk: `full_history` sees **both** A and A′ and must pick the current one. If it answers A′, no temporal memory is needed (a finding in itself); if it answers A, that's the first crack in the baseline, and `temporal_graph_memory` (explicit A→A′ supersession) could win on *correctness*. Needs only ~15–25 turns (enough for prod to evict A), not 30–60, calibrated by `context_stats`. → partial v2 run so far: `full_history` contradiction **2/2**, `prod` **0/3** (directional, not final).

- **Q1 — cost at long horizon (the open F16 question, less product-critical).** Once sessions are long enough that `full_history` is genuinely expensive *even at the cache discount* (F2/F9), can `incontext_history_retrieval` (retrieve only the relevant old turns) or `delta_summarization` (append-only running summary, no prefix rewrite — the F2 counter-hypothesis) **match its recall at lower all-in cost**? `incontext` already works but was short-session-neutral on ≤13-turn sessions (F16); this is the test that could finally beat the baseline *on cost*. Requires a token-calibrated long-session tier (~30–60+ turns) and must report chat cost, latency, AND Cohere embedding overhead explicitly (the in-context middleware embeds older turn-blocks at run time). Gated on whether real tutor telemetry shows sessions actually get that long.

**Which strategies each question would justify building:** Q2 failing → `temporal_graph_memory`. Q1 looking promising → `delta_summarization` (and confirms `incontext_history_retrieval`'s cost case). The plan's first v2 run tests `full_history` + `prod` + `incontext` + *active* `profile_memory` together, since that cheaply answers both questions and tells us what, if anything, to build next.
