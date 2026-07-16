# Evals background: research notes & rationale

Reference material behind `evals.md`: source attributions, metric design rationale, methodology, pitfalls.

Goal: build an eval harness that lets us say, with evidence, "memory/context method A got these results, method B regressed quality but cut cost 40%, method C improved both" — across latency, tokens, cost, retrieval accuracy, and answer quality — and turn that into a workshop on comparing these methods.

Sources read: Hamel Husain's evals FAQ + related posts, howtoeval.com (Ben Hylak), and the OpenAI cookbook "Macro Evals for Agentic Systems". Codebase and `data/academy_discussion_eval.jsonl` inspected directly.

---

## 1. The single most important framing: you are doing TWO different things

Everything in the sources clicks into place once you separate two activities that look similar but have different methodologies:

1. **Product evals (quality assurance).** "Is our tutor good? Where does it fail?" This is Hamel's world: error analysis → failure taxonomy → targeted binary checks and validated LLM judges. The output is a list of failure modes with frequencies, and automated checks that catch regressions.

2. **Controlled experiments (benchmarking variants).** "Does summarization middleware beat full history?" This is classic experiment design: freeze everything (model, prompt, dataset, retrieval), vary exactly one component (the memory/context strategy), run the **identical** scenario battery through each variant, compare metrics with paired statistics. This is what the workshop is actually about, and it's closest to the OpenAI cookbook's approach.

**Activity 2 depends on activity 1.** You can't compare variants until you have (a) a trustworthy dataset, (b) metrics you believe, and (c) a judge validated against human labels. And when variant B scores worse, only error analysis tells you *why* (the macro-evals point: a regression number without failure-pattern analysis is a dead end). Hamel gives you the machinery for (a)–(c); the cookbook gives you the cross-population analysis; howtoeval gives you the harness philosophy (assert on trajectories, golden cases gate shipping, small high-signal datasets).

This two-activity distinction is also a great workshop narrative: *most teams try to jump straight to activity 2 and get numbers they can't trust.*

---

## 2. What the three sources say (distilled)

### Hamel Husain (hamel.dev/blog/posts/evals-faq + evals + llm-judge + field-guide)
- **Error analysis first, always.** Read 100+ real traces, take free-form notes ("open coding"), group into a failure taxonomy ("axial coding"), count frequencies. Stop when ~20 consecutive traces yield no new failure category ("theoretical saturation"). Re-run on 100+ fresh traces each cycle. This is "the single highest-ROI activity"; expect 60–80% of dev time on evals overall.
- **Binary pass/fail + written critique, never 1–5 Likert scores.** Decompose nuance into multiple binary checks (e.g., each expected fact = one binary check). One domain-expert "benevolent dictator" labels; don't outsource, don't committee.
- **LLM-as-judge via "critique shadowing":** expert labels ~30 diverse examples binary + critique → few-shot judge prompt built from the critiques → iterate until >90% agreement with expert on a held-out set → measure **TPR/TNR**, not raw agreement (classes are imbalanced). Use the strongest model you can afford as judge.
- **Generic metric dashboards (hallucination score, helpfulness 1–5) are an anti-pattern** — "waste time and create false confidence." Build checks for failures you observed, not failures you imagine. Don't do "eval-driven development."
- **RAG:** evaluate retrieval and generation **separately**. Retrieval = classic IR metrics (Recall@k, Precision@k, MRR) against query→document ground truth. Generation = error analysis + targeted checks (faithfulness to context, answers the question).
- **Multi-turn:** start with one conversation-level pass/fail ("did the conversation meet the user's goal?"); annotate only the **first** failure (downstream failures cascade). For test generation, **N-1 replay** (feed real turns 1..N-1, evaluate the model's turn N) beats fully simulated users.
- **Agents:** two phases — (1) end-to-end black-box success rate, (2) step-level diagnostics: tool choice, argument quality, error recovery, context retention across steps, efficiency (steps, seconds, tokens). Build **transition failure matrices** (last good step × first failed step) and compare them **across experiments** — directly applicable to our variant comparison.
- **CI:** small purpose-built suites (often ~100 examples), prefer cheap code assertions over judges; 100% pass means your evals are too easy (~70% is healthier).

### howtoeval.com (Ben Hylak, Raindrop — one long guide, May 2026)
- Frame: **floor-raising** (eliminate worst-case failures on critical paths) over benchmark-maxxing. For agents, "the path matters as much as the answer" — offline evals should be **code-aware tests** that run the real agent and assert on outputs *and* tool-call sequences (pytest/vitest style), not prompt-scoring.
- **Golden cases:** start with 5–10 critical-path scenarios; "if your agent starts failing your golden cases, you do not ship." Datasets come from real production failures; "20 high-signal cases beats 200 low-signal ones"; prune cases that haven't failed in 3 months.
- **Skeptical of LLM-as-judge** (calibration brittleness, Goodhart's law) and of hosted eval dashboards; weights production monitoring + A/B on real traffic for unverifiable hypotheses. Useful tension to present in the workshop against Hamel's pro-judge stance — the resolution is Hamel's validation discipline: a judge is only as good as its measured agreement with a human expert.
- Budget ~10–20% of dev time (vs Hamel's 60–80% — present both, the truth depends on maturity/stakes).

### OpenAI cookbook — "Macro Evals for Agentic Systems" (May 2026, w/ Promptfoo)
- **Micro evals** grade one run against a rubric; **macro evals** look *across* hundreds of graded runs to find which failure patterns repeat, where they concentrate, and what to inspect first. Rationale: "a final answer is only the last event in a longer workflow" — final-answer inspection alone misses upstream failures.
- Four labels per run: **case_type** (the scenario setup) → **run_outcome** (how it ended) → **eval_finding** (which rubric check failed) → **behavior_pattern** (cluster discovered across the population). Slice analysis via lift = pattern share in slice ÷ overall share.
- **Trace bundles**: persist a complete evidence packet per run (events, tool calls, outcomes, judge labels) so you can **re-grade offline without re-running the agent**. "The quality of the trace document is part of the evaluation design."
- Tracks per-trace counters (tool calls, handoffs, loops, retries) as severity/complexity inputs. Notably it does **not** cover latency/token/cost — none of the three sources do. We have to define that layer ourselves (it's the easy, objective part).

---

## 3. What we should measure (the metric stack)

Five layers, cheapest/most-objective first. Layers 1–3 are code; only layer 4 needs a judge; layer 5 is the one that actually differentiates memory methods.

### Layer 1 — Operational telemetry (free, fully objective)
Per turn and cumulative per conversation:
- **Latency**: time-to-first-token and total wall-clock per turn (we stream, so TTFT is what users feel).
- **Tokens**: input/output per LLM call, summed per turn and per conversation. Input tokens are where memory methods differ most — full history grows O(n²) cumulative input tokens over a conversation; summarization/editing flattens that curve. *The cumulative-input-token curve per conversation turn is the workshop's signature plot.*
- **Cost**: tokens × provider price sheet (account for cache-read pricing if applicable; summarization itself costs extra LLM calls — count them, that's part of the honest comparison).
- **Trajectory counters**: number of LLM calls, tool calls per turn (retrieve vs kb_command vs web), KB-command budget consumption, summarization/context-edit trigger firings.

LangSmith already captures token usage and latency per run (runs are tagged with model + metadata; add a `memory_variant` tag). The `stream_chat` ChatEvent stream gives tool-call-level detail.

### Layer 2 — Retrieval accuracy (objective, we have ground truth)
- The dataset has `source_key` and `lesson_url` per question → **Recall@k / MRR**: did the right source/lesson appear in `retrieve_tutor_context` results? (Hook into the SearchResult list on `tool_call_completed` events; results carry source + URL.)
- **Tool-routing correctness**: when `expected_behavior == answer_from_corpus`, did the agent call retrieval/KB at all? When `redirect_to_support`, did it correctly *not* go down a retrieval rabbit hole?
- Why this matters for memory comparisons: compressed/summarized context can degrade the agent's *query formulation* on later turns (it lost the details needed to write a good retrieval query). Retrieval recall on late turns is a sensitive early indicator of memory damage.

### Layer 3 — Behavioral correctness (programmatic binary checks)
- **Expected-behavior match**: dataset annotates `expected_behavior` ∈ {answer_from_corpus, answer_general, redirect_to_support, acknowledge_feedback}. A small classifier-judge or keyword check per behavior → binary pass.
- **Citation validity**: we already resolve inline citations against current-turn evidence + KB manifest (`app/kb_manifest.py`) — assert citations resolve, and that corpus answers carry ≥1 citation.
- **Format/safety assertions**: answered in scope, didn't fabricate course logistics, didn't promise human follow-up, etc. (grow this list from error analysis, not imagination).

### Layer 4 — Answer quality (LLM-as-judge, validated)
- **Key-point coverage** (primary): annotations include 1–4 `key_points` extracted from real staff answers. For each key point, one binary judge call: "Does the answer correctly convey this point? pass/fail + one-line critique." Report coverage = fraction of key points passed. This is exactly Hamel's "decompose gradations into separate binary checks" and is far more reliable than holistic scoring.
- **Faithfulness**: binary — is every substantive claim in the answer supported by the retrieved evidence / KB content in the trace? (Catches hallucination specifically on corpus questions.)
- **Holistic pass/fail** (secondary): "Would the course staff member have approved sending this answer? pass/fail + critique" — few-shot prompt built from *our own labeled critiques*, validated to >90% TPR/TNR agreement against ~30–50 human-labeled traces before we trust it. The team member who actually answers academy questions is the "benevolent dictator" labeler.

### Layer 5 — Memory-specific probes (the differentiator — this is OUR design; no source covers it)
Memory methods only differ when context pressure exists. Single-turn evals will show **zero difference** between variants (our SummarizationMiddleware triggers at 30k tokens, ContextEditing at 5k tool-tokens — a one-shot question never trips them). So:
- **Recall-after-compression probes**: plant a fact early (turn 1–2: "I'm on the Agentic AI course, lesson 4, using Python 3.13 on Windows, my API key is set via .env"), drive 5–10 heavy turns (each invoking retrieval/KB to inflate tool tokens past the triggers), then probe: "given my setup, why might X fail?" Binary: does the answer use the planted facts?
- **Consistency**: does turn N contradict what the tutor said in turn 2? (Judge check on conversation pairs.)
- **No re-asking**: does the agent ask for information the user already provided? (Strong signal of memory loss; cheap judge check.)
- **Instruction persistence**: user says "explain everything assuming I'm a beginner / always show code in Python" at turn 1 — is turn 8 still complying?
- **Anaphora resolution under compression**: late-turn question that says "the second approach you mentioned" — resolvable only if earlier assistant content survived.

Score each probe binary, report **memory probe accuracy** per variant. This is the column where full-history wins and aggressive compression loses — the interesting result is *how much* quality each method trades for its token savings.

---

## 4. Agents vs one-turn chatbots — and tutors vs coding agents

How agent evals differ from single-turn chatbot evals (all three sources agree):
1. **The trajectory is a first-class eval target.** A correct answer reached via 14 redundant KB commands is a different (worse) result than the same answer in 2 tool calls. Assert on tool-call sequences and counts, not just final text.
2. **Failures cascade.** Annotate the *first* failure per trace; build transition matrices (where in the pipeline do failures start — query formulation? retrieval? synthesis? citation?). Fixing late-stage symptoms of early-stage failures is wasted work.
3. **Final-answer inspection alone is insufficient** — an answer can look fine while the trace shows the agent ignored retrieved evidence and answered from parametric knowledge (a faithfulness failure that will bite on corpus-specific content).
4. **Two-phase evaluation**: end-to-end success rate first (cheap, comparable), step-level diagnostics second (explains the deltas).
5. **Multi-turn adds a conversation-level unit of analysis**: "did the session meet the student's goal" is judged over the whole conversation, with per-turn metrics underneath.

How a **tutor** differs from a **coding agent** (your instinct is right):
- Coding agents have **executable ground truth** (tests pass / build compiles). We don't — there is no compiler for "good pedagogical answer." That pushes us toward (a) human-aligned LLM judges and (b) squeezing every drop out of the *programmatic* ground truth we *do* have: retrieval ground truth (`source_key`/`lesson_url`), citation resolvability, expected-behavior routing, and key-point coverage against real staff answers. We're actually unusually well-positioned: most chatbot teams have no reference answers; we have 135 real staff answers.
- Tutor-specific quality dimensions worth tracking once error analysis confirms they occur: grounding in *our* course material vs generic internet answers (a student asks about lesson 4's notebook; a generically-correct-but-course-ignorant answer is a failure), level-appropriateness, and scope discipline (redirect platform issues instead of hallucinating refund policies).
- The agentic surface we *do* share with coding agents: tool selection (retrieve vs browse KB vs web), query formulation quality, and budget efficiency — evaluate those the same way coding-agent evals do (trajectory assertions + efficiency counters).

---

## 5. The dataset: what we have, and the gap we must fill

`data/academy_discussion_eval.jsonl` — 151 real student questions, LLM-annotated (gemini-3.5-flash, 2026-06-10):

| Dimension | Distribution |
|---|---|
| eval_quality | gold 62 · usable 61 · weak 9 · exclude 19 |
| category | debugging 47 · course_feedback 39 · conceptual 31 · platform_issue 20 · other 14 |
| expected_behavior | answer_general 49 · acknowledge_feedback 41 · answer_from_corpus 40 · redirect_to_support 21 |
| has reference_answer | 135 yes · 16 no |
| key_points | 91 cases have ≥1 (60 have none) |
| time_bound | **52 true** · 99 false |
| source | full_stack 98 · agentic 37 · python 15 · none 1 |

**Strengths**: real user distribution (not synthetic), real staff reference answers, behavior labels that map directly to binary checks, **81 cases that are gold/usable + key_points + reference answer** — that's the core single-turn battery, right in Hamel's "~100-example purpose-built suite" range.

**Caveats**:
- **Verify the annotations.** They're LLM-generated and unreviewed. Before trusting them as ground truth, the domain expert should review at least the 62 gold cases (Hamel: never delegate ground-truth labels). This doubles as your first error-analysis session. Expect to demote some.
- **52 time_bound cases**: answers reference point-in-time state (broken links, current platform behavior). Exclude from the comparison battery or re-verify; they'll produce noise, not signal.
- **Skew**: 65% from one course; fine for now, note it when reporting.
- **It's almost entirely single-turn** (82 threads have 1 reply, 14 have none; the few longer threads are discussion back-and-forth, not chat sessions). **This dataset alone cannot differentiate memory methods** (§3 layer 5). 

**The gap-filler: a synthetic multi-turn "study session" suite.** Following Hamel's synthetic-inputs guidance (generate *inputs* only; run them through the real system):
1. Take related gold cases from the same course/lesson; chain 4–8 of them into a plausible study session script (student persona working through a lesson, hitting issues).
2. Insert planted facts early (persona, environment, constraints) and 1–3 **memory probes** late (§3 layer 5), positioned *after* enough tool-heavy turns that the 5k/30k triggers have demonstrably fired (log trigger events to verify — a memory eval where compression never activated measures nothing).
3. Hand-write ~5 session scripts first (Hamel: hand-write ~20 seed tuples before LLM-generating more), then LLM-generate variants along dimensions: course × persona (beginner/advanced) × session length × probe type. Target ~30–50 sessions.
4. Also use **N-1 replay** on the 9 real threads with ≥3 turns — replay real turns, evaluate the next one. Small but real.

---

## 6. Error analysis and LLM-as-judge, in plain terms (and exactly what to do)

**Error analysis** = systematically reading your system's transcripts and writing down what went wrong, then grouping the notes into a counted taxonomy. No math, no models — structured journaling. It is unanimously (all three sources) the highest-ROI step and the one everyone skips. Concretely for us:
1. Run the current system (default config) over the ~80-case single-turn battery. Persist full trace bundles (every ChatEvent: tool calls, retrieval results, reasoning, final answer) as JSONL — the cookbook's "grade offline, re-grade cheaply" pattern.
2. Omar (or whoever answers academy questions) reads every trace next to the staff reference answer, marks binary good/bad, writes a one-to-three-sentence note on the *first* thing that went wrong.
3. Group notes into a failure taxonomy (LLM can help cluster; human validates) and count. Expect things like: retrieval missed the lesson → answered generically; ignored retrieved evidence; over-eager web search; wrong behavior routing (answered a platform issue instead of redirecting); citation errors.
4. **Those observed failure modes — not a generic metric list — become the automated checks in layers 3–4.** Hamel's strongest warning is against bolting on prefab "hallucination/helpfulness" scores; build checks for what actually breaks.
5. A tiny annotation viewer (Streamlit, one screen: trace + reference answer + pass/fail button + notes box) is "the single most impactful investment" — buildable in an afternoon, ~10x labeling speed.

**LLM-as-judge** = using a strong LLM to grade outputs at a scale humans can't. The catch: an unvalidated judge is a random-ish number generator with confident prose. The discipline (Hamel's critique-shadowing): human-label 30–50 diverse traces binary+critique → build the judge prompt with those critiques as few-shot examples → run judge on held-out labeled traces → measure TPR/TNR → iterate the prompt until agreement >90% → only then run it at scale, and re-spot-check periodically. Our `key_points` decomposition makes the judge's job nearly mechanical ("does the answer say X?"), which is exactly where judges are most reliable. Hylak's skepticism (Goodhart, calibration drift) is the reason for the validation step, not a reason to skip judges — we have no executable ground truth, so a validated judge is the only scalable quality measure available.

---

## 7. The controlled experiment: "method A vs B vs C"

### Variants (all parameterizable from our codebase — see §8 for the needed refactor)
| Variant | Description |
|---|---|
| **A. Full history** (baseline) | No summarization, no context editing. Upper bound on quality/memory, worst tokens/cost. |
| **B. Current prod** | ContextEditing(trigger 5k tool-tokens, keep 5) + Summarization(trigger 30k, keep last 20 msgs). |
| **C. Summarization only** | Isolate the summarizer's contribution. |
| **D. Context-editing only** | Isolate tool-result pruning. |
| **E. Aggressive compression** | e.g. Summarization @ 8k keep 8, editing @ 2k keep 2. The "how bad can cheap get" point. |
| **F+. SOTA methods (workshop highlights)** | One or more of: **long-term memory store** (LangMem/LangGraph Store: extract salient facts per turn into a store, retrieve-into-context on later turns — MemGPT/Letta lineage); **retrieval-over-history** (embed past turns, RAG over your own conversation instead of carrying it); **structured note-taking / compaction** (agent maintains a running scratchpad of session state, à la Anthropic's context-management work and Claude Code's compaction); **observation offloading** (truncate tool outputs to references, re-fetch on demand). |

### Protocol
- **Hold constant**: model (run the full matrix on one model first; model×memory interaction is a separate, second experiment), system prompt, retrieval config, dataset, tool config, temperature.
- **Vary**: the memory/context configuration only. Tag every run/trace with `memory_variant` (LangSmith metadata + trace bundle field).
- **Repeat trials**: LLM nondeterminism is real. 3 trials per case per variant minimum; report mean ± and also **consistency** (pass^3: passed all 3 trials — workshops love seeing that a method is not just better on average but more *reliable*).
- **Paired comparison**: every variant sees the identical cases, so compare per-case (paired bootstrap for rates, McNemar for binary). With ~80 single-turn cases + ~40 sessions × 3 trials, ~10-point pass-rate differences will be clearly resolvable (none of the sources give significance guidance — this is standard stats we add ourselves).
- **Two-tier reporting**: single-turn battery (sanity tier: variants should be ~equal here — if a memory method hurts single-turn answers, that's a bug, and it cleanly isolates *memory* effects to the multi-turn tier) + multi-turn session battery (where the methods actually separate).

### The results matrix (the workshop money-slide)
Rows = variants A–F. Columns = answer pass rate · key-point coverage · faithfulness · memory-probe accuracy · retrieval Recall@5 (late-turn) · expected-behavior accuracy · cumulative input tokens/session · cost/session · p50/p95 TTFT · summarizer overhead tokens. Plus the signature plot: **cumulative input tokens vs turn number, one line per variant**, annotated with where each variant's quality started dropping.

### Explaining the deltas (macro layer)
When a variant regresses, don't stop at the number: diff the failure taxonomies between variants (which failure modes did compression *create*?), and use first-failure/transition analysis (did failures move upstream into query formulation?). With a few hundred graded traces per variant, even the cookbook's lightweight version — group failed traces by eval_finding × case_type, look for concentration — is enough; full clustering machinery is optional.

---

## 8. Implementation notes (grounded in our code)

*Status (2026-07-15): everything in this section has since been built — memory presets (`app/memory_presets.py`), the runner/bundles/telemetry (`evals/run_battery.py`, `context_stats`), and the trigger gate (`evals/check_triggers.py`). See evals.md "How it runs". Line references below are historical.*

1. **Runner**: a Python script calling `stream_chat(ChatRequest(...))` directly (no HTTP needed) — `app/chat_service.py` is the single entry point. Multi-turn sessions: reuse one `thread_id` across turns (InMemorySaver is in-process, so the runner keeps state naturally). Persist one **trace bundle** JSONL per run: case_id, variant, trial, every ChatEvent, per-call token usage, timings, final answer.
2. **Parameterize the middlewares**: `build_agent` hard-codes ContextEditing/Summarization params (`app/chat_service.py:832–851`). Add a memory-config parameter (env var or ChatRequest field) selecting variant presets, and include it in the agent cache key. This is the one refactor the harness requires.
3. **Token/latency capture**: LangChain returns `usage_metadata` per model call; also already in LangSmith runs (tagged with model/thread; add `memory_variant` to the metadata dict at `app/chat_service.py:979–992`). Capture both — bundles for offline analysis, LangSmith for trace browsing during error analysis.
4. **Retrieval ground truth hook**: `tool_call_completed` events carry SearchResult matches (source, URL, score) — compute Recall@k/MRR in the runner, no retriever changes needed.
5. **Verify triggers fire**: log summarization/context-edit activations per session; assert the multi-turn suite actually trips them (else lengthen sessions).
6. **LangSmith vs roll-your-own**: use LangSmith for trace inspection + telemetry; keep the runner/dataset/judging in-repo (plain JSONL + pytest-style checks, per howtoeval) so the workshop materials are self-contained and reproducible without a SaaS dependency.

### Phased build (each phase ships something usable alone)
1. **Curate** (≈1 day): expert-review the 62 gold annotations; freeze a v1 battery (~80 single-turn cases, time_bound excluded); pick 5–10 **golden cases** that gate any future ship.
2. **Runner + bundles** (≈1–2 days): batch runner, trace persistence, telemetry. Layer-1 metrics work immediately.
3. **Error analysis** (≈1 day of expert time): run baseline, read all traces in a small Streamlit viewer, build the failure taxonomy. *Do this before writing any judge.*
4. **Programmatic checks** (≈1 day): retrieval recall, behavior routing, citation validity, + checks derived from step 3.
5. **Judge** (≈2 days incl. labeling): key-point coverage + faithfulness + holistic, validated to >90% TPR/TNR against expert labels.
6. **Multi-turn suite** (≈2 days): 5 hand-written sessions → ~30–50 generated; memory probes; N-1 replay of the 9 real multi-turn threads.
7. **Experiment matrix** (compute-bound): variants × battery × 3 trials; results matrix; failure-taxonomy diff per variant.

---

## 9. Workshop framing suggestions

- The build order **is** the talk: why generic metrics fail → error analysis on real student questions (show real traces!) → decomposed binary checks → judge validation (show the TPR/TNR table — audiences rarely see *judge* evals) → controlled comparison → results matrix → "why did E fail" trace autopsy.
- The honest tension between sources is great material: Hamel (judges, 60–80% time) vs Hylak (judge-skeptic, golden cases, 10–20%) vs OpenAI (population-scale macro analysis). Resolution: maturity and stakes determine the dose; validation discipline determines whether judges are trustworthy at all.
- The memory-probe design (plant facts → inflate context past compression triggers → probe) is the novel, reusable artifact attendees take home — none of the public sources cover it.
- Have one **counterintuitive result** ready (e.g., "aggressive compression cut cost 70% and memory-probe accuracy only dropped 8 points" or the reverse) — that's what gets shared.

## 10. The two-week workshop cut (what to actually do before the talk)

*Status (2026-07-15): shipped — see evals.md Parts B/C and the findings log.*

The full plan above is research-grade; the talk needs **demo-grade numbers on a meter plus one honest bake-off table**. The cut:

**Keep (the talk literally requires these):**
1. **Layer-1 telemetry = the meter.** Build the `context_stats` instrumentation once so both the UI meter and a headless runner consume it. Capture usage from the raw stream / `aget_state` *before* display filtering (compaction-ON runs undercount otherwise).
2. **Demo session scripts = the multi-turn suite, miniaturized.** The 13-turn compaction session and the 5-fact memory session ARE recall-after-compression probes. Write them deliberately: plant facts early, verify the 30k/5k triggers actually fire (log trigger events), probe late. 2–3 scripted sessions, rehearsed.
3. **Rehearsal = error analysis lite.** Read every trace from rehearsal runs. The failures found become the "failure first" demo beats — the talk format (failure → fix → number) *requires* harvesting real failures, which is exactly error analysis.
4. **Mini bake-off**: 3–4 configs (no-memory RAG baseline · compaction OFF/ON · + profile memory) × ~15–20 expert-verified gold single-turn cases × 2–3 scripted sessions, 1–2 trials. **Hand-grade** answers binary against key_points (≈60–80 readings, a half-day) instead of building a validated judge. Report counts honestly ("20 real student questions, 2 trials") — small-n with real data beats large-n synthetic for a talk.
5. **Replace every placeholder number before slides freeze** (4k→45k→8k, 0/5→5/5, 4k→800) with measured ones, and cache/pre-record fallback runs for the live demo.

**Cut / defer to post-workshop (and pitch as course content):** judge validation to >90% TPR/TNR, the 6-variant × 80-case × 3-trial matrix, N-1 replay, paired statistics, macro failure-pattern analysis, variants D/E/F beyond profile memory.

**Three fixes the talk outline needs:**
- **Stale code paths**: the outline targets `scripts/chat_service.py` + Gradio (`main.py`, `gradio_presenter.py`), but the repo is now `app/chat_service.py` + Next.js (Gradio was removed). The agent-side changes (toggles in `build_agent`, `MemoryMiddleware` via the `SourcePreferenceMiddleware` pattern, `InMemoryStore`, `context_stats` event) port ~1:1; the UI meter must be a Next.js component fed by the SSE stream (or the workshop branch resurrects the old Gradio snapshot — decide which, it changes the estimate).
- **The bake-off as written can't show a memory effect.** "No-memory RAG vs profile memory on the same questions" yields identical results if the questions are single-turn and profile-independent (§3 layer 5). The bake-off questions must be personalization-sensitive: phrased against the stored profile ("given my level / my project / my language preference…") or run as session turn-N probes. Otherwise the table shows memory ≈ baseline and undermines the demo.
- **Show compaction's tradeoff, not only the win.** 45k→8k tokens is the headline, but include one probe where summarization *loses* a planted detail (and full history doesn't). That's the credible, memorable beat — and it sets up "this is why you measure" for the course upsell.

**Rough schedule (10 working days):** D1–3 workshop-branch code (meter, toggles, memory, skills) · D4–5 curate ~20 gold cases + write session scripts + headless runner · D6–8 bake-off runs, hand-grading, build the table, harvest failure beats, lock real numbers · D9–10 rehearse live paths + cached fallbacks.

## 11. Pitfalls checklist

- ☐ Don't compare variants on single-turn cases only — memory methods are indistinguishable there.
- ☐ Don't trust the LLM-generated annotations without expert review (esp. key_points and eval_quality).
- ☐ Exclude/handle the 52 time_bound cases.
- ☐ Don't use an unvalidated judge; report its TPR/TNR alongside results.
- ☐ Count the *full* cost of each method (summarizer LLM calls, memory-store writes) — not just the saved prompt tokens.
- ☐ One trial per case is noise; pair the comparisons.
- ☐ No prefab metric dashboards; every automated check traces back to an observed failure or a dataset ground truth.
- ☐ Verify compression triggers actually fired in the multi-turn suite.
