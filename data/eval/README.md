# Evaluation batteries (v1, frozen 2026-06-11)

The datasets for evaluating the AI tutor (see `evals.md` at the repo root for the overall effort and results). **This README is the single reference for what every file, field, and term means.**

Built from `data/academy_discussion_eval.jsonl` — real student posts from the academy discussion boards with real staff answers — plus authored content (sessions, personas).

## Ground rules

- **Never edit a battery in place.** To change cases: edit the verdict/source files, rerun assembly, write `_v2`. Results are only comparable within one battery version.
- **All `*.jsonl` here is gitignored** (repo-wide rule) and contains real student text. Treat like the private HF dataset; never publish or paste into slides.
- **Inputs only.** No file contains LLM-generated answers to grade against. Ground truth is either real staff replies (distilled into key points) or facts we authored ourselves (sessions, personas).

## What each battery is for

| File | Tests | Main metrics it feeds | Graded by |
|---|---|---|---|
| `battery_singleturn_v1.jsonl` | One-shot answer quality and behavior routing. Memory presets should score ~equal here (sanity tier); differences indicate bugs, not memory effects. | key-point coverage · behavior accuracy · retrieval recall@k · citation validity · all ops metrics (tokens/cost/latency) | code + judge/hand-grade |
| `battery_sessions_v1.jsonl` | **Where memory presets separate.** Long sessions inflate context past compression triggers, then probes test what survived. | memory probe accuracy (per probe type) · cumulative tokens per session · trigger firings · late-turn retrieval recall | code (string checks) + judge for nuanced probes |
| `battery_personas_v1.jsonl` | The `profile_memory` preset specifically: does a stored student profile actually shape answers across sessions? | personalization pass rate | mostly code (regex), few judge checks |
| `replay_n1_v1.jsonl` | Multi-turn answering on *real* conversation prefixes (no synthetic anything). Secondary battery. | reply quality vs the real staff reply | judge or hand-grade |

"Metrics" here map to the metric layers defined in `evals.md` (ops, trajectory, retrieval, behavior, quality, memory).

## Glossary

- **Case / session / probe**: one gradeable unit. A *case* is a single question (single-turn, persona, replay). A *session* is an ordered multi-turn conversation run on one `thread_id`. A *probe* is a turn inside a session that gets graded (other turns are ungraded context).
- **eval_quality — `gold` / `usable` / `weak` / `exclude`**: reviewer confidence that the case is a fair, gradeable test. `gold` = clear question, verifiable expectation, safe to treat a failure as the system's fault. `usable` = gradeable but with a caveat (thin reference answer, fragile specifics) — failures deserve a second look before blaming the system. `weak`/`exclude` never enter a battery. "60 cases: gold 49 / usable 11" means 49 high-confidence + 11 caveated cases.
- **key_points**: 1–5 atomic, binary-checkable claims that a good answer SHOULD contain, distilled from the real staff answer (only claims still true today). Each is graded independently pass/fail ("does the answer correctly convey this point?") → the **key-point coverage** metric (fraction of points passed). This decomposition is what makes grading reliable: small binary checks instead of a holistic 1–5 score. Empty list is intentional for redirect/feedback cases — those are graded on behavior, not content.
- **expected_behavior**: which of the tutor's four correct moves this case demands. `answer_from_corpus` = retrieve and ground the answer in course/docs content. `answer_general` = legitimate AI/programming question outside the corpus; answer from general knowledge without fake citations. `redirect_to_support` = platform/billing/submission issue; empathize and point to human support, never invent platform answers. `acknowledge_feedback` = course feedback; thank and acknowledge, don't promise fixes. Graded as **behavior accuracy** (did the tutor do the right *kind* of thing), separate from answer content.
- **standalone question** (`question` field): the student's post rewritten so it works without seeing the discussion page — names the course/lesson, keeps code and error text, strips student names. This is the text actually sent to the tutor.
- **reference_answer / reference_reply / reference_links**: the real staff reply. **Grader context only — never shown to the tutor.** Key points are derived from it; graders read it to resolve ambiguity.
- **time_bound**: the staff answer depended on a point in time (broken link since fixed, "we'll add this soon", old library version). Time-bound cases were excluded unless durable behavior can still be graded — the 9 included `time_bound: true` cases are all platform issues where the *redirect behavior* is what's graded, not the stale facts.
- **requires_notebook**: fully answering needs the Colab notebook contents (cells/outputs/files), which our KB does not contain — verified against lesson markdown with file/line evidence (`notebook_evidence`), not guessed. All 60 included cases are `false`; the one true case in review was excluded. If retrieval misses on a flagged-adjacent case, check `notebook_evidence` before blaming the retriever.
- **requires_media**: the post referenced a screenshot we don't have. The 2 included cases remain gradeable without it.
- **planted facts**: personal facts (environment, goal, level, weak topic, preference/constraint) a session's student states in turn 0, which later probes test. The session-level analog of a profile.
- **Probe types** (`probe_type`): `fact_recall` — answer must use a planted fact from many turns ago. `preference_compliance` — answer must honor a stated preference (OS, tooling, language, style). `anaphora` — question refers to "the X you explained earlier"; unanswerable without resolving it to an earlier turn. `anaphora_consistency` — answer must stay consistent with the tutor's own earlier recommendation. `fact_update` — the student CHANGED a fact mid-session; using the old value = fail (catches summaries that freeze stale state — the signature compaction failure). `behavior_routing` — a platform issue dropped mid-session; the tutor must still redirect properly under context pressure.
- **expected_facts / check_note**: per probe, the specific facts a correct answer must use, and a one-line pass/fail rule for the grader. Grade probes binary → **memory probe accuracy**, reported overall and per probe type per memory preset.
- **profile_seed / facts**: the persona's stored profile. Seed it with `app.chat_service.set_student_profile(student_id, profile_seed)` before running that persona's questions; `facts` is the same content structured, for slicing results by fact type.
- **Self-grading checks** (`checks` / `anti_patterns`): persona questions grade themselves because we authored the ground truth. `{"type": "regex_any", "patterns": [...]}` = case-insensitive regex over the answer; any match passes that check. `{"type": "llm", "instruction": ...}` = a judge call with that instruction (use a strong Anthropic model; binary verdict). `anti_patterns` = case-insensitive regexes whose match FAILS the question (e.g. bash `export` advice for a Windows persona). A question passes when every check passes and no anti-pattern matches.
- **N-1 replay**: take a real thread, feed turns 1..N-1 as history, have the tutor produce turn N, grade against what the staff actually replied. Tests multi-turn behavior on fully real data (Hamel's recommended alternative to simulated users).
- **Retrieval ground truth**: every discussion-derived case carries `source_key` + `lesson_url` — the course/lesson the question came from. Computing whether retrieval surfaced that source/lesson gives **recall@k / MRR** with zero extra labeling.
- **review / verdict / review log**: every single-turn case carries the reviewer's `changes` (diff vs the original Gemini annotation) and `notes`. `review_log_v1.md` is the human-readable audit: systematic findings, the 11 judgment-call cases, all 92 verdicts, all 32 exclusions. Machine-readable verdicts: `review_batches/verdicts_batch_*.jsonl`.
- **Memory preset**: a named memory/context configuration (`full_history`, `prod`, `aggressive`, `profile_memory`, ...) — see `app/memory_presets.py`. The experiment variable these batteries measure.

## File schemas

### `battery_singleturn_v1.jsonl` — 60 cases, one JSON object per line

| Field | Meaning |
|---|---|
| `case_id` | Stable id (`st_<post_id>`). Use in run logs/bundles. |
| `post_id` | Provenance: the original academy discussion post (`academy.towardsai.net/manage/discussion/posts/<post_id>`). |
| `course`, `source_key`, `lesson_name`, `lesson_url` | Where the question came from; `source_key`+`lesson_url` double as retrieval ground truth. |
| `asked_at` | Original post date (for staleness debugging). |
| `question` | The standalone question — the prompt sent to the tutor. |
| `category` | `conceptual` / `debugging` / `platform_issue` / `course_feedback` / `other` — for slicing results. |
| `expected_behavior` | See glossary. Drives the behavior-accuracy check. |
| `key_points` | See glossary. Empty for behavior-only cases (13 of 60). |
| `eval_quality` | `gold` (49) or `usable` (11) — see glossary. |
| `time_bound` | 9 true: grade behavior only, ignore stale specifics. |
| `requires_notebook`, `notebook_evidence` | All false; evidence records what was checked in the KB. |
| `requires_media` | 2 true; still gradeable. |
| `reference_answer`, `reference_links` | Real staff reply — grader-only. |
| `review.{changes,notes,batch}` | Reviewer audit trail per case. |

Distribution to keep in mind when reporting: behaviors 37 corpus / 10 redirect / 7 general / 6 feedback; courses 41 full-stack / 16 agentic / 3 beginner-python (skewed — say so on slides).

### `battery_sessions_v1.jsonl` — 32 sessions (337 turns, 113 probes)

| Field | Meaning |
|---|---|
| `session_id` | e.g. `s08_fullstack_gamedev_update_12t` (s01–s05 hand-written, s06–s32 generated + spot-checked). |
| `course`, `source_key` | Course context; pass `source_keys=(source_key,)` to the tutor. |
| `persona` | One-line description of the student (for readers, not the tutor). |
| `planted_facts` | The facts turn 0 establishes (and mid-session updates, marked "UPDATED at turn k"). |
| `turns` | Ordered user messages. Run ALL sequentially on ONE `thread_id`. Middle turns are real corpus questions (post_ids in `notes`) — ungraded context whose job is to inflate tokens past compression triggers. Some reference since-changed content; that's realistic, leave it. |
| `probes` | The graded turns: `turn_index` (0-based into `turns` — the probe IS that turn), `probe_type` (glossary), `expected_facts`, `check_note`. |
| `notes` | Provenance: which post_ids / lesson names the turns use. |

Probe mix: fact_recall 55 · preference_compliance 27 · anaphora 12 · anaphora_consistency 9 · fact_update 7 (sessions s08, s11, s18, s21, s24, s31) · behavior_routing 3.
**Before the bake-off**: smoke-run a few sessions and confirm compression actually fired before the probe turns (`context_stats.summary_messages` / `cleared_tool_outputs` > 0); lengthen sessions if not — a memory eval where compaction never triggered measures nothing.

### `battery_personas_v1.jsonl` — 10 personas × 4 questions

| Field | Meaning |
|---|---|
| `persona_id` | Also use as the `student_id` for seeding/runs. |
| `course`, `source_key` | Course context. |
| `profile_seed` | Text to write into the store via `set_student_profile()` before the run. |
| `facts` | Same profile, structured (level/goal/environment/weak_topic/preference) for slicing. |
| `questions[].question_id` | e.g. `p01_q3`. |
| `questions[].question` | Sent on a FRESH `thread_id` per question — isolates long-term memory from working memory. Deliberately unanswerable-correctly without the profile. |
| `questions[].expected_facts_used` | Which profile facts a correct answer must draw on. |
| `questions[].checks`, `anti_patterns`, `check_note` | Self-grading rules — see glossary. |

Run each persona under `profile_memory` AND under a no-memory preset: the gap between the two pass rates is the personalization effect.

### `replay_n1_v1.jsonl` — 30 cases from 24 real threads

| Field | Meaning |
|---|---|
| `replay_id` | `replay_<post_id>_t<k>`: replays up to staff turn k. |
| `post_id`, `course`, `source_key`, `lesson_url`, `category` | Provenance + retrieval ground truth. |
| `eval_quality`, `time_bound` | 17 gold / 13 usable; 8 time_bound — filter those for most uses. |
| `history` | Real conversation prefix: original question + thread turns, mapped student→`user`, staff→`assistant`. Feed as the request history; the last entry is always a user message. |
| `reference_reply` | The real staff reply to grade against (grader-only). |

Contains real names inside message text (inherent to replay) — private.

### Provenance files

- `review_batches/batch_*.json` — the 92 raw cases as sent to reviewers; `review_batches/verdicts_batch_*.jsonl` — machine-readable verdicts. To overturn a verdict: edit the line, rerun assembly, bump the battery to v2.
- `review_log_v1.md` — the human audit document for the single-turn battery.
- `sessions_handwritten.jsonl`, `sessions_generated_{a,b,c}.jsonl` — raw session inputs merged into `battery_sessions_v1.jsonl`.

## How grading will work (Part A3/B of the experiments plan)

- **Single-turn**: behavior routing + citation validity as code assertions; key-point coverage hand-graded for the workshop (≈60–80 binary readings), later automated by a judge validated against those hand labels (product plan P4). Judges: strong Anthropic model via API, binary verdict + one-line critique per check.
- **Sessions**: probe `expected_facts` checked by string match where unambiguous, judge with `check_note` as the rubric otherwise; trigger-firing verified from `context_stats` telemetry.
- **Personas**: fully programmatic except `type: "llm"` checks.
- **Replay**: judge or hand-grade against `reference_reply`.
- Every reported number should name its battery version (e.g. "key-point coverage, singleturn_v1") and, for judge-graded metrics, the judge's measured agreement with human labels (TPR/TNR).
