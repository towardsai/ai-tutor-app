# Evaluation batteries (v1, frozen 2026-06-11)

The datasets for evaluating the AI tutor (see `evals.md` at the repo root for the overall effort and results). **This README is the single reference for what every file, field, and term means.** Files added after the v1 freeze: `battery_sessions_v2.jsonl` (deprecated for new runs — see the v2 note in the probe-type glossary), `battery_sessions_v2_1.jsonl` (its repaired successor), and `compaction/` (the F29/F30 lesson-compaction batteries, documented in `evals/compaction.md` / `evals/slm_compaction.md`; note the compaction scripts default to `data/compaction/`, not this folder — either regenerate there with `compaction_study build` or point `BATTERY=`/`QUESTIONS_FILE=` at these copies).

Built from `data/academy_discussion_eval.jsonl` — real student posts from the academy discussion boards with real staff answers — plus authored content (sessions, personas).

## Quick mental model

- A **battery** is one JSONL file full of related tests. For example, the single-turn battery is one-question chats, while the sessions battery is longer conversations.
- A **case** is one test item. Depending on the battery, a case might be one question, one persona question, one full session, or one replay.
- A **run** executes a battery with one model and one memory preset. The output is saved in `runs/.../bundles.jsonl`.
- A **bundle** is the saved evidence for one tutor turn: input, answer, tool calls, source matches, timing, token usage, and errors.
- A **grade** is computed later from bundles. Some grades are automatic; others are filled by a human in a blinded workbook.

## Ground rules

- **Never edit a battery in place.** To change cases: edit the verdict/source files, rerun assembly, write `_v2`. Results are only comparable within one battery version.
- **All `*.jsonl` here is gitignored** (repo-wide rule) and contains real student text. Treat like the private HF dataset; never publish or paste into slides.
- **Inputs only.** No file contains LLM-generated answers to grade against. Ground truth is either real staff replies (distilled into key points) or facts we authored ourselves (sessions, personas).

## What each battery is for

| File | Tests | Main metrics it feeds | Graded by |
|---|---|---|---|
| `battery_singleturn_v1.jsonl` | One-question chats from real student posts. These check whether the tutor answers correctly and chooses the right kind of response. Memory presets should score about the same here; if they do not, the memory setting may be changing behavior in a bad way. | Key-point coverage, behavior accuracy, whether retrieval found the right source, citation validity, tokens, cost, latency. | Automatic checks plus human or validated-judge grades (F14). |
| `battery_sessions_v1.jsonl` | Longer study sessions. Early turns plant facts about the student, middle turns make the conversation large, and later graded turns test whether the tutor still remembers or updates important facts. | Memory probe accuracy, cumulative tokens, whether compaction fired before probes, late-turn retrieval behavior. | Automatic trigger checks plus human or judge grades for nuanced probes. |
| `battery_personas_v1.jsonl` | Stored student profiles plus fresh questions. Each question is designed so the best answer needs profile details, such as OS, skill level, or goal. | Personalization pass rate. | Mostly automatic regex and anti-pattern checks; a few rows need human or judge review. |
| `replay_n1_v1.jsonl` | Real discussion threads turned into "what should the next staff reply be?" tests. The tutor receives the real conversation up to just before a staff reply, then writes the next reply. | How close the tutor's reply is to the real staff reply. | Human or validated judge grade. |

"Metrics" here map to the layers defined in `evals.md`: runtime/cost, tool behavior, retrieval, response behavior, answer quality, and memory.

## Glossary

- **Case / session / probe**: A *case* is one test item. A *session* is a multi-turn conversation run on one `thread_id`. A *probe* is a specific turn inside a session that gets graded. Other session turns are there to create realistic context and make the conversation long enough to test memory.
- **eval_quality — `gold` / `usable` / `weak` / `exclude`**: reviewer confidence that the case is a fair, gradeable test. `gold` = clear question, verifiable expectation, safe to treat a failure as the system's fault. `usable` = gradeable but with a caveat (thin reference answer, fragile specifics) — failures deserve a second look before blaming the system. `weak`/`exclude` never enter a battery. "60 cases: gold 49 / usable 11" means 49 high-confidence + 11 caveated cases.
- **key_points**: 1–5 atomic, binary-checkable claims that a good answer SHOULD contain, distilled from the real staff answer (only claims still true today). Each is graded independently pass/fail ("does the answer correctly convey this point?") → the **key-point coverage** metric (fraction of points passed). This decomposition is what makes grading reliable: small binary checks instead of a holistic 1–5 score. Empty list is intentional for redirect/feedback cases — those are graded on behavior, not content.
- **expected_behavior**: which of the tutor's four correct moves this case demands. `answer_from_corpus` = retrieve and ground the answer in course/docs content. `answer_general` = legitimate AI/programming question outside the corpus; answer from general knowledge without fake citations. `redirect_to_support` = platform/billing/submission issue; empathize and point to human support, never invent platform answers. `acknowledge_feedback` = course feedback; thank and acknowledge, don't promise fixes. Graded as **behavior accuracy** (did the tutor do the right *kind* of thing), separate from answer content.
- **standalone question** (`question` field): the student's post rewritten so it works without seeing the discussion page — names the course/lesson, keeps code and error text, strips student names. This is the text actually sent to the tutor.
- **reference_answer / reference_reply / reference_links**: the real staff reply. **Grader context only — never shown to the tutor.** Key points are derived from it; graders read it to resolve ambiguity.
- **time_bound**: the staff answer depended on a point in time (broken link since fixed, "we'll add this soon", old library version). Time-bound cases were excluded unless durable behavior can still be graded — the 9 included `time_bound: true` cases are all platform issues where the *redirect behavior* is what's graded, not the stale facts.
- **requires_notebook**: fully answering needs the Colab notebook contents (cells/outputs/files), which our KB does not contain — verified against lesson markdown with file/line evidence (`notebook_evidence`), not guessed. All 60 included cases are `false`; the one true case in review was excluded. If retrieval misses on a flagged-adjacent case, check `notebook_evidence` before blaming the retriever.
- **requires_media**: the post referenced a screenshot we don't have. The 2 included cases remain gradeable without it.
- **planted facts**: personal facts (environment, goal, level, weak topic, preference/constraint) a session's student states in turn 0, which later probes test. The session-level analog of a profile.
- **Probe types** (`probe_type`): `fact_recall` means the answer must use a fact the student gave many turns ago. `preference_compliance` means the tutor must honor a stated preference, such as OS, tooling, language, or style. `anaphora` means the user asks about "that thing from earlier," so the tutor must resolve the reference. `anaphora_consistency` means the answer must stay consistent with the tutor's own earlier recommendation. `fact_update` means the student changed a fact mid-session; using the old value fails. `behavior_routing` means a support/platform issue appears inside a long chat, and the tutor must still redirect appropriately. Battery v2 (`battery_sessions_v2.jsonl`, built 2026-06-16: 6 sessions, `tier`/`tags` metadata, filler recycled from v1 corpus questions) added `contradiction` (a fact changed early and probed only after it would be evicted — note the eviction is not guaranteed in capped arms; see evals.md F36), `longhorizon_recall`, and `entity_isolation` (two entities in one session, no fact bleed). **Use `battery_sessions_v2_1.jsonl` for new runs**: the 2026-07-15 audit found v2's recycled filler included v1 persona/update/pivot turns whose first-person claims collide with the planted facts (7 turns across 4 sessions; details in evals.md harness corrections); v2.1 replaces those turns with neutral same-course corpus questions, probes and plants unchanged, and the runners refuse v2 without `--allow-deprecated-battery`. `cross_session_recall` was an earlier idea for `sleeptime_consolidation` and is deferred with the multi-thread battery; the v1 types here are frozen. See `evals/part_c_plan.md` → "Battery v2" for the build plan.
- **expected_facts / check_note**: per probe, the specific facts a correct answer must use, and a one-line pass/fail rule for the grader. Grade probes binary → **memory probe accuracy**, reported overall and per probe type per memory preset.
- **profile_seed / facts**: the persona's stored profile. Seed it with `app.chat_service.set_student_profile(student_id, profile_seed)` before running that persona's questions; `facts` is the same content structured, for slicing results by fact type.
- **Self-grading checks** (`checks` / `anti_patterns`): persona questions grade themselves because we authored the ground truth. `{"type": "regex_any", "patterns": [...]}` = case-insensitive regex over the answer; any match passes that check. `{"type": "llm", "instruction": ...}` = a judge call with that instruction (use a strong Anthropic model; binary verdict). `anti_patterns` = case-insensitive regexes whose match FAILS the question (e.g. bash `export` advice for a Windows persona). A question passes when every check passes and no anti-pattern matches.
- **Replay / N-1 replay**: A replay is a real multi-turn discussion converted into a prediction task. "N-1" means "show the tutor every turn before the next staff reply, then ask it to produce that next reply." The real staff reply is kept hidden from the tutor and used only for grading.
- **Retrieval ground truth**: every discussion-derived case carries `source_key` + `lesson_url`, the course and lesson the question came from. If retrieval shows the model a chunk from that source or lesson, it gets retrieval credit.
- **recall@shown**: The share of cases where the correct source or lesson appeared somewhere in the retrieval results shown to the model. This does not measure whether the source exists in the whole database; it measures whether the model actually saw it during the turn.
- **MRR**: Mean reciprocal rank. If the correct lesson is first in the shown results, that case scores 1.0; second scores 0.5; third scores 0.33; missing scores 0. Higher means the right evidence appeared earlier.
- **review / verdict / review log**: every single-turn case carries the reviewer's `changes` (diff vs the original Gemini annotation) and `notes`. `review_log_v1.md` is the human-readable audit: systematic findings, the 11 judgment-call cases, all 92 verdicts, all 32 exclusions. Machine-readable verdicts: `review_batches/verdicts_batch_*.jsonl`.
- **Memory preset**: a named memory/context configuration (`full_history`, `prod`, `aggressive`, `profile_memory`, ...) — see `app/memory_presets.py`. This is the main experiment variable.
- **Compaction**: reducing the prompt by summarizing old conversation turns or clearing old tool outputs. It saves prompt space, but can also remove details the tutor later needs.
- **Human grade / hand grade**: a person reads the question, answer, and criterion, then marks pass or fail.
- **Judge grade**: an LLM marks pass or fail. Judge grades are only reportable after the judge is validated against held-out human grades.

## File schemas

### `battery_singleturn_v1.jsonl` — 60 cases, one JSON object per line

Use this file to test one-turn behavior. The tutor gets only `question`, not the staff answer or review fields.

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

Use this file to test memory over a longer conversation. Run every message in `turns` sequentially on one thread. Only the turns listed in `probes` are graded.

| Field | Meaning |
|---|---|
| `session_id` | e.g. `s08_fullstack_gamedev_update_12t` (s01–s05 hand-written, s06–s32 generated + spot-checked). |
| `course`, `source_key` | Course context; pass `source_keys=(source_key,)` to the tutor. |
| `persona` | One-line description of the student (for readers, not the tutor). |
| `planted_facts` | The facts turn 0 establishes (and mid-session updates, marked "UPDATED at turn k"). |
| `turns` | Ordered user messages. Run ALL sequentially on ONE `thread_id`. Middle turns are real corpus questions (post_ids in `notes`) — ungraded context whose job is to inflate tokens past compression triggers. Some reference since-changed content; that's realistic, leave it. |
| `probes` | The graded turns: `turn_index` (0-based into `turns` — the probe IS that turn), `probe_type` (glossary), `expected_facts`, `check_note`. |
| `notes` | Provenance: which post_ids / lesson names the turns use. |
| `tier`, `tags` | Optional v2 metadata for staged runs, for example `tier1_contradiction`, `tier2_longhorizon`, `tier3_entity`. The runner can filter these with `--tags`; v1 sessions usually omit them. |

Probe mix: fact_recall 55 · preference_compliance 27 · anaphora 12 · anaphora_consistency 9 · fact_update 7 (sessions s08, s11, s18, s21, s24, s31) · behavior_routing 3.
**Before a comparison run**: smoke-run a few sessions and confirm compression actually fired before the probe turns (`context_stats.summary_messages` > 0, or `evals.check_triggers`; for clearing arms use input-token deltas — `cleared_tool_outputs` is structurally always 0, see evals.md F24); lengthen sessions if not — a memory eval where compaction never triggered measures nothing.

### `battery_personas_v1.jsonl` — 10 personas × 4 questions

Use this file to test stored student profiles. Before each persona question, write `profile_seed` into the profile store. Each question starts on a fresh thread so the answer must come from long-term profile memory, not from recent chat history.

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

### `replay_n1_v1.jsonl` — 30 "write the next staff reply" cases from 24 real threads

Use this file to test multi-turn behavior without simulated users. The runner feeds the tutor the real conversation prefix in `history`; the last history entry is a user message. The tutor then answers, and graders compare its answer with `reference_reply`, the real staff reply that came next.

| Field | Meaning |
|---|---|
| `replay_id` | `replay_<post_id>_t<k>`: identifies the original post and which staff reply is being predicted. |
| `post_id`, `course`, `source_key`, `lesson_url`, `category` | Provenance + retrieval ground truth. |
| `eval_quality`, `time_bound` | 17 gold / 13 usable; 8 time_bound — filter those for most uses. |
| `history` | Real conversation prefix: original question + thread turns, mapped student→`user`, staff→`assistant`. Feed as the request history; the last entry is always a user message. |
| `reference_reply` | The real staff reply to grade against (grader-only). |

Contains real names inside message text (inherent to replay) — private.

### Provenance files

- `review_batches/batch_*.json` — the 92 raw cases as sent to reviewers; `review_batches/verdicts_batch_*.jsonl` — machine-readable verdicts. To overturn a verdict: edit the line, rerun assembly, bump the battery to v2.
- `review_log_v1.md` — the human audit document for the single-turn battery.
- `sessions_handwritten.jsonl`, `sessions_generated_{a,b,c}.jsonl` — raw session inputs merged into `battery_sessions_v1.jsonl`.

## How grading works

- **Single-turn**: Code checks whether the tutor used retrieval when expected and whether citations were present. Humans grade key points and final behavior correctness.
- **Sessions**: Humans or a validated judge grade the probe turn using `expected_facts` and `check_note`. Code also verifies that compaction had actually fired before probe turns when the preset is supposed to compact.
- **Personas**: Most rows grade automatically with regex checks and anti-patterns. Rows with `{"type": "llm"}` need a human or validated judge.
- **Replay**: Humans or a validated judge compare the tutor answer with `reference_reply`, the real staff answer.
- Every reported number should name its battery version, for example "key-point coverage, singleturn_v1". Judge-graded metrics should also report the judge's measured agreement with human labels.
