"""State-tracking context-rot: does keep-everything preserve the STATE of a real
investigation across a long, messy history, or does it lose the thread?

The single-fact grid found ~90-100% recall to 800k, but that is the easy case:
one distinctive fact is a lexical grab, exactly what these models are trained for
(Louis's review: "we tested exactly what the models are built to do"). This probe
removes that shortcut. It plants ONE real debugging thread whose state evolves
over time, buries it among many other issues, and asks a synthesis question that
can only be answered by reasoning across the whole history.

Two modes (``--mode``), so you can show the contrast on one slide:

  easy  - the original probe (kept as a CONTROL). The resolved thread literally
          says "RESOLVED / closing as fixed", the root cause and rejected fix are
          stated verbatim and adjacent, and it is the only thread with resolution
          language. So "which bug was fixed" is a keyword grab and every component
          is a copy. This is why it scored 4/4 -- it is a harder NIAH, not
          reasoning. Run it to reproduce the 4/4 baseline.

  hard  - the reasoning probe. The shortcuts are removed:
            * NO "RESOLVED" beacon: the fix is confirmed only by an offhand,
              later, unrelated note ("the freeze reports stopped after X shipped").
            * Several DISTRACTORS also claim a fix, so the resolved issue must be
              discriminated, not keyword-matched.
            * The failed attempt is NOT labelled "did not work": the symptom
              simply persists after it, so its failure must be INFERRED.
            * The trap (an abandoned hypothesis) is planted LATE and CONFIDENT and
              is never retracted, so recency bias pulls toward the wrong answer.
            * Root cause and fix are scattered, not adjacent, so the answer has to
              be assembled across positions.
          If components recovered fall as the history grows, OR the trap rate
          climbs, that is real lost-in-the-middle on REASONING, not retrieval.

Validity fixes vs the first version: (1) trials are now INDEPENDENT -- each trial
draws a different bug case + jittered beat positions + a shuffled distractor set
(seeded via ``--seed``), so n>1 actually means something instead of repeating one
temperature-0 run; (2) truncated cells (answer hit the output cap or came back
empty) are EXCLUDED from the headline mean and reported separately, so the 300- /
8000-token truncation artifact can never again be counted as a reasoning miss.

Same filler, model, and judge as the single-fact grid; retrieval OFF. Format
defaults to CONVERSATION (a real multi-turn chat history, the tutor's shape, and
the shape Omar tested); pass --format document for the stuffed-log shape. Hard
mode draws from 6 scenarios; --trials are randomized LAYOUTS (jittered positions
+ shuffled distractors), so >6 trials repeat scenarios with fresh layouts.

  # reproduce the easy 4/4 baseline (control)
  uv run --env-file .env -m evals.context_rot_state --mode easy

  # the real test: reasoning sweep on DeepSeek's big window (conversation default)
  uv run --env-file .env -m evals.context_rot_state --mode hard \
      --fills 8000 100000 400000 800000 --trials 12 --out data/context_rot_state_hard

  # same probe as a stuffed document, to compare shapes (document vs conversation)
  uv run --env-file .env -m evals.context_rot_state --mode hard --format document \
      --fills 8000 100000 400000 800000 --trials 12 --out data/context_rot_state_hard_doc

  # local 32B contrast (in-window only; overflow cliff already shown elsewhere)
  uv run --env-file .env -m evals.context_rot_state --mode hard \
      --provider ollama --model ollama_chat/qwen2.5:32b-instruct --num-ctx 32768 \
      --fills 8000 16000 24000 --trials 12 --out data/context_rot_state_hard_local
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean

from .context_rot import (
    PROVIDERS,
    Needle,
    judge,
    load_filler,
)
from .knowledge_compaction import complete

logger = logging.getLogger("evals.context_rot_state")

# Generous output budget: a 4-part synthesis answer is naturally long, and a
# reasoning model may think before it answers. A truncation guard flags any cell
# that still hits the cap; flagged cells are INVALID and excluded from the mean
# (overridable with --answer-max-tokens; pair with --no-think on verbose models).
ANSWER_MAX_TOKENS = 8000


# ======================================================================== EASY
# The original probe, kept verbatim as a CONTROL. Its 4/4 is the artifact we are
# contrasting against: the resolved thread self-announces, components are copies,
# and it is the only thread with resolution language.

EASY_TARGET_BEATS: tuple[str, ...] = (
    "Support thread #4412 (streaming): a user reports that streaming responses "
    "freeze right after a tool call runs. Tokens stop arriving and the chat UI "
    "just spins. It reproduces whenever the agent calls a tool mid-answer.",
    "Re: #4412 streaming freeze. First theory: maybe this is a CORS problem "
    "between the Next.js frontend and the FastAPI backend. We chased that angle "
    "for a while.",
    "Re: #4412 streaming freeze. We tried bumping the request timeout way up, in "
    "case the backend was just slow to finish after the tool call. No luck, the "
    "freeze still happened. So the timeout was not the cause.",
    "Re: #4412 streaming freeze. Found it: the UI was waiting for a 'message.done' "
    "event to close out the turn, but after a tool call the backend was emitting "
    "'response.done' instead, so the UI never saw the event it was waiting for and "
    "hung.",
    "Re: #4412 streaming freeze, RESOLVED. Fix: we normalized the backend stream "
    "events to the AI SDK UI-message protocol, so the UI now receives the exact "
    "events it expects. Verified that streaming no longer freezes after tool "
    "calls. Closing this one as fixed.",
)

EASY_DISTRACTOR_THREADS: tuple[str, ...] = (
    "Thread #3001 (retrieval): Chroma returned zero chunks after a re-ingest. "
    "Suspected an index/dimension mismatch. Still investigating, not resolved.",
    "Thread #3002 (rerank): Cohere rerank intermittently times out on large "
    "batches. Considering smaller batches. No fix confirmed yet.",
    "Thread #3003 (citations): a citation card sometimes renders without its "
    "source URL. Looks like the manifest lookup misses. Open.",
    "Thread #3004 (kb): run_kb_command failed with a path error when the query "
    "used an absolute path; the sandbox jail likely rejected it. Unconfirmed.",
    "Thread #3005 (caching): Gemini implicit cache seemed to miss on long prompts "
    "and cost spiked. Maybe a prompt-prefix change broke the cache. Still digging.",
    "Thread #3006 (routing): OpenRouter routed the same model id to different "
    "quant backends across days, drifting eval scores. Noted, no fix.",
    "Thread #3007 (frontend): Next.js logged a hydration mismatch warning on the "
    "chat page. Possibly a server/client timestamp diff. Not yet fixed.",
    "Thread #3008 (streaming): an SSE connection occasionally dropped mid-stream "
    "on flaky networks. A retry idea was floated but not shipped.",
    "Thread #3009 (tools): a tool call returned malformed JSON and the agent "
    "retried forever. Capping retries is a candidate fix, still testing.",
    "Thread #3010 (embeddings): switching to embed-v4.0 changed vector dimensions "
    "and broke an old collection. Rebuild pending.",
    "Thread #3011 (auth): the HF token expired and first-start downloads failed. "
    "Rotating the token is the plan.",
    "Thread #3012 (latency): p95 latency jumped after enabling web search; the "
    "fetch tool is suspected. Investigating.",
    "Thread #3013 (memory): the summarization preset dropped a student's earlier "
    "detail. Considering keep-everything. Open.",
    "Thread #3014 (ui): the message list scrolled to the top on each new token. A "
    "scroll-anchor tweak is proposed, not merged.",
    "Thread #3015 (protocol): a 'done' event arrived twice for one turn, "
    "double-rendering the final card. Possibly a duplicate emit. Unconfirmed.",
    "Thread #3016 (rerank): rerank scored a generic chunk above the exact match on "
    "one query. A hybrid-weight change is under consideration.",
    "Thread #3017 (kb): headings.jsonl went stale after a corpus update, so "
    "navigation pointed at old anchors. Regen pending.",
    "Thread #3018 (cost): DeepSeek via OpenRouter showed cache hits ~50x cheaper, "
    "but only sometimes. Logging more before acting.",
    "Thread #3019 (frontend): a tool-status popover flickered on hover; a CSS race "
    "is suspected. Low priority, open.",
    "Thread #3020 (ingest): a Notion export had broken image links that failed "
    "processing. Skipping images is the workaround under review.",
    "Thread #3021 (cors): an actual CORS error blocked the dev frontend from the "
    "API on a newly added port. Adding the origin to the allowlist is proposed.",
    "Thread #3022 (timeout): a long web_fetch hit the request timeout and errored. "
    "Raising the timeout for that specific call is being tested.",
    "Thread #3023 (judge): the eval judge occasionally returned unparseable JSON, "
    "failing a grade. A regex fallback is proposed.",
    "Thread #3024 (vectors): create_vector_stores ran out of memory on the full "
    "corpus. Batching the upserts is planned.",
    "Thread #3025 (streaming): first-token latency felt slow on cold start. "
    "Warming the model is one idea. Open.",
    "Thread #3026 (citations): two sources merged into one card when their URLs "
    "differed only by a fragment. The dedup key needs the fragment. Open.",
)

EASY_COMPONENTS: tuple[Needle, ...] = (
    Needle(
        fact="",
        probe="Which bug was actually fixed?",
        answer="the bug where streaming responses freeze after a tool call",
        distractors=(),
    ),
    Needle(
        fact="",
        probe="What was the final root cause of that bug?",
        answer="the UI was waiting for a message.done event but the backend "
        "emitted response.done after tool calls",
        distractors=(),
    ),
    Needle(
        fact="",
        probe="Which attempted fix was tried but rejected because it did not work?",
        answer="increasing the request timeout (it did not fix it)",
        distractors=(),
    ),
    Needle(
        fact="",
        probe="What was the final code-level change that resolved it?",
        answer="normalizing the backend stream events to the AI SDK UI-message "
        "protocol",
        distractors=(),
    ),
)

EASY_QUESTION = (
    "In the debugging history above, several streaming-related problems are "
    "mentioned, but only ONE was actually fixed. Answer all four, numbered:\n"
    "1) Which streaming bug was actually fixed?\n"
    "2) What was the final root cause?\n"
    "3) Which attempted fix was tried but rejected because it did not work?\n"
    "4) What exact code-level change finally resolved it?\n"
    "Answer concisely. If something truly is not present, say you cannot find it."
)

EASY_TRAP_CLAIM = (
    "the streaming bug was fixed or resolved by increasing/raising the timeout, "
    "or by fixing a CORS issue"
)


# ======================================================================== HARD
# The reasoning probe. Each case is ONE bug whose state must be reconstructed:
#   symptom -> failed attempt (NOT labelled failed) -> real root cause (mid) ->
#   real fix (mid) -> a LATE, confident, never-retracted wrong hypothesis (the
#   trap) ; and the ONLY confirmation the fix worked is a separate, later,
#   offhand note (resolution_signal). No "RESOLVED" beacon anywhere.

# Six beats placed across the whole window; the trap sits late (0.84) so recency
# bias favours it, the real fix sits earlier (0.62) so it cannot win on recency.
THREAD_POSITIONS: tuple[float, ...] = (0.08, 0.20, 0.34, 0.50, 0.62, 0.84)
RESOLUTION_POSITION = 0.95


@dataclass(frozen=True)
class HardCase:
    key: str
    # (role, text) in chronological order, exactly len(THREAD_POSITIONS) beats.
    thread: tuple[tuple[str, str], ...]
    # The only confirmation the fix worked: a later, unrelated, offhand note.
    resolution_signal: str
    components: tuple[Needle, ...]  # 4 gradeable ground truths
    trap_claim: str  # what crediting the abandoned hypothesis looks like


HARD_CASES: tuple[HardCase, ...] = (
    HardCase(
        key="streaming",
        thread=(
            (
                "user",
                "Streaming answers freeze right after the agent runs a tool. "
                "Tokens stop arriving and the chat just spins. It only happens when a "
                "tool is called in the middle of an answer.",
            ),
            (
                "user",
                "I raised the request timeout to 120s, figured the backend was "
                "just slow to finish after the tool call.",
            ),
            (
                "user",
                "Still freezing after the timeout change. Same spinner, same "
                "place, no difference at all.",
            ),
            (
                "assistant",
                "Watching the stream: the UI only ends the turn when it "
                "receives the stream's final message-completion event, but after a tool "
                "call the backend sends a response-completion event instead, so the UI "
                "waits for an event that never arrives and hangs.",
            ),
            (
                "assistant",
                "I normalized the backend's streaming events to match what "
                "the UI's protocol expects, so the UI receives the completion event it "
                "is waiting for.",
            ),
            (
                "user",
                "I'm still not convinced. The network tab showed some red on "
                "the backend calls. I think this is really a CORS problem between the "
                "frontend and the API and that's what we should chase.",
            ),
        ),
        resolution_signal="Weekly note (different thread): since the streaming "
        "events were normalized, the freeze-after-a-tool-call reports have stopped "
        "coming in entirely.",
        components=(
            Needle(
                fact="",
                probe="Which problem was actually resolved?",
                answer="streaming answers freezing after a tool call",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="What was the final root cause?",
                answer="the UI waited for the stream's message-completion event, "
                "but after a tool call the backend emitted a response-completion "
                "event instead, so the UI hung",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="Which attempted fix was tried but did NOT work?",
                answer="raising the request timeout (the freeze persisted)",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="What change finally resolved it?",
                answer="normalizing the backend's streaming events to match the "
                "UI's expected protocol",
                distractors=(),
            ),
        ),
        trap_claim="the issue was resolved by fixing a CORS problem, or by raising "
        "the request timeout",
    ),
    HardCase(
        key="retrieval",
        thread=(
            (
                "user",
                "After last night's re-ingest, retrieval recall fell off a "
                "cliff. Relevant lessons just stop showing up for queries that used to "
                "work.",
            ),
            (
                "user",
                "I bumped rerank top_k from 5 to 20 to give it more candidates "
                "to work with.",
            ),
            (
                "user",
                "Recall is still bad even at top_k 20. The right chunk simply "
                "isn't anywhere in the candidate set.",
            ),
            (
                "assistant",
                "Checked the vectors: the re-ingest re-embedded documents "
                "with embed-v4.0, but the stored query path still used embed-v3, so "
                "query and document vectors live in different spaces and never match.",
            ),
            (
                "assistant",
                "I re-embedded the whole collection with embed-v4.0 so the "
                "query and document vectors are back in the same space.",
            ),
            (
                "user",
                "I still suspect the reranker. rerank-v4.0-fast looked like it "
                "was demoting the exact matches below generic chunks. Pretty sure "
                "that's the real culprit.",
            ),
        ),
        resolution_signal="From standup (different topic): recall metrics snapped "
        "back to baseline the moment the collection finished re-embedding on v4.",
        components=(
            Needle(
                fact="",
                probe="Which problem was actually resolved?",
                answer="retrieval recall collapsing after the re-ingest",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="What was the final root cause?",
                answer="query and document vectors were embedded with different "
                "model versions (v3 vs v4), so they were in mismatched spaces",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="Which attempted fix was tried but did NOT work?",
                answer="increasing rerank top_k from 5 to 20 (recall stayed bad)",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="What change finally resolved it?",
                answer="re-embedding the whole collection with embed-v4.0 so the "
                "vectors match",
                distractors=(),
            ),
        ),
        trap_claim="the issue was resolved by changing the reranker, or by raising "
        "the rerank top_k",
    ),
    HardCase(
        key="cost",
        thread=(
            (
                "user",
                "Our API bill jumped ~4x overnight with no change in traffic. "
                "Per-turn cost roughly quadrupled.",
            ),
            (
                "user",
                "I switched us to a cheaper model variant to bring the bill down.",
            ),
            (
                "user",
                "Bill is still high on the cheaper variant. Per-turn cost "
                "barely moved, the spike is still there.",
            ),
            (
                "assistant",
                "Found it: we started prepending a per-turn timestamp to "
                "the system prompt. That changes the cached prefix on every single "
                "call, so implicit prompt caching never hits and every token bills at "
                "the full uncached rate.",
            ),
            (
                "assistant",
                "I moved the timestamp out of the cached prefix into a "
                "later message, so the stable prefix caches again.",
            ),
            (
                "user",
                "Honestly I think the provider just raised their prices. We "
                "should plan a migration to a different vendor, that's the real fix.",
            ),
        ),
        resolution_signal="Finance note (separate thread): spend dropped back to "
        "normal the same day the system-prompt timestamp change was reverted.",
        components=(
            Needle(
                fact="",
                probe="Which problem was actually resolved?",
                answer="the ~4x API cost/billing spike",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="What was the final root cause?",
                answer="a per-turn timestamp in the system prompt changed the "
                "cached prefix every call and broke implicit prompt caching, so "
                "tokens billed uncached",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="Which attempted fix was tried but did NOT work?",
                answer="switching to a cheaper model variant (the bill stayed high)",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="What change finally resolved it?",
                answer="removing the timestamp from the cached prefix so the "
                "stable prefix caches again",
                distractors=(),
            ),
        ),
        trap_claim="the issue was resolved by a provider price change, or by "
        "switching models or migrating vendors",
    ),
    HardCase(
        key="memory",
        thread=(
            (
                "user",
                "The tutor keeps forgetting things the student told it earlier "
                "in the same session. By the end of a long chat it has lost the setup "
                "the student gave at the start.",
            ),
            (
                "user",
                "I increased the summary length budget so the running summary "
                "keeps more of the conversation.",
            ),
            (
                "user",
                "Still losing the earliest details even with the bigger "
                "summary. The very first things the student said are just gone.",
            ),
            (
                "assistant",
                "Traced it: summarization fires on a fixed turn count and "
                "drops the oldest turns to make room, but it summarizes the kept window "
                "only, so the earliest turns are evicted before they are ever folded "
                "into the summary.",
            ),
            (
                "assistant",
                "I switched that preset to keep the full history under the "
                "cache instead of summarizing, so the early turns stay in context.",
            ),
            (
                "user",
                "I still think the long-term profile store is just stale. We "
                "should rebuild the embedding memory, that's probably the real issue.",
            ),
        ),
        resolution_signal="From the support channel (separate topic): once those "
        "sessions stopped being summarized, the 'it forgot what I told it' "
        "complaints dried up.",
        components=(
            Needle(
                fact="",
                probe="Which problem was actually resolved?",
                answer="the tutor forgetting facts the student gave earlier in "
                "the same session",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="What was the final root cause?",
                answer="summarization evicted the oldest turns before they were "
                "ever folded into the summary, so the earliest facts were lost",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="Which attempted fix was tried but did NOT work?",
                answer="increasing the summary length budget",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="What change finally resolved it?",
                answer="keeping the full history under the cache instead of "
                "summarizing those sessions",
                distractors=(),
            ),
        ),
        trap_claim="the issue was resolved by rebuilding the long-term profile / "
        "embedding memory store",
    ),
    HardCase(
        key="chunking",
        thread=(
            (
                "user",
                "The tutor cites the right lesson but hands back a broken code "
                "snippet. The prose is correct, the code is subtly wrong.",
            ),
            (
                "user",
                "I raised the retrieval token budget so it pulls in more "
                "context around the match.",
            ),
            (
                "user",
                "Still a broken snippet even with more context pulled. More "
                "tokens did not help at all.",
            ),
            (
                "assistant",
                "Found it: the chunker was splitting in the middle of code "
                "blocks, so a retrieved chunk held only half a function and the model "
                "completed the rest from guesswork.",
            ),
            (
                "assistant",
                "I made chunking heading-aware so it never splits inside a "
                "code block; each chunk now carries a whole snippet.",
            ),
            (
                "user",
                "I think the model just hallucinates code regardless. We should "
                "add a strong warning to the system prompt, that is the real fix.",
            ),
        ),
        resolution_signal="Standup note (unrelated): after the chunker stopped "
        "cutting code blocks in half, the broken-snippet reports went away.",
        components=(
            Needle(
                fact="",
                probe="Which problem was actually resolved?",
                answer="the tutor returning broken code snippets while citing the "
                "right lesson",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="What was the final root cause?",
                answer="the chunker split inside code blocks, so retrieved chunks "
                "held only part of a function",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="Which attempted fix was tried but did NOT work?",
                answer="increasing the retrieval token budget",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="What change finally resolved it?",
                answer="making chunking heading-aware so it never splits inside a "
                "code block",
                distractors=(),
            ),
        ),
        trap_claim="the issue was resolved by adding a warning to the system prompt "
        "(against hallucinated code)",
    ),
    HardCase(
        key="ratelimit",
        thread=(
            (
                "user",
                "Eval runs randomly die partway through with rate-limit errors "
                "during the big batches. Short runs are fine, long ones blow up.",
            ),
            ("user", "I dropped concurrency to a single worker to ease off the API."),
            (
                "user",
                "Still hitting rate limits even single-threaded once a run gets "
                "long. Serial did not save it.",
            ),
            (
                "assistant",
                "Root cause: the client retries immediately and ignores "
                "the Retry-After header, so a single slow window snowballs into a burst "
                "of retries that trips the limit.",
            ),
            (
                "assistant",
                "I added exponential backoff that honors Retry-After, so "
                "the client waits the requested time instead of hammering.",
            ),
            (
                "user",
                "Honestly I think our API tier is just too low. We should "
                "upgrade the plan, that will actually fix it.",
            ),
        ),
        resolution_signal="Note from the eval channel (different thread): since the "
        "backoff went in, the long batches run clean end to end.",
        components=(
            Needle(
                fact="",
                probe="Which problem was actually resolved?",
                answer="eval runs failing with rate-limit errors on long batches",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="What was the final root cause?",
                answer="the client retried immediately and ignored Retry-After, "
                "so retries burst and tripped the rate limit",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="Which attempted fix was tried but did NOT work?",
                answer="reducing concurrency to a single worker",
                distractors=(),
            ),
            Needle(
                fact="",
                probe="What change finally resolved it?",
                answer="adding exponential backoff that honors the Retry-After header",
                distractors=(),
            ),
        ),
        trap_claim="the issue was resolved by upgrading the API plan / tier",
    ),
)

# Diverse "past discussions of very different questions" (Louis's framing). Several
# CLAIM a fix (shipped/resolved/fixed) so the genuinely-resolved issue cannot be
# found by keyword; some of those claimed fixes even regress, so only the target's
# downstream confirmation makes it the uniquely, verifiably resolved one.
HARD_DISTRACTORS: tuple[str, ...] = (
    "Thread #3007 (frontend): Next.js logged a hydration mismatch on the chat "
    "page. Pinned the server timestamp and shipped it; no follow-up to confirm yet.",
    "Thread #3008 (sse): an SSE connection dropped mid-stream on flaky networks. "
    "Added a reconnect, looked fixed, but it dropped again two days later. Reopened.",
    "Thread #3009 (tools): a tool returned malformed JSON and the agent retried "
    "forever. Capped retries at 3; seems fine so far, not closed.",
    "Thread #3010 (embeddings): an old collection broke when dimensions changed. "
    "Rebuild is still pending, no fix yet.",
    "Thread #3011 (auth): the HF token expired and downloads failed. Rotated the "
    "token; assumed fixed, not re-checked.",
    "Thread #3012 (latency): p95 latency rose after enabling web search. Suspect "
    "the fetch tool. Still investigating.",
    "Thread #3013 (memory): summarization dropped a student's earlier detail. "
    "Considering keep-everything; nothing changed yet.",
    "Thread #3014 (ui): the message list jumped to the top on each token. Added a "
    "scroll anchor and shipped it; no complaints yet, but unverified.",
    "Thread #3015 (protocol): a duplicate completion event double-rendered the "
    "final card. Deduped on event id; marked fixed, then a report came back this week.",
    "Thread #3016 (chunking): headings split mid-code-block on one source. "
    "Tightened the splitter; probably fine, not checked.",
    "Thread #3017 (kb): headings.jsonl went stale after a corpus update. Regen is "
    "queued, not run yet.",
    "Thread #3018 (routing): OpenRouter sent the same model id to different quant "
    "backends across days, drifting scores. Noted, no action taken.",
    "Thread #3019 (frontend): a tool-status popover flickered on hover. A CSS race "
    "is suspected. Low priority, open.",
    "Thread #3020 (ingest): a Notion export had broken image links. Added a skip "
    "for missing images and shipped it; not verified in prod.",
    "Thread #3021 (cors): a real CORS error blocked the dev frontend on a new port. "
    "Added the origin to the allowlist; not yet verified.",
    "Thread #3022 (webfetch): a long web_fetch hit its timeout and errored. Raised "
    "the timeout; looked fixed, then it timed out again on a slow upstream. Reopened.",
    "Thread #3023 (judge): the eval judge sometimes returned unparseable JSON. "
    "Added a regex fallback and shipped it; not yet seen across a full run.",
    "Thread #3024 (vectors): create_vector_stores ran out of memory on the full "
    "corpus. Batching the upserts is planned, not done.",
    "Thread #3025 (coldstart): first-token latency felt slow on cold start. Tried "
    "warming the model; it helped a little, still being tuned.",
    "Thread #3026 (citations): two sources merged into one card when URLs differed "
    "only by a fragment. Added the fragment to the dedup key; assume fixed, untested.",
    "Thread #3027 (rate-limit): Cohere rerank hit a rate limit during a big eval. "
    "Added backoff, seemed fine, then tripped again under load. Reopened.",
    "Thread #3028 (notebooks): a course notebook failed on a pinned package. Bumped "
    "the pin; one clean run, still watching it.",
    "Thread #3029 (prompt): the system prompt leaked a source path into an answer. "
    "Masked it in post-processing; presumed fixed, not confirmed.",
    "Thread #3030 (i18n): non-ASCII source titles rendered as mojibake in a card. "
    "Forced UTF-8 on the response; looks right locally, not confirmed in prod.",
)

HARD_SYSTEM = (
    "You are reviewing a long support history covering many separate, unrelated "
    "issues. Some are still open, and several CLAIM to be fixed, but a claim of a "
    "fix is not proof: only treat an issue as resolved if its fix is later "
    "confirmed to have actually worked. Answer concisely and numbered. If "
    "something is genuinely not stated, say you cannot find it."
)

HARD_QUESTION = (
    "Across the entire history above, many issues are discussed and several claim a "
    "fix, but only ONE issue is actually, verifiably resolved (its fix is later "
    "confirmed to have worked). For that one resolved issue, answer, numbered:\n"
    "1) Which problem was it?\n"
    "2) What was the final root cause?\n"
    "3) Which attempted fix was tried but did NOT actually work?\n"
    "4) What change finally resolved it?\n"
    "Be concise. If something is genuinely not stated, say you cannot find it."
)


# ----------------------------------------------------------------- context build
# An "item" is (frac, role, kind, text): kind in {"beat","aside","distractor"};
# role is the chat role for conversation mode. Filler prose is sliced in between.


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _hard_items(case: HardCase, distractors: list[str], rng: random.Random) -> list:
    """Lay out one hard case: thread beats at jittered positions, the resolution
    aside near the end, and the shuffled distractors spread across the window."""
    items: list[tuple[float, str, str, str]] = []
    for (role, text), base in zip(case.thread, THREAD_POSITIONS):
        items.append(
            (_clamp(base + rng.uniform(-0.03, 0.03), 0.02, 0.93), role, "beat", text)
        )
    items.append((RESOLUTION_POSITION, "assistant", "aside", case.resolution_signal))
    m = len(distractors)
    for i, d in enumerate(distractors):
        frac = 0.03 + 0.94 * (i / max(1, m - 1))
        items.append(
            (
                _clamp(frac + rng.uniform(-0.02, 0.02), 0.0, 0.99),
                "user",
                "distractor",
                d,
            )
        )
    items.sort(key=lambda x: x[0])
    return items


def _easy_items() -> list:
    """The original layout: beats at 0.10..0.90, distractors evenly spread, all as
    bare inserted text (no speaker labels) -- reproduces the 4/4 baseline."""
    items: list[tuple[float, str, str, str]] = []
    for frac, beat in zip((0.10, 0.30, 0.50, 0.70, 0.90), EASY_TARGET_BEATS):
        items.append((frac, "user", "distractor", beat))
    m = len(EASY_DISTRACTOR_THREADS)
    for i, d in enumerate(EASY_DISTRACTOR_THREADS):
        items.append((0.03 + 0.94 * (i / max(1, m - 1)), "user", "distractor", d))
    items.sort(key=lambda x: x[0])
    return items


def _parts(enc, filler_tokens, fill_tokens, items) -> list:
    """Interleave filler-prose slices with the placed items, in order. Returns a
    list of (kind, role, text); kind 'filler' for prose."""
    body = filler_tokens[:fill_tokens]
    n = len(body)
    parts: list[tuple[str, str, str]] = []
    prev = 0
    for frac, role, kind, text in items:
        idx = int(_clamp(frac, 0.0, 1.0) * n)
        idx = max(prev, min(n, idx))
        parts.append(("filler", "user", enc.decode(body[prev:idx])))
        parts.append((kind, role, text))
        prev = idx
    parts.append(("filler", "user", enc.decode(body[prev:])))
    return parts


def _render_document(parts, label_speakers: bool) -> str:
    out: list[str] = []
    for kind, role, text in parts:
        if kind == "beat" and label_speakers:
            out.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")
        else:
            out.append(text)
    return "\n\n".join(p for p in out if p.strip())


def _render_conversation(parts, question: str, system: str) -> list[dict]:
    """A role-faithful conversation: planted beats keep their real role, filler is
    delivered as user turns, and consecutive same-role messages are coalesced so
    the sequence alternates (provider-safe). Ends on the user question."""
    seq: list[tuple[str, str]] = []
    for kind, role, text in parts:
        if not text.strip():
            continue
        seq.append(("user" if kind == "filler" else role, text))
    coalesced: list[list[str]] = []
    for role, text in seq:
        if coalesced and coalesced[-1][0] == role:
            coalesced[-1][1] += "\n\n" + text
        else:
            coalesced.append([role, text])
    msgs = [{"role": "system", "content": system}]
    for role, text in coalesced:
        msgs.append({"role": role, "content": text})
    if msgs[-1]["role"] == "user":
        msgs[-1]["content"] += "\n\n" + question
    else:
        msgs.append({"role": "user", "content": question})
    return msgs


def answer_state(parts, cfg, fmt, question, system, max_tokens):
    """retrieval OFF: the model must read the whole history to answer. fmt=
    'document' = one stuffed user message (log-shaped); fmt='conversation' =
    role-faithful multi-turn chat."""
    label = any(kind == "beat" for kind, _, _ in parts)  # hard mode labels speakers
    if fmt == "conversation":
        return complete(
            _render_conversation(parts, question, system),
            max_tokens=max_tokens,
            cfg=cfg,
        )
    return complete(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Support history:\n{_render_document(parts, label)}\n\n{question}",
            },
        ],
        max_tokens=max_tokens,
        cfg=cfg,
    )


_TRAP_RE = re.compile(r'"pass"\s*:\s*(true|false)', re.IGNORECASE)


def trap_check(ans, trap_claim, judge_cfg) -> bool:
    """Did the answer credit an ABANDONED hypothesis as the actual fix? That is the
    failure mode the hard probe most wants to catch."""
    text, *_ = complete(
        [
            {
                "role": "system",
                "content": "You check whether an ANSWER makes a "
                'specific WRONG claim. Output ONLY JSON: {"pass": true/false}.',
            },
            {
                "role": "user",
                "content": f"Answer pass=true ONLY if the ANSWER claims "
                f"that {trap_claim}. Those were abandoned hypotheses, not the real "
                f"fix.\n\nANSWER: {ans}",
            },
        ],
        max_tokens=100,
        cfg=judge_cfg,
    )
    m = _TRAP_RE.search(text)
    return bool(m and m.group(1).lower() == "true")


@dataclass
class StateCell:
    mode: str
    case_key: str
    fill_tokens: int
    context_tokens: int
    trial: int
    components_recovered: int  # 0..4
    n_components: int
    per_component: list
    trap_triggered: bool
    input_tokens: int
    output_tokens: int
    latency_s: float
    truncated: bool  # hit the output cap or empty -> INVALID, excluded from mean
    answer_text: str
    format: str = "document"


def run_cell(
    enc,
    filler,
    fill_tokens,
    trial,
    cfg,
    judge_cfg,
    bundle,
    *,
    mode,
    fmt,
    max_tokens,
    seed,
):
    if mode == "hard":
        rng = random.Random(seed * 1000 + trial)
        case = HARD_CASES[trial % len(HARD_CASES)]
        distractors = list(HARD_DISTRACTORS)
        rng.shuffle(distractors)
        distractors = distractors[:22]
        items = _hard_items(case, distractors, rng)
        components, trap_claim = case.components, case.trap_claim
        question, system, case_key = HARD_QUESTION, HARD_SYSTEM, case.key
    else:
        items = _easy_items()
        components, trap_claim = EASY_COMPONENTS, EASY_TRAP_CLAIM
        question = EASY_QUESTION
        system = (
            "You are an assistant answering from the provided debugging "
            "history ONLY. Answer concisely and numbered. If something is "
            "truly not present, say you cannot find it."
        )
        case_key = "streaming"

    parts = _parts(enc, filler, fill_tokens, items)
    ctx_tok = len(enc.encode(_render_document(parts, True), disallowed_special=()))

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            ans, in_tok, out_tok, lat = answer_state(
                parts, cfg, fmt, question, system, max_tokens
            )
            break
        except Exception as e:  # noqa: BLE001 - transient provider/network errors
            last_err = e
            logger.warning(
                "fill=%dk t%d attempt %d failed: %s",
                fill_tokens // 1000,
                trial,
                attempt + 1,
                str(e)[:120],
            )
            time.sleep(5 * (attempt + 1))
    else:
        logger.error(
            "fill=%dk t%d SKIPPED after retries: %s",
            fill_tokens // 1000,
            trial,
            str(last_err)[:120],
        )
        return None

    truncated = out_tok >= max_tokens or not ans.strip()
    per_component = []
    recovered = 0
    for comp in components:
        jp, jr = judge(comp.probe, ans, comp, judge_cfg)
        per_component.append({"probe": comp.probe, "pass": jp, "reason": jr})
        recovered += int(jp)
    trap = trap_check(ans, trap_claim, judge_cfg)
    cell = StateCell(
        mode=mode,
        case_key=case_key,
        fill_tokens=fill_tokens,
        context_tokens=ctx_tok,
        trial=trial,
        components_recovered=recovered,
        n_components=len(components),
        per_component=per_component,
        trap_triggered=trap,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_s=lat,
        truncated=truncated,
        answer_text=ans[:2000],
        format=fmt,
    )
    with open(bundle, "a", encoding="utf-8") as h:
        h.write(json.dumps(asdict(cell), ensure_ascii=False) + "\n")
    logger.info(
        "[%s/%s] fill=%dk t%d -> %d/%d%s (ctx %dk, out=%d, %.1fs)%s",
        mode,
        case_key,
        fill_tokens // 1000,
        trial,
        recovered,
        len(components),
        " TRAP" if trap else "",
        ctx_tok // 1000,
        out_tok,
        lat,
        " TRUNCATED(INVALID)" if truncated else "",
    )
    return cell


def report(cells, fills, cfg, judge_cfg, mode):
    lines = [
        "",
        f"# State-tracking context-rot ({mode} mode, retrieval OFF)",
        "",
        f"Model: {cfg.model} | judge {judge_cfg.model} | format: "
        f"{cells[0].format if cells else '?'} | 4 components: which issue, root "
        "cause, rejected fix, final fix",
        (
            f"{len({c.case_key for c in cells})} scenarios x {len(cells)} randomized "
            "layouts (jittered positions + shuffled distractors): a trial is a "
            "layout, not an independent bug."
            if mode == "hard"
            else ""
        ),
        "Truncated cells (output cap hit / empty) are INVALID and excluded from the "
        "mean.",
        "",
        "| fill (ctx) | n valid | mean comp /4 | trap rate | truncated | per-trial |",
        "|---|---|---|---|---|---|",
    ]
    for ft in fills:
        cc = [c for c in cells if c.fill_tokens == ft]
        vc = [c for c in cc if not c.truncated]
        if not cc:
            continue
        ctxk = int(mean(c.context_tokens for c in cc)) // 1000
        mc = f"{mean(c.components_recovered for c in vc):.2f}" if vc else "-"
        trap = f"{mean(int(c.trap_triggered) for c in vc):.0%}" if vc else "-"
        ntr = sum(1 for c in cc if c.truncated)
        per = ", ".join(f"{c.components_recovered}/4" for c in vc)
        lines.append(
            f"| {ft // 1000}k ({ctxk}k) | {len(vc)} | {mc} | {trap} | {ntr} | {per} |"
        )
    comps = HARD_CASES[0].components if mode == "hard" else EASY_COMPONENTS
    lines += ["", "Per-component recovery by length (valid cells only):"]
    for i, comp in enumerate(comps):
        row = []
        for ft in fills:
            vc = [c for c in cells if c.fill_tokens == ft and not c.truncated]
            if vc:
                pr = mean(int(c.per_component[i]["pass"]) for c in vc)
                row.append(f"{ft // 1000}k={pr:.0%}")
        lines.append(f"  - {comp.probe} " + "  ".join(row))
    if mode == "hard":
        lines += [
            "",
            "Rot signature: mean components falls and/or trap rate climbs "
            "as fill grows. Flat-and-high = reasoning survives the length.",
        ]
    return "\n".join(lines) + "\n"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["easy", "hard"],
        default="hard",
        help="easy = original NIAH-ish control (reproduces 4/4); "
        "hard = the reasoning probe (no beacon, late trap, inferred state).",
    )
    p.add_argument("--provider", choices=sorted(PROVIDERS), default="deepseek_or")
    p.add_argument("--model", help="Override the litellm model id.")
    p.add_argument("--api-base")
    p.add_argument("--num-ctx", type=int, help="Ollama context window.")
    p.add_argument("--judge-provider", choices=sorted(PROVIDERS), default="gemini")
    p.add_argument("--fills", type=int, nargs="*", default=[8000, 100000, 400000])
    p.add_argument(
        "--trials",
        type=int,
        default=6,
        help="Independent trials per fill: each draws a different case + "
        "jittered positions + shuffled distractors (hard mode).",
    )
    p.add_argument(
        "--seed", type=int, default=7, help="RNG seed (reproducible layouts)."
    )
    p.add_argument("--answer-max-tokens", type=int, default=ANSWER_MAX_TOKENS)
    p.add_argument(
        "--no-think",
        action="store_true",
        help="Disable a hybrid model's thinking (reasoning_effort=none) "
        "to keep verbose reasoning from blowing the output cap at length.",
    )
    p.add_argument(
        "--format",
        choices=["document", "conversation"],
        default="conversation",
        help="conversation (default) = role-faithful multi-turn chat (the "
        "tutor's real shape); document = one stuffed user message "
        "(the stuffed-log framing).",
    )
    p.add_argument("--out", default="data/context_rot_state_hard")
    args = p.parse_args()

    cfg = PROVIDERS[args.provider]
    if args.model:
        cfg = replace(cfg, model=args.model)
    if args.api_base is not None:
        cfg = replace(cfg, api_base=args.api_base)
    if args.num_ctx is not None:
        cfg = replace(cfg, num_ctx=args.num_ctx)
    if args.no_think:
        cfg = replace(cfg, reasoning_effort="none")
    judge_cfg = PROVIDERS[args.judge_provider]
    fills, trials = args.fills, args.trials

    from app.chroma_rag import get_token_encoding

    enc = get_token_encoding()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "bundles.jsonl"
    bundle.write_text("")

    logger.info(
        "mode=%s model=%s judge=%s | fills=%s trials=%d seed=%d fmt=%s",
        args.mode,
        cfg.model,
        judge_cfg.model,
        fills,
        trials,
        args.seed,
        args.format,
    )
    filler = load_filler(enc, max(fills))
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "model": cfg.model,
                "judge": judge_cfg.model,
                "n_distractors": len(
                    HARD_DISTRACTORS if args.mode == "hard" else EASY_DISTRACTOR_THREADS
                ),
                "n_cases": len(HARD_CASES) if args.mode == "hard" else 1,
                "fills": fills,
                "trials": trials,
                "seed": args.seed,
                "answer_max_tokens": args.answer_max_tokens,
                "format": args.format,
            },
            indent=2,
        )
    )

    cells = []
    for ft in fills:
        for t in range(trials):
            c = run_cell(
                enc,
                filler,
                ft,
                t,
                cfg,
                judge_cfg,
                bundle,
                mode=args.mode,
                fmt=args.format,
                max_tokens=args.answer_max_tokens,
                seed=args.seed,
            )
            if c:
                cells.append(c)

    out = report(cells, fills, cfg, judge_cfg, args.mode)
    (out_dir / "report.md").write_text(out)
    print(out)
    print(f"Bundles: {bundle} | {len(cells)} cells")


if __name__ == "__main__":
    main()
