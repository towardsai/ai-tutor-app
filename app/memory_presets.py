"""Memory/context-management presets.

A preset bundles every knob that changes how the agent manages conversation
context: the compaction middlewares (summarization, tool-output clearing) and
long-term student-profile memory. ``build_agent()`` assembles its middleware
stack from the active preset, so experiment runs can compare configurations by
name while the API default stays on ``prod``.

Selection order: explicit request value > ``AI_TUTOR_MEMORY_PRESET`` env var >
``DEFAULT_MEMORY_PRESET``. Unknown names raise instead of falling back; a
mislabeled experiment run is worse than a failed one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --- Custom summarization prompts (Part C / Axis A) --------------------------
# SummarizationMiddleware formats these with the older messages via
# ``.format(messages=...)``, so they must contain ``{messages}`` and no other
# bare braces. The default prompt is agentic-task flavored ("artifacts", "next
# steps"); these are tutoring-session flavored, aimed at the facts the session
# probes actually test (student facts, preferences, the current thread).

SELECTIVE_RETENTION_SUMMARY_PROMPT = """<role>
Tutoring-session memory extractor
</role>

You are compacting an AI tutor's conversation with a student. Your summary will
REPLACE the older messages below, so anything you omit is forgotten. Preserve the
durable facts a good tutor must not lose; drop chit-chat and resolved tangents.

Fill each section, or write "None":

## STUDENT FACTS
Stable facts the student stated about themselves: level, goal, operating system,
tools/frameworks in use, language preference, and any hard constraint (for
example "no GPU", "must use conda"). If the student changed a fact mid-session,
record the CURRENT value only.

## PREFERENCES AND DECISIONS
Preferences the tutor agreed to honor, and any recommendation or approach already
chosen, so later answers stay consistent with them.

## OPEN THREAD
What the student is working on right now and what was just discussed, in enough
detail to resolve a later "that thing from earlier".

Respond ONLY with the extracted context.

<messages>
{messages}
</messages>"""

CONTEXT_RESET_SUMMARY_PROMPT = """<role>
Session state snapshot
</role>

The conversation below will be discarded and replaced by your snapshot (a context
reset). Write the minimal state needed to keep helping this student without
re-asking: their level, goal, environment, and constraints (current values), the
approach currently in progress, and the immediate question. Be brief; omit
everything else.

Respond ONLY with the snapshot.

<messages>
{messages}
</messages>"""


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    name: str
    summarization: bool = True
    summarization_trigger_tokens: int = 30_000
    summarization_keep_messages: int = 20
    # Custom SummarizationMiddleware prompt (None = the library default). Used by
    # the selective_retention / context_reset arms; must template {messages}.
    summary_prompt: str | None = None
    context_editing: bool = True
    context_editing_trigger_tokens: int = 5_000
    context_editing_keep: int = 5
    # When True (prod default), ClearToolUsesEdit leaves retrieval results alone
    # (F3: that is where the tokens are, so clearing rarely fires). False makes
    # the clear_retrieval_kb variant clear retrieval + KB outputs too.
    clear_excludes_retrieval: bool = True
    longterm_memory: bool = False
    # --- Part C / Axis A per-call-view mechanisms --------------------------
    # These reshape only the message list sent to the model (not the checkpoint),
    # so they report via app.telemetry's turn-signal registry rather than a
    # checkpoint marker. Each is a single-axis arm; see the preset notes below.
    sliding_window_keep: int | None = None  # keep last N messages, drop older
    truncate_tool_outputs: bool = False  # head/tail-truncate large tool outputs
    truncate_head_chars: int = 2_000
    truncate_tail_chars: int = 500
    truncate_trigger_chars: int = 4_000
    compress_prompt: bool = False  # deterministic per-call text compaction
    # In-context history retrieval (Axis A subsystem): keep the last N turn-blocks
    # and retrieve only the top-k most relevant older blocks. None disables it.
    history_retrieval_keep_recent: int | None = None
    history_retrieval_top_k: int = 3


MEMORY_PRESETS: dict[str, MemoryConfig] = {
    # No compaction at all: the quality/memory upper bound and the
    # token-cost worst case.
    "full_history": MemoryConfig(
        name="full_history", summarization=False, context_editing=False
    ),
    # What production runs today.
    "prod": MemoryConfig(name="prod"),
    "summarization_only": MemoryConfig(
        name="summarization_only", context_editing=False
    ),
    "editing_only": MemoryConfig(name="editing_only", summarization=False),
    # How bad can cheap get: compaction fires early and keeps little.
    "aggressive": MemoryConfig(
        name="aggressive",
        summarization_trigger_tokens=8_000,
        summarization_keep_messages=8,
        context_editing_trigger_tokens=2_000,
        context_editing_keep=2,
    ),
    # prod compaction + long-term semantic memory (student profile store).
    "profile_memory": MemoryConfig(name="profile_memory", longterm_memory=True),
    # Part C / Axis B (F3): prod, but ClearToolUsesEdit also clears retrieval + KB
    # outputs (where the tokens are), so context editing actually fires.
    "clear_retrieval_kb": MemoryConfig(
        name="clear_retrieval_kb", clear_excludes_retrieval=False
    ),
    # --- Part C / Axis B: retrieval & tool outputs -------------------------
    # prod + head/tail truncation of large tool outputs (incl. KB). Tests F1
    # (tokens live in tool outputs) by trimming the dominant source while
    # keeping the gist + any citation. Summarization/clearing stay on (prod).
    "observation_truncation": MemoryConfig(
        name="observation_truncation", truncate_tool_outputs=True
    ),
    # --- Part C / Axis A: memory & context management ----------------------
    # Each Axis-A arm isolates ONE history-compaction method. The two
    # "alternative to summarization" arms (sliding_window, prompt_compression)
    # turn summarization OFF and keep prod's tool-output clearing ON, so the only
    # change vs prod is how chat history is reduced. The two summarization-style
    # arms (selective_retention, context_reset) keep prod's stack and change only
    # the summary prompt (+ keep/trigger for the reset).
    #
    # Recency-only memory: keep the last N messages, drop older ones from the
    # model's view (no LLM summary). Cuts on a user-turn boundary to avoid
    # orphaning tool results. Expected to be cheap but to drop planted facts
    # the probes need (F10).
    "sliding_window": MemoryConfig(
        name="sliding_window",
        summarization=False,
        sliding_window_keep=12,
    ),
    # Deterministic per-call text compaction instead of summarization. A cheap,
    # model-free stand-in for prompt compression: it screens the cost/cache
    # effect of rewriting the prompt prefix each turn (F2) without an extra LLM
    # call. Not a faithful LLMLingua-style compressor.
    "prompt_compression": MemoryConfig(
        name="prompt_compression",
        summarization=False,
        compress_prompt=True,
    ),
    # Quality-preserving compaction: prod, but the summary prompt is told to keep
    # the student facts/preferences/decisions the probes test. Isolates "what the
    # summary preserves" from prod's generic summary.
    "selective_retention": MemoryConfig(
        name="selective_retention",
        summary_prompt=SELECTIVE_RETENTION_SUMMARY_PROMPT,
    ),
    # Aggressive context reset seeded with a minimal state snapshot: summarize
    # early and keep few recent messages. A prefix rewrite, so it shares the F2
    # cache confound; the report shows tokens AND dollars to expose it.
    "context_reset": MemoryConfig(
        name="context_reset",
        summary_prompt=CONTEXT_RESET_SUMMARY_PROMPT,
        summarization_trigger_tokens=15_000,
        summarization_keep_messages=4,
    ),
    # The principled answer to F9 (Axis A subsystem): instead of carrying or
    # summarizing all history, keep the last 2 turn-blocks and retrieve only the
    # top-3 most relevant older blocks for the current question. Summarization
    # off (retrieval replaces it); prod clearing stays on.
    "incontext_history_retrieval": MemoryConfig(
        name="incontext_history_retrieval",
        summarization=False,
        history_retrieval_keep_recent=2,
        history_retrieval_top_k=3,
    ),
}

DEFAULT_MEMORY_PRESET = "prod"


def resolve_memory_preset(name: str | None = None) -> MemoryConfig:
    requested = (
        (name or "").strip()
        or os.environ.get("AI_TUTOR_MEMORY_PRESET", "").strip()
        or DEFAULT_MEMORY_PRESET
    )
    config = MEMORY_PRESETS.get(requested)
    if config is None:
        known = ", ".join(sorted(MEMORY_PRESETS))
        raise ValueError(f"Unknown memory preset {requested!r}. Known presets: {known}")
    return config
