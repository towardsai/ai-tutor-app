"""Memory/context-management presets.

A preset bundles every knob that changes how the agent manages conversation
context: the compaction middlewares (summarization, tool-output clearing) and
long-term student-profile memory. ``build_agent()`` assembles its middleware
stack from the active preset, so experiment runs can compare configurations by
name while the API selects a model-compatible production preset.

Selection order: explicit request value > ``AI_TUTOR_MEMORY_PRESET`` env var >
the production preset compatible with the requested model. Unknown names and
incompatible selections from either override source raise instead of falling
back; a mislabeled experiment run is worse than a failed one.
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

DELTA_SUMMARY_PROMPT = """<role>
Running-summary updater (delta compaction)
</role>

The messages below contain the running summary so far (if one exists) followed by
newer turns. Produce an UPDATED running summary: start from the existing summary
and fold in only what the newer turns add or change. Keep the durable facts,
decisions, and lesson content the student is asking about; do not drop anything
important that was already summarized; stay concise and non-redundant.

Respond ONLY with the updated running summary.

<messages>
{messages}
</messages>"""


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    name: str
    summarization: bool = True
    summarization_trigger_tokens: int = 30_000
    summarization_keep_messages: int = 20
    # Experiment arms use token-based retention so a single large tool message
    # cannot make the post-compaction window vary by hundreds of thousands of
    # tokens. None preserves the historical message-count behavior.
    summarization_keep_tokens: int | None = None
    # LangChain defaults this to 4k. None deliberately sends the entire selected
    # older history to the summarizer (the corrected long-context experiment).
    summarization_trim_tokens: int | None = 4_000
    # Fail rather than silently trim if a full-input experimental summary would
    # approach the provider's context ceiling. None disables the guard.
    summarization_input_guard_tokens: int | None = None
    # Custom SummarizationMiddleware prompt (None = the library default). Used by
    # the selective_retention / context_reset arms; must template {messages}.
    summary_prompt: str | None = None
    # ``xml`` is LangChain's historical behavior: serialize selected messages
    # into one new prompt string. ``structured_prefix`` keeps the original
    # system message, tool schemas, model settings, and selected message prefix
    # byte-for-byte at the message boundary, then appends one checkpoint
    # instruction. The latter is experiment-only because it changes request
    # shape and checkpoint installation semantics.
    summarization_strategy: str = "xml"
    context_editing: bool = True
    context_editing_trigger_tokens: int = 5_000
    context_editing_keep: int = 5
    # When True (legacy prod default), ClearToolUsesEdit leaves retrieval results alone
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
    # Persistent insertion-time cap: unlike truncate_tool_outputs, this changes
    # the checkpoint itself, so every later model call and the summarizer see the
    # same stable text. The experiment uses 40k UTF-8 bytes as a nominal 10k-token
    # cap, matching the reproducible approximation used by the Codex harness.
    tool_output_cap_bytes: int | None = None
    # Enables explanatory compaction telemetry and DeepSeek user_id isolation.
    # Kept off for production and historical presets so this study is additive.
    experiment_mode: bool = False
    # Fail before an experimental agent call exceeds this approximate input
    # size. The approximation over-counted the provider-reported input by about
    # 29k near 870k, so 990k preserves real headroom inside DeepSeek's 1M window
    # without prematurely truncating the full-history control.
    experiment_request_guard_tokens: int = 990_000
    compress_prompt: bool = False  # deterministic per-call text compaction
    # In-context history retrieval (Axis A subsystem): keep the last N turn-blocks
    # and retrieve only the top-k most relevant older blocks. None disables it.
    history_retrieval_keep_recent: int | None = None
    history_retrieval_top_k: int = 3
    # Hierarchical summarization (Axis A): map-reduce the older messages
    # (summarize groups, then summarize the group summaries) into one summary in
    # the model's view, cached by content. None/False disables it.
    hierarchical_summarize: bool = False
    hierarchical_trigger_tokens: int = 8_000
    hierarchical_keep_recent: int = 6
    hierarchical_group_size: int = 5


MEMORY_PRESETS: dict[str, MemoryConfig] = {
    # No compaction at all: the quality/memory upper bound and the
    # token-cost worst case.
    "full_history": MemoryConfig(
        name="full_history", summarization=False, context_editing=False
    ),
    # --- DeepSeek long-context compaction experiment ----------------------
    # Four mechanism-isolation arms. All disable age-based context editing;
    # the capped arms instead perform one stable rewrite when tool output first
    # enters history. C200 arms summarize the complete selected prefix at 200k
    # and retain a controlled 50k-token recent tail.
    "exp_fh_raw": MemoryConfig(
        name="exp_fh_raw",
        summarization=False,
        context_editing=False,
        experiment_mode=True,
    ),
    "exp_fh_cap10k": MemoryConfig(
        name="exp_fh_cap10k",
        summarization=False,
        context_editing=False,
        tool_output_cap_bytes=40_000,
        experiment_mode=True,
    ),
    "exp_c200_raw": MemoryConfig(
        name="exp_c200_raw",
        summarization_trigger_tokens=200_000,
        summarization_keep_tokens=50_000,
        summarization_trim_tokens=None,
        summarization_input_guard_tokens=900_000,
        context_editing=False,
        experiment_mode=True,
    ),
    "exp_c200_cap10k": MemoryConfig(
        name="exp_c200_cap10k",
        summarization_trigger_tokens=200_000,
        summarization_keep_tokens=50_000,
        summarization_trim_tokens=None,
        summarization_input_guard_tokens=900_000,
        context_editing=False,
        tool_output_cap_bytes=40_000,
        experiment_mode=True,
    ),
    # Cache-friendly version of exp_c200_cap10k. It deliberately remains a
    # separate arm so the completed XML run stays reproducible and comparable.
    "exp_c200_cap10k_structured": MemoryConfig(
        name="exp_c200_cap10k_structured",
        summarization_trigger_tokens=200_000,
        summarization_keep_tokens=50_000,
        summarization_trim_tokens=None,
        summarization_input_guard_tokens=900_000,
        summarization_strategy="structured_prefix",
        context_editing=False,
        tool_output_cap_bytes=40_000,
        experiment_mode=True,
    ),
    # Stage-2 threshold sensitivity arms share the exact same cap, summary
    # input, and post-compaction retention. Only the trigger changes.
    "exp_c400_cap10k": MemoryConfig(
        name="exp_c400_cap10k",
        summarization_trigger_tokens=400_000,
        summarization_keep_tokens=50_000,
        summarization_trim_tokens=None,
        summarization_input_guard_tokens=900_000,
        context_editing=False,
        tool_output_cap_bytes=40_000,
        experiment_mode=True,
    ),
    "exp_c800_cap10k": MemoryConfig(
        name="exp_c800_cap10k",
        summarization_trigger_tokens=800_000,
        summarization_keep_tokens=50_000,
        summarization_trim_tokens=None,
        summarization_input_guard_tokens=900_000,
        context_editing=False,
        tool_output_cap_bytes=40_000,
        experiment_mode=True,
    ),
    # Historical production baseline. Keep this immutable: existing eval
    # findings and saved run labels named "prod" refer to these exact settings.
    "prod": MemoryConfig(name="prod"),
    # Current long-context production policy. This is the structured-prefix
    # stage-1 arm with its trigger moved from 200k to 800k; every other setting
    # stays identical so the summary call remains a strict cache-prefix
    # extension and tool outputs are capped once at insertion time.
    "prod_v2": MemoryConfig(
        name="prod_v2",
        summarization_trigger_tokens=800_000,
        summarization_keep_tokens=50_000,
        summarization_trim_tokens=None,
        summarization_input_guard_tokens=900_000,
        summarization_strategy="structured_prefix",
        context_editing=False,
        tool_output_cap_bytes=40_000,
        experiment_mode=True,
    ),
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
    # Delta summarization: a single running summary updated each trigger with only
    # what changed (the prompt folds the prior summary + new turns into a fresh
    # running summary). Summarization-style arm; isolates the summary STRATEGY vs
    # prod/selective. Watch the F2 cache confound (prefix rewrite each trigger).
    "delta_summarization": MemoryConfig(
        name="delta_summarization",
        context_editing=False,
        summary_prompt=DELTA_SUMMARY_PROMPT,
    ),
    # Hierarchical summarization: map-reduce the older messages (summarize groups,
    # then summarize the summaries) into one layered summary. Expected to preserve
    # more structure than single-pass on long content; its extra summarization
    # LLM calls are the cost it must justify. Summarization off (this replaces it).
    "hierarchical_summarization": MemoryConfig(
        name="hierarchical_summarization",
        summarization=False,
        hierarchical_summarize=True,
    ),
}

# One switch controls the long-context production policy. Providers outside the
# allowlist stay on the historical, conservative preset until they have a
# provider-appropriate long-context configuration (Claude Haiku 4.5, for
# example, has a 200k input window and cannot safely wait for an 800k trigger).
PRODUCTION_MEMORY_PRESET = "prod_v2"
PRODUCTION_FALLBACK_MEMORY_PRESET = "prod"
PRODUCTION_LONG_CONTEXT_PROVIDERS = frozenset({"deepseek", "google-genai"})

# Backward-compatible import for callers that need a single default name. New
# runtime code should call resolve_memory_preset(..., model_name=...) so model
# compatibility is applied.
DEFAULT_MEMORY_PRESET = PRODUCTION_MEMORY_PRESET

PRESET_PROVIDER_ALLOWLIST: dict[str, frozenset[str]] = {
    "prod_v2": PRODUCTION_LONG_CONTEXT_PROVIDERS,
}


def _model_provider(model_name: str) -> str:
    normalized = (model_name or "").strip()
    if ":" in normalized:
        return normalized.partition(":")[0]
    if normalized.startswith("gpt-"):
        return "openai"
    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith("gemini"):
        return "google-genai"
    if normalized.startswith("deepseek"):
        return "deepseek"
    return ""


def production_memory_preset_name(model_name: str = "") -> str:
    """Return the production preset compatible with ``model_name``.

    An empty model means the application's default model, currently DeepSeek,
    so it resolves to the primary production preset.
    """
    provider = _model_provider(model_name)
    if not provider or provider in PRODUCTION_LONG_CONTEXT_PROVIDERS:
        return PRODUCTION_MEMORY_PRESET
    return PRODUCTION_FALLBACK_MEMORY_PRESET


def memory_preset_supports_model(preset_name: str, model_name: str) -> bool:
    """Whether a selected preset supports the requested provider."""
    allowed = PRESET_PROVIDER_ALLOWLIST.get(preset_name)
    if allowed is None or not model_name:
        return True
    return _model_provider(model_name) in allowed


def resolve_memory_preset(
    name: str | None = None, *, model_name: str = ""
) -> MemoryConfig:
    explicit = (name or "").strip()
    environment = os.environ.get("AI_TUTOR_MEMORY_PRESET", "").strip()
    requested = explicit or environment or production_memory_preset_name(model_name)
    config = MEMORY_PRESETS.get(requested)
    if config is None:
        known = ", ".join(sorted(MEMORY_PRESETS))
        raise ValueError(f"Unknown memory preset {requested!r}. Known presets: {known}")
    if not memory_preset_supports_model(requested, model_name):
        provider = _model_provider(model_name) or "unknown"
        allowed = ", ".join(sorted(PRESET_PROVIDER_ALLOWLIST[requested]))
        raise ValueError(
            f"Memory preset {requested!r} does not support provider {provider!r}; "
            f"supported providers: {allowed}"
        )
    return config
