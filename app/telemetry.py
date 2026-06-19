"""Per-turn token/cost/latency telemetry, independent of LangSmith.

Token counts come from the ``usage_metadata`` every chat-model call reports,
collected by a callback handler attached to the agent run. That includes
SummarizationMiddleware's internal summary calls and the student-profile
update call, so a preset's overhead is part of its own bill. Nothing here
talks to LangSmith: eval runs can stream thousands of turns with tracing
disabled and still get complete numbers.

Costs are estimates from the local price table below. Raw token counts are
always emitted alongside, so costs can be recomputed offline whenever prices
change; an unknown model yields ``est_cost_usd=None`` rather than a wrong
number.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import LLMResult


# --- Turn-scoped middleware signals ------------------------------------------
# context_window_stats() can only report markers that survive into the
# CHECKPOINTED state (summarization tags, cleared-output placeholders). A
# middleware that only reshapes the per-call view (sliding window, observation
# truncation, prompt compression, in-context history retrieval, ...) leaves the
# checkpoint untouched, so it would be invisible both to telemetry and to the
# probe gate (which would then false-fail: "memory eval where compaction never
# fired"). Such middlewares instead tally here, keyed by the turn's message_id,
# and stream_chat merges the tally into the context_stats event for that turn.
#
# A module-level dict (not a ContextVar) on purpose: LangChain may run sync
# wrap_model_call in a worker thread, where a ContextVar update lands on a copy
# and is lost. A plain global + lock is visible from any thread.
_TURN_SIGNALS: dict[str, dict[str, int]] = {}
_TURN_SIGNALS_LOCK = threading.Lock()
_MAX_TRACKED_TURNS = 256

# Per-call-view turn signals that mean "a context-compaction mechanism fired
# this turn". The eval gate mirrors this set in evals/common.py
# (COMPACTION_SIGNAL_KEYS, plus the two checkpoint-only markers) because the
# harness must not import app code (so old bundles re-grade forever); a unit test
# enforces the two stay in sync. List ONLY names a middleware actually emits.
# (selective_retention / context_reset are SummarizationMiddleware variants, so
# they gate via the summary_messages checkpoint marker and need no entry here.)
COMPACTION_SIGNAL_NAMES = (
    "dropped_messages",  # sliding_window (also set by history retrieval)
    "truncated_tool_outputs",  # observation_truncation
    "compressed_messages",  # prompt_compression
    "history_retrievals",  # incontext_history_retrieval
)


def reset_turn_signals(turn_id: str) -> None:
    """Start a fresh signal tally for a turn (called once, at turn start)."""
    if not turn_id:
        return
    with _TURN_SIGNALS_LOCK:
        # Bound memory without a race: an aborted turn never pops its entry, so
        # evict the OLDEST tallies (dict insertion order) once over the cap.
        # Never clear() the whole dict — that would wipe concurrent in-flight
        # turns' signals and could mis-grade a probe as "compaction never fired".
        while len(_TURN_SIGNALS) >= _MAX_TRACKED_TURNS:
            _TURN_SIGNALS.pop(next(iter(_TURN_SIGNALS)), None)
        _TURN_SIGNALS[turn_id] = {}


def record_turn_signal(turn_id: str, name: str, amount: int = 1) -> None:
    """Accumulate a middleware signal for one turn (thread-safe)."""
    if not turn_id or not amount:
        return
    with _TURN_SIGNALS_LOCK:
        bucket = _TURN_SIGNALS.setdefault(turn_id, {})
        bucket[name] = bucket.get(name, 0) + int(amount)


def record_turn_signal_max(turn_id: str, name: str, value: int) -> None:
    """Record the per-turn MAXIMUM of a signal (thread-safe).

    Use this for per-call-view magnitudes (messages dropped, outputs truncated):
    a middleware's wrap_model_call fires once per model call within a turn, so
    summing would re-count overlapping prefixes each call. The largest single
    call is the meaningful per-turn figure.
    """
    if not turn_id:
        return
    with _TURN_SIGNALS_LOCK:
        bucket = _TURN_SIGNALS.setdefault(turn_id, {})
        bucket[name] = max(bucket.get(name, 0), int(value))


def pop_turn_signals(turn_id: str) -> dict[str, int]:
    """Take and clear a turn's accumulated signals (called when emitting stats)."""
    if not turn_id:
        return {}
    with _TURN_SIGNALS_LOCK:
        return _TURN_SIGNALS.pop(turn_id, {})


class TurnUsageHandler(UsageMetadataCallbackHandler):
    """Aggregate per-model usage for one turn and count chat-model calls."""

    def __init__(self) -> None:
        super().__init__()
        self.llm_calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        with self._lock:
            self.llm_calls += 1
        super().on_llm_end(response, **kwargs)


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """USD per million tokens. ``cache_write`` of None bills cache-creation
    tokens at the plain input rate (providers without a write surcharge)."""

    input: float
    output: float
    cache_read: float
    cache_write: float | None = None


# Verify against the provider price sheets before quoting cost numbers anywhere
# public; prices move. Keys match by longest prefix so dated variants
# ("claude-haiku-4-5-20251001") hit their family entry.
#
# Sources checked 2026-06-13:
# - https://ai.google.dev/gemini-api/docs/pricing
# - https://platform.claude.com/docs/en/about-claude/pricing
MODEL_PRICING: dict[str, ModelPricing] = {
    "gemini-3.5-flash": ModelPricing(input=1.50, output=9.00, cache_read=0.15),
    "claude-haiku-4-5": ModelPricing(
        input=1.00, output=5.00, cache_read=0.10, cache_write=1.25
    ),
    # DeepSeek-V4-Flash via OpenRouter (slug "deepseek/deepseek-v4-flash").
    # Kept for the openrouter provider path; unused now that evals run on the
    # first-party API. Note OpenRouter's own list price differs ($0.09/$0.18).
    "deepseek/deepseek-v4-flash": ModelPricing(
        input=0.14, output=0.28, cache_read=0.0028
    ),
    # DeepSeek-V4-Flash, DeepSeek FIRST-PARTY API (base https://api.deepseek.com,
    # model id "deepseek-v4-flash" — the id the API echoes back, so this is the
    # key usage_by_model resolves against). Per-1M-token pricing verified against
    # DeepSeek's price page 2026-06-19: $0.14 cache-miss input / $0.28 output /
    # $0.0028 cache-hit input (the ~50x cache discount that drives the cost test).
    "deepseek-v4-flash": ModelPricing(
        input=0.14, output=0.28, cache_read=0.0028
    ),
}


def pricing_for_model(model_key: str) -> ModelPricing | None:
    best: tuple[int, ModelPricing] | None = None
    for prefix, pricing in MODEL_PRICING.items():
        if model_key.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), pricing)
    return best[1] if best else None


def usage_totals(usage_by_model: dict[str, Any]) -> dict[str, int]:
    """Sum usage across models into the fields the stats event reports."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    for usage in usage_by_model.values():
        totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        details = usage.get("input_token_details") or {}
        totals["cache_read_tokens"] += int(details.get("cache_read", 0) or 0)
        totals["cache_creation_tokens"] += int(details.get("cache_creation", 0) or 0)
    return totals


def estimate_cost_usd(usage_by_model: dict[str, Any]) -> float | None:
    """Estimated turn cost, or None when any used model has no price entry.

    Cached input tokens are billed at the cache-read rate; ``input_tokens``
    includes them (LangChain's UsageMetadata convention), so they are carved
    out of the plain-input bucket rather than added on top.
    """
    total = 0.0
    for model_key, usage in usage_by_model.items():
        pricing = pricing_for_model(model_key)
        if pricing is None:
            return None
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        details = usage.get("input_token_details") or {}
        cache_read = int(details.get("cache_read", 0) or 0)
        cache_creation = int(details.get("cache_creation", 0) or 0)
        plain_input = max(0, input_tokens - cache_read - cache_creation)
        write_rate = (
            pricing.cache_write if pricing.cache_write is not None else pricing.input
        )
        total += (
            plain_input * pricing.input
            + cache_read * pricing.cache_read
            + cache_creation * write_rate
            + output_tokens * pricing.output
        ) / 1_000_000
    return total


def context_window_stats(
    messages: list[BaseMessage], cleared_placeholder: str
) -> dict[str, int]:
    """Describe the checkpointed context after a turn.

    ``summary_messages`` counts SummarizationMiddleware's summary insertions
    (tagged ``lc_source: summarization``); ``cleared_tool_outputs`` counts
    tool results ContextEditingMiddleware replaced with the placeholder. A
    runner diffs these across turns to verify compaction actually fired.
    """
    summary_messages = 0
    cleared_tool_outputs = 0
    for message in messages:
        additional = getattr(message, "additional_kwargs", None) or {}
        if additional.get("lc_source") == "summarization":
            summary_messages += 1
        if getattr(message, "type", "") == "tool":
            content = message.content
            text = content if isinstance(content, str) else str(content)
            if text.startswith(cleared_placeholder):
                cleared_tool_outputs += 1
    return {
        "context_messages": len(messages),
        "context_tokens_approx": (
            int(count_tokens_approximately(messages)) if messages else 0
        ),
        "summary_messages": summary_messages,
        "cleared_tool_outputs": cleared_tool_outputs,
    }
