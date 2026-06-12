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


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    name: str
    summarization: bool = True
    summarization_trigger_tokens: int = 30_000
    summarization_keep_messages: int = 20
    context_editing: bool = True
    context_editing_trigger_tokens: int = 5_000
    context_editing_keep: int = 5
    longterm_memory: bool = False


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
