from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class SourceMatch:
    doc_id: str
    title: str
    url: str
    source_key: str
    source_label: str
    score: float
    group: str = ""
    # KB-root-relative file path ("raw/docs/...") for manifest-backed matches;
    # lets the client map inline `raw/...` citations to this source's real URL.
    path: str = ""


@dataclass(frozen=True, slots=True)
class ChatRequest:
    query: str
    history: tuple[ChatTurn, ...] = ()
    source_keys: tuple[str, ...] = ()
    model_name: str = ""
    include_reasoning: bool = False
    thread_id: str = ""
    enabled_tools: tuple[str, ...] = ()
    # Memory/context-management preset name (see app/memory_presets.py).
    # Empty means the env-var/default resolution order.
    memory_preset: str = ""
    # Long-term memory key: profile-memory presets read and update the stored
    # student profile under this id. Empty disables profile memory I/O.
    student_id: str = ""


@dataclass(frozen=True, slots=True)
class ChatEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
