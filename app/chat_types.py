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
    # Experiment-only DeepSeek KV-cache namespace. The eval runner generates a
    # stable opaque id per arm/session/trial to prevent cross-arm cache warming.
    cache_user_id: str = ""
    # Part C / Axis B ablation: drop the run_kb_command tool (and its prompt
    # section) while keeping retrieval, to measure whether KB browsing helps.
    disable_kb: bool = False
    # Part C / Axis B sweep: per-request retrieval token budget (caps the chunks
    # the retriever returns). None keeps the default 100k budget.
    retrieval_budget: int | None = None
    # Experiment (GraphRAG vs RAG): which retrieval backend retrieve_tutor_context
    # uses. "" / "classical" = the default hybrid LocalChromaRetriever; "graphrag"
    # = the GraphRAG retriever over the prebuilt graph index. Opt-in; default
    # leaves production behavior unchanged.
    retriever: str = ""


@dataclass(frozen=True, slots=True)
class ChatEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
