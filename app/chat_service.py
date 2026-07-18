from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ExtendedModelResponse,
    SummarizationMiddleware,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately, get_buffer_string
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from .chat_types import ChatEvent, ChatRequest, ChatTurn, SourceMatch
from .deepseek_chat import TutorChatDeepSeek
from .memory_presets import (
    MemoryConfig,
    resolve_memory_preset,
)
from .telemetry import (
    TurnUsageHandler,
    aggregate_cost_breakdown,
    context_window_stats,
    estimate_cost_usd,
    pop_turn_events,
    pop_turn_signals,
    record_turn_event,
    record_turn_signal,
    record_turn_signal_max,
    reset_turn_signals,
    usage_totals,
)
from .chroma_rag import LocalChromaRetriever, format_tool_payload, parse_tool_payload
from .kb_shell import (
    KbCommandError,
    format_command_payload,
    run_kb_command as execute_kb_command,
)
from .kb_manifest import (
    citation_dedupe_key,
    extract_raw_paths,
    parse_markdown_citations,
    resolve_manifest_reference,
    source_match_key,
    source_match_payload,
)
from .prompts import build_system_prompt, ensure_kb_agents_instructions
from .provider_events import (
    GoogleSearchActivity,
    extract_anthropic_source_matches,
    extract_reasoning_deltas,
)
from .config import (
    BM25_INDEX_PATH,
    COURSE_SOURCE_KEYS,
    DEEPSEEK_DIRECT_MODEL_NAME,
    DEFAULT_SELECTED_SOURCE_KEYS,
    DOCUMENT_DICT_PATH,
    GEMINI_FALLBACK_MODEL_NAME,
    SOURCE_KEY_TO_LABEL,
    VECTOR_COLLECTION_NAME,
    VECTOR_DB_DIR,
    ensure_local_vector_db,
)

logger = logging.getLogger(__name__)

# Thread state for every model, keyed only by thread_id -- so nothing here
# stops a caller from pointing two providers at one thread. Do not read that as
# permission: a checkpoint holds PROVIDER-NATIVE messages (Gemini thought
# signatures, Anthropic signed thinking blocks, per-provider server-tool
# blocks), and replaying them to a different provider is a hard 400, not a
# degraded answer. One thread is one provider; see build_chat_model for the
# verified failure modes and sync_thread_with_history for the safe way to
# change provider (branch to a fresh thread seeded with plain text).
CHECKPOINTER = InMemorySaver()
# Long-term memory (student profiles), keyed by namespace ("student", <id>).
# In-process like the checkpointer: profiles survive across threads/sessions
# within one server lifetime, which is what the profile-memory experiments
# need. Swap for a persistent LangGraph store to survive restarts.
STORE = InMemoryStore()
CLEARED_TOOL_OUTPUT_PLACEHOLDER = "[tool output cleared to save context]"
# Visible transcript per thread, recorded as each turn finishes streaming.
# The checkpointed messages stop mirroring the visible transcript once
# SummarizationMiddleware rewrites thread state, so thread reuse is decided
# against this record first; the checkpoint-derived transcript is only a
# fallback for threads with no record (e.g. created before a restart).
_THREAD_TRANSCRIPTS: dict[str, tuple[ChatTurn, ...]] = {}
# Fork points per thread: visible turn count -> checkpoint_id of the thread
# state at that turn boundary. When the client edits or regenerates, its
# history is an exact prefix of the tracked transcript; running the agent
# with that checkpoint_id forks the thread from the prefix's real state
# (tool outputs and summaries included) instead of a plain-text rebuild.
_THREAD_FORK_POINTS: dict[str, dict[int, str]] = {}
# Provider whose native message blocks are in each thread's checkpoint, taken
# from the model that ACTUALLY served the turn (the DeepSeek fallback means the
# serving model is not always the requested one). A turn on a different provider
# cannot replay those blocks, so sync_thread_with_history branches on mismatch
# rather than letting the request 400.
_THREAD_PROVIDERS: dict[str, str] = {}
# Threads live in process memory only, so idle ones (abandoned tabs, "New
# chat") must be evicted or the saver grows forever. Eviction is graceful:
# if the client comes back, the request restores its visible history into
# the same thread id as plain text.
THREAD_IDLE_TTL_SECONDS = 60.0 * 60.0
_THREAD_SWEEP_INTERVAL_SECONDS = 5.0 * 60.0
_THREAD_LAST_USED: dict[str, float] = {}
_THREAD_SWEEP_STATE = {"last": 0.0}
_THREAD_TRANSCRIPT_LOCK = Lock()
_RETRIEVER_INIT_LOCK = Lock()
DEFAULT_KB_COMMAND_LIMIT = 20
_KB_COMMAND_COUNTS: dict[str, int] = {}
_KB_COMMAND_COUNT_LOCK = Lock()


@dataclass(frozen=True)
class AppContext:
    allowed_sources: tuple[str, ...]
    kb_session_id: str = ""
    kb_command_limit: int = DEFAULT_KB_COMMAND_LIMIT
    student_id: str = ""
    # DeepSeek experiment-only cache namespace. Stable within one trajectory,
    # distinct across arm/session/trial, and intentionally contains no PII.
    cache_user_id: str = ""
    # Per-request retrieval token budget (Part C / Axis B sweep); None keeps the
    # retriever's DEFAULT_CONTEXT_TOKEN_BUDGET.
    retrieval_budget: int | None = None
    # Which retrieval backend retrieve_tutor_context uses (GraphRAG experiment).
    # "" / "classical" = hybrid LocalChromaRetriever (default); "graphrag" = the
    # GraphRAG retriever over the prebuilt graph index.
    retriever_kind: str = ""


def get_student_profile(student_id: str) -> str:
    if not student_id:
        return ""
    item = STORE.get(("student", student_id), "profile")
    if item is None:
        return ""
    return str((item.value or {}).get("profile", "")).strip()


def set_student_profile(student_id: str, profile: str) -> None:
    """Write a student profile directly; eval runners use this to seed
    personas without spending a turn on the on-the-fly write path."""
    if not student_id:
        return
    STORE.put(("student", student_id), "profile", {"profile": profile.strip()})


def _claim_kb_command_budget(session_id: str, limit: int) -> tuple[bool, int]:
    if not session_id:
        return True, 1
    with _KB_COMMAND_COUNT_LOCK:
        used = _KB_COMMAND_COUNTS.get(session_id, 0)
        if used >= limit:
            return False, used
        used += 1
        _KB_COMMAND_COUNTS[session_id] = used
        return True, used


def _clear_kb_command_budget(session_id: str) -> None:
    if not session_id:
        return
    with _KB_COMMAND_COUNT_LOCK:
        _KB_COMMAND_COUNTS.pop(session_id, None)


def _get_thread_transcript(thread_id: str) -> tuple[ChatTurn, ...] | None:
    with _THREAD_TRANSCRIPT_LOCK:
        return _THREAD_TRANSCRIPTS.get(thread_id)


def _record_thread_transcript(thread_id: str, transcript: tuple[ChatTurn, ...]) -> None:
    with _THREAD_TRANSCRIPT_LOCK:
        _THREAD_TRANSCRIPTS[thread_id] = transcript


def _get_thread_provider(thread_id: str) -> str:
    with _THREAD_TRANSCRIPT_LOCK:
        return _THREAD_PROVIDERS.get(thread_id, "")


def _record_thread_provider(thread_id: str, provider: str) -> None:
    """Remember which provider's blocks are in this thread's checkpoint.

    Recorded from the model that ACTUALLY served the turn, not the one the
    request asked for: when the DeepSeek fallback fires, Gemini is what wrote
    the messages, and that is what the next turn has to replay.
    """
    if not provider:
        return
    with _THREAD_TRANSCRIPT_LOCK:
        _THREAD_PROVIDERS[thread_id] = provider


def _get_fork_point(thread_id: str, turn_count: int) -> str:
    with _THREAD_TRANSCRIPT_LOCK:
        return _THREAD_FORK_POINTS.get(thread_id, {}).get(turn_count, "")


def _record_fork_point(thread_id: str, turn_count: int, checkpoint_id: str) -> None:
    if not checkpoint_id:
        return
    with _THREAD_TRANSCRIPT_LOCK:
        _THREAD_FORK_POINTS.setdefault(thread_id, {})[turn_count] = checkpoint_id


def _prune_fork_points(thread_id: str, keep_up_to: int) -> None:
    """Forget fork points past a fork: they map turn counts of the abandoned
    branch, and the new branch records its own as turns complete."""
    with _THREAD_TRANSCRIPT_LOCK:
        points = _THREAD_FORK_POINTS.get(thread_id)
        if not points:
            return
        for turn_count in [count for count in points if count > keep_up_to]:
            del points[turn_count]


def _drop_thread_record(thread_id: str) -> None:
    with _THREAD_TRANSCRIPT_LOCK:
        _THREAD_TRANSCRIPTS.pop(thread_id, None)
        _THREAD_FORK_POINTS.pop(thread_id, None)
        _THREAD_LAST_USED.pop(thread_id, None)
        _THREAD_PROVIDERS.pop(thread_id, None)


def _touch_thread(thread_id: str, now: float | None = None) -> None:
    with _THREAD_TRANSCRIPT_LOCK:
        _THREAD_LAST_USED[thread_id] = time.monotonic() if now is None else now


def _evict_idle_threads(now: float | None = None, *, force: bool = False) -> list[str]:
    """Delete checkpoints and records of threads idle past the TTL.

    Called opportunistically per request; the actual scan runs at most once
    per sweep interval unless ``force`` is set.
    """
    if now is None:
        now = time.monotonic()
    with _THREAD_TRANSCRIPT_LOCK:
        if not force and now - _THREAD_SWEEP_STATE["last"] < (
            _THREAD_SWEEP_INTERVAL_SECONDS
        ):
            return []
        _THREAD_SWEEP_STATE["last"] = now
        expired = [
            thread_id
            for thread_id, last_used in _THREAD_LAST_USED.items()
            if now - last_used > THREAD_IDLE_TTL_SECONDS
        ]
        for thread_id in expired:
            _THREAD_LAST_USED.pop(thread_id, None)
            _THREAD_TRANSCRIPTS.pop(thread_id, None)
            _THREAD_FORK_POINTS.pop(thread_id, None)
            _THREAD_PROVIDERS.pop(thread_id, None)
    for thread_id in expired:
        CHECKPOINTER.delete_thread(thread_id)
    if expired:
        logger.info("Evicted %d idle thread(s).", len(expired))
    return expired


def _latest_checkpoint_id(agent, thread_id: str) -> str:
    state = agent.get_state(thread_config(thread_id))
    config = getattr(state, "config", None) or {}
    return str(config.get("configurable", {}).get("checkpoint_id", "") or "")


def _seed_checkpoint_id(update_config: Any) -> str:
    if not isinstance(update_config, dict):
        return ""
    return str(update_config.get("configurable", {}).get("checkpoint_id", "") or "")


@lru_cache(maxsize=1)
def _build_retriever() -> LocalChromaRetriever:
    ensure_local_vector_db()
    cohere_api_key = os.environ["COHERE_API_KEY"]
    return LocalChromaRetriever(
        db_path=VECTOR_DB_DIR,
        collection_name=VECTOR_COLLECTION_NAME,
        document_dict_path=DOCUMENT_DICT_PATH,
        bm25_index_path=BM25_INDEX_PATH,
        cohere_api_key=cohere_api_key,
    )


def get_retriever() -> LocalChromaRetriever:
    with _RETRIEVER_INIT_LOCK:
        return _build_retriever()


def select_retriever(retriever_kind: str = ""):
    """Pick the retrieval backend for retrieve_tutor_context.

    Default ("" / "classical") is the hybrid LocalChromaRetriever. "graphrag"
    returns the GraphRAG retriever over the prebuilt local graph index (used by
    the GraphRAG-vs-RAG experiment); both expose the same ``search(...)``
    signature returning ``list[SearchResult]``.
    """
    if (retriever_kind or "").lower() == "graphrag":
        from .graph_rag import get_graphrag_retriever

        with _RETRIEVER_INIT_LOCK:
            return get_graphrag_retriever()
    return get_retriever()


def warm_up_retriever() -> None:
    if not os.environ.get("COHERE_API_KEY"):
        return
    try:
        get_retriever()
    except Exception as exc:  # pragma: no cover - diagnostic logging only
        logger.warning(
            "Retriever warm-up failed; first retrieval call may retry. error=%s",
            exc,
        )


RETRIEVE_TUTOR_CONTEXT_SCHEMA = {
    "title": "retrieve_tutor_context",
    "description": "Retrieve relevant course and documentation context for an AI tutor question.",
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The question or topic to search for in the course and documentation corpus.",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


@tool(args_schema=RETRIEVE_TUTOR_CONTEXT_SCHEMA)
def retrieve_tutor_context(
    query: str, runtime: ToolRuntime[AppContext], **unsupported: Any
) -> str:
    """Retrieve relevant course and documentation context for an AI tutor question."""
    if unsupported:
        names = ", ".join(sorted(unsupported))
        logger.warning(
            "retrieve_tutor_context received unsupported arguments: %s", names
        )
        return (
            "retrieve_tutor_context could not run because it received unsupported "
            f"argument(s): {names}. Retry the tool with only the query argument."
        )
    try:
        results = select_retriever(
            getattr(runtime.context, "retriever_kind", "")
        ).search(
            query=query,
            allowed_sources=list(runtime.context.allowed_sources),
            token_budget=getattr(runtime.context, "retrieval_budget", None),
        )
    except Exception as exc:
        # Degrade instead of killing the turn: retrieval depends on Cohere
        # (embed + rerank), so on failure return a soft message and let the
        # agent fall back to run_kb_command or general knowledge.
        logger.warning("retrieve_tutor_context failed; degrading. error=%s", exc)
        return (
            "retrieve_tutor_context is temporarily unavailable. Use run_kb_command "
            "to browse the knowledge base or answer from general knowledge, and let "
            "the user know retrieval was unavailable."
        )
    return format_tool_payload(query, results)


RUN_KB_COMMAND_SCHEMA = {
    "title": "run_kb_command",
    "description": "Run a safe, read-only terminal-style command inside the local KB.",
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": (
                "Single read-only command to run under data/kb. Supported commands: "
                "rg, grep, find, ls, sed, head, cat, wc. Pipes, redirects, command "
                "chaining, network commands, and writes are not allowed."
            ),
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Command timeout in seconds, capped by the runtime.",
            "default": 8,
            "minimum": 1,
            "maximum": 30,
        },
        "timeout": {
            "type": "integer",
            "description": (
                "Alias for timeout_seconds. The runtime still caps the command "
                "at 30 seconds."
            ),
            "minimum": 1,
            "maximum": 30,
        },
        "max_output_chars": {
            "type": "integer",
            "description": "Maximum stdout/stderr characters to return, capped by the runtime.",
            "default": 40000,
            "minimum": 1000,
            "maximum": 80000,
        },
    },
    "required": ["command"],
    "additionalProperties": False,
}


@tool(args_schema=RUN_KB_COMMAND_SCHEMA)
def run_kb_command(
    command: str,
    runtime: ToolRuntime[AppContext],
    timeout_seconds: int | None = None,
    max_output_chars: int = 40000,
    timeout: int | None = None,
    **unsupported: Any,
) -> str:
    """Run a safe, read-only terminal-style command inside the local KB."""
    if unsupported:
        names = ", ".join(sorted(unsupported))
        logger.warning("run_kb_command received unsupported arguments: %s", names)
        return (
            f"$ {command}\n"
            f"error: unsupported run_kb_command argument(s): {names}. "
            "Use command, timeout_seconds (or timeout), and max_output_chars."
        )
    effective_timeout = timeout_seconds if timeout_seconds is not None else timeout
    if effective_timeout is None:
        effective_timeout = 8
    allowed, used = _claim_kb_command_budget(
        runtime.context.kb_session_id,
        runtime.context.kb_command_limit,
    )
    if not allowed:
        return (
            f"$ {command}\n"
            "error: KB command budget exceeded for this turn "
            f"({used}/{runtime.context.kb_command_limit}). "
            "Use the evidence already collected to answer now."
        )
    try:
        ensure_local_vector_db()
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.warning("KB artifact download/check failed. error=%s", exc)
    try:
        result = execute_kb_command(
            command,
            timeout_seconds=effective_timeout,
            max_output_chars=max_output_chars,
        )
        return format_command_payload(result)
    except (KbCommandError, OSError, TypeError, ValueError) as exc:
        return f"$ {command}\nerror: {exc}"


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "get") and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "".join(parts)
    return str(content)


def normalize_history(history: tuple[ChatTurn, ...]) -> tuple[ChatTurn, ...]:
    """Keep user/assistant turns, trimming whitespace so incoming history
    compares equal to checkpoint-derived history."""
    normalized: list[ChatTurn] = []
    for turn in history:
        if turn.role not in {"user", "assistant"}:
            continue
        normalized.append(ChatTurn(role=turn.role, content=turn.content.strip()))
    return tuple(normalized)


def checkpoint_messages_to_history(messages: list[BaseMessage]) -> tuple[ChatTurn, ...]:
    """Collapse a checkpointed message list into the user-visible transcript.

    A tool-using turn is checkpointed as several AI messages (one per
    tool-call round, then the final answer) with ToolMessages in between,
    while the frontend renders the whole turn as a single assistant message
    whose text parts join without a separator. Mirror that here, merging
    each run of AI messages into one assistant turn before trimming, so the
    result compares equal to the history the client sends back.
    """
    history: list[ChatTurn] = []
    for message in messages:
        message_type = getattr(message, "type", None)
        if message_type == "human":
            history.append(ChatTurn("user", message_content_to_text(message.content)))
            continue
        if message_type == "ai":
            text = message_content_to_text(message.content)
            if history and history[-1].role == "assistant":
                history[-1] = ChatTurn("assistant", history[-1].content + text)
            else:
                history.append(ChatTurn("assistant", text))
    return tuple(ChatTurn(turn.role, turn.content.strip()) for turn in history)


def history_to_langgraph_messages(history: tuple[ChatTurn, ...]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for message in history:
        if message.role == "user":
            messages.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            messages.append(AIMessage(content=message.content))
    return messages


def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def new_thread_id() -> str:
    return uuid4().hex


def sync_thread_with_history(
    agent,
    thread_id: str,
    history: tuple[ChatTurn, ...],
    provider: str = "",
) -> tuple[str, str]:
    """Pick the thread (and optional fork checkpoint) for this request.

    Returns ``(thread_id, fork_checkpoint_id)``. A non-empty checkpoint id
    means the run must fork the thread from that checkpoint (LangGraph
    time travel) instead of continuing from its latest state.

    The branch path at the bottom (fresh thread seeded from the visible
    transcript via history_to_langgraph_messages) is the ONLY safe way to move
    a conversation onto a different provider: it reseeds plain user/assistant
    text and drops the checkpoint's provider-native state -- tool outputs,
    summaries, thought signatures -- which is exactly what makes it portable.
    That data loss is the feature, not a bug to fix. Continuing or forking a
    thread across providers instead replays unportable blocks and 400s; see
    build_chat_model.

    ``provider`` is the provider serving THIS turn. When it differs from the one
    that wrote the thread's checkpointed messages, every other path here would
    replay unportable blocks, so the mismatch branches unconditionally.
    """
    owner = _get_thread_provider(thread_id)
    if provider and owner and owner != provider:
        # The DeepSeek->Gemini fallback is the common way to get here: a
        # rescued turn checkpoints Gemini-native messages, so the next DeepSeek
        # turn would 400 and silently re-fall-back to Gemini forever (verified:
        # a healthy DeepSeek key still 400s on that thread). Branching resets
        # the conversation onto portable plain text so the requested model
        # actually serves it.
        logger.info(
            "Thread %s was written by %s but this turn runs on %s; branching to a "
            "fresh thread seeded with plain text (tool outputs and summaries are "
            "dropped -- provider-native blocks cannot be replayed across providers).",
            thread_id,
            owner,
            provider,
        )
        return _branch_to_fresh_thread(agent, thread_id, history), ""

    tracked = _get_thread_transcript(thread_id)
    if tracked == history:
        # The client sent back exactly what this thread streamed to it, so
        # continue the thread even when summarization or context editing has
        # rewritten the checkpointed messages.
        return thread_id, ""

    if tracked is None:
        state = agent.get_state(thread_config(thread_id))
        checkpoint_history = checkpoint_messages_to_history(
            state.values.get("messages", [])
        )

        if checkpoint_history == history:
            return thread_id, ""

        if not checkpoint_history:
            restored_messages = history_to_langgraph_messages(history)
            if restored_messages:
                update_config = agent.update_state(
                    thread_config(thread_id), {"messages": restored_messages}
                )
                _record_fork_point(
                    thread_id, len(history), _seed_checkpoint_id(update_config)
                )
            return thread_id, ""
    elif history == tracked[: len(history)]:
        # The client kept an exact prefix of this thread's transcript (edit
        # or regenerate). Fork from the checkpoint at that turn boundary so
        # the prefix keeps its real state: tool outputs, summaries, etc.
        fork_checkpoint_id = _get_fork_point(thread_id, len(history))
        if fork_checkpoint_id:
            _prune_fork_points(thread_id, keep_up_to=len(history))
            return thread_id, fork_checkpoint_id

    # No checkpoint maps to the history the client kept: branch to a fresh
    # thread seeded with the visible messages as plain text.
    return _branch_to_fresh_thread(agent, thread_id, history), ""


def _branch_to_fresh_thread(
    agent,
    thread_id: str,
    history: tuple[ChatTurn, ...],
) -> str:
    """Reseed the visible transcript as plain text on a brand-new thread.

    Drops everything provider-native in the old checkpoint (tool outputs,
    summaries, thought/thinking signatures). That loss is the point: plain
    user/assistant text is the only representation every provider accepts.
    """
    branched_thread_id = new_thread_id()
    restored_messages = history_to_langgraph_messages(history)
    if restored_messages:
        update_config = agent.update_state(
            thread_config(branched_thread_id),
            {"messages": restored_messages},
        )
        _record_fork_point(
            branched_thread_id, len(history), _seed_checkpoint_id(update_config)
        )
    # The client follows thread_started to the branched id, so the superseded
    # thread is unreachable; drop it or it lives in the in-memory saver forever.
    CHECKPOINTER.delete_thread(thread_id)
    _drop_thread_record(thread_id)
    return branched_thread_id


def collect_retrieval_source_matches(payload: str) -> list[SourceMatch]:
    matches: list[SourceMatch] = []
    for match in parse_tool_payload(payload):
        matches.append(
            SourceMatch(
                doc_id=match.doc_id,
                title=match.title,
                url=match.url,
                source_key=match.source,
                source_label=SOURCE_KEY_TO_LABEL.get(match.source, match.source),
                score=match.score,
                group="courses" if match.source in COURSE_SOURCE_KEYS else "docs",
            )
        )
    return matches


def _source_match_record(match: SourceMatch) -> dict[str, Any]:
    """Store lightweight source metadata alongside a persistently capped tool."""
    return {
        "doc_id": match.doc_id,
        "title": match.title,
        "url": match.url,
        "source_key": match.source_key,
        "source_label": match.source_label,
        "score": match.score,
        "group": match.group,
        "path": match.path,
    }


def _source_matches_from_records(records: Any) -> list[SourceMatch]:
    """Restore source metadata retained outside a truncated retrieval payload."""
    if not isinstance(records, list):
        return []
    matches: list[SourceMatch] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            matches.append(
                SourceMatch(
                    doc_id=str(record.get("doc_id", "")),
                    title=str(record.get("title", "")),
                    url=str(record.get("url", "")),
                    source_key=str(record.get("source_key", "")),
                    source_label=str(record.get("source_label", "")),
                    score=float(record.get("score", 0.0)),
                    group=str(record.get("group", "")),
                    path=str(record.get("path", "")),
                )
            )
        except (TypeError, ValueError):
            continue
    return matches


def _record_evidence(
    target: dict[str, SourceMatch], matches: list[SourceMatch]
) -> None:
    for match in matches:
        key = source_match_key(match)
        existing = target.get(key)
        if existing and existing.score >= match.score:
            continue
        target[key] = match


def _index_evidence(matches: dict[str, SourceMatch]) -> dict[str, SourceMatch]:
    index: dict[str, SourceMatch] = {}
    for match in matches.values():
        for key in (match.doc_id, match.url, match.title.strip().lower()):
            if key:
                index[key] = match
    return index


def _external_web_source(reference: str, label: str) -> SourceMatch:
    """A cited http(s) URL that matched no evidence bucket and no manifest entry.

    Only surfaced when ``keep_unresolved_sources=True``; tagged ``group="web"`` so
    the UI renders it as a low-trust external chip.
    """
    return SourceMatch(
        doc_id="",
        title=label.strip() or reference,
        url=reference,
        source_key="web",
        source_label="Web",
        score=0.0,
        group="web",
    )


def _match_evidence(
    reference: str,
    label: str,
    evidence_indexes: list[dict[str, SourceMatch]],
) -> SourceMatch | None:
    for index in evidence_indexes:
        for candidate in (reference, label.strip().lower()):
            if candidate and candidate in index:
                return index[candidate]
    return None


def _match_manifest_via_shell(
    reference: str,
    label: str,
    shell_index: dict[str, SourceMatch],
) -> SourceMatch | None:
    """Resolve a reference through the KB manifest, but only trust it when the
    same doc was actually browsed this turn via ``run_kb_command`` (i.e. it is in
    shell evidence)."""
    manifest_match = resolve_manifest_reference(reference, label=label)
    if not manifest_match:
        return None
    for key in (
        manifest_match.doc_id,
        manifest_match.url,
        manifest_match.title.strip().lower(),
    ):
        if key and key in shell_index:
            return shell_index[key]
    return None


def resolve_answer_citations(
    answer: str,
    *,
    retrieval_evidence: dict[str, SourceMatch],
    shell_evidence: dict[str, SourceMatch],
    web_evidence: dict[str, SourceMatch],
    keep_unresolved_sources: bool = False,
) -> list[SourceMatch]:
    """Turn the model's inline citations into trusted source cards.

    For each inline citation in ``answer`` the reference (URL, title,
    ``kb://doc/<id>``, or raw path) is matched against the current turn's
    evidence:

    * ``retrieval_evidence`` / ``web_evidence`` — anything
      ``retrieve_tutor_context`` or the web tools surfaced. **Web-search results
      are recorded in ``web_evidence``**, so a web source the model cites inline
      resolves here exactly like a corpus source — no special handling needed.
    * ``shell_evidence`` (+ KB manifest) — files browsed via ``run_kb_command``.

    Matches are deduped by URL and returned in **citation order** (the order the
    links appear in the answer).

    ``keep_unresolved_sources`` governs *only* inline http(s) URLs that match no
    evidence bucket and no manifest entry — links the model produced from memory
    or lifted from a doc body that no tool actually surfaced. Default ``False``
    drops them (they still render as plain links in the answer prose); ``True``
    surfaces them as low-trust "Web" chips.
    """
    evidence_indexes = [
        _index_evidence(retrieval_evidence),
        _index_evidence(shell_evidence),
        _index_evidence(web_evidence),
    ]
    shell_index = _index_evidence(shell_evidence)
    resolved: list[SourceMatch] = []
    seen: set[str] = set()
    for label, reference in parse_markdown_citations(answer):
        match = _match_evidence(reference, label, evidence_indexes)
        if match is None:
            match = _match_manifest_via_shell(reference, label, shell_index)
        if (
            match is None
            and keep_unresolved_sources
            and reference.startswith(("http://", "https://"))
        ):
            match = _external_web_source(reference, label)
        if match is None:
            continue
        key = citation_dedupe_key(match)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(match)
    return resolved


# Commands whose output is the body of the file they were pointed at. raw/
# links printed by these are references the model *saw* in that text, not docs
# it consulted (e.g. the source list of a cat'ed wiki topic page).
_KB_READ_EXECUTABLES = frozenset({"cat", "sed", "head"})


@dataclass(frozen=True, slots=True)
class ShellSourceMatches:
    """Corpus docs a KB command touched, split by how directly.

    ``browsed`` — docs the command actually read (cat/sed/head file args) or
    returned as search hits (raw/ paths printed by rg/grep/find/ls). Attached
    to the tool event, so the activity feed counts only consulted docs.
    ``referenced`` — ``browsed`` plus docs merely linked in the printed text.
    Recorded as citation evidence so the model may still cite a doc it saw
    referenced in a wiki page, without that mention inflating the feed.
    """

    browsed: list[SourceMatch]
    referenced: list[SourceMatch]


def _resolve_shell_paths(paths: list[str]) -> list[SourceMatch]:
    matches: list[SourceMatch] = []
    seen: set[str] = set()
    for path in paths:
        match = resolve_manifest_reference(path)
        if not match:
            continue
        key = source_match_key(match)
        if key in seen:
            continue
        seen.add(key)
        matches.append(match)
    return matches


def extract_shell_source_matches(command: str, output_text: str) -> ShellSourceMatches:
    output_paths = extract_raw_paths(output_text)
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = []
    executable = tokens[0] if tokens else ""
    arg_paths: list[str] = []
    if executable == "cat":
        arg_paths.extend(tokens[1:])
    elif executable == "sed" and len(tokens) >= 4:
        arg_paths.append(tokens[-1])
    elif executable == "head":
        start = 1
        if len(tokens) > 2 and tokens[1] == "-n":
            start = 3
        arg_paths.extend(tokens[start:])
    browsed_paths = (
        arg_paths if executable in _KB_READ_EXECUTABLES else [*arg_paths, *output_paths]
    )
    return ShellSourceMatches(
        browsed=_resolve_shell_paths(browsed_paths),
        referenced=_resolve_shell_paths([*arg_paths, *output_paths]),
    )


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    if ":" in normalized:
        return normalized
    if normalized.startswith("gpt-"):
        return f"openai:{normalized}"
    if normalized.startswith("claude"):
        return f"anthropic:{normalized}"
    if normalized.startswith("gemini"):
        return f"google-genai:{normalized}"
    if normalized.startswith("deepseek"):
        return f"deepseek:{normalized}"
    return normalized


def is_google_genai_model(model_name: str) -> bool:
    provider_model = normalize_model_name(model_name)
    provider, _, _actual_model = provider_model.partition(":")
    return provider == "google-genai"


def is_anthropic_model(model_name: str) -> bool:
    provider_model = normalize_model_name(model_name)
    provider, _, _actual_model = provider_model.partition(":")
    return provider == "anthropic"


def is_deepseek_model(model_name: str) -> bool:
    provider_model = normalize_model_name(model_name)
    provider, _, _actual_model = provider_model.partition(":")
    return provider == "deepseek"


def supports_gemini_tool_combination(model_name: str) -> bool:
    """True when a Gemini model can mix built-in tools with function tools.

    Gemini calls this "tool context circulation", and it is a Gemini 3+
    preview: only then does `tool_config.include_server_side_tool_invocations`
    exist, and only then may google_search/url_context be bound alongside our
    custom function tools. On gemini-2.5-* the flag is a 400 ("Tool call
    context circulation is not enabled for models/gemini-2.5-flash") and the
    tool mix is rejected regardless -- and the agent always binds
    retrieve_tutor_context + run_kb_command, so there is no combination-free
    path for a pre-3 model to use web tools here.
    """
    if not is_google_genai_model(model_name):
        return False
    _provider, _, actual_model = normalize_model_name(model_name).partition(":")
    # Leading digits only: the generation is followed by "." on point releases
    # (gemini-3.5-flash) but by "-" on major ones (gemini-3-pro).
    generation = ""
    for char in actual_model.removeprefix("gemini-"):
        if not char.isdigit():
            break
        generation += char
    return bool(generation) and int(generation) >= 3


def format_tool_args(args: Any) -> str:
    if isinstance(args, dict):
        query = str(args.get("query", "")).strip()
        if query:
            return query
        command = str(args.get("command", "")).strip()
        if command:
            return command
        if args:
            return json.dumps(args, ensure_ascii=False, sort_keys=True)
        return ""
    if args is None:
        return ""
    return str(args).strip()


def _has_google_genai_key() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _has_deepseek_key() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def _build_chat_model_client(provider_model: str, include_thoughts: bool = False):
    provider, _, actual_model = provider_model.partition(":")

    if provider == "openai":
        return ChatOpenAI(model=actual_model, temperature=1)
    if provider == "openrouter":
        # OpenAI-compatible gateway for open models (DeepSeek, Qwen, ...).
        # stream_usage=True makes the streamed response carry token usage, so
        # context_stats / cost telemetry populates (some OpenAI-compatible
        # endpoints, e.g. Ollama, omit usage and leave token counts at 0).
        # Routing inside OpenRouter is left with its own fallbacks enabled:
        # pinning a single provider with allow_fallbacks=False made batches die
        # on a backend's transient 429 instead of routing around it.
        return ChatOpenAI(
            model=actual_model,
            temperature=1,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            stream_usage=True,
            # Long agentic sessions fire hundreds of calls against OpenRouter's
            # shared pool; bump retries (with the client's exponential backoff)
            # to ride out transient upstream 429s instead of failing a whole
            # session. A BYOK DeepSeek key is the real fix for sustained limits.
            max_retries=12,
        )
    if provider == "deepseek":
        # Use the provider-specific adapter: the generic ChatOpenAI wrapper
        # intentionally discards DeepSeek's non-standard reasoning_content
        # field even though the transport itself is OpenAI-compatible.
        # stream_usage=True so the streamed response carries token usage ->
        # context_stats / cost telemetry populates, including the cached-prefix
        # tokens that drive the cost comparison (DeepSeek caches prefixes
        # automatically; cache-hit input is ~50x cheaper than cache-miss). Reads
        # DEEPSEEK_API_KEY.
        return TutorChatDeepSeek(
            model=actual_model,
            temperature=None if include_thoughts else 1,
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            stream_usage=True,
            extra_body={
                "thinking": {"type": "enabled" if include_thoughts else "disabled"}
            },
            # A few retries ride out transient 429/5xx on long agentic sessions
            # without failing the whole session (we run evals at concurrency 1).
            max_retries=6,
        )
    if provider == "ollama":
        # Local SLM via Ollama's OpenAI-compatible endpoint (experiments only,
        # e.g. the SLM compaction study). The model's context window is set on
        # the Ollama side (a num_ctx Modelfile variant), not per request.
        return ChatOpenAI(
            model=actual_model,
            temperature=1,
            base_url=os.environ.get(
                "OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1"
            ),
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            # Ollama streams usage only when asked; without this the telemetry's
            # usage_metadata is empty and token counts come back as 0.
            stream_usage=True,
        )
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic support requires langchain-anthropic and anthropic in the environment. Run uv sync after updating dependencies."
            ) from exc

        if include_thoughts:
            # Haiku-tier models use the budget_tokens thinking surface
            # (adaptive thinking is unsupported); the interleaved beta lets
            # thinking blocks appear between tool calls within a turn.
            # Thinking requires temperature=1 and budget_tokens >= 1024.
            # max_tokens stays at the model-profile default (64k for Haiku
            # 4.5), same as the non-thinking path: thinking shares the
            # response's max_tokens, so a low cap silently truncates answers.
            return ChatAnthropic(
                model=actual_model,
                temperature=1,
                thinking={"type": "enabled", "budget_tokens": 2048},
                betas=["interleaved-thinking-2025-05-14"],
            )
        return ChatAnthropic(model=actual_model, temperature=1)
    if provider == "google-genai":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "Gemini support requires langchain-google-genai and google-genai in the environment. Run uv sync after updating dependencies."
            ) from exc

        return ChatGoogleGenerativeAI(
            model=actual_model,
            temperature=1,
            include_thoughts=include_thoughts,
        )

    raise ValueError(
        "Unsupported model provider. Use openai, openrouter, deepseek, anthropic, google-genai, or ollama."
    )


def build_chat_model(model_name: str, include_thoughts: bool = False):
    """Build the chat client for a turn, with the DeepSeek->Gemini rescue path.

    ONE THREAD IS ONE PROVIDER. Read this before touching the fallback.

    A thread's checkpoint stores PROVIDER-NATIVE message objects, not plain
    text: Gemini reasoning parts carry thought signatures, Anthropic thinking
    blocks carry a required cryptographic `signature`, and each provider's
    server-side tool calls are its own block types. Those payloads are not
    portable. Replaying one provider's checkpointed history to another is a
    hard API error, not a degraded answer. Both directions are verified:

      Gemini history -> DeepSeek     400 (OpenAI-compatible parser rejects it)
      Gemini history -> Anthropic    400 "messages.1.content.0.thinking.
                                     signature: Field required"

    So a fallback is NOT a free "try another model" switch. It is safe here
    only because of two properties, and it stops being safe if either breaks:

      1. The fallback answers the SAME request the primary just failed, so it
         reads its own request payload, never a foreign checkpoint.
      2. build_agent binds tools from the REQUESTED model (deepseek), which has
         no provider-native web tools, so the fallback is never handed Gemini's
         google_search/url_context alongside our custom function tools -- a
         combination gemini-2.5-flash cannot accept (see config.
         GEMINI_FALLBACK_MODEL_NAME for why).

    A rescued turn still writes GEMINI-shaped messages into a DeepSeek thread's
    checkpoint. That used to strand the thread: the next DeepSeek turn replayed
    foreign history, 400'd, and silently fell back again, migrating the thread
    to Gemini at ~10x the token price forever. It is contained OUTSIDE this
    function -- sync_thread_with_history records the serving provider per thread
    and branches to a fresh plain-text thread on mismatch. Do not "fix" that by
    adding provider translation or another fallback here.

    If you are here to add a fallback, a retry-on-another-provider, or
    mid-conversation model switching: do not wire it at this layer. Route it
    through sync_thread_with_history's plain-text branch, which drops the
    unportable state on purpose.
    """
    provider_model = normalize_model_name(model_name)
    if provider_model != DEEPSEEK_DIRECT_MODEL_NAME:
        return _build_chat_model_client(
            provider_model,
            include_thoughts=include_thoughts,
        )
    if not _has_deepseek_key() and _has_google_genai_key():
        logger.warning(
            "No DEEPSEEK_API_KEY is set for %s; using Gemini fallback %s.",
            DEEPSEEK_DIRECT_MODEL_NAME,
            GEMINI_FALLBACK_MODEL_NAME,
        )
        return _build_chat_model_client(
            GEMINI_FALLBACK_MODEL_NAME,
            include_thoughts=include_thoughts,
        )
    model = _build_chat_model_client(provider_model, include_thoughts=include_thoughts)
    if not _has_google_genai_key():
        logger.warning(
            "Gemini fallback %s is configured for %s, but no GOOGLE_API_KEY or "
            "GEMINI_API_KEY is set; using DeepSeek only.",
            GEMINI_FALLBACK_MODEL_NAME,
            DEEPSEEK_DIRECT_MODEL_NAME,
        )
        return model
    fallback = _build_chat_model_client(
        GEMINI_FALLBACK_MODEL_NAME,
        include_thoughts=include_thoughts,
    )
    return model.with_fallbacks([fallback])


class GeminiServerSideToolsMiddleware(AgentMiddleware):
    """Enable server-side tool invocation for Gemini 3 tool combinations.

    Combining built-in tools (e.g. `google_search`, `url_context`) with
    user-defined function declarations requires
    `include_server_side_tool_invocations=True` on Gemini's `ToolConfig`.
    We inject it via `model_settings` so it flows through LangChain's
    `bind_tools(..., tool_config=...)` path.
    """

    def _inject(self, request):
        existing = request.model_settings.get("tool_config") or {}
        if isinstance(existing, dict):
            next_tool_config: Any = {
                **existing,
                "include_server_side_tool_invocations": True,
            }
        else:
            next_tool_config = existing
        new_settings = {**request.model_settings, "tool_config": next_tool_config}
        return request.override(model_settings=new_settings)

    def wrap_model_call(self, request, handler):
        return handler(self._inject(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._inject(request))


class SourcePreferenceMiddleware(AgentMiddleware):
    """Append a selected-sources hint to the system prompt per request."""

    def _build_note(self, sources: tuple[str, ...]) -> str | None:
        if not sources:
            return None
        # Skip when the user kept the default (all sources). The note is only
        # useful when the user narrowed the picker.
        if set(sources) >= set(DEFAULT_SELECTED_SOURCE_KEYS):
            return None
        lines = [
            "## Selected sources for this turn",
            "",
            "Prefer these paths when using `run_kb_command`:",
        ]
        for key in sources:
            label = SOURCE_KEY_TO_LABEL.get(key, key)
            group = "courses" if key in COURSE_SOURCE_KEYS else "docs"
            wiki_dir = "courses" if key in COURSE_SOURCE_KEYS else "frameworks"
            lines.append(f"- {label}: `raw/{group}/{key}/`, `wiki/{wiki_dir}/{key}.md`")
        lines.append("")
        lines.append(
            "Only branch out to other KB sources if these don't have the answer."
        )
        return "\n".join(lines)

    def _inject(self, request):
        sources: tuple[str, ...] = ()
        runtime = getattr(request, "runtime", None)
        ctx = getattr(runtime, "context", None) if runtime else None
        if ctx is not None:
            sources = getattr(ctx, "allowed_sources", ()) or ()
        note = self._build_note(sources)
        if not note:
            return request
        sys_msg = request.system_message
        if sys_msg is None:
            new_sys = SystemMessage(content=note)
        else:
            new_sys = SystemMessage(content=f"{sys_msg.content}\n\n{note}")
        return request.override(system_message=new_sys)

    def wrap_model_call(self, request, handler):
        return handler(self._inject(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._inject(request))


class StudentProfileMiddleware(AgentMiddleware):
    """Append the stored student profile to the system prompt.

    Long-term semantic memory: the profile lives in the LangGraph store under
    ``("student", <student_id>)`` and is updated after each turn by
    ``stream_chat`` (see ``_update_student_profile``). InMemoryStore reads are
    dict lookups, so the sync ``store.get`` is fine on the async path too.
    """

    def _inject(self, request):
        runtime = getattr(request, "runtime", None)
        ctx = getattr(runtime, "context", None) if runtime else None
        student_id = getattr(ctx, "student_id", "") if ctx else ""
        if not student_id:
            return request
        store = getattr(runtime, "store", None) or STORE
        item = store.get(("student", student_id), "profile")
        profile = ""
        if item is not None:
            profile = str((item.value or {}).get("profile", "")).strip()
        if not profile:
            return request
        note = (
            "## Student profile (long-term memory)\n\n"
            f"{profile}\n\n"
            "Use this profile to calibrate level, language, and examples. "
            "Do not re-ask for information it already answers."
        )
        sys_msg = request.system_message
        if sys_msg is None:
            new_sys = SystemMessage(content=note)
        else:
            new_sys = SystemMessage(content=f"{sys_msg.content}\n\n{note}")
        return request.override(system_message=new_sys)

    def wrap_model_call(self, request, handler):
        return handler(self._inject(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._inject(request))


def _turn_id_for(request: Any) -> str:
    """The current turn's message_id from AppContext (the telemetry key)."""
    runtime = getattr(request, "runtime", None)
    ctx = getattr(runtime, "context", None) if runtime else None
    return getattr(ctx, "kb_session_id", "") if ctx else ""


def _cache_user_id_for_runtime(runtime: Any) -> str:
    ctx = getattr(runtime, "context", None) if runtime else None
    return str(getattr(ctx, "cache_user_id", "") or "") if ctx else ""


class DeepSeekCacheIsolationMiddleware(AgentMiddleware):
    """Attach DeepSeek ``user_id`` and guard the experimental request size."""

    def __init__(self, max_request_tokens: int | None = None) -> None:
        super().__init__()
        self.max_request_tokens = max_request_tokens

    def _isolate(self, request: Any) -> Any:
        request_messages = list(getattr(request, "messages", None) or [])
        system_message = getattr(request, "system_message", None)
        if system_message is not None:
            request_messages.insert(0, system_message)
        request_tokens = int(count_tokens_approximately(request_messages))
        if (
            self.max_request_tokens is not None
            and request_tokens > self.max_request_tokens
        ):
            raise RuntimeError(
                "Agent request exceeds the experiment safety guard: "
                f"{request_tokens:,} > {self.max_request_tokens:,} "
                "approximate tokens."
            )
        user_id = _cache_user_id_for_runtime(getattr(request, "runtime", None))
        if not user_id:
            return request
        settings = dict(getattr(request, "model_settings", None) or {})
        model = getattr(request, "model", None)
        model_extra_body: Any = None
        # Runnable bindings/fallbacks wrap the underlying provider model. Keep
        # its DeepSeek thinking toggle when adding the experiment-only user_id;
        # a call-time extra_body otherwise replaces the model-level mapping.
        for _ in range(4):
            model_extra_body = getattr(model, "extra_body", None)
            if model_extra_body is not None:
                break
            model = getattr(model, "runnable", None) or getattr(model, "bound", None)
            if model is None:
                break
        extra_body = dict(model_extra_body or {})
        extra_body.update(settings.get("extra_body") or {})
        extra_body["user_id"] = user_id
        settings["extra_body"] = extra_body
        return request.override(model_settings=settings)

    def wrap_model_call(self, request, handler):
        return handler(self._isolate(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._isolate(request))


class StableToolOutputCapMiddleware(AgentMiddleware):
    """Persistently cap tool output once, when it first enters agent history."""

    def __init__(self, max_bytes: int) -> None:
        super().__init__()
        self.max_bytes = max(1_024, int(max_bytes))

    @staticmethod
    def _decode_fragment(fragment: bytes) -> str:
        return fragment.decode("utf-8", errors="ignore")

    def _cap(self, request: Any, result: Any) -> Any:
        if not isinstance(result, ToolMessage):
            return result
        text = message_content_to_text(result.content)
        raw = text.encode("utf-8")
        if len(raw) <= self.max_bytes:
            return result

        marker = (
            f"\n\n[... tool output truncated at stable {self.max_bytes}-byte cap; "
            "middle omitted ...]\n\n"
        ).encode("utf-8")
        payload_budget = max(0, self.max_bytes - len(marker))
        head_bytes = payload_budget // 2
        tail_bytes = payload_budget - head_bytes
        capped = (
            self._decode_fragment(raw[:head_bytes])
            + marker.decode("utf-8")
            + self._decode_fragment(raw[-tail_bytes:] if tail_bytes else b"")
        )
        capped_bytes = len(capped.encode("utf-8"))
        metadata = {
            "original_bytes": len(raw),
            "original_chars": len(text),
            "retained_bytes": capped_bytes,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "max_bytes": self.max_bytes,
        }
        if getattr(result, "name", "") == "retrieve_tutor_context":
            # Capping inserts a head/tail marker and intentionally makes the
            # JSON content unparsable. Preserve only lightweight source
            # metadata so citations and the UI's chunk count stay accurate
            # without keeping the discarded chunk bodies in the checkpoint.
            retrieval_matches = collect_retrieval_source_matches(text)
            metadata["retrieval_matches"] = [
                _source_match_record(match) for match in retrieval_matches
            ]
        turn = _turn_id_for(request)
        record_turn_signal(turn, "tool_outputs_capped", 1)
        record_turn_signal(turn, "tool_output_original_bytes", len(raw))
        record_turn_signal(turn, "tool_output_retained_bytes", capped_bytes)
        additional = dict(result.additional_kwargs or {})
        additional["stable_tool_cap"] = metadata
        return result.model_copy(
            update={"content": capped, "additional_kwargs": additional}
        )

    def wrap_tool_call(self, request, handler):
        return self._cap(request, handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._cap(request, await handler(request))


class InstrumentedSummarizationMiddleware(SummarizationMiddleware):
    """SummarizationMiddleware with full event telemetry and loud failures."""

    MAX_SUMMARY_ATTEMPTS = 3
    RETRY_BASE_DELAY_SECONDS = 1.0

    def __init__(
        self,
        *args: Any,
        summary_input_guard_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.summary_input_guard_tokens = summary_input_guard_tokens

    def _configured_token_trigger(self) -> int | None:
        return next(
            (
                int(clause["tokens"])
                for clause in self._trigger_clauses
                if "tokens" in clause
            ),
            None,
        )

    @staticmethod
    def _last_ai_reported_tokens(messages: list[Any]) -> int:
        last_ai_message = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, AIMessage)
            ),
            None,
        )
        usage = getattr(last_ai_message, "usage_metadata", None) or {}
        return int(usage.get("total_tokens") or 0)

    def _token_trigger_evidence(
        self, messages: list[Any], approximate_tokens: int
    ) -> tuple[str, int]:
        trigger_tokens = self._configured_token_trigger()
        if trigger_tokens is None:
            return "non_token", 0
        approximate_met = approximate_tokens >= trigger_tokens
        provider_reported_met = self._should_summarize_based_on_reported_tokens(
            messages, float(trigger_tokens)
        )
        reported_tokens = self._last_ai_reported_tokens(messages)
        if approximate_met and provider_reported_met:
            return "approximate_and_provider_reported", reported_tokens
        if approximate_met:
            return "approximate", reported_tokens
        if provider_reported_met:
            return "provider_reported", reported_tokens
        return "other", reported_tokens

    def _plan_compaction(self, state: Any) -> dict[str, Any] | None:
        messages = state["messages"]
        self._ensure_message_ids(messages)
        total_tokens = int(self.token_counter(messages))
        if not self._should_summarize(messages, total_tokens):
            return None
        trigger_source, trigger_reported_tokens = self._token_trigger_evidence(
            messages, total_tokens
        )
        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None
        selected, preserved = self._partition_messages(messages, cutoff_index)
        trimmed = self._trim_messages_for_summary(selected)
        if not trimmed:
            raise RuntimeError("Summarization selected no usable input messages.")
        summary_input_tokens = int(self._partial_token_counter(trimmed))
        if (
            self.summary_input_guard_tokens is not None
            and summary_input_tokens > self.summary_input_guard_tokens
        ):
            raise RuntimeError(
                "Summarization input exceeds the experiment safety guard: "
                f"{summary_input_tokens:,} > "
                f"{self.summary_input_guard_tokens:,} approximate tokens."
            )
        return {
            "messages": messages,
            "pre_tokens": total_tokens,
            "trigger_source": trigger_source,
            "trigger_reported_tokens": trigger_reported_tokens,
            "selected": selected,
            "preserved": preserved,
            "trimmed": trimmed,
            "summary_input_tokens": summary_input_tokens,
        }

    def _summary_prompt_text(self, trimmed: list[Any]) -> str:
        formatted = get_buffer_string(trimmed, format="xml")
        return self.summary_prompt.format(messages=formatted).rstrip()

    def _summary_model(self, runtime: Any) -> Any:
        user_id = _cache_user_id_for_runtime(runtime)
        if not user_id:
            return self.model
        return self.model.bind(extra_body={"user_id": user_id})

    @staticmethod
    def _is_retryable_exception(exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError, httpx.TransportError)):
            return True
        status = getattr(exc, "status_code", None)
        if status is None:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status in {408, 409, 425, 429} or status >= 500
        return type(exc).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
        }

    @staticmethod
    def _retry_reason(exc: BaseException) -> str:
        detail = str(exc).strip()
        return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__

    @classmethod
    def _retry_delay(cls, attempt: int) -> float:
        return cls.RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))

    def _log_summary_retry(
        self, runtime: Any, attempt: int, reason: str, delay: float
    ) -> None:
        ctx = getattr(runtime, "context", None)
        message_id = str(getattr(ctx, "kb_session_id", "") or "") if ctx else ""
        logger.warning(
            "Retrying summarization after attempt %d/%d in %.1fs. "
            "message_id=%s reason=%s",
            attempt,
            self.MAX_SUMMARY_ATTEMPTS,
            delay,
            message_id,
            reason,
        )

    def _record_compaction(
        self, runtime: Any, plan: dict[str, Any], summary: str
    ) -> dict[str, Any]:
        if not summary.strip():
            raise RuntimeError("Summarization returned an empty summary.")
        if summary.startswith("Error generating summary:"):
            raise RuntimeError(summary)
        new_messages = self._build_new_messages(summary)
        preserved = plan["preserved"]
        turn = _cache_user_id_for_runtime(runtime)
        ctx = getattr(runtime, "context", None)
        message_id = str(getattr(ctx, "kb_session_id", "") or "") if ctx else ""
        event = {
            "event": "summarization",
            "summary_strategy": str(plan.get("summary_strategy") or "xml"),
            "configured_trigger_tokens": self._configured_token_trigger(),
            "trigger_source": plan["trigger_source"],
            "trigger_reported_tokens": int(plan["trigger_reported_tokens"]),
            "pre_compaction_tokens_approx": int(plan["pre_tokens"]),
            "pre_compaction_messages": len(plan["messages"]),
            "selected_messages": len(plan["selected"]),
            "selected_tokens_approx": int(
                self._partial_token_counter(plan["selected"])
            ),
            "summary_input_messages": len(plan["trimmed"]),
            "summary_input_tokens_approx": int(plan["summary_input_tokens"]),
            "summary_input_untrimmed": self.trim_tokens_to_summarize is None
            and len(plan["trimmed"]) == len(plan["selected"]),
            "summary_attempts": int(plan.get("summary_attempts") or 1),
            "summary_retry_reasons": list(plan.get("summary_retry_reasons") or []),
            "summary_output_tokens_approx": int(
                count_tokens_approximately([HumanMessage(content=summary)])
            ),
            "retained_tail_messages": len(preserved),
            "retained_tail_tokens_approx": int(self._partial_token_counter(preserved)),
            "post_compaction_tokens_approx": int(
                self._partial_token_counter([*new_messages, *preserved])
            ),
            "cache_user_id_present": bool(turn),
        }
        event.update(plan.get("summary_request_telemetry") or {})
        event.update(plan.get("summary_provider_telemetry") or {})
        record_turn_signal(message_id, "compactions_this_turn", 1)
        record_turn_signal_max(
            message_id,
            "max_pre_compaction_tokens_approx",
            int(plan["pre_tokens"]),
        )
        record_turn_event(message_id, event)
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved,
            ]
        }

    def before_model(self, state, runtime):
        plan = self._plan_compaction(state)
        if plan is None:
            return None
        model = self._summary_model(runtime)
        prompt = self._summary_prompt_text(plan["trimmed"])
        retry_reasons: list[str] = []
        for attempt in range(1, self.MAX_SUMMARY_ATTEMPTS + 1):
            try:
                response = model.invoke(
                    prompt,
                    config={"metadata": {"lc_source": "summarization"}},
                )
                plan["summary_provider_telemetry"] = self._summary_provider_telemetry(
                    response
                )
                summary = message_content_to_text(response.content).strip()
            except Exception as exc:
                if (
                    attempt >= self.MAX_SUMMARY_ATTEMPTS
                    or not self._is_retryable_exception(exc)
                ):
                    raise
                reason = self._retry_reason(exc)
            else:
                if summary:
                    plan["summary_attempts"] = attempt
                    plan["summary_retry_reasons"] = retry_reasons
                    return self._record_compaction(runtime, plan, summary)
                if attempt >= self.MAX_SUMMARY_ATTEMPTS:
                    raise RuntimeError(
                        "Summarization returned an empty summary after "
                        f"{attempt} attempts."
                    )
                reason = "empty response"
            retry_reasons.append(reason)
            delay = self._retry_delay(attempt)
            self._log_summary_retry(runtime, attempt, reason, delay)
            time.sleep(delay)
        raise AssertionError("unreachable")

    async def abefore_model(self, state, runtime):
        plan = self._plan_compaction(state)
        if plan is None:
            return None
        model = self._summary_model(runtime)
        prompt = self._summary_prompt_text(plan["trimmed"])
        retry_reasons: list[str] = []
        for attempt in range(1, self.MAX_SUMMARY_ATTEMPTS + 1):
            try:
                response = await model.ainvoke(
                    prompt,
                    config={"metadata": {"lc_source": "summarization"}},
                )
                plan["summary_provider_telemetry"] = self._summary_provider_telemetry(
                    response
                )
                summary = message_content_to_text(response.content).strip()
            except Exception as exc:
                if (
                    attempt >= self.MAX_SUMMARY_ATTEMPTS
                    or not self._is_retryable_exception(exc)
                ):
                    raise
                reason = self._retry_reason(exc)
            else:
                if summary:
                    plan["summary_attempts"] = attempt
                    plan["summary_retry_reasons"] = retry_reasons
                    return self._record_compaction(runtime, plan, summary)
                if attempt >= self.MAX_SUMMARY_ATTEMPTS:
                    raise RuntimeError(
                        "Summarization returned an empty summary after "
                        f"{attempt} attempts."
                    )
                reason = "empty response"
            retry_reasons.append(reason)
            delay = self._retry_delay(attempt)
            self._log_summary_retry(runtime, attempt, reason, delay)
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _summary_provider_telemetry(response: Any) -> dict[str, Any]:
        """Expose provider cache accounting on the compaction event itself."""
        usage = dict(getattr(response, "usage_metadata", None) or {})
        details = dict(usage.get("input_token_details") or {})
        input_tokens = int(usage.get("input_tokens") or 0)
        cache_read = int(details.get("cache_read") or 0)
        cache_creation = int(details.get("cache_creation") or 0)
        cache_miss = max(0, input_tokens - cache_read - cache_creation)
        return {
            "summary_provider_usage_reported": bool(usage),
            "summary_provider_cache_details_reported": "cache_read" in details,
            "summary_provider_input_tokens": input_tokens,
            "summary_provider_cache_read_tokens": cache_read,
            "summary_provider_cache_creation_tokens": cache_creation,
            "summary_provider_cache_miss_tokens": cache_miss,
            "summary_provider_output_tokens": int(usage.get("output_tokens") or 0),
            "summary_provider_cache_hit_ratio": (
                cache_read / input_tokens if input_tokens else None
            ),
        }


PREFIX_PRESERVING_COMPACTION_PROMPT = """Create a durable checkpoint summary of the older conversation prefix above.

The first {selected_messages} conversation messages will be replaced by your checkpoint. The final {retained_messages} messages (approximately {retained_tokens} tokens) will remain verbatim. Summarize only the older prefix; do not duplicate the retained tail except where a short reference is necessary to explain a dependency between old and recent work.

Preserve concrete facts, current values after corrections, decisions, constraints, unresolved work, prior tool evidence needed later, and enough causal detail to continue without re-running tools. Omit transient chatter and redundant wording. Never call a tool and do not answer the conversation's latest question.

Return only the checkpoint summary."""


class PrefixPreservingCompactionMiddleware(InstrumentedSummarizationMiddleware):
    """Compact through a structured prefix-extension request.

    LangChain's stock summarizer serializes selected messages into a new XML
    prompt, so even the summary-generation call loses the provider's cached
    prefix. This middleware instead runs after request-shaping middleware and
    binds the exact same model, tool schemas, tool choice, model settings, and
    system message. It then sends the unchanged *entire* current request prefix
    followed by one checkpoint instruction. The resulting checkpoint summary
    may overlap the retained tail; that conservative duplication matches the
    cache-friendly local Codex pattern and reduces the chance of lost evidence.

    Installing the resulting summary necessarily changes the *next* agent
    prefix; no client-side middleware can avoid that boundary without a
    provider-native opaque continuation/compaction primitive.
    """

    def before_model(self, state, runtime):
        # Planning must happen after all request-shaping middleware has produced
        # the final ModelRequest, so wrap_model_call owns the operation.
        return None

    async def abefore_model(self, state, runtime):
        return None

    def _checkpoint_instruction(self, plan: dict[str, Any]) -> HumanMessage:
        return HumanMessage(
            content=PREFIX_PRESERVING_COMPACTION_PROMPT.format(
                selected_messages=len(plan["selected"]),
                retained_messages=len(plan["preserved"]),
                retained_tokens=int(self._partial_token_counter(plan["preserved"])),
            )
        )

    def _summary_messages(
        self, request: Any, plan: dict[str, Any]
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        if request.system_message is not None:
            messages.append(request.system_message)
        messages.extend(request.messages)
        messages.append(self._checkpoint_instruction(plan))
        return messages

    def _prepare_summary_request(
        self, request: Any, plan: dict[str, Any]
    ) -> tuple[Any, list[BaseMessage]]:
        if request.response_format is not None:
            raise RuntimeError(
                "Structured-prefix compaction does not support a structured "
                "agent response format."
            )
        if list(request.messages) != list(plan["messages"]):
            raise RuntimeError(
                "Structured-prefix compaction requires the finalized model view "
                "to match checkpoint history exactly; an earlier middleware "
                "changed request.messages without persisting that change."
            )
        messages = self._summary_messages(request, plan)
        request_tokens = int(count_tokens_approximately(messages))
        if (
            self.summary_input_guard_tokens is not None
            and request_tokens > self.summary_input_guard_tokens
        ):
            raise RuntimeError(
                "Structured-prefix summary request exceeds the experiment safety "
                f"guard: {request_tokens:,} > "
                f"{self.summary_input_guard_tokens:,} approximate tokens."
            )

        settings = dict(request.model_settings or {})
        if request.tools:
            summary_model = request.model.bind_tools(
                request.tools,
                tool_choice=request.tool_choice,
                **settings,
            )
        else:
            summary_model = request.model.bind(**settings)

        system_tokens = int(
            count_tokens_approximately([request.system_message])
            if request.system_message is not None
            else 0
        )
        instruction_tokens = int(count_tokens_approximately([messages[-1]]))
        plan["summary_strategy"] = "structured_prefix"
        plan["summary_request_telemetry"] = {
            "summary_prefix_messages": len(request.messages),
            "summary_prefix_tokens_approx": int(
                count_tokens_approximately(request.messages)
            ),
            "summary_selected_messages": len(plan["trimmed"]),
            "summary_selected_tokens_approx": int(plan["summary_input_tokens"]),
            "summary_request_messages": len(messages),
            "summary_request_tokens_approx": request_tokens,
            "summary_request_is_strict_extension": True,
            "summary_instruction_selected_messages": len(plan["selected"]),
            "summary_instruction_retained_messages": len(plan["preserved"]),
            "summary_instruction_retained_tokens_approx": int(
                self._partial_token_counter(plan["preserved"])
            ),
            "summary_system_tokens_approx": system_tokens,
            "summary_instruction_tokens_approx": instruction_tokens,
            "summary_system_message_present": request.system_message is not None,
            "summary_tools_bound": len(request.tools or []),
            "summary_tool_choice_preserved": True,
            "summary_tool_choice_value": (
                str(request.tool_choice) if request.tool_choice is not None else None
            ),
            "summary_model_settings_keys": sorted(settings),
            "summary_cache_user_id_preserved": bool(
                (settings.get("extra_body") or {}).get("user_id")
            ),
        }
        return summary_model, messages

    def _invoke_prefix_summary(self, request: Any, plan: dict[str, Any]) -> str:
        model, messages = self._prepare_summary_request(request, plan)
        retry_reasons: list[str] = []
        for attempt in range(1, self.MAX_SUMMARY_ATTEMPTS + 1):
            try:
                response = model.invoke(
                    messages,
                    config={
                        "metadata": {
                            "lc_source": "summarization",
                            "compaction_strategy": "structured_prefix",
                        }
                    },
                )
                plan["summary_provider_telemetry"] = self._summary_provider_telemetry(
                    response
                )
                summary = message_content_to_text(response.content).strip()
            except Exception as exc:
                if (
                    attempt >= self.MAX_SUMMARY_ATTEMPTS
                    or not self._is_retryable_exception(exc)
                ):
                    raise
                reason = self._retry_reason(exc)
            else:
                if summary:
                    plan["summary_attempts"] = attempt
                    plan["summary_retry_reasons"] = retry_reasons
                    return summary
                if attempt >= self.MAX_SUMMARY_ATTEMPTS:
                    raise RuntimeError(
                        "Structured-prefix summarization returned an empty summary "
                        f"after {attempt} attempts."
                    )
                reason = "empty response"
            retry_reasons.append(reason)
            delay = self._retry_delay(attempt)
            self._log_summary_retry(request.runtime, attempt, reason, delay)
            time.sleep(delay)
        raise AssertionError("unreachable")

    async def _ainvoke_prefix_summary(self, request: Any, plan: dict[str, Any]) -> str:
        model, messages = self._prepare_summary_request(request, plan)
        retry_reasons: list[str] = []
        for attempt in range(1, self.MAX_SUMMARY_ATTEMPTS + 1):
            try:
                response = await model.ainvoke(
                    messages,
                    config={
                        "metadata": {
                            "lc_source": "summarization",
                            "compaction_strategy": "structured_prefix",
                        }
                    },
                )
                plan["summary_provider_telemetry"] = self._summary_provider_telemetry(
                    response
                )
                summary = message_content_to_text(response.content).strip()
            except Exception as exc:
                if (
                    attempt >= self.MAX_SUMMARY_ATTEMPTS
                    or not self._is_retryable_exception(exc)
                ):
                    raise
                reason = self._retry_reason(exc)
            else:
                if summary:
                    plan["summary_attempts"] = attempt
                    plan["summary_retry_reasons"] = retry_reasons
                    return summary
                if attempt >= self.MAX_SUMMARY_ATTEMPTS:
                    raise RuntimeError(
                        "Structured-prefix summarization returned an empty summary "
                        f"after {attempt} attempts."
                    )
                reason = "empty response"
            retry_reasons.append(reason)
            delay = self._retry_delay(attempt)
            self._log_summary_retry(request.runtime, attempt, reason, delay)
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    def _compacted_request_and_command(
        self, request: Any, plan: dict[str, Any], summary: str
    ) -> tuple[Any, list[BaseMessage]]:
        update = self._record_compaction(request.runtime, plan, summary)
        compacted = list(update["messages"])[1:]
        compacted_state = dict(request.state)
        compacted_state["messages"] = compacted
        return request.override(messages=compacted, state=compacted_state), compacted

    @staticmethod
    def _with_checkpoint_command(
        response: Any, compacted: list[BaseMessage]
    ) -> ExtendedModelResponse:
        # The model response command is applied first by create_agent. The
        # additional command then replaces the old checkpoint and re-adds that
        # same response, so it survives REMOVE_ALL_MESSAGES exactly once.
        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={
                    "messages": [
                        RemoveMessage(id=REMOVE_ALL_MESSAGES),
                        *compacted,
                        *response.result,
                    ]
                }
            ),
        )

    def wrap_model_call(self, request, handler):
        plan = self._plan_compaction(request.state)
        if plan is None:
            return handler(request)
        summary = self._invoke_prefix_summary(request, plan)
        compacted_request, compacted = self._compacted_request_and_command(
            request, plan, summary
        )
        response = handler(compacted_request)
        return self._with_checkpoint_command(response, compacted)

    async def awrap_model_call(self, request, handler):
        plan = self._plan_compaction(request.state)
        if plan is None:
            return await handler(request)
        summary = await self._ainvoke_prefix_summary(request, plan)
        compacted_request, compacted = self._compacted_request_and_command(
            request, plan, summary
        )
        response = await handler(compacted_request)
        return self._with_checkpoint_command(response, compacted)


class SlidingWindowMiddleware(AgentMiddleware):
    """Keep only the last N messages in the model's view; drop older ones.

    Recency-only working memory, no summary (Part C / Axis A). The drop is
    per-call: the checkpoint keeps the full thread, so this just shrinks each
    request without mutating stored state. The cut is advanced to the next user
    message so a kept window never begins with an orphaned tool result (which
    providers reject). Reports ``dropped_messages`` via the turn-signal registry,
    since nothing it does survives into the checkpoint for context_window_stats.
    """

    def __init__(self, keep: int) -> None:
        super().__init__()
        self._keep = max(1, int(keep))

    def _trim(self, request: Any) -> Any:
        messages = request.messages
        if len(messages) <= self._keep:
            return request
        naive_cut = len(messages) - self._keep
        # Cut at the nearest user message AT OR BEFORE the naive cut: keeps the
        # last ~N messages aligned to a turn boundary (so the current turn stays
        # intact and no tool result is orphaned), drops whole older turns, and
        # never silently no-ops when there is prior history to drop -- the
        # long-tool-turn case, where the naive cut lands inside the current
        # turn's tool loop and a forward scan would find no later user message.
        cut = 0
        for index in range(naive_cut, -1, -1):
            if getattr(messages[index], "type", "") == "human":
                cut = index
                break
        if cut <= 0:
            return request
        record_turn_signal_max(_turn_id_for(request), "dropped_messages", cut)
        return request.override(messages=messages[cut:])

    def wrap_model_call(self, request, handler):
        return handler(self._trim(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._trim(request))


class ObservationTruncationMiddleware(AgentMiddleware):
    """Head/tail-truncate large tool outputs in the model's view (Part C / Axis B).

    The dominant input-token source is tool output, not chat history (F1). This
    keeps the first and last slice of each oversized tool result so the model
    keeps the gist and any citation while shedding the bulk. Per-call only (the
    checkpoint is untouched). Reports ``truncated_tool_outputs`` and
    ``chars_saved``.
    """

    def __init__(self, head_chars: int, tail_chars: int, trigger_chars: int) -> None:
        super().__init__()
        self._head = max(0, int(head_chars))
        self._tail = max(0, int(tail_chars))
        self._trigger = max(self._head + self._tail + 1, int(trigger_chars))

    def _truncate(self, request: Any) -> Any:
        new_messages: list[Any] = []
        truncated = 0
        saved = 0
        for message in request.messages:
            content = getattr(message, "content", None)
            if (
                getattr(message, "type", "") == "tool"
                and isinstance(content, str)
                and len(content) > self._trigger
            ):
                dropped = len(content) - self._head - self._tail
                kept = (
                    f"{content[: self._head]}\n\n"
                    f"... [{dropped} chars truncated to save context] ...\n\n"
                    f"{content[-self._tail :]}"
                )
                # The marker boilerplate has a fixed cost; for content only just
                # over the trigger it can exceed the original. Never inflate.
                if len(kept) >= len(content):
                    new_messages.append(message)
                    continue
                saved += len(content) - len(kept)
                truncated += 1
                new_messages.append(message.model_copy(update={"content": kept}))
            else:
                new_messages.append(message)
        if not truncated:
            return request
        turn = _turn_id_for(request)
        record_turn_signal_max(turn, "truncated_tool_outputs", truncated)
        record_turn_signal_max(turn, "chars_saved", saved)
        return request.override(messages=new_messages)

    def wrap_model_call(self, request, handler):
        return handler(self._truncate(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._truncate(request))


_COMPRESS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[ \t]+"), " "),  # runs of spaces/tabs -> one space
    (re.compile(r"\n[ \t]+"), "\n"),  # strip line-leading whitespace
    (re.compile(r"\n{3,}"), "\n\n"),  # collapse blank-line runs
)


def _compress_text(text: str) -> str:
    for pattern, repl in _COMPRESS_PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


class PromptCompressionMiddleware(AgentMiddleware):
    """Deterministic per-call text compaction: a cheap prompt-compression stand-in.

    Collapses redundant whitespace in every message the model sees (Part C /
    Axis A). No LLM and no checkpoint change; it screens the cost/cache effect
    (F2) of rewriting the prompt prefix each turn. It is NOT a faithful
    token-importance compressor (LLMLingua-style), so its savings are a lower
    bound on what learned compression could achieve. Reports
    ``compressed_messages`` and ``chars_saved``.
    """

    def _compress(self, request: Any) -> Any:
        new_messages: list[Any] = []
        compressed = 0
        saved = 0
        for message in request.messages:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                shrunk = _compress_text(content)
                if len(shrunk) < len(content):
                    saved += len(content) - len(shrunk)
                    compressed += 1
                    new_messages.append(message.model_copy(update={"content": shrunk}))
                    continue
            new_messages.append(message)
        if not compressed:
            return request
        turn = _turn_id_for(request)
        record_turn_signal_max(turn, "compressed_messages", compressed)
        record_turn_signal_max(turn, "chars_saved", saved)
        return request.override(messages=new_messages)

    def wrap_model_call(self, request, handler):
        return handler(self._compress(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._compress(request))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _history_block_text(block: list[Any]) -> str:
    """Concatenate the str content of a turn-block for embedding."""
    parts = [m.content for m in block if isinstance(getattr(m, "content", None), str)]
    return "\n".join(parts).strip() or " "


def _default_history_embedder(texts: list[str]) -> list[list[float]]:
    """Embed turn-blocks with the same Cohere model retrieval uses (run-time cost)."""
    from .chroma_rag import DEFAULT_EMBED_MODEL, embed_texts

    retriever = get_retriever()
    return embed_texts(
        retriever._cohere,
        texts,
        input_type="search_document",
        model=DEFAULT_EMBED_MODEL,
    )


class InContextHistoryRetrievalMiddleware(AgentMiddleware):
    """Retrieve the relevant older turns instead of carrying or summarizing all.

    The principled answer to F9/F10 (Part C / Axis A subsystem): keep the last
    ``keep_recent`` turn-blocks (recency + the current turn) and, for everything
    older, embed each prior turn-block plus the current question and inject only
    the top-k most similar older blocks. Works at turn-BLOCK granularity (a user
    message plus the assistant/tool messages answering it) so a retrieved or
    dropped block never orphans a tool result. Per-call view only; reports
    ``history_retrievals`` (and ``dropped_messages``). Embeds via Cohere at run
    time -- that cost is part of what this arm is measured on (vs carrying the
    raw history). Offline tests inject a stub embedder.
    """

    def __init__(self, keep_recent: int, top_k: int, embed_fn: Any = None) -> None:
        super().__init__()
        self._keep_recent = max(1, int(keep_recent))
        self._top_k = max(0, int(top_k))
        self._embed_fn = embed_fn or _default_history_embedder

    @staticmethod
    def _blocks(messages: list[Any]) -> list[list[Any]]:
        blocks: list[list[Any]] = []
        current: list[Any] = []
        for message in messages:
            if getattr(message, "type", "") == "human" and current:
                blocks.append(current)
                current = []
            current.append(message)
        if current:
            blocks.append(current)
        return blocks

    def _select(self, request: Any) -> Any:
        messages = request.messages
        blocks = self._blocks(messages)
        if len(blocks) <= self._keep_recent:
            return request
        older = blocks[: -self._keep_recent]
        recent = blocks[-self._keep_recent :]
        if self._top_k >= len(older):
            return request  # all older blocks already fit; nothing to drop
        query = _history_block_text(recent[-1])
        older_texts = [_history_block_text(block) for block in older]
        turn = _turn_id_for(request)
        try:
            embed_inputs = [query] + older_texts
            vectors = self._embed_fn(embed_inputs)
        except Exception as exc:  # degrade to carrying full history
            logger.warning(
                "history retrieval embed failed; carrying history. error=%s", exc
            )
            return request
        query_vec, older_vecs = vectors[0], vectors[1:]
        ranked = sorted(
            range(len(older)),
            key=lambda i: _cosine_similarity(query_vec, older_vecs[i]),
            reverse=True,
        )
        keep_idx = set(ranked[: self._top_k])
        kept_older = [older[i] for i in range(len(older)) if i in keep_idx]
        new_messages = [m for block in (kept_older + recent) for m in block]
        if len(new_messages) >= len(messages):
            return request
        record_turn_signal_max(turn, "history_embedding_texts", len(embed_inputs))
        record_turn_signal_max(
            turn, "history_embedding_chars", sum(len(text) for text in embed_inputs)
        )
        record_turn_signal_max(turn, "history_retrievals", len(kept_older))
        record_turn_signal_max(
            turn, "dropped_messages", len(messages) - len(new_messages)
        )
        return request.override(messages=new_messages)

    def wrap_model_call(self, request, handler):
        return handler(self._select(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._select(request))


_HIERARCHICAL_SUMMARY_CACHE: dict[str, str] = {}


class HierarchicalSummarizationMiddleware(AgentMiddleware):
    """Map-reduce summary of older messages into the model's view (Part C / Axis A).

    Splits the message list into older (to compress) and the last ``keep_recent``;
    when the older block exceeds the trigger, summarizes it in groups, then
    summarizes the group summaries into one layered summary that replaces the
    older block in the request view. Per-call view only (checkpoint untouched).
    The summary is cached by older-block content hash, so a static long document
    (e.g. a lesson loaded turn 0) is summarized once and reused, not every turn.

    Caveat: the extra summarization calls go straight to the model (not through
    the turn's usage handler), so they are reported via ``summary_messages`` but
    are not added into the turn's est_cost -- hierarchical's summarization cost is
    a one-time-per-session extra on top of the per-turn numbers.
    """

    def __init__(
        self, model: Any, trigger_tokens: int, keep_recent: int, group_size: int
    ) -> None:
        super().__init__()
        self._model = model
        self._trigger = max(1, int(trigger_tokens))
        self._keep = max(1, int(keep_recent))
        self._group = max(1, int(group_size))

    def _text(self, message: Any) -> str:
        return message_content_to_text(getattr(message, "content", ""))

    def _summarize(self, text: str, instruction: str) -> str:
        response = self._model.invoke(
            [HumanMessage(content=f"{instruction}\n\n{text}")]
        )
        return message_content_to_text(response.content).strip()

    def _unit_summary(self, text: str) -> str:
        """Map step for one stable chunk, cached by content so a static long
        document is summarized once and reused across turns."""
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cached = _HIERARCHICAL_SUMMARY_CACHE.get(key)
        if cached is None:
            cached = self._summarize(
                text,
                "Summarize this part of a tutoring conversation/lesson, keeping "
                "concrete facts, steps, names, and examples:",
            )
            _HIERARCHICAL_SUMMARY_CACHE[key] = cached
        return cached

    def _maybe(self, request: Any) -> Any:
        messages = request.messages
        if len(messages) <= self._keep + 1:
            return request
        older = messages[: -self._keep]
        recent = messages[-self._keep :]
        older_text = "\n\n".join(self._text(m) for m in older)
        # Cheap ~4-chars/token estimate to avoid encoding the whole prefix each call.
        if len(older_text) < self._trigger * 4:
            return request
        # Stable map units: each older message, with oversized messages (a long
        # lesson loaded in one turn) split into fixed char-chunks. Boundaries are
        # stable across turns, so each unit's map summary is cached and computed
        # once; only newly-arrived messages and the final reduce re-run as the
        # session grows -- the realistic incremental cost, not a per-turn redo.
        chunk_chars = max(2_000, self._group * 1_200)
        units: list[str] = []
        for message in older:
            text = self._text(message)
            if not text.strip():
                continue
            if len(text) > chunk_chars:
                units += [
                    text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)
                ]
            else:
                units.append(text)
        partials = [self._unit_summary(u) for u in units]
        reduce_key = hashlib.md5("\n\n".join(partials).encode("utf-8")).hexdigest()
        summary = _HIERARCHICAL_SUMMARY_CACHE.get(reduce_key)
        if summary is None:
            summary = self._summarize(
                "\n\n".join(partials),
                "Combine these section summaries into one coherent hierarchical "
                "study summary (grouped by topic, concrete details kept, no "
                "redundancy):",
            )
            _HIERARCHICAL_SUMMARY_CACHE[reduce_key] = summary
        turn = _turn_id_for(request)
        record_turn_signal_max(turn, "summary_messages", 1)
        record_turn_signal_max(turn, "dropped_messages", len(older))
        new_messages = [
            SystemMessage(
                content=f"## Summary of earlier context (hierarchical)\n\n{summary}"
            ),
            *recent,
        ]
        return request.override(messages=new_messages)

    def wrap_model_call(self, request, handler):
        return handler(self._maybe(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._maybe(request))


def build_agent_middleware(
    model: Any, memory_config: MemoryConfig
) -> list[AgentMiddleware]:
    """Assemble the compaction/memory middleware stack for one preset."""
    if memory_config.summarization_strategy not in {"xml", "structured_prefix"}:
        raise ValueError(
            f"Unknown summarization strategy: {memory_config.summarization_strategy!r}"
        )
    if (
        memory_config.summarization_strategy == "structured_prefix"
        and not memory_config.summarization
    ):
        raise ValueError(
            "structured_prefix summarization strategy requires summarization=True"
        )
    middleware: list[AgentMiddleware] = []
    prefix_compactor: PrefixPreservingCompactionMiddleware | None = None
    if memory_config.experiment_mode:
        middleware.append(
            DeepSeekCacheIsolationMiddleware(
                memory_config.experiment_request_guard_tokens
            )
        )
    if memory_config.tool_output_cap_bytes is not None:
        middleware.append(
            StableToolOutputCapMiddleware(memory_config.tool_output_cap_bytes)
        )
    if memory_config.context_editing:
        middleware.append(
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=memory_config.context_editing_trigger_tokens,
                        keep=memory_config.context_editing_keep,
                        # Default (prod): retrieval results stay, only shell
                        # outputs clear (F3). The clear_retrieval_kb variant flips
                        # this to clear retrieval + KB outputs too, where the
                        # tokens actually are.
                        exclude_tools=(
                            ("retrieve_tutor_context",)
                            if memory_config.clear_excludes_retrieval
                            else ()
                        ),
                        placeholder=CLEARED_TOOL_OUTPUT_PLACEHOLDER,
                    )
                ],
                token_count_method="approximate",
            )
        )
    if memory_config.summarization:
        keep: tuple[str, int]
        if memory_config.summarization_keep_tokens is not None:
            keep = ("tokens", memory_config.summarization_keep_tokens)
        else:
            keep = ("messages", memory_config.summarization_keep_messages)
        summarization_kwargs: dict[str, Any] = {
            "model": model,
            "trigger": ("tokens", memory_config.summarization_trigger_tokens),
            "keep": keep,
            "trim_tokens_to_summarize": memory_config.summarization_trim_tokens,
        }
        # A custom summary prompt (selective_retention / context_reset) overrides
        # the library default; None keeps it.
        if memory_config.summary_prompt:
            summarization_kwargs["summary_prompt"] = memory_config.summary_prompt
        if memory_config.experiment_mode:
            summarization_kwargs["summary_input_guard_tokens"] = (
                memory_config.summarization_input_guard_tokens
            )
        if memory_config.summarization_strategy == "structured_prefix":
            if not memory_config.experiment_mode:
                raise ValueError(
                    "structured_prefix summarization is restricted to "
                    "experiment-mode presets"
                )
            prefix_compactor = PrefixPreservingCompactionMiddleware(
                **summarization_kwargs
            )
        else:
            summary_middleware = (
                InstrumentedSummarizationMiddleware
                if memory_config.experiment_mode
                else SummarizationMiddleware
            )
            middleware.append(summary_middleware(**summarization_kwargs))
    # Part C per-call-view mechanisms (each preset enables at most one). They
    # reshape only the request, not the checkpoint, and report via the
    # turn-signal registry.
    if memory_config.sliding_window_keep is not None:
        middleware.append(SlidingWindowMiddleware(memory_config.sliding_window_keep))
    if memory_config.truncate_tool_outputs:
        middleware.append(
            ObservationTruncationMiddleware(
                head_chars=memory_config.truncate_head_chars,
                tail_chars=memory_config.truncate_tail_chars,
                trigger_chars=memory_config.truncate_trigger_chars,
            )
        )
    if memory_config.compress_prompt:
        middleware.append(PromptCompressionMiddleware())
    if memory_config.history_retrieval_keep_recent is not None:
        middleware.append(
            InContextHistoryRetrievalMiddleware(
                keep_recent=memory_config.history_retrieval_keep_recent,
                top_k=memory_config.history_retrieval_top_k,
            )
        )
    if memory_config.hierarchical_summarize:
        middleware.append(
            HierarchicalSummarizationMiddleware(
                model,
                trigger_tokens=memory_config.hierarchical_trigger_tokens,
                keep_recent=memory_config.hierarchical_keep_recent,
                group_size=memory_config.hierarchical_group_size,
            )
        )
    if memory_config.longterm_memory:
        middleware.append(StudentProfileMiddleware())
    middleware.append(SourcePreferenceMiddleware())
    if prefix_compactor is not None:
        # wrap_model_call middleware compose left-to-right (first is outermost).
        # Keep this last so it observes the finalized system message from every
        # preceding request-shaping middleware before issuing the prefix call.
        middleware.append(prefix_compactor)
    return middleware


@lru_cache(maxsize=32)
def build_agent(
    model_name: str,
    enabled_tools: tuple[str, ...] = (),
    include_thoughts: bool = False,
    kb_agents_instructions: str | None = None,
    include_local_tools: bool = True,
    disable_kb: bool = False,
    memory_config: MemoryConfig | None = None,
):
    # kb_agents_instructions is part of the cache key on purpose: an agent
    # built before data/kb/AGENTS.md existed must not pin its degraded
    # system prompt for the process lifetime. memory_config (frozen, hashable)
    # is too: each preset gets its own agent.
    if memory_config is None:
        memory_config = resolve_memory_preset(model_name=model_name)
    model = build_chat_model(model_name, include_thoughts=include_thoughts)
    # An explicit empty source selection turns the knowledge base off: no
    # retrieval, no KB browsing, and a system prompt that says so.
    tools: list[Any]
    if not include_local_tools:
        tools = []
    elif disable_kb:
        tools = [retrieve_tutor_context]
    else:
        tools = [retrieve_tutor_context, run_kb_command]
    middleware = build_agent_middleware(model, memory_config)
    enabled = set(enabled_tools)
    # Provider-native web tools are bound from the REQUESTED model, so a
    # fallback client never receives them (see build_chat_model).
    if supports_gemini_tool_combination(model_name):
        if "web_search" in enabled:
            tools.append({"google_search": {}})
        if "url_context" in enabled:
            tools.append({"url_context": {}})
        if enabled & {"web_search", "url_context"}:
            middleware.append(GeminiServerSideToolsMiddleware())
    elif is_anthropic_model(model_name):
        # Anthropic combines server-side tools with function tools natively --
        # no flag, no generation gate (unlike Gemini above); they just share
        # this list.
        #
        # allowed_callers=["direct"] is LOAD-BEARING, not boilerplate. The
        # _20260209 tools default to allowing the code-execution caller
        # (programmatic tool calling), which Haiku 4.5 does not support:
        # dropping this field 400s with "'claude-haiku-4-5-20251001' does not
        # support programmatic tool calling ... Explicitly set
        # allowed_callers=["direct"]". Verified live against both web_fetch
        # variants.
        if "web_search" in enabled:
            tools.append(
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "allowed_callers": ["direct"],
                }
            )
        if "web_fetch" in enabled:
            tools.append(
                {
                    "type": "web_fetch_20260209",
                    "name": "web_fetch",
                    "allowed_callers": ["direct"],
                }
            )
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=build_system_prompt(
            model_name,
            enabled_tools,
            kb_agents_instructions,
            include_local_tools=include_local_tools,
            disable_kb=disable_kb,
        ),
        context_schema=AppContext,
        checkpointer=CHECKPOINTER,
        store=STORE,
        middleware=middleware,
    )


def model_provider_and_name(model_name: str) -> tuple[str, str]:
    provider_model = normalize_model_name(model_name)
    provider, _, actual_model = provider_model.partition(":")
    return provider or "unknown", actual_model or provider_model


def effective_tool_names(
    model_name: str,
    enabled_tools: tuple[str, ...],
    include_local_tools: bool = True,
    disable_kb: bool = False,
) -> tuple[str, ...]:
    if not include_local_tools:
        names = []
    elif disable_kb:
        names = ["retrieve_tutor_context"]
    else:
        names = ["retrieve_tutor_context", "run_kb_command"]
    enabled = set(enabled_tools)
    if supports_gemini_tool_combination(model_name):
        if "web_search" in enabled:
            names.append("google_search")
        if "url_context" in enabled:
            names.append("url_context")
    elif is_anthropic_model(model_name):
        if "web_search" in enabled:
            names.append("web_search")
        if "web_fetch" in enabled:
            names.append("web_fetch")
    return tuple(names)


_QUERY_URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]\"']+")


def extract_query_urls(text: str) -> list[str]:
    """Pull http(s) URLs out of a user query, in order, deduped."""
    urls: list[str] = []
    for raw_url in _QUERY_URL_PATTERN.findall(text):
        url = raw_url.rstrip(".,;:!?")
        if url and url not in urls:
            urls.append(url)
    return urls


def url_context_evidence(query: str) -> list[SourceMatch]:
    """Evidence for URLs the user pasted when `url_context` is enabled.

    langchain-google-genai (4.2.4) never surfaces Gemini's
    `url_context_metadata`, so fetches are invisible to the stream. Treating
    pasted URLs as web evidence at least lets the model's citation of a
    fetched page resolve to a source card instead of being dropped.
    """
    return [
        SourceMatch(
            doc_id=f"url_context::{url}",
            title=url,
            url=url,
            source_key="url_context",
            source_label="Web page",
            score=1.0,
            group="web",
        )
        for url in extract_query_urls(query)
    ]


def agent_run_config(
    request: ChatRequest,
    active_thread_id: str,
    message_id: str,
    memory_preset: str = "",
) -> dict[str, Any]:
    provider, actual_model = model_provider_and_name(request.model_name)
    tools = effective_tool_names(
        request.model_name,
        request.enabled_tools,
        include_local_tools=bool(request.source_keys),
        disable_kb=bool(request.disable_kb),
    )
    source_labels = [
        SOURCE_KEY_TO_LABEL.get(source_key, source_key)
        for source_key in request.source_keys
    ]
    preset = memory_preset or resolve_memory_preset(model_name=request.model_name).name
    config = thread_config(active_thread_id)
    config.update(
        {
            "run_name": "ai-tutor-agent-turn",
            "tags": [
                "ai-tutor-app",
                "knowledge-base-chatbot",
                f"provider:{provider}",
                f"model:{actual_model}",
                f"memory:{preset}",
                *(f"tool:{tool_name}" for tool_name in tools),
            ],
            "metadata": {
                "app": "ai-tutor-app",
                "thread_id": active_thread_id,
                "conversation_id": active_thread_id,
                "message_id": message_id,
                "model_provider": provider,
                "model_name": actual_model,
                "requested_model": request.model_name,
                "available_tools": list(tools),
                "enabled_tool_toggles": list(request.enabled_tools),
                "source_keys": list(request.source_keys),
                "source_labels": source_labels,
                "include_reasoning": bool(request.include_reasoning),
                "memory_preset": preset,
                "student_id": request.student_id,
                "cache_user_id": request.cache_user_id,
            },
        }
    )
    return config


_PROFILE_UPDATE_PROMPT = """\
You maintain a short profile of a student for an AI tutor.

Current profile (may be empty):
{profile}

Latest exchange:
Student: {query}
Tutor: {answer}

Rewrite the profile in at most 5 short lines. Keep only durable facts useful
for future tutoring: skill level, goals, preferred language/tools, weak
topics, current course/lesson. Drop one-off details. Return only the profile
text; return NONE if nothing durable is known yet."""


async def _update_student_profile(
    request: ChatRequest, answer: str, usage_handler: TurnUsageHandler
) -> None:
    """Refresh the stored profile from this turn (one small extra LLM call).

    The call reports into ``usage_handler`` so profile-memory presets pay for
    their own upkeep in the turn's token/cost numbers.
    """
    model = build_chat_model(request.model_name)
    prompt = _PROFILE_UPDATE_PROMPT.format(
        profile=get_student_profile(request.student_id) or "(empty)",
        query=request.query[:1500],
        answer=answer[:3000],
    )
    response = await model.ainvoke(
        [HumanMessage(content=prompt)],
        config={
            "run_name": "student-profile-update",
            "callbacks": [usage_handler],
            "metadata": {"lc_source": "student_profile_update"},
        },
    )
    text = message_content_to_text(response.content).strip()
    if not text or text.upper() == "NONE":
        return
    set_student_profile(request.student_id, text[:1200])


async def stream_chat(request: ChatRequest) -> AsyncIterator[ChatEvent]:
    turn_started = time.monotonic()
    # Raises on unknown/incompatible preset names: a mislabeled experiment run
    # must fail, not silently fall back. The API layer pre-validates to a 422.
    memory_config = resolve_memory_preset(
        request.memory_preset,
        model_name=request.model_name,
    )
    # Token usage and call counts come from the model calls themselves
    # (middleware summarization and profile updates included), so eval runs
    # never depend on LangSmith being enabled or within plan limits.
    usage_handler = TurnUsageHandler()
    first_text_at: float | None = None
    # First streamed token of ANY kind (reasoning, tool call, or answer text),
    # vs first_text_at which waits for the visible answer. On an agentic turn
    # the first token is usually the model's tool-call decision, so this
    # measures "when did the model start moving" (a UX/responsiveness signal)
    # separately from time-to-first-answer (ttft_ms), which trails the whole
    # tool-call loop.
    first_token_at: float | None = None
    normalized_history = normalize_history(request.history)
    retrieval_evidence: dict[str, SourceMatch] = {}
    shell_evidence: dict[str, SourceMatch] = {}
    web_evidence: dict[str, SourceMatch] = {}
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    answer_chunks: list[str] = []
    completed_answer = ""
    message_id = uuid4().hex
    # Per-turn tally for middlewares whose effect never reaches the checkpoint
    # (sliding window, truncation, ...); merged into context_stats below.
    reset_turn_signals(message_id)
    include_reasoning = bool(request.include_reasoning) and (
        is_google_genai_model(request.model_name)
        or is_anthropic_model(request.model_name)
        or is_deepseek_model(request.model_name)
    )
    # Gemini streams each thought summary as one complete block; Anthropic
    # streams partial fragments of a single thought. The encoder uses this to
    # decide whether consecutive deltas need a paragraph break between them.
    reasoning_deltas_are_blocks = is_google_genai_model(request.model_name)
    google_search = GoogleSearchActivity(message_id, web_evidence)
    include_local_tools = bool(request.source_keys)
    disable_kb = bool(request.disable_kb)
    effective_tools = effective_tool_names(
        request.model_name,
        request.enabled_tools,
        include_local_tools=include_local_tools,
        disable_kb=disable_kb,
    )
    if "url_context" in effective_tools:
        _record_evidence(web_evidence, url_context_evidence(request.query))

    logger.info("Running query: %s", request.query)

    # Agent construction (model client + middlewares on a cache miss) and the
    # checkpointer sync are synchronous; run them off the event loop so one
    # request's setup doesn't stall every other in-flight SSE stream.
    def _build_agent_for_request():
        return build_agent(
            request.model_name,
            enabled_tools=tuple(request.enabled_tools),
            include_thoughts=include_reasoning,
            kb_agents_instructions=(
                ensure_kb_agents_instructions()
                if (include_local_tools and not disable_kb)
                else None
            ),
            include_local_tools=include_local_tools,
            disable_kb=disable_kb,
            memory_config=memory_config,
        )

    agent = await asyncio.to_thread(_build_agent_for_request)
    _evict_idle_threads()
    active_thread_id, fork_checkpoint_id = await asyncio.to_thread(
        sync_thread_with_history,
        agent,
        request.thread_id.strip() or new_thread_id(),
        normalized_history,
        model_provider_and_name(request.model_name)[0],
    )
    _touch_thread(active_thread_id)
    run_config = agent_run_config(
        request, active_thread_id, message_id, memory_config.name
    )
    run_config["callbacks"] = [usage_handler]
    if fork_checkpoint_id:
        # Time travel: run from the checkpoint matching the history the
        # client kept; the turns after it become an abandoned branch.
        run_config["configurable"]["checkpoint_id"] = fork_checkpoint_id

    yield ChatEvent("thread_started", {"thread_id": active_thread_id})
    yield ChatEvent("message_started", {"message_id": message_id})

    try:
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": request.query}]},
            run_config,
            context=AppContext(
                allowed_sources=request.source_keys,
                kb_session_id=message_id,
                kb_command_limit=DEFAULT_KB_COMMAND_LIMIT,
                student_id=request.student_id,
                cache_user_id=request.cache_user_id,
                retrieval_budget=request.retrieval_budget,
                retriever_kind=request.retriever,
            ),
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if chunk["type"] == "messages":
                token, metadata = chunk["data"]
                if not isinstance(token, AIMessageChunk):
                    continue

                # SummarizationMiddleware's internal LLM call is tagged this;
                # skip so its summary template doesn't leak into the answer.
                if metadata.get("lc_source") == "summarization":
                    continue

                step = str(metadata.get("langgraph_node", ""))
                if include_reasoning:
                    thought_text = "\n\n".join(extract_reasoning_deltas(token))
                    if thought_text:
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                        yield ChatEvent(
                            "reasoning_delta",
                            {
                                "message_id": message_id,
                                "step": step,
                                "text": thought_text,
                                "is_block": reasoning_deltas_are_blocks,
                            },
                        )

                # Anthropic/OpenAI stream a tool call across several chunks:
                # only the first fragment carries the provider id + name, the
                # rest are partial-args continuations. Announce each call once,
                # on the id-bearing fragment.
                for tool_call in getattr(token, "tool_calls", []) or []:
                    tool_call_id = str(tool_call.get("id") or "")
                    if not tool_call_id or tool_call_id in tool_calls_by_id:
                        continue
                    tool_calls_by_id[tool_call_id] = tool_call
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    yield ChatEvent(
                        "tool_call_started",
                        {
                            "message_id": message_id,
                            "call_id": tool_call_id,
                            "tool_name": str(tool_call.get("name", "tool")),
                            "args": tool_call.get("args"),
                            "args_text": format_tool_args(tool_call.get("args")),
                        },
                    )

                text_delta = token.text or ""
                if not text_delta and token.content:
                    text_delta = message_content_to_text(token.content)
                if text_delta:
                    if first_text_at is None:
                        first_text_at = time.monotonic()
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    answer_chunks.append(text_delta)
                    yield ChatEvent(
                        "text_delta",
                        {
                            "message_id": message_id,
                            "text": text_delta,
                        },
                    )

                search_started = google_search.observe(
                    getattr(token, "response_metadata", None)
                )
                if search_started:
                    yield search_started
                continue

            if chunk["type"] != "updates":
                continue

            for step, update in chunk["data"].items():
                if not isinstance(update, dict):
                    continue
                messages = update.get("messages")
                if not messages:
                    continue
                message = messages[-1]
                # Only the tools node reports fresh completions. Middleware
                # state rewrites (e.g. SummarizationMiddleware replacing
                # history mid-turn) also end in the preserved most-recent
                # ToolMessage and must not re-emit it as new tool activity.
                if step == "tools" and getattr(message, "type", None) == "tool":
                    payload = message_content_to_text(message.content)
                    cap_metadata = dict(
                        (getattr(message, "additional_kwargs", None) or {}).get(
                            "stable_tool_cap"
                        )
                        or {}
                    )
                    tool_call_id = str(
                        getattr(message, "tool_call_id", "") or uuid4().hex
                    )
                    tool_name = str(getattr(message, "name", "tool"))
                    call_matches: list[SourceMatch] = []
                    if tool_name == "retrieve_tutor_context":
                        call_matches = _source_matches_from_records(
                            cap_metadata.get("retrieval_matches")
                        )
                        if not call_matches:
                            call_matches = collect_retrieval_source_matches(payload)
                        _record_evidence(retrieval_evidence, call_matches)

                    tool_call = tool_calls_by_id.get(
                        tool_call_id,
                        {"name": getattr(message, "name", "tool"), "args": None},
                    )
                    if tool_name == "run_kb_command":
                        args = tool_call.get("args")
                        command = str(
                            args.get("command") if isinstance(args, dict) else ""
                        )
                        shell_matches = extract_shell_source_matches(command, payload)
                        _record_evidence(shell_evidence, shell_matches.referenced)
                        call_matches = shell_matches.browsed
                    yield ChatEvent(
                        "tool_call_completed",
                        {
                            "message_id": message_id,
                            "step": step,
                            "call_id": tool_call_id,
                            "tool_name": str(
                                tool_call.get(
                                    "name",
                                    getattr(message, "name", "tool"),
                                )
                            ),
                            "args": tool_call.get("args"),
                            "args_text": format_tool_args(tool_call.get("args")),
                            "output_text": payload,
                            "output_was_capped": bool(cap_metadata),
                            "output_original_bytes": cap_metadata.get("original_bytes"),
                            "output_original_chars": cap_metadata.get("original_chars"),
                            "output_retained_bytes": cap_metadata.get("retained_bytes"),
                            "output_sha256": cap_metadata.get("sha256"),
                            "matches": [
                                source_match_payload(
                                    match,
                                    message_id=message_id,
                                    call_id=tool_call_id,
                                )
                                for match in call_matches
                            ],
                        },
                    )
                    continue

                if step != "model" or getattr(message, "type", None) != "ai":
                    continue

                search_started = google_search.observe(
                    getattr(message, "response_metadata", None)
                )
                if search_started:
                    yield search_started
                # This model step is complete, so any search it ran is done:
                # close the activity now instead of when the whole turn ends.
                search_step_completed = google_search.completed_event()
                if search_step_completed:
                    yield search_step_completed

                if is_anthropic_model(request.model_name):
                    anthropic_updates, anthropic_tool_uses = (
                        extract_anthropic_source_matches(
                            message.content,
                            web_evidence,
                        )
                    )
                    for tool_use_id, tool_use in anthropic_tool_uses.items():
                        if tool_use_id in tool_calls_by_id:
                            continue
                        tool_calls_by_id[tool_use_id] = tool_use
                        yield ChatEvent(
                            "tool_call_started",
                            {
                                "message_id": message_id,
                                "call_id": tool_use_id,
                                "tool_name": tool_use["name"],
                                "args": tool_use.get("args"),
                                "args_text": format_tool_args(tool_use.get("args")),
                            },
                        )
                    for tool_use_id, new_matches in anthropic_updates.items():
                        if not tool_use_id:
                            continue
                        tool_call = tool_calls_by_id.get(tool_use_id, {})
                        tool_name = str(tool_call.get("name") or "web_search")
                        plural = "" if len(new_matches) == 1 else "s"
                        output_text = (
                            f"{tool_name} returned {len(new_matches)} result{plural}."
                        )
                        yield ChatEvent(
                            "tool_call_completed",
                            {
                                "message_id": message_id,
                                "step": step,
                                "call_id": tool_use_id,
                                "tool_name": tool_name,
                                "args": tool_call.get("args"),
                                "args_text": format_tool_args(tool_call.get("args")),
                                "output_text": output_text,
                                "matches": [
                                    source_match_payload(
                                        match,
                                        message_id=message_id,
                                        call_id=tool_use_id,
                                    )
                                    for match in new_matches
                                ],
                            },
                        )

                if getattr(message, "tool_calls", None):
                    # The completed message has fully parsed args; streamed
                    # fragments may have announced the call with empty args.
                    # Publish the completed arguments now, before the tools
                    # node returns, so a long-running search shows its query
                    # while it is running rather than only with its result.
                    for tool_call in message.tool_calls:
                        call_id = str(tool_call.get("id") or "")
                        if not call_id:
                            continue
                        previous = tool_calls_by_id.get(call_id)
                        tool_calls_by_id[call_id] = tool_call
                        if previous is None:
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            yield ChatEvent(
                                "tool_call_started",
                                {
                                    "message_id": message_id,
                                    "call_id": call_id,
                                    "tool_name": str(tool_call.get("name", "tool")),
                                    "args": tool_call.get("args"),
                                    "args_text": format_tool_args(
                                        tool_call.get("args")
                                    ),
                                },
                            )
                            continue
                        args = tool_call.get("args")
                        if args == previous.get("args"):
                            continue
                        yield ChatEvent(
                            "tool_call_args_available",
                            {
                                "message_id": message_id,
                                "call_id": call_id,
                                "tool_name": str(tool_call.get("name", "tool")),
                                "args": args,
                                "args_text": format_tool_args(args),
                            },
                        )
                    continue
                completed_answer = message_content_to_text(message.content)
    finally:
        _clear_kb_command_budget(message_id)
        answer = "".join(answer_chunks).strip() or completed_answer.strip()
        # Record the turn as the client renders it (including a partial answer
        # from an aborted stream) so the next request can be matched back to
        # this thread even after summarization rewrites the checkpoint.
        transcript = normalized_history + (
            ChatTurn("user", request.query.strip()),
            ChatTurn("assistant", answer),
        )
        _record_thread_transcript(active_thread_id, transcript)
        _record_fork_point(
            active_thread_id,
            len(transcript),
            _latest_checkpoint_id(agent, active_thread_id),
        )
        # Take ownership from the model that actually answered, which is not
        # request.model_name when the DeepSeek fallback fired: Gemini is what
        # wrote the checkpointed blocks, so Gemini is what the next turn would
        # have to replay. usage_metadata is keyed by served model and only a
        # call that reached on_llm_end reports usage, so a failed primary never
        # appears; the last key is what wrote the final messages.
        served_models = [model for model in usage_handler.usage_metadata if model]
        if served_models:
            _record_thread_provider(
                active_thread_id,
                model_provider_and_name(served_models[-1])[0],
            )
        _touch_thread(active_thread_id)

    search_completed = google_search.completed_event()
    if search_completed:
        yield search_completed

    # Run citation resolution off the loop: it can shell out to the KB
    # sandbox per unresolved reference and, on first use, parse the whole
    # corpus manifest.
    matched_sources = await asyncio.to_thread(
        resolve_answer_citations,
        answer,
        retrieval_evidence=retrieval_evidence,
        shell_evidence=shell_evidence,
        web_evidence=web_evidence,
        # url_context is the only tool whose fetches are invisible to the
        # stream (the other web tools report results into web_evidence),
        # so only then is a cited-but-unmatched URL plausibly a fetched
        # page rather than a memory link: keep it as a low-trust Web chip.
        keep_unresolved_sources="url_context" in effective_tools,
    )

    # Fallback when the model writes no inline `[label](url)` (Gemini grounds
    # via metadata segments instead): surface collected evidence as cards.
    if not matched_sources:
        seen_keys: set[str] = set()
        for evidence in (web_evidence, retrieval_evidence, shell_evidence):
            for src in evidence.values():
                key = source_match_key(src)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                matched_sources.append(src)

    for source_match in matched_sources:
        yield ChatEvent(
            "source_match",
            source_match_payload(source_match, message_id=message_id),
        )

    # Long-term memory upkeep happens before the stats event so the profile
    # update's tokens land in this turn's bill.
    if memory_config.longterm_memory and request.student_id and answer:
        try:
            await _update_student_profile(request, answer, usage_handler)
        except Exception as exc:
            logger.warning("Student profile update failed. error=%s", exc)

    # Aborted streams never reach this event; eval runners consume the full
    # stream, and the UI treats the meter as best-effort.
    state_messages = agent.get_state(thread_config(active_thread_id)).values.get(
        "messages", []
    )
    totals = usage_totals(usage_handler.usage_metadata)
    cost_breakdown = aggregate_cost_breakdown(usage_handler.usage_metadata)
    model_calls = list(usage_handler.model_calls)
    summarization_cost = sum(
        float((call.get("cost") or {}).get("total_usd") or 0)
        for call in model_calls
        if call.get("source") == "summarization"
    )
    turn_events = pop_turn_events(message_id)
    turn_signals = pop_turn_signals(message_id)
    total_ms = int((time.monotonic() - turn_started) * 1000)
    yield ChatEvent(
        "context_stats",
        {
            "message_id": message_id,
            "thread_id": active_thread_id,
            "memory_preset": memory_config.name,
            "llm_calls": usage_handler.llm_calls,
            # Per-model breakdown for trace bundles; the totals below are
            # what the UI meter renders.
            "usage_by_model": {
                model_key: dict(usage)
                for model_key, usage in usage_handler.usage_metadata.items()
            },
            "model_calls": model_calls,
            **totals,
            "cache_miss_tokens": max(
                0,
                totals["input_tokens"]
                - totals["cache_read_tokens"]
                - totals["cache_creation_tokens"],
            ),
            "est_cost_usd": estimate_cost_usd(usage_handler.usage_metadata),
            "cost_breakdown": cost_breakdown,
            "summarization_cost_usd": summarization_cost,
            "max_request_context_tokens_approx": max(
                (
                    int(call.get("request_context_tokens_approx") or 0)
                    for call in model_calls
                ),
                default=0,
            ),
            "compaction_events": turn_events,
            "ttft_ms": (
                int((first_text_at - turn_started) * 1000)
                if first_text_at is not None
                else None
            ),
            # Time to the first streamed token of any kind (reasoning/tool/text);
            # <= ttft_ms. Isolates raw model responsiveness from the tool-call
            # loop that ttft_ms includes. See first_token_at above.
            "time_to_first_token_ms": (
                int((first_token_at - turn_started) * 1000)
                if first_token_at is not None
                else None
            ),
            "total_ms": total_ms,
            **context_window_stats(state_messages, CLEARED_TOOL_OUTPUT_PLACEHOLDER),
            # Signals from per-call-view middlewares (this turn only); absent
            # keys simply mean that mechanism did not fire.
            **turn_signals,
        },
    )
    yield ChatEvent(
        "message_completed",
        {
            "message_id": message_id,
            "thread_id": active_thread_id,
            "answer": answer,
        },
    )
