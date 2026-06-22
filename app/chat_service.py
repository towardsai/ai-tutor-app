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

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    SummarizationMiddleware,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from .chat_types import ChatEvent, ChatRequest, ChatTurn, SourceMatch
from .memory_presets import (
    DEFAULT_MEMORY_PRESET,
    MEMORY_PRESETS,
    MemoryConfig,
    resolve_memory_preset,
)
from .telemetry import (
    TurnUsageHandler,
    context_window_stats,
    estimate_cost_usd,
    pop_turn_signals,
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
    extract_thought_summaries,
)
from .config import (
    BM25_INDEX_PATH,
    COURSE_SOURCE_KEYS,
    DEFAULT_SELECTED_SOURCE_KEYS,
    DOCUMENT_DICT_PATH,
    SOURCE_KEY_TO_LABEL,
    VECTOR_COLLECTION_NAME,
    VECTOR_DB_DIR,
    ensure_local_vector_db,
)

logger = logging.getLogger(__name__)

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
}


@tool(args_schema=RETRIEVE_TUTOR_CONTEXT_SCHEMA)
def retrieve_tutor_context(query: str, runtime: ToolRuntime[AppContext]) -> str:
    """Retrieve relevant course and documentation context for an AI tutor question."""
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
        },
        "max_output_chars": {
            "type": "integer",
            "description": "Maximum stdout/stderr characters to return, capped by the runtime.",
            "default": 40000,
        },
    },
    "required": ["command"],
}


@tool(args_schema=RUN_KB_COMMAND_SCHEMA)
def run_kb_command(
    command: str,
    runtime: ToolRuntime[AppContext],
    timeout_seconds: int = 8,
    max_output_chars: int = 40000,
) -> str:
    """Run a safe, read-only terminal-style command inside the local KB."""
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
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )
        return format_command_payload(result)
    except (KbCommandError, OSError) as exc:
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
) -> tuple[str, str]:
    """Pick the thread (and optional fork checkpoint) for this request.

    Returns ``(thread_id, fork_checkpoint_id)``. A non-empty checkpoint id
    means the run must fork the thread from that checkpoint (LangGraph
    time travel) instead of continuing from its latest state.
    """
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
    return branched_thread_id, ""


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


def extract_shell_source_matches(command: str, output_text: str) -> list[SourceMatch]:
    raw_paths = extract_raw_paths(output_text)
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = []
    if tokens:
        executable = tokens[0]
        if executable == "cat":
            raw_paths.extend(tokens[1:])
        elif executable == "sed" and len(tokens) >= 4:
            raw_paths.append(tokens[-1])
        elif executable == "head":
            start = 1
            if len(tokens) > 2 and tokens[1] == "-n":
                start = 3
            raw_paths.extend(tokens[start:])
    matches: list[SourceMatch] = []
    seen: set[str] = set()
    for path in raw_paths:
        match = resolve_manifest_reference(path)
        if not match:
            continue
        key = source_match_key(match)
        if key in seen:
            continue
        seen.add(key)
        matches.append(match)
    return matches


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


def build_chat_model(model_name: str, include_thoughts: bool = False):
    provider_model = normalize_model_name(model_name)
    provider, _, actual_model = provider_model.partition(":")

    if provider == "openai":
        return ChatOpenAI(model=actual_model, temperature=1)
    if provider == "openrouter":
        # OpenAI-compatible gateway for open models (DeepSeek, Qwen, ...).
        # stream_usage=True makes the streamed response carry token usage, so
        # context_stats / cost telemetry populates (some OpenAI-compatible
        # endpoints, e.g. Ollama, omit usage and leave token counts at 0).
        # Routing is left to OpenRouter with fallbacks enabled: pinning a single
        # provider with allow_fallbacks=False made batches die on a backend's
        # transient 429 instead of routing around it. Fallback across providers
        # is what keeps a long eval run reliable.
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
        # DeepSeek first-party API (OpenAI-compatible, base https://api.deepseek.com).
        # stream_usage=True so the streamed response carries token usage ->
        # context_stats / cost telemetry populates, including the cached-prefix
        # tokens that drive the cost comparison (DeepSeek caches prefixes
        # automatically; cache-hit input is ~50x cheaper than cache-miss). Reads
        # DEEPSEEK_API_KEY.
        return ChatOpenAI(
            model=actual_model,
            temperature=1,
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            stream_usage=True,
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
    middleware: list[AgentMiddleware] = []
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
        summarization_kwargs: dict[str, Any] = {
            "model": model,
            "trigger": ("tokens", memory_config.summarization_trigger_tokens),
            "keep": ("messages", memory_config.summarization_keep_messages),
        }
        # A custom summary prompt (selective_retention / context_reset) overrides
        # the library default; None keeps it.
        if memory_config.summary_prompt:
            summarization_kwargs["summary_prompt"] = memory_config.summary_prompt
        middleware.append(SummarizationMiddleware(**summarization_kwargs))
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
    return middleware


@lru_cache(maxsize=32)
def build_agent(
    model_name: str,
    enabled_tools: tuple[str, ...] = (),
    include_thoughts: bool = False,
    kb_agents_instructions: str | None = None,
    include_local_tools: bool = True,
    disable_kb: bool = False,
    memory_config: MemoryConfig = MEMORY_PRESETS[DEFAULT_MEMORY_PRESET],
):
    # kb_agents_instructions is part of the cache key on purpose: an agent
    # built before data/kb/AGENTS.md existed must not pin its degraded
    # system prompt for the process lifetime. memory_config (frozen, hashable)
    # is too: each preset gets its own agent.
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
    if is_google_genai_model(model_name):
        if "web_search" in enabled:
            tools.append({"google_search": {}})
        if "url_context" in enabled:
            tools.append({"url_context": {}})
        if enabled & {"web_search", "url_context"}:
            middleware.append(GeminiServerSideToolsMiddleware())
    elif is_anthropic_model(model_name):
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
    if is_google_genai_model(model_name):
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
    preset = memory_preset or DEFAULT_MEMORY_PRESET
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
    # Raises on unknown preset names: a mislabeled experiment run must fail,
    # not silently fall back to prod. The API layer pre-validates to a 422.
    memory_config = resolve_memory_preset(request.memory_preset)
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
                    thought_text = "\n\n".join(extract_thought_summaries(token.content))
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
                    tool_call_id = str(
                        getattr(message, "tool_call_id", "") or uuid4().hex
                    )
                    tool_name = str(getattr(message, "name", "tool"))
                    call_matches: list[SourceMatch] = []
                    if tool_name == "retrieve_tutor_context":
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
                        call_matches = extract_shell_source_matches(command, payload)
                        _record_evidence(shell_evidence, call_matches)
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
                    for tool_call in message.tool_calls:
                        call_id = str(tool_call.get("id") or "")
                        if call_id:
                            tool_calls_by_id[call_id] = tool_call
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
            **totals,
            "est_cost_usd": estimate_cost_usd(usage_handler.usage_metadata),
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
            **pop_turn_signals(message_id),
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
