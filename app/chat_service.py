from __future__ import annotations

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

from .chat_types import ChatEvent, ChatRequest, ChatTurn, SourceMatch
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
        results = get_retriever().search(
            query=query,
            allowed_sources=list(runtime.context.allowed_sources),
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
            return ChatAnthropic(
                model=actual_model,
                temperature=1,
                max_tokens=8192,
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
        "Unsupported model provider. Use openai, anthropic, or google-genai."
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


@lru_cache(maxsize=32)
def build_agent(
    model_name: str,
    enabled_tools: tuple[str, ...] = (),
    include_thoughts: bool = False,
    kb_agents_instructions: str | None = None,
):
    # kb_agents_instructions is part of the cache key on purpose: an agent
    # built before data/kb/AGENTS.md existed must not pin its degraded
    # system prompt for the process lifetime.
    model = build_chat_model(model_name, include_thoughts=include_thoughts)
    tools: list[Any] = [
        retrieve_tutor_context,
        run_kb_command,
    ]
    middleware: list[AgentMiddleware] = [
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=5_000,
                    keep=5,
                    # Retrieval results stay; only shell outputs get cleared.
                    exclude_tools=("retrieve_tutor_context",),
                    placeholder="[tool output cleared to save context]",
                )
            ],
            token_count_method="approximate",
        ),
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 30_000),
            keep=("messages", 20),
        ),
        SourcePreferenceMiddleware(),
    ]
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
            model_name, enabled_tools, kb_agents_instructions
        ),
        context_schema=AppContext,
        checkpointer=CHECKPOINTER,
        middleware=middleware,
    )


def model_provider_and_name(model_name: str) -> tuple[str, str]:
    provider_model = normalize_model_name(model_name)
    provider, _, actual_model = provider_model.partition(":")
    return provider or "unknown", actual_model or provider_model


def effective_tool_names(
    model_name: str,
    enabled_tools: tuple[str, ...],
) -> tuple[str, ...]:
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
) -> dict[str, Any]:
    provider, actual_model = model_provider_and_name(request.model_name)
    tools = effective_tool_names(request.model_name, request.enabled_tools)
    source_labels = [
        SOURCE_KEY_TO_LABEL.get(source_key, source_key)
        for source_key in request.source_keys
    ]
    config = thread_config(active_thread_id)
    config.update(
        {
            "run_name": "ai-tutor-agent-turn",
            "tags": [
                "ai-tutor-app",
                "knowledge-base-chatbot",
                f"provider:{provider}",
                f"model:{actual_model}",
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
            },
        }
    )
    return config


async def stream_chat(request: ChatRequest) -> AsyncIterator[ChatEvent]:
    normalized_history = normalize_history(request.history)
    retrieval_evidence: dict[str, SourceMatch] = {}
    shell_evidence: dict[str, SourceMatch] = {}
    web_evidence: dict[str, SourceMatch] = {}
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    answer_chunks: list[str] = []
    completed_answer = ""
    message_id = uuid4().hex
    include_reasoning = bool(request.include_reasoning) and (
        is_google_genai_model(request.model_name)
        or is_anthropic_model(request.model_name)
    )
    # Gemini streams each thought summary as one complete block; Anthropic
    # streams partial fragments of a single thought. The encoder uses this to
    # decide whether consecutive deltas need a paragraph break between them.
    reasoning_deltas_are_blocks = is_google_genai_model(request.model_name)
    google_search = GoogleSearchActivity(message_id, web_evidence)
    effective_tools = effective_tool_names(request.model_name, request.enabled_tools)
    if "url_context" in effective_tools:
        _record_evidence(web_evidence, url_context_evidence(request.query))

    logger.info("Running query: %s", request.query)
    agent = build_agent(
        request.model_name,
        enabled_tools=tuple(request.enabled_tools),
        include_thoughts=include_reasoning,
        kb_agents_instructions=ensure_kb_agents_instructions(),
    )
    _evict_idle_threads()
    active_thread_id, fork_checkpoint_id = sync_thread_with_history(
        agent,
        request.thread_id.strip() or new_thread_id(),
        normalized_history,
    )
    _touch_thread(active_thread_id)
    run_config = agent_run_config(request, active_thread_id, message_id)
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

                step = str(metadata.get("langgraph_step", ""))
                if include_reasoning:
                    thought_text = "\n\n".join(extract_thought_summaries(token.content))
                    if thought_text:
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
                if getattr(message, "type", None) == "tool":
                    payload = message_content_to_text(message.content)
                    tool_call_id = str(
                        getattr(message, "tool_call_id", "") or uuid4().hex
                    )
                    tool_name = str(getattr(message, "name", "tool"))
                    if tool_name == "retrieve_tutor_context":
                        _record_evidence(
                            retrieval_evidence,
                            collect_retrieval_source_matches(payload),
                        )

                    tool_call = tool_calls_by_id.get(
                        tool_call_id,
                        {"name": getattr(message, "name", "tool"), "args": None},
                    )
                    if tool_name == "run_kb_command":
                        args = tool_call.get("args")
                        command = str(
                            args.get("command") if isinstance(args, dict) else ""
                        )
                        _record_evidence(
                            shell_evidence,
                            extract_shell_source_matches(command, payload),
                        )
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

    matched_sources = list(
        resolve_answer_citations(
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
    yield ChatEvent(
        "message_completed",
        {
            "message_id": message_id,
            "thread_id": active_thread_id,
            "answer": answer,
        },
    )
