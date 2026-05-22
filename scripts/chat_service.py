from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, AsyncIterator
from uuid import uuid4

import logfire
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
    extract_raw_paths,
    parse_markdown_citations,
    resolve_manifest_reference,
    source_match_key,
    source_match_payload,
)
from .prompts import build_system_prompt
from .setup import (
    BM25_INDEX_PATH,
    COURSE_SOURCE_KEYS,
    DEFAULT_SELECTED_SOURCE_KEYS,
    DOCUMENT_DICT_PATH,
    SOURCE_KEY_TO_LABEL,
    VECTOR_COLLECTION_NAME,
    VECTOR_DB_DIR,
    ensure_local_vector_db,
)

SOURCES_HEADER = "📝 Here are the sources I used to answer your question:"
ACTIVITY_BLOCK_START = "<!-- MODEL_ACTIVITY_START -->"
ACTIVITY_BLOCK_END = "<!-- MODEL_ACTIVITY_END -->"
THOUGHTS_BLOCK_START = "<!-- GEMINI_THOUGHTS_START -->"
THOUGHTS_BLOCK_END = "<!-- GEMINI_THOUGHTS_END -->"
ANSWER_HEADER = "**Answer**"
LEGACY_THOUGHTS_DETAILS_OPEN = "<details><summary>Gemini thoughts</summary>"
LEGACY_THOUGHTS_DETAILS_OPEN_EXPANDED = "<details open><summary>Gemini thoughts</summary>"
CHECKPOINTER = InMemorySaver()
_RETRIEVER_INIT_LOCK = Lock()
KB_TOOL_NAMES = ("run_kb_command",)
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
        logfire.warn(
            "Retriever warm-up failed; first retrieval call may retry.",
            error=str(exc),
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
    results = get_retriever().search(
        query=query,
        allowed_sources=list(runtime.context.allowed_sources),
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
        logfire.warn("KB artifact download/check failed.", error=str(exc))
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


def strip_sources_block(text: str) -> str:
    separator = f"\n\n{SOURCES_HEADER}"
    body, marker, _ = text.partition(separator)
    if marker:
        return body.strip()
    if text.startswith(SOURCES_HEADER):
        return ""
    return text.strip()


def strip_hidden_block(text: str, start_marker: str, end_marker: str) -> str:
    stripped = text
    while True:
        start = stripped.find(start_marker)
        if start == -1:
            return stripped
        end = stripped.find(end_marker, start)
        if end == -1:
            return stripped[:start]
        stripped = stripped[:start] + stripped[end + len(end_marker) :]


def strip_activity_block(text: str) -> str:
    stripped = strip_hidden_block(text, ACTIVITY_BLOCK_START, ACTIVITY_BLOCK_END)
    stripped = strip_hidden_block(stripped, THOUGHTS_BLOCK_START, THOUGHTS_BLOCK_END)

    for marker in (
        LEGACY_THOUGHTS_DETAILS_OPEN_EXPANDED,
        LEGACY_THOUGHTS_DETAILS_OPEN,
    ):
        separator = f"\n\n{marker}"
        body, found, _ = stripped.partition(separator)
        if found:
            return body.strip()
        if stripped.startswith(marker):
            return ""
    return stripped.strip()


def strip_answer_header(text: str) -> str:
    stripped = text.strip()
    answer_prefix = f"{ANSWER_HEADER}\n\n"
    if stripped.startswith(answer_prefix):
        return stripped[len(answer_prefix) :].strip()
    if stripped == ANSWER_HEADER:
        return ""
    return stripped


def strip_display_blocks(text: str) -> str:
    return strip_answer_header(strip_activity_block(strip_sources_block(text)))


def normalize_history(
    history: list[dict[str, Any]] | tuple[ChatTurn, ...] | list[ChatTurn],
) -> tuple[ChatTurn, ...]:
    if not history:
        return ()

    normalized: list[ChatTurn] = []
    for message in history:
        if isinstance(message, ChatTurn):
            role = message.role
            content = message.content
        else:
            role = str(message.get("role", ""))
            content = message_content_to_text(message.get("content"))

        if role not in {"user", "assistant"}:
            continue
        if role == "assistant":
            content = strip_display_blocks(content)
        normalized.append(ChatTurn(role=role, content=content))
    return tuple(normalized)


def checkpoint_messages_to_history(messages: list[BaseMessage]) -> tuple[ChatTurn, ...]:
    history: list[ChatTurn] = []
    for message in messages:
        message_type = getattr(message, "type", None)
        if message_type == "human":
            history.append(ChatTurn("user", message_content_to_text(message.content)))
            continue
        if message_type == "ai":
            history.append(
                ChatTurn(
                    "assistant",
                    strip_display_blocks(message_content_to_text(message.content)),
                )
            )
    return tuple(history)


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
) -> str:
    state = agent.get_state(thread_config(thread_id))
    checkpoint_history = checkpoint_messages_to_history(
        state.values.get("messages", [])
    )

    if checkpoint_history == history:
        return thread_id

    if not checkpoint_history:
        restored_messages = history_to_langgraph_messages(history)
        if restored_messages:
            agent.update_state(
                thread_config(thread_id), {"messages": restored_messages}
            )
        return thread_id

    branched_thread_id = new_thread_id()
    restored_messages = history_to_langgraph_messages(history)
    if restored_messages:
        agent.update_state(
            thread_config(branched_thread_id),
            {"messages": restored_messages},
        )
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


def _record_evidence(target: dict[str, SourceMatch], matches: list[SourceMatch]) -> None:
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


def resolve_answer_citations(
    answer: str,
    *,
    retrieval_evidence: dict[str, SourceMatch],
    shell_evidence: dict[str, SourceMatch],
    web_evidence: dict[str, SourceMatch],
) -> list[SourceMatch]:
    evidence_indexes = [
        _index_evidence(retrieval_evidence),
        _index_evidence(shell_evidence),
        _index_evidence(web_evidence),
    ]
    resolved: list[SourceMatch] = []
    seen: set[str] = set()
    for label, reference in parse_markdown_citations(answer):
        candidates = [reference, label.strip().lower()]
        match: SourceMatch | None = None
        for index in evidence_indexes:
            for candidate in candidates:
                if candidate and candidate in index:
                    match = index[candidate]
                    break
            if match:
                break
        if not match:
            manifest_match = resolve_manifest_reference(reference, label=label)
            if manifest_match:
                shell_index = _index_evidence(shell_evidence)
                if (
                    manifest_match.doc_id in shell_index
                    or manifest_match.url in shell_index
                    or manifest_match.title.strip().lower() in shell_index
                ):
                    match = shell_index.get(manifest_match.doc_id) or shell_index.get(manifest_match.url) or shell_index.get(manifest_match.title.strip().lower())
        if not match:
            continue
        key = source_match_key(match)
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


def extract_thought_summaries(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []

    thoughts: list[str] = []
    for item in content:
        if not hasattr(item, "get"):
            continue

        item_type = item.get("type")
        if item_type == "thinking":
            thought = str(item.get("thinking", "")).strip()
        elif item_type == "reasoning":
            thought = str(item.get("reasoning", "")).strip()
        else:
            continue

        if thought:
            thoughts.append(thought)
    return thoughts


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
            lines.append(
                f"- {label}: `raw/{group}/{key}/`, `wiki/{wiki_dir}/{key}.md`"
            )
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
):
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
                    keep=3,
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
        system_prompt=build_system_prompt(model_name, enabled_tools),
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
    names = ["retrieve_tutor_context", *KB_TOOL_NAMES]
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


def extract_web_search_queries(response_metadata: Any) -> list[str]:
    """Pull the queries Gemini ran against google_search from grounding metadata."""
    if not isinstance(response_metadata, dict):
        return []
    grounding = response_metadata.get("grounding_metadata") or {}
    queries = grounding.get("web_search_queries") or []
    return [str(q).strip() for q in queries if isinstance(q, str) and str(q).strip()]


def extract_grounding_source_matches(
    response_metadata: Any,
    matches_by_doc_id: dict[str, SourceMatch],
) -> list[SourceMatch]:
    """Turn Gemini grounding metadata into source matches (deduped by URI)."""
    if not isinstance(response_metadata, dict):
        return []
    grounding = response_metadata.get("grounding_metadata") or {}
    chunks = grounding.get("grounding_chunks") or []
    if not chunks:
        return []

    confidence_by_index: dict[int, float] = {}
    for support in grounding.get("grounding_supports") or []:
        indices = support.get("grounding_chunk_indices") or []
        scores = support.get("confidence_scores") or []
        for idx, score in zip(indices, scores):
            if not isinstance(idx, int):
                continue
            numeric = float(score) if isinstance(score, (int, float)) else 0.0
            if numeric > confidence_by_index.get(idx, 0.0):
                confidence_by_index[idx] = numeric

    updated: list[SourceMatch] = []
    for idx, chunk in enumerate(chunks):
        web = (chunk or {}).get("web") or {}
        uri = str(web.get("uri") or "").strip()
        if not uri:
            continue
        title = str(web.get("title") or uri).strip()
        doc_id = f"google_search::{uri}"
        if doc_id in matches_by_doc_id:
            continue
        score = confidence_by_index.get(idx, 1.0)
        source_match = SourceMatch(
            doc_id=doc_id,
            title=title,
            url=uri,
            source_key="google_search",
            source_label="Web",
            score=score,
            group="web",
        )
        matches_by_doc_id[doc_id] = source_match
        updated.append(source_match)
    return updated


GOOGLE_SEARCH_TOOL_NAME = "google_search"
ANTHROPIC_SERVER_TOOL_NAMES = frozenset({"web_search", "web_fetch"})
ANTHROPIC_RESULT_BLOCK_TYPES = {
    "web_search_tool_result": ("web_search", "Web"),
    "web_fetch_tool_result": ("web_fetch", "Web page"),
}


def extract_anthropic_source_matches(
    content: Any,
    matches_by_doc_id: dict[str, SourceMatch],
) -> tuple[dict[str, list[SourceMatch]], dict[str, dict[str, Any]]]:
    """Parse Claude's server-side web tool invocations and their results.

    Scans ``message.content`` for three kinds of blocks emitted when Claude
    runs the built-in ``web_search`` / ``web_fetch`` tools:

    * ``tool_use`` — the model's call (id, name, input args)
    * ``web_search_tool_result`` / ``web_fetch_tool_result`` — the server's
      response, keyed by ``tool_use_id``
    * ``text`` blocks with ``citations`` — fallback for citations without a
      matching result block

    Returns ``(matches_by_tool_use_id, tool_use_index)`` where
    ``tool_use_index`` maps tool_use id → ``{"name", "args"}`` so the caller
    can emit ``tool_call_started`` events with the right metadata.
    ``langchain-anthropic`` does not always surface server-side tool_use in
    ``AIMessage.tool_calls``, so we read them off the content blocks directly.
    """
    if not isinstance(content, list):
        return {}, {}

    updates: dict[str, list[SourceMatch]] = {}
    tool_use_index: dict[str, dict[str, Any]] = {}

    for block in content:
        if not hasattr(block, "get"):
            continue

        block_type = block.get("type")

        if block_type in ("server_tool_use", "tool_use"):
            tool_use_id = str(block.get("id") or "")
            tool_name = str(block.get("name") or "")
            if tool_use_id and tool_name in ANTHROPIC_SERVER_TOOL_NAMES:
                args = block.get("input") or {}
                if not args:
                    partial = block.get("partial_json")
                    if isinstance(partial, str) and partial.strip():
                        try:
                            parsed = json.loads(partial)
                        except json.JSONDecodeError:
                            parsed = None
                        if isinstance(parsed, dict):
                            args = parsed
                tool_use_index[tool_use_id] = {
                    "id": tool_use_id,
                    "name": tool_name,
                    "args": args,
                }
            continue

        mapping = ANTHROPIC_RESULT_BLOCK_TYPES.get(block_type)
        if mapping:
            source_key, source_label = mapping
            tool_use_id = str(block.get("tool_use_id") or "")
            results = block.get("content") or []
            if not isinstance(results, list):
                continue
            for result in results:
                if not hasattr(result, "get"):
                    continue
                url = str(result.get("url") or "").strip()
                if not url:
                    continue
                title = str(result.get("title") or url).strip()
                doc_id = f"{source_key}::{url}"
                if doc_id in matches_by_doc_id:
                    continue
                source_match = SourceMatch(
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    source_key=source_key,
                    source_label=source_label,
                    score=1.0,
                    group="web",
                )
                matches_by_doc_id[doc_id] = source_match
                updates.setdefault(tool_use_id, []).append(source_match)
            continue

        if block_type == "text":
            for citation in block.get("citations") or []:
                if not hasattr(citation, "get"):
                    continue
                url = str(citation.get("url") or "").strip()
                if not url:
                    continue
                title = str(citation.get("title") or url).strip()
                doc_id = f"web_search::{url}"
                if doc_id in matches_by_doc_id:
                    continue
                source_match = SourceMatch(
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    source_key="web_search",
                    source_label="Web",
                    score=1.0,
                    group="web",
                )
                matches_by_doc_id[doc_id] = source_match
                updates.setdefault("", []).append(source_match)

    return updates, tool_use_index


async def stream_chat(request: ChatRequest) -> AsyncIterator[ChatEvent]:
    normalized_history = normalize_history(request.history)
    retrieval_evidence: dict[str, SourceMatch] = {}
    shell_evidence: dict[str, SourceMatch] = {}
    web_evidence: dict[str, SourceMatch] = {}
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    answer_chunks: list[str] = []
    completed_answer = ""
    message_id = uuid4().hex
    include_reasoning = bool(request.include_reasoning) and is_google_genai_model(
        request.model_name
    )
    google_search_call_id = ""
    google_search_queries: list[str] = []
    google_search_match_count = 0

    logfire.info("Running query", query=request.query)
    agent = build_agent(
        request.model_name,
        enabled_tools=tuple(request.enabled_tools),
        include_thoughts=include_reasoning,
    )
    active_thread_id = sync_thread_with_history(
        agent,
        request.thread_id.strip() or new_thread_id(),
        normalized_history,
    )

    yield ChatEvent("thread_started", {"thread_id": active_thread_id})
    yield ChatEvent("message_started", {"message_id": message_id})

    try:
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": request.query}]},
            agent_run_config(request, active_thread_id, message_id),
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
                    thought_text = "\n\n".join(
                        extract_thought_summaries(token.content)
                    )
                    if thought_text:
                        yield ChatEvent(
                            "reasoning_delta",
                            {
                                "message_id": message_id,
                                "step": step,
                                "text": thought_text,
                            },
                        )

                for tool_call in getattr(token, "tool_calls", []) or []:
                    tool_call_id = str(tool_call.get("id") or uuid4().hex)
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

                token_metadata = getattr(token, "response_metadata", None)
                new_queries = [
                    q
                    for q in extract_web_search_queries(token_metadata)
                    if q not in google_search_queries
                ]
                new_grounding = extract_grounding_source_matches(
                    token_metadata,
                    web_evidence,
                )
                if (new_queries or new_grounding) and not google_search_call_id:
                    google_search_call_id = uuid4().hex
                    yield ChatEvent(
                        "tool_call_started",
                        {
                            "message_id": message_id,
                            "call_id": google_search_call_id,
                            "tool_name": GOOGLE_SEARCH_TOOL_NAME,
                            "args": {
                                "query": "; ".join(new_queries)
                                if new_queries
                                else ""
                            },
                            "args_text": "; ".join(new_queries),
                        },
                    )
                if new_queries:
                    google_search_queries.extend(new_queries)
                google_search_match_count += len(new_grounding)
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

                message_metadata = getattr(message, "response_metadata", None)
                new_queries = [
                    q
                    for q in extract_web_search_queries(message_metadata)
                    if q not in google_search_queries
                ]
                new_grounding = extract_grounding_source_matches(
                    message_metadata,
                    web_evidence,
                )
                if (new_queries or new_grounding) and not google_search_call_id:
                    google_search_call_id = uuid4().hex
                    yield ChatEvent(
                        "tool_call_started",
                        {
                            "message_id": message_id,
                            "call_id": google_search_call_id,
                            "tool_name": GOOGLE_SEARCH_TOOL_NAME,
                            "args": {
                                "query": "; ".join(new_queries)
                                if new_queries
                                else ""
                            },
                            "args_text": "; ".join(new_queries),
                        },
                    )
                if new_queries:
                    google_search_queries.extend(new_queries)
                google_search_match_count += len(new_grounding)

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
                    continue
                completed_answer = message_content_to_text(message.content)
    finally:
        _clear_kb_command_budget(message_id)

    if google_search_call_id:
        joined_queries = "; ".join(google_search_queries)
        if google_search_match_count == 0:
            output_text = "Google search ran but returned no grounding results."
        else:
            plural = "" if google_search_match_count == 1 else "s"
            output_text = (
                f"Google search returned {google_search_match_count} web result{plural}."
            )
        yield ChatEvent(
            "tool_call_completed",
            {
                "message_id": message_id,
                "call_id": google_search_call_id,
                "tool_name": GOOGLE_SEARCH_TOOL_NAME,
                "args": {"query": joined_queries},
                "args_text": joined_queries,
                "output_text": output_text,
            },
        )

    answer = "".join(answer_chunks).strip() or completed_answer.strip()
    matched_sources = list(
        resolve_answer_citations(
            answer,
            retrieval_evidence=retrieval_evidence,
            shell_evidence=shell_evidence,
            web_evidence=web_evidence,
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
