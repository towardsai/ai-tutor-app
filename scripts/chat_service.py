from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, AsyncIterator
from uuid import uuid4

import logfire
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from .chat_types import ChatEvent, ChatRequest, ChatTurn, SourceMatch
from .chroma_rag import LocalChromaRetriever, format_tool_payload, parse_tool_payload
from .prompts import system_message_openai_agent
from .setup import (
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


@dataclass(frozen=True)
class AppContext:
    allowed_sources: tuple[str, ...]


@lru_cache(maxsize=1)
def get_retriever() -> LocalChromaRetriever:
    ensure_local_vector_db()
    cohere_api_key = os.environ["COHERE_API_KEY"]
    return LocalChromaRetriever(
        db_path=VECTOR_DB_DIR,
        collection_name=VECTOR_COLLECTION_NAME,
        document_dict_path=DOCUMENT_DICT_PATH,
        cohere_api_key=cohere_api_key,
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


def collect_updated_source_matches(
    matches_by_doc_id: dict[str, SourceMatch],
    payload: str,
) -> list[SourceMatch]:
    updated: list[SourceMatch] = []
    for match in parse_tool_payload(payload):
        source_match = SourceMatch(
            doc_id=match.doc_id,
            title=match.title,
            url=match.url,
            source_key=match.source,
            source_label=SOURCE_KEY_TO_LABEL.get(match.source, match.source),
            score=match.score,
        )
        existing = matches_by_doc_id.get(match.doc_id)
        if existing and existing.score >= source_match.score:
            continue
        matches_by_doc_id[match.doc_id] = source_match
        updated.append(source_match)
    return updated


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


@lru_cache(maxsize=8)
def build_agent(model_name: str, include_thoughts: bool = False):
    model = build_chat_model(model_name, include_thoughts=include_thoughts)
    tools: list[Any] = [retrieve_tutor_context]
    middleware: list[AgentMiddleware] = []
    if is_google_genai_model(model_name):
        tools.append({"google_search": {}})
        tools.append({"url_context": {}})
        middleware.append(GeminiServerSideToolsMiddleware())
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_message_openai_agent,
        context_schema=AppContext,
        checkpointer=CHECKPOINTER,
        middleware=middleware,
    )


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
        )
        matches_by_doc_id[doc_id] = source_match
        updated.append(source_match)
    return updated


GOOGLE_SEARCH_TOOL_NAME = "google_search"


async def stream_chat(request: ChatRequest) -> AsyncIterator[ChatEvent]:
    normalized_history = normalize_history(request.history)
    matches_by_doc_id: dict[str, SourceMatch] = {}
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
    agent = build_agent(request.model_name, include_thoughts=include_reasoning)
    active_thread_id = sync_thread_with_history(
        agent,
        request.thread_id.strip() or new_thread_id(),
        normalized_history,
    )

    yield ChatEvent(
        "thread_started",
        {"thread_id": active_thread_id},
    )
    yield ChatEvent(
        "message_started",
        {"message_id": message_id},
    )

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": request.query}]},
        thread_config(active_thread_id),
        context=AppContext(allowed_sources=request.source_keys),
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if not isinstance(token, AIMessageChunk):
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
                matches_by_doc_id,
            )
            if (new_queries or new_grounding) and not google_search_call_id:
                google_search_call_id = uuid4().hex
                yield ChatEvent(
                    "tool_call_started",
                    {
                        "message_id": message_id,
                        "call_id": google_search_call_id,
                        "tool_name": GOOGLE_SEARCH_TOOL_NAME,
                        "args": {"query": "; ".join(new_queries) if new_queries else ""},
                        "args_text": "; ".join(new_queries),
                    },
                )
            if new_queries:
                google_search_queries.extend(new_queries)
            for source_match in new_grounding:
                google_search_match_count += 1
                yield ChatEvent(
                    "source_match",
                    {
                        "message_id": message_id,
                        "doc_id": source_match.doc_id,
                        "title": source_match.title,
                        "url": source_match.url,
                        "source_key": source_match.source_key,
                        "source_label": source_match.source_label,
                        "score": source_match.score,
                        "call_id": google_search_call_id,
                    },
                )
            continue

        if chunk["type"] != "updates":
            continue

        for step, update in chunk["data"].items():
            message = update["messages"][-1]
            if (
                getattr(message, "type", None) == "tool"
                and getattr(message, "name", "") == "retrieve_tutor_context"
            ):
                payload = message_content_to_text(message.content)
                tool_call_id = str(getattr(message, "tool_call_id", "") or uuid4().hex)
                for source_match in collect_updated_source_matches(
                    matches_by_doc_id,
                    payload,
                ):
                    yield ChatEvent(
                        "source_match",
                        {
                            "message_id": message_id,
                            "doc_id": source_match.doc_id,
                            "title": source_match.title,
                            "url": source_match.url,
                            "source_key": source_match.source_key,
                            "source_label": source_match.source_label,
                            "score": source_match.score,
                            "call_id": tool_call_id,
                        },
                    )

                tool_call = tool_calls_by_id.get(
                    tool_call_id,
                    {"name": getattr(message, "name", "tool"), "args": None},
                )
                yield ChatEvent(
                    "tool_call_completed",
                    {
                        "message_id": message_id,
                        "step": step,
                        "call_id": tool_call_id,
                        "tool_name": str(tool_call.get("name", getattr(message, "name", "tool"))),
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
                matches_by_doc_id,
            )
            if (new_queries or new_grounding) and not google_search_call_id:
                google_search_call_id = uuid4().hex
                yield ChatEvent(
                    "tool_call_started",
                    {
                        "message_id": message_id,
                        "call_id": google_search_call_id,
                        "tool_name": GOOGLE_SEARCH_TOOL_NAME,
                        "args": {"query": "; ".join(new_queries) if new_queries else ""},
                        "args_text": "; ".join(new_queries),
                    },
                )
            if new_queries:
                google_search_queries.extend(new_queries)
            for source_match in new_grounding:
                google_search_match_count += 1
                yield ChatEvent(
                    "source_match",
                    {
                        "message_id": message_id,
                        "doc_id": source_match.doc_id,
                        "title": source_match.title,
                        "url": source_match.url,
                        "source_key": source_match.source_key,
                        "source_label": source_match.source_label,
                        "score": source_match.score,
                        "call_id": google_search_call_id,
                    },
                )

            if getattr(message, "tool_calls", None):
                continue
            completed_answer = message_content_to_text(message.content)

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
    yield ChatEvent(
        "message_completed",
        {
            "message_id": message_id,
            "thread_id": active_thread_id,
            "answer": answer,
        },
    )
