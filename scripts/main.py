from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from uuid import uuid4

import gradio as gr
import logfire
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from .chroma_rag import LocalChromaRetriever, format_tool_payload, parse_tool_payload
from .prompts import system_message_openai_agent
from .setup import (
    AVAILABLE_SOURCES_UI,
    DOCUMENT_DICT_PATH,
    SOURCE_UI_TO_KEY,
    VECTOR_COLLECTION_NAME,
    VECTOR_DB_DIR,
    ensure_local_vector_db,
)


@dataclass(frozen=True)
class AppContext:
    allowed_sources: tuple[str, ...]


@dataclass
class ActivityEvent:
    key: str
    kind: str
    body: str


SOURCES_HEADER = "📝 Here are the sources I used to answer your question:"
ACTIVITY_BLOCK_START = "<!-- MODEL_ACTIVITY_START -->"
ACTIVITY_BLOCK_END = "<!-- MODEL_ACTIVITY_END -->"
THOUGHTS_BLOCK_START = "<!-- GEMINI_THOUGHTS_START -->"
THOUGHTS_BLOCK_END = "<!-- GEMINI_THOUGHTS_END -->"
THOUGHTS_HEADER = "**Thinking**"
THOUGHTS_HINT = "_Reasoning summary from Gemini. This is not the final answer._"
TOOL_HEADER = "**Tool**"
TOOL_PENDING_HINT = "_Searching the selected sources..._"
ANSWER_HEADER = "**Answer**"
LEGACY_THOUGHTS_DETAILS_OPEN = "<details><summary>Gemini thoughts</summary>"
LEGACY_THOUGHTS_DETAILS_OPEN_EXPANDED = (
    "<details open><summary>Gemini thoughts</summary>"
)
CHECKPOINTER = InMemorySaver()
SOURCE_KEY_TO_LABEL = {value: key for key, value in SOURCE_UI_TO_KEY.items()}


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


@tool
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
    history: list[dict[str, Any]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in history:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message_content_to_text(message.get("content"))
        if role == "assistant":
            content = strip_display_blocks(content)
        normalized.append({"role": role, "content": content})
    return normalized


def checkpoint_messages_to_history(
    messages: list[BaseMessage],
) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages:
        message_type = getattr(message, "type", None)
        if message_type == "human":
            history.append(
                {"role": "user", "content": message_content_to_text(message.content)}
            )
            continue
        if message_type == "ai":
            history.append(
                {
                    "role": "assistant",
                    "content": strip_display_blocks(
                        message_content_to_text(message.content)
                    ),
                }
            )
    return history


def history_to_langgraph_messages(
    history: list[dict[str, str]],
) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for message in history:
        role = message["role"]
        content = message["content"]
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def new_thread_id() -> str:
    return uuid4().hex


def sync_thread_with_history(
    agent,
    thread_id: str,
    history: list[dict[str, str]],
) -> str:
    state = agent.get_state(thread_config(thread_id))
    checkpoint_history = checkpoint_messages_to_history(
        state.values.get("messages", [])
    )

    if not checkpoint_history:
        restored_messages = history_to_langgraph_messages(history)
        if restored_messages:
            agent.update_state(
                thread_config(thread_id), {"messages": restored_messages}
            )
        return thread_id

    if not history:
        return thread_id

    if checkpoint_history == history:
        return thread_id

    branched_thread_id = new_thread_id()
    restored_messages = history_to_langgraph_messages(history)
    if restored_messages:
        agent.update_state(
            thread_config(branched_thread_id),
            {"messages": restored_messages},
        )
    return branched_thread_id


def update_source_matches(
    matches_by_doc_id: dict[str, dict[str, Any]],
    payload: str,
) -> None:
    for match in parse_tool_payload(payload):
        existing = matches_by_doc_id.get(match.doc_id)
        if existing and existing["score"] >= match.score:
            continue
        matches_by_doc_id[match.doc_id] = {
            "title": match.title,
            "source": match.source,
            "url": match.url,
            "score": match.score,
        }


def format_sources(matches_by_doc_id: dict[str, dict[str, Any]]) -> str:
    if not matches_by_doc_id:
        return ""

    lines = [SOURCES_HEADER]
    sorted_matches = sorted(
        matches_by_doc_id.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    for match in sorted_matches:
        lines.append(
            f"- [🔗 {match['source']}: {match['title']}]({match['url']}), relevance: {match['score']:.2f}"
        )
    return "\n".join(lines)


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


def as_blockquote(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    return "\n".join("> " + line if line else ">" for line in lines)


def merge_stream_text(existing: str, incoming: str) -> str:
    current = existing.strip()
    update = incoming.strip()
    if not update:
        return current
    if not current:
        return update
    if update == current or update in current:
        return current
    if current in update:
        return update
    return f"{current}\n\n{update}"


def upsert_activity_event(
    events: list[ActivityEvent],
    *,
    key: str,
    kind: str,
    body: str,
    replace: bool = False,
) -> None:
    if not body.strip():
        return

    for index, event in enumerate(events):
        if event.key != key:
            continue
        events[index] = ActivityEvent(
            key=key,
            kind=kind,
            body=body.strip() if replace else merge_stream_text(event.body, body),
        )
        return

    events.append(ActivityEvent(key=key, kind=kind, body=body.strip()))


def format_tool_args(args: Any) -> str:
    if isinstance(args, dict):
        query = str(args.get("query", "")).strip()
        if query:
            trimmed_query = query[:157] + "..." if len(query) > 160 else query
            return f'Query: "{trimmed_query}"'
        if args:
            serialized = json.dumps(args, ensure_ascii=False, sort_keys=True)
            trimmed = serialized[:197] + "..." if len(serialized) > 200 else serialized
            return f"Args: `{trimmed}`"
        return ""
    if args is None:
        return ""
    serialized = str(args).strip()
    if not serialized:
        return ""
    trimmed = serialized[:197] + "..." if len(serialized) > 200 else serialized
    return f"Args: `{trimmed}`"


def summarize_activity_sources(sources: list[str], *, max_items: int = 3) -> str:
    if not sources:
        return ""
    if len(sources) <= max_items:
        return ", ".join(sources)
    remaining = len(sources) - max_items
    return f"{', '.join(sources[:max_items])}, and {remaining} more"


def summarize_tool_result(tool_name: str, payload: str) -> str:
    if tool_name != "retrieve_tutor_context":
        return "_Tool completed._"

    matches = parse_tool_payload(payload)
    if not matches:
        return "_No matching sources found in the selected sources._"

    ordered_sources: list[str] = []
    seen_sources: set[str] = set()
    for match in matches:
        source_label = SOURCE_KEY_TO_LABEL.get(match.source, match.source)
        if source_label in seen_sources:
            continue
        seen_sources.add(source_label)
        ordered_sources.append(source_label)

    source_summary = summarize_activity_sources(ordered_sources)
    match_count = len(matches)
    match_label = "match" if match_count == 1 else "matches"
    return f"_Found {match_count} {match_label} from {source_summary}._"


def format_tool_event(tool_name: str, args: Any, status_line: str) -> str:
    lines = [f"Using `{tool_name}`"]
    args_line = format_tool_args(args)
    if args_line:
        lines.append(args_line)
    lines.append(status_line)
    return "\n".join(lines)


def render_activity_block(events: list[ActivityEvent]) -> str:
    sections: list[str] = []
    for event in events:
        if event.kind == "thinking":
            sections.append(
                "\n".join(
                    [
                        THOUGHTS_HEADER,
                        THOUGHTS_HINT,
                        "",
                        as_blockquote(event.body),
                    ]
                ).strip()
            )
            continue
        if event.kind == "tool":
            sections.append(f"{TOOL_HEADER}\n{event.body}".strip())

    if not sections:
        return ""

    rendered_sections = "\n\n".join(section for section in sections if section)
    return f"{ACTIVITY_BLOCK_START}\n\n{rendered_sections}\n\n{ACTIVITY_BLOCK_END}"


def render_output(
    answer: str, activity_block: str = "", sources_block: str = ""
) -> str:
    visible_answer = answer.strip()
    visible_activity = activity_block.strip()
    visible_sources = sources_block.strip()

    output_parts = [visible_activity]
    if visible_answer:
        if visible_activity:
            output_parts.append(f"{ANSWER_HEADER}\n\n{visible_answer}")
        else:
            output_parts.append(visible_answer)
    output_parts.append(visible_sources)
    return "\n\n".join(part for part in output_parts if part)


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


@lru_cache(maxsize=8)
def build_agent(model_name: str, include_thoughts: bool = False):
    model = build_chat_model(model_name, include_thoughts=include_thoughts)
    return create_agent(
        model=model,
        tools=[retrieve_tutor_context],
        system_prompt=system_message_openai_agent,
        context_schema=AppContext,
        checkpointer=CHECKPOINTER,
    )


async def generate_completion(
    query: str,
    history,
    sources,
    model,
    show_gemini_thoughts,
    thread_id,
):
    source_keys = tuple(
        SOURCE_UI_TO_KEY[source] for source in sources if source in SOURCE_UI_TO_KEY
    )
    normalized_history = normalize_history(history)
    matches_by_doc_id: dict[str, dict[str, Any]] = {}
    activity_events: list[ActivityEvent] = []
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    answer_chunks: list[str] = []
    completed_answer = ""
    last_emitted = ""
    include_thoughts = bool(show_gemini_thoughts) and is_google_genai_model(model)
    show_activity = include_thoughts

    logfire.info("Running query", query=query)
    agent = build_agent(model, include_thoughts=include_thoughts)
    active_thread_id = sync_thread_with_history(
        agent,
        thread_id.strip() or new_thread_id(),
        normalized_history,
    )
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": query}]},
        thread_config(active_thread_id),
        context=AppContext(allowed_sources=source_keys),
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if not isinstance(token, AIMessageChunk):
                continue

            step = str(metadata.get("langgraph_step", ""))
            if show_activity:
                thought_text = "\n\n".join(extract_thought_summaries(token.content))
                if thought_text:
                    upsert_activity_event(
                        activity_events,
                        key=f"thinking:{step}",
                        kind="thinking",
                        body=thought_text,
                    )

                for tool_call in getattr(token, "tool_calls", []) or []:
                    tool_call_id = str(tool_call.get("id") or uuid4().hex)
                    tool_calls_by_id[tool_call_id] = tool_call
                    upsert_activity_event(
                        activity_events,
                        key=f"tool:{tool_call_id}",
                        kind="tool",
                        body=format_tool_event(
                            str(tool_call.get("name", "tool")),
                            tool_call.get("args"),
                            TOOL_PENDING_HINT,
                        ),
                        replace=True,
                    )

            text_delta = token.text or ""
            if not text_delta and token.content:
                text_delta = message_content_to_text(token.content)
            if text_delta:
                answer_chunks.append(text_delta)

            current_output = render_output(
                "".join(answer_chunks),
                render_activity_block(activity_events) if show_activity else "",
            )
            if current_output and current_output != last_emitted:
                last_emitted = current_output
                yield current_output, active_thread_id
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
                update_source_matches(
                    matches_by_doc_id,
                    payload,
                )
                if show_activity:
                    tool_call_id = str(
                        getattr(message, "tool_call_id", "") or uuid4().hex
                    )
                    tool_call = tool_calls_by_id.get(
                        tool_call_id,
                        {"name": getattr(message, "name", "tool"), "args": None},
                    )
                    upsert_activity_event(
                        activity_events,
                        key=f"tool:{tool_call_id}",
                        kind="tool",
                        body=format_tool_event(
                            str(
                                tool_call.get("name", getattr(message, "name", "tool"))
                            ),
                            tool_call.get("args"),
                            summarize_tool_result(
                                str(getattr(message, "name", "tool")),
                                payload,
                            ),
                        ),
                        replace=True,
                    )
                    current_output = render_output(
                        "".join(answer_chunks),
                        render_activity_block(activity_events),
                    )
                    if current_output and current_output != last_emitted:
                        last_emitted = current_output
                        yield current_output, active_thread_id
                continue

            if step != "model" or getattr(message, "type", None) != "ai":
                continue
            if getattr(message, "tool_calls", None):
                continue
            completed_answer = message_content_to_text(message.content)

    answer = "".join(answer_chunks).strip() or completed_answer.strip()
    activity_block = render_activity_block(activity_events) if show_activity else ""
    sources_block = format_sources(matches_by_doc_id)
    final_output = render_output(answer, activity_block, sources_block)
    if final_output != last_emitted:
        yield final_output, active_thread_id


accordion = gr.Accordion(label="Customize Sources (Click to expand)", open=False)
sources = gr.CheckboxGroup(
    AVAILABLE_SOURCES_UI,
    label="Sources",
    value=[
        "Agentic AI Engineering",
        "Master AI For Work",
        "Advanced LLM Developer",
        "8 Hour Primer",
        "Python Primer",
        "Towards AI Blog",
        "Transformers Docs",
        "PEFT Docs",
        "TRL Docs",
        "LlamaIndex Docs",
        "LangChain Docs",
        "OpenAI Cookbooks",
    ],
    interactive=True,
)
model = gr.Textbox(
    label="Model (provider:model)",
    value="google-genai:gemini-flash-latest",
    interactive=False,
    placeholder="openai:gpt-5.4-mini | anthropic:claude-opus-4-6 | google-genai:gemini-3.1-pro-preview",
)
show_gemini_thoughts = gr.Checkbox(
    label="Show Gemini thinking and tool timeline",
    value=True,
    info="Uses Gemini include_thoughts and shows tool activity when supported.",
)
thread_id = gr.Textbox(
    label="Thread ID",
    value="",
    visible=False,
    container=False,
)

with gr.Blocks(
    title="Towards AI 🤖",
    analytics_enabled=True,
    fill_height=True,
) as demo:

    def reset_thread_id():
        return ""

    chatbot = gr.Chatbot(
        scale=20,
        placeholder="<strong>Towards AI 🤖: A Question-Answering Bot for anything AI-related</strong><br>",
        show_label=False,
        buttons=["copy"],
    )
    chatbot.clear(
        reset_thread_id,
        None,
        [thread_id],
        api_visibility="undocumented",
        queue=False,
    )
    chatbot.undo(
        reset_thread_id,
        None,
        [thread_id],
        api_visibility="undocumented",
        queue=False,
    )
    chatbot.retry(
        reset_thread_id,
        None,
        [thread_id],
        api_visibility="undocumented",
        queue=False,
    )
    chatbot.edit(
        reset_thread_id,
        None,
        [thread_id],
        api_visibility="undocumented",
        queue=False,
    )
    gr.ChatInterface(
        fn=generate_completion,
        chatbot=chatbot,
        additional_inputs=[sources, model, show_gemini_thoughts, thread_id],
        additional_outputs=[thread_id],
        additional_inputs_accordion=accordion,
        api_name="chat",
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=False, share=False)
