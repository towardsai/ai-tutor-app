from __future__ import annotations

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


SOURCES_HEADER = "📝 Here are the sources I used to answer your question:"
THOUGHTS_BLOCK_START = "<!-- GEMINI_THOUGHTS_START -->"
THOUGHTS_BLOCK_END = "<!-- GEMINI_THOUGHTS_END -->"
THOUGHTS_HEADER = "**Thinking**"
THOUGHTS_HINT = "_Reasoning summary from Gemini. This is not the final answer._"
ANSWER_HEADER = "**Answer**"
LEGACY_THOUGHTS_DETAILS_OPEN = "<details><summary>Gemini thoughts</summary>"
LEGACY_THOUGHTS_DETAILS_OPEN_EXPANDED = "<details open><summary>Gemini thoughts</summary>"
CHECKPOINTER = InMemorySaver()


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


def strip_thoughts_block(text: str) -> str:
    stripped = text

    while True:
        start = stripped.find(THOUGHTS_BLOCK_START)
        if start == -1:
            break
        end = stripped.find(THOUGHTS_BLOCK_END, start)
        if end == -1:
            stripped = stripped[:start]
            break
        stripped = (
            stripped[:start] + stripped[end + len(THOUGHTS_BLOCK_END) :]
        )

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


def strip_display_blocks(text: str) -> str:
    return strip_thoughts_block(strip_sources_block(text))


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


def format_thoughts(content: Any) -> str:
    thoughts = extract_thought_summaries(content)
    if not thoughts:
        return ""
    quoted_thoughts = "\n\n".join(as_blockquote(thought) for thought in thoughts)
    return (
        f"{THOUGHTS_BLOCK_START}\n\n"
        f"{THOUGHTS_HEADER}\n"
        f"{THOUGHTS_HINT}\n\n"
        f"{quoted_thoughts}\n\n"
        f"{THOUGHTS_BLOCK_END}"
    )


def render_output(
    answer: str, thoughts_block: str = "", sources_block: str = ""
) -> str:
    visible_answer = answer.strip()
    visible_thoughts = thoughts_block.strip()
    visible_sources = sources_block.strip()

    output_parts = [visible_thoughts]
    if visible_answer:
        if visible_thoughts:
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
    streamed_message: AIMessageChunk | None = None
    completed_answer = ""
    completed_thoughts = ""
    last_emitted = ""
    include_thoughts = bool(show_gemini_thoughts) and is_google_genai_model(model)

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
            token, _metadata = chunk["data"]
            if not isinstance(token, AIMessageChunk):
                continue
            if not token.content and not token.text:
                continue

            streamed_message = (
                token if streamed_message is None else streamed_message + token
            )
            current_answer = message_content_to_text(streamed_message.content)
            current_thoughts = format_thoughts(streamed_message.content)
            current_output = render_output(current_answer, current_thoughts)
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
                update_source_matches(
                    matches_by_doc_id,
                    message_content_to_text(message.content),
                )
                continue

            if step != "model" or getattr(message, "type", None) != "ai":
                continue
            if getattr(message, "tool_calls", None):
                continue
            completed_answer = message_content_to_text(message.content)
            completed_thoughts = format_thoughts(message.content)

    answer = (
        message_content_to_text(streamed_message.content).strip()
        if streamed_message is not None
        else ""
    ) or completed_answer.strip()
    thoughts_block = (
        format_thoughts(streamed_message.content)
        if streamed_message is not None
        else ""
    ) or completed_thoughts.strip()
    sources_block = format_sources(matches_by_doc_id)
    final_output = render_output(answer, thoughts_block, sources_block)
    if final_output != last_emitted:
        yield final_output, active_thread_id


accordion = gr.Accordion(label="Customize Sources (Click to expand)", open=False)
sources = gr.CheckboxGroup(
    AVAILABLE_SOURCES_UI,
    label="Sources",
    value=[
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
    label="Show Gemini thought summaries",
    value=True,
    info="Uses Gemini include_thoughts when the selected model supports it.",
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
