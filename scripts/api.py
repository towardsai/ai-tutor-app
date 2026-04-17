from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

import logfire
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chat_service import (
    get_retriever,
    is_anthropic_model,
    is_google_genai_model,
    message_content_to_text,
    stream_chat,
)
from .chat_types import ChatEvent, ChatRequest, ChatTurn
from .setup import (
    AVAILABLE_MODELS,
    AVAILABLE_SOURCES,
    AVAILABLE_SOURCES_UI,
    COURSE_SOURCE_KEYS,
    DEFAULT_MODEL_NAME,
    DEFAULT_SELECTED_SOURCE_KEYS,
    DEFAULT_SELECTED_SOURCES_UI,
    SOURCE_UI_TO_KEY,
)


class ApiChatTurn(BaseModel):
    role: str
    content: str


class ApiChatRequest(BaseModel):
    query: str | None = None
    history: list[ApiChatTurn] = Field(default_factory=list)
    messages: list[dict[str, Any]] | None = None
    sourceKeys: list[str] | None = None
    model: str | None = None
    includeReasoning: bool = True
    threadId: str = ""


def parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*")
    return [origin.strip() for origin in raw.split(",") if origin.strip()] or ["*"]


def parse_bind_port() -> int:
    raw = os.getenv("AI_TUTOR_API_PORT") or os.getenv("PORT") or "8000"
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid API port: {raw!r}") from exc


def parse_bind_host() -> str:
    return (os.getenv("AI_TUTOR_API_HOST") or os.getenv("HOST") or "0.0.0.0").strip()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Warm up the Chroma retriever before serving any requests.

    LangGraph's ToolNode runs sync tools in a threadpool, and parallel
    `retrieve_tutor_context` calls can race into `PersistentClient(...)`
    at the same time on the first turn. That race occasionally surfaces
    `Could not connect to tenant default_tenant`. Touching the retriever
    at startup forces single-threaded tenant init.
    """
    if os.environ.get("COHERE_API_KEY"):
        try:
            get_retriever()
        except Exception as exc:  # pragma: no cover - diagnostic logging only
            logfire.warn(
                "Retriever warm-up failed; first retrieval call may retry.",
                error=str(exc),
            )
    yield


app = FastAPI(title="AI Tutor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse_frame(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n"


def extract_turns_from_messages(messages: list[dict[str, Any]]) -> tuple[ChatTurn, ...]:
    turns: list[ChatTurn] = []
    for message in messages:
        role = str(message.get("role", ""))
        if role not in {"user", "assistant"}:
            continue
        content = message.get("parts")
        if not content:
            content = message.get("content", "")
        turns.append(ChatTurn(role=role, content=message_content_to_text(content)))
    return tuple(turns)


def build_chat_request(payload: ApiChatRequest) -> ChatRequest:
    history = tuple(
        ChatTurn(role=turn.role, content=turn.content)
        for turn in payload.history
        if turn.role in {"user", "assistant"}
    )
    query = (payload.query or "").strip()

    if payload.messages:
        message_turns = extract_turns_from_messages(payload.messages)
        if not query:
            if not message_turns or message_turns[-1].role != "user":
                raise HTTPException(
                    status_code=422,
                    detail="messages must end with a user message when query is not provided",
                )
            query = message_turns[-1].content.strip()
            history = message_turns[:-1]
        elif message_turns:
            history = message_turns

    if not query:
        raise HTTPException(status_code=422, detail="query is required")

    allowed_source_keys = set(AVAILABLE_SOURCES)
    requested_source_keys = payload.sourceKeys or list(DEFAULT_SELECTED_SOURCE_KEYS)
    source_keys = tuple(
        dict.fromkeys(
            key for key in requested_source_keys if key in allowed_source_keys
        )
    ) or DEFAULT_SELECTED_SOURCE_KEYS
    return ChatRequest(
        query=query,
        history=history,
        source_keys=source_keys,
        model_name=(payload.model or DEFAULT_MODEL_NAME).strip(),
        include_reasoning=bool(payload.includeReasoning),
        thread_id=(payload.threadId or "").strip(),
    )


class UIMessageStreamEncoder:
    def __init__(self) -> None:
        self.message_id = ""
        self.text_block_id = ""
        self.active_reasoning_id = ""
        self.source_matches_by_call_id: dict[str, list[dict[str, Any]]] = {}
        self.closed = False

    def close_reasoning_block(self) -> list[dict[str, Any]]:
        if not self.active_reasoning_id:
            return []

        parts = [{"type": "reasoning-end", "id": self.active_reasoning_id}]
        self.active_reasoning_id = ""
        return parts

    def encode(self, event: ChatEvent) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []

        if event.type == "thread_started":
            parts.append(
                {
                    "type": "data-thread",
                    "data": {"threadId": str(event.data.get("thread_id", ""))},
                }
            )
            return parts

        if event.type == "message_started":
            self.message_id = str(event.data.get("message_id", uuid4().hex))
            self.active_reasoning_id = ""
            parts.append({"type": "start", "messageId": self.message_id})
            parts.append({"type": "start-step"})
            return parts

        if event.type == "text_delta":
            parts.extend(self.close_reasoning_block())
            if not self.text_block_id:
                self.text_block_id = f"text_{self.message_id or uuid4().hex}"
                parts.append({"type": "text-start", "id": self.text_block_id})
            parts.append(
                {
                    "type": "text-delta",
                    "id": self.text_block_id,
                    "delta": str(event.data.get("text", "")),
                }
            )
            return parts

        if event.type == "reasoning_delta":
            if not self.active_reasoning_id:
                self.active_reasoning_id = f"reasoning_{uuid4().hex[:8]}"
                parts.append({"type": "reasoning-start", "id": self.active_reasoning_id})
            parts.append(
                {
                    "type": "reasoning-delta",
                    "id": self.active_reasoning_id,
                    "delta": str(event.data.get("text", "")),
                }
            )
            return parts

        if event.type == "tool_call_started":
            parts.extend(self.close_reasoning_block())
            call_id = str(event.data.get("call_id", uuid4().hex))
            tool_name = str(event.data.get("tool_name", "tool"))
            parts.append(
                {
                    "type": "tool-input-start",
                    "toolCallId": call_id,
                    "toolName": tool_name,
                }
            )
            args = event.data.get("args")
            args_text = str(event.data.get("args_text", "")).strip()
            if isinstance(args, dict):
                parts.append(
                    {
                        "type": "tool-input-available",
                        "toolCallId": call_id,
                        "toolName": tool_name,
                        "input": args,
                    }
                )
            elif args_text:
                parts.append(
                    {
                        "type": "tool-input-available",
                        "toolCallId": call_id,
                        "toolName": tool_name,
                        "input": {"text": args_text},
                    }
                )
            return parts

        if event.type == "source_match":
            call_id = str(event.data.get("call_id", ""))
            source_data = {
                "docId": str(event.data.get("doc_id", "")),
                "title": str(event.data.get("title", "")),
                "url": str(event.data.get("url", "")),
                "sourceKey": str(event.data.get("source_key", "")),
                "sourceLabel": str(event.data.get("source_label", "")),
                "score": float(event.data.get("score", 0.0)),
            }
            if call_id:
                self.source_matches_by_call_id.setdefault(call_id, []).append(source_data)
            parts.append(
                {
                    "type": "source-url",
                    "sourceId": source_data["docId"] or source_data["url"],
                    "url": source_data["url"],
                }
            )
            parts.append(
                {
                    "type": "source-document",
                    "sourceId": source_data["docId"] or source_data["url"],
                    "mediaType": "text/html",
                    "title": source_data["title"],
                }
            )
            parts.append({"type": "data-source", "data": source_data})
            return parts

        if event.type == "tool_call_completed":
            parts.extend(self.close_reasoning_block())
            call_id = str(event.data.get("call_id", uuid4().hex))
            output = {
                "text": str(event.data.get("output_text", "")),
                "matches": self.source_matches_by_call_id.get(call_id, []),
            }
            parts.append(
                {
                    "type": "tool-output-available",
                    "toolCallId": call_id,
                    "output": output,
                }
            )
            return parts

        if event.type == "message_completed":
            parts.extend(self.close_reasoning_block())
            answer = str(event.data.get("answer", "")).strip()
            if answer and not self.text_block_id:
                self.text_block_id = f"text_{self.message_id or uuid4().hex}"
                parts.append({"type": "text-start", "id": self.text_block_id})
                parts.append(
                    {
                        "type": "text-delta",
                        "id": self.text_block_id,
                        "delta": answer,
                    }
            )
            if self.text_block_id:
                parts.append({"type": "text-end", "id": self.text_block_id})
            parts.append({"type": "finish-step"})
            parts.append({"type": "finish"})
            self.closed = True
            return parts

        return parts

    def finish_error(self, error_text: str) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = [{"type": "error", "errorText": error_text}]
        if not self.closed:
            parts.extend(self.close_reasoning_block())
            if self.text_block_id:
                parts.append({"type": "text-end", "id": self.text_block_id})
            if self.message_id:
                parts.append({"type": "finish-step"})
                parts.append({"type": "finish"})
            self.closed = True
        return parts


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


SOURCE_VERSIONS_PATH = Path("data/source_versions.json")


def _load_source_versions() -> dict[str, dict[str, Any]]:
    if not SOURCE_VERSIONS_PATH.exists():
        return {}
    try:
        with SOURCE_VERSIONS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


_SOURCE_VERSIONS: dict[str, dict[str, Any]] = _load_source_versions()


def _source_entries() -> list[dict[str, Any]]:
    defaults = set(DEFAULT_SELECTED_SOURCES_UI)
    entries: list[dict[str, Any]] = []
    for label in AVAILABLE_SOURCES_UI:
        key = SOURCE_UI_TO_KEY[label]
        group = "courses" if key in COURSE_SOURCE_KEYS else "docs"
        entry: dict[str, Any] = {
            "label": label,
            "key": key,
            "group": group,
            "selectedByDefault": label in defaults,
        }
        if group == "docs":
            version_info = _SOURCE_VERSIONS.get(key, {})
            entry["version"] = version_info.get("version")
            entry["indexedAt"] = version_info.get("indexedAt")
        entries.append(entry)
    return entries


def _tool_catalog(model_name: str) -> list[dict[str, Any]]:
    retrieval_tool: dict[str, Any] = {
        "key": "retrieval",
        "label": "Retrieval",
        "kind": "configurable",
        "active": True,
        "sources": _source_entries(),
    }
    tools: list[dict[str, Any]] = [retrieval_tool]
    if is_google_genai_model(model_name):
        tools.append(
            {
                "key": "web_search",
                "label": "Web search",
                "kind": "toggle",
                "active": True,
            }
        )
        tools.append(
            {
                "key": "url_context",
                "label": "URL reading",
                "kind": "toggle",
                "active": True,
            }
        )
    elif is_anthropic_model(model_name):
        tools.append(
            {
                "key": "web_search",
                "label": "Web search",
                "kind": "toggle",
                "active": True,
            }
        )
        tools.append(
            {
                "key": "web_fetch",
                "label": "Web fetch",
                "kind": "toggle",
                "active": True,
            }
        )
    return tools


@app.get("/api/tools")
def list_tools(model: str | None = None) -> dict[str, Any]:
    model_name = (model or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    return {
        "model": model_name,
        "availableModels": list(AVAILABLE_MODELS),
        "tools": _tool_catalog(model_name),
    }


@app.post("/api/chat")
async def chat(payload: ApiChatRequest) -> StreamingResponse:
    chat_request = build_chat_request(payload)

    async def event_stream():
        encoder = UIMessageStreamEncoder()
        try:
            async for event in stream_chat(chat_request):
                for part in encoder.encode(event):
                    yield sse_frame(part)
        except HTTPException:
            raise
        except Exception as exc:
            for part in encoder.finish_error(str(exc)):
                yield sse_frame(part)
        else:
            if not encoder.closed:
                for part in encoder.finish_error("stream ended without completion"):
                    yield sse_frame(part)
        yield sse_frame("[DONE]")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "x-vercel-ai-ui-message-stream": "v1",
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "scripts.api:app",
        host=parse_bind_host(),
        port=parse_bind_port(),
        reload=False,
    )
