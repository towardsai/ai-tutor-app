from __future__ import annotations

import json
import os
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chat_service import message_content_to_text, stream_chat
from .chat_types import ChatEvent, ChatRequest, ChatTurn
from .setup import (
    AVAILABLE_SOURCES_UI,
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


app = FastAPI(title="AI Tutor API")
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

    source_keys = tuple(payload.sourceKeys or DEFAULT_SELECTED_SOURCE_KEYS)
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
        self.reasoning_block_ids: dict[str, str] = {}
        self.source_matches_by_call_id: dict[str, list[dict[str, Any]]] = {}
        self.closed = False

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
            parts.append({"type": "start", "messageId": self.message_id})
            parts.append({"type": "start-step"})
            return parts

        if event.type == "text_delta":
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
            step = str(event.data.get("step", "")) or "default"
            reasoning_id = self.reasoning_block_ids.get(step)
            if reasoning_id is None:
                reasoning_id = f"reasoning_{step}_{uuid4().hex[:8]}"
                self.reasoning_block_ids[step] = reasoning_id
                parts.append({"type": "reasoning-start", "id": reasoning_id})
            parts.append(
                {
                    "type": "reasoning-delta",
                    "id": reasoning_id,
                    "delta": str(event.data.get("text", "")),
                }
            )
            return parts

        if event.type == "tool_call_started":
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
            for reasoning_id in self.reasoning_block_ids.values():
                parts.append({"type": "reasoning-end", "id": reasoning_id})
            parts.append({"type": "finish-step"})
            parts.append({"type": "finish"})
            self.closed = True
            return parts

        return parts

    def finish_error(self, error_text: str) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = [{"type": "error", "errorText": error_text}]
        if not self.closed:
            if self.text_block_id:
                parts.append({"type": "text-end", "id": self.text_block_id})
            for reasoning_id in self.reasoning_block_ids.values():
                parts.append({"type": "reasoning-end", "id": reasoning_id})
            if self.message_id:
                parts.append({"type": "finish-step"})
                parts.append({"type": "finish"})
            self.closed = True
        return parts


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sources")
def list_sources() -> dict[str, Any]:
    defaults = set(DEFAULT_SELECTED_SOURCES_UI)
    return {
        "sources": [
            {
                "label": label,
                "key": SOURCE_UI_TO_KEY[label],
                "selectedByDefault": label in defaults,
            }
            for label in AVAILABLE_SOURCES_UI
        ]
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
    uvicorn.run("scripts.api:app", host="0.0.0.0", port=8000, reload=False)
