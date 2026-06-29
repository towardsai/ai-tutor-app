from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .chat_service import (
    is_anthropic_model,
    is_google_genai_model,
    message_content_to_text,
    stream_chat,
    warm_up_retriever,
)
from .chat_types import ChatEvent, ChatRequest, ChatTurn
from .kb_manifest import available_source_keys, load_manifest_entries
from .memory_presets import MEMORY_PRESETS
from .config import (
    AVAILABLE_MODELS,
    AVAILABLE_SOURCES,
    AVAILABLE_SOURCES_UI,
    COURSE_SOURCE_KEYS,
    DEFAULT_MODEL_NAME,
    DEFAULT_SELECTED_SOURCE_KEYS,
    DEFAULT_SELECTED_SOURCES_UI,
    SOURCE_DISPLAY_INFO,
    SOURCE_UI_TO_KEY,
)

logger = logging.getLogger(__name__)


# Request-size bounds. The public Space takes anonymous traffic, and every
# accepted byte flows into the model call and the in-memory checkpointer, so
# absurd payloads must die at validation. The caps are generous for real use:
# a query is one pasted question/snippet, a turn is one rendered message, and
# an AI SDK message can carry several tool outputs (each token-budgeted).
MAX_QUERY_CHARS = 32_000
MAX_TURN_CHARS = 32_000
MAX_HISTORY_TURNS = 100
MAX_CLIENT_MESSAGES = 100
MAX_MESSAGE_JSON_CHARS = 200_000
MAX_BODY_BYTES = 2 * 1024 * 1024


class ApiChatTurn(BaseModel):
    role: str = Field(max_length=32)
    content: str = Field(max_length=MAX_TURN_CHARS)


class ApiChatRequest(BaseModel):
    query: str | None = Field(default=None, max_length=MAX_QUERY_CHARS)
    history: list[ApiChatTurn] = Field(
        default_factory=list, max_length=MAX_HISTORY_TURNS
    )
    messages: list[dict[str, Any]] | None = Field(
        default=None, max_length=MAX_CLIENT_MESSAGES
    )
    sourceKeys: list[str] | None = Field(default=None, max_length=100)
    enabledTools: list[str] | None = Field(default=None, max_length=50)
    model: str | None = Field(default=None, max_length=200)
    includeReasoning: bool = True
    threadId: str = Field(default="", max_length=128)
    # Memory/context-management preset (experiments + workshop toggles).
    # Omitted means the server-side default resolution order.
    memoryPreset: str | None = Field(default=None, max_length=64)
    # Long-term memory key for profile-memory presets.
    studentId: str = Field(default="", max_length=128)

    @field_validator("messages")
    @classmethod
    def _cap_message_size(
        cls, value: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        for message in value or []:
            if len(json.dumps(message, ensure_ascii=False)) > MAX_MESSAGE_JSON_CHARS:
                raise ValueError(
                    f"message exceeds {MAX_MESSAGE_JSON_CHARS} serialized characters"
                )
        return value


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

    Also pre-parse the KB corpus manifest (lru-cached): citation resolution
    reads it on first use, and warming it here keeps that synchronous parse
    off the event loop during the first answer.
    """
    warm_up_retriever()
    load_manifest_entries()
    yield


app = FastAPI(title="AI Tutor API", lifespan=lifespan)


@app.middleware("http")
async def reject_oversized_bodies(request: Request, call_next):
    """Refuse multi-megabyte bodies before parsing them. Pydantic caps bound
    the fields, but only after the whole body is read and JSON-decoded."""
    content_length = request.headers.get("content-length")
    try:
        too_large = content_length is not None and int(content_length) > MAX_BODY_BYTES
    except ValueError:
        too_large = False
    if too_large:
        return JSONResponse(
            {"detail": "Request body too large"},
            status_code=413,
        )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(),
    # Nothing uses cookies or HTTP auth, and credentials combined with the
    # default wildcard origin makes Starlette reflect any Origin with
    # Access-Control-Allow-Credentials: true, which would silently undermine
    # any future cookie-based auth.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse_frame(payload: dict[str, Any] | str) -> str:
    data = (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )
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
            # A client sending the full AI-SDK message list AND query set to
            # its trailing user turn means them as the same question; keeping
            # the turn in history would make the model see it twice.
            if history[-1].role == "user" and history[-1].content.strip() == query:
                history = history[:-1]

    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    # The Field cap only bounds payload.query; a query extracted from the
    # trailing message must obey the same model-input bound.
    if len(query) > MAX_QUERY_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"query exceeds {MAX_QUERY_CHARS} characters",
        )

    allowed_source_keys = set(AVAILABLE_SOURCES)
    if payload.sourceKeys is not None and not payload.sourceKeys:
        # An explicit empty selection is the user turning the knowledge base
        # off; it must not silently coerce to the defaults. Only an *omitted*
        # field means "use the defaults".
        source_keys: tuple[str, ...] = ()
    else:
        requested_source_keys = (
            payload.sourceKeys
            if payload.sourceKeys is not None
            else list(DEFAULT_SELECTED_SOURCE_KEYS)
        )
        source_keys = (
            tuple(
                dict.fromkeys(
                    key for key in requested_source_keys if key in allowed_source_keys
                )
            )
            or DEFAULT_SELECTED_SOURCE_KEYS
        )
    model_name = (payload.model or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    if model_name not in {model["id"] for model in AVAILABLE_MODELS}:
        raise HTTPException(status_code=422, detail="Unknown model")
    memory_preset = (payload.memoryPreset or "").strip()
    if memory_preset and memory_preset not in MEMORY_PRESETS:
        raise HTTPException(status_code=422, detail="Unknown memory preset")
    if payload.enabledTools is None:
        # `active` only governs the frontend's initial checkbox state; the
        # browser always sends an explicit enabledTools list (possibly []). A
        # direct API caller that OMITS the field falls through to here and gets
        # every toggle tool enabled regardless of `active` -- so url_context is
        # on, and keep_unresolved_sources with it. To keep url_context off (the
        # UI default), send an explicit enabledTools list, e.g. [] or
        # ["web_search"].
        enabled_tools = tuple(
            tool["key"]
            for tool in _tool_catalog(model_name)
            if tool["kind"] == "toggle"
        )
    else:
        allowed_tool_keys = {
            tool["key"]
            for tool in _tool_catalog(model_name)
            if tool["kind"] == "toggle"
        }
        enabled_tools = tuple(
            dict.fromkeys(
                key for key in payload.enabledTools if key in allowed_tool_keys
            )
        )
    return ChatRequest(
        query=query,
        history=history,
        source_keys=source_keys,
        model_name=model_name,
        include_reasoning=bool(payload.includeReasoning),
        thread_id=(payload.threadId or "").strip(),
        enabled_tools=enabled_tools,
        memory_preset=memory_preset,
        student_id=payload.studentId.strip(),
    )


class UIMessageStreamEncoder:
    def __init__(self) -> None:
        self.message_id = ""
        self.thread_id = ""
        self.text_block_id = ""
        self.text_block_count = 0
        self.emitted_text = False
        self.active_reasoning_id = ""
        self.open_tool_call_ids: list[str] = []
        self.announced_tool_call_ids: set[str] = set()
        self.closed = False

    def close_reasoning_block(self) -> list[dict[str, Any]]:
        if not self.active_reasoning_id:
            return []

        parts = [{"type": "reasoning-end", "id": self.active_reasoning_id}]
        self.active_reasoning_id = ""
        return parts

    def close_text_block(self) -> list[dict[str, Any]]:
        if not self.text_block_id:
            return []

        parts = [{"type": "text-end", "id": self.text_block_id}]
        self.text_block_id = ""
        return parts

    def next_text_block_id(self) -> str:
        self.text_block_count += 1
        return f"text_{self.message_id or uuid4().hex}_{self.text_block_count}"

    @staticmethod
    def source_data(data: dict[str, Any]) -> dict[str, Any]:
        return {
            "docId": str(data.get("doc_id", "")),
            "title": str(data.get("title", "")),
            "url": str(data.get("url", "")),
            "sourceKey": str(data.get("source_key", "")),
            "sourceLabel": str(data.get("source_label", "")),
            "score": float(data.get("score", 0.0)),
            "group": str(data.get("group", "")),
            # KB-root-relative path ("raw/docs/...") when the source is a KB
            # file; the client uses it to resolve inline `raw/...` citations
            # to this source's real URL.
            "path": str(data.get("path", "")),
        }

    def encode(self, event: ChatEvent) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []

        if event.type == "thread_started":
            self.thread_id = str(event.data.get("thread_id", ""))
            parts.append(
                {
                    "type": "data-thread",
                    "data": {"threadId": self.thread_id},
                    # Delivered to onData only, never stored in message.parts:
                    # nothing renders it, and a non-transient part emitted
                    # before "start" relies on undocumented AI SDK ordering
                    # behavior (the in-flight message is stored by reference).
                    "transient": True,
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
                self.text_block_id = self.next_text_block_id()
                parts.append({"type": "text-start", "id": self.text_block_id})
            parts.append(
                {
                    "type": "text-delta",
                    "id": self.text_block_id,
                    "delta": str(event.data.get("text", "")),
                }
            )
            self.emitted_text = True
            return parts

        if event.type == "reasoning_delta":
            delta = str(event.data.get("text", ""))
            if not self.active_reasoning_id:
                self.active_reasoning_id = f"reasoning_{uuid4().hex[:8]}"
                parts.append(
                    {"type": "reasoning-start", "id": self.active_reasoning_id}
                )
            elif event.data.get("is_block"):
                # Each block-granularity delta is a complete thought summary;
                # appending to an open block needs a paragraph break or the
                # next summary's title glues onto the previous sentence.
                delta = f"\n\n{delta}"
            parts.append(
                {
                    "type": "reasoning-delta",
                    "id": self.active_reasoning_id,
                    "delta": delta,
                }
            )
            return parts

        if event.type == "tool_call_started":
            parts.extend(self.close_reasoning_block())
            # Close any open text block (e.g. a preamble before the tool
            # call) so the answer streamed after the tools becomes its own
            # part and the activity block renders before it, not after.
            parts.extend(self.close_text_block())
            call_id = str(event.data.get("call_id", uuid4().hex))
            tool_name = str(event.data.get("tool_name", "tool"))
            if call_id not in self.open_tool_call_ids:
                self.open_tool_call_ids.append(call_id)
            self.announced_tool_call_ids.add(call_id)
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
            source_data = self.source_data(event.data)
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
            args = event.data.get("args")
            if (
                isinstance(args, dict) and args
            ) or call_id not in self.announced_tool_call_ids:
                # Providers that stream tool calls incrementally announce the
                # call before its args are parsed; refresh the input now that
                # the full args are known. A call id the stream never
                # announced (e.g. a ToolMessage with a missing id) must also
                # get an input part first: the AI SDK client throws on an
                # output for an unknown tool call and drops the rest of the
                # stream.
                parts.append(
                    {
                        "type": "tool-input-available",
                        "toolCallId": call_id,
                        "toolName": str(event.data.get("tool_name", "tool")),
                        "input": args if isinstance(args, dict) else {},
                    }
                )
                self.announced_tool_call_ids.add(call_id)
            output = {
                "text": str(event.data.get("output_text", "")),
                "matches": [
                    self.source_data(match)
                    for match in event.data.get("matches") or []
                    if isinstance(match, dict)
                ],
            }
            if call_id in self.open_tool_call_ids:
                self.open_tool_call_ids.remove(call_id)
            parts.append(
                {
                    "type": "tool-output-available",
                    "toolCallId": call_id,
                    "output": output,
                }
            )
            return parts

        if event.type == "context_stats":
            data = event.data
            parts.append(
                {
                    "type": "data-context-stats",
                    "data": {
                        "messageId": str(data.get("message_id", "")),
                        "memoryPreset": str(data.get("memory_preset", "")),
                        "llmCalls": data.get("llm_calls"),
                        "inputTokens": data.get("input_tokens"),
                        "outputTokens": data.get("output_tokens"),
                        "totalTokens": data.get("total_tokens"),
                        "cacheReadTokens": data.get("cache_read_tokens"),
                        "cacheCreationTokens": data.get("cache_creation_tokens"),
                        # None when a used model has no price-table entry;
                        # the client must render that as unknown, not $0.
                        "estCostUsd": data.get("est_cost_usd"),
                        "ttftMs": data.get("ttft_ms"),
                        "totalMs": data.get("total_ms"),
                        "contextMessages": data.get("context_messages"),
                        "contextTokensApprox": data.get("context_tokens_approx"),
                        "summaryMessages": data.get("summary_messages"),
                        "clearedToolOutputs": data.get("cleared_tool_outputs"),
                    },
                    # Delivered to onData for a live meter; not stored in
                    # message.parts (nothing renders it as message content).
                    "transient": True,
                }
            )
            return parts

        if event.type == "message_completed":
            parts.extend(self.close_reasoning_block())
            answer = str(event.data.get("answer", "")).strip()
            # `answer` is every delta of the whole turn joined: when any text
            # already streamed (even in a block a tool call closed), emitting
            # it again would duplicate the rendered text. The fallback is only
            # for answers that never streamed.
            if answer and not self.text_block_id and not self.emitted_text:
                self.text_block_id = self.next_text_block_id()
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
            # Close tool calls the turn never finished, or the UI shows
            # them as running forever.
            for call_id in self.open_tool_call_ids:
                parts.append(
                    {
                        "type": "tool-output-error",
                        "toolCallId": call_id,
                        "errorText": "The response ended before this tool call finished.",
                    }
                )
            self.open_tool_call_ids = []
            if self.message_id:
                parts.append({"type": "finish-step"})
                parts.append({"type": "finish"})
            self.closed = True
        return parts


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


# Anchored to the repo root (like FRONTEND_OUT_DIR), not the process CWD:
# the server must find it regardless of where uvicorn was started from.
SOURCE_VERSIONS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "source_versions.json"
)


def _load_source_versions() -> dict[str, dict[str, Any]]:
    if not SOURCE_VERSIONS_PATH.exists():
        logger.warning(
            "source versions file missing; docs sources get version=null. path=%s",
            SOURCE_VERSIONS_PATH,
        )
        return {}
    try:
        with SOURCE_VERSIONS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning(
            "source versions file unreadable; docs sources get version=null. path=%s",
            SOURCE_VERSIONS_PATH,
            exc_info=True,
        )
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "source versions file is not a JSON object; docs sources get "
            "version=null. path=%s",
            SOURCE_VERSIONS_PATH,
        )
        return {}
    return data


_SOURCE_VERSIONS: dict[str, dict[str, Any]] = _load_source_versions()


def _source_entries() -> list[dict[str, Any]]:
    defaults = set(DEFAULT_SELECTED_SOURCES_UI)
    # On the public docs-only bundle the course sources are absent, so hide
    # them from the picker instead of offering sources that return nothing.
    # `None` means the bundle is not loaded yet (advertise everything, the
    # prod default); otherwise only advertise sources present in the bundle.
    available = available_source_keys()
    entries: list[dict[str, Any]] = []
    for label in AVAILABLE_SOURCES_UI:
        key = SOURCE_UI_TO_KEY[label]
        if available is not None and key not in available:
            continue
        group = "courses" if key in COURSE_SOURCE_KEYS else "docs"
        display = SOURCE_DISPLAY_INFO.get(key, {})
        entry: dict[str, Any] = {
            "label": label,
            # Display metadata is registry-owned (single source of truth);
            # the frontend renders these verbatim instead of keeping its own
            # per-source maps or reshaping the label.
            "shortLabel": display.get("ui_label") or label,
            "description": display.get("description"),
            "infoUrl": display.get("url"),
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
        "label": "Knowledge base",
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
                # Default OFF. The frontend seeds its initial toggle state from
                # `active`, so url_context starts unchecked on a fresh load.
                # Enabling it flips keep_unresolved_sources=True for that turn
                # (chat_service.stream_chat), which surfaces cited-but-ungrounded
                # URLs as low-trust Web chips; users opt in per turn.
                "kind": "toggle",
                "active": False,
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


# Reverse proxies (HF Spaces' router included) enforce idle timeouts; a slow
# provider step or a long tool batch can keep the wire silent past them. SSE
# comment frames are ignored by clients but count as traffic.
SSE_HEARTBEAT_SECONDS = 15.0


class _ThreadRunSlot:
    __slots__ = ("lock", "refs")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.refs = 0


# Concurrent runs on the same client-supplied thread id race the
# read-compare-write in sync_thread_with_history and interleave checkpoint
# writes on the shared InMemorySaver; serialize them per thread. Slots are
# refcounted so the dict only holds ids with an active or queued run.
_THREAD_RUN_SLOTS: dict[str, _ThreadRunSlot] = {}


def _claim_thread_slot(thread_id: str) -> _ThreadRunSlot | None:
    if not thread_id:
        return None
    slot = _THREAD_RUN_SLOTS.setdefault(thread_id, _ThreadRunSlot())
    slot.refs += 1
    return slot


def _release_thread_slot(thread_id: str, slot: _ThreadRunSlot) -> None:
    slot.refs -= 1
    if slot.refs <= 0 and _THREAD_RUN_SLOTS.get(thread_id) is slot:
        del _THREAD_RUN_SLOTS[thread_id]


@app.post("/api/chat")
async def chat(payload: ApiChatRequest) -> StreamingResponse:
    chat_request = build_chat_request(payload)

    async def event_stream():
        encoder = UIMessageStreamEncoder()
        slot = _claim_thread_slot(chat_request.thread_id)
        holds_lock = False
        events = stream_chat(chat_request)
        next_event: asyncio.Task | None = None
        try:
            if slot is not None:
                # Wait for any in-flight run on this thread, heartbeating so
                # the queued request isn't idled out by proxies meanwhile.
                acquire = asyncio.ensure_future(slot.lock.acquire())
                try:
                    while True:
                        done, _pending = await asyncio.wait(
                            {acquire}, timeout=SSE_HEARTBEAT_SECONDS
                        )
                        if done:
                            acquire.result()
                            holds_lock = True
                            break
                        yield ": ping\n\n"
                finally:
                    if not holds_lock:
                        acquire.cancel()

            next_event = asyncio.ensure_future(anext(events))
            while True:
                # asyncio.wait (not wait_for): a timeout must leave the
                # pending anext running — cancelling it would unwind the
                # generator mid-turn.
                done, _pending = await asyncio.wait(
                    {next_event}, timeout=SSE_HEARTBEAT_SECONDS
                )
                if not done:
                    yield ": ping\n\n"
                    continue
                task, next_event = next_event, None
                try:
                    event = task.result()
                except StopAsyncIteration:
                    break
                for part in encoder.encode(event):
                    yield sse_frame(part)
                next_event = asyncio.ensure_future(anext(events))
        except Exception:
            ref = encoder.message_id or uuid4().hex
            logger.exception(
                "chat stream failed ref=%s thread=%s",
                ref,
                encoder.thread_id or "?",
            )
            message = (
                "Something went wrong while answering. Please try again. "
                f"If it keeps happening, reference: {ref}"
            )
            for part in encoder.finish_error(message):
                yield sse_frame(part)
        else:
            if not encoder.closed:
                for part in encoder.finish_error("stream ended without completion"):
                    yield sse_frame(part)
        finally:
            # Client disconnect (or any exit) must unwind stream_chat so its
            # cleanup (KB budget, transcript recording) runs promptly.
            if next_event is not None:
                next_event.cancel()
            if holds_lock:
                slot.lock.release()
            if slot is not None:
                _release_thread_slot(chat_request.thread_id, slot)
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


FRONTEND_OUT_DIR = Path(__file__).resolve().parent.parent / "frontend" / "out"

if FRONTEND_OUT_DIR.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_OUT_DIR), html=True),
        name="frontend",
    )
else:
    logger.info(
        "Frontend static export not found; skipping static mount. path=%s",
        FRONTEND_OUT_DIR,
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host=parse_bind_host(),
        port=parse_bind_port(),
        reload=False,
    )
