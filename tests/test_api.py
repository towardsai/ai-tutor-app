from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import unittest
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

import pytest
import uvicorn
from fastapi.testclient import TestClient

from app.api import app
from app.chat_types import ChatEvent


def parse_sse_payloads(raw_text: str) -> list[str]:
    return [
        line[len("data: ") :]
        for line in raw_text.splitlines()
        if line.startswith("data: ")
    ]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ApiTestCase(unittest.TestCase):
    def test_healthcheck(self) -> None:
        with TestClient(app) as client:
            response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_list_tools(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/tools")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("tools", body)
        tools = body["tools"]
        retrieval = next(tool for tool in tools if tool["key"] == "retrieval")
        self.assertEqual(retrieval["kind"], "configurable")
        self.assertTrue(retrieval["sources"])
        self.assertTrue(
            any(source["selectedByDefault"] for source in retrieval["sources"])
        )
        self.assertTrue(
            all(
                source["group"] in {"courses", "docs"}
                for source in retrieval["sources"]
            )
        )
        # Gemini is the default model, so web search + url reading are present.
        tool_keys = {tool["key"] for tool in tools}
        self.assertIn("web_search", tool_keys)
        self.assertIn("url_context", tool_keys)

    def test_list_tools_for_anthropic_model(self) -> None:
        with TestClient(app) as client:
            response = client.get(
                "/api/tools", params={"model": "anthropic:claude-sonnet-4-6"}
            )

        self.assertEqual(response.status_code, 200)
        tool_keys = {tool["key"] for tool in response.json()["tools"]}
        self.assertIn("retrieval", tool_keys)
        self.assertIn("web_search", tool_keys)
        self.assertIn("web_fetch", tool_keys)
        self.assertNotIn("url_context", tool_keys)

    def test_chat_stream_returns_ai_sdk_parts(self) -> None:
        async def fake_stream_chat(request):
            self.assertEqual(request.query, "What is RAG?")
            self.assertEqual(request.history[0].role, "assistant")
            self.assertEqual(request.history[0].content, "Previous answer")
            self.assertEqual(request.source_keys, ("langchain", "transformers"))
            self.assertEqual(request.enabled_tools, ("web_search",))
            self.assertEqual(request.thread_id, "thread_0")
            yield ChatEvent("thread_started", {"thread_id": "thread_1"})
            yield ChatEvent("message_started", {"message_id": "message_1"})
            yield ChatEvent(
                "reasoning_delta",
                {"message_id": "message_1", "step": "model", "text": "Need retrieval"},
            )
            yield ChatEvent(
                "tool_call_started",
                {
                    "message_id": "message_1",
                    "call_id": "call_1",
                    "tool_name": "retrieve_tutor_context",
                    "args": {"query": "What is RAG?"},
                    "args_text": "What is RAG?",
                },
            )
            yield ChatEvent(
                "source_match",
                {
                    "message_id": "message_1",
                    "call_id": "call_1",
                    "doc_id": "doc_1",
                    "title": "RAG overview",
                    "url": "https://example.com/rag",
                    "source_key": "langchain",
                    "source_label": "LangChain Docs",
                    "score": 0.91,
                },
            )
            yield ChatEvent(
                "tool_call_completed",
                {
                    "message_id": "message_1",
                    "call_id": "call_1",
                    "tool_name": "retrieve_tutor_context",
                    "args": {"query": "What is RAG?"},
                    "args_text": "What is RAG?",
                    "output_text": "payload",
                },
            )
            yield ChatEvent(
                "text_delta", {"message_id": "message_1", "text": "RAG combines "}
            )
            yield ChatEvent(
                "text_delta",
                {"message_id": "message_1", "text": "retrieval with generation."},
            )
            yield ChatEvent(
                "message_completed",
                {
                    "message_id": "message_1",
                    "thread_id": "thread_1",
                    "answer": "RAG combines retrieval with generation.",
                },
            )

        payload = {
            "messages": [
                {"role": "assistant", "content": "Previous answer"},
                {
                    "role": "user",
                    "parts": [{"type": "text", "text": "What is RAG?"}],
                },
            ],
            "sourceKeys": ["langchain", "transformers"],
            "enabledTools": ["web_search", "not_a_real_tool"],
            "threadId": "thread_0",
        }

        with patch("app.api.stream_chat", fake_stream_chat):
            with TestClient(app) as client:
                with client.stream("POST", "/api/chat", json=payload) as response:
                    body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["x-vercel-ai-ui-message-stream"],
            "v1",
        )

        payloads = parse_sse_payloads(body)
        self.assertEqual(payloads[-1], "[DONE]")
        parts = [json.loads(item) for item in payloads[:-1]]
        part_types = [part["type"] for part in parts]

        self.assertIn("data-thread", part_types)
        self.assertIn("start", part_types)
        self.assertIn("start-step", part_types)
        self.assertIn("reasoning-start", part_types)
        self.assertIn("reasoning-delta", part_types)
        self.assertIn("tool-input-start", part_types)
        self.assertIn("tool-input-available", part_types)
        self.assertIn("source-url", part_types)
        self.assertIn("source-document", part_types)
        self.assertIn("data-source", part_types)
        self.assertIn("tool-output-available", part_types)
        self.assertIn("text-start", part_types)
        self.assertIn("text-delta", part_types)
        self.assertIn("text-end", part_types)
        self.assertIn("finish-step", part_types)
        self.assertIn("finish", part_types)

    def test_chat_stream_emits_sanitized_error_part(self) -> None:
        async def broken_stream_chat(_request):
            yield ChatEvent("thread_started", {"thread_id": "thread_1"})
            yield ChatEvent("message_started", {"message_id": "message_1"})
            raise RuntimeError("backend failed: cohere trace secret-123")

        with patch("app.api.stream_chat", broken_stream_chat):
            with TestClient(app) as client:
                with client.stream(
                    "POST", "/api/chat", json={"query": "Hello"}
                ) as response:
                    body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        payloads = parse_sse_payloads(body)
        self.assertEqual(payloads[-1], "[DONE]")
        parts = [json.loads(item) for item in payloads[:-1]]
        error_parts = [part for part in parts if part.get("type") == "error"]
        self.assertEqual(len(error_parts), 1)
        error_text = error_parts[0]["errorText"]
        # The raw upstream error must not leak to the client...
        self.assertNotIn("backend failed", error_text)
        self.assertNotIn("secret-123", error_text)
        # ...but the correlation ref (message_id) is surfaced for support.
        self.assertIn("message_1", error_text)

    def test_chat_stream_restarts_reasoning_after_tool_activity(self) -> None:
        async def fake_stream_chat(_request):
            yield ChatEvent("message_started", {"message_id": "message_1"})
            yield ChatEvent(
                "reasoning_delta", {"message_id": "message_1", "text": "First thought"}
            )
            yield ChatEvent(
                "tool_call_started",
                {
                    "message_id": "message_1",
                    "call_id": "call_1",
                    "tool_name": "retrieve_tutor_context",
                    "args": {"query": "What is RAG?"},
                    "args_text": "What is RAG?",
                },
            )
            yield ChatEvent(
                "tool_call_completed",
                {
                    "message_id": "message_1",
                    "call_id": "call_1",
                    "output_text": "payload",
                },
            )
            yield ChatEvent(
                "reasoning_delta", {"message_id": "message_1", "text": "Second thought"}
            )
            yield ChatEvent(
                "text_delta", {"message_id": "message_1", "text": "Final answer"}
            )
            yield ChatEvent(
                "message_completed",
                {
                    "message_id": "message_1",
                    "answer": "Final answer",
                },
            )

        with patch("app.api.stream_chat", fake_stream_chat):
            with TestClient(app) as client:
                with client.stream(
                    "POST", "/api/chat", json={"query": "Hello"}
                ) as response:
                    body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        payloads = parse_sse_payloads(body)
        parts = [json.loads(item) for item in payloads[:-1]]
        part_types = [part["type"] for part in parts]

        self.assertEqual(part_types.count("reasoning-start"), 2)
        self.assertEqual(part_types.count("reasoning-end"), 2)
        self.assertLess(
            part_types.index("tool-input-start"), part_types.index("text-start")
        )


LIVE_API_E2E = pytest.mark.skipif(
    os.getenv("RUN_LIVE_API_E2E") != "1",
    reason="Set RUN_LIVE_API_E2E=1 to run the live frontend API smoke test.",
)


def require_live_api_e2e_prereqs() -> None:
    if shutil.which("curl") is None:
        pytest.skip("curl is required for the live API smoke test.")
    if not os.getenv("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY is required for live retrieval.")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY is required for the live model.")
    if not os.path.exists("data/kb/wiki/index.md"):
        pytest.skip("data/kb artifacts must exist before running the live smoke test.")


@contextmanager
def live_api_server() -> Iterator[int]:
    port = free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            health = subprocess.run(
                ["curl", "-s", f"http://127.0.0.1:{port}/healthz"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if health.returncode == 0 and "ok" in health.stdout:
                break
            time.sleep(0.25)
        else:
            pytest.fail("API server did not become healthy")
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def live_chat_payload(
    messages: list[dict],
    thread_id: str = "",
    enabled_tools: list[str] | None = None,
) -> dict:
    return {
        "messages": messages,
        "sourceKeys": ["peft", "transformers"],
        "enabledTools": enabled_tools or [],
        "model": os.getenv("LIVE_API_E2E_MODEL", "google-genai:gemini-3.5-flash"),
        "includeReasoning": False,
        "threadId": thread_id,
    }


def post_live_api_chat(port: int, payload: dict) -> list[dict]:
    response = subprocess.run(
        [
            "curl",
            "-sN",
            f"http://127.0.0.1:{port}/api/chat",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=240,
    )
    payloads = parse_sse_payloads(response.stdout)
    assert payloads[-1] == "[DONE]"
    return [json.loads(item) for item in payloads[:-1]]


def run_live_api_chat(prompt: str) -> list[dict]:
    with live_api_server() as port:
        return post_live_api_chat(port, live_chat_payload([live_user_message(prompt)]))


def live_user_message(text: str) -> dict:
    return {"role": "user", "parts": [{"type": "text", "text": text}]}


def live_assistant_message(parts: list[dict]) -> dict:
    """Rebuild the assistant UIMessage the AI SDK keeps from streamed parts:
    one text part per text block id, deltas concatenated in arrival order."""
    text_blocks: dict[str, str] = {}
    for part in parts:
        if part["type"] == "text-delta":
            block_id = part["id"]
            text_blocks[block_id] = text_blocks.get(block_id, "") + part["delta"]
    return {
        "role": "assistant",
        "parts": [{"type": "text", "text": text} for text in text_blocks.values()],
    }


def live_thread_id(parts: list[dict]) -> str:
    for part in parts:
        if part["type"] == "data-thread":
            return str(part["data"].get("threadId", ""))
    return ""


def live_part_types(parts: list[dict]) -> list[str]:
    return [part["type"] for part in parts]


def live_tool_names(parts: list[dict]) -> set[str]:
    return {
        part.get("toolName")
        for part in parts
        if part["type"] in {"tool-input-start", "tool-input-available"}
    }


def live_answer_text(parts: list[dict]) -> str:
    return "\n".join(
        part.get("delta", "") for part in parts if part["type"] == "text-delta"
    )


def live_sources(parts: list[dict]) -> list[dict]:
    return [part["data"] for part in parts if part["type"] == "data-source"]


@LIVE_API_E2E
def test_live_api_stream_exposes_frontend_parts() -> None:
    require_live_api_e2e_prereqs()

    parts = run_live_api_chat(
        "Use both retrieve_tutor_context and run_kb_command to answer: "
        "how does PEFT configure LoRA with LoraConfig?"
    )
    part_types = [part["type"] for part in parts]
    assert "data-thread" in part_types
    assert "tool-input-start" in part_types
    assert "tool-output-available" in part_types
    assert "text-delta" in part_types
    assert "source-url" in part_types
    assert "source-document" in part_types
    assert "data-source" in part_types
    assert "finish" in part_types
    tool_names = live_tool_names(parts)
    assert {"retrieve_tutor_context", "run_kb_command"} <= tool_names
    final_text = live_answer_text(parts)
    assert "LoRA" in final_text
    assert re.search(r"\[[^\]]+\]\((?:https://|raw/|kb://doc/)", final_text)
    assert "### Sources" not in final_text
    sources = live_sources(parts)
    assert any(source["sourceKey"] in {"peft", "transformers"} for source in sources)


@LIVE_API_E2E
def test_live_api_follow_up_reuses_thread_after_tool_turn() -> None:
    """A follow-up that echoes the streamed turn back must keep the thread id.

    This is the continuity regression: tool-using turns checkpoint as several
    AI messages, and a naive transcript comparison branched to a new thread on
    every follow-up, discarding checkpointed tool context and summaries.
    """
    require_live_api_e2e_prereqs()

    first_prompt = (
        "Use retrieve_tutor_context to answer: how does PEFT configure LoRA "
        "with LoraConfig?"
    )
    follow_up_prompt = (
        "Thanks. In one sentence, restate the key point of your previous answer."
    )

    with live_api_server() as port:
        first_parts = post_live_api_chat(
            port, live_chat_payload([live_user_message(first_prompt)])
        )
        first_thread = live_thread_id(first_parts)
        assert first_thread
        assert live_tool_names(first_parts), "first turn must exercise a tool"
        assert "finish" in live_part_types(first_parts)

        follow_up_messages = [
            live_user_message(first_prompt),
            live_assistant_message(first_parts),
            live_user_message(follow_up_prompt),
        ]
        second_parts = post_live_api_chat(
            port, live_chat_payload(follow_up_messages, thread_id=first_thread)
        )

        assert "finish" in live_part_types(second_parts)
        assert live_answer_text(second_parts)
        second_thread = live_thread_id(second_parts)
        assert second_thread == first_thread, (
            f"follow-up branched to a new thread: {first_thread} -> {second_thread}"
        )


@LIVE_API_E2E
def test_live_api_edit_forks_thread_from_checkpoint() -> None:
    """Editing a non-first message must fork the thread, not branch it.

    The history the client keeps is an exact prefix of the thread's tracked
    transcript, so sync resolves it to the checkpoint at that turn boundary
    (LangGraph time travel) and the thread id stays stable. A plain rebuild
    would surface as a fresh thread id in data-thread.
    """
    require_live_api_e2e_prereqs()

    first_prompt = (
        "Use retrieve_tutor_context to answer briefly: how does PEFT configure "
        "LoRA with LoraConfig?"
    )
    second_prompt = "In one sentence, what does the r parameter control?"
    edited_second_prompt = "In one sentence, what does lora_alpha control?"

    with live_api_server() as port:
        first_parts = post_live_api_chat(
            port, live_chat_payload([live_user_message(first_prompt)])
        )
        thread_id = live_thread_id(first_parts)
        assert thread_id

        history = [
            live_user_message(first_prompt),
            live_assistant_message(first_parts),
        ]
        second_parts = post_live_api_chat(
            port,
            live_chat_payload(
                [*history, live_user_message(second_prompt)],
                thread_id=thread_id,
            ),
        )
        assert live_thread_id(second_parts) == thread_id

        # Edit the second question: the client keeps only the first turn.
        edited_parts = post_live_api_chat(
            port,
            live_chat_payload(
                [*history, live_user_message(edited_second_prompt)],
                thread_id=thread_id,
            ),
        )

        assert "finish" in live_part_types(edited_parts)
        assert live_answer_text(edited_parts)
        edited_thread = live_thread_id(edited_parts)
        assert edited_thread == thread_id, (
            f"edit branched instead of forking: {thread_id} -> {edited_thread}"
        )


@LIVE_API_E2E
def test_live_api_pasted_url_gets_source_card_with_url_context() -> None:
    """The paste-a-URL flow must not lose its source card.

    Gemini's url_context fetches never reach the stream (the library drops
    url_context_metadata), so the pasted URL is synthesized as web evidence;
    whether the model cites it inline or cites nothing, a source card for
    the page must surface.
    """
    require_live_api_e2e_prereqs()

    page_url = "https://huggingface.co/docs/peft/index"
    prompt = (
        f"Read {page_url} and answer in two sentences: what is PEFT? "
        "Cite that page inline in your answer."
    )

    with live_api_server() as port:
        parts = post_live_api_chat(
            port,
            live_chat_payload(
                [live_user_message(prompt)],
                enabled_tools=["url_context"],
            ),
        )

    assert "finish" in live_part_types(parts)
    assert live_answer_text(parts)
    sources = live_sources(parts)
    assert any(source["url"] == page_url for source in sources), (
        f"pasted URL missing from source cards: {sources}"
    )


@LIVE_API_E2E
def test_live_api_stream_obeys_retrieval_only_instruction() -> None:
    require_live_api_e2e_prereqs()

    parts = run_live_api_chat(
        "Use only the retrieve_tutor_context tool. Do not call run_kb_command. "
        "Answer: how does PEFT configure LoRA with LoraConfig?"
    )

    assert "data-source" in live_part_types(parts)
    tool_names = live_tool_names(parts)
    assert "retrieve_tutor_context" in tool_names
    assert "run_kb_command" not in tool_names
    assert "LoRA" in live_answer_text(parts)
    assert live_sources(parts)


@LIVE_API_E2E
def test_live_api_stream_obeys_shell_only_instruction() -> None:
    require_live_api_e2e_prereqs()

    parts = run_live_api_chat(
        "Use only the run_kb_command tool. Do not call retrieve_tutor_context. "
        "Inspect raw PEFT docs and answer: how does PEFT configure LoRA with "
        "LoraConfig?"
    )

    assert "data-source" in live_part_types(parts)
    tool_names = live_tool_names(parts)
    assert "run_kb_command" in tool_names
    assert "retrieve_tutor_context" not in tool_names
    assert "LoRA" in live_answer_text(parts)
    assert live_sources(parts)


if __name__ == "__main__":
    unittest.main()
