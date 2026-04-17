from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts.api import app
from scripts.chat_types import ChatEvent


def parse_sse_payloads(raw_text: str) -> list[str]:
    return [
        line[len("data: ") :]
        for line in raw_text.splitlines()
        if line.startswith("data: ")
    ]


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
            all(source["group"] in {"courses", "docs"} for source in retrieval["sources"])
        )
        # Gemini is the default model, so web search + url reading are present.
        tool_keys = {tool["key"] for tool in tools}
        self.assertIn("web_search", tool_keys)
        self.assertIn("url_context", tool_keys)

    def test_list_tools_for_non_gemini_model(self) -> None:
        with TestClient(app) as client:
            response = client.get(
                "/api/tools", params={"model": "anthropic:claude-sonnet-4-6"}
            )

        self.assertEqual(response.status_code, 200)
        tool_keys = {tool["key"] for tool in response.json()["tools"]}
        self.assertIn("retrieval", tool_keys)
        self.assertNotIn("web_search", tool_keys)
        self.assertNotIn("url_context", tool_keys)

    def test_chat_stream_returns_ai_sdk_parts(self) -> None:
        async def fake_stream_chat(request):
            self.assertEqual(request.query, "What is RAG?")
            self.assertEqual(request.history[0].role, "assistant")
            self.assertEqual(request.history[0].content, "Previous answer")
            self.assertEqual(request.source_keys, ("langchain", "transformers"))
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
            yield ChatEvent("text_delta", {"message_id": "message_1", "text": "RAG combines "})
            yield ChatEvent("text_delta", {"message_id": "message_1", "text": "retrieval with generation."})
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
            "threadId": "thread_0",
        }

        with patch("scripts.api.stream_chat", fake_stream_chat):
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

    def test_chat_stream_emits_error_part(self) -> None:
        async def broken_stream_chat(_request):
            raise RuntimeError("backend failed")
            yield  # pragma: no cover

        with patch("scripts.api.stream_chat", broken_stream_chat):
            with TestClient(app) as client:
                with client.stream("POST", "/api/chat", json={"query": "Hello"}) as response:
                    body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        payloads = parse_sse_payloads(body)
        self.assertEqual(payloads[-1], "[DONE]")
        parts = [json.loads(item) for item in payloads[:-1]]
        self.assertEqual(parts[0]["type"], "error")
        self.assertEqual(parts[0]["errorText"], "backend failed")

    def test_chat_stream_restarts_reasoning_after_tool_activity(self) -> None:
        async def fake_stream_chat(_request):
            yield ChatEvent("message_started", {"message_id": "message_1"})
            yield ChatEvent("reasoning_delta", {"message_id": "message_1", "text": "First thought"})
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
            yield ChatEvent("reasoning_delta", {"message_id": "message_1", "text": "Second thought"})
            yield ChatEvent("text_delta", {"message_id": "message_1", "text": "Final answer"})
            yield ChatEvent(
                "message_completed",
                {
                    "message_id": "message_1",
                    "answer": "Final answer",
                },
            )

        with patch("scripts.api.stream_chat", fake_stream_chat):
            with TestClient(app) as client:
                with client.stream("POST", "/api/chat", json={"query": "Hello"}) as response:
                    body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        payloads = parse_sse_payloads(body)
        parts = [json.loads(item) for item in payloads[:-1]]
        part_types = [part["type"] for part in parts]

        self.assertEqual(part_types.count("reasoning-start"), 2)
        self.assertEqual(part_types.count("reasoning-end"), 2)
        self.assertLess(part_types.index("tool-input-start"), part_types.index("text-start"))


if __name__ == "__main__":
    unittest.main()
