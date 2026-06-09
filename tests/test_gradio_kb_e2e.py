from __future__ import annotations

import socket
import os

import pytest
from gradio_client import Client

from scripts.chat_types import ChatEvent


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_gradio_chat_api_surfaces_shell_and_retrieval_activity(monkeypatch) -> None:
    import scripts.main as main

    async def fake_stream_chat(_request):
        yield ChatEvent("thread_started", {"thread_id": "thread_kb"})
        yield ChatEvent("message_started", {"message_id": "message_kb"})
        yield ChatEvent(
            "tool_call_started",
            {
                "message_id": "message_kb",
                "call_id": "call_search",
                "tool_name": "retrieve_tutor_context",
                "args": {"query": "generate_next_queries_tool"},
                "args_text": "generate_next_queries_tool",
            },
        )
        yield ChatEvent(
            "tool_call_completed",
            {
                "message_id": "message_kb",
                "call_id": "call_search",
                "tool_name": "retrieve_tutor_context",
                "args": {"query": "generate_next_queries_tool"},
                "args_text": "generate_next_queries_tool",
                "output_text": (
                    '{"matches": [{"doc_id": "agentic_ai_engineering:lesson-18", '
                    '"title": "Lesson 18: Research Loop", '
                    '"url": "https://academy.towardsai.net/lesson-18", '
                    '"source": "agentic_ai_engineering", "score": 10.0}]}'
                ),
            },
        )
        yield ChatEvent(
            "tool_call_started",
            {
                "message_id": "message_kb",
                "call_id": "call_rg",
                "tool_name": "run_kb_command",
                "args": {"command": "rg generate_next_queries_tool raw"},
                "args_text": "rg generate_next_queries_tool raw",
            },
        )
        yield ChatEvent(
            "tool_call_completed",
            {
                "message_id": "message_kb",
                "call_id": "call_rg",
                "tool_name": "run_kb_command",
                "args": {"command": "rg generate_next_queries_tool raw"},
                "args_text": "rg generate_next_queries_tool raw",
                "output_text": "$ rg generate_next_queries_tool raw\ncwd: data/kb\nexit_code: 0\nstdout:\nraw/courses/agentic_ai_engineering/lesson-18.md:4:Call `generate_next_queries_tool`.",
            },
        )
        yield ChatEvent(
            "text_delta",
            {
                "message_id": "message_kb",
                "text": "The symbol appears in Lesson 18: Research Loop.",
            },
        )
        yield ChatEvent(
            "source_match",
            {
                "message_id": "message_kb",
                "doc_id": "agentic_ai_engineering:lesson-18",
                "title": "Lesson 18: Research Loop",
                "url": "https://academy.towardsai.net/lesson-18",
                "source_key": "agentic_ai_engineering",
                "source_label": "Agentic AI Engineering",
                "score": 10.0,
                "group": "courses",
            },
        )
        yield ChatEvent(
            "message_completed",
            {
                "message_id": "message_kb",
                "thread_id": "thread_kb",
                "answer": "The symbol appears in Lesson 18: Research Loop.",
            },
        )

    monkeypatch.setattr(main, "stream_chat", fake_stream_chat)
    port = free_port()
    main.demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        quiet=True,
    )
    try:
        client = Client(f"http://127.0.0.1:{port}")
        result = client.predict(
            "Where is generate_next_queries_tool discussed?",
            [],
            ["Agentic AI Engineering"],
            "google-genai:gemini-3.5-flash",
            "",
            False,
            False,
            api_name="/chat",
        )
    finally:
        main.demo.close()

    rendered = result[0] if isinstance(result, tuple) else str(result)
    assert "Using `retrieve_tutor_context`" in rendered
    assert "Using `run_kb_command`" in rendered
    assert "rg generate_next_queries_tool raw" in rendered
    assert "Lesson 18: Research Loop" in rendered
    assert "Agentic AI Engineering" in rendered


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GRADIO_CURL_E2E") != "1",
    reason="Set RUN_LIVE_GRADIO_CURL_E2E=1 to run the live model-backed Gradio curl smoke test.",
)
def test_live_gradio_curl_can_use_retrieval_and_shell() -> None:
    if os.getenv("RUN_LIVE_GRADIO_CURL_E2E") != "1":
        pytest.skip("Set RUN_LIVE_GRADIO_CURL_E2E=1 to run this test.")
    import scripts.main as main

    import json
    import shutil
    import subprocess

    if shutil.which("curl") is None:
        pytest.skip("curl is required for the live Gradio curl smoke test.")
    if not os.getenv("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY is required for live retrieval fallback.")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY is required for the live model.")
    if not os.path.exists("data/kb/wiki/index.md"):
        pytest.skip("data/kb artifacts must exist before running the live smoke test.")

    port = free_port()
    main.demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        quiet=True,
    )
    try:
        post = subprocess.run(
            [
                "curl",
                "-s",
                f"http://127.0.0.1:{port}/gradio_api/call/chat",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(
                    {
                        "data": [
                            "Use both retrieve_tutor_context and run_kb_command to answer: how does PEFT configure LoRA with LoraConfig?",
                            [],
                            ["PEFT Docs", "Transformers Docs"],
                            os.getenv(
                                "LIVE_GRADIO_E2E_MODEL", "google-genai:gemini-3.5-flash"
                            ),
                            "",
                            False,
                            False,
                        ]
                    }
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        event_id = json.loads(post.stdout)["event_id"]
        stream = subprocess.run(
            [
                "curl",
                "-N",
                f"http://127.0.0.1:{port}/gradio_api/call/chat/{event_id}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    finally:
        main.demo.close()

    rendered = stream.stdout
    assert "Using `retrieve_tutor_context`" in rendered
    assert "Using `run_kb_command`" in rendered
    assert "LoRA" in rendered
    assert "PEFT" in rendered or "Transformers" in rendered
