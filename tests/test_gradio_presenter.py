from __future__ import annotations

import unittest

from scripts.chat_types import ChatEvent
from scripts.gradio_presenter import (
    GradioPresenterState,
    SOURCES_HEADER,
    summarize_tool_result,
)


class GradioPresenterTestCase(unittest.TestCase):
    def test_sources_render_only_after_message_completion(self) -> None:
        presenter = GradioPresenterState(show_activity=True)

        presenter.apply(ChatEvent("thread_started", {"thread_id": "thread_1"}))
        presenter.apply(ChatEvent("text_delta", {"message_id": "m1", "text": "Answer"}))
        presenter.apply(
            ChatEvent(
                "source_match",
                {
                    "message_id": "m1",
                    "call_id": "c1",
                    "doc_id": "doc_1",
                    "title": "RAG overview",
                    "url": "https://example.com/rag",
                    "source_label": "LangChain Docs",
                    "score": 0.92,
                },
            )
        )

        in_progress = presenter.render()
        self.assertIn("Answer", in_progress)
        self.assertNotIn(SOURCES_HEADER, in_progress)

        presenter.apply(
            ChatEvent(
                "message_completed",
                {
                    "message_id": "m1",
                    "thread_id": "thread_1",
                    "answer": "Answer",
                },
            )
        )

        completed = presenter.render()
        self.assertIn(SOURCES_HEADER, completed)
        self.assertIn("RAG overview", completed)

    def test_retrieval_summary_uses_tool_payload_matches(self) -> None:
        event = ChatEvent(
            "tool_call_completed",
            {
                "tool_name": "retrieve_tutor_context",
                "output_text": (
                    '{"matches": [{"title": "LoRA", "source_label": "PEFT Docs"}]}'
                ),
            },
        )

        self.assertEqual(
            summarize_tool_result(event),
            "_Found 1 match from PEFT Docs._",
        )

    def test_run_kb_command_summary_confirms_command_execution(self) -> None:
        event = ChatEvent(
            "tool_call_completed",
            {
                "tool_name": "run_kb_command",
                "output_text": (
                    "$ rg LoraConfig raw\n"
                    "cwd: data/kb\n"
                    "exit_code: 0\n"
                    "stdout:\n"
                    "raw/docs/peft/lora.md:3:LoraConfig"
                ),
            },
        )

        self.assertEqual(
            summarize_tool_result(event),
            "_Ran `rg LoraConfig raw` with exit code 0._",
        )


if __name__ == "__main__":
    unittest.main()
