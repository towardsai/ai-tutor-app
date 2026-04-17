from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from scripts.chat_service import sync_thread_with_history
from scripts.chat_types import ChatTurn


class FakeAgent:
    def __init__(self, messages):
        self._messages = list(messages)
        self.updated_states: list[tuple[dict[str, object], dict[str, object]]] = []

    def get_state(self, _config):
        return types.SimpleNamespace(values={"messages": list(self._messages)})

    def update_state(self, config, payload):
        self.updated_states.append((config, payload))


class ChatServiceTestCase(unittest.TestCase):
    def test_shorter_history_branches_to_fresh_thread(self) -> None:
        agent = FakeAgent(
            [
                HumanMessage(content="How do I create an agent?"),
                AIMessage(content="Use a model and tools."),
            ]
        )

        with patch("scripts.chat_service.new_thread_id", return_value="thread_regen"):
            active_thread_id = sync_thread_with_history(agent, "thread_0", ())

        self.assertEqual(active_thread_id, "thread_regen")
        self.assertEqual(agent.updated_states, [])

    def test_edited_history_restores_messages_into_branched_thread(self) -> None:
        agent = FakeAgent(
            [
                HumanMessage(content="How do I create an agent?"),
                AIMessage(content="Use a model and tools."),
            ]
        )
        edited_history = (
            ChatTurn(role="user", content="How do I create a RAG agent?"),
            ChatTurn(role="assistant", content="Use retrieval and a model."),
        )

        with patch("scripts.chat_service.new_thread_id", return_value="thread_edit"):
            active_thread_id = sync_thread_with_history(
                agent,
                "thread_0",
                edited_history,
            )

        self.assertEqual(active_thread_id, "thread_edit")
        self.assertEqual(len(agent.updated_states), 1)
        config, payload = agent.updated_states[0]
        self.assertEqual(config, {"configurable": {"thread_id": "thread_edit"}})
        restored_messages = payload["messages"]
        self.assertEqual(len(restored_messages), 2)
        self.assertEqual(restored_messages[0].content, "How do I create a RAG agent?")
        self.assertEqual(restored_messages[1].content, "Use retrieval and a model.")


if __name__ == "__main__":
    unittest.main()
