from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from scripts.chat_service import (
    agent_run_config,
    build_agent,
    effective_tool_names,
    sync_thread_with_history,
)
from scripts.chat_types import ChatRequest, ChatTurn


class FakeAgent:
    def __init__(self, messages):
        self._messages = list(messages)
        self.updated_states: list[tuple[dict[str, object], dict[str, object]]] = []

    def get_state(self, _config):
        return types.SimpleNamespace(values={"messages": list(self._messages)})

    def update_state(self, config, payload):
        self.updated_states.append((config, payload))


class ChatServiceTestCase(unittest.TestCase):
    def test_effective_tool_names_follow_provider(self) -> None:
        self.assertEqual(
            effective_tool_names(
                "google-genai:gemini-3.5-flash",
                ("web_search", "url_context", "web_fetch"),
            ),
            ("retrieve_tutor_context", "google_search", "url_context"),
        )
        self.assertEqual(
            effective_tool_names(
                "anthropic:claude-haiku-4-5",
                ("web_search", "url_context", "web_fetch"),
            ),
            ("retrieve_tutor_context", "web_search", "web_fetch"),
        )

    def test_agent_run_config_adds_langsmith_metadata(self) -> None:
        request = ChatRequest(
            query="What is RAG?",
            source_keys=("langchain", "transformers"),
            model_name="google-genai:gemini-3.5-flash",
            include_reasoning=True,
            enabled_tools=("web_search",),
        )

        config = agent_run_config(request, "thread_123", "message_456")

        self.assertEqual(config["configurable"], {"thread_id": "thread_123"})
        self.assertEqual(config["run_name"], "ai-tutor-agent-turn")
        self.assertIn("provider:google-genai", config["tags"])
        self.assertIn("tool:retrieve_tutor_context", config["tags"])
        self.assertIn("tool:google_search", config["tags"])
        self.assertEqual(config["metadata"]["thread_id"], "thread_123")
        self.assertEqual(config["metadata"]["conversation_id"], "thread_123")
        self.assertEqual(config["metadata"]["message_id"], "message_456")
        self.assertEqual(
            config["metadata"]["available_tools"],
            ["retrieve_tutor_context", "google_search"],
        )
        self.assertEqual(
            config["metadata"]["source_keys"],
            ["langchain", "transformers"],
        )

    def test_build_agent_cache_keys_include_tool_toggles(self) -> None:
        build_agent.cache_clear()
        created_agents = []

        def fake_create_agent(**kwargs):
            agent = types.SimpleNamespace(kwargs=kwargs)
            created_agents.append(agent)
            return agent

        try:
            with (
                patch("scripts.chat_service.build_chat_model", return_value=object()),
                patch(
                    "scripts.chat_service.build_system_prompt", return_value="prompt"
                ),
                patch(
                    "scripts.chat_service.create_agent", side_effect=fake_create_agent
                ),
            ):
                with_web_tools = build_agent(
                    "google-genai:gemini-3.5-flash",
                    enabled_tools=("web_search", "url_context"),
                    include_thoughts=True,
                )
                without_web_tools = build_agent(
                    "google-genai:gemini-3.5-flash",
                    enabled_tools=(),
                    include_thoughts=True,
                )
                without_web_tools_again = build_agent(
                    "google-genai:gemini-3.5-flash",
                    enabled_tools=(),
                    include_thoughts=True,
                )
        finally:
            build_agent.cache_clear()

        self.assertIsNot(with_web_tools, without_web_tools)
        self.assertIs(without_web_tools, without_web_tools_again)
        self.assertEqual(len(created_agents), 2)

        enabled_tool_defs = created_agents[0].kwargs["tools"]
        disabled_tool_defs = created_agents[1].kwargs["tools"]
        self.assertIn({"google_search": {}}, enabled_tool_defs)
        self.assertIn({"url_context": {}}, enabled_tool_defs)
        self.assertEqual(disabled_tool_defs, [enabled_tool_defs[0]])
        self.assertEqual(len(created_agents[0].kwargs["middleware"]), 1)
        self.assertEqual(created_agents[1].kwargs["middleware"], [])

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
