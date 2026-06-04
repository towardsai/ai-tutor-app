from __future__ import annotations

import asyncio
import types
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from scripts.chat_service import (
    _claim_kb_command_budget,
    _clear_kb_command_budget,
    agent_run_config,
    build_agent,
    effective_tool_names,
    resolve_answer_citations,
    sync_thread_with_history,
    stream_chat,
)
from scripts.chat_types import ChatRequest, ChatTurn, SourceMatch


class FakeAgent:
    def __init__(self, messages):
        self._messages = list(messages)
        self.updated_states: list[tuple[dict[str, object], dict[str, object]]] = []

    def get_state(self, _config):
        return types.SimpleNamespace(values={"messages": list(self._messages)})

    def update_state(self, config, payload):
        self.updated_states.append((config, payload))


class FakeStreamingAgent(FakeAgent):
    async def astream(self, *_args, **_kwargs):
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_rg",
                            "name": "run_kb_command",
                            "args": {"command": "rg LoraConfig raw"},
                        }
                    ],
                ),
                {"langgraph_step": "model"},
            ),
        }
        yield {
            "type": "updates",
            "data": {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content=(
                                "$ rg LoraConfig raw\n"
                                "cwd: /tmp/kb\n"
                                "exit_code: 0\n"
                                "stdout:\n"
                                "raw/docs/peft/lora.md:3:LoraConfig"
                            ),
                            name="run_kb_command",
                            tool_call_id="call_rg",
                        )
                    ]
                }
            },
        }
        yield {
            "type": "updates",
            "data": {
                "model": {
                    "messages": [
                        AIMessage(
                            content=(
                                "LoraConfig is documented in LoRA "
                                "[LoRA](raw/docs/peft/lora.md)."
                            )
                        )
                    ]
                }
            },
        }


class ChatServiceTestCase(unittest.TestCase):
    def test_effective_tool_names_follow_provider(self) -> None:
        self.assertEqual(
            effective_tool_names(
                "google-genai:gemini-3.5-flash",
                ("web_search", "url_context", "web_fetch"),
            ),
            (
                "retrieve_tutor_context",
                "run_kb_command",
                "google_search",
                "url_context",
            ),
        )
        self.assertEqual(
            effective_tool_names(
                "anthropic:claude-haiku-4-5",
                ("web_search", "url_context", "web_fetch"),
            ),
            (
                "retrieve_tutor_context",
                "run_kb_command",
                "web_search",
                "web_fetch",
            ),
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
        self.assertIn("tool:run_kb_command", config["tags"])
        self.assertIn("tool:google_search", config["tags"])
        self.assertEqual(config["metadata"]["thread_id"], "thread_123")
        self.assertEqual(config["metadata"]["conversation_id"], "thread_123")
        self.assertEqual(config["metadata"]["message_id"], "message_456")
        self.assertEqual(
            config["metadata"]["available_tools"],
            [
                "retrieve_tutor_context",
                "run_kb_command",
                "google_search",
            ],
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
                patch(
                    "scripts.chat_service.build_chat_model",
                    # SummarizationMiddleware reads model._llm_type at init, so the
                    # stub needs that attribute (a bare object() would AttributeError).
                    return_value=types.SimpleNamespace(_llm_type="fake-chat-model"),
                ),
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
        self.assertEqual(len(disabled_tool_defs), 2)
        disabled_tool_names = {tool.name for tool in disabled_tool_defs}
        self.assertEqual(
            disabled_tool_names,
            {
                "retrieve_tutor_context",
                "run_kb_command",
            },
        )
        # Enabling Gemini web tools adds exactly one toggle-specific middleware
        # (GeminiServerSideToolsMiddleware) on top of the shared base middlewares.
        web_middleware = [type(m).__name__ for m in created_agents[0].kwargs["middleware"]]
        plain_middleware = [type(m).__name__ for m in created_agents[1].kwargs["middleware"]]
        self.assertIn("GeminiServerSideToolsMiddleware", web_middleware)
        self.assertNotIn("GeminiServerSideToolsMiddleware", plain_middleware)
        self.assertEqual(len(web_middleware), len(plain_middleware) + 1)
        self.assertGreater(len(plain_middleware), 0)

    def test_kb_command_budget_blocks_after_limit(self) -> None:
        session_id = "test_budget_session"
        _clear_kb_command_budget(session_id)
        try:
            self.assertEqual(_claim_kb_command_budget(session_id, 2), (True, 1))
            self.assertEqual(_claim_kb_command_budget(session_id, 2), (True, 2))
            self.assertEqual(_claim_kb_command_budget(session_id, 2), (False, 2))
        finally:
            _clear_kb_command_budget(session_id)

    def test_resolve_answer_citations_uses_current_turn_evidence(self) -> None:
        retrieval = SourceMatch(
            doc_id="peft:lora",
            title="LoRA",
            url="https://example.com/lora",
            source_key="peft",
            source_label="PEFT Docs",
            score=12.0,
            group="docs",
        )

        resolved = resolve_answer_citations(
            "See [LoRA](https://example.com/lora) and [Other](https://example.com/other).",
            retrieval_evidence={"peft:lora": retrieval},
            shell_evidence={},
            web_evidence={},
        )

        self.assertEqual(resolved, [retrieval])

    def test_resolve_answer_citations_ignores_unseen_kb_paths(self) -> None:
        resolved = resolve_answer_citations(
            "See [LoRA](raw/docs/peft/lora.md).",
            retrieval_evidence={},
            shell_evidence={},
            web_evidence={},
        )

        self.assertEqual(resolved, [])

    def test_resolve_answer_citations_dedupes_repeated_citation(self) -> None:
        retrieval = SourceMatch(
            doc_id="peft:lora",
            title="LoRA",
            url="https://example.com/lora",
            source_key="peft",
            source_label="PEFT Docs",
            score=12.0,
            group="docs",
        )

        resolved = resolve_answer_citations(
            "See [LoRA](https://example.com/lora). More on [LoRA](https://example.com/lora).",
            retrieval_evidence={"peft:lora": retrieval},
            shell_evidence={},
            web_evidence={},
        )

        self.assertEqual(resolved, [retrieval])

    def test_resolve_answer_citations_resolves_cited_web_source(self) -> None:
        web = SourceMatch(
            doc_id="web_search::https://example.com/post",
            title="A blog post",
            url="https://example.com/post",
            source_key="web_search",
            source_label="Web",
            score=1.0,
            group="web",
        )

        resolved = resolve_answer_citations(
            "As noted in [the post](https://example.com/post).",
            retrieval_evidence={},
            shell_evidence={},
            web_evidence={"web_search::https://example.com/post": web},
        )

        self.assertEqual(resolved, [web])

    def test_resolve_answer_citations_keep_unresolved_sources_flag(self) -> None:
        answer = "From [some site](https://unsourced.example/page)."

        gated = resolve_answer_citations(
            answer,
            retrieval_evidence={},
            shell_evidence={},
            web_evidence={},
        )
        self.assertEqual(gated, [])

        kept = resolve_answer_citations(
            answer,
            retrieval_evidence={},
            shell_evidence={},
            web_evidence={},
            keep_unresolved_sources=True,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].url, "https://unsourced.example/page")
        self.assertEqual(kept[0].group, "web")

    def test_stream_chat_resolves_shell_citation_after_final_answer(self) -> None:
        agent = FakeStreamingAgent([])
        request = ChatRequest(
            query="Use rg to find LoraConfig",
            source_keys=("peft",),
            model_name="google-genai:gemini-3.5-flash",
            include_reasoning=False,
            enabled_tools=(),
        )

        async def collect_events():
            return [event async for event in stream_chat(request)]

        shell_match = SourceMatch(
            doc_id="peft:lora",
            title="LoRA",
            url="https://example.com/lora",
            source_key="peft",
            source_label="PEFT Docs",
            score=1.0,
            group="docs",
        )

        def fake_resolve_manifest_reference(reference, **_kwargs):
            return shell_match if "raw/docs/peft/lora.md" in reference else None

        with (
            patch("scripts.chat_service.build_agent", return_value=agent),
            patch("scripts.chat_service.new_thread_id", return_value="thread_rg"),
            patch(
                "scripts.chat_service.resolve_manifest_reference",
                side_effect=fake_resolve_manifest_reference,
            ),
        ):
            events = asyncio.run(collect_events())

        started = [
            event
            for event in events
            if event.type == "tool_call_started"
            and event.data.get("tool_name") == "run_kb_command"
        ]
        completed = [
            event
            for event in events
            if event.type == "tool_call_completed"
            and event.data.get("tool_name") == "run_kb_command"
        ]
        source_matches = [
            event
            for event in events
            if event.type == "source_match"
        ]

        self.assertEqual(started[0].data["args_text"], "rg LoraConfig raw")
        self.assertIn("rg LoraConfig raw", completed[0].data["output_text"])
        self.assertEqual(source_matches[0].data["source_key"], "peft")
        self.assertNotIn("call_id", source_matches[0].data)

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
