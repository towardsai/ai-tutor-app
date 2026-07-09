from __future__ import annotations

import asyncio
import os
import types
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.runnables.fallbacks import RunnableWithFallbacks

from app.config import DEEPSEEK_DIRECT_MODEL_NAME, GEMINI_FALLBACK_MODEL_NAME
from app.chat_service import (
    THREAD_IDLE_TTL_SECONDS,
    _claim_kb_command_budget,
    _clear_kb_command_budget,
    _drop_thread_record,
    _evict_idle_threads,
    _get_fork_point,
    _get_thread_transcript,
    _record_fork_point,
    _record_thread_transcript,
    _touch_thread,
    agent_run_config,
    build_agent,
    build_chat_model,
    checkpoint_messages_to_history,
    effective_tool_names,
    extract_query_urls,
    extract_shell_source_matches,
    resolve_answer_citations,
    retrieve_tutor_context,
    sync_thread_with_history,
    stream_chat,
)
from app.chat_types import ChatRequest, ChatTurn, SourceMatch


class FakeAgent:
    def __init__(self, messages, checkpoint_id="ckpt_latest"):
        self._messages = list(messages)
        self.checkpoint_id = checkpoint_id
        self.updated_states: list[tuple[dict[str, object], dict[str, object]]] = []

    def get_state(self, _config):
        return types.SimpleNamespace(
            values={"messages": list(self._messages)},
            config={"configurable": {"checkpoint_id": self.checkpoint_id}},
        )

    def update_state(self, config, payload):
        self.updated_states.append((config, payload))
        return {
            "configurable": {"checkpoint_id": f"ckpt_seed_{len(self.updated_states)}"}
        }


class FakeStreamingAgent(FakeAgent):
    async def astream(self, *args, **kwargs):
        self.astream_configs = getattr(self, "astream_configs", [])
        self.astream_configs.append(args[1] if len(args) > 1 else kwargs.get("config"))
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
                {"langgraph_node": "model"},
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


class FakeAnswerAgent(FakeAgent):
    """Agent that streams nothing and answers with one final AI message."""

    def __init__(self, answer: str):
        super().__init__([])
        self.answer = answer

    async def astream(self, *args, **kwargs):
        yield {
            "type": "updates",
            "data": {"model": {"messages": [AIMessage(content=self.answer)]}},
        }


class FakeToolThenTextAgent(FakeAgent):
    """Streams a tool-call token first, then streamed answer text, so a turn
    sets first_token_at (the tool call) strictly before first_text_at (the
    answer) -- the case that distinguishes time_to_first_token_ms from ttft_ms."""

    async def astream(self, *args, **kwargs):
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_ls",
                            "name": "run_kb_command",
                            "args": {"command": "ls raw"},
                        }
                    ],
                ),
                {"langgraph_node": "model"},
            ),
        }
        yield {
            "type": "updates",
            "data": {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content="$ ls raw\nok",
                            name="run_kb_command",
                            tool_call_id="call_ls",
                        )
                    ]
                }
            },
        }
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(content="Here is the answer."),
                {"langgraph_node": "model"},
            ),
        }
        yield {
            "type": "updates",
            "data": {"model": {"messages": [AIMessage(content="Here is the answer.")]}},
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
                    "app.chat_service.build_chat_model",
                    # SummarizationMiddleware reads model._llm_type at init, so the
                    # stub needs that attribute (a bare object() would AttributeError).
                    return_value=types.SimpleNamespace(_llm_type="fake-chat-model"),
                ),
                patch("app.chat_service.build_system_prompt", return_value="prompt"),
                patch("app.chat_service.create_agent", side_effect=fake_create_agent),
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
        web_middleware = [
            type(m).__name__ for m in created_agents[0].kwargs["middleware"]
        ]
        plain_middleware = [
            type(m).__name__ for m in created_agents[1].kwargs["middleware"]
        ]
        self.assertIn("GeminiServerSideToolsMiddleware", web_middleware)
        self.assertNotIn("GeminiServerSideToolsMiddleware", plain_middleware)
        self.assertEqual(len(web_middleware), len(plain_middleware) + 1)
        self.assertGreater(len(plain_middleware), 0)

    def test_anthropic_reasoning_keeps_profile_output_budget(self) -> None:
        # Thinking shares the response's max_tokens; a hardcoded low cap made
        # reasoning mode (the frontend default) silently truncate long
        # answers mid-stream while the non-reasoning path got the model
        # profile's 64k. Both paths must use the same profile default.
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            plain = build_chat_model("anthropic:claude-haiku-4-5")
            reasoning = build_chat_model(
                "anthropic:claude-haiku-4-5", include_thoughts=True
            )

        self.assertEqual(reasoning.max_tokens, plain.max_tokens)
        self.assertGreater(reasoning.max_tokens, 8192)
        self.assertEqual(reasoning.thinking, {"type": "enabled", "budget_tokens": 2048})

    def test_deepseek_direct_default_has_gemini_fallback(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "deepseek-test-key",
                "GEMINI_API_KEY": "gemini-test-key",
            },
            clear=True,
        ):
            model = build_chat_model(DEEPSEEK_DIRECT_MODEL_NAME)

        self.assertIsInstance(model, RunnableWithFallbacks)
        self.assertEqual(model.runnable.model_name, "deepseek-v4-flash")
        self.assertEqual(str(model.runnable.openai_api_base), "https://api.deepseek.com")
        self.assertEqual(len(model.fallbacks), 1)
        self.assertEqual(
            model.fallbacks[0].model,
            GEMINI_FALLBACK_MODEL_NAME.partition(":")[2],
        )

    def test_deepseek_direct_default_skips_fallback_without_gemini_key(
        self,
    ) -> None:
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "deepseek-test-key"},
            clear=True,
        ):
            model = build_chat_model(DEEPSEEK_DIRECT_MODEL_NAME)

        self.assertNotIsInstance(model, RunnableWithFallbacks)
        self.assertEqual(model.model_name, "deepseek-v4-flash")

    def test_deepseek_direct_default_uses_gemini_when_deepseek_key_missing(
        self,
    ) -> None:
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "gemini-test-key"},
            clear=True,
        ):
            model = build_chat_model(DEEPSEEK_DIRECT_MODEL_NAME)

        self.assertNotIsInstance(model, RunnableWithFallbacks)
        self.assertEqual(model.model, GEMINI_FALLBACK_MODEL_NAME.partition(":")[2])

    def test_kb_command_budget_blocks_after_limit(self) -> None:
        session_id = "test_budget_session"
        _clear_kb_command_budget(session_id)
        try:
            self.assertEqual(_claim_kb_command_budget(session_id, 2), (True, 1))
            self.assertEqual(_claim_kb_command_budget(session_id, 2), (True, 2))
            self.assertEqual(_claim_kb_command_budget(session_id, 2), (False, 2))
        finally:
            _clear_kb_command_budget(session_id)

    def test_retrieve_tutor_context_degrades_on_retriever_failure(self) -> None:
        runtime = types.SimpleNamespace(
            context=types.SimpleNamespace(allowed_sources=("transformers",))
        )
        failing_retriever = MagicMock()
        failing_retriever.search.side_effect = RuntimeError("cohere 500 boom")
        with patch("app.chat_service.get_retriever", return_value=failing_retriever):
            result = retrieve_tutor_context.func(query="What is RAG?", runtime=runtime)
        # The turn survives with a soft fallback instead of a raised error...
        self.assertIn("temporarily unavailable", result)
        self.assertIn("run_kb_command", result)
        # ...and the raw provider error is not exposed in the tool output.
        self.assertNotIn("cohere 500 boom", result)

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

    def test_extract_shell_source_matches_reader_body_links_are_reference_only(
        self,
    ) -> None:
        # Regression for LangSmith trace 019f29f4-4e21-7a01-86b8-053161eb08d6:
        # cat'ing one wiki topic page put all 20 docs it links to into the tool
        # event, so the activity pill showed "20 sources" for a turn that read
        # 2 docs. Links in a read file's body stay citable evidence but must
        # not count as consulted sources.
        def fake_resolve(reference, **_kwargs):
            key = reference.rsplit("/", 1)[-1].removesuffix(".md")
            if "raw/" not in reference:
                return None
            return SourceMatch(
                doc_id=f"doc:{key}",
                title=key,
                url=f"https://example.com/{key}",
                source_key="peft",
                source_label="PEFT Docs",
                score=1.0,
                group="docs",
            )

        wiki_body = (
            "## Evaluation\n"
            "- [Lesson 29](raw/courses/agentic_ai_engineering/lesson-29.md)\n"
            "- [Best practices](raw/docs/openai_docs/evaluation-best-practices.md)\n"
        )
        with patch(
            "app.chat_service.resolve_manifest_reference", side_effect=fake_resolve
        ):
            wiki_cat = extract_shell_source_matches(
                "cat wiki/topics/evaluation.md", wiki_body
            )
            doc_cat = extract_shell_source_matches(
                "cat raw/courses/agentic_ai_engineering/lesson-29.md",
                "Lesson body links [elsewhere](raw/docs/peft/lora.md).",
            )
            doc_sed = extract_shell_source_matches(
                "sed -n '1,40p' raw/docs/peft/lora.md", "chunk of the lesson body"
            )

        # The wiki page itself is not a corpus doc; its linked docs are
        # evidence only.
        self.assertEqual(wiki_cat.browsed, [])
        self.assertEqual(
            [match.doc_id for match in wiki_cat.referenced],
            ["doc:lesson-29", "doc:evaluation-best-practices"],
        )
        # Reading a corpus doc (fully or partially) counts it as browsed; the
        # docs its body links to remain evidence only.
        self.assertEqual([match.doc_id for match in doc_cat.browsed], ["doc:lesson-29"])
        self.assertEqual(
            [match.doc_id for match in doc_cat.referenced],
            ["doc:lesson-29", "doc:lora"],
        )
        self.assertEqual([match.doc_id for match in doc_sed.browsed], ["doc:lora"])

    def test_extract_shell_source_matches_search_hits_count_as_browsed(self) -> None:
        def fake_resolve(reference, **_kwargs):
            if "raw/docs/peft/lora.md" not in reference:
                return None
            return SourceMatch(
                doc_id="peft:lora",
                title="LoRA",
                url="https://example.com/lora",
                source_key="peft",
                source_label="PEFT Docs",
                score=1.0,
                group="docs",
            )

        with patch(
            "app.chat_service.resolve_manifest_reference", side_effect=fake_resolve
        ):
            matches = extract_shell_source_matches(
                "rg -n LoraConfig raw/docs",
                "raw/docs/peft/lora.md:12:class LoraConfig:",
            )

        # A search hit printed the doc's own lines: the model read part of it.
        self.assertEqual([match.doc_id for match in matches.browsed], ["peft:lora"])
        self.assertEqual(matches.referenced, matches.browsed)

    def test_stream_chat_emits_time_to_first_token(self) -> None:
        agent = FakeToolThenTextAgent([])
        self.addCleanup(_drop_thread_record, "thread_ttft")
        request = ChatRequest(
            query="ls the kb",
            source_keys=("peft",),
            model_name="google-genai:gemini-3.5-flash",
            include_reasoning=False,
            enabled_tools=(),
        )

        async def collect_events():
            return [event async for event in stream_chat(request)]

        with (
            patch("app.chat_service.build_agent", return_value=agent),
            patch("app.chat_service.new_thread_id", return_value="thread_ttft"),
        ):
            events = asyncio.run(collect_events())

        stats = [event for event in events if event.type == "context_stats"]
        self.assertEqual(len(stats), 1)
        data = stats[0].data
        self.assertIn("time_to_first_token_ms", data)
        first_token = data["time_to_first_token_ms"]
        ttft = data["ttft_ms"]
        self.assertIsInstance(first_token, int)
        self.assertIsInstance(ttft, int)
        self.assertGreaterEqual(first_token, 0)
        # The tool-call token precedes the first answer text, so first-token
        # latency never exceeds time-to-first-answer.
        self.assertLessEqual(first_token, ttft)

    def test_stream_chat_resolves_shell_citation_after_final_answer(self) -> None:
        agent = FakeStreamingAgent([])
        self.addCleanup(_drop_thread_record, "thread_rg")
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
            patch("app.chat_service.build_agent", return_value=agent),
            patch("app.chat_service.new_thread_id", return_value="thread_rg"),
            patch(
                "app.chat_service.resolve_manifest_reference",
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
        source_matches = [event for event in events if event.type == "source_match"]

        self.assertEqual(started[0].data["args_text"], "rg LoraConfig raw")
        self.assertIn("rg LoraConfig raw", completed[0].data["output_text"])
        # The completion event carries its own evidence so the encoder can
        # populate output.matches (the tool row's source count in the UI).
        completed_matches = completed[0].data["matches"]
        self.assertEqual(len(completed_matches), 1)
        self.assertEqual(completed_matches[0]["doc_id"], "peft:lora")
        self.assertEqual(completed_matches[0]["call_id"], "call_rg")
        self.assertEqual(source_matches[0].data["source_key"], "peft")
        self.assertNotIn("call_id", source_matches[0].data)

    def test_stream_chat_disables_local_tools_when_no_sources_selected(self) -> None:
        # An explicit empty source selection (UI "Knowledge base: off") must
        # build the agent without retrieval/KB tools instead of silently
        # retrieving from the defaults.
        agent = FakeStreamingAgent([])
        build_agent_mock = MagicMock(return_value=agent)
        self.addCleanup(_drop_thread_record, "thread_no_sources")
        request = ChatRequest(
            query="What is RAG?",
            source_keys=(),
            model_name="google-genai:gemini-3.5-flash",
            include_reasoning=False,
            enabled_tools=(),
        )

        async def collect_events():
            return [event async for event in stream_chat(request)]

        with (
            patch("app.chat_service.build_agent", build_agent_mock),
            patch("app.chat_service.new_thread_id", return_value="thread_no_sources"),
        ):
            asyncio.run(collect_events())

        build_agent_mock.assert_called_once()
        self.assertFalse(build_agent_mock.call_args.kwargs["include_local_tools"])

        self.assertEqual(
            effective_tool_names(
                "google-genai:gemini-3.5-flash",
                ("web_search",),
                include_local_tools=False,
            ),
            ("google_search",),
        )

    def test_checkpoint_history_collapses_tool_using_turn(self) -> None:
        # A tool-using turn checkpoints as [Human, AI(tool_calls, empty text),
        # ToolMessage, AI(answer)]; the visible transcript has one assistant
        # turn, so the collapsed history must match that shape.
        messages = [
            HumanMessage(content="What is LoRA?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "retrieve_tutor_context",
                        "args": {"query": "LoRA"},
                    }
                ],
            ),
            ToolMessage(
                content="payload",
                name="retrieve_tutor_context",
                tool_call_id="call_1",
            ),
            AIMessage(content="LoRA is a parameter-efficient fine-tuning method."),
        ]

        self.assertEqual(
            checkpoint_messages_to_history(messages),
            (
                ChatTurn(role="user", content="What is LoRA?"),
                ChatTurn(
                    role="assistant",
                    content="LoRA is a parameter-efficient fine-tuning method.",
                ),
            ),
        )

    def test_checkpoint_history_merges_preamble_like_ui_text_parts(self) -> None:
        # The AI SDK UIMessage joins its text parts with no separator, so a
        # preamble streamed before the tool call must concatenate raw (inner
        # whitespace preserved, only the merged turn trimmed).
        messages = [
            HumanMessage(content="What is LoRA?"),
            AIMessage(
                content="Let me check the docs.\n\n",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "retrieve_tutor_context",
                        "args": {"query": "LoRA"},
                    }
                ],
            ),
            ToolMessage(
                content="payload",
                name="retrieve_tutor_context",
                tool_call_id="call_1",
            ),
            AIMessage(content="LoRA is a fine-tuning method."),
        ]

        self.assertEqual(
            checkpoint_messages_to_history(messages),
            (
                ChatTurn(role="user", content="What is LoRA?"),
                ChatTurn(
                    role="assistant",
                    content="Let me check the docs.\n\nLoRA is a fine-tuning method.",
                ),
            ),
        )

    def test_tool_using_turn_reuses_checkpointed_thread(self) -> None:
        agent = FakeAgent(
            [
                HumanMessage(content="What is LoRA?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "retrieve_tutor_context",
                            "args": {"query": "LoRA"},
                        }
                    ],
                ),
                ToolMessage(
                    content="payload",
                    name="retrieve_tutor_context",
                    tool_call_id="call_1",
                ),
                AIMessage(content="LoRA is a fine-tuning method."),
            ]
        )
        history = (
            ChatTurn(role="user", content="What is LoRA?"),
            ChatTurn(role="assistant", content="LoRA is a fine-tuning method."),
        )

        active_thread_id, fork_checkpoint_id = sync_thread_with_history(
            agent, "thread_0", history
        )

        self.assertEqual(active_thread_id, "thread_0")
        self.assertEqual(agent.updated_states, [])

    def test_summarized_thread_reuses_tracked_transcript(self) -> None:
        # After SummarizationMiddleware rewrites thread state, the checkpoint
        # transcript (summary-as-user-turn + tail) can never equal the visible
        # history again; the tracked transcript must keep the thread alive.
        agent = FakeAgent(
            [
                HumanMessage(content="Here is a summary of the conversation so far."),
                AIMessage(content="a2"),
            ]
        )
        history = (
            ChatTurn(role="user", content="q1"),
            ChatTurn(role="assistant", content="a1"),
            ChatTurn(role="user", content="q2"),
            ChatTurn(role="assistant", content="a2"),
        )
        self.addCleanup(_drop_thread_record, "thread_sum")
        _record_thread_transcript("thread_sum", history)

        active_thread_id, fork_checkpoint_id = sync_thread_with_history(
            agent, "thread_sum", history
        )

        self.assertEqual(active_thread_id, "thread_sum")
        self.assertEqual(agent.updated_states, [])

    def test_edit_branches_despite_tracked_transcript_and_drops_it(self) -> None:
        agent = FakeAgent(
            [
                HumanMessage(content="q1"),
                AIMessage(content="a1"),
            ]
        )
        self.addCleanup(_drop_thread_record, "thread_0")
        _record_thread_transcript(
            "thread_0",
            (
                ChatTurn(role="user", content="q1"),
                ChatTurn(role="assistant", content="a1"),
            ),
        )
        edited_history = (ChatTurn(role="user", content="q1 edited"),)

        with (
            patch("app.chat_service.new_thread_id", return_value="thread_fork"),
            patch("app.chat_service.CHECKPOINTER") as checkpointer,
        ):
            active_thread_id, fork_checkpoint_id = sync_thread_with_history(
                agent,
                "thread_0",
                edited_history,
            )

        self.assertEqual(active_thread_id, "thread_fork")
        checkpointer.delete_thread.assert_called_once_with("thread_0")
        self.assertIsNone(_get_thread_transcript("thread_0"))

    def test_edit_forks_thread_from_recorded_checkpoint(self) -> None:
        # Editing q2 sends history = transcript[:2]; the thread must fork
        # from the end-of-turn-1 checkpoint instead of branching to a new
        # thread, and fork points past the fork must be pruned.
        agent = FakeAgent([])
        tracked = (
            ChatTurn(role="user", content="q1"),
            ChatTurn(role="assistant", content="a1"),
            ChatTurn(role="user", content="q2"),
            ChatTurn(role="assistant", content="a2"),
        )
        self.addCleanup(_drop_thread_record, "thread_tt")
        _record_thread_transcript("thread_tt", tracked)
        _record_fork_point("thread_tt", 2, "ckpt_turn_1")
        _record_fork_point("thread_tt", 4, "ckpt_turn_2")

        active_thread_id, fork_checkpoint_id = sync_thread_with_history(
            agent,
            "thread_tt",
            tracked[:2],
        )

        self.assertEqual(active_thread_id, "thread_tt")
        self.assertEqual(fork_checkpoint_id, "ckpt_turn_1")
        self.assertEqual(agent.updated_states, [])
        self.assertEqual(_get_fork_point("thread_tt", 2), "ckpt_turn_1")
        self.assertEqual(_get_fork_point("thread_tt", 4), "")

    def test_edit_without_fork_point_falls_back_to_branch(self) -> None:
        agent = FakeAgent([])
        tracked = (
            ChatTurn(role="user", content="q1"),
            ChatTurn(role="assistant", content="a1"),
            ChatTurn(role="user", content="q2"),
            ChatTurn(role="assistant", content="a2"),
        )
        self.addCleanup(_drop_thread_record, "thread_nofp")
        self.addCleanup(_drop_thread_record, "thread_plain")
        _record_thread_transcript("thread_nofp", tracked)

        with (
            patch("app.chat_service.new_thread_id", return_value="thread_plain"),
            patch("app.chat_service.CHECKPOINTER") as checkpointer,
        ):
            active_thread_id, fork_checkpoint_id = sync_thread_with_history(
                agent,
                "thread_nofp",
                tracked[:2],
            )

        self.assertEqual(active_thread_id, "thread_plain")
        self.assertEqual(fork_checkpoint_id, "")
        self.assertEqual(len(agent.updated_states), 1)
        checkpointer.delete_thread.assert_called_once_with("thread_nofp")
        self.assertIsNone(_get_thread_transcript("thread_nofp"))

    def test_stream_chat_forks_from_checkpoint_on_edit(self) -> None:
        agent = FakeStreamingAgent([])
        tracked = (
            ChatTurn(role="user", content="q1"),
            ChatTurn(role="assistant", content="a1"),
            ChatTurn(role="user", content="q2"),
            ChatTurn(role="assistant", content="a2"),
        )
        self.addCleanup(_drop_thread_record, "thread_live")
        _record_thread_transcript("thread_live", tracked)
        _record_fork_point("thread_live", 2, "ckpt_turn_1")
        request = ChatRequest(
            query="q2 edited",
            history=tracked[:2],
            source_keys=("peft",),
            model_name="google-genai:gemini-3.5-flash",
            include_reasoning=False,
            thread_id="thread_live",
            enabled_tools=(),
        )

        async def collect_events():
            return [event async for event in stream_chat(request)]

        with (
            patch("app.chat_service.build_agent", return_value=agent),
            patch("app.chat_service.resolve_manifest_reference", return_value=None),
        ):
            events = asyncio.run(collect_events())

        run_config = agent.astream_configs[0]
        self.assertEqual(run_config["configurable"]["thread_id"], "thread_live")
        self.assertEqual(run_config["configurable"]["checkpoint_id"], "ckpt_turn_1")
        thread_started = next(e for e in events if e.type == "thread_started")
        self.assertEqual(thread_started.data["thread_id"], "thread_live")
        self.assertEqual(
            _get_thread_transcript("thread_live"),
            tracked[:2]
            + (
                ChatTurn(role="user", content="q2 edited"),
                ChatTurn(
                    role="assistant",
                    content=(
                        "LoraConfig is documented in LoRA "
                        "[LoRA](raw/docs/peft/lora.md)."
                    ),
                ),
            ),
        )
        # The new branch records its own fork point at the new turn boundary.
        self.assertEqual(_get_fork_point("thread_live", 4), "ckpt_latest")

    def test_summarization_rewrite_does_not_reemit_tool_completion(self) -> None:
        # SummarizationMiddleware.before_model replaces thread state with
        # [summary, *preserved]; when that fires mid-turn the preserved tail
        # ends in the already-reported ToolMessage. Only the tools node may
        # report completions.
        tool_message = ToolMessage(
            content="$ rg LoraConfig raw\nstdout:\nraw/docs/peft/lora.md:3",
            name="run_kb_command",
            tool_call_id="call_rg",
        )

        class FakeSummarizingAgent(FakeAgent):
            async def astream(self, *_args, **_kwargs):
                yield {
                    "type": "updates",
                    "data": {"tools": {"messages": [tool_message]}},
                }
                yield {
                    "type": "updates",
                    "data": {
                        "SummarizationMiddleware.before_model": {
                            "messages": [
                                HumanMessage(content="Summary of the conversation."),
                                tool_message,
                            ]
                        }
                    },
                }
                yield {
                    "type": "updates",
                    "data": {"model": {"messages": [AIMessage(content="answer")]}},
                }

        agent = FakeSummarizingAgent([])
        request = ChatRequest(
            query="long tool-heavy question",
            source_keys=("peft",),
            model_name="google-genai:gemini-3.5-flash",
            include_reasoning=False,
            enabled_tools=(),
        )
        self.addCleanup(_drop_thread_record, "thread_sumdup")

        async def collect_events():
            return [event async for event in stream_chat(request)]

        with (
            patch("app.chat_service.build_agent", return_value=agent),
            patch("app.chat_service.new_thread_id", return_value="thread_sumdup"),
            patch("app.chat_service.resolve_manifest_reference", return_value=None),
        ):
            events = asyncio.run(collect_events())

        completed = [e for e in events if e.type == "tool_call_completed"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].data["call_id"], "call_rg")
        self.assertEqual(completed[0].data["step"], "tools")

    def test_stream_chat_records_thread_transcript(self) -> None:
        agent = FakeStreamingAgent([])
        request = ChatRequest(
            query="Use rg to find LoraConfig",
            source_keys=("peft",),
            model_name="google-genai:gemini-3.5-flash",
            include_reasoning=False,
            enabled_tools=(),
        )
        self.addCleanup(_drop_thread_record, "thread_transcript")

        async def collect_events():
            return [event async for event in stream_chat(request)]

        with (
            patch("app.chat_service.build_agent", return_value=agent),
            patch("app.chat_service.new_thread_id", return_value="thread_transcript"),
            patch("app.chat_service.resolve_manifest_reference", return_value=None),
        ):
            asyncio.run(collect_events())

        self.assertEqual(
            _get_thread_transcript("thread_transcript"),
            (
                ChatTurn(role="user", content="Use rg to find LoraConfig"),
                ChatTurn(
                    role="assistant",
                    content=(
                        "LoraConfig is documented in LoRA "
                        "[LoRA](raw/docs/peft/lora.md)."
                    ),
                ),
            ),
        )

    def test_build_agent_cache_busts_when_kb_instructions_appear(self) -> None:
        build_agent.cache_clear()
        created_agents = []

        def fake_create_agent(**kwargs):
            agent = types.SimpleNamespace(kwargs=kwargs)
            created_agents.append(agent)
            return agent

        try:
            with (
                patch(
                    "app.chat_service.build_chat_model",
                    return_value=types.SimpleNamespace(_llm_type="fake-chat-model"),
                ),
                patch("app.chat_service.create_agent", side_effect=fake_create_agent),
            ):
                degraded = build_agent(
                    "google-genai:gemini-3.5-flash",
                    enabled_tools=(),
                    include_thoughts=False,
                    kb_agents_instructions="",
                )
                healthy = build_agent(
                    "google-genai:gemini-3.5-flash",
                    enabled_tools=(),
                    include_thoughts=False,
                    kb_agents_instructions="# KB rules marker",
                )
                healthy_again = build_agent(
                    "google-genai:gemini-3.5-flash",
                    enabled_tools=(),
                    include_thoughts=False,
                    kb_agents_instructions="# KB rules marker",
                )
        finally:
            build_agent.cache_clear()

        # A degraded (no KB instructions) build must not pin the cache entry.
        self.assertIsNot(degraded, healthy)
        self.assertIs(healthy, healthy_again)
        self.assertNotIn(
            "## Local KB Instructions", created_agents[0].kwargs["system_prompt"]
        )
        self.assertIn("# KB rules marker", created_agents[1].kwargs["system_prompt"])

    def test_stream_chat_passes_ensured_kb_instructions_to_build_agent(self) -> None:
        agent = FakeAnswerAgent("Answer.")
        request = ChatRequest(
            query="hello",
            source_keys=("peft",),
            model_name="google-genai:gemini-3.5-flash",
            include_reasoning=False,
            enabled_tools=(),
        )
        self.addCleanup(_drop_thread_record, "thread_kbwire")

        async def collect_events():
            return [event async for event in stream_chat(request)]

        with (
            patch(
                "app.chat_service.build_agent", return_value=agent
            ) as build_agent_mock,
            patch(
                "app.chat_service.ensure_kb_agents_instructions",
                return_value="# Ensured rules",
            ),
            patch("app.chat_service.new_thread_id", return_value="thread_kbwire"),
        ):
            asyncio.run(collect_events())

        self.assertEqual(
            build_agent_mock.call_args.kwargs["kb_agents_instructions"],
            "# Ensured rules",
        )

    def test_extract_query_urls_dedupes_and_trims_punctuation(self) -> None:
        self.assertEqual(
            extract_query_urls(
                "Read https://example.com/post, then https://example.com/post "
                "and (https://other.example/page)."
            ),
            ["https://example.com/post", "https://other.example/page"],
        )
        self.assertEqual(extract_query_urls("no links here"), [])

    def run_answer_stream(
        self,
        answer: str,
        *,
        query: str,
        enabled_tools: tuple[str, ...],
        thread_id: str,
    ) -> list:
        agent = FakeAnswerAgent(answer)
        request = ChatRequest(
            query=query,
            source_keys=("peft",),
            model_name="google-genai:gemini-3.5-flash",
            include_reasoning=False,
            enabled_tools=enabled_tools,
        )
        self.addCleanup(_drop_thread_record, thread_id)

        async def collect_events():
            return [event async for event in stream_chat(request)]

        with (
            patch("app.chat_service.build_agent", return_value=agent),
            patch("app.chat_service.new_thread_id", return_value=thread_id),
            patch("app.chat_service.resolve_manifest_reference", return_value=None),
        ):
            events = asyncio.run(collect_events())
        return [event for event in events if event.type == "source_match"]

    def test_cited_pasted_url_resolves_via_url_context_evidence(self) -> None:
        sources = self.run_answer_stream(
            "Summary. See [the post](https://example.com/post).",
            query="Read https://example.com/post and summarize it",
            enabled_tools=("url_context",),
            thread_id="thread_urlctx",
        )

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].data["source_key"], "url_context")
        self.assertEqual(sources[0].data["url"], "https://example.com/post")
        self.assertEqual(sources[0].data["group"], "web")

    def test_unmatched_web_citation_kept_only_for_url_context(self) -> None:
        # Only url_context fetches invisibly; the other web tools report
        # their results into evidence, so an unmatched citation there still
        # means a memory link and stays gated.
        answer = "From [some site](https://unsourced.example/page)."

        kept = self.run_answer_stream(
            answer,
            query="What does that site say?",
            enabled_tools=("url_context",),
            thread_id="thread_webkeep",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].data["source_key"], "web")
        self.assertEqual(kept[0].data["group"], "web")

        for enabled_tools, thread_id in (
            (("web_search",), "thread_webdrop_search"),
            ((), "thread_webdrop_none"),
        ):
            dropped = self.run_answer_stream(
                answer,
                query="What does that site say?",
                enabled_tools=enabled_tools,
                thread_id=thread_id,
            )
            self.assertEqual(dropped, [])

    def test_pasted_url_surfaces_in_fallback_when_nothing_cited(self) -> None:
        sources = self.run_answer_stream(
            "Here is a summary with no inline citations.",
            query="Summarize https://example.com/post please",
            enabled_tools=("url_context",),
            thread_id="thread_urlfallback",
        )

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].data["source_key"], "url_context")
        self.assertEqual(sources[0].data["url"], "https://example.com/post")

    def test_idle_threads_evicted_with_checkpoints_and_records(self) -> None:
        self.addCleanup(_drop_thread_record, "thread_stale")
        self.addCleanup(_drop_thread_record, "thread_fresh")
        stale_history = (
            ChatTurn(role="user", content="q"),
            ChatTurn(role="assistant", content="a"),
        )
        _record_thread_transcript("thread_stale", stale_history)
        _record_fork_point("thread_stale", 2, "ckpt_stale")
        _touch_thread("thread_stale", now=0.0)
        _record_thread_transcript("thread_fresh", stale_history)
        _touch_thread("thread_fresh", now=1000.0)

        with patch("app.chat_service.CHECKPOINTER") as checkpointer:
            evicted = _evict_idle_threads(now=THREAD_IDLE_TTL_SECONDS + 1.0, force=True)

        self.assertEqual(evicted, ["thread_stale"])
        checkpointer.delete_thread.assert_called_once_with("thread_stale")
        self.assertIsNone(_get_thread_transcript("thread_stale"))
        self.assertEqual(_get_fork_point("thread_stale", 2), "")
        # The fresh thread was used within the TTL and survives untouched.
        self.assertEqual(_get_thread_transcript("thread_fresh"), stale_history)

    def test_branching_deletes_superseded_thread(self) -> None:
        agent = FakeAgent(
            [
                HumanMessage(content="How do I create an agent?"),
                AIMessage(content="Use a model and tools."),
            ]
        )
        edited_history = (
            ChatTurn(role="user", content="How do I create a RAG agent?"),
        )
        self.addCleanup(_drop_thread_record, "thread_edit")

        with (
            patch("app.chat_service.new_thread_id", return_value="thread_edit"),
            patch("app.chat_service.CHECKPOINTER") as checkpointer,
        ):
            active_thread_id, fork_checkpoint_id = sync_thread_with_history(
                agent,
                "thread_0",
                edited_history,
            )

        self.assertEqual(active_thread_id, "thread_edit")
        self.assertEqual(fork_checkpoint_id, "")
        checkpointer.delete_thread.assert_called_once_with("thread_0")

    def test_shorter_history_branches_to_fresh_thread(self) -> None:
        agent = FakeAgent(
            [
                HumanMessage(content="How do I create an agent?"),
                AIMessage(content="Use a model and tools."),
            ]
        )

        with patch("app.chat_service.new_thread_id", return_value="thread_regen"):
            active_thread_id, fork_checkpoint_id = sync_thread_with_history(
                agent, "thread_0", ()
            )

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
        self.addCleanup(_drop_thread_record, "thread_edit")

        with patch("app.chat_service.new_thread_id", return_value="thread_edit"):
            active_thread_id, fork_checkpoint_id = sync_thread_with_history(
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
