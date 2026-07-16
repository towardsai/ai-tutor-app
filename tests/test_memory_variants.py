"""Tests for the Part C memory/context variants and their wiring.

Covers the per-call-view middlewares (sliding window, observation truncation,
prompt compression), preset resolution + middleware assembly, the generalized
compaction gate, and the per-request retrieval-budget override. All offline:
no model client, no API keys, no vector DB.
"""

from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace
from unittest import mock

import tiktoken
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from app.chat_service import (
    AppContext,
    DeepSeekCacheIsolationMiddleware,
    InContextHistoryRetrievalMiddleware,
    InstrumentedSummarizationMiddleware,
    ObservationTruncationMiddleware,
    PromptCompressionMiddleware,
    PrefixPreservingCompactionMiddleware,
    SlidingWindowMiddleware,
    StableToolOutputCapMiddleware,
    build_agent_middleware,
)
from app.chroma_rag import LocalChromaRetriever
from app.memory_presets import resolve_memory_preset
from app.telemetry import (
    COMPACTION_SIGNAL_NAMES,
    TurnUsageHandler,
    estimate_cost_usd,
    pop_turn_events,
    pop_turn_signals,
    reset_turn_signals,
    usage_totals,
)
from evals.common import COMPACTION_SIGNAL_KEYS, compaction_active


def make_request(messages: list, turn_id: str = "t1") -> SimpleNamespace:
    """A stand-in for langchain's ModelRequest with .messages/.runtime/.override."""
    runtime = SimpleNamespace(context=SimpleNamespace(kb_session_id=turn_id))
    req = SimpleNamespace(messages=messages, runtime=runtime)
    req.override = lambda messages=None, **_: make_request(
        req.messages if messages is None else messages, turn_id
    )
    return req


class SlidingWindowTests(unittest.TestCase):
    def test_no_trim_within_window(self) -> None:
        reset_turn_signals("t1")
        out = SlidingWindowMiddleware(keep=5)._trim(
            make_request([HumanMessage("q1"), AIMessage("a1")])
        )
        self.assertEqual(len(out.messages), 2)
        self.assertEqual(pop_turn_signals("t1"), {})

    def test_cut_lands_on_user_boundary(self) -> None:
        msgs = [
            HumanMessage("q1"),
            AIMessage("a1"),
            HumanMessage("q2"),
            AIMessage(""),
            ToolMessage(content="r", tool_call_id="c1"),
            AIMessage("a2"),
            HumanMessage("q3"),
        ]
        reset_turn_signals("t1")
        out = SlidingWindowMiddleware(keep=5)._trim(make_request(msgs))
        self.assertEqual(out.messages[0].type, "human")
        self.assertEqual(pop_turn_signals("t1")["dropped_messages"], 2)

    def test_advances_past_tool_to_avoid_orphan(self) -> None:
        # A naive last-3 cut would start on the tool result; the window must
        # advance to the next user message so no tool result is orphaned.
        msgs = [
            HumanMessage("q1"),
            AIMessage("a1"),
            HumanMessage("q2"),
            AIMessage(""),
            ToolMessage(content="r", tool_call_id="c1"),
            AIMessage("a2"),
            HumanMessage("q3"),
        ]
        reset_turn_signals("t1")
        out = SlidingWindowMiddleware(keep=3)._trim(make_request(msgs))
        self.assertNotEqual(out.messages[0].type, "tool")
        self.assertEqual(out.messages[0].type, "human")

    def test_trims_long_tool_turn_instead_of_no_op(self) -> None:
        # The current turn's tool loop is longer than keep, so the naive cut
        # lands inside it (no later user message). Must still drop the prior
        # turn and keep the current turn intact, not return the full list.
        msgs = [
            HumanMessage("old"),
            AIMessage("oldA"),  # prior turn -> droppable
            HumanMessage("current"),
            AIMessage(""),
            ToolMessage(content="r1", tool_call_id="c1"),
            AIMessage(""),
            ToolMessage(content="r2", tool_call_id="c2"),
            AIMessage("final"),
        ]
        reset_turn_signals("t1")
        out = SlidingWindowMiddleware(keep=3)._trim(make_request(msgs))
        self.assertEqual(out.messages[0].content, "current")  # current turn kept
        self.assertEqual(pop_turn_signals("t1")["dropped_messages"], 2)


class ObservationTruncationTests(unittest.TestCase):
    def _mw(self) -> ObservationTruncationMiddleware:
        return ObservationTruncationMiddleware(
            head_chars=10, tail_chars=5, trigger_chars=30
        )

    def test_truncates_large_tool_output(self) -> None:
        msgs = [HumanMessage("q"), ToolMessage(content="X" * 100, tool_call_id="c1")]
        reset_turn_signals("t1")
        out = self._mw()._truncate(make_request(msgs))
        tool_msg = out.messages[1]
        self.assertEqual(tool_msg.type, "tool")
        self.assertLess(len(tool_msg.content), 100)
        self.assertTrue(tool_msg.content.startswith("X" * 10))
        signals = pop_turn_signals("t1")
        self.assertEqual(signals["truncated_tool_outputs"], 1)
        self.assertGreater(signals["chars_saved"], 0)

    def test_small_output_and_non_tool_untouched(self) -> None:
        msgs = [HumanMessage("Y" * 100), ToolMessage(content="tiny", tool_call_id="c1")]
        reset_turn_signals("t1")
        out = self._mw()._truncate(make_request(msgs))
        self.assertEqual(out.messages[0].content, "Y" * 100)  # human untouched
        self.assertEqual(out.messages[1].content, "tiny")  # below trigger
        self.assertEqual(pop_turn_signals("t1"), {})

    def test_just_over_trigger_never_inflates(self) -> None:
        # Content barely over the trigger: the marker boilerplate would make the
        # "truncated" copy longer, so it must be left untouched (no inflation,
        # no negative chars_saved).
        mw = ObservationTruncationMiddleware(
            head_chars=20, tail_chars=5, trigger_chars=26
        )
        content = "Z" * 27  # > trigger, but head+tail+marker >> 27
        reset_turn_signals("t1")
        out = mw._truncate(
            make_request([ToolMessage(content=content, tool_call_id="c")])
        )
        self.assertEqual(out.messages[0].content, content)
        self.assertEqual(pop_turn_signals("t1"), {})


OVERSIZED_TOOL_OUTPUT = "HEAD" + "x" * 20_000 + "TAIL"


@tool
def big_lookup(query: str) -> str:
    """Return deliberately oversized evidence."""
    del query
    return OVERSIZED_TOOL_OUTPUT


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Scripted model that accepts tool binding (the base class raises)."""

    def bind_tools(self, tools, **kwargs):
        return self


class StableToolOutputCapTests(unittest.TestCase):
    def test_cap_persists_to_checkpoint_and_summarizer_never_sees_raw(self) -> None:
        model = ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "big_lookup",
                            "args": {"query": "q"},
                            "id": "call-big",
                        }
                    ],
                ),
                AIMessage(content="final answer"),
            ]
        )
        agent = create_agent(
            model=model,
            tools=[big_lookup],
            middleware=[StableToolOutputCapMiddleware(2_048)],
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "cap-thread"}}
        reset_turn_signals("cap-e2e")
        agent.invoke(
            {"messages": [HumanMessage(content="look this up")]},
            config=config,
            context=AppContext(allowed_sources=(), kb_session_id="cap-e2e"),
        )

        checkpointed = agent.get_state(config).values["messages"]
        tool_messages = [m for m in checkpointed if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_messages), 1)
        capped = tool_messages[0]
        self.assertLessEqual(len(capped.content.encode("utf-8")), 2_048)
        self.assertIn("truncated at stable 2048-byte cap", capped.content)
        self.assertNotIn(OVERSIZED_TOOL_OUTPUT, capped.content)
        metadata = capped.additional_kwargs["stable_tool_cap"]
        self.assertEqual(
            metadata["sha256"],
            hashlib.sha256(OVERSIZED_TOOL_OUTPUT.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            metadata["original_bytes"], len(OVERSIZED_TOOL_OUTPUT.encode("utf-8"))
        )
        self.assertEqual(pop_turn_signals("cap-e2e")["tool_outputs_capped"], 1)

        # XML path: the summarizer's prompt is built from the checkpointed
        # (capped) ToolMessage, never the raw oversized output.
        summarizer = InstrumentedSummarizationMiddleware(
            model=ExperimentCompactionMiddlewareTests.FakeModel(),
            trigger=("tokens", 100),
            keep=("tokens", 30),
            trim_tokens_to_summarize=None,
        )
        plan = summarizer._plan_compaction({"messages": checkpointed})
        self.assertIsNotNone(plan)
        planned_tool = next(m for m in plan["trimmed"] if isinstance(m, ToolMessage))
        self.assertEqual(planned_tool.content, capped.content)
        prompt = summarizer._summary_prompt_text(plan["trimmed"])
        self.assertIn("truncated at stable 2048-byte cap", prompt)
        self.assertNotIn(OVERSIZED_TOOL_OUTPUT, prompt)

        # Structured path: the summary request extends the same checkpointed
        # prefix, so it carries the identical capped ToolMessage.
        structured = PrefixPreservingCompactionMiddleware(
            model=ExperimentCompactionMiddlewareTests.FakeModel(),
            trigger=("tokens", 100),
            keep=("tokens", 30),
            trim_tokens_to_summarize=None,
        )
        request = ModelRequest(
            model=ExperimentCompactionMiddlewareTests.FakeModel(),
            messages=list(checkpointed),
            system_message=SystemMessage(content="system"),
            tools=[],
            tool_choice=None,
            response_format=None,
            model_settings={},
            state={"messages": list(checkpointed)},
            runtime=SimpleNamespace(
                context=SimpleNamespace(kb_session_id="cap-e2e", cache_user_id="")
            ),
        )
        structured_plan = structured._plan_compaction(request.state)
        self.assertIsNotNone(structured_plan)
        _, summary_request_messages = structured._prepare_summary_request(
            request, structured_plan
        )
        request_tools = [
            m for m in summary_request_messages if isinstance(m, ToolMessage)
        ]
        self.assertEqual([m.content for m in request_tools], [capped.content])

    def test_cap_is_persistent_bounded_and_auditable(self) -> None:
        raw = "HEAD" + ("é" * 30_000) + "TAIL"
        request = make_request([], "stable-cap")
        reset_turn_signals("stable-cap")
        result = StableToolOutputCapMiddleware(40_000)._cap(
            request,
            ToolMessage(content=raw, tool_call_id="call-cap"),
        )
        self.assertLessEqual(len(result.content.encode("utf-8")), 40_000)
        self.assertTrue(result.content.startswith("HEAD"))
        self.assertTrue(result.content.endswith("TAIL"))
        metadata = result.additional_kwargs["stable_tool_cap"]
        self.assertEqual(metadata["original_bytes"], len(raw.encode("utf-8")))
        self.assertEqual(len(metadata["sha256"]), 64)
        signals = pop_turn_signals("stable-cap")
        self.assertEqual(signals["tool_outputs_capped"], 1)
        self.assertGreater(
            signals["tool_output_original_bytes"],
            signals["tool_output_retained_bytes"],
        )


class ExperimentCompactionMiddlewareTests(unittest.TestCase):
    class FakeModel:
        _llm_type = "fake-chat-model"

        def __init__(self, responses: list[str] | None = None) -> None:
            self.bound: list[dict] = []
            self.prompts: list[str] = []
            self.responses = list(responses or ["durable full-input summary"])

        def bind(self, **kwargs):
            self.bound.append(kwargs)
            return self

        def invoke(self, prompt, config=None):
            self.prompts.append(prompt)
            return AIMessage(content=self.responses.pop(0))

        async def ainvoke(self, prompt, config=None):
            return self.invoke(prompt, config=config)

        def _get_ls_params(self):
            return {"ls_provider": "deepseek"}

    class StructuredFakeModel(FakeModel):
        def __init__(self, responses: list[str] | None = None) -> None:
            super().__init__(responses)
            self.bound_tools: list[tuple[list, dict]] = []
            self.invocations: list[tuple[list, dict | None]] = []

        def bind_tools(self, tools, **kwargs):
            self.bound_tools.append((list(tools), dict(kwargs)))
            return self

        def invoke(self, prompt, config=None):
            self.invocations.append((list(prompt), config))
            return AIMessage(
                content=self.responses.pop(0),
                usage_metadata={
                    "input_tokens": 10_000,
                    "output_tokens": 100,
                    "total_tokens": 10_100,
                    "input_token_details": {"cache_read": 9_000},
                },
                response_metadata={"model_name": "deepseek-v4-flash"},
            )

    class ScriptedMessageModel(FakeModel):
        """Structured-path fake returning prebuilt AIMessage responses verbatim."""

        def __init__(self, responses: list[AIMessage]) -> None:
            super().__init__([])
            self.message_responses = list(responses)
            self.invocations: list[tuple[list, dict | None]] = []

        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, prompt, config=None):
            self.invocations.append((list(prompt), config))
            return self.message_responses.pop(0)

    @staticmethod
    def _structured_request(model, messages: list, turn_id: str) -> ModelRequest:
        return ModelRequest(
            model=model,
            messages=messages,
            system_message=SystemMessage(content="system"),
            tools=[],
            tool_choice=None,
            response_format=None,
            model_settings={},
            state={"messages": messages},
            runtime=SimpleNamespace(
                context=SimpleNamespace(kb_session_id=turn_id, cache_user_id="")
            ),
        )

    def test_cache_user_id_is_added_to_agent_model_settings(self) -> None:
        runtime = SimpleNamespace(
            context=SimpleNamespace(cache_user_id="eval_abc", kb_session_id="turn")
        )
        request = SimpleNamespace(
            runtime=runtime, model_settings={}, messages=[], system_message=None
        )
        request.override = lambda **updates: SimpleNamespace(
            runtime=runtime,
            model_settings=updates.get("model_settings", request.model_settings),
        )
        isolated = DeepSeekCacheIsolationMiddleware()._isolate(request)
        self.assertEqual(isolated.model_settings["extra_body"], {"user_id": "eval_abc"})

    def test_agent_request_guard_fails_before_model_handler(self) -> None:
        runtime = SimpleNamespace(
            context=SimpleNamespace(cache_user_id="eval_guard", kb_session_id="turn")
        )
        request = SimpleNamespace(
            runtime=runtime,
            model_settings={},
            messages=[HumanMessage(content="x" * 4_000)],
            system_message=None,
        )
        with self.assertRaisesRegex(RuntimeError, "Agent request exceeds"):
            DeepSeekCacheIsolationMiddleware(100)._isolate(request)

    def test_full_selected_history_reaches_summarizer_and_records_event(self) -> None:
        model = self.FakeModel()
        middleware = InstrumentedSummarizationMiddleware(
            model=model,
            trigger=("tokens", 1_000),
            keep=("tokens", 500),
            trim_tokens_to_summarize=None,
        )
        messages = [
            (HumanMessage if index % 2 == 0 else AIMessage)(content="x" * 1_000)
            for index in range(28)
        ]
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                kb_session_id="summary-turn", cache_user_id="eval_summary"
            )
        )
        reset_turn_signals("summary-turn")
        update = middleware.before_model({"messages": messages}, runtime)
        self.assertIsNotNone(update)
        events = pop_turn_events("summary-turn")
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertTrue(event["summary_input_untrimmed"])
        self.assertEqual(event["configured_trigger_tokens"], 1_000)
        self.assertGreater(event["summary_input_tokens_approx"], 4_000)
        self.assertLessEqual(event["retained_tail_tokens_approx"], 500)
        self.assertEqual(pop_turn_signals("summary-turn")["compactions_this_turn"], 1)
        self.assertEqual(model.bound[-1]["extra_body"], {"user_id": "eval_summary"})
        self.assertGreater(len(model.prompts[-1]), 20_000)

    def test_provider_reported_tokens_can_trigger_below_approximation(self) -> None:
        model = self.FakeModel()
        middleware = InstrumentedSummarizationMiddleware(
            model=model,
            trigger=("tokens", 200_000),
            keep=("tokens", 500),
            trim_tokens_to_summarize=None,
        )
        messages = [HumanMessage(content="x" * 2_000) for _ in range(6)]
        messages.append(
            AIMessage(
                content="previous answer",
                usage_metadata={
                    "input_tokens": 205_664,
                    "output_tokens": 1_672,
                    "total_tokens": 207_336,
                },
                response_metadata={"model_provider": "deepseek"},
            )
        )
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                kb_session_id="reported-trigger", cache_user_id="eval_reported"
            )
        )
        reset_turn_signals("reported-trigger")
        with mock.patch.object(middleware, "token_counter", return_value=199_567):
            update = middleware.before_model({"messages": messages}, runtime)
        self.assertIsNotNone(update)
        event = pop_turn_events("reported-trigger")[0]
        self.assertEqual(event["pre_compaction_tokens_approx"], 199_567)
        self.assertEqual(event["trigger_reported_tokens"], 207_336)
        self.assertEqual(event["trigger_source"], "provider_reported")

    def test_summary_input_guard_fails_before_provider_call(self) -> None:
        model = self.FakeModel()
        middleware = InstrumentedSummarizationMiddleware(
            model=model,
            trigger=("tokens", 100),
            keep=("tokens", 100),
            trim_tokens_to_summarize=None,
            summary_input_guard_tokens=200,
        )
        messages = [HumanMessage(content="x" * 2_000) for _ in range(4)]
        runtime = SimpleNamespace(
            context=SimpleNamespace(kb_session_id="guard", cache_user_id="eval_guard")
        )
        with self.assertRaisesRegex(RuntimeError, "safety guard"):
            middleware.before_model({"messages": messages}, runtime)
        self.assertEqual(model.prompts, [])

    def test_empty_summary_is_retried_and_recorded(self) -> None:
        model = self.FakeModel(["", "durable retry summary"])
        middleware = InstrumentedSummarizationMiddleware(
            model=model,
            trigger=("tokens", 100),
            keep=("tokens", 100),
            trim_tokens_to_summarize=None,
        )
        messages = [HumanMessage(content="x" * 2_000) for _ in range(4)]
        runtime = SimpleNamespace(
            context=SimpleNamespace(kb_session_id="retry", cache_user_id="eval_retry")
        )
        reset_turn_signals("retry")
        with mock.patch("app.chat_service.time.sleep") as sleep:
            update = middleware.before_model({"messages": messages}, runtime)
        self.assertIsNotNone(update)
        self.assertEqual(len(model.prompts), 2)
        sleep.assert_called_once_with(1.0)
        event = pop_turn_events("retry")[0]
        self.assertEqual(event["summary_attempts"], 2)
        self.assertEqual(event["summary_retry_reasons"], ["empty response"])

    def test_non_retryable_summary_failure_is_not_retried(self) -> None:
        model = self.FakeModel()
        model.invoke = mock.Mock(side_effect=ValueError("invalid request"))
        middleware = InstrumentedSummarizationMiddleware(
            model=model,
            trigger=("tokens", 100),
            keep=("tokens", 100),
            trim_tokens_to_summarize=None,
        )
        messages = [HumanMessage(content="x" * 2_000) for _ in range(4)]
        runtime = SimpleNamespace(
            context=SimpleNamespace(kb_session_id="no-retry", cache_user_id="eval")
        )
        with self.assertRaisesRegex(ValueError, "invalid request"):
            middleware.before_model({"messages": messages}, runtime)
        self.assertEqual(model.invoke.call_count, 1)

    def test_real_agent_attributes_summary_and_agent_calls_separately(self) -> None:
        def response(text: str, input_tokens: int, output_tokens: int) -> AIMessage:
            return AIMessage(
                content=text,
                usage_metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "input_token_details": {"cache_read": 0},
                },
                response_metadata={"model_name": "deepseek-v4-flash"},
            )

        model = FakeMessagesListChatModel(
            responses=[response("summary", 6_000, 10), response("answer", 700, 20)]
        )
        summary = InstrumentedSummarizationMiddleware(
            model=model,
            trigger=("tokens", 1_000),
            keep=("tokens", 500),
            trim_tokens_to_summarize=None,
            summary_input_guard_tokens=900_000,
        )
        agent = create_agent(
            model=model,
            tools=[],
            middleware=[DeepSeekCacheIsolationMiddleware(900_000), summary],
        )
        messages = [
            (HumanMessage if index % 2 == 0 else AIMessage)(content="x" * 1_000)
            for index in range(28)
        ]
        handler = TurnUsageHandler()
        reset_turn_signals("integration-turn")
        result = agent.invoke(
            {"messages": messages},
            config={"callbacks": [handler]},
            context=AppContext(
                allowed_sources=(),
                kb_session_id="integration-turn",
                cache_user_id="eval_integration",
            ),
        )
        self.assertEqual(handler.llm_calls, 2)
        self.assertEqual(
            [call["source"] for call in handler.model_calls],
            ["summarization", "agent"],
        )
        self.assertEqual(result["messages"][-1].content, "answer")
        event = pop_turn_events("integration-turn")[0]
        self.assertGreater(event["summary_input_tokens_approx"], 4_000)
        self.assertEqual(
            pop_turn_signals("integration-turn")["compactions_this_turn"], 1
        )

    def test_structured_prefix_preserves_request_shape_and_persists_checkpoint(
        self,
    ) -> None:
        model = self.StructuredFakeModel(["durable structured summary"])
        middleware = PrefixPreservingCompactionMiddleware(
            model=model,
            trigger=("tokens", 1_000),
            keep=("tokens", 500),
            trim_tokens_to_summarize=None,
            summary_input_guard_tokens=900_000,
        )
        messages = [
            (HumanMessage if index % 2 == 0 else AIMessage)(
                content=f"m{index}:" + "x" * 1_000
            )
            for index in range(28)
        ]
        system = SystemMessage(content="stable system prompt")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup evidence",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                kb_session_id="prefix-turn", cache_user_id="stable-prefix-user"
            )
        )
        request = ModelRequest(
            model=model,
            messages=messages,
            system_message=system,
            tools=tools,
            tool_choice=None,
            response_format=None,
            model_settings={"extra_body": {"user_id": "stable-prefix-user"}},
            state={"messages": messages},
            runtime=runtime,
        )
        expected_plan = middleware._plan_compaction(request.state)
        self.assertIsNotNone(expected_plan)
        handled: list[ModelRequest] = []

        def handler(compacted_request):
            handled.append(compacted_request)
            return ModelResponse(
                result=[AIMessage(content="final answer", id="answer")]
            )

        reset_turn_signals("prefix-turn")
        result = middleware.wrap_model_call(request, handler)

        self.assertEqual(model.bound_tools[0][0], tools)
        self.assertEqual(
            model.bound_tools[0][1]["extra_body"],
            {"user_id": "stable-prefix-user"},
        )
        summary_messages, summary_config = model.invocations[0]
        self.assertIs(summary_messages[0], system)
        self.assertEqual(summary_messages[1:-1], messages)
        self.assertNotIn("<messages>", summary_messages[-1].content)
        self.assertIn(
            f"final {len(expected_plan['preserved'])} messages",
            summary_messages[-1].content,
        )
        self.assertIn(
            f"approximately {middleware._partial_token_counter(expected_plan['preserved'])} tokens",
            summary_messages[-1].content,
        )
        self.assertEqual(summary_config["metadata"]["lc_source"], "summarization")
        self.assertEqual(
            summary_config["metadata"]["compaction_strategy"],
            "structured_prefix",
        )

        compacted = handled[0].messages
        self.assertEqual(
            compacted[0].additional_kwargs.get("lc_source"), "summarization"
        )
        self.assertEqual(handled[0].state["messages"], compacted)
        command_messages = result.command.update["messages"]
        self.assertIsInstance(command_messages[0], RemoveMessage)
        self.assertEqual(command_messages[0].id, REMOVE_ALL_MESSAGES)
        self.assertEqual(command_messages[-1].content, "final answer")

        event = pop_turn_events("prefix-turn")[0]
        self.assertEqual(event["summary_strategy"], "structured_prefix")
        self.assertTrue(event["summary_request_is_strict_extension"])
        self.assertEqual(event["summary_prefix_messages"], len(messages))
        self.assertLess(event["summary_selected_messages"], len(messages))
        self.assertEqual(
            event["summary_instruction_retained_messages"],
            len(expected_plan["preserved"]),
        )
        self.assertTrue(event["summary_system_message_present"])
        self.assertEqual(event["summary_tools_bound"], 1)
        self.assertTrue(event["summary_cache_user_id_preserved"])
        self.assertEqual(event["summary_provider_input_tokens"], 10_000)
        self.assertEqual(event["summary_provider_cache_read_tokens"], 9_000)
        self.assertEqual(event["summary_provider_cache_miss_tokens"], 1_000)
        self.assertEqual(event["summary_provider_cache_hit_ratio"], 0.9)

    def test_structured_prefix_safe_tail_never_starts_with_orphaned_tool(self) -> None:
        model = self.StructuredFakeModel(["summary"])
        middleware = PrefixPreservingCompactionMiddleware(
            model=model,
            trigger=("tokens", 100),
            keep=("tokens", 120),
            trim_tokens_to_summarize=None,
        )
        messages = [
            HumanMessage(content="old " + "x" * 2_000),
            AIMessage(content="old answer " + "x" * 2_000),
            HumanMessage(content="tool turn"),
            AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {}, "id": "call-1"}],
            ),
            ToolMessage(content="evidence", tool_call_id="call-1"),
            AIMessage(content="tool answer"),
            HumanMessage(content="current"),
        ]
        plan = middleware._plan_compaction({"messages": messages})
        self.assertIsNotNone(plan)
        self.assertTrue(plan["preserved"])
        self.assertNotIsInstance(plan["preserved"][0], ToolMessage)

    def test_structured_prefix_rejects_an_unpersisted_message_view(self) -> None:
        model = self.StructuredFakeModel(["summary"])
        middleware = PrefixPreservingCompactionMiddleware(
            model=model,
            trigger=("tokens", 100),
            keep=("tokens", 100),
            trim_tokens_to_summarize=None,
        )
        state_messages = [HumanMessage(content="x" * 2_000) for _ in range(4)]
        request = ModelRequest(
            model=model,
            messages=state_messages[1:],
            system_message=SystemMessage(content="system"),
            tools=[],
            response_format=None,
            state={"messages": state_messages},
            runtime=SimpleNamespace(
                context=SimpleNamespace(kb_session_id="mismatched-view")
            ),
        )
        plan = middleware._plan_compaction(request.state)
        self.assertIsNotNone(plan)
        with self.assertRaisesRegex(RuntimeError, "match checkpoint history"):
            middleware._prepare_summary_request(request, plan)

    def test_structured_prefix_empty_summary_is_retried_and_recorded(self) -> None:
        # Empty text and tool_calls-with-empty-text are both "empty" responses:
        # each is retried and shows up in summary_retry_reasons.
        model = self.ScriptedMessageModel(
            [
                AIMessage(content=""),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "lookup", "args": {}, "id": "call-empty"}],
                ),
                AIMessage(content="structured retry checkpoint"),
            ]
        )
        middleware = PrefixPreservingCompactionMiddleware(
            model=model,
            trigger=("tokens", 100),
            keep=("tokens", 100),
            trim_tokens_to_summarize=None,
        )
        messages = [HumanMessage(content="x" * 2_000, id=f"h{i}") for i in range(4)]
        request = self._structured_request(model, messages, "structured-retry")
        handled: list[ModelRequest] = []

        def handler(compacted_request):
            handled.append(compacted_request)
            return ModelResponse(result=[AIMessage(content="answer", id="a")])

        reset_turn_signals("structured-retry")
        with mock.patch("app.chat_service.time.sleep") as sleep:
            middleware.wrap_model_call(request, handler)
        self.assertEqual(len(model.invocations), 3)
        self.assertEqual(sleep.call_args_list, [mock.call(1.0), mock.call(2.0)])
        self.assertEqual(len(handled), 1)
        self.assertIn("structured retry checkpoint", handled[0].messages[0].content)
        event = pop_turn_events("structured-retry")[0]
        self.assertEqual(event["summary_attempts"], 3)
        self.assertEqual(
            event["summary_retry_reasons"], ["empty response", "empty response"]
        )
        self.assertEqual(
            pop_turn_signals("structured-retry")["compactions_this_turn"], 1
        )

    def test_structured_prefix_empty_summary_raises_after_max_attempts(self) -> None:
        model = self.ScriptedMessageModel([AIMessage(content="")] * 3)
        middleware = PrefixPreservingCompactionMiddleware(
            model=model,
            trigger=("tokens", 100),
            keep=("tokens", 100),
            trim_tokens_to_summarize=None,
        )
        messages = [HumanMessage(content="x" * 2_000, id=f"h{i}") for i in range(4)]
        request = self._structured_request(model, messages, "structured-exhausted")
        handled: list[ModelRequest] = []
        reset_turn_signals("structured-exhausted")
        with mock.patch("app.chat_service.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "empty summary after 3 attempts"):
                middleware.wrap_model_call(request, handled.append)
        self.assertEqual(len(model.invocations), 3)
        # The agent call never runs on a failed checkpoint, and nothing is
        # recorded as a successful compaction.
        self.assertEqual(handled, [])
        self.assertEqual(pop_turn_events("structured-exhausted"), [])
        self.assertNotIn(
            "compactions_this_turn", pop_turn_signals("structured-exhausted")
        )

    def test_structured_prefix_uses_text_and_ignores_summary_tool_calls(self) -> None:
        model = self.ScriptedMessageModel(
            [
                AIMessage(
                    content="checkpoint despite tool call",
                    tool_calls=[{"name": "lookup", "args": {}, "id": "call-x"}],
                )
            ]
        )
        middleware = PrefixPreservingCompactionMiddleware(
            model=model,
            trigger=("tokens", 100),
            keep=("tokens", 100),
            trim_tokens_to_summarize=None,
        )
        messages = [HumanMessage(content="x" * 2_000, id=f"h{i}") for i in range(4)]
        request = self._structured_request(model, messages, "structured-toolcall")
        handled: list[ModelRequest] = []

        def handler(compacted_request):
            handled.append(compacted_request)
            return ModelResponse(result=[AIMessage(content="answer", id="a")])

        reset_turn_signals("structured-toolcall")
        result = middleware.wrap_model_call(request, handler)
        self.assertEqual(len(model.invocations), 1)
        self.assertIn("checkpoint despite tool call", handled[0].messages[0].content)
        # The summarizer's AIMessage never enters state, so its tool calls can
        # never be executed: no message anywhere carries call-x.
        command_messages = result.command.update["messages"]
        self.assertFalse(
            any(
                call["id"] == "call-x"
                for message in [*handled[0].messages, *command_messages]
                if isinstance(message, AIMessage)
                for call in (message.tool_calls or [])
            )
        )
        event = pop_turn_events("structured-toolcall")[0]
        self.assertEqual(event["summary_attempts"], 1)
        self.assertEqual(event["summary_retry_reasons"], [])
        self.assertEqual(
            pop_turn_signals("structured-toolcall")["compactions_this_turn"], 1
        )

    def test_real_agent_structured_compaction_rewrites_checkpoint_once(self) -> None:
        def response(text: str, input_tokens: int, cache_read: int) -> AIMessage:
            return AIMessage(
                content=text,
                usage_metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": 10,
                    "total_tokens": input_tokens + 10,
                    "input_token_details": {"cache_read": cache_read},
                },
                response_metadata={"model_name": "deepseek-v4-flash"},
            )

        model = FakeMessagesListChatModel(
            responses=[
                response("structured checkpoint", 6_000, 5_500),
                response("answer after checkpoint", 700, 0),
            ]
        )
        compactor = PrefixPreservingCompactionMiddleware(
            model=model,
            trigger=("tokens", 1_000),
            keep=("tokens", 500),
            trim_tokens_to_summarize=None,
            summary_input_guard_tokens=900_000,
        )
        agent = create_agent(
            model=model,
            tools=[],
            system_prompt="stable system",
            middleware=[DeepSeekCacheIsolationMiddleware(900_000), compactor],
        )
        messages = [
            (HumanMessage if index % 2 == 0 else AIMessage)(
                content=f"old-{index}:" + "x" * 1_000
            )
            for index in range(28)
        ]
        handler = TurnUsageHandler()
        reset_turn_signals("structured-integration")
        result = agent.invoke(
            {"messages": messages},
            config={"callbacks": [handler]},
            context=AppContext(
                allowed_sources=(),
                kb_session_id="structured-integration",
                cache_user_id="eval-structured-integration",
            ),
        )

        self.assertEqual(handler.llm_calls, 2)
        self.assertEqual(
            [call["source"] for call in handler.model_calls],
            ["summarization", "agent"],
        )
        summaries = [
            message
            for message in result["messages"]
            if message.additional_kwargs.get("lc_source") == "summarization"
        ]
        self.assertEqual(len(summaries), 1)
        self.assertIn("structured checkpoint", summaries[0].content)
        answers = [
            message
            for message in result["messages"]
            if isinstance(message, AIMessage)
            and message.content == "answer after checkpoint"
        ]
        self.assertEqual(len(answers), 1)
        self.assertFalse(
            any(
                message.content == messages[0].content for message in result["messages"]
            )
        )
        event = pop_turn_events("structured-integration")[0]
        self.assertEqual(event["summary_provider_cache_read_tokens"], 5_500)
        self.assertEqual(event["summary_provider_cache_miss_tokens"], 500)
        self.assertEqual(
            pop_turn_signals("structured-integration")["compactions_this_turn"],
            1,
        )


class ExperimentCompactionMiddlewareAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_empty_summary_is_retried(self) -> None:
        model = ExperimentCompactionMiddlewareTests.FakeModel(["", "async summary"])
        middleware = InstrumentedSummarizationMiddleware(
            model=model,
            trigger=("tokens", 100),
            keep=("tokens", 100),
            trim_tokens_to_summarize=None,
        )
        messages = [HumanMessage(content="x" * 2_000) for _ in range(4)]
        runtime = SimpleNamespace(
            context=SimpleNamespace(
                kb_session_id="async-retry", cache_user_id="eval_async_retry"
            )
        )
        reset_turn_signals("async-retry")
        with mock.patch("app.chat_service.asyncio.sleep") as sleep:
            update = await middleware.abefore_model({"messages": messages}, runtime)
        self.assertIsNotNone(update)
        self.assertEqual(len(model.prompts), 2)
        sleep.assert_awaited_once_with(1.0)
        event = pop_turn_events("async-retry")[0]
        self.assertEqual(event["summary_attempts"], 2)


class CompactionPathEquivalenceTests(unittest.TestCase):
    """Same history + config: both paths must agree on the compaction boundary."""

    TRIGGER = ("tokens", 1_000)
    KEEP = ("tokens", 500)

    def _history(self) -> list:
        # Pre-assigned ids let boundary selection be compared across paths.
        messages = [
            (HumanMessage if index % 2 == 0 else AIMessage)(
                content=f"m{index}:" + "x" * 1_000, id=f"m{index}"
            )
            for index in range(24)
        ]
        messages += [
            AIMessage(
                content="",
                id="m-toolcall",
                tool_calls=[{"name": "lookup", "args": {}, "id": "call-1"}],
            ),
            ToolMessage(content="evidence", tool_call_id="call-1", id="m-tool"),
            AIMessage(content="tool answer", id="m-tool-answer"),
            HumanMessage(content="current question", id="m-current"),
        ]
        return messages

    def test_both_paths_select_the_identical_boundary(self) -> None:
        history = self._history()
        xml_plan = InstrumentedSummarizationMiddleware(
            model=ExperimentCompactionMiddlewareTests.FakeModel(),
            trigger=self.TRIGGER,
            keep=self.KEEP,
            trim_tokens_to_summarize=None,
        )._plan_compaction({"messages": history})
        structured_plan = PrefixPreservingCompactionMiddleware(
            model=ExperimentCompactionMiddlewareTests.StructuredFakeModel(),
            trigger=self.TRIGGER,
            keep=self.KEEP,
            trim_tokens_to_summarize=None,
        )._plan_compaction({"messages": history})
        self.assertIsNotNone(xml_plan)
        self.assertIsNotNone(structured_plan)
        self.assertEqual(
            [m.id for m in xml_plan["selected"]],
            [m.id for m in structured_plan["selected"]],
        )
        self.assertEqual(
            [m.id for m in xml_plan["preserved"]],
            [m.id for m in structured_plan["preserved"]],
        )
        # The boundary is a clean partition of the full history.
        self.assertEqual(
            [m.id for m in [*xml_plan["selected"], *xml_plan["preserved"]]],
            [m.id for m in history],
        )
        self.assertGreater(len(xml_plan["selected"]), 0)
        self.assertGreater(len(xml_plan["preserved"]), 0)

    def test_both_paths_install_the_same_post_compaction_structure(self) -> None:
        history = self._history()

        xml_middleware = InstrumentedSummarizationMiddleware(
            model=ExperimentCompactionMiddlewareTests.FakeModel(["xml summary"]),
            trigger=self.TRIGGER,
            keep=self.KEEP,
            trim_tokens_to_summarize=None,
        )
        reset_turn_signals("eq-xml")
        xml_update = xml_middleware.before_model(
            {"messages": list(history)},
            SimpleNamespace(
                context=SimpleNamespace(kb_session_id="eq-xml", cache_user_id="")
            ),
        )
        pop_turn_events("eq-xml")
        pop_turn_signals("eq-xml")

        structured_model = ExperimentCompactionMiddlewareTests.StructuredFakeModel(
            ["structured summary"]
        )
        structured_middleware = PrefixPreservingCompactionMiddleware(
            model=structured_model,
            trigger=self.TRIGGER,
            keep=self.KEEP,
            trim_tokens_to_summarize=None,
        )
        request = ExperimentCompactionMiddlewareTests._structured_request(
            structured_model, list(history), "eq-structured"
        )
        handled: list[ModelRequest] = []

        def handler(compacted_request):
            handled.append(compacted_request)
            return ModelResponse(result=[AIMessage(content="answer", id="a")])

        reset_turn_signals("eq-structured")
        result = structured_middleware.wrap_model_call(request, handler)
        pop_turn_events("eq-structured")
        pop_turn_signals("eq-structured")

        self.assertIsInstance(xml_update["messages"][0], RemoveMessage)
        self.assertEqual(xml_update["messages"][0].id, REMOVE_ALL_MESSAGES)
        xml_summary, xml_tail = xml_update["messages"][1], xml_update["messages"][2:]
        compacted = handled[0].messages
        structured_summary, structured_tail = compacted[0], compacted[1:]

        for summary in (xml_summary, structured_summary):
            self.assertIsInstance(summary, HumanMessage)
            self.assertEqual(
                summary.additional_kwargs.get("lc_source"), "summarization"
            )
            self.assertTrue(
                summary.content.startswith(
                    "Here is a summary of the conversation to date:"
                )
            )
        self.assertEqual([m.id for m in xml_tail], [m.id for m in structured_tail])
        self.assertEqual(
            [m.content for m in xml_tail], [m.content for m in structured_tail]
        )
        # Only the summary text differs between the two paths.
        self.assertIn("xml summary", xml_summary.content)
        self.assertIn("structured summary", structured_summary.content)
        # The structured path's checkpoint command installs the same structure.
        command_messages = result.command.update["messages"]
        self.assertIsInstance(command_messages[0], RemoveMessage)
        self.assertIs(command_messages[1], structured_summary)
        self.assertEqual(
            [m.id for m in command_messages[2:-1]],
            [m.id for m in structured_tail],
        )
        self.assertEqual(command_messages[-1].content, "answer")


class MultiCompactionTests(unittest.TestCase):
    def test_xml_second_compaction_replaces_prior_summary_without_orphans(
        self,
    ) -> None:
        model = ExperimentCompactionMiddlewareTests.FakeModel(
            ["first summary", "second summary"]
        )
        middleware = InstrumentedSummarizationMiddleware(
            model=model,
            trigger=("tokens", 1_000),
            keep=("tokens", 500),
            trim_tokens_to_summarize=None,
        )
        runtime = SimpleNamespace(
            context=SimpleNamespace(kb_session_id="multi-xml", cache_user_id="")
        )
        history = [
            (HumanMessage if index % 2 == 0 else AIMessage)(
                content=f"m{index}:" + "x" * 1_000, id=f"m{index}"
            )
            for index in range(24)
        ]
        reset_turn_signals("multi-xml")
        first_update = middleware.before_model({"messages": history}, runtime)
        self.assertIsNotNone(first_update)
        summarized_state = list(first_update["messages"][1:])

        second_turn = [
            HumanMessage(content="new question " + "y" * 3_000, id="n0"),
            AIMessage(
                content="",
                id="n1",
                tool_calls=[{"name": "lookup", "args": {}, "id": "call-2"}],
            ),
            ToolMessage(
                content="evidence " + "y" * 1_000, tool_call_id="call-2", id="n2"
            ),
            AIMessage(content="answer two", id="n3"),
        ]
        second_update = middleware.before_model(
            {"messages": [*summarized_state, *second_turn]}, runtime
        )
        self.assertIsNotNone(second_update)

        final_state = list(second_update["messages"][1:])
        summaries = [
            message
            for message in final_state
            if message.additional_kwargs.get("lc_source") == "summarization"
        ]
        self.assertEqual(len(summaries), 1)
        self.assertIn("second summary", summaries[0].content)
        self.assertNotIn("first summary", summaries[0].content)
        # The first summary fed the second summarization instead of surviving.
        self.assertIn("first summary", model.prompts[1])
        for index, message in enumerate(final_state):
            if isinstance(message, ToolMessage):
                self.assertGreater(index, 0)
                previous = final_state[index - 1]
                self.assertIsInstance(previous, AIMessage)
                self.assertIn(
                    message.tool_call_id,
                    [call["id"] for call in previous.tool_calls],
                )
        self.assertLessEqual(middleware._partial_token_counter(final_state[1:]), 500)
        self.assertEqual(len(pop_turn_events("multi-xml")), 2)
        self.assertEqual(pop_turn_signals("multi-xml")["compactions_this_turn"], 2)

    def test_real_agent_structured_second_compaction_on_summarized_thread(
        self,
    ) -> None:
        def response(text: str) -> AIMessage:
            return AIMessage(
                content=text,
                usage_metadata={
                    "input_tokens": 1_000,
                    "output_tokens": 10,
                    "total_tokens": 1_010,
                    "input_token_details": {"cache_read": 0},
                },
                response_metadata={"model_name": "deepseek-v4-flash"},
            )

        model = FakeMessagesListChatModel(
            responses=[
                response("checkpoint one"),
                response("answer one"),
                response("checkpoint two"),
                response("answer two"),
            ]
        )
        compactor = PrefixPreservingCompactionMiddleware(
            model=model,
            trigger=("tokens", 1_000),
            keep=("tokens", 500),
            trim_tokens_to_summarize=None,
            summary_input_guard_tokens=900_000,
        )
        agent = create_agent(
            model=model,
            tools=[],
            system_prompt="stable system",
            middleware=[compactor],
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "structured-multi"}}
        first_turn = [
            (HumanMessage if index % 2 == 0 else AIMessage)(
                content=f"old-{index}:" + "x" * 1_000
            )
            for index in range(28)
        ]
        reset_turn_signals("structured-multi-t1")
        agent.invoke(
            {"messages": first_turn},
            config=config,
            context=AppContext(allowed_sources=(), kb_session_id="structured-multi-t1"),
        )
        state = agent.get_state(config).values["messages"]
        first_summaries = [
            message
            for message in state
            if message.additional_kwargs.get("lc_source") == "summarization"
        ]
        self.assertEqual(len(first_summaries), 1)
        self.assertIn("checkpoint one", first_summaries[0].content)
        self.assertEqual(len(pop_turn_events("structured-multi-t1")), 1)
        self.assertEqual(
            pop_turn_signals("structured-multi-t1")["compactions_this_turn"], 1
        )

        reset_turn_signals("structured-multi-t2")
        agent.invoke(
            {"messages": [HumanMessage(content="second wave " + "y" * 6_000)]},
            config=config,
            context=AppContext(allowed_sources=(), kb_session_id="structured-multi-t2"),
        )
        state = agent.get_state(config).values["messages"]
        summaries = [
            message
            for message in state
            if message.additional_kwargs.get("lc_source") == "summarization"
        ]
        self.assertEqual(len(summaries), 1)
        self.assertIn("checkpoint two", summaries[0].content)
        contents = [str(message.content) for message in state]
        self.assertFalse(any("checkpoint one" in content for content in contents))
        self.assertFalse(any("old-0:" in content for content in contents))
        self.assertFalse(any(isinstance(m, ToolMessage) for m in state))
        # Retained tail survives verbatim, followed by the new answer.
        self.assertTrue(any(content.startswith("second wave") for content in contents))
        self.assertEqual(contents.count("answer two"), 1)
        event = pop_turn_events("structured-multi-t2")[0]
        self.assertEqual(event["summary_strategy"], "structured_prefix")
        self.assertEqual(event["summary_instruction_retained_messages"], 1)
        self.assertEqual(
            pop_turn_signals("structured-multi-t2")["compactions_this_turn"], 1
        )


class TurnUsageAccountingInvariantTests(unittest.TestCase):
    def test_model_call_rows_sum_to_the_billed_usage_totals(self) -> None:
        # est_cost_usd is computed from usage_by_model; the per-call rows are
        # the explanation. If a call were double-counted (or dropped) on either
        # side, the two aggregates would disagree.
        def response(
            text: str, input_tokens: int, cache_read: int, cache_creation: int
        ) -> AIMessage:
            return AIMessage(
                content=text,
                usage_metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": 40,
                    "total_tokens": input_tokens + 40,
                    "input_token_details": {
                        "cache_read": cache_read,
                        "cache_creation": cache_creation,
                    },
                },
                response_metadata={"model_name": "deepseek-v4-flash"},
            )

        model = FakeMessagesListChatModel(
            responses=[
                response("summary", 6_000, 5_000, 500),
                response("answer", 700, 100, 50),
            ]
        )
        summary = InstrumentedSummarizationMiddleware(
            model=model,
            trigger=("tokens", 1_000),
            keep=("tokens", 500),
            trim_tokens_to_summarize=None,
        )
        agent = create_agent(model=model, tools=[], middleware=[summary])
        messages = [
            (HumanMessage if index % 2 == 0 else AIMessage)(content="x" * 1_000)
            for index in range(28)
        ]
        handler = TurnUsageHandler()
        reset_turn_signals("usage-invariant")
        agent.invoke(
            {"messages": messages},
            config={"callbacks": [handler]},
            context=AppContext(allowed_sources=(), kb_session_id="usage-invariant"),
        )
        pop_turn_events("usage-invariant")
        pop_turn_signals("usage-invariant")

        self.assertEqual(handler.llm_calls, 2)
        self.assertEqual(len(handler.model_calls), 2)
        self.assertEqual(
            sorted(call["source"] for call in handler.model_calls),
            ["agent", "summarization"],
        )
        totals = usage_totals(handler.usage_metadata)
        summed = {
            field: sum(call[field] for call in handler.model_calls)
            for field in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cache_read_tokens",
                "cache_creation_tokens",
            )
        }
        self.assertEqual(summed, totals)
        # Known scripted usage pins the absolute numbers, not just consistency.
        self.assertEqual(totals["input_tokens"], 6_700)
        self.assertEqual(totals["output_tokens"], 80)
        self.assertEqual(totals["cache_read_tokens"], 5_100)
        self.assertEqual(totals["cache_creation_tokens"], 550)
        estimated = estimate_cost_usd(handler.usage_metadata)
        self.assertIsNotNone(estimated)
        self.assertAlmostEqual(
            estimated,
            sum(call["cost"]["total_usd"] for call in handler.model_calls),
        )


class PromptCompressionTests(unittest.TestCase):
    def test_collapses_whitespace_and_reports(self) -> None:
        reset_turn_signals("t1")
        out = PromptCompressionMiddleware()._compress(
            make_request([HumanMessage("a   b\n\n\n\nc   ")])
        )
        self.assertEqual(out.messages[0].content, "a b\n\nc")
        signals = pop_turn_signals("t1")
        self.assertEqual(signals["compressed_messages"], 1)
        self.assertGreater(signals["chars_saved"], 0)

    def test_already_compact_untouched(self) -> None:
        reset_turn_signals("t1")
        PromptCompressionMiddleware()._compress(make_request([HumanMessage("a b\nc")]))
        self.assertEqual(pop_turn_signals("t1"), {})


def _stub_embed(texts: list) -> list:
    """Deterministic embedder: blocks containing 'KEEPME' point one way."""
    return [[1.0, 0.0] if "KEEPME" in t else [0.0, 1.0] for t in texts]


class InContextHistoryRetrievalTests(unittest.TestCase):
    def _msgs(self) -> list:
        return [
            HumanMessage("blah one"),
            AIMessage("a1"),
            HumanMessage("the KEEPME fact is important"),
            AIMessage("a2"),
            HumanMessage("blah three"),
            AIMessage("a3"),
            HumanMessage("recall the KEEPME thing"),  # current turn
        ]

    def test_retrieves_relevant_older_block_drops_rest(self) -> None:
        mw = InContextHistoryRetrievalMiddleware(
            keep_recent=1, top_k=1, embed_fn=_stub_embed
        )
        reset_turn_signals("t1")
        out = mw._select(make_request(self._msgs()))
        contents = [m.content for m in out.messages]
        self.assertIn("the KEEPME fact is important", contents)  # retrieved
        self.assertIn("recall the KEEPME thing", contents)  # current turn
        self.assertNotIn("blah one", contents)  # irrelevant, dropped
        self.assertNotIn("blah three", contents)
        self.assertEqual(out.messages[0].type, "human")  # no orphaned tool/ai
        signals = pop_turn_signals("t1")
        self.assertEqual(signals["history_retrievals"], 1)
        self.assertGreater(signals["dropped_messages"], 0)

    def test_no_op_when_nothing_old_enough(self) -> None:
        mw = InContextHistoryRetrievalMiddleware(
            keep_recent=2, top_k=3, embed_fn=_stub_embed
        )
        reset_turn_signals("t1")
        msgs = [HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2")]
        out = mw._select(make_request(msgs))
        self.assertEqual(len(out.messages), 3)
        self.assertEqual(pop_turn_signals("t1"), {})


class PresetResolutionTests(unittest.TestCase):
    NEW_PRESETS = (
        "observation_truncation",
        "sliding_window",
        "prompt_compression",
        "selective_retention",
        "context_reset",
        "clear_retrieval_kb",
        "incontext_history_retrieval",
    )

    def test_all_new_presets_resolve(self) -> None:
        for name in self.NEW_PRESETS:
            self.assertEqual(resolve_memory_preset(name).name, name)

    def test_axis_a_alternatives_disable_summarization(self) -> None:
        self.assertFalse(resolve_memory_preset("sliding_window").summarization)
        self.assertFalse(resolve_memory_preset("prompt_compression").summarization)

    def test_summary_prompt_variants_are_valid_templates(self) -> None:
        for name in ("selective_retention", "context_reset"):
            cfg = resolve_memory_preset(name)
            self.assertTrue(cfg.summarization)
            self.assertIsNotNone(cfg.summary_prompt)
            self.assertIn("{messages}", cfg.summary_prompt)
            # Only {messages} is a field, so .format must not raise.
            cfg.summary_prompt.format(messages="X")


class BuildMiddlewareTests(unittest.TestCase):
    def test_sliding_window_stack(self) -> None:
        mws = build_agent_middleware(
            model=None, memory_config=resolve_memory_preset("sliding_window")
        )
        names = {type(m).__name__ for m in mws}
        self.assertIn("SlidingWindowMiddleware", names)
        self.assertNotIn("SummarizationMiddleware", names)

    def test_prompt_compression_stack(self) -> None:
        mws = build_agent_middleware(
            model=None, memory_config=resolve_memory_preset("prompt_compression")
        )
        self.assertIn("PromptCompressionMiddleware", {type(m).__name__ for m in mws})

    def test_incontext_history_retrieval_stack(self) -> None:
        mws = build_agent_middleware(
            model=None,
            memory_config=resolve_memory_preset("incontext_history_retrieval"),
        )
        names = {type(m).__name__ for m in mws}
        self.assertIn("InContextHistoryRetrievalMiddleware", names)
        self.assertNotIn("SummarizationMiddleware", names)


class CompactionGateTests(unittest.TestCase):
    def test_recognizes_old_and_new_signals(self) -> None:
        self.assertTrue(compaction_active({"summary_messages": 2}))
        self.assertTrue(compaction_active({"cleared_tool_outputs": 1}))
        self.assertTrue(compaction_active({"dropped_messages": 3}))
        self.assertTrue(compaction_active({"truncated_tool_outputs": 1}))

    def test_no_signal_means_inactive(self) -> None:
        self.assertFalse(
            compaction_active({"summary_messages": 0, "dropped_messages": 0})
        )
        self.assertFalse(compaction_active({}))
        self.assertFalse(compaction_active(None))

    def test_app_and_eval_signal_lists_stay_in_sync(self) -> None:
        # Bidirectional: the per-call-view turn signals (app) must equal the eval
        # gate keys minus the two checkpoint-only markers. Pins both lists and
        # flags any phantom signal name that has no producer.
        checkpoint_markers = {"summary_messages", "cleared_tool_outputs"}
        self.assertEqual(
            set(COMPACTION_SIGNAL_NAMES),
            set(COMPACTION_SIGNAL_KEYS) - checkpoint_markers,
        )

    def test_signal_names_disjoint_from_reserved_stats(self) -> None:
        # Turn signals spread last into the context_stats event, so a name that
        # collided with a real metric would silently clobber it.
        reserved = {
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "est_cost_usd",
            "llm_calls",
            "ttft_ms",
            "total_ms",
            "summary_messages",
            "cleared_tool_outputs",
            "context_messages",
            "context_tokens_approx",
        }
        self.assertEqual(set(COMPACTION_SIGNAL_NAMES) & reserved, set())


class RetrievalBudgetTests(unittest.TestCase):
    def _retriever(self, default_budget: int) -> LocalChromaRetriever:
        retriever = LocalChromaRetriever.__new__(LocalChromaRetriever)
        retriever._encoding = tiktoken.get_encoding("cl100k_base")
        retriever._token_budget = default_budget
        return retriever

    def test_override_caps_results_below_default(self) -> None:
        retriever = self._retriever(10_000)
        results = [SimpleNamespace(score=0.9, content="word " * 50) for _ in range(3)]
        self.assertEqual(len(retriever._apply_token_budget(results)), 3)
        self.assertEqual(
            len(retriever._apply_token_budget(results, token_budget=60)), 1
        )

    def test_low_score_filtered(self) -> None:
        retriever = self._retriever(10_000)
        results = [SimpleNamespace(score=0.05, content="x")]
        self.assertEqual(retriever._apply_token_budget(results), [])


if __name__ == "__main__":
    unittest.main()
