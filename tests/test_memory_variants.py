"""Tests for the Part C memory/context variants and their wiring.

Covers the per-call-view middlewares (sliding window, observation truncation,
prompt compression), preset resolution + middleware assembly, the generalized
compaction gate, and the per-request retrieval-budget override. All offline:
no model client, no API keys, no vector DB.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.chat_service import (
    InContextHistoryRetrievalMiddleware,
    ObservationTruncationMiddleware,
    PromptCompressionMiddleware,
    SlidingWindowMiddleware,
    build_agent_middleware,
)
from app.chroma_rag import LocalChromaRetriever
from app.memory_presets import resolve_memory_preset
from app.telemetry import COMPACTION_SIGNAL_NAMES, pop_turn_signals, reset_turn_signals
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
