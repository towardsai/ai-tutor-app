from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.chat_service import CLEARED_TOOL_OUTPUT_PLACEHOLDER
from app.telemetry import (
    TurnUsageHandler,
    context_window_stats,
    estimate_cost_usd,
    pricing_for_model,
    usage_totals,
)


class UsageTotalsTests(unittest.TestCase):
    def test_sums_across_models_including_cache_details(self) -> None:
        usage = {
            "gemini-3.5-flash": {
                "input_tokens": 1_000,
                "output_tokens": 200,
                "total_tokens": 1_200,
                "input_token_details": {"cache_read": 400},
            },
            "claude-haiku-4-5-20251001": {
                "input_tokens": 500,
                "output_tokens": 100,
                "total_tokens": 600,
                "input_token_details": {"cache_read": 50, "cache_creation": 25},
            },
        }
        totals = usage_totals(usage)
        self.assertEqual(totals["input_tokens"], 1_500)
        self.assertEqual(totals["output_tokens"], 300)
        self.assertEqual(totals["total_tokens"], 1_800)
        self.assertEqual(totals["cache_read_tokens"], 450)
        self.assertEqual(totals["cache_creation_tokens"], 25)

    def test_missing_details_default_to_zero(self) -> None:
        totals = usage_totals({"m": {"input_tokens": 10, "output_tokens": 5}})
        self.assertEqual(totals["cache_read_tokens"], 0)
        self.assertEqual(totals["total_tokens"], 0)


class CostEstimateTests(unittest.TestCase):
    def test_dated_model_name_matches_family_pricing(self) -> None:
        self.assertIsNotNone(pricing_for_model("claude-haiku-4-5-20251001"))
        self.assertIsNone(pricing_for_model("some-unknown-model"))

    def test_cache_tokens_priced_separately(self) -> None:
        # claude-haiku-4-5: input $1, output $5, cache read $0.10, write $1.25
        # per MTok. input_tokens includes the cached buckets, so the plain
        # bucket is 1M - 400k - 100k = 500k.
        usage = {
            "claude-haiku-4-5-20251001": {
                "input_tokens": 1_000_000,
                "output_tokens": 200_000,
                "input_token_details": {
                    "cache_read": 400_000,
                    "cache_creation": 100_000,
                },
            }
        }
        expected = 0.5 * 1.00 + 0.4 * 0.10 + 0.1 * 1.25 + 0.2 * 5.00
        self.assertAlmostEqual(estimate_cost_usd(usage), expected)

    def test_cache_creation_without_write_rate_bills_as_input(self) -> None:
        # gemini-3.5-flash has no cache_write entry.
        usage = {
            "gemini-3.5-flash": {
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "input_token_details": {"cache_creation": 200_000},
            }
        }
        expected = 0.8 * 0.30 + 0.2 * 0.30
        self.assertAlmostEqual(estimate_cost_usd(usage), expected)

    def test_unknown_model_yields_none_not_zero(self) -> None:
        self.assertIsNone(estimate_cost_usd({"mystery-model": {"input_tokens": 10}}))
        mixed = {
            "gemini-3.5-flash": {"input_tokens": 10, "output_tokens": 1},
            "mystery-model": {"input_tokens": 10, "output_tokens": 1},
        }
        self.assertIsNone(estimate_cost_usd(mixed))


class ContextWindowStatsTests(unittest.TestCase):
    def test_counts_summaries_and_cleared_tool_outputs(self) -> None:
        messages = [
            HumanMessage(
                content="Here is a summary of the conversation to date: ...",
                additional_kwargs={"lc_source": "summarization"},
            ),
            HumanMessage(content="What is RAG?"),
            AIMessage(content="RAG is retrieval-augmented generation."),
            ToolMessage(
                content=CLEARED_TOOL_OUTPUT_PLACEHOLDER,
                tool_call_id="call_1",
            ),
            ToolMessage(content="$ ls\nwiki", tool_call_id="call_2"),
        ]
        stats = context_window_stats(messages, CLEARED_TOOL_OUTPUT_PLACEHOLDER)
        self.assertEqual(stats["context_messages"], 5)
        self.assertEqual(stats["summary_messages"], 1)
        self.assertEqual(stats["cleared_tool_outputs"], 1)
        self.assertGreater(stats["context_tokens_approx"], 0)

    def test_empty_context(self) -> None:
        stats = context_window_stats([], CLEARED_TOOL_OUTPUT_PLACEHOLDER)
        self.assertEqual(stats["context_messages"], 0)
        self.assertEqual(stats["context_tokens_approx"], 0)


class TurnUsageHandlerTests(unittest.TestCase):
    def test_counts_calls_and_aggregates_usage(self) -> None:
        handler = TurnUsageHandler()

        def result(input_tokens: int, output_tokens: int) -> LLMResult:
            message = AIMessage(
                content="ok",
                usage_metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                response_metadata={"model_name": "gemini-3.5-flash"},
            )
            return LLMResult(generations=[[ChatGeneration(message=message)]])

        handler.on_llm_end(result(100, 20))
        handler.on_llm_end(result(50, 10))

        self.assertEqual(handler.llm_calls, 2)
        usage = handler.usage_metadata["gemini-3.5-flash"]
        self.assertEqual(usage["input_tokens"], 150)
        self.assertEqual(usage["output_tokens"], 30)


if __name__ == "__main__":
    unittest.main()
