from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from app.chat_service import (
    DeepSeekCacheIsolationMiddleware,
    InstrumentedSummarizationMiddleware,
    PrefixPreservingCompactionMiddleware,
    SourcePreferenceMiddleware,
    StableToolOutputCapMiddleware,
    StudentProfileMiddleware,
    build_agent,
    build_agent_middleware,
)
from app.memory_presets import (
    DEFAULT_MEMORY_PRESET,
    MEMORY_PRESETS,
    PRODUCTION_FALLBACK_MEMORY_PRESET,
    PRODUCTION_MEMORY_PRESET,
    memory_preset_supports_model,
    production_memory_preset_name,
    resolve_memory_preset,
)
from langchain.agents.middleware import (
    ContextEditingMiddleware,
    SummarizationMiddleware,
)


def _stub_model():
    # SummarizationMiddleware reads model._llm_type at init.
    return types.SimpleNamespace(_llm_type="fake-chat-model")


class MemoryPresetResolutionTests(unittest.TestCase):
    def test_legacy_prod_preset_remains_immutable(self) -> None:
        prod = MEMORY_PRESETS["prod"]
        self.assertTrue(prod.summarization)
        self.assertEqual(prod.summarization_trigger_tokens, 30_000)
        self.assertEqual(prod.summarization_keep_messages, 20)
        self.assertTrue(prod.context_editing)
        self.assertEqual(prod.context_editing_trigger_tokens, 5_000)
        self.assertEqual(prod.context_editing_keep, 5)
        self.assertFalse(prod.longterm_memory)

    def test_prod_v2_matches_structured_800k_policy(self) -> None:
        prod_v2 = MEMORY_PRESETS["prod_v2"]
        self.assertEqual(prod_v2.summarization_trigger_tokens, 800_000)
        self.assertEqual(prod_v2.summarization_keep_tokens, 50_000)
        self.assertIsNone(prod_v2.summarization_trim_tokens)
        self.assertEqual(prod_v2.summarization_input_guard_tokens, 900_000)
        self.assertEqual(prod_v2.summarization_strategy, "structured_prefix")
        self.assertFalse(prod_v2.context_editing)
        self.assertEqual(prod_v2.tool_output_cap_bytes, 40_000)
        self.assertTrue(prod_v2.experiment_mode)
        self.assertEqual(prod_v2.experiment_request_guard_tokens, 990_000)

    def test_default_resolution(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("AI_TUTOR_MEMORY_PRESET", None)
            self.assertEqual(resolve_memory_preset("").name, DEFAULT_MEMORY_PRESET)
            self.assertEqual(resolve_memory_preset(None).name, DEFAULT_MEMORY_PRESET)
            self.assertEqual(DEFAULT_MEMORY_PRESET, PRODUCTION_MEMORY_PRESET)
            self.assertEqual(
                resolve_memory_preset(
                    None, model_name="deepseek:deepseek-v4-flash"
                ).name,
                "prod_v2",
            )
            self.assertEqual(
                resolve_memory_preset(
                    None, model_name="google-genai:gemini-2.5-flash"
                ).name,
                "prod_v2",
            )
            self.assertEqual(
                resolve_memory_preset(
                    None, model_name="anthropic:claude-haiku-4-5"
                ).name,
                "prod",
            )
            self.assertEqual(
                production_memory_preset_name("openai:gpt-5.6"),
                PRODUCTION_FALLBACK_MEMORY_PRESET,
            )

    def test_prod_v2_rejects_short_context_providers(self) -> None:
        self.assertTrue(
            memory_preset_supports_model("prod_v2", "deepseek:deepseek-v4-flash")
        )
        self.assertTrue(
            memory_preset_supports_model("prod_v2", "google-genai:gemini-2.5-flash")
        )
        self.assertFalse(
            memory_preset_supports_model("prod_v2", "anthropic:claude-haiku-4-5")
        )
        with self.assertRaisesRegex(ValueError, "does not support provider"):
            resolve_memory_preset("prod_v2", model_name="anthropic:claude-haiku-4-5")

    def test_env_var_default_and_explicit_override(self) -> None:
        with patch.dict("os.environ", {"AI_TUTOR_MEMORY_PRESET": "full_history"}):
            self.assertEqual(resolve_memory_preset("").name, "full_history")
            # An explicit request value beats the env default.
            self.assertEqual(resolve_memory_preset("aggressive").name, "aggressive")

    def test_unknown_preset_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_memory_preset("does_not_exist")
        with patch.dict("os.environ", {"AI_TUTOR_MEMORY_PRESET": "typo"}):
            with self.assertRaises(ValueError):
                resolve_memory_preset("")

    def test_deepseek_experiment_arms_are_single_axis_configs(self) -> None:
        raw = MEMORY_PRESETS["exp_fh_raw"]
        capped = MEMORY_PRESETS["exp_fh_cap10k"]
        compact = MEMORY_PRESETS["exp_c200_cap10k"]
        self.assertTrue(raw.experiment_mode)
        self.assertFalse(raw.summarization)
        self.assertFalse(raw.context_editing)
        self.assertEqual(capped.tool_output_cap_bytes, 40_000)
        self.assertEqual(compact.summarization_trigger_tokens, 200_000)
        self.assertEqual(compact.summarization_keep_tokens, 50_000)
        self.assertIsNone(compact.summarization_trim_tokens)
        self.assertEqual(compact.summarization_input_guard_tokens, 900_000)
        self.assertEqual(compact.experiment_request_guard_tokens, 990_000)
        self.assertFalse(compact.context_editing)
        structured = MEMORY_PRESETS["exp_c200_cap10k_structured"]
        self.assertEqual(structured.summarization_strategy, "structured_prefix")
        self.assertEqual(structured.summarization_trigger_tokens, 200_000)
        self.assertEqual(structured.summarization_keep_tokens, 50_000)
        self.assertEqual(structured.tool_output_cap_bytes, 40_000)


class MiddlewareAssemblyTests(unittest.TestCase):
    def test_full_history_disables_compaction(self) -> None:
        middleware = build_agent_middleware(
            _stub_model(), MEMORY_PRESETS["full_history"]
        )
        self.assertEqual(len(middleware), 1)
        self.assertIsInstance(middleware[0], SourcePreferenceMiddleware)

    def test_prod_has_editing_then_summarization(self) -> None:
        middleware = build_agent_middleware(_stub_model(), MEMORY_PRESETS["prod"])
        self.assertIsInstance(middleware[0], ContextEditingMiddleware)
        self.assertIsInstance(middleware[1], SummarizationMiddleware)
        self.assertIsInstance(middleware[-1], SourcePreferenceMiddleware)
        self.assertEqual(len(middleware), 3)

    def test_single_technique_presets(self) -> None:
        summarization_only = build_agent_middleware(
            _stub_model(), MEMORY_PRESETS["summarization_only"]
        )
        self.assertFalse(
            any(isinstance(m, ContextEditingMiddleware) for m in summarization_only)
        )
        self.assertTrue(
            any(isinstance(m, SummarizationMiddleware) for m in summarization_only)
        )
        editing_only = build_agent_middleware(
            _stub_model(), MEMORY_PRESETS["editing_only"]
        )
        self.assertTrue(
            any(isinstance(m, ContextEditingMiddleware) for m in editing_only)
        )
        self.assertFalse(
            any(isinstance(m, SummarizationMiddleware) for m in editing_only)
        )

    def test_profile_memory_adds_student_profile_middleware(self) -> None:
        middleware = build_agent_middleware(
            _stub_model(), MEMORY_PRESETS["profile_memory"]
        )
        self.assertTrue(
            any(isinstance(m, StudentProfileMiddleware) for m in middleware)
        )

    def test_experiment_stack_has_isolation_cap_and_full_input_summary(self) -> None:
        middleware = build_agent_middleware(
            _stub_model(), MEMORY_PRESETS["exp_c200_cap10k"]
        )
        self.assertTrue(
            any(isinstance(m, DeepSeekCacheIsolationMiddleware) for m in middleware)
        )
        self.assertTrue(
            any(isinstance(m, StableToolOutputCapMiddleware) for m in middleware)
        )
        summary = next(
            m for m in middleware if isinstance(m, InstrumentedSummarizationMiddleware)
        )
        self.assertEqual(summary.keep, ("tokens", 50_000))
        self.assertIsNone(summary.trim_tokens_to_summarize)
        self.assertEqual(summary.summary_input_guard_tokens, 900_000)
        self.assertFalse(
            any(isinstance(m, ContextEditingMiddleware) for m in middleware)
        )

    def test_structured_compactor_runs_after_request_shaping_middleware(self) -> None:
        middleware = build_agent_middleware(
            _stub_model(), MEMORY_PRESETS["exp_c200_cap10k_structured"]
        )
        self.assertIsInstance(middleware[0], DeepSeekCacheIsolationMiddleware)
        self.assertTrue(
            any(isinstance(m, StableToolOutputCapMiddleware) for m in middleware)
        )
        self.assertIsInstance(middleware[-2], SourcePreferenceMiddleware)
        self.assertIsInstance(middleware[-1], PrefixPreservingCompactionMiddleware)
        self.assertFalse(
            any(type(m) is InstrumentedSummarizationMiddleware for m in middleware)
        )

    def test_prod_v2_uses_the_structured_experiment_stack(self) -> None:
        middleware = build_agent_middleware(_stub_model(), MEMORY_PRESETS["prod_v2"])
        self.assertEqual(
            [type(item) for item in middleware],
            [
                DeepSeekCacheIsolationMiddleware,
                StableToolOutputCapMiddleware,
                SourcePreferenceMiddleware,
                PrefixPreservingCompactionMiddleware,
            ],
        )
        compactor = middleware[-1]
        self.assertEqual(compactor.trigger, ("tokens", 800_000))
        self.assertEqual(compactor.keep, ("tokens", 50_000))
        self.assertIsNone(compactor.trim_tokens_to_summarize)
        self.assertEqual(compactor.summary_input_guard_tokens, 900_000)

    def test_build_agent_cache_keys_include_memory_config(self) -> None:
        build_agent.cache_clear()
        created = []

        def fake_create_agent(**kwargs):
            agent = types.SimpleNamespace(kwargs=kwargs)
            created.append(agent)
            return agent

        try:
            with (
                patch(
                    "app.chat_service.build_chat_model",
                    return_value=_stub_model(),
                ),
                patch("app.chat_service.build_system_prompt", return_value="prompt"),
                patch("app.chat_service.create_agent", side_effect=fake_create_agent),
            ):
                prod = build_agent(
                    "google-genai:gemini-3.5-flash",
                    memory_config=MEMORY_PRESETS["prod"],
                )
                full_history = build_agent(
                    "google-genai:gemini-3.5-flash",
                    memory_config=MEMORY_PRESETS["full_history"],
                )
                full_history_again = build_agent(
                    "google-genai:gemini-3.5-flash",
                    memory_config=MEMORY_PRESETS["full_history"],
                )
        finally:
            build_agent.cache_clear()

        self.assertIsNot(prod, full_history)
        self.assertIs(full_history, full_history_again)
        self.assertEqual(len(created), 2)
        # Long-term memory needs the store wired into every agent.
        self.assertIn("store", created[0].kwargs)


if __name__ == "__main__":
    unittest.main()
