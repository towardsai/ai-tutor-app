from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from app.chat_service import (
    SourcePreferenceMiddleware,
    StudentProfileMiddleware,
    build_agent,
    build_agent_middleware,
)
from app.memory_presets import (
    DEFAULT_MEMORY_PRESET,
    MEMORY_PRESETS,
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
    def test_prod_preset_matches_production_constants(self) -> None:
        prod = MEMORY_PRESETS["prod"]
        self.assertTrue(prod.summarization)
        self.assertEqual(prod.summarization_trigger_tokens, 30_000)
        self.assertEqual(prod.summarization_keep_messages, 20)
        self.assertTrue(prod.context_editing)
        self.assertEqual(prod.context_editing_trigger_tokens, 5_000)
        self.assertEqual(prod.context_editing_keep, 5)
        self.assertFalse(prod.longterm_memory)

    def test_default_resolution(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("AI_TUTOR_MEMORY_PRESET", None)
            self.assertEqual(resolve_memory_preset("").name, DEFAULT_MEMORY_PRESET)
            self.assertEqual(resolve_memory_preset(None).name, DEFAULT_MEMORY_PRESET)

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
