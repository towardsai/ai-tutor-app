from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts.agent_tracing import (
    DEFAULT_LANGSMITH_PROJECT,
    configure_langsmith_environment,
    langsmith_tracing_enabled,
    parse_env_bool,
)


class AgentTracingTestCase(unittest.TestCase):
    def test_parse_env_bool(self) -> None:
        self.assertIs(parse_env_bool(None), None)
        self.assertTrue(parse_env_bool("true"))
        self.assertTrue(parse_env_bool("1"))
        self.assertFalse(parse_env_bool("false"))
        self.assertFalse(parse_env_bool("0"))
        self.assertIs(parse_env_bool("maybe"), None)

    def test_api_key_enables_langsmith_by_default(self) -> None:
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "ls_test"}, clear=True):
            configure_langsmith_environment()

            self.assertTrue(langsmith_tracing_enabled())
            self.assertEqual(os.environ["LANGSMITH_TRACING"], "true")
            self.assertEqual(os.environ["LANGSMITH_PROJECT"], DEFAULT_LANGSMITH_PROJECT)

    def test_explicit_disable_wins_over_api_key(self) -> None:
        with patch.dict(
            os.environ,
            {"LANGSMITH_API_KEY": "ls_test", "LANGSMITH_TRACING": "false"},
            clear=True,
        ):
            configure_langsmith_environment()

            self.assertFalse(langsmith_tracing_enabled())
            self.assertNotIn("LANGSMITH_PROJECT", os.environ)

    def test_project_is_preserved(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LANGSMITH_API_KEY": "ls_test",
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_PROJECT": "custom-project",
            },
            clear=True,
        ):
            configure_langsmith_environment()

            self.assertEqual(os.environ["LANGSMITH_PROJECT"], "custom-project")


if __name__ == "__main__":
    unittest.main()
