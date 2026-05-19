from __future__ import annotations

import os

DEFAULT_LANGSMITH_PROJECT = "ai-tutor-app"

TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def parse_env_bool(value: str | None) -> bool | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def langsmith_tracing_enabled() -> bool:
    explicit = parse_env_bool(os.getenv("LANGSMITH_TRACING"))
    if explicit is not None:
        return explicit
    return bool(os.getenv("LANGSMITH_API_KEY"))


def configure_langsmith_environment() -> None:
    """Apply app defaults for LangSmith without requiring code changes in deploys."""
    if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING") is None:
        os.environ["LANGSMITH_TRACING"] = "true"

    if not langsmith_tracing_enabled():
        return

    if not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = DEFAULT_LANGSMITH_PROJECT
