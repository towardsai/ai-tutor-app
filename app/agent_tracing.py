from __future__ import annotations

import os

DEFAULT_LANGSMITH_PROJECT = "ai-tutor-app"
DEFAULT_LOCAL_DEPLOYMENT_ENVIRONMENT = "local"
DEFAULT_HF_DEPLOYMENT_ENVIRONMENT = "huggingface-space"
LOCAL_DEPLOYMENT_HOST = "local"

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


def langsmith_deployment_identity() -> tuple[str, str]:
    """Return stable environment and host labels for LangSmith traces."""
    space_host = os.getenv("SPACE_HOST", "").strip()
    environment = os.getenv("AI_TUTOR_DEPLOYMENT_ENV", "").strip()
    if not environment:
        environment = (
            DEFAULT_HF_DEPLOYMENT_ENVIRONMENT
            if space_host
            else DEFAULT_LOCAL_DEPLOYMENT_ENVIRONMENT
        )
    return environment, space_host or LOCAL_DEPLOYMENT_HOST


def configure_langsmith_environment() -> None:
    """Apply app defaults for LangSmith without requiring code changes in deploys."""
    if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_TRACING") is None:
        os.environ["LANGSMITH_TRACING"] = "true"

    if not langsmith_tracing_enabled():
        return

    if not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = DEFAULT_LANGSMITH_PROJECT
