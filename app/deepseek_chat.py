"""DeepSeek chat-model compatibility helpers.

``ChatDeepSeek`` preserves the provider's streamed ``reasoning_content`` in
LangChain message metadata.  DeepSeek thinking-mode tool loops additionally
require that metadata to be serialized back onto assistant tool-call messages;
the upstream integration does not currently do that replay step.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek


class TutorChatDeepSeek(ChatDeepSeek):
    """ChatDeepSeek with reasoning replay for thinking-mode tool calls."""

    @staticmethod
    def _thinking_enabled(payload: dict[str, Any]) -> bool:
        extra_body = payload.get("extra_body")
        if not isinstance(extra_body, dict):
            return False
        thinking = extra_body.get("thinking")
        return isinstance(thinking, dict) and thinking.get("type") == "enabled"

    def _original_messages(self, input_: LanguageModelInput) -> list[BaseMessage]:
        return self._convert_input(input_).to_messages()

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if not self._thinking_enabled(payload):
            return payload

        original_messages = self._original_messages(input_)
        for index, message in enumerate(payload.get("messages") or []):
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            reasoning_content = ""
            if index < len(original_messages):
                value = original_messages[index].additional_kwargs.get(
                    "reasoning_content"
                )
                if isinstance(value, str):
                    reasoning_content = value
            # DeepSeek requires the top-level field to exist on every replayed
            # assistant tool-call message in thinking mode. An empty value is
            # still preferable to omitting the field and receiving a 400.
            message["reasoning_content"] = reasoning_content
        return payload
