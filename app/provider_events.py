"""Provider-specific response parsing for the chat stream.

Gemini and Anthropic surface server-side tool activity (web search, URL
fetch) and reasoning through provider-shaped response metadata and content
blocks rather than regular tool messages. This module turns those payloads
into the app's neutral `ChatEvent` / `SourceMatch` shapes so
`chat_service.stream_chat` stays provider-agnostic.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from .chat_types import ChatEvent, SourceMatch


def extract_thought_summaries(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []

    thoughts: list[str] = []
    for item in content:
        if not hasattr(item, "get"):
            continue

        item_type = item.get("type")
        if item_type == "thinking":
            thought = str(item.get("thinking", "")).strip()
        elif item_type == "reasoning":
            thought = str(item.get("reasoning", "")).strip()
        else:
            continue

        if thought:
            thoughts.append(thought)
    return thoughts


def extract_web_search_queries(response_metadata: Any) -> list[str]:
    """Pull the queries Gemini ran against google_search from grounding metadata."""
    if not isinstance(response_metadata, dict):
        return []
    grounding = response_metadata.get("grounding_metadata") or {}
    queries = grounding.get("web_search_queries") or []
    return [str(q).strip() for q in queries if isinstance(q, str) and str(q).strip()]


def extract_grounding_source_matches(
    response_metadata: Any,
    matches_by_doc_id: dict[str, SourceMatch],
) -> list[SourceMatch]:
    """Turn Gemini grounding metadata into source matches (deduped by URI)."""
    if not isinstance(response_metadata, dict):
        return []
    grounding = response_metadata.get("grounding_metadata") or {}
    chunks = grounding.get("grounding_chunks") or []
    if not chunks:
        return []

    confidence_by_index: dict[int, float] = {}
    for support in grounding.get("grounding_supports") or []:
        indices = support.get("grounding_chunk_indices") or []
        scores = support.get("confidence_scores") or []
        for idx, score in zip(indices, scores):
            if not isinstance(idx, int):
                continue
            numeric = float(score) if isinstance(score, (int, float)) else 0.0
            if numeric > confidence_by_index.get(idx, 0.0):
                confidence_by_index[idx] = numeric

    updated: list[SourceMatch] = []
    for idx, chunk in enumerate(chunks):
        web = (chunk or {}).get("web") or {}
        uri = str(web.get("uri") or "").strip()
        if not uri:
            continue
        title = str(web.get("title") or uri).strip()
        doc_id = f"google_search::{uri}"
        if doc_id in matches_by_doc_id:
            continue
        score = confidence_by_index.get(idx, 1.0)
        source_match = SourceMatch(
            doc_id=doc_id,
            title=title,
            url=uri,
            source_key="google_search",
            source_label="Web",
            score=score,
            group="web",
        )
        matches_by_doc_id[doc_id] = source_match
        updated.append(source_match)
    return updated


GOOGLE_SEARCH_TOOL_NAME = "google_search"


class GoogleSearchActivity:
    """Surface Gemini's server-side google_search activity as tool events.

    Gemini reports search grounding via response metadata instead of tool
    messages, so queries and grounding results are accumulated from every
    metadata payload and exposed as a single synthetic tool call per turn.
    """

    def __init__(self, message_id: str, web_evidence: dict[str, SourceMatch]) -> None:
        self._message_id = message_id
        self._web_evidence = web_evidence
        self._call_id = ""
        self._queries: list[str] = []
        self._match_count = 0

    def observe(self, response_metadata: Any) -> ChatEvent | None:
        """Record metadata; return a tool_call_started event on first activity."""
        new_queries = [
            q
            for q in extract_web_search_queries(response_metadata)
            if q not in self._queries
        ]
        new_grounding = extract_grounding_source_matches(
            response_metadata,
            self._web_evidence,
        )
        started: ChatEvent | None = None
        if (new_queries or new_grounding) and not self._call_id:
            self._call_id = uuid4().hex
            joined = "; ".join(new_queries)
            started = ChatEvent(
                "tool_call_started",
                {
                    "message_id": self._message_id,
                    "call_id": self._call_id,
                    "tool_name": GOOGLE_SEARCH_TOOL_NAME,
                    "args": {"query": joined},
                    "args_text": joined,
                },
            )
        self._queries.extend(new_queries)
        self._match_count += len(new_grounding)
        return started

    def completed_event(self) -> ChatEvent | None:
        if not self._call_id:
            return None
        joined = "; ".join(self._queries)
        if self._match_count == 0:
            output_text = "Google search ran but returned no grounding results."
        else:
            plural = "" if self._match_count == 1 else "s"
            output_text = (
                f"Google search returned {self._match_count} web result{plural}."
            )
        return ChatEvent(
            "tool_call_completed",
            {
                "message_id": self._message_id,
                "call_id": self._call_id,
                "tool_name": GOOGLE_SEARCH_TOOL_NAME,
                "args": {"query": joined},
                "args_text": joined,
                "output_text": output_text,
            },
        )


ANTHROPIC_SERVER_TOOL_NAMES = frozenset({"web_search", "web_fetch"})
ANTHROPIC_RESULT_BLOCK_TYPES = {
    "web_search_tool_result": ("web_search", "Web"),
    "web_fetch_tool_result": ("web_fetch", "Web page"),
}


def extract_anthropic_source_matches(
    content: Any,
    matches_by_doc_id: dict[str, SourceMatch],
) -> tuple[dict[str, list[SourceMatch]], dict[str, dict[str, Any]]]:
    """Parse Claude's server-side web tool invocations and their results.

    Scans ``message.content`` for three kinds of blocks emitted when Claude
    runs the built-in ``web_search`` / ``web_fetch`` tools:

    * ``tool_use`` — the model's call (id, name, input args)
    * ``web_search_tool_result`` / ``web_fetch_tool_result`` — the server's
      response, keyed by ``tool_use_id``
    * ``text`` blocks with ``citations`` — fallback for citations without a
      matching result block

    Returns ``(matches_by_tool_use_id, tool_use_index)`` where
    ``tool_use_index`` maps tool_use id → ``{"name", "args"}`` so the caller
    can emit ``tool_call_started`` events with the right metadata.
    ``langchain-anthropic`` does not always surface server-side tool_use in
    ``AIMessage.tool_calls``, so we read them off the content blocks directly.
    """
    if not isinstance(content, list):
        return {}, {}

    updates: dict[str, list[SourceMatch]] = {}
    tool_use_index: dict[str, dict[str, Any]] = {}

    for block in content:
        if not hasattr(block, "get"):
            continue

        block_type = block.get("type")

        if block_type in ("server_tool_use", "tool_use"):
            tool_use_id = str(block.get("id") or "")
            tool_name = str(block.get("name") or "")
            if tool_use_id and tool_name in ANTHROPIC_SERVER_TOOL_NAMES:
                args = block.get("input") or {}
                if not args:
                    partial = block.get("partial_json")
                    if isinstance(partial, str) and partial.strip():
                        try:
                            parsed = json.loads(partial)
                        except json.JSONDecodeError:
                            parsed = None
                        if isinstance(parsed, dict):
                            args = parsed
                tool_use_index[tool_use_id] = {
                    "id": tool_use_id,
                    "name": tool_name,
                    "args": args,
                }
            continue

        mapping = ANTHROPIC_RESULT_BLOCK_TYPES.get(block_type)
        if mapping:
            source_key, source_label = mapping
            tool_use_id = str(block.get("tool_use_id") or "")
            results = block.get("content") or []
            if not isinstance(results, list):
                continue
            for result in results:
                if not hasattr(result, "get"):
                    continue
                url = str(result.get("url") or "").strip()
                if not url:
                    continue
                title = str(result.get("title") or url).strip()
                doc_id = f"{source_key}::{url}"
                if doc_id in matches_by_doc_id:
                    continue
                source_match = SourceMatch(
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    source_key=source_key,
                    source_label=source_label,
                    score=1.0,
                    group="web",
                )
                matches_by_doc_id[doc_id] = source_match
                updates.setdefault(tool_use_id, []).append(source_match)
            continue

        if block_type == "text":
            for citation in block.get("citations") or []:
                if not hasattr(citation, "get"):
                    continue
                url = str(citation.get("url") or "").strip()
                if not url:
                    continue
                title = str(citation.get("title") or url).strip()
                doc_id = f"web_search::{url}"
                if doc_id in matches_by_doc_id:
                    continue
                source_match = SourceMatch(
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    source_key="web_search",
                    source_label="Web",
                    score=1.0,
                    group="web",
                )
                matches_by_doc_id[doc_id] = source_match
                updates.setdefault("", []).append(source_match)

    return updates, tool_use_index
