from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .chat_types import ChatEvent

SOURCES_HEADER = "📝 Here are the sources I used to answer your question:"
ACTIVITY_BLOCK_START = "<!-- MODEL_ACTIVITY_START -->"
ACTIVITY_BLOCK_END = "<!-- MODEL_ACTIVITY_END -->"
THOUGHTS_HEADER = "**Thinking**"
THOUGHTS_HINT = "_Reasoning summary from Gemini. This is not the final answer._"
TOOL_HEADER = "**Tool**"
TOOL_PENDING_HINT = "_Searching the selected sources..._"
ANSWER_HEADER = "**Answer**"


@dataclass
class ActivityEvent:
    key: str
    kind: str
    body: str


def as_blockquote(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    return "\n".join("> " + line if line else ">" for line in lines)


def merge_stream_text(existing: str, incoming: str) -> str:
    current = existing.strip()
    update = incoming.strip()
    if not update:
        return current
    if not current:
        return update
    if update == current or update in current:
        return current
    if current in update:
        return update
    return f"{current}\n\n{update}"


def upsert_activity_event(
    events: list[ActivityEvent],
    *,
    key: str,
    kind: str,
    body: str,
    replace: bool = False,
) -> None:
    if not body.strip():
        return

    for index, event in enumerate(events):
        if event.key != key:
            continue
        events[index] = ActivityEvent(
            key=key,
            kind=kind,
            body=body.strip() if replace else merge_stream_text(event.body, body),
        )
        return

    events.append(ActivityEvent(key=key, kind=kind, body=body.strip()))


def format_tool_args(args: Any, args_text: str = "") -> str:
    query = ""
    url = ""
    if isinstance(args, dict):
        query = str(args.get("query", "")).strip()
        url = str(args.get("url", "")).strip()

    if not query and args_text.strip():
        query = args_text.strip()

    if query:
        trimmed = query[:157] + "..." if len(query) > 160 else query
        return f'Query: "{trimmed}"'

    if url:
        trimmed = url[:157] + "..." if len(url) > 160 else url
        return f"URL: `{trimmed}`"

    if isinstance(args, dict) and args:
        serialized = json.dumps(args, ensure_ascii=False, sort_keys=True)
        trimmed = serialized[:197] + "..." if len(serialized) > 200 else serialized
        return f"Args: `{trimmed}`"

    if args is None:
        return ""
    serialized = str(args).strip()
    if not serialized:
        return ""
    trimmed = serialized[:197] + "..." if len(serialized) > 200 else serialized
    return f"Args: `{trimmed}`"


def summarize_activity_sources(sources: list[str], *, max_items: int = 3) -> str:
    if not sources:
        return ""
    if len(sources) <= max_items:
        return ", ".join(sources)
    remaining = len(sources) - max_items
    return f"{', '.join(sources[:max_items])}, and {remaining} more"


def summarize_tool_result(event: ChatEvent) -> str:
    tool_name = str(event.data.get("tool_name", ""))
    matches = event.data.get("matches", [])
    match_count = len(matches)

    if tool_name == "retrieve_tutor_context":
        if not matches:
            return "_No matching sources found in the selected sources._"
        ordered_sources: list[str] = []
        seen_sources: set[str] = set()
        for match in matches:
            source_label = str(match.get("source_label", match.get("source_key", "unknown")))
            if source_label in seen_sources:
                continue
            seen_sources.add(source_label)
            ordered_sources.append(source_label)
        source_summary = summarize_activity_sources(ordered_sources)
        match_label = "match" if match_count == 1 else "matches"
        return f"_Found {match_count} {match_label} from {source_summary}._"

    output_text = str(event.data.get("output_text", "")).strip()
    if output_text:
        return f"_{output_text}_"

    if tool_name in ("google_search", "web_search"):
        if match_count == 0:
            return "_Search ran but returned no results._"
        label = "result" if match_count == 1 else "results"
        return f"_Searched the web — {match_count} {label} used as sources._"

    if tool_name in ("url_context", "web_fetch"):
        if match_count == 0:
            return "_URL fetched._"
        label = "page" if match_count == 1 else "pages"
        return f"_Fetched {match_count} {label}._"

    return "_Tool completed._"


def format_tool_event(
    tool_name: str,
    args: Any,
    status_line: str,
    *,
    args_text: str = "",
) -> str:
    lines = [f"Using `{tool_name}`"]
    args_line = format_tool_args(args, args_text=args_text)
    if args_line:
        lines.append(args_line)
    lines.append(status_line)
    return "\n".join(lines)


def render_activity_block(events: list[ActivityEvent]) -> str:
    sections: list[str] = []
    for event in events:
        if event.kind == "thinking":
            sections.append(
                "\n".join(
                    [
                        THOUGHTS_HEADER,
                        THOUGHTS_HINT,
                        "",
                        as_blockquote(event.body),
                    ]
                ).strip()
            )
            continue
        if event.kind == "tool":
            sections.append(f"{TOOL_HEADER}\n{event.body}".strip())

    if not sections:
        return ""

    rendered_sections = "\n\n".join(section for section in sections if section)
    return (
        f"{ACTIVITY_BLOCK_START}\n\n"
        f"{rendered_sections}\n\n"
        f"{ACTIVITY_BLOCK_END}"
    )


def format_sources(matches_by_doc_id: dict[str, dict[str, Any]]) -> str:
    if not matches_by_doc_id:
        return ""

    lines = [SOURCES_HEADER]
    sorted_matches = sorted(
        matches_by_doc_id.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    for match in sorted_matches:
        lines.append(
            f"- [🔗 {match['source_label']}: {match['title']}]({match['url']}), relevance: {match['score']:.2f}"
        )
    return "\n".join(lines)


def render_output(
    answer: str,
    activity_block: str = "",
    sources_block: str = "",
) -> str:
    visible_answer = answer.strip()
    visible_activity = activity_block.strip()
    visible_sources = sources_block.strip()

    output_parts = [visible_activity]
    if visible_answer:
        if visible_activity:
            output_parts.append(f"{ANSWER_HEADER}\n\n{visible_answer}")
        else:
            output_parts.append(visible_answer)
    output_parts.append(visible_sources)
    return "\n\n".join(part for part in output_parts if part)


@dataclass
class GradioPresenterState:
    show_activity: bool = False
    thread_id: str = ""
    message_completed: bool = False
    matches_by_doc_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    activity_events: list[ActivityEvent] = field(default_factory=list)
    answer_chunks: list[str] = field(default_factory=list)
    completed_answer: str = ""
    tool_matches_by_call_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def apply(self, event: ChatEvent) -> None:
        if event.type == "thread_started":
            self.thread_id = str(event.data.get("thread_id", self.thread_id))
            return

        if event.type == "text_delta":
            text = str(event.data.get("text", ""))
            if text:
                self.answer_chunks.append(text)
            return

        if event.type == "reasoning_delta":
            if not self.show_activity:
                return
            step = str(event.data.get("step", ""))
            body = str(event.data.get("text", ""))
            upsert_activity_event(
                self.activity_events,
                key=f"thinking:{step}",
                kind="thinking",
                body=body,
            )
            return

        if event.type == "tool_call_started":
            if not self.show_activity:
                return
            call_id = str(event.data.get("call_id", ""))
            upsert_activity_event(
                self.activity_events,
                key=f"tool:{call_id}",
                kind="tool",
                body=format_tool_event(
                    str(event.data.get("tool_name", "tool")),
                    event.data.get("args"),
                    TOOL_PENDING_HINT,
                    args_text=str(event.data.get("args_text", "")),
                ),
                replace=True,
            )
            return

        if event.type == "source_match":
            doc_id = str(event.data.get("doc_id", ""))
            existing = self.matches_by_doc_id.get(doc_id)
            incoming_score = float(event.data.get("score", 0.0))
            if existing and float(existing["score"]) >= incoming_score:
                return
            source_data = {
                "title": str(event.data.get("title", "")),
                "url": str(event.data.get("url", "")),
                "source_label": str(event.data.get("source_label", "")),
                "score": incoming_score,
            }
            self.matches_by_doc_id[doc_id] = source_data

            call_id = str(event.data.get("call_id", ""))
            if call_id:
                self.tool_matches_by_call_id.setdefault(call_id, []).append(source_data)
            return

        if event.type == "tool_call_completed":
            if not self.show_activity:
                return
            call_id = str(event.data.get("call_id", ""))
            matches = self.tool_matches_by_call_id.get(call_id, [])
            event_with_matches = ChatEvent(
                type=event.type,
                data={**event.data, "matches": matches},
            )
            upsert_activity_event(
                self.activity_events,
                key=f"tool:{call_id}",
                kind="tool",
                body=format_tool_event(
                    str(event.data.get("tool_name", "tool")),
                    event.data.get("args"),
                    summarize_tool_result(event_with_matches),
                    args_text=str(event.data.get("args_text", "")),
                ),
                replace=True,
            )
            return

        if event.type == "message_completed":
            self.message_completed = True
            self.thread_id = str(event.data.get("thread_id", self.thread_id))
            self.completed_answer = str(event.data.get("answer", "")).strip()

    def render(self) -> str:
        answer = "".join(self.answer_chunks).strip() or self.completed_answer
        activity_block = (
            render_activity_block(self.activity_events) if self.show_activity else ""
        )
        sources_block = (
            format_sources(self.matches_by_doc_id) if self.message_completed else ""
        )
        return render_output(answer, activity_block, sources_block)
