from __future__ import annotations

from app.provider_events import (
    GoogleSearchActivity,
    extract_anthropic_source_matches,
)


def _grounding_metadata(query: str, uri: str, title: str) -> dict:
    return {
        "grounding_metadata": {
            "web_search_queries": [query],
            "grounding_chunks": [{"web": {"uri": uri, "title": title}}],
            "grounding_supports": [],
        }
    }


def test_google_search_activity_completes_once_per_cycle() -> None:
    evidence: dict = {}
    activity = GoogleSearchActivity("msg_1", evidence)

    started = activity.observe(_grounding_metadata("q1", "https://a.example", "A"))
    assert started is not None
    assert started.data["tool_name"] == "google_search"

    completed = activity.completed_event()
    assert completed is not None
    assert "1 web result" in completed.data["output_text"]
    assert completed.data["call_id"] == started.data["call_id"]
    # Idempotent: the post-turn fallback must not duplicate the event.
    assert activity.completed_event() is None


def test_google_search_activity_restarts_after_completion() -> None:
    # A later model step running another search gets its own activity cycle
    # instead of being swallowed by the completed one.
    activity = GoogleSearchActivity("msg_1", {})
    first = activity.observe(_grounding_metadata("q1", "https://a.example", "A"))
    assert activity.completed_event() is not None

    second = activity.observe(_grounding_metadata("q2", "https://b.example", "B"))
    assert second is not None
    assert second.data["call_id"] != first.data["call_id"]
    assert second.data["args_text"] == "q2"

    completed = activity.completed_event()
    assert completed is not None
    assert "1 web result" in completed.data["output_text"]


def test_google_search_activity_repeated_query_same_evidence_restarts() -> None:
    # Scenario (a): a later model step re-runs the SAME query and grounds
    # only already-seen URLs. Query dedupe is cycle-scoped, so the repeat
    # must still start a fresh, visible cycle with its query text.
    evidence: dict = {}
    activity = GoogleSearchActivity("msg_1", evidence)
    first = activity.observe(_grounding_metadata("q1", "https://a.example", "A"))
    assert first is not None
    assert activity.completed_event() is not None

    second = activity.observe(_grounding_metadata("q1", "https://a.example", "A"))
    assert second is not None
    assert second.data["call_id"] != first.data["call_id"]
    assert second.data["args_text"] == "q1"

    completed = activity.completed_event()
    assert completed is not None
    assert completed.data["args_text"] == "q1"
    # Evidence stays turn-deduped: the repeated URL is not double-recorded,
    # so the repeat cycle reports no (new) grounding results.
    assert list(evidence) == ["google_search::https://a.example"]
    assert "no grounding results" in completed.data["output_text"]


def test_google_search_activity_repeated_query_new_evidence_keeps_query() -> None:
    # Scenario (b): a later model step re-runs the SAME query but grounds a
    # new URL. The fresh cycle must display the query text, not "".
    evidence: dict = {}
    activity = GoogleSearchActivity("msg_1", evidence)
    assert activity.observe(_grounding_metadata("q1", "https://a.example", "A"))
    assert activity.completed_event() is not None

    second = activity.observe(_grounding_metadata("q1", "https://b.example", "B"))
    assert second is not None
    assert second.data["args_text"] == "q1"

    completed = activity.completed_event()
    assert completed is not None
    assert completed.data["args_text"] == "q1"
    assert "1 web result" in completed.data["output_text"]
    assert set(evidence) == {
        "google_search::https://a.example",
        "google_search::https://b.example",
    }


def test_google_search_activity_dedupes_within_active_cycle() -> None:
    # Streaming repeats the same cumulative metadata within one model step
    # (chunk deltas, then the step-final message). Within an active cycle
    # that repeat must not start a second cycle or duplicate the query.
    activity = GoogleSearchActivity("msg_1", {})
    metadata = _grounding_metadata("q1", "https://a.example", "A")
    assert activity.observe(metadata) is not None
    assert activity.observe(metadata) is None

    completed = activity.completed_event()
    assert completed is not None
    assert completed.data["args_text"] == "q1"


def test_web_fetch_result_object_recorded_as_evidence() -> None:
    # Real shape returned by web_fetch_20260209: content is one
    # web_fetch_result document, not a list of results.
    content = [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_fetch",
            "name": "web_fetch",
            "input": {"url": "https://example.com"},
        },
        {
            "type": "web_fetch_tool_result",
            "tool_use_id": "srvtoolu_fetch",
            "content": {
                "type": "web_fetch_result",
                "url": "https://example.com",
                "retrieved_at": "2026-06-10T17:36:00Z",
                "content": {
                    "type": "document",
                    "title": "Example Domain",
                    "source": {"type": "text", "data": "..."},
                },
            },
        },
    ]
    evidence: dict = {}

    updates, tool_uses = extract_anthropic_source_matches(content, evidence)

    assert "srvtoolu_fetch" in tool_uses
    matches = updates["srvtoolu_fetch"]
    assert len(matches) == 1
    assert matches[0].url == "https://example.com"
    assert matches[0].title == "Example Domain"
    assert matches[0].source_key == "web_fetch"
    assert matches[0].group == "web"
    assert "web_fetch::https://example.com" in evidence


def test_web_fetch_error_result_records_no_evidence() -> None:
    content = [
        {
            "type": "web_fetch_tool_result",
            "tool_use_id": "srvtoolu_fetch",
            "content": {
                "type": "web_fetch_tool_result_error",
                "error_code": "url_not_accessible",
            },
        },
    ]
    evidence: dict = {}

    updates, _tool_uses = extract_anthropic_source_matches(content, evidence)

    assert updates == {}
    assert evidence == {}


def test_web_search_result_list_still_recorded() -> None:
    content = [
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_search",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://example.com/post",
                    "title": "A blog post",
                },
            ],
        },
    ]
    evidence: dict = {}

    updates, _tool_uses = extract_anthropic_source_matches(content, evidence)

    matches = updates["srvtoolu_search"]
    assert len(matches) == 1
    assert matches[0].title == "A blog post"
    assert matches[0].source_key == "web_search"
