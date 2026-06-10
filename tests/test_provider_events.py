from __future__ import annotations

from app.provider_events import extract_anthropic_source_matches


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
