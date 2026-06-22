from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.prompts import (
    build_system_prompt,
    ensure_kb_agents_instructions,
    load_kb_agents_instructions,
)


def test_build_system_prompt_includes_local_kb_agents_instructions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    agents_path = tmp_path / "AGENTS.md"
    agents_path.write_text(
        "# Test KB Instructions\n\n"
        "- Read `wiki/index.md` before broad questions.\n"
        "- Treat `raw/` as source authority.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TUTOR_KB_AGENTS_PATH", str(agents_path))

    prompt = build_system_prompt("google-genai:gemini-3.5-flash", ())

    assert "## Local KB Instructions" in prompt
    assert "loaded from `data/kb/AGENTS.md`" in prompt
    assert "# Test KB Instructions" in prompt
    assert "Treat `raw/` as source authority." in prompt


def test_build_system_prompt_omits_local_kb_section_when_file_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AI_TUTOR_KB_AGENTS_PATH", str(tmp_path / "missing.md"))

    prompt = build_system_prompt("google-genai:gemini-3.5-flash", ())

    assert "## Local KB Instructions" not in prompt


def test_load_kb_agents_instructions_returns_empty_for_missing_path(
    tmp_path: Path,
) -> None:
    assert load_kb_agents_instructions(tmp_path / "missing.md") == ""


def test_build_system_prompt_without_local_tools_describes_none_of_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Explicit empty source selection: the prompt must not describe or
    # instruct retrieval/KB tools the agent does not have this turn.
    agents_path = tmp_path / "AGENTS.md"
    agents_path.write_text("# From disk\n", encoding="utf-8")
    monkeypatch.setenv("AI_TUTOR_KB_AGENTS_PATH", str(agents_path))

    prompt = build_system_prompt(
        "google-genai:gemini-3.5-flash",
        ("web_search",),
        include_local_tools=False,
    )

    assert "retrieve_tutor_context" not in prompt
    assert "run_kb_command" not in prompt
    assert "## Local KB Instructions" not in prompt
    assert "## Knowledge base disabled" in prompt
    assert "google_search" in prompt

    no_tools = build_system_prompt(
        "google-genai:gemini-3.5-flash",
        (),
        include_local_tools=False,
    )
    assert "tools available" not in no_tools
    assert "one tool available" not in no_tools


def test_build_system_prompt_includes_course_reproduction_policy() -> None:
    # The policy must reach the model whenever it can touch the corpus: both the
    # full KB-browsing path and the retrieval-only path (disable_kb).
    kb_on = build_system_prompt(
        "google-genai:gemini-3.5-flash", (), kb_agents_instructions=""
    )
    retrieval_only = build_system_prompt(
        "google-genai:gemini-3.5-flash", (), disable_kb=True
    )
    for prompt in (kb_on, retrieval_only):
        assert "## Course content reproduction policy" in prompt
        assert "raw/courses/" in prompt
        assert "raw/docs/" in prompt
        # Affordability redirect, gated on the student raising cost.
        assert "louis@towardsai.net" in prompt
        assert "never offer it unprompted" in prompt


def test_course_reproduction_policy_absent_without_corpus_access() -> None:
    # No sources selected: the agent has no corpus to copy from, so the policy
    # (which only governs corpus text) must not appear.
    prompt = build_system_prompt(
        "google-genai:gemini-3.5-flash", (), include_local_tools=False
    )
    assert "## Course content reproduction policy" not in prompt


def test_build_system_prompt_uses_explicit_instructions_without_file_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # The file on disk must be ignored when instructions are passed in.
    agents_path = tmp_path / "AGENTS.md"
    agents_path.write_text("# From disk\n", encoding="utf-8")
    monkeypatch.setenv("AI_TUTOR_KB_AGENTS_PATH", str(agents_path))

    explicit = build_system_prompt(
        "google-genai:gemini-3.5-flash",
        (),
        kb_agents_instructions="# Explicit KB rules",
    )
    assert "# Explicit KB rules" in explicit
    assert "# From disk" not in explicit

    empty = build_system_prompt(
        "google-genai:gemini-3.5-flash",
        (),
        kb_agents_instructions="",
    )
    assert "## Local KB Instructions" not in empty


def test_ensure_kb_agents_instructions_materializes_from_template(
    tmp_path: Path,
    monkeypatch,
) -> None:
    template_path = tmp_path / "kb_agents_template.md"
    template_path.write_text("# Generated KB rules\n", encoding="utf-8")
    agents_path = tmp_path / "kb" / "AGENTS.md"
    monkeypatch.setenv("AI_TUTOR_KB_AGENTS_PATH", str(agents_path))

    with (
        patch("app.config.KB_AGENTS_PATH", str(agents_path)),
        patch("app.config.KB_AGENTS_TEMPLATE_PATH", str(template_path)),
    ):
        instructions = ensure_kb_agents_instructions()

    assert instructions == "# Generated KB rules"
    assert agents_path.is_file()


def test_ensure_kb_agents_instructions_empty_when_template_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    agents_path = tmp_path / "kb" / "AGENTS.md"
    monkeypatch.setenv("AI_TUTOR_KB_AGENTS_PATH", str(agents_path))

    with (
        patch("app.config.KB_AGENTS_PATH", str(agents_path)),
        patch("app.config.KB_AGENTS_TEMPLATE_PATH", str(tmp_path / "missing.md")),
    ):
        assert ensure_kb_agents_instructions() == ""
