from __future__ import annotations

from pathlib import Path

from scripts.prompts import build_system_prompt, load_kb_agents_instructions


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
