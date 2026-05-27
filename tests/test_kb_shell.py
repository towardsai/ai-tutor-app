from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from data.scraping_scripts.build_kb_artifacts import build_kb_artifacts
from data.scraping_scripts.update_kb_wiki import update_kb_wiki
from scripts.kb_shell import KbCommandError, format_command_payload, run_kb_command


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture()
def kb_dir(tmp_path: Path) -> Path:
    input_file = tmp_path / "all_sources_data.jsonl"
    output_dir = tmp_path / "kb"
    write_jsonl(
        input_file,
        [
            {
                "doc_id": "peft:lora",
                "name": "LoRA",
                "url": "https://example.com/peft/lora",
                "source": "peft",
                "source_path": "package_reference/lora.md",
                "tokens": 300,
                "retrieve_doc": True,
                "content": "# LoRA\n\nUse `LoraConfig` with `get_peft_model`.",
            },
            {
                "doc_id": "agentic_ai_engineering:lesson-18",
                "name": "Lesson 18: Research Loop",
                "url": "https://academy.towardsai.net/lesson-18",
                "source": "agentic_ai_engineering",
                "tokens": 500,
                "retrieve_doc": True,
                "content": "# Lesson 18\n\nCall `generate_next_queries_tool`.",
            },
        ],
    )
    build_kb_artifacts(input_file, output_dir)
    update_kb_wiki(output_dir, seed_defaults=True)
    return output_dir


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is required")
def test_run_kb_command_runs_rg_and_returns_plain_transcript(kb_dir: Path) -> None:
    result = run_kb_command(
        "rg LoraConfig raw/docs/peft",
        root=kb_dir,
    )

    assert result.command == "rg LoraConfig raw/docs/peft"
    assert result.argv[:4] == ["rg", "--color=never", "--line-number", "--no-heading"]
    assert result.exit_code == 0
    assert not result.timed_out
    assert "LoraConfig" in result.stdout

    payload = format_command_payload(result)
    assert "$ rg LoraConfig raw/docs/peft" in payload
    assert "exit_code: 0" in payload
    assert "stdout:" in payload
    assert "matches" not in payload


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is required")
def test_run_kb_command_can_read_all_raw_sources(kb_dir: Path) -> None:
    result = run_kb_command(
        "rg generate_next_queries_tool raw/courses/agentic_ai_engineering",
        root=kb_dir,
    )

    assert result.exit_code == 0
    assert "generate_next_queries_tool" in result.stdout
    assert "agentic_ai_engineering" in result.stdout


def test_run_kb_command_rejects_shell_chaining(kb_dir: Path) -> None:
    with pytest.raises(KbCommandError):
        run_kb_command("rg LoraConfig | head", root=kb_dir)


def test_run_kb_command_rejects_path_traversal(kb_dir: Path) -> None:
    with pytest.raises(KbCommandError):
        run_kb_command("cat ../outside.md", root=kb_dir)


def test_run_kb_command_rejects_unbounded_broad_raw_search(kb_dir: Path) -> None:
    with pytest.raises(KbCommandError, match="requires -m"):
        run_kb_command("rg LoraConfig raw", root=kb_dir)

    result = run_kb_command("rg -m 20 LoraConfig raw", root=kb_dir)

    assert result.exit_code == 0
    assert "LoraConfig" in result.stdout
