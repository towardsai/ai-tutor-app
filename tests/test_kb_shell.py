from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from data.scraping_scripts.build_kb_artifacts import build_kb_artifacts
from data.scraping_scripts.update_kb_wiki import update_kb_wiki
from app.kb_shell import (
    KbCommandError,
    build_kb_command_argv,
    format_command_payload,
    run_kb_command,
)


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


def test_run_kb_command_caps_large_output_without_buffering_all(kb_dir: Path) -> None:
    # A command that emits far more than the cap must be truncated to the cap;
    # the streaming reader retains at most `output_limit` chars regardless of how
    # much the child writes, so peak memory is bounded by the cap, not the file.
    big = kb_dir / "raw" / "big.txt"
    big.write_text("A" * 200_000)

    result = run_kb_command("cat raw/big.txt", root=kb_dir, max_output_chars=1000)

    assert result.truncated
    assert len(result.stdout) <= 1000
    assert "output truncated" in result.stdout


def test_run_kb_command_rejects_shell_chaining(kb_dir: Path) -> None:
    with pytest.raises(KbCommandError):
        run_kb_command("rg LoraConfig | head", root=kb_dir)


def test_run_kb_command_rejects_path_traversal(kb_dir: Path) -> None:
    with pytest.raises(KbCommandError):
        run_kb_command("cat ../outside.md", root=kb_dir)


def test_rg_pattern_is_separated_by_double_dash(kb_dir: Path) -> None:
    # A pattern beginning with `-`/`--` (e.g. ripgrep's `--pre`, which spawns an
    # external preprocessor binary) is rejected in option position with a
    # corrective error; after an explicit `--` it is forced into the positional
    # pattern slot by the emitted separator so rg can never re-parse it as a flag.
    with pytest.raises(KbCommandError, match="Unsupported rg option"):
        build_kb_command_argv("rg --pre=/bin/sh raw/docs/peft", root=kb_dir)

    argv, _ = build_kb_command_argv("rg -- --pre=/bin/sh raw/docs/peft", root=kb_dir)
    assert "--" in argv
    sep = argv.index("--")
    assert argv[sep + 1 :] == ["--pre=/bin/sh", "raw/docs/peft"]
    # The dangerous token never precedes the separator (i.e. is never an option).
    assert "--pre=/bin/sh" not in argv[:sep]


def test_grep_pattern_is_separated_by_double_dash(kb_dir: Path) -> None:
    with pytest.raises(KbCommandError, match="Unsupported grep option"):
        build_kb_command_argv("grep --label=foo raw/docs/peft", root=kb_dir)

    argv, _ = build_kb_command_argv("grep -- --label=foo raw/docs/peft", root=kb_dir)
    assert "--" in argv
    sep = argv.index("--")
    assert argv[sep + 1 :] == ["--label=foo", "raw/docs/peft"]
    assert "--label=foo" not in argv[:sep]


def test_run_kb_command_rejects_unbounded_broad_raw_search(kb_dir: Path) -> None:
    with pytest.raises(KbCommandError, match="requires -m"):
        run_kb_command("rg LoraConfig raw", root=kb_dir)

    result = run_kb_command("rg -m 20 LoraConfig raw", root=kb_dir)

    assert result.exit_code == 0
    assert "LoraConfig" in result.stdout


def test_head_accepts_dash_count_shorthands(kb_dir: Path) -> None:
    # `head -200 FILE` and `head -n200 FILE` are standard shorthands for
    # `head -n 200 FILE`; all three must normalize to the same argv.
    sample = kb_dir / "raw" / "sample.txt"
    sample.write_text("line\n" * 10)

    expected = ["head", "-n", "200", "raw/sample.txt"]
    for command in (
        "head -200 raw/sample.txt",
        "head -n200 raw/sample.txt",
        "head -n 200 raw/sample.txt",
    ):
        argv, _ = build_kb_command_argv(command, root=kb_dir)
        assert argv == expected


def test_head_rejects_unsupported_option_with_corrective_error(kb_dir: Path) -> None:
    # The old behavior resolved `-c` as a path and reported "path does not
    # exist: -c", which misled the agent; the error must name the supported form.
    with pytest.raises(KbCommandError, match="Unsupported head option") as excinfo:
        build_kb_command_argv("head -c 5 raw/sample.txt", root=kb_dir)
    assert "head [-n N | -N] FILE" in str(excinfo.value)


def test_cat_wc_ls_reject_unsupported_options(kb_dir: Path) -> None:
    with pytest.raises(KbCommandError, match="Unsupported cat option") as excinfo:
        build_kb_command_argv("cat -n wiki/index.md", root=kb_dir)
    assert "only file paths" in str(excinfo.value)

    with pytest.raises(KbCommandError, match="Unsupported wc option") as excinfo:
        build_kb_command_argv("wc -L wiki/index.md", root=kb_dir)
    assert "-l -w -c -m" in str(excinfo.value)

    with pytest.raises(KbCommandError, match="Unsupported ls option") as excinfo:
        build_kb_command_argv("ls -R raw", root=kb_dir)
    assert "-1 -a -l -la -al" in str(excinfo.value)


def test_rg_and_grep_reject_unknown_options_with_corrective_error(
    kb_dir: Path,
) -> None:
    # Unknown dash tokens in option position must not be silently demoted to
    # patterns; the error names the supported options and the `--` escape hatch.
    with pytest.raises(KbCommandError, match="Unsupported rg option") as excinfo:
        build_kb_command_argv("rg -A 3 LoraConfig raw/docs/peft", root=kb_dir)
    message = str(excinfo.value)
    assert "-m/--max-count" in message
    assert "put `--` before it" in message

    with pytest.raises(KbCommandError, match="Unsupported grep option") as excinfo:
        build_kb_command_argv("grep -o LoraConfig raw/docs/peft", root=kb_dir)
    message = str(excinfo.value)
    assert "-m/--max-count" in message
    assert "put `--` before it" in message


def test_rg_double_dash_still_allows_dash_leading_patterns(kb_dir: Path) -> None:
    argv, _ = build_kb_command_argv("rg -- -dashpattern raw/docs/peft", root=kb_dir)
    sep = argv.index("--")
    assert argv[sep + 1 :] == ["-dashpattern", "raw/docs/peft"]


def test_grep_is_bounded_like_rg_and_does_not_follow_symlinks(kb_dir: Path) -> None:
    # grep must reject an unbounded broad raw recursion, mirroring rg.
    with pytest.raises(KbCommandError, match="requires -m"):
        run_kb_command("grep LoraConfig raw", root=kb_dir)

    # An explicit -m/--max-count satisfies the bound and works end-to-end.
    argv, _ = build_kb_command_argv("grep -m 20 LoraConfig raw", root=kb_dir)
    assert argv[:4] == ["grep", "-r", "-n", "--color=never"]
    assert "-R" not in argv  # never the symlink-following recursion mode
    assert "-m" in argv

    result = run_kb_command("grep -m 20 LoraConfig raw/docs/peft", root=kb_dir)
    assert result.exit_code == 0
    assert "LoraConfig" in result.stdout
