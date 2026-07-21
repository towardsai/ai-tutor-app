from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.scraping_scripts import lint_kb_wiki
from data.scraping_scripts.lint_kb_wiki import lint_kb_wiki as run_lint


def write_manifest(kb_dir: Path, raw_paths: list[Path]) -> None:
    manifest_path = kb_dir / "generated" / "corpus_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, raw_path in enumerate(raw_paths):
            row = {
                "doc_id": f"doc:{index}",
                "title": raw_path.stem,
                "source": raw_path.parent.name,
                "source_group": raw_path.parts[-3] if len(raw_path.parts) > 2 else "",
                # Real manifests store paths relative to the repo root
                # ("data/kb/raw/..."); tests use tmp-path absolutes, which the
                # shell_path normalizer resolves the same way.
                "path": (kb_dir / raw_path).as_posix(),
            }
            handle.write(json.dumps(row) + "\n")


def make_kb(tmp_path: Path) -> Path:
    """A minimal, internally consistent KB: raw pages, manifest, wiki pages."""
    kb_dir = tmp_path / "kb"
    raw_paths = [
        Path("raw/docs/peft/package_reference/lora.md"),
        Path("raw/courses/agentic_ai_engineering/lesson-1.md"),
    ]
    for raw_path in raw_paths:
        page = kb_dir / raw_path
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(f"# {raw_path.stem}\n", encoding="utf-8")
    write_manifest(kb_dir, raw_paths)
    (kb_dir / "generated" / "headings.jsonl").write_text("", encoding="utf-8")

    wiki_dir = kb_dir / "wiki"
    (wiki_dir / "topics").mkdir(parents=True)
    (wiki_dir / "frameworks").mkdir(parents=True)
    (wiki_dir / "index.md").write_text(
        "# Index\n\n"
        "- peft: `wiki/frameworks/peft.md`\n"
        "- Manifest: `generated/corpus_manifest.jsonl`\n"
        "- Headings: `generated/headings.jsonl`\n",
        encoding="utf-8",
    )
    (wiki_dir / "frameworks" / "peft.md").write_text(
        "# peft\n\n"
        "- LoRA: `raw/docs/peft/package_reference/lora.md`\n"
        "- Browse `raw/docs/peft/` with `rg`.\n"
        "- See [rag](../topics/rag.md) and [lesson 1]"
        "(../../raw/courses/agentic_ai_engineering/lesson-1.md#intro).\n",
        encoding="utf-8",
    )
    (wiki_dir / "topics" / "rag.md").write_text(
        "# RAG\n\n- Course intro: `raw/courses/agentic_ai_engineering/lesson-1.md`\n",
        encoding="utf-8",
    )
    return kb_dir


def test_clean_kb_passes(tmp_path: Path) -> None:
    kb_dir = make_kb(tmp_path)

    assert run_lint(kb_dir) == []


def test_conservative_extraction_ignores_templates_urls_and_prose(
    tmp_path: Path,
) -> None:
    kb_dir = make_kb(tmp_path)
    (kb_dir / "wiki" / "log.md").write_text(
        "# KB Log\n\n"
        "- Open `wiki/frameworks/{name}.md` or any `wiki/topics/*.md` page.\n"
        "- Model docs live in `raw/docs/transformers/model_doc/{name}.md`.\n"
        "- See https://example.com/raw/docs/nope.md and "
        "[docs](https://example.com/wiki/nope.md).\n"
        "- openai_cookbooks has no raw/docs/openai_cookbooks tree yet.\n",
        encoding="utf-8",
    )

    assert run_lint(kb_dir) == []


def test_missing_raw_file_fails(tmp_path: Path) -> None:
    kb_dir = make_kb(tmp_path)
    (kb_dir / "wiki" / "topics" / "rag.md").write_text(
        "# RAG\n\n- Deleted page: `raw/docs/peft/package_reference/gone.md`\n",
        encoding="utf-8",
    )

    findings = run_lint(kb_dir)

    assert len(findings) == 1
    assert findings[0].page == "wiki/topics/rag.md"
    assert findings[0].line == 3
    assert "missing KB path: raw/docs/peft/package_reference/gone.md" in str(
        findings[0]
    )


def test_raw_file_missing_from_manifest_fails(tmp_path: Path) -> None:
    kb_dir = make_kb(tmp_path)
    orphan = kb_dir / "raw" / "docs" / "peft" / "orphan.md"
    orphan.write_text("# Orphan\n", encoding="utf-8")
    (kb_dir / "wiki" / "topics" / "rag.md").write_text(
        "# RAG\n\n- Orphan: `raw/docs/peft/orphan.md`\n",
        encoding="utf-8",
    )

    findings = run_lint(kb_dir)

    assert len(findings) == 1
    assert "raw file not in generated/corpus_manifest.jsonl" in findings[0].message
    assert "raw/docs/peft/orphan.md" in findings[0].message


def test_broken_relative_wiki_link_fails(tmp_path: Path) -> None:
    kb_dir = make_kb(tmp_path)
    (kb_dir / "wiki" / "frameworks" / "peft.md").write_text(
        "# peft\n\n- See [retired topic](../topics/retired.md#anchor).\n",
        encoding="utf-8",
    )

    findings = run_lint(kb_dir)

    assert len(findings) == 1
    assert findings[0].page == "wiki/frameworks/peft.md"
    assert "broken relative link: ../topics/retired.md" in findings[0].message
    assert "wiki/topics/retired.md" in findings[0].message


def test_fenced_shell_command_paths_are_checked(tmp_path: Path) -> None:
    kb_dir = make_kb(tmp_path)
    (kb_dir / "wiki" / "topics" / "rag.md").write_text(
        "# RAG\n\n"
        "```bash\n"
        'rg -n "LoraConfig" raw/docs/peft/package_reference/lora.md\n'
        "sed -n 1,40p raw/docs/peft/stale-page.md\n"
        "```\n",
        encoding="utf-8",
    )

    findings = run_lint(kb_dir)

    assert [finding.message for finding in findings] == [
        "missing KB path: raw/docs/peft/stale-page.md"
    ]


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    kb_dir = make_kb(tmp_path)
    (kb_dir / "generated" / "corpus_manifest.jsonl").unlink()

    findings = run_lint(kb_dir)

    assert len(findings) == 1
    assert "corpus_manifest.jsonl not found" in findings[0].message


def test_main_exits_nonzero_on_findings(tmp_path: Path, monkeypatch) -> None:
    kb_dir = make_kb(tmp_path)
    (kb_dir / "wiki" / "index.md").write_text(
        "# Index\n\n- `wiki/frameworks/deleted.md`\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "sys.argv", ["lint_kb_wiki", "--kb-dir", str(kb_dir)], raising=False
    )

    with pytest.raises(SystemExit) as excinfo:
        lint_kb_wiki.main()

    assert excinfo.value.code == 1


def test_main_exits_zero_on_clean_kb(tmp_path: Path, monkeypatch, capsys) -> None:
    kb_dir = make_kb(tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["lint_kb_wiki", "--kb-dir", str(kb_dir)], raising=False
    )

    lint_kb_wiki.main()

    assert "KB wiki lint passed" in capsys.readouterr().out
