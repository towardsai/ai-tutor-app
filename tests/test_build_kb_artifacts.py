from __future__ import annotations

import json
from pathlib import Path

from data.scraping_scripts import build_kb_artifacts as builder


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_kb_artifacts_writes_markdown_and_indexes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_file = tmp_path / "all_sources_data.jsonl"
    output_dir = tmp_path / "kb"
    docs_dir = tmp_path / "temp_docs_md_files"
    docs_page = docs_dir / "package_reference" / "lora.mdx"
    docs_page.parent.mkdir(parents=True)
    docs_content = """---
sidebarTitle: LoRA Guide
---
# LoRA Guide

Use `LoraConfig` with `get_peft_model`.
"""
    docs_page.write_text(docs_content, encoding="utf-8")
    monkeypatch.setitem(
        builder.SOURCE_CONFIGS,
        "temp_docs",
        {
            "base_url": "https://example.com/docs/",
            "input_directory": str(docs_dir),
            "output_file": str(tmp_path / "temp_docs_data.jsonl"),
            "source_name": "temp_docs",
            "use_include_list": False,
            "included_dirs": [],
            "excluded_dirs": [],
            "excluded_root_files": [],
            "included_root_files": [],
            "url_extension": "",
        },
    )
    write_jsonl(
        input_file,
        [
            {
                "doc_id": "legacy-random-id",
                "name": "LoRA Guide",
                "url": "https://example.com/docs/package_reference/lora",
                "source": "temp_docs",
                "tokens": 400,
                "retrieve_doc": True,
                "content": docs_content,
            },
            {
                "doc_id": "0a4fe6fa-928c-4cbf-951c-d0fd9dce01f3",
                "name": "Lesson 18: Research Loop",
                "url": "https://academy.towardsai.net/courses/take/agent-engineering/multimedia/70289117-lesson-18-the-research-loop",
                "source": "agentic_ai_engineering",
                "tokens": 800,
                "retrieve_doc": True,
                "content": "# Lesson 18: Research Loop\n\nCall `generate_next_queries_tool`.",
            },
        ],
    )

    summary = builder.build_kb_artifacts(input_file, output_dir)

    assert summary == {"documents": 2, "manifest_rows": 2}
    manifest_path = output_dir / "generated" / "corpus_manifest.jsonl"
    headings_path = output_dir / "generated" / "headings.jsonl"
    symbols_path = output_dir / "generated" / "symbols.tsv"
    assert manifest_path.exists()
    assert headings_path.exists()
    assert symbols_path.exists()

    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows[0]["doc_id"] == "temp_docs:docs-package-reference-lora"
    assert manifest_rows[0]["content_hash"].startswith("sha256:")
    assert manifest_rows[0]["source_path"] == "package_reference/lora.mdx"
    assert manifest_rows[0]["original_path"] == docs_page.as_posix()

    markdown_path = Path(manifest_rows[0]["path"])
    assert markdown_path == output_dir / "raw" / "docs" / "temp_docs" / "package_reference" / "lora.mdx"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert 'doc_id: "temp_docs:docs-package-reference-lora"' in markdown
    assert 'source_path: "package_reference/lora.mdx"' in markdown
    assert "sidebarTitle" not in markdown
    assert "# LoRA Guide" in markdown
    assert "`LoraConfig`" in markdown

    course_row = manifest_rows[1]
    assert course_row["source_group"] == "courses"
    assert course_row["source_path"] == ""

    symbols = symbols_path.read_text(encoding="utf-8")
    assert "LoraConfig" in symbols
    assert "generate_next_queries_tool" in symbols
