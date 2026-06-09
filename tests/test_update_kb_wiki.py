from __future__ import annotations

import json
from pathlib import Path

from data.scraping_scripts import build_kb_artifacts as builder
from data.scraping_scripts.build_kb_artifacts import build_kb_artifacts
from data.scraping_scripts.update_kb_wiki import update_kb_wiki


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def seed_peft_original_markdown(tmp_path: Path, monkeypatch) -> None:
    """Point the peft source at a tmp original-markdown tree so the mirrored
    `raw/docs/peft/package_reference/lora.md` path is produced on any machine,
    instead of depending on the locally downloaded corpus (data/peft_md_files)."""
    md_page = tmp_path / "peft_md_files" / "package_reference" / "lora.md"
    md_page.parent.mkdir(parents=True)
    md_page.write_text(
        "# LoRA\n\nUse `LoraConfig` for adapter fine-tuning.\n", encoding="utf-8"
    )
    peft_config = dict(builder.SOURCE_CONFIGS["peft"])
    peft_config["input_directory"] = str(tmp_path / "peft_md_files")
    monkeypatch.setitem(builder.SOURCE_CONFIGS, "peft", peft_config)


def test_update_kb_wiki_seeds_navigation_pages(tmp_path: Path, monkeypatch) -> None:
    seed_peft_original_markdown(tmp_path, monkeypatch)
    input_file = tmp_path / "all_sources_data.jsonl"
    kb_dir = tmp_path / "kb"
    write_jsonl(
        input_file,
        [
            {
                "doc_id": "peft:lora",
                "name": "LoRA",
                "url": "https://example.com/lora",
                "source": "peft",
                "source_path": "package_reference/lora.md",
                "tokens": 300,
                "retrieve_doc": True,
                "content": "# LoRA\n\nUse `LoraConfig` for adapter fine-tuning.",
            },
            {
                "doc_id": "agentic_ai_engineering:lesson-1",
                "name": "Lesson 1",
                "url": "https://example.com/lesson-1",
                "source": "agentic_ai_engineering",
                "tokens": 300,
                "retrieve_doc": True,
                "content": "# Lesson 1\n\nAgent lesson.",
            },
        ],
    )
    build_kb_artifacts(input_file, kb_dir)

    update_kb_wiki(kb_dir, seed_defaults=True)

    assert (kb_dir / "AGENTS.md").exists()
    assert (kb_dir / "wiki" / "index.md").exists()
    assert (kb_dir / "wiki" / "frameworks" / "peft.md").exists()
    assert (kb_dir / "wiki" / "courses" / "agentic_ai_engineering.md").exists()
    agents = (kb_dir / "AGENTS.md").read_text(encoding="utf-8")
    assert "## Ground Truth" in agents
    assert "## First Command Rule" in agents
    assert "run_kb_command" in agents
    index = (kb_dir / "wiki" / "index.md").read_text(encoding="utf-8")
    assert "`wiki/courses/agentic_ai_engineering.md`" in index
    assert "`wiki/frameworks/peft.md`" in index
    topic = (kb_dir / "wiki" / "topics" / "lora.md").read_text(encoding="utf-8")
    assert "LoRA" in topic
    assert "`raw/docs/peft/package_reference/lora.md`" in topic
    assert "lookup_tutor_symbol" not in topic


def test_update_kb_wiki_preserves_authored_topic_content_and_appends_log(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seed_peft_original_markdown(tmp_path, monkeypatch)
    input_file = tmp_path / "all_sources_data.jsonl"
    kb_dir = tmp_path / "kb"
    write_jsonl(
        input_file,
        [
            {
                "doc_id": "peft:lora",
                "name": "LoRA",
                "url": "https://example.com/lora",
                "source": "peft",
                "source_path": "package_reference/lora.md",
                "tokens": 300,
                "retrieve_doc": True,
                "content": "# LoRA\n\nUse `LoraConfig` for adapter fine-tuning.",
            }
        ],
    )
    build_kb_artifacts(input_file, kb_dir)
    topic_path = kb_dir / "wiki" / "topics" / "lora.md"
    topic_path.parent.mkdir(parents=True)
    topic_path.write_text("# Lora\n\nAuthored synthesis stays.\n", encoding="utf-8")

    update_kb_wiki(kb_dir, seed_defaults=False)
    update_kb_wiki(kb_dir, seed_defaults=False)

    topic = topic_path.read_text(encoding="utf-8")
    assert "Authored synthesis stays." in topic
    assert "`raw/docs/peft/package_reference/lora.md`" in topic
    log = (kb_dir / "wiki" / "log.md").read_text(encoding="utf-8")
    assert log.count("Generated or refreshed") == 2
