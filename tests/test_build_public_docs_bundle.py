from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from app import kb_manifest
from data.scraping_scripts import build_public_docs_bundle as builder
from data.scraping_scripts.source_registry import (
    COURSE_SOURCE_KEYS,
    DOC_SOURCE_KEYS,
    SOURCE_CONFIGS,
)

# Use real registry keys so the COURSE/_DOC split matches production.
COURSE_KEY = "master_ai_for_work"
DOC_KEY = "transformers"


def test_registry_classifies_every_source() -> None:
    """Every source must be classified as doc or course — no third bucket.

    The public bundle publishes exactly DOC_SOURCE_KEYS; this invariant makes
    "I added a source to SOURCE_CONFIGS but forgot the grouping tuples" a CI
    failure instead of a silent misclassification. (The build script itself
    fails closed — an unclassified source is dropped, not published — but it
    should never get that far.)
    """
    docs = set(DOC_SOURCE_KEYS)
    courses = set(COURSE_SOURCE_KEYS)
    assert not docs & courses, "a source cannot be both doc and course"
    assert set(SOURCE_CONFIGS) == docs | courses, (
        "unclassified source(s): add them to DOC_SOURCE_KEYS or "
        f"COURSE_SOURCE_KEYS in source_registry.py: "
        f"{sorted(set(SOURCE_CONFIGS) - docs - courses)}"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_filter_jsonl_drops_course_rows(tmp_path: Path) -> None:
    path = tmp_path / "headings.jsonl"
    _write_jsonl(
        path,
        [
            {"source": DOC_KEY, "heading": "Install"},
            {"source": COURSE_KEY, "heading": "Lesson 1"},
            {"source": DOC_KEY, "heading": "Usage"},
        ],
    )
    kept = builder._filter_jsonl(path)
    assert kept == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {row["source"] for row in rows} == {DOC_KEY}


def test_filter_jsonl_fails_closed_on_unclassified_sources(tmp_path: Path) -> None:
    """Allowlist semantics: unknown/missing sources are dropped, not kept."""
    path = tmp_path / "headings.jsonl"
    _write_jsonl(
        path,
        [
            {"source": DOC_KEY, "heading": "Install"},
            {"source": "brand_new_unregistered_source", "heading": "?"},
            {"heading": "no source field at all"},
        ],
    )
    kept = builder._filter_jsonl(path)
    assert kept == 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {row["source"] for row in rows} == {DOC_KEY}


def test_filter_tsv_drops_course_rows_keeps_header(tmp_path: Path) -> None:
    path = tmp_path / "symbols.tsv"
    path.write_text(
        "symbol\tsource\ttitle\tpath\theading\tdoc_id\n"
        f"AutoModel\t{DOC_KEY}\tT\tp\th\td1\n"
        f"Lesson\t{COURSE_KEY}\tT\tp\th\td2\n",
        encoding="utf-8",
    )
    kept = builder._filter_tsv(path)
    assert kept == 1
    lines = path.read_text().splitlines()
    assert lines[0].startswith("symbol\tsource")  # header preserved
    assert COURSE_KEY not in path.read_text()
    assert DOC_KEY in path.read_text()


def test_stage_kb_prunes_course_content(tmp_path: Path) -> None:
    source_dir = tmp_path / "data"
    kb = source_dir / "kb"
    (kb / "raw" / "docs" / DOC_KEY).mkdir(parents=True)
    (kb / "raw" / "docs" / DOC_KEY / "a.md").write_text("# A", encoding="utf-8")
    (kb / "raw" / "courses" / COURSE_KEY).mkdir(parents=True)
    (kb / "raw" / "courses" / COURSE_KEY / "l1.md").write_text("# L1", encoding="utf-8")
    (kb / "wiki" / "courses").mkdir(parents=True)
    (kb / "wiki" / "courses" / "c.md").write_text("course", encoding="utf-8")
    (kb / "wiki").joinpath("index.md").write_text("# Index", encoding="utf-8")
    (kb / "MAINTAINER.md").write_text(
        "Examples cite `raw/courses/...` paths", encoding="utf-8"
    )

    generated = kb / "generated"
    generated.mkdir(parents=True)
    _write_jsonl(
        generated / "corpus_manifest.jsonl",
        [
            {"source": DOC_KEY, "source_group": "docs", "path": "data/kb/raw/docs/x"},
            {"source": COURSE_KEY, "source_group": "courses", "path": "y"},
        ],
    )
    _write_jsonl(
        generated / "headings.jsonl",
        [{"source": DOC_KEY}, {"source": COURSE_KEY}],
    )
    (generated / "symbols.tsv").write_text(
        "symbol\tsource\ttitle\tpath\theading\tdoc_id\n"
        f"X\t{DOC_KEY}\tT\tp\th\td1\n"
        f"Y\t{COURSE_KEY}\tT\tp\th\td2\n",
        encoding="utf-8",
    )

    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    summary = builder.stage_kb(source_dir, stage_dir)

    staged_kb = stage_dir / "kb"
    assert not (staged_kb / "raw" / "courses").exists()
    assert (staged_kb / "raw" / "docs" / DOC_KEY / "a.md").exists()
    assert not (staged_kb / "wiki" / "courses").exists()
    assert (staged_kb / "wiki" / "index.md").exists()  # docs wiki kept
    # Maintainer manual ships only in the private bundle (quotes course paths).
    assert not (staged_kb / "MAINTAINER.md").exists()

    manifest = [
        json.loads(line)
        for line in (staged_kb / "generated" / "corpus_manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {row["source"] for row in manifest} == {DOC_KEY}
    assert summary["manifest_rows"] == 1
    assert COURSE_KEY not in (staged_kb / "generated" / "symbols.tsv").read_text()


def test_rebuild_retrieval_pkls_drops_course_documents(tmp_path: Path) -> None:
    source_dir = tmp_path / "data"
    rows = [
        {
            "doc_id": "doc-transformers",
            "name": "Transformers",
            "url": "https://example.com/t",
            "source": DOC_KEY,
            "content": "# Heading\n\nUse AutoModel to load weights.",
            "retrieve_doc": True,
            "tokens": 9,
        },
        {
            "doc_id": "doc-course",
            "name": "Course Lesson",
            "url": "https://example.com/c",
            "source": COURSE_KEY,
            "content": "# Lesson\n\nPrivate course material.",
            "retrieve_doc": True,
            "tokens": 7,
        },
    ]
    _write_jsonl(source_dir / "all_sources_data.jsonl", rows)
    (tmp_path / "stage" / builder.VECTOR_DB_DIR).mkdir(parents=True)

    summary = builder.rebuild_retrieval_pkls(source_dir, tmp_path / "stage")
    assert summary["documents"] == 1

    dict_path = tmp_path / "stage" / builder.VECTOR_DB_DIR / builder.DOCUMENT_DICT_FILE
    with dict_path.open("rb") as handle:
        document_dict = pickle.load(handle)
    assert set(document_dict) == {"doc-transformers"}


def test_public_allow_patterns_toggles_contextual() -> None:
    base = builder.public_allow_patterns(include_contextual=False)
    assert base == ["chroma-db-all_sources/**", "kb/**", "README.md"]
    with_ctx = builder.public_allow_patterns(include_contextual=True)
    assert "all_sources_contextual_nodes.pkl" in with_ctx


def test_dataset_card_names_only_public_sources(tmp_path: Path) -> None:
    builder.write_dataset_card(tmp_path)
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    for key in builder._PUBLIC_KEYS:
        assert f"`{key}`" in card
    for token in builder._prose_prune_tokens():
        assert token not in card


def test_available_source_keys_excludes_absent_sources(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    generated = kb_dir / "generated"
    generated.mkdir(parents=True)
    # A docs-only manifest (the public bundle): courses are simply not present.
    _write_jsonl(
        generated / "corpus_manifest.jsonl",
        [
            {"doc_id": "1", "source": DOC_KEY, "source_group": "docs"},
            {"doc_id": "2", "source": "langchain", "source_group": "docs"},
        ],
    )
    kb_manifest._MANIFEST_CACHE.pop(str(kb_dir), None)
    keys = kb_manifest.available_source_keys(str(kb_dir))
    assert keys == frozenset({DOC_KEY, "langchain"})
    assert COURSE_KEY not in keys


def test_available_source_keys_none_when_manifest_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no_kb"
    kb_manifest._MANIFEST_CACHE.pop(str(missing), None)
    assert kb_manifest.available_source_keys(str(missing)) is None


TOKENS = builder._prose_prune_tokens()


def test_prune_course_prose_drops_course_bullets_keeps_doc_bullets() -> None:
    text = (
        "# Rag\n"
        "\n"
        "## Where to look first\n"
        "\n"
        "- For *concepts*: `raw/courses/agentic_ai_engineering/lesson-9.md`\n"
        "- For *vector stores*: `raw/docs/langchain/vectorstores/index.mdx`\n"
        "- For *the 2x2 matrix*: see [llm_primer](../courses/llm_primer.md).\n"
    )
    pruned, removed = builder.prune_course_prose(text, TOKENS)
    assert removed == 2
    assert "raw/courses" not in pruned
    assert "llm_primer" not in pruned
    assert "raw/docs/langchain/vectorstores/index.mdx" in pruned


def test_prune_course_prose_strips_sentences_inside_prose_lines() -> None:
    text = (
        "RAG pairs a retriever with a generator. "
        "See `raw/courses/full_stack_ai_engineering/rag.md` for a walkthrough. "
        "Production stores are covered in `raw/docs/langchain/`.\n"
    )
    pruned, removed = builder.prune_course_prose(text, TOKENS)
    assert removed == 1
    assert "raw/courses" not in pruned
    assert pruned.startswith("RAG pairs a retriever with a generator.")
    assert "raw/docs/langchain/" in pruned


def test_prune_course_prose_leaves_marker_blocks_alone() -> None:
    """Scaffolder-owned content is regenerated, never text-pruned."""
    text = (
        "Prose mentioning `raw/courses/x.md` gets pruned.\n"
        "\n"
        "<!-- AUTO-GENERATED:START -->\n"
        "- Lesson: `raw/courses/agentic_ai_engineering/lesson-9.md`\n"
        "<!-- AUTO-GENERATED:END -->\n"
    )
    pruned, removed = builder.prune_course_prose(text, TOKENS)
    assert removed == 1
    # The marker block keeps its (scaffolder-owned) course line verbatim.
    assert "raw/courses/agentic_ai_engineering/lesson-9.md" in pruned
    assert "gets pruned" not in pruned


def _minimal_public_kb(stage_dir: Path) -> Path:
    kb = stage_dir / "kb"
    (kb / "wiki").mkdir(parents=True)
    (kb / "wiki" / "index.md").write_text("# Index\n\nDocs only.\n", encoding="utf-8")
    generated = kb / "generated"
    generated.mkdir()
    _write_jsonl(generated / "corpus_manifest.jsonl", [{"source": DOC_KEY}])
    return kb


def test_audit_staged_kb_passes_on_clean_bundle(tmp_path: Path) -> None:
    _minimal_public_kb(tmp_path)
    builder.audit_staged_kb(tmp_path)  # must not raise


def test_audit_staged_kb_fails_on_course_rows_in_indexes(tmp_path: Path) -> None:
    kb = _minimal_public_kb(tmp_path)
    _write_jsonl(
        kb / "generated" / "corpus_manifest.jsonl",
        [{"source": DOC_KEY}, {"source": COURSE_KEY}],
    )
    with pytest.raises(SystemExit):
        builder.audit_staged_kb(tmp_path)


def test_audit_staged_kb_fails_on_course_mentions_in_wiki(tmp_path: Path) -> None:
    kb = _minimal_public_kb(tmp_path)
    (kb / "wiki" / "topics").mkdir()
    (kb / "wiki" / "topics" / "rag.md").write_text(
        "See `raw/courses/agentic_ai_engineering/lesson-9.md`.\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        builder.audit_staged_kb(tmp_path)


def test_audit_staged_kb_fails_on_surviving_course_dirs(tmp_path: Path) -> None:
    kb = _minimal_public_kb(tmp_path)
    (kb / "raw" / "courses").mkdir(parents=True)
    with pytest.raises(SystemExit):
        builder.audit_staged_kb(tmp_path)


def test_audit_staged_kb_covers_kb_root_files_but_exempts_agents_md(
    tmp_path: Path,
) -> None:
    kb = _minimal_public_kb(tmp_path)
    # AGENTS.md comes from the repo template, which names the wiki/courses/
    # layout generically; it must not trip the audit.
    (kb / "AGENTS.md").write_text(
        "cat wiki/courses/<source>.md when present", encoding="utf-8"
    )
    builder.audit_staged_kb(tmp_path)  # passes
    # But any other kb-root markdown mentioning course content must fail.
    (kb / "MAINTAINER.md").write_text(
        "Example: `raw/courses/x/lesson-1.md`", encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        builder.audit_staged_kb(tmp_path)
