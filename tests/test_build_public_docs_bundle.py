from __future__ import annotations

import json
import pickle
from pathlib import Path

from app import kb_manifest
from data.scraping_scripts import build_public_docs_bundle as builder

# Use real registry keys so the COURSE/_DOC split matches production.
COURSE_KEY = "master_ai_for_work"
DOC_KEY = "transformers"


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
    assert base == ["chroma-db-all_sources/**", "kb/**"]
    with_ctx = builder.public_allow_patterns(include_contextual=True)
    assert "all_sources_contextual_nodes.pkl" in with_ctx


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
