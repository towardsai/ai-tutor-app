"""Tests for the incremental contextual-node selection/merge in the workflows.

Regression coverage for two data bugs in new-content-only mode:

1. Data destruction: the workflows used to drop every existing node whose
   *source* had any new doc, gutting incrementally updated sources. The merge
   must instead preserve all existing nodes except those for doc_ids being
   (re)processed in the run.
2. Stale context: doc_ids are path-based and stable across content edits, so a
   membership-only reprocess filter never re-contextualized a doc whose
   content changed in place — its pkl nodes kept the old text forever. The
   selection must compare the row's ``content_hash`` against the
   ``doc_content_hash`` stamped in node metadata, while treating legacy nodes
   without the hash as unchanged (no surprise full-corpus Gemini reprocess).
"""

import asyncio
import json
from unittest.mock import patch

from app.chroma_rag import ChunkRecord
from data.scraping_scripts.update_docs_workflow import (
    DOC_CONTENT_HASH_METADATA_KEY,
    build_doc_hash_map,
    merge_contextual_nodes,
    select_docs_to_process,
)


def make_node(
    chunk_id: str, doc_id: str, source: str, content_hash: str | None = None
) -> ChunkRecord:
    metadata = {"doc_id": doc_id, "source": source}
    if content_hash is not None:
        metadata[DOC_CONTENT_HASH_METADATA_KEY] = content_hash
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"text for {chunk_id}",
        metadata=metadata,
    )


def make_doc(doc_id: str, content_hash: str | None = "sha256:aaa") -> dict:
    doc = {
        "doc_id": doc_id,
        "content": f"content for {doc_id}",
        "name": doc_id,
        "url": f"https://example.com/{doc_id}",
        "source": doc_id.split(":")[0],
        "retrieve_doc": False,
        "tokens": 10,
    }
    if content_hash is not None:
        doc["content_hash"] = content_hash
    return doc


def test_new_doc_in_existing_source_keeps_all_existing_nodes():
    # A source with already-indexed docs gains one new doc: every existing
    # node survives (same source or not) and the new doc's nodes are appended.
    existing = [
        make_node("t-p1-c1", "transformers:page-1", "transformers"),
        make_node("t-p1-c2", "transformers:page-1", "transformers"),
        make_node("t-p2-c1", "transformers:page-2", "transformers"),
        make_node("l-p1-c1", "langchain:page-1", "langchain"),
    ]
    new = [
        make_node("t-p3-c1", "transformers:page-3", "transformers"),
        make_node("t-p3-c2", "transformers:page-3", "transformers"),
    ]

    merged = merge_contextual_nodes(existing, new, {"transformers:page-3"})

    assert merged == existing + new


def test_reprocessed_doc_id_replaces_only_its_own_nodes():
    # If a doc_id in the new batch already has nodes in the pkl, those old
    # nodes are replaced by the freshly generated ones; sibling docs from the
    # same source are untouched.
    stale = [
        make_node("a-c1", "course:lesson-a", "course"),
        make_node("a-c2", "course:lesson-a", "course"),
    ]
    sibling = make_node("b-c1", "course:lesson-b", "course")
    regenerated = [make_node("a-c1-new", "course:lesson-a", "course")]

    merged = merge_contextual_nodes(stale + [sibling], regenerated, {"course:lesson-a"})

    assert merged == [sibling] + regenerated


def test_nodes_with_undetermined_doc_id_are_kept():
    class OpaqueNode:
        """No doc_id, metadata, or source_node: doc_id lookup raises."""

    opaque = OpaqueNode()
    parseable = make_node("x-c1", "src:doc-x", "src")
    new = [make_node("y-c1", "src:doc-y", "src")]

    merged = merge_contextual_nodes([opaque, parseable], new, {"src:doc-y"})

    assert merged == [opaque, parseable] + new


def test_empty_existing_pkl_yields_only_new_nodes():
    new = [make_node("n-c1", "src:doc-n", "src")]

    assert merge_contextual_nodes([], new, {"src:doc-n"}) == new


# --- Hash-forward selection: new + changed docs, legacy nodes untouched ---


def test_changed_content_hash_is_reprocessed_and_old_nodes_replaced():
    # A doc whose content changed in place (same path-stable doc_id, new
    # content_hash) must be selected for reprocessing, and its old nodes must
    # be replaced by the regenerated ones when merged back.
    existing = [
        make_node("d1-c1", "src:doc-1", "src", content_hash="sha256:old"),
        make_node("d1-c2", "src:doc-1", "src", content_hash="sha256:old"),
        make_node("d2-c1", "src:doc-2", "src", content_hash="sha256:same"),
    ]
    all_docs = [
        make_doc("src:doc-1", "sha256:new"),
        make_doc("src:doc-2", "sha256:same"),
    ]

    selected, stats = select_docs_to_process(all_docs, build_doc_hash_map(existing))

    assert [doc["doc_id"] for doc in selected] == ["src:doc-1"]
    assert stats == {"new": 0, "changed": 1, "legacy_unhashed": 0}

    regenerated = [
        make_node("d1-c1-new", "src:doc-1", "src", content_hash="sha256:new")
    ]
    merged = merge_contextual_nodes(
        existing, regenerated, {doc["doc_id"] for doc in selected}
    )

    assert merged == [existing[2]] + regenerated


def test_unchanged_content_hash_is_not_reprocessed():
    existing = [make_node("d1-c1", "src:doc-1", "src", content_hash="sha256:same")]

    selected, stats = select_docs_to_process(
        [make_doc("src:doc-1", "sha256:same")], build_doc_hash_map(existing)
    )

    assert selected == []
    assert stats == {"new": 0, "changed": 0, "legacy_unhashed": 0}


def test_legacy_nodes_without_hash_are_treated_as_unchanged():
    # Nodes written before hashes were stamped must NOT be reprocessed, even
    # if the row's hash cannot be matched: --process-all-context is the
    # explicit one-time baseline, not an implicit mass Gemini run.
    existing = [make_node("d1-c1", "src:doc-1", "src")]
    stored_hashes = build_doc_hash_map(existing)

    assert stored_hashes == {"src:doc-1": None}

    selected, stats = select_docs_to_process(
        [make_doc("src:doc-1", "sha256:new")], stored_hashes
    )

    assert selected == []
    assert stats == {"new": 0, "changed": 0, "legacy_unhashed": 1}


def test_new_doc_is_still_detected():
    existing = [make_node("d1-c1", "src:doc-1", "src", content_hash="sha256:a")]

    selected, stats = select_docs_to_process(
        [make_doc("src:doc-1", "sha256:a"), make_doc("src:doc-2", "sha256:b")],
        build_doc_hash_map(existing),
    )

    assert [doc["doc_id"] for doc in selected] == ["src:doc-2"]
    assert stats == {"new": 1, "changed": 0, "legacy_unhashed": 0}


def test_row_without_content_hash_is_treated_as_unchanged():
    # Older JSONL rows may lack content_hash entirely; with no basis for
    # comparison the doc must not be reprocessed.
    selected, stats = select_docs_to_process(
        [make_doc("src:doc-1", content_hash=None)], {"src:doc-1": "sha256:a"}
    )

    assert selected == []
    assert stats == {"new": 0, "changed": 0, "legacy_unhashed": 0}


def test_mixed_hash_doc_uses_any_stored_hash():
    # A doc with some hashed and some legacy-unhashed nodes (shouldn't exist,
    # but must resolve sanely): any stored hash counts, in either node order.
    hashed_first = [
        make_node("d1-c1", "src:doc-1", "src", content_hash="sha256:a"),
        make_node("d1-c2", "src:doc-1", "src"),
    ]
    unhashed_first = list(reversed(hashed_first))

    assert build_doc_hash_map(hashed_first) == {"src:doc-1": "sha256:a"}
    assert build_doc_hash_map(unhashed_first) == {"src:doc-1": "sha256:a"}

    selected, stats = select_docs_to_process(
        [make_doc("src:doc-1", "sha256:b")], build_doc_hash_map(unhashed_first)
    )

    assert [doc["doc_id"] for doc in selected] == ["src:doc-1"]
    assert stats == {"new": 0, "changed": 1, "legacy_unhashed": 0}


def test_nodes_with_undetermined_doc_id_are_skipped_in_hash_map():
    class OpaqueNode:
        """No doc_id, metadata, or source_node: doc_id lookup raises."""

    assert build_doc_hash_map(
        [OpaqueNode(), make_node("d1-c1", "src:doc-1", "src", content_hash="sha256:a")]
    ) == {"src:doc-1": "sha256:a"}


# --- Producer side: the context step must stamp the hash it reads back ---


def test_metadata_key_constant_matches_producer():
    # update_docs_workflow duplicates the key to stay import-light; the copies
    # must never drift from the producer's.
    from data.scraping_scripts.add_context_to_nodes import (
        DOC_CONTENT_HASH_METADATA_KEY as producer_key,
    )

    assert producer_key == DOC_CONTENT_HASH_METADATA_KEY


def test_process_stamps_doc_content_hash_on_every_chunk(tmp_path):
    # create_docs -> process is the single funnel for the full-rebuild path,
    # the incremental path, and create_vector_stores' non-pickle path; every
    # resulting chunk must carry the row's content_hash. No Gemini call: the
    # context generator is stubbed out.
    from data.scraping_scripts.add_context_to_nodes import create_docs, process

    row = make_doc("src:doc-1", "sha256:abc123")
    row["content"] = "# Doc One\n\nSome content about incremental context."
    jsonl_path = tmp_path / "docs.jsonl"
    jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    documents = create_docs(str(jsonl_path))
    assert documents[0].metadata["content_hash"] == "sha256:abc123"

    async def fake_situate_context(_doc: str, _chunk: str) -> str:
        return "situated"

    with patch(
        "data.scraping_scripts.add_context_to_nodes.situate_context",
        fake_situate_context,
    ):
        records = asyncio.run(process(documents, semaphore_limit=1))

    assert records
    assert all(
        record.metadata[DOC_CONTENT_HASH_METADATA_KEY] == "sha256:abc123"
        for record in records
    )
    # The selection round-trip closes: unchanged hash -> no reprocess,
    # changed hash -> reprocess.
    stored_hashes = build_doc_hash_map(records)
    unchanged, _ = select_docs_to_process([row], stored_hashes)
    changed, _ = select_docs_to_process(
        [make_doc("src:doc-1", "sha256:def456")], stored_hashes
    )
    assert unchanged == []
    assert [doc["doc_id"] for doc in changed] == ["src:doc-1"]


def test_legacy_row_without_hash_produces_unhashed_nodes(tmp_path):
    # Rows without content_hash (older JSONLs) must not stamp the field at
    # all, so those docs land in the legacy-unhashed bucket downstream.
    from data.scraping_scripts.add_context_to_nodes import create_docs, process

    row = make_doc("src:doc-1", content_hash=None)
    row["content"] = "# Doc One\n\nLegacy row without a content hash."
    jsonl_path = tmp_path / "docs.jsonl"
    jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    documents = create_docs(str(jsonl_path))
    assert "content_hash" not in documents[0].metadata

    async def fake_situate_context(_doc: str, _chunk: str) -> str:
        return "situated"

    with patch(
        "data.scraping_scripts.add_context_to_nodes.situate_context",
        fake_situate_context,
    ):
        records = asyncio.run(process(documents, semaphore_limit=1))

    assert records
    assert all(
        DOC_CONTENT_HASH_METADATA_KEY not in record.metadata for record in records
    )
    assert build_doc_hash_map(records) == {"src:doc-1": None}
