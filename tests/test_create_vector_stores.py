"""Tests for content-aware embedding reuse in create_vector_stores.

chunk_id (``f"{doc_id}:{index}"``) is stable across content edits, so reuse
keyed on ids alone served stale text/embeddings forever when an upstream
document changed in place. process_source now compares the stored Chroma
document text against the incoming record text and re-embeds on mismatch.

Uses a real chromadb PersistentClient on a tmp path; embed_texts is mocked
(no Cohere calls).
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import unittest
import uuid
from unittest.mock import patch

import chromadb

import data.scraping_scripts.create_vector_stores as cvs
from app.chroma_rag import ChunkRecord
from data.scraping_scripts.source_registry import ACTIVE_SOURCE_KEYS


class _OpaqueRecord:
    """A pickled record with no metadata anywhere: source is undeterminable."""


def _fake_embed(client, texts, **kwargs):
    return [[float(len(text)), 1.0] for text in texts]


def _record(doc_id: str, index: int, text: str, heading: str = "") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"{doc_id}:{index}",
        doc_id=doc_id,
        text=text,
        metadata={
            "doc_id": doc_id,
            "title": "Guide",
            "url": "https://example.com/guide",
            "source": "testsource",
            "retrieve_doc": False,
            "tokens": 100,
            "chunk_index": index,
            "heading_path": heading,
        },
    )


class ProcessSourceContentAwareReuseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._old_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        self.addCleanup(os.chdir, self._old_cwd)
        os.makedirs("data", exist_ok=True)

        # Unique per test: chromadb caches client instances by path, and
        # process_source uses paths relative to the (per-test) cwd, so a
        # repeated relative path would resurface another test's client.
        self.source = f"testsource-{uuid.uuid4().hex[:8]}"
        self.db_name = f"chroma-db-{self.source}"
        self.db_path = f"data/{self.db_name}"

        config = {
            "input_file": f"data/{self.source}_data.jsonl",
            "db_name": self.db_name,
            "document_dict_file": f"document_dict_{self.source}.pkl",
            "bm25_index_file": f"bm25_index_{self.source}.pkl",
        }
        with open(config["input_file"], "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "doc_id": "doc-1",
                        "content": "# Guide\n\nSome document content.",
                        "name": "Guide",
                        "url": "https://example.com/guide",
                        "source": self.source,
                        "retrieve_doc": False,
                        "tokens": 100,
                    }
                )
                + "\n"
            )

        for patcher in (
            patch.dict(cvs.SOURCE_CONFIGS, {self.source: config}),
            patch.dict(os.environ, {"COHERE_API_KEY": "fake-key"}),
            patch.object(cvs, "cohere"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run(self, records: list[ChunkRecord], **kwargs) -> list[list[str]]:
        """Run process_source with mocked embeddings; return embed call texts."""
        calls: list[list[str]] = []

        def tracking_embed(client, texts, **embed_kwargs):
            calls.append(list(texts))
            return _fake_embed(client, texts)

        with (
            patch.object(cvs, "load_or_create_chunk_records", return_value=records),
            patch.object(cvs, "embed_texts", side_effect=tracking_embed),
        ):
            cvs.process_source(self.source, **kwargs)
        return calls

    def _collection(self):
        return chromadb.PersistentClient(path=self.db_path).get_collection(
            name=self.db_name
        )

    def _stored_document(self, chunk_id: str) -> str | None:
        result = self._collection().get(ids=[chunk_id], include=["documents"])
        documents = result.get("documents") or [None]
        return documents[0]

    def test_unchanged_chunks_are_reused_without_reembedding(self) -> None:
        records = [_record("doc-1", 0, "chunk zero"), _record("doc-1", 1, "chunk one")]

        first_calls = self._run(records)
        self.assertEqual(first_calls, [["chunk zero", "chunk one"]])

        second_calls = self._run(records)
        self.assertEqual(second_calls, [])
        self.assertEqual(self._collection().count(), 2)

    def test_changed_content_with_same_chunk_id_is_reembedded(self) -> None:
        self._run([_record("doc-1", 0, "old text"), _record("doc-1", 1, "chunk one")])

        changed = _record("doc-1", 0, "new text", heading="Guide > Install")
        calls = self._run([changed, _record("doc-1", 1, "chunk one")])

        # Only the changed chunk is re-embedded; the unchanged one is reused.
        self.assertEqual(calls, [["new text"]])
        # Stored document text and metadata are refreshed for the changed id.
        self.assertEqual(self._stored_document("doc-1:0"), "new text")
        result = self._collection().get(ids=["doc-1:0"], include=["metadatas"])
        self.assertEqual(result["metadatas"][0]["heading_path"], "Guide > Install")
        self.assertEqual(self._stored_document("doc-1:1"), "chunk one")

    def test_removed_chunks_are_deleted(self) -> None:
        self._run([_record("doc-1", 0, "chunk zero"), _record("doc-1", 1, "chunk one")])

        self._run([_record("doc-1", 0, "chunk zero")])

        collection = self._collection()
        self.assertEqual(collection.count(), 1)
        self.assertEqual(collection.get(include=[])["ids"], ["doc-1:0"])

    def test_legacy_chunks_with_matching_stored_text_are_reused(self) -> None:
        # Pre-fix collections always stored documents at upsert time, so an
        # upgrade run over an unchanged corpus must not re-embed anything.
        record = _record("doc-1", 0, "legacy text")
        collection = chromadb.PersistentClient(
            path=self.db_path
        ).get_or_create_collection(name=self.db_name)
        collection.add(
            ids=[record.chunk_id],
            embeddings=[[1.0, 2.0]],
            documents=[record.text],
            metadatas=[record.metadata],
        )

        calls = self._run([record])

        self.assertEqual(calls, [])

    def test_chunk_missing_stored_document_is_reembedded(self) -> None:
        # Defensive path: a chunk stored without document text cannot be
        # compared, so it is treated as changed and rewritten.
        collection = chromadb.PersistentClient(
            path=self.db_path
        ).get_or_create_collection(name=self.db_name)
        collection.add(ids=["doc-1:0"], embeddings=[[1.0, 2.0]])

        calls = self._run([_record("doc-1", 0, "recovered text")])

        self.assertEqual(calls, [["recovered text"]])
        self.assertEqual(self._stored_document("doc-1:0"), "recovered text")

    def test_force_rebuild_reembeds_everything(self) -> None:
        records = [_record("doc-1", 0, "chunk zero")]
        self._run(records)

        # force_rebuild rmtree's the Chroma directory; drop chromadb's cached
        # client for the old path first, as a fresh CLI invocation would.
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        calls = self._run(records, force_rebuild=True)

        self.assertEqual(calls, [["chunk zero"]])


class LoadOrCreateChunkRecordsTestCase(unittest.TestCase):
    def test_unparseable_records_fail_open_with_warning(self) -> None:
        active_source = next(iter(ACTIVE_SOURCE_KEYS))
        active = ChunkRecord(
            chunk_id="doc-a:0",
            doc_id="doc-a",
            text="active",
            metadata={"doc_id": "doc-a", "source": active_source},
        )
        retired = ChunkRecord(
            chunk_id="doc-b:0",
            doc_id="doc-b",
            text="retired",
            metadata={"doc_id": "doc-b", "source": "definitely-retired-source"},
        )
        opaque = _OpaqueRecord()

        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                os.makedirs("data", exist_ok=True)
                with open("data/all_sources_contextual_nodes.pkl", "wb") as handle:
                    pickle.dump([opaque, active, retired], handle)

                with self.assertLogs(cvs.logger, level="WARNING") as logs:
                    kept = cvs.load_or_create_chunk_records("all_sources")
            finally:
                os.chdir(old_cwd)

        self.assertIn(active, kept)
        self.assertNotIn(retired, kept)
        self.assertTrue(any(isinstance(item, _OpaqueRecord) for item in kept))
        self.assertTrue(
            any("could not be determined" in message for message in logs.output)
        )
        self.assertTrue(any("_OpaqueRecord" in message for message in logs.output))


if __name__ == "__main__":
    unittest.main()
