from __future__ import annotations

import asyncio
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb
import tiktoken
from data.scraping_scripts.add_context_to_nodes import process
from data.scraping_scripts.create_vector_stores import write_retrieval_artifacts
from llama_index.core import Document
from app.chroma_rag import (
    BM25Index,
    ChunkRecord,
    LocalChromaRetriever,
    build_chunk_records,
    heading_aware_markdown_chunks,
    reciprocal_rank_fusion,
    load_bm25_index,
    rerank_results,
    SearchResult,
)


class ChromaRagTestCase(unittest.TestCase):
    def test_heading_aware_chunks_keep_code_blocks_intact(self) -> None:
        code_lines = "\n".join(f"print({index})" for index in range(120))
        markdown = f"""# Guide

## Install

Use `pip install`.

## Example

```python
{code_lines}
```

After the example.
"""

        chunks = heading_aware_markdown_chunks(
            markdown,
            title="Guide",
            chunk_size=80,
        )

        code_chunks = [chunk for chunk in chunks if "print(0)" in chunk.text]
        self.assertEqual(len(code_chunks), 1)
        self.assertIn("print(119)", code_chunks[0].text)
        self.assertIn("Example", code_chunks[0].heading_path)

    def test_build_chunk_records_adds_heading_metadata(self) -> None:
        records = build_chunk_records(
            [
                {
                    "doc_id": "doc-1",
                    "name": "Guide",
                    "url": "https://example.com/guide",
                    "source": "transformers",
                    "retrieve_doc": False,
                    "tokens": 1000,
                    "content": "# Guide\n\n## Install\n\nUse `AutoModel`.",
                }
            ]
        )

        self.assertEqual(records[0].metadata["heading_path"], "Guide")
        self.assertEqual(records[1].metadata["heading_path"], "Guide > Install")
        self.assertIn("source_version", records[0].metadata)

    def test_bm25_search_finds_keywords_and_filters_sources(self) -> None:
        records = [
            ChunkRecord(
                chunk_id="a",
                doc_id="doc-a",
                text="Use AutoModel.from_pretrained for model loading.",
                metadata={"doc_id": "doc-a", "source": "transformers"},
            ),
            ChunkRecord(
                chunk_id="b",
                doc_id="doc-b",
                text="Create a prompt template for chains.",
                metadata={"doc_id": "doc-b", "source": "langchain"},
            ),
        ]
        index = BM25Index.build(records)

        hits = index.search(
            "AutoModel.from_pretrained", allowed_sources=["transformers"]
        )

        self.assertEqual([record.chunk_id for record, _score in hits], ["a"])
        self.assertEqual(index.search("AutoModel", allowed_sources=["langchain"]), [])

    def test_retrieval_artifact_writer_persists_bm25_and_document_dict(self) -> None:
        document_rows = [
            {
                "doc_id": "doc-1",
                "name": "Transformers Loading",
                "url": "https://example.com/loading",
                "source": "transformers",
                "retrieve_doc": False,
                "tokens": 1200,
                "content": "# Loading\n\n## AutoModel\n\nUse `AutoModel.from_pretrained`.",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir)
            count = write_retrieval_artifacts(
                config={
                    "document_dict_file": "document_dict_test.pkl",
                    "bm25_index_file": "bm25_index_test.json.gz",
                },
                document_rows=document_rows,
                db_path=str(db_path),
            )

            document_dict_path = db_path / "document_dict_test.pkl"
            bm25_path = db_path / "bm25_index_test.json.gz"

            self.assertGreaterEqual(count, 1)
            self.assertTrue(document_dict_path.exists())
            self.assertTrue(bm25_path.exists())

            with open(document_dict_path, "rb") as handle:
                document_dict = pickle.load(handle)
            self.assertEqual(document_dict["doc-1"]["name"], "Transformers Loading")

            index = load_bm25_index(str(bm25_path))
            self.assertIsNotNone(index)
            assert index is not None
            hits = index.search("AutoModel.from_pretrained")
            self.assertEqual(hits[0][0].doc_id, "doc-1")
            self.assertTrue(
                any(record.metadata["heading_path"] for record in index.records)
            )

    def test_context_processing_uses_heading_chunks_and_raw_text_metadata(self) -> None:
        async def fake_situate_context(_doc: str, chunk: str) -> str:
            return f"Situated {chunk.splitlines()[0]}"

        document = Document(
            doc_id="doc-1",
            text="# Guide\n\n## Setup\n\nUse `AutoModel.from_pretrained`.",
            metadata={
                "title": "Guide",
                "url": "https://example.com/guide",
                "tokens": 1000,
                "retrieve_doc": False,
                "source": "transformers",
            },
        )

        with patch(
            "data.scraping_scripts.add_context_to_nodes.situate_context",
            fake_situate_context,
        ):
            records = asyncio.run(process([document], semaphore_limit=1))

        self.assertGreaterEqual(len(records), 1)
        setup_record = next(
            record
            for record in records
            if record.metadata["heading_path"] == "Guide > Setup"
        )
        self.assertIn("raw_text", setup_record.metadata)
        self.assertIn("Title: Guide", setup_record.text)
        self.assertIn("Heading path: Guide > Setup", setup_record.text)
        self.assertIn("Context: Situated", setup_record.text)

    def test_rerank_scores_matched_chunk_for_retrieve_doc_results(self) -> None:
        # retrieve_doc results carry the whole document in `content`; the
        # reranker must score the matched chunk (`chunk_content`) instead, so
        # relevance is not diluted toward the doc average and the payload stays
        # within Cohere's per-document token limit.
        full_doc = "Intro paragraph.\n" * 500
        results = [
            SearchResult(
                chunk_id="doc-chunk",
                doc_id="doc-1",
                title="Doc",
                url="",
                source="test",
                retrieve_doc=True,
                tokens=4000,
                score=0.5,
                content=full_doc,
                chunk_content="the matched chunk about AutoModel",
                heading_path="section",
                retrieval_method="dense",
            ),
            SearchResult(
                chunk_id="plain-chunk",
                doc_id="doc-2",
                title="Plain",
                url="",
                source="test",
                retrieve_doc=False,
                tokens=100,
                score=0.4,
                content="formatted chunk body",
                chunk_content="raw chunk body",
                heading_path="section",
                retrieval_method="dense",
            ),
        ]

        captured: dict[str, list[str]] = {}

        class _FakeItem:
            def __init__(self, index: int, score: float) -> None:
                self.index = index
                self.relevance_score = score

        class _FakeResponse:
            def __init__(self, items: list["_FakeItem"]) -> None:
                self.results = items

        class _FakeCohere:
            def rerank(self, *, model, query, documents, top_n):  # type: ignore[no-untyped-def]
                captured["documents"] = list(documents)
                return _FakeResponse(
                    [
                        _FakeItem(i, 1.0 - i * 0.1)
                        for i in range(min(top_n, len(documents)))
                    ]
                )

        reranked = rerank_results(_FakeCohere(), "AutoModel", results)

        # The full document never reaches the reranker; the matched chunk does.
        self.assertEqual(
            captured["documents"],
            ["the matched chunk about AutoModel", "formatted chunk body"],
        )
        # The returned result still carries the full document for the answer.
        self.assertEqual(reranked[0].content, full_doc)

    def test_rrf_prefers_overlap_across_ranked_lists(self) -> None:
        dense_only = self._result("dense-only", 0.9, "dense")
        overlap_dense = self._result("overlap", 0.7, "dense")
        overlap_bm25 = self._result("overlap", 4.0, "bm25")
        bm25_only = self._result("bm25-only", 5.0, "bm25")

        fused = reciprocal_rank_fusion(
            [[dense_only, overlap_dense], [bm25_only, overlap_bm25]],
            top_k=4,
        )

        self.assertEqual(fused[0].chunk_id, "overlap")
        self.assertEqual(fused[0].retrieval_method, "hybrid")

    def test_rrf_counts_each_key_once_per_ranked_list(self) -> None:
        # A section split into several chunks can land at multiple ranks of ONE
        # retriever's list (same dedupe key). Standard RRF scores a key once per
        # list, at its best rank; per-occurrence accumulation would let one
        # retriever's duplicates masquerade as cross-retriever consensus.
        dup_top = self._result("dup:0", 0.9, "dense", doc_id="dup-doc")
        dup_mid = self._result("dup:1", 0.8, "dense", doc_id="dup-doc")
        dup_low = self._result("dup:2", 0.7, "dense", doc_id="dup-doc")
        consensus_dense = self._result("uni:0", 0.6, "dense", doc_id="consensus-doc")
        consensus_bm25 = self._result("uni:0", 5.0, "bm25", doc_id="consensus-doc")

        fused = reciprocal_rank_fusion(
            [[dup_top, dup_mid, dup_low, consensus_dense], [consensus_bm25]],
            top_k=5,
        )

        by_doc = {result.doc_id: result for result in fused}
        # One contribution at the best rank (1), nothing from ranks 2-3.
        self.assertAlmostEqual(by_doc["dup-doc"].score, 1.0 / 61)
        # Rank 4 in dense + rank 1 in bm25.
        self.assertAlmostEqual(by_doc["consensus-doc"].score, 1.0 / 64 + 1.0 / 61)
        # Genuine cross-retriever consensus outranks single-list duplication.
        self.assertEqual(fused[0].doc_id, "consensus-doc")
        # Representative selection still works: best-scoring dense duplicate.
        self.assertEqual(by_doc["dup-doc"].chunk_id, "dup:0")

    def _result(
        self,
        chunk_id: str,
        score: float,
        method: str,
        *,
        doc_id: str | None = None,
        content: str | None = None,
        retrieve_doc: bool = False,
    ) -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            doc_id=doc_id if doc_id is not None else chunk_id,
            title=chunk_id,
            url="",
            source="test",
            retrieve_doc=retrieve_doc,
            tokens=10,
            score=score,
            content=content if content is not None else chunk_id,
            chunk_content=content if content is not None else chunk_id,
            heading_path="section",
            retrieval_method=method,
        )


class TokenBudgetTestCase(unittest.TestCase):
    def _retriever(self) -> LocalChromaRetriever:
        retriever = LocalChromaRetriever.__new__(LocalChromaRetriever)
        retriever._encoding = tiktoken.get_encoding("cl100k_base")
        retriever._token_budget = 100_000
        return retriever

    def _result(self, chunk_id: str, score: float, content: str) -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            doc_id=chunk_id,
            title=chunk_id,
            url="",
            source="test",
            retrieve_doc=False,
            tokens=10,
            score=score,
            content=content,
            chunk_content=content,
            heading_path="section",
            retrieval_method="dense",
        )

    def test_budget_skips_oversized_result_and_fills_with_smaller(self) -> None:
        # A rank-1 retrieve_doc result whose full document exceeds a small
        # per-request budget must not empty the whole result list: it is
        # skipped and the budget is filled with lower-ranked results that fit,
        # in rank order.
        retriever = self._retriever()
        oversized = self._result("big", 0.9, "word " * 500)
        small_one = self._result("small-1", 0.8, "word " * 15)
        small_two = self._result("small-2", 0.7, "word " * 15)

        kept = retriever._apply_token_budget(
            [oversized, small_one, small_two], token_budget=50
        )
        self.assertEqual([result.chunk_id for result in kept], ["small-1", "small-2"])

        # With the default (large) budget everything still fits.
        kept_all = retriever._apply_token_budget([oversized, small_one, small_two])
        self.assertEqual(len(kept_all), 3)


class CollectionOpenTestCase(unittest.TestCase):
    def _write_document_dict(self, directory: str) -> str:
        path = Path(directory) / "document_dict_test.pkl"
        with open(path, "wb") as handle:
            pickle.dump({}, handle)
        return str(path)

    def test_init_fails_loudly_when_collection_missing(self) -> None:
        # A broken/mismatched bundle must raise at startup instead of silently
        # creating an empty collection that returns zero dense hits forever.
        with tempfile.TemporaryDirectory() as temp_dir:
            document_dict_path = self._write_document_dict(temp_dir)

            with self.assertRaises(RuntimeError) as ctx:
                LocalChromaRetriever(
                    db_path=temp_dir,
                    collection_name="missing-collection",
                    document_dict_path=document_dict_path,
                    cohere_api_key="fake",
                )

            message = str(ctx.exception)
            self.assertIn("missing-collection", message)
            self.assertIn(temp_dir, message)

    def test_init_opens_collection_created_beforehand(self) -> None:
        # Mirrors production: create_vector_stores creates the collection; the
        # retriever only opens it.
        with tempfile.TemporaryDirectory() as temp_dir:
            chromadb.PersistentClient(path=temp_dir).create_collection(
                name="test-collection"
            )
            document_dict_path = self._write_document_dict(temp_dir)

            retriever = LocalChromaRetriever(
                db_path=temp_dir,
                collection_name="test-collection",
                document_dict_path=document_dict_path,
                cohere_api_key="fake",
            )

            self.assertEqual(retriever._collection.name, "test-collection")


if __name__ == "__main__":
    unittest.main()
