from __future__ import annotations

import asyncio
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data.scraping_scripts.add_context_to_nodes import process
from data.scraping_scripts.create_vector_stores import write_retrieval_artifacts
from llama_index.core import Document
from scripts.chroma_rag import (
    BM25Index,
    ChunkRecord,
    build_chunk_records,
    heading_aware_markdown_chunks,
    reciprocal_rank_fusion,
    load_bm25_index,
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

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp_dir:
            db_path = Path(temp_dir)
            count = write_retrieval_artifacts(
                config={
                    "document_dict_file": "document_dict_test.pkl",
                    "bm25_index_file": "bm25_index_test.pkl",
                },
                document_rows=document_rows,
                db_path=str(db_path),
            )

            document_dict_path = db_path / "document_dict_test.pkl"
            bm25_path = db_path / "bm25_index_test.pkl"

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

    def _result(self, chunk_id: str, score: float, method: str) -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            doc_id=chunk_id,
            title=chunk_id,
            url="",
            source="test",
            retrieve_doc=False,
            tokens=10,
            score=score,
            content=chunk_id,
            chunk_content=chunk_id,
            heading_path="section",
            retrieval_method=method,
        )


if __name__ == "__main__":
    unittest.main()
