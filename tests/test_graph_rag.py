from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.chat_types import ChatRequest
from app.graph_rag import (
    GRAPHRAG_COMMUNITY_SOURCE,
    GraphRAGIndexNotBuilt,
    GraphRAGRetriever,
    graphrag_index_exists,
)


class _FakeSearch:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def limit(self, _k: int) -> "_FakeSearch":
        return self

    def to_pandas(self) -> pd.DataFrame:
        return self._df


class _FakeEntityTable:
    """Stand-in for a LanceDB table: search(vec).limit(k).to_pandas()."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def search(self, _vec):
        return _FakeSearch(self._df)


def _bare_retriever() -> GraphRAGRetriever:
    """A retriever instance without running __init__ (no index/keys needed),
    used to unit-test the pure mapping logic offline."""
    return GraphRAGRetriever.__new__(GraphRAGRetriever)


class GraphRagIndexTestCase(unittest.TestCase):
    def test_index_exists_false_then_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output"
            out.mkdir(parents=True)
            from app import graph_rag

            original = graph_rag.GRAPHRAG_OUTPUT_DIR
            graph_rag.GRAPHRAG_OUTPUT_DIR = str(out)
            try:
                self.assertFalse(graphrag_index_exists())
                (out / "entities.parquet").write_text("x")
                self.assertTrue(graphrag_index_exists())
            finally:
                graph_rag.GRAPHRAG_OUTPUT_DIR = original

    def test_constructor_raises_when_index_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Index check happens before any key/cohere import, so a fake key
            # is fine; the missing output dir must raise.
            with self.assertRaises(GraphRAGIndexNotBuilt):
                GraphRAGRetriever(
                    cohere_api_key="fake",
                    output_dir=str(Path(tmp) / "missing"),
                )


class ChatRequestTestCase(unittest.TestCase):
    def test_retriever_defaults_to_classical(self) -> None:
        self.assertEqual(ChatRequest(query="hi").retriever, "")


class GraphRagMappingTestCase(unittest.TestCase):
    def test_entity_linked_text_units_map_to_real_source(self) -> None:
        r = _bare_retriever()
        r._entity_top_k = 10
        r._embed_query = lambda _q: [0.0, 0.0]  # type: ignore[assignment]
        r._entity_table = _FakeEntityTable(pd.DataFrame({"id": ["e1", "e2"]}))
        r._entities = pd.DataFrame(
            {
                "id": ["e1", "e2", "e3"],
                "text_unit_ids": [["t1"], ["t2"], ["t3"]],
            }
        )
        r._text_unit_by_id = {
            "t1": {
                "id": "t1",
                "text": "alpha",
                "document_id": "src_a:doc",
                "n_tokens": 3,
            },
            "t2": {
                "id": "t2",
                "text": "beta",
                "document_id": "src_b:doc",
                "n_tokens": 2,
            },
            "t3": {
                "id": "t3",
                "text": "gamma",
                "document_id": "src_a:doc2",
                "n_tokens": 1,
            },
        }
        r._manifest = {
            "src_a:doc": {"source": "src_a", "url": "http://a/1", "title": "A1"},
            "src_b:doc": {"source": "src_b", "url": "http://b/1", "title": "B1"},
        }

        # No source filter: top entities e1, e2 -> text units t1, t2.
        results = r._entity_linked_text_units("q", None)
        by_source = {res.source: res for res in results}
        self.assertIn("src_a", by_source)
        self.assertIn("src_b", by_source)
        self.assertEqual(by_source["src_a"].url, "http://a/1")
        self.assertEqual(by_source["src_a"].content, "alpha")
        self.assertEqual(by_source["src_a"].retrieval_method, "graphrag")

        # allowed_sources filter drops src_b.
        only_a = r._entity_linked_text_units("q", ["src_a"])
        self.assertEqual({res.source for res in only_a}, {"src_a"})

    def test_community_reports_are_context_only(self) -> None:
        r = _bare_retriever()
        r._community_top_k = 2
        r._reports = pd.DataFrame(
            {
                "community": [1, 2],
                "title": ["C1", "C2"],
                "full_content": ["report one", "report two"],
                "rank": [9.0, 5.0],
            }
        )
        results = r._community_context("q")
        self.assertEqual(len(results), 2)
        for res in results:
            # Synthetic source -> never matches a real ground-truth source, and
            # no url -> never resolves as a cited source card.
            self.assertEqual(res.source, GRAPHRAG_COMMUNITY_SOURCE)
            self.assertEqual(res.url, "")
        # Sorted by rank desc: highest-rank community first.
        self.assertEqual(results[0].title, "C1")


if __name__ == "__main__":
    unittest.main()
