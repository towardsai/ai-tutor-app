from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest

# The GraphRAG retriever is an optional eval-only extra (see pyproject
# [project.optional-dependencies].graphrag). Skip this module unless it is
# installed, so a default/prod `uv sync` (which omits it) still collects clean.
pytest.importorskip("graphrag", reason="install the optional 'graphrag' extra to run")

import pandas as pd

from app.chat_types import ChatRequest
from app.chroma_rag import SearchResult, get_token_encoding
from app.graph_rag import (
    COMMUNITY_RERANK_CANDIDATES,
    GRAPHRAG_COMMUNITY_SOURCE,
    GraphRAGIndexNotBuilt,
    GraphRAGRetriever,
    graphrag_index_exists,
)


class _FakeRerankItem:
    def __init__(self, index: int, score: float) -> None:
        self.index = index
        self.relevance_score = score


class _FakeRerankResponse:
    def __init__(self, items: list[_FakeRerankItem]) -> None:
        self.results = items


class _FakeCohere:
    """Captures rerank calls; ranks documents containing the query text first."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def rerank(self, *, model, query, documents, top_n):  # type: ignore[no-untyped-def]
        self.calls.append(
            {"model": model, "query": query, "documents": list(documents)}
        )
        order = sorted(
            range(len(documents)),
            key=lambda i: (query.lower() in documents[i].lower(), -i),
            reverse=True,
        )
        return _FakeRerankResponse(
            [
                _FakeRerankItem(doc_index, 1.0 - rank * 0.1)
                for rank, doc_index in enumerate(order[:top_n])
            ]
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
            self.assertFalse(graphrag_index_exists(str(out)))
            (out / "entities.parquet").write_text("x")
            self.assertTrue(graphrag_index_exists(str(out)))

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
        r._entity_text_units = {"e1": ["t1"], "e2": ["t2"], "e3": ["t3"]}
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
        # With candidates <= community_top_k every report is returned without a
        # rerank call (no _cohere set on the bare retriever proves that).
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
            self.assertEqual(res.score, 0.0)
        # Sorted by rank desc: highest-rank community first.
        self.assertEqual(results[0].title, "C1")

    def test_community_reports_selected_by_query_relevance(self) -> None:
        # The query, not the static community rank, decides which reports come
        # back: the top-ranked candidates are reranked against the query.
        r = _bare_retriever()
        r._community_top_k = 1
        r._rerank_model = "fake-rerank"
        fake = _FakeCohere()
        r._cohere = fake
        r._reports = pd.DataFrame(
            {
                "community": [1, 2, 3, 4],
                "title": ["C1", "C2", "C3", "C4"],
                "full_content": [
                    "agents and tools",
                    "prompt engineering tips",
                    "vector databases overview",
                    "fine-tuning walkthrough",
                ],
                "rank": [9.0, 8.0, 7.0, 6.0],
            }
        )

        results = r._community_context("vector databases")

        # Not C1 (highest static rank): the query-relevant report wins.
        self.assertEqual([res.title for res in results], ["C3"])
        # Every candidate reached the reranker in one call.
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len(fake.calls[0]["documents"]), 4)
        # Still context-only: synthetic source, no url, score pinned to 0.0.
        for res in results:
            self.assertEqual(res.source, GRAPHRAG_COMMUNITY_SOURCE)
            self.assertEqual(res.url, "")
            self.assertEqual(res.score, 0.0)
            self.assertEqual(res.retrieval_method, "graphrag_community")

    def test_community_rerank_candidate_pool_is_bounded(self) -> None:
        # Cost stays bounded: only the top COMMUNITY_RERANK_CANDIDATES reports
        # by static rank are sent to the reranker.
        r = _bare_retriever()
        r._community_top_k = 1
        r._rerank_model = "fake-rerank"
        fake = _FakeCohere()
        r._cohere = fake
        total = COMMUNITY_RERANK_CANDIDATES + 5
        r._reports = pd.DataFrame(
            {
                "community": list(range(total)),
                "title": [f"C{i}" for i in range(total)],
                "full_content": [f"report {i}" for i in range(total)],
                "rank": [float(total - i) for i in range(total)],
            }
        )

        r._community_context("q")

        documents = fake.calls[0]["documents"]
        self.assertEqual(len(documents), COMMUNITY_RERANK_CANDIDATES)
        self.assertIn("report 0", documents)  # highest static rank kept
        self.assertNotIn(f"report {total - 1}", documents)  # lowest dropped


class GraphRagTokenBudgetTestCase(unittest.TestCase):
    def _retriever(self) -> GraphRAGRetriever:
        r = _bare_retriever()
        r._encoding = get_token_encoding(None)
        r._token_budget = 100_000
        return r

    def _result(
        self,
        chunk_id: str,
        score: float,
        *,
        content: str = "x",
        source: str = "src_a",
        method: str = "graphrag",
    ) -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            doc_id=chunk_id,
            title=chunk_id,
            url="",
            source=source,
            retrieve_doc=False,
            tokens=10,
            score=score,
            content=content,
            chunk_content=content,
            heading_path="",
            retrieval_method=method,
        )

    def test_budget_skips_oversized_result_and_fills_with_smaller(self) -> None:
        # Same regression as the classical retriever: an oversized rank-1
        # result under a small per-request budget must not empty the list.
        r = self._retriever()
        oversized = self._result("big", 0.9, content="word " * 500)
        small = self._result("small", 0.8, content="word " * 15)

        kept = r._apply_token_budget([oversized, small], token_budget=50)

        self.assertEqual([res.chunk_id for res in kept], ["small"])

    def test_low_score_floor_exempts_community_reports(self) -> None:
        # Fairness with the classical arm: weak reranked text units are dropped
        # by the same score floor, but community reports (context-only chunks
        # pinned at score 0.0) are exempt.
        r = self._retriever()
        strong = self._result("strong", 0.5)
        weak = self._result("weak", 0.05)
        community = self._result(
            "community:1",
            0.0,
            source=GRAPHRAG_COMMUNITY_SOURCE,
            method="graphrag_community",
        )

        kept = r._apply_token_budget([strong, weak, community], token_budget=None)

        self.assertEqual([res.chunk_id for res in kept], ["strong", "community:1"])


if __name__ == "__main__":
    unittest.main()
