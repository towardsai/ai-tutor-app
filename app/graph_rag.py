"""GraphRAG retriever for the GraphRAG-vs-classical-RAG experiment.

This is the second retrieval backend behind ``retrieve_tutor_context`` (the
default is the hybrid ``LocalChromaRetriever`` in ``chroma_rag.py``). It reads a
prebuilt Microsoft GraphRAG index (entities, relationships, text units, and
community reports) produced offline by ``graphrag index`` over our corpus, and
exposes the SAME interface as ``LocalChromaRetriever``::

    retriever.search(query, allowed_sources=..., token_budget=...) -> list[SearchResult]

so it drops straight into the agent tool and grades with the existing eval
metrics (the matches it returns carry the real ``source``/``url`` recovered from
``corpus_manifest`` via each text unit's ``document_id``).

Design constraints for the experiment (see evals.md / branch
experiment/graphrag-vs-rag):

* **Context provider only, no generation.** GraphRAG normally runs an LLM to
  synthesize an answer in local/global search. We must NOT do that: the eval's
  only generation model is the agent's 3.5 Flash. So this assembles the
  GraphRAG *context* (entity-linked text units + relevant community reports)
  and returns it as ``SearchResult`` chunks; the agent writes the answer.
* **Same rerank/budget as classical RAG** so the comparison is fair: candidate
  text units are reranked with Cohere and capped by the same token budget.
* The graph index is **local-only** (``data/graphrag/output``); it is never part
  of the HF vector-db bundle, so this backend is unavailable unless the index
  has been built on the machine.

The index is built with Gemini 2.5 Flash; this module only reads its artifacts.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .chroma_rag import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_RERANK_MODEL,
    RERANK_SCORE_FLOOR,
    SearchResult,
    get_token_encoding,
    rerank_results,
)

logger = logging.getLogger(__name__)

GRAPHRAG_DIR = os.getenv("AI_TUTOR_GRAPHRAG_DIR", "data/graphrag")
GRAPHRAG_OUTPUT_DIR = f"{GRAPHRAG_DIR}/output"
GRAPHRAG_LANCEDB_DIR = f"{GRAPHRAG_OUTPUT_DIR}/lancedb"
# Embedding model used at INDEX time (settings.yaml embedding_models). The query
# must be embedded with the same model for entity similarity to be meaningful.
GRAPHRAG_EMBED_MODEL_LITELLM = "openai/text-embedding-3-small"

# Synthetic source key for community-report context (graph-level synthesis with
# no single backing document). It is intentionally NOT a real corpus source, so
# recall@source can never falsely credit it.
GRAPHRAG_COMMUNITY_SOURCE = "graphrag_community"

# Local-search-ish knobs.
DEFAULT_ENTITY_TOP_K = 20
DEFAULT_COMMUNITY_TOP_K = 3
DEFAULT_RERANK_TOP_K = 5
# Candidate pool for query-relevant community-report selection: only the top
# reports by the index's static rank are reranked against the query, so the
# per-query cost stays bounded (one rerank call over at most this many docs).
COMMUNITY_RERANK_CANDIDATES = 30


class GraphRAGIndexNotBuilt(RuntimeError):
    """Raised when the GraphRAG output artifacts are missing.

    Build them with::

        uv run -m data.scraping_scripts.graphrag_prep_input
        uv run --env-file .env graphrag index --root data/graphrag
    """


def graphrag_index_exists(output_dir: str = GRAPHRAG_OUTPUT_DIR) -> bool:
    out = Path(output_dir)
    return out.is_dir() and any(out.glob("entities.parquet"))


def _load_manifest_lookup(manifest_path: str) -> dict[str, dict[str, Any]]:
    """doc_id -> {source, url, title} from corpus_manifest.jsonl."""
    import json

    lookup: dict[str, dict[str, Any]] = {}
    if not os.path.exists(manifest_path):
        return lookup
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = record.get("doc_id")
            if doc_id:
                lookup[doc_id] = {
                    "source": record.get("source", ""),
                    "url": record.get("url", ""),
                    "title": record.get("title", ""),
                }
    return lookup


class GraphRAGRetriever:
    """Local-search-style context assembly over a prebuilt GraphRAG index.

    NOTE: the exact LanceDB table name and parquet column names are validated
    against a real ``graphrag index`` run (the smoke index) before this backend
    is used in an eval; the loaders below discover them defensively.
    """

    def __init__(
        self,
        *,
        cohere_api_key: str,
        output_dir: str = GRAPHRAG_OUTPUT_DIR,
        lancedb_dir: str = GRAPHRAG_LANCEDB_DIR,
        manifest_path: str | None = None,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        embed_model_litellm: str = GRAPHRAG_EMBED_MODEL_LITELLM,
        entity_top_k: int = DEFAULT_ENTITY_TOP_K,
        community_top_k: int = DEFAULT_COMMUNITY_TOP_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
        answer_model_name: str | None = None,
    ) -> None:
        if not graphrag_index_exists(output_dir):
            raise GraphRAGIndexNotBuilt(
                f"GraphRAG index not found at {output_dir}. Build it with "
                "`uv run -m data.scraping_scripts.graphrag_prep_input` then "
                "`uv run --env-file .env graphrag index --root data/graphrag`."
            )
        import cohere
        from .config import KB_MANIFEST_PATH

        self._output_dir = output_dir
        self._lancedb_dir = lancedb_dir
        self._rerank_model = rerank_model
        self._embed_model_litellm = embed_model_litellm
        self._entity_top_k = entity_top_k
        self._community_top_k = community_top_k
        self._rerank_top_k = rerank_top_k
        self._token_budget = token_budget
        self._encoding = get_token_encoding(answer_model_name)
        self._cohere = cohere.ClientV2(api_key=cohere_api_key)
        self._manifest = _load_manifest_lookup(manifest_path or KB_MANIFEST_PATH)

        self._load_tables()

    def _load_tables(self) -> None:
        """Load GraphRAG output parquets and open the entity-embedding store.

        Column/table names follow the graphrag 3.1 data model
        (TextUnit.document_id, Entity.text_unit_ids/community_ids); confirmed
        against the smoke index output before first eval use.
        """
        import pandas as pd

        out = Path(self._output_dir)
        self._entities = pd.read_parquet(out / "entities.parquet")
        self._text_units = pd.read_parquet(out / "text_units.parquet")
        reports_path = out / "community_reports.parquet"
        self._reports = pd.read_parquet(reports_path) if reports_path.exists() else None
        # entity id -> its text unit ids, and text unit id -> row, both O(1) so a
        # query only touches the top-k nearest entities (not every entity/chunk).
        self._entity_text_units = {
            str(eid): list(tus) if tus is not None else []
            for eid, tus in zip(self._entities["id"], self._entities["text_unit_ids"])
        }
        self._text_unit_by_id = {
            str(row["id"]): row for _, row in self._text_units.iterrows()
        }
        self._entity_table = self._open_entity_vector_table()

    def _open_entity_vector_table(self):
        """Open the LanceDB table holding entity-description embeddings.

        graphrag names it like ``default-entity-description``; discover it by
        pattern so a version bump doesn't silently break retrieval.
        """
        try:
            import lancedb

            db = lancedb.connect(self._lancedb_dir)
            names = list(db.table_names())
        except Exception as exc:  # pragma: no cover - exercised post-index
            logger.warning("GraphRAG LanceDB open failed (%s); entity search off.", exc)
            return None
        preferred = [n for n in names if "entity" in n and "description" in n]
        chosen = preferred or [n for n in names if "entity" in n] or names
        if not chosen:
            logger.warning("No GraphRAG LanceDB tables found in %s.", self._lancedb_dir)
            return None
        return db.open_table(chosen[0])

    def _embed_query(self, query: str) -> list[float]:
        import litellm

        response = litellm.embedding(model=self._embed_model_litellm, input=[query])
        return list(response.data[0]["embedding"])

    def search(
        self,
        query: str,
        *,
        allowed_sources: list[str] | None = None,
        token_budget: int | None = None,
    ) -> list[SearchResult]:
        """Assemble GraphRAG context as reranked SearchResult chunks.

        Steps: embed query -> nearest entities (LanceDB) -> their text units
        (mapped to real source/url via document_id) + query-relevant community
        reports -> Cohere rerank -> token budget. Returns [] on any failure so the agent
        degrades to KB browsing, matching the classical backend's contract.
        """
        try:
            text_unit_hits = self._entity_linked_text_units(query, allowed_sources)
            reranked = self._rerank(query, text_unit_hits)
            community_hits = self._community_context(query)
            return self._apply_token_budget(reranked + community_hits, token_budget)
        except Exception as exc:  # pragma: no cover - exercised post-index
            logger.warning("GraphRAG search failed; returning no matches. %s", exc)
            return []

    # --- internals (finalized against the real index) ---------------------

    def _entity_linked_text_units(
        self, query: str, allowed_sources: list[str] | None
    ) -> list[SearchResult]:
        if self._entity_table is None:
            return []
        query_vec = self._embed_query(query)
        hits = (
            self._entity_table.search(query_vec).limit(self._entity_top_k).to_pandas()
        )
        # Walk the nearest entities in similarity order, collecting the text
        # units they were extracted from (deduped, order preserved).
        text_unit_ids: list[str] = []
        seen: set[str] = set()
        for entity_id in hits.get("id", []):
            for tu_id in self._entity_text_units.get(str(entity_id), []):
                tu_id = str(tu_id)
                if tu_id not in seen:
                    seen.add(tu_id)
                    text_unit_ids.append(tu_id)
        results: list[SearchResult] = []
        allowed = set(allowed_sources or [])
        for tu_id in text_unit_ids:
            row = self._text_unit_by_id.get(tu_id)
            if row is None:
                continue
            doc_id = str(row.get("document_id") or "")
            meta = self._manifest.get(doc_id, {})
            source = meta.get("source", "")
            if allowed and source not in allowed:
                continue
            text = str(row.get("text") or "")
            results.append(
                SearchResult(
                    chunk_id=tu_id,
                    doc_id=doc_id,
                    title=meta.get("title", "") or doc_id,
                    url=meta.get("url", ""),
                    source=source or "unknown",
                    retrieve_doc=False,
                    tokens=int(row.get("n_tokens") or 0),
                    score=0.0,
                    content=text,
                    chunk_content=text,
                    heading_path="",
                    retrieval_method="graphrag",
                )
            )
        return results

    def _rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []
        return rerank_results(
            self._cohere,
            query,
            results,
            model=self._rerank_model,
            top_n=self._rerank_top_k,
        )

    def _community_context(self, query: str) -> list[SearchResult]:
        """Query-relevant community reports as context-only chunks (synthetic
        source, no url -> no recall/citation credit, by design).

        Candidates are the ``COMMUNITY_RERANK_CANDIDATES`` highest-ranked
        reports (the index's static rank); the ``community_top_k`` most relevant
        to the query are selected with the same Cohere rerank used for text
        units. Scores are pinned back to 0.0 so the reports stay context-only
        chunks (and stay exempt from the rerank-score floor in
        ``_apply_token_budget``).
        """
        if self._reports is None or self._community_top_k <= 0:
            return []
        reports = self._reports
        if "rank" in reports.columns:
            reports = reports.sort_values("rank", ascending=False)
        candidates: list[SearchResult] = []
        for _, row in reports.head(COMMUNITY_RERANK_CANDIDATES).iterrows():
            content = str(row.get("full_content") or row.get("summary") or "")
            if not content:
                continue
            candidates.append(
                SearchResult(
                    chunk_id=f"community:{row.get('community', '')}",
                    doc_id="",
                    title=str(row.get("title") or "Community report"),
                    url="",
                    source=GRAPHRAG_COMMUNITY_SOURCE,
                    retrieve_doc=False,
                    tokens=0,
                    score=0.0,
                    content=content,
                    chunk_content=content,
                    heading_path="",
                    retrieval_method="graphrag_community",
                )
            )
        if len(candidates) <= self._community_top_k:
            return candidates
        selected = rerank_results(
            self._cohere,
            query,
            candidates,
            model=self._rerank_model,
            top_n=self._community_top_k,
        )
        for result in selected:
            result.score = 0.0
        return selected

    def _apply_token_budget(
        self, results: list[SearchResult], token_budget: int | None
    ) -> list[SearchResult]:
        budget = self._token_budget if token_budget is None else token_budget
        filtered: list[SearchResult] = []
        total = 0
        for result in results:
            # Same low-relevance floor as the classical retriever, so both eval
            # arms drop weak reranked hits (the module's fairness contract).
            # Community reports are context-only chunks pinned at score 0.0 and
            # are exempt, or the floor would drop them all.
            if (
                result.source != GRAPHRAG_COMMUNITY_SOURCE
                and result.score < RERANK_SCORE_FLOOR
            ):
                continue
            result_tokens = len(
                self._encoding.encode(result.content, disallowed_special=())
            )
            if total + result_tokens > budget:
                # Skip an oversized result instead of cutting off the whole
                # list; keep filling the budget with later results that fit.
                continue
            total += result_tokens
            filtered.append(result)
        return filtered


@lru_cache(maxsize=1)
def _build_graphrag_retriever() -> GraphRAGRetriever:
    cohere_api_key = os.environ["COHERE_API_KEY"]
    return GraphRAGRetriever(cohere_api_key=cohere_api_key)


def get_graphrag_retriever() -> GraphRAGRetriever:
    return _build_graphrag_retriever()
