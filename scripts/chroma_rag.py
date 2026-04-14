from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import chromadb
import cohere
import logfire
import tiktoken
from tqdm.auto import tqdm


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_DENSE_TOP_K = 15
DEFAULT_RERANK_TOP_K = 5
DEFAULT_CONTEXT_TOKEN_BUDGET = 100_000
DEFAULT_EMBED_MODEL = "embed-v4.0"
DEFAULT_RERANK_MODEL = "rerank-v4.0-fast"
DEFAULT_ENCODING = "cl100k_base"
DEFAULT_OUTPUT_DIMENSION = 1024


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class SearchResult:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    source: str
    retrieve_doc: bool
    tokens: int
    score: float
    content: str
    chunk_content: str


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_jsonl_documents(input_file: str) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(json.loads(line))
    return documents


def get_token_encoding(model_name: str | None = None) -> tiktoken.Encoding:
    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception:
            pass
    return tiktoken.get_encoding(DEFAULT_ENCODING)


def token_window_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[str]:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text, disallowed_special=())
    if not tokens:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            continue
        decoded = encoding.decode(window).strip()
        if decoded:
            chunks.append(decoded)
        if start + chunk_size >= len(tokens):
            break
    return chunks


def build_chunk_records(
    documents: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    chunk_records: list[ChunkRecord] = []
    for document in documents:
        chunks = token_window_chunks(
            document["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for index, chunk_text in enumerate(chunks):
            metadata = {
                "doc_id": document["doc_id"],
                "title": document["name"],
                "url": document["url"],
                "source": document["source"],
                "retrieve_doc": document["retrieve_doc"],
                "tokens": document["tokens"],
                "chunk_index": index,
            }
            chunk_records.append(
                ChunkRecord(
                    chunk_id=f'{document["doc_id"]}:{index}',
                    doc_id=document["doc_id"],
                    text=chunk_text,
                    metadata=metadata,
                )
            )
    return chunk_records


def build_document_dict(
    documents: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        document["doc_id"]: {
            "doc_id": document["doc_id"],
            "content": document["content"],
            "name": document["name"],
            "url": document["url"],
            "source": document["source"],
            "retrieve_doc": document["retrieve_doc"],
            "tokens": document["tokens"],
        }
        for document in documents
    }


def get_chunk_record_id(record: Any) -> str:
    chunk_id = getattr(record, "chunk_id", None)
    if chunk_id is not None:
        return str(chunk_id)

    node_id = getattr(record, "node_id", None)
    if node_id is not None:
        return str(node_id)

    raise TypeError("Unsupported chunk record type: missing chunk identifier.")


def get_chunk_record_doc_id(record: Any) -> str:
    doc_id = getattr(record, "doc_id", None)
    if doc_id is not None:
        return str(doc_id)

    metadata = get_chunk_record_metadata(record)
    metadata_doc_id = metadata.get("doc_id")
    if metadata_doc_id is not None:
        return str(metadata_doc_id)

    source_node = getattr(record, "source_node", None)
    source_node_id = getattr(source_node, "node_id", None)
    if source_node_id is not None:
        return str(source_node_id)

    raise TypeError("Unsupported chunk record type: missing document identifier.")


def get_chunk_record_text(record: Any) -> str:
    text = getattr(record, "text", None)
    if text is not None:
        return str(text)

    get_content = getattr(record, "get_content", None)
    if callable(get_content):
        return str(get_content())

    raise TypeError("Unsupported chunk record type: missing chunk text.")


def get_chunk_record_metadata(record: Any) -> dict[str, Any]:
    metadata = getattr(record, "metadata", None)
    if metadata is not None:
        return dict(metadata)

    source_node = getattr(record, "source_node", None)
    source_metadata = getattr(source_node, "metadata", None)
    if source_metadata is not None:
        return dict(source_metadata)

    raise TypeError("Unsupported chunk record type: missing metadata.")


def get_chunk_record_source(record: Any) -> str | None:
    return get_chunk_record_metadata(record).get("source")


def normalize_chunk_record(record: Any) -> ChunkRecord:
    doc_id = get_chunk_record_doc_id(record)
    metadata = get_chunk_record_metadata(record)
    metadata.setdefault("doc_id", doc_id)
    return ChunkRecord(
        chunk_id=get_chunk_record_id(record),
        doc_id=doc_id,
        text=get_chunk_record_text(record),
        metadata=metadata,
    )


def get_full_doc_content(full_doc: Any) -> str:
    if isinstance(full_doc, dict):
        return str(full_doc["content"])
    if hasattr(full_doc, "get_content"):
        return str(full_doc.get_content())
    if hasattr(full_doc, "text"):
        return str(full_doc.text)
    raise TypeError("Unsupported full document type in document dictionary.")


def _is_missing_metadata_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        normalized = value.strip()
        return not normalized or normalized.lower() == "nan"
    return False


def _string_metadata_value(*candidates: Any, default: str = "") -> str:
    for candidate in candidates:
        if _is_missing_metadata_value(candidate):
            continue
        return str(candidate).strip()
    return default


def _title_from_url(url: str) -> str:
    if not url:
        return ""
    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    if not slug:
        return ""
    return unquote(slug).replace("-", " ").replace("_", " ").strip()


def _cohere_embeddings_list(response: Any) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        raise ValueError("Cohere embed response did not contain embeddings.")

    float_vectors = getattr(embeddings, "float", None)
    if float_vectors is not None:
        return [list(vector) for vector in float_vectors]

    if isinstance(embeddings, list):
        return [list(vector) for vector in embeddings]

    raise ValueError("Unsupported Cohere embed response shape.")


def embed_texts(
    client: cohere.ClientV2,
    texts: list[str],
    input_type: str,
    model: str = DEFAULT_EMBED_MODEL,
    output_dimension: int = DEFAULT_OUTPUT_DIMENSION,
    batch_size: int = 96,
    show_progress: bool = False,
    progress_desc: str = "Embedding",
) -> list[list[float]]:
    vectors: list[list[float]] = []
    progress = None
    if show_progress and texts:
        progress = tqdm(total=len(texts), desc=progress_desc, unit="chunk")

    try:
        for batch in batched(texts, batch_size):
            response = client.embed(
                model=model,
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=output_dimension,
                texts=batch,
            )
            vectors.extend(_cohere_embeddings_list(response))
            if progress is not None:
                progress.update(len(batch))
    finally:
        if progress is not None:
            progress.close()

    return vectors


def rerank_results(
    client: cohere.ClientV2,
    query: str,
    results: list[SearchResult],
    model: str = DEFAULT_RERANK_MODEL,
    top_n: int = DEFAULT_RERANK_TOP_K,
) -> list[SearchResult]:
    if not results:
        return []

    response = client.rerank(
        model=model,
        query=query,
        documents=[result.content for result in results],
        top_n=min(top_n, len(results)),
    )

    reranked: list[SearchResult] = []
    for item in response.results:
        result = results[item.index]
        reranked.append(
            SearchResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                title=result.title,
                url=result.url,
                source=result.source,
                retrieve_doc=result.retrieve_doc,
                tokens=result.tokens,
                score=float(item.relevance_score),
                content=result.content,
                chunk_content=result.chunk_content,
            )
        )
    return reranked


def build_where_filter(allowed_sources: list[str] | None) -> dict[str, Any] | None:
    if not allowed_sources:
        return None
    if len(allowed_sources) == 1:
        return {"source": {"$eq": allowed_sources[0]}}
    return {"source": {"$in": allowed_sources}}


def _distance_to_score(distance: float | None) -> float:
    if distance is None:
        return 0.0
    if math.isnan(distance):
        return 0.0
    return max(0.0, 1.0 - distance)


def _flatten_query_results(values: list[list[Any]] | None) -> list[Any]:
    if not values:
        return []
    return values[0]


class LocalChromaRetriever:
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        document_dict_path: str,
        *,
        cohere_api_key: str,
        embed_model: str = DEFAULT_EMBED_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        dense_top_k: int = DEFAULT_DENSE_TOP_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        answer_model_name: str | None = None,
        token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    ) -> None:
        self._db_path = db_path
        self._collection_name = collection_name
        self._document_dict_path = document_dict_path
        self._dense_top_k = dense_top_k
        self._rerank_top_k = rerank_top_k
        self._token_budget = token_budget
        self._embed_model = embed_model
        self._rerank_model = rerank_model
        self._encoding = get_token_encoding(answer_model_name)

        client = chromadb.PersistentClient(path=db_path)
        self._collection = client.get_or_create_collection(name=collection_name)
        with open(document_dict_path, "rb") as handle:
            self._document_dict: dict[str, dict[str, Any]] = pickle.load(handle)

        self._cohere = cohere.ClientV2(api_key=cohere_api_key)

    def search(
        self,
        query: str,
        *,
        allowed_sources: list[str] | None = None,
    ) -> list[SearchResult]:
        query_embedding = embed_texts(
            self._cohere,
            [query],
            input_type="search_query",
            model=self._embed_model,
        )[0]

        where = build_where_filter(allowed_sources)
        raw_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=self._dense_top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunk_ids = _flatten_query_results(raw_results.get("ids"))
        documents = _flatten_query_results(raw_results.get("documents"))
        metadatas = _flatten_query_results(raw_results.get("metadatas"))
        distances = _flatten_query_results(raw_results.get("distances"))

        dense_hits: list[SearchResult] = []
        seen_doc_ids: set[str] = set()
        for chunk_id, chunk_text, metadata, distance in zip(
            chunk_ids, documents, metadatas, distances, strict=False
        ):
            if metadata is None:
                continue

            doc_id = str(metadata["doc_id"])
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            full_doc = self._document_dict.get(doc_id)
            if metadata.get("retrieve_doc") and full_doc is not None:
                content = get_full_doc_content(full_doc)
            else:
                content = chunk_text

            full_doc_name = full_doc.get("name") if isinstance(full_doc, dict) else None
            url = _string_metadata_value(
                metadata.get("url"),
                full_doc.get("url") if isinstance(full_doc, dict) else None,
            )
            title = _string_metadata_value(
                metadata.get("title"),
                full_doc_name,
                _title_from_url(url),
                doc_id,
            )
            source = _string_metadata_value(
                metadata.get("source"),
                full_doc.get("source") if isinstance(full_doc, dict) else None,
                default="unknown",
            )
            retrieve_doc = bool(
                metadata.get("retrieve_doc")
                if "retrieve_doc" in metadata
                else (
                    full_doc.get("retrieve_doc") if isinstance(full_doc, dict) else False
                )
            )
            tokens_value = metadata.get("tokens")
            if _is_missing_metadata_value(tokens_value) and isinstance(full_doc, dict):
                tokens_value = full_doc.get("tokens")
            try:
                tokens = int(tokens_value)
            except (TypeError, ValueError):
                tokens = 0

            dense_hits.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    source=source,
                    retrieve_doc=retrieve_doc,
                    tokens=tokens,
                    score=_distance_to_score(distance),
                    content=content,
                    chunk_content=str(chunk_text),
                )
            )

        reranked = rerank_results(
            self._cohere,
            query,
            dense_hits,
            model=self._rerank_model,
            top_n=self._rerank_top_k,
        )
        return self._apply_token_budget(reranked)

    def _apply_token_budget(self, results: list[SearchResult]) -> list[SearchResult]:
        filtered: list[SearchResult] = []
        total_tokens = 0
        for result in results:
            if result.score < 0.10:
                continue

            result_tokens = len(self._encoding.encode(result.content))
            if total_tokens + result_tokens > self._token_budget:
                break

            total_tokens += result_tokens
            filtered.append(result)
        return filtered


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_document_dict(document_dict: dict[str, dict[str, Any]], output_file: str) -> None:
    ensure_parent_dir(output_file)
    with open(output_file, "wb") as handle:
        pickle.dump(document_dict, handle)


def format_tool_payload(query: str, results: list[SearchResult]) -> str:
    payload = {
        "query": query,
        "matches": [asdict(result) for result in results],
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_tool_payload(content: str) -> list[SearchResult]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []

    matches = payload.get("matches", [])
    parsed: list[SearchResult] = []
    for match in matches:
        try:
            parsed.append(SearchResult(**match))
        except TypeError:
            logfire.warn("Skipping malformed retrieval payload entry")
    return parsed
