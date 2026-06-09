from __future__ import annotations

import json
import logging
import math
import os
import pickle
import random
import re
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import chromadb
import cohere
import tiktoken
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MAX_CHUNK_TOKENS = 1200
DEFAULT_DENSE_TOP_K = 15
DEFAULT_BM25_TOP_K = 30
DEFAULT_FUSION_TOP_K = 30
DEFAULT_RERANK_TOP_K = 5
DEFAULT_RRF_K = 60
DEFAULT_CONTEXT_TOKEN_BUDGET = 100_000
DEFAULT_EMBED_MODEL = "embed-v4.0"
DEFAULT_RERANK_MODEL = "rerank-v4.0-fast"
DEFAULT_ENCODING = "cl100k_base"
DEFAULT_OUTPUT_DIMENSION = 1024
DEFAULT_COHERE_EMBED_BATCH_SIZE = 96
DEFAULT_COHERE_EMBED_INPUTS_PER_MINUTE = 2_000
DEFAULT_COHERE_EMBED_TPM_LIMIT = 0
DEFAULT_COHERE_EMBED_RPM_LIMIT = 0
DEFAULT_COHERE_EMBED_RATE_LIMIT_MARGIN = 0.8
DEFAULT_COHERE_EMBED_WINDOW_SECONDS = 60.0
DEFAULT_COHERE_EMBED_RETRY_ATTEMPTS = 8
DEFAULT_SOURCE_VERSIONS_PATH = "data/source_versions.json"

BM25_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./:-]*|\d+(?:\.\d+)*")
CAMEL_CASE_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+")
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
MARKDOWN_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
BM25_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "use",
        "what",
        "when",
        "where",
        "which",
        "with",
        "you",
        "your",
    }
)


class SyncWindowLimiter:
    def __init__(self, units_per_window: int, window_seconds: float) -> None:
        self.units_per_window = max(1, units_per_window)
        self.window_seconds = window_seconds
        self._events: deque[tuple[float, int]] = deque()
        self._used_units = 0
        self._lock = threading.Lock()

    def acquire(self, units: int) -> None:
        units = min(max(1, units), self.units_per_window)

        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)

                if self._used_units + units <= self.units_per_window:
                    self._events.append((now, units))
                    self._used_units += units
                    return

                oldest_at, _ = self._events[0]
                delay = max(0.1, self.window_seconds - (now - oldest_at))

            time.sleep(delay + random.uniform(0.1, 0.75))

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= self.window_seconds:
            _, units = self._events.popleft()
            self._used_units -= units


_cohere_limiter_lock = threading.Lock()
_cohere_limiters: dict[tuple[str, int, float], SyncWindowLimiter] = {}


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class MarkdownUnit:
    text: str
    heading_path: tuple[str, ...]
    is_code: bool = False


@dataclass(slots=True)
class MarkdownChunk:
    text: str
    heading_path: tuple[str, ...]
    tokens: int


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
    heading_path: str = ""
    retrieval_method: str = ""


@dataclass(slots=True)
class BM25Index:
    records: list[ChunkRecord]
    postings: dict[str, list[tuple[int, int]]]
    document_frequencies: dict[str, int]
    document_lengths: list[int]
    average_document_length: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(
        cls,
        records: list[ChunkRecord],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        postings: dict[str, list[tuple[int, int]]] = {}
        document_frequencies: dict[str, int] = {}
        document_lengths: list[int] = []

        for doc_index, record in enumerate(records):
            terms = tokenize_for_bm25(
                format_chunk_for_retrieval(record.text, record.metadata)
            )
            counts = Counter(terms)
            document_lengths.append(sum(counts.values()))

            for term, term_frequency in counts.items():
                postings.setdefault(term, []).append((doc_index, term_frequency))
            for term in counts:
                document_frequencies[term] = document_frequencies.get(term, 0) + 1

        average_document_length = (
            sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
        )
        return cls(
            records=records,
            postings=postings,
            document_frequencies=document_frequencies,
            document_lengths=document_lengths,
            average_document_length=average_document_length,
            k1=k1,
            b=b,
        )

    def search(
        self,
        query: str,
        *,
        allowed_sources: list[str] | None = None,
        top_k: int = DEFAULT_BM25_TOP_K,
    ) -> list[tuple[ChunkRecord, float]]:
        query_terms = tokenize_for_bm25(query)
        if not query_terms or not self.records:
            return []

        allowed = set(allowed_sources or [])
        total_documents = len(self.records)
        scores: dict[int, float] = {}

        for term in set(query_terms):
            postings = self.postings.get(term)
            if not postings:
                continue

            document_frequency = self.document_frequencies.get(term, len(postings))
            idf = math.log(
                1.0
                + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

            for doc_index, term_frequency in postings:
                record = self.records[doc_index]
                if allowed and record.metadata.get("source") not in allowed:
                    continue

                document_length = self.document_lengths[doc_index]
                if document_length <= 0 or self.average_document_length <= 0:
                    continue

                denominator = term_frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b * document_length / self.average_document_length
                )
                scores[doc_index] = scores.get(doc_index, 0.0) + idf * (
                    term_frequency * (self.k1 + 1.0) / denominator
                )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            (self.records[doc_index], score)
            for doc_index, score in ranked[: max(0, top_k)]
            if score > 0.0
        ]


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


def load_source_versions(
    path: str = DEFAULT_SOURCE_VERSIONS_PATH,
) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(source): dict(metadata)
        for source, metadata in data.items()
        if isinstance(metadata, dict)
    }


def source_version_for(
    source: str,
    source_versions: dict[str, dict[str, Any]] | None = None,
) -> str:
    metadata = (source_versions or {}).get(source, {})
    for key in ("version", "sha", "indexedAt"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def clean_heading_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("#").strip())


def parse_markdown_units(
    text: str,
    *,
    default_heading_path: tuple[str, ...] = (),
) -> list[MarkdownUnit]:
    lines = text.splitlines()
    units: list[MarkdownUnit] = []
    heading_stack: list[str] = []
    text_buffer: list[str] = []

    def active_heading_path() -> tuple[str, ...]:
        return tuple(heading_stack) or default_heading_path

    def flush_text_buffer() -> None:
        nonlocal text_buffer
        paragraph: list[str] = []
        for buffered_line in text_buffer:
            if buffered_line.strip():
                paragraph.append(buffered_line)
                continue
            if paragraph:
                units.append(
                    MarkdownUnit(
                        text="\n".join(paragraph).strip(),
                        heading_path=active_heading_path(),
                    )
                )
                paragraph = []
        if paragraph:
            units.append(
                MarkdownUnit(
                    text="\n".join(paragraph).strip(),
                    heading_path=active_heading_path(),
                )
            )
        text_buffer = []

    index = 0
    while index < len(lines):
        line = lines[index]
        fence_match = MARKDOWN_FENCE_RE.match(line)
        if fence_match:
            flush_text_buffer()
            fence = fence_match.group(1)
            fence_char = fence[0]
            fence_len = len(fence)
            code_lines = [line]
            index += 1
            while index < len(lines):
                code_line = lines[index]
                code_lines.append(code_line)
                close_match = MARKDOWN_FENCE_RE.match(code_line)
                if (
                    close_match
                    and close_match.group(1)[0] == fence_char
                    and len(close_match.group(1)) >= fence_len
                ):
                    index += 1
                    break
                index += 1
            units.append(
                MarkdownUnit(
                    text="\n".join(code_lines).strip(),
                    heading_path=active_heading_path(),
                    is_code=True,
                )
            )
            continue

        heading_match = MARKDOWN_HEADING_RE.match(line)
        if heading_match:
            flush_text_buffer()
            level = len(heading_match.group(1))
            title = clean_heading_text(heading_match.group(2))
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            units.append(
                MarkdownUnit(
                    text=line.strip(),
                    heading_path=active_heading_path(),
                )
            )
            index += 1
            continue

        text_buffer.append(line)
        index += 1

    flush_text_buffer()
    return [unit for unit in units if unit.text.strip()]


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


def _heading_path_text(heading_path: tuple[str, ...] | list[str] | str | None) -> str:
    if not heading_path:
        return ""
    if isinstance(heading_path, str):
        return heading_path
    return " > ".join(str(part) for part in heading_path if str(part).strip())


def _split_large_unit(
    unit: MarkdownUnit,
    *,
    encoding: tiktoken.Encoding,
    chunk_size: int,
    chunk_overlap: int,
) -> list[MarkdownUnit]:
    if unit.is_code:
        return [unit]

    token_count = len(encoding.encode(unit.text, disallowed_special=()))
    if token_count <= chunk_size:
        return [unit]
    return [
        MarkdownUnit(text=chunk, heading_path=unit.heading_path)
        for chunk in token_window_chunks(
            unit.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    ]


def heading_aware_markdown_chunks(
    text: str,
    *,
    title: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[MarkdownChunk]:
    encoding = tiktoken.get_encoding(encoding_name)
    default_heading_path = (title,) if title else ()
    units = parse_markdown_units(text, default_heading_path=default_heading_path)
    if not units:
        return []

    budget_size = min(max(chunk_size, 1), DEFAULT_MAX_CHUNK_TOKENS)
    overlap_size = chunk_overlap

    def unit_size(value: str) -> int:
        return len(encoding.encode(value, disallowed_special=()))

    chunks: list[MarkdownChunk] = []
    current_parts: list[str] = []
    current_heading_path: tuple[str, ...] = ()
    current_size = 0

    def flush_current() -> None:
        nonlocal current_parts, current_heading_path, current_size
        text_value = "\n\n".join(part for part in current_parts if part.strip()).strip()
        if text_value:
            chunks.append(
                MarkdownChunk(
                    text=text_value,
                    heading_path=current_heading_path,
                    tokens=len(encoding.encode(text_value, disallowed_special=())),
                )
            )
        current_parts = []
        current_heading_path = ()
        current_size = 0

    for unit in units:
        split_units = _split_large_unit(
            unit,
            encoding=encoding,
            chunk_size=budget_size,
            chunk_overlap=overlap_size,
        )
        for split_unit in split_units:
            size = unit_size(split_unit.text)
            heading_changed = (
                current_heading_path and split_unit.heading_path != current_heading_path
            )
            would_exceed = current_parts and current_size + size > budget_size
            if heading_changed or would_exceed:
                flush_current()

            if not current_parts:
                current_heading_path = split_unit.heading_path
            current_parts.append(split_unit.text)
            current_size += size

    flush_current()
    return chunks


def build_chunk_retrieval_header(metadata: dict[str, Any]) -> str:
    lines: list[str] = []
    title = _string_metadata_value(metadata.get("title"), metadata.get("name"))
    source = _string_metadata_value(metadata.get("source"))
    version = _string_metadata_value(metadata.get("source_version"))
    heading_path = _string_metadata_value(metadata.get("heading_path"))

    if title:
        lines.append(f"Title: {title}")
    if source:
        lines.append(f"Source: {source}")
    if version:
        lines.append(f"Version: {version}")
    if heading_path:
        lines.append(f"Heading path: {heading_path}")
    return "\n".join(lines)


def format_chunk_for_retrieval(text: str, metadata: dict[str, Any]) -> str:
    text = text.strip()
    header = build_chunk_retrieval_header(metadata)
    if not header:
        return text
    if not text:
        return header
    return f"{header}\n\n{text}"


def build_chunk_records(
    documents: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    chunk_records: list[ChunkRecord] = []
    source_versions = load_source_versions()
    for document in documents:
        chunks = heading_aware_markdown_chunks(
            document["content"],
            title=str(document.get("name") or ""),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        source = str(document["source"])
        source_version = source_version_for(source, source_versions)
        for index, chunk in enumerate(chunks):
            heading_path = _heading_path_text(chunk.heading_path)
            metadata = {
                "doc_id": document["doc_id"],
                "title": document["name"],
                "url": document["url"],
                "source": source,
                "source_version": source_version,
                "retrieve_doc": document["retrieve_doc"],
                "tokens": document["tokens"],
                "chunk_tokens": chunk.tokens,
                "chunk_index": index,
                "heading_path": heading_path,
            }
            chunk_records.append(
                ChunkRecord(
                    chunk_id=f"{document['doc_id']}:{index}",
                    doc_id=document["doc_id"],
                    text=chunk.text,
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


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _cohere_rate_limited_units(
    limit: int | None,
    env_name: str,
    default: int,
    margin: float,
) -> int | None:
    configured_limit = _env_int(env_name, default) if limit is None else limit
    if configured_limit <= 0:
        return None
    return max(1, int(configured_limit * margin))


def _get_cohere_limiter(
    name: str,
    units_per_window: int,
    window_seconds: float,
) -> SyncWindowLimiter:
    key = (name, units_per_window, window_seconds)
    with _cohere_limiter_lock:
        limiter = _cohere_limiters.get(key)
        if limiter is None:
            limiter = SyncWindowLimiter(units_per_window, window_seconds)
            _cohere_limiters[key] = limiter
        return limiter


def _count_embed_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text, disallowed_special=()))


def _iter_cohere_embed_batches(
    texts: list[str],
    token_counts: list[int],
    *,
    batch_size: int,
    max_batch_tokens: int | None,
) -> Iterable[tuple[list[str], int]]:
    batch: list[str] = []
    batch_tokens = 0

    for text, token_count in zip(texts, token_counts, strict=True):
        should_flush = len(batch) >= batch_size
        if max_batch_tokens is not None and batch:
            should_flush = should_flush or batch_tokens + token_count > max_batch_tokens

        if should_flush:
            yield batch, batch_tokens
            batch = []
            batch_tokens = 0

        batch.append(text)
        batch_tokens += max(1, token_count)

    if batch:
        yield batch, batch_tokens


def _is_cohere_rate_limit_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    if exc.__class__.__name__ == "TooManyRequestsError":
        return True
    return "rate limit" in str(exc).lower() and "429" in str(exc)


def _cohere_retry_after_seconds(exc: BaseException) -> float | None:
    headers = getattr(exc, "headers", None)
    if not isinstance(headers, dict):
        return None

    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None

    try:
        return float(retry_after)
    except ValueError:
        return None


def _wait_for_cohere_retry(
    exc: BaseException,
    attempt: int,
    window_seconds: float,
) -> None:
    retry_after = _cohere_retry_after_seconds(exc)
    if retry_after is not None:
        delay = retry_after
    else:
        delay = min(window_seconds, max(15.0, 2.0**attempt))

    time.sleep(delay + random.uniform(0.5, 2.0))


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
    batch_size: int = DEFAULT_COHERE_EMBED_BATCH_SIZE,
    max_inputs_per_minute: int | None = None,
    max_tokens_per_minute: int | None = None,
    max_requests_per_minute: int | None = None,
    show_progress: bool = False,
    progress_desc: str = "Embedding",
) -> list[list[float]]:
    batch_size = max(1, batch_size)
    rate_limit_margin = _env_float(
        "COHERE_EMBED_RATE_LIMIT_MARGIN",
        DEFAULT_COHERE_EMBED_RATE_LIMIT_MARGIN,
    )
    window_seconds = _env_float(
        "COHERE_EMBED_WINDOW_SECONDS",
        DEFAULT_COHERE_EMBED_WINDOW_SECONDS,
    )
    token_window = _cohere_rate_limited_units(
        max_tokens_per_minute,
        "COHERE_EMBED_TPM_LIMIT",
        DEFAULT_COHERE_EMBED_TPM_LIMIT,
        rate_limit_margin,
    )
    input_window = _cohere_rate_limited_units(
        max_inputs_per_minute,
        "COHERE_EMBED_INPUTS_PER_MINUTE",
        DEFAULT_COHERE_EMBED_INPUTS_PER_MINUTE,
        rate_limit_margin,
    )
    request_window = _cohere_rate_limited_units(
        max_requests_per_minute,
        "COHERE_EMBED_RPM_LIMIT",
        DEFAULT_COHERE_EMBED_RPM_LIMIT,
        rate_limit_margin,
    )
    retry_attempts = max(
        1,
        _env_int("COHERE_EMBED_RETRY_ATTEMPTS", DEFAULT_COHERE_EMBED_RETRY_ATTEMPTS),
    )
    encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
    token_counts = [_count_embed_tokens(text, encoding) for text in texts]
    token_limiter = (
        _get_cohere_limiter("tokens", token_window, window_seconds)
        if token_window is not None
        else None
    )
    input_limiter = (
        _get_cohere_limiter("inputs", input_window, window_seconds)
        if input_window is not None
        else None
    )
    request_limiter = (
        _get_cohere_limiter("requests", request_window, window_seconds)
        if request_window is not None
        else None
    )

    vectors: list[list[float]] = []
    progress = None
    if show_progress and texts:
        progress = tqdm(total=len(texts), desc=progress_desc, unit="chunk")

    try:
        for batch, batch_tokens in _iter_cohere_embed_batches(
            texts,
            token_counts,
            batch_size=batch_size,
            max_batch_tokens=token_window,
        ):
            for attempt in range(1, retry_attempts + 1):
                if token_limiter is not None:
                    token_limiter.acquire(batch_tokens)
                if input_limiter is not None:
                    input_limiter.acquire(len(batch))
                if request_limiter is not None:
                    request_limiter.acquire(1)

                try:
                    response = client.embed(
                        model=model,
                        input_type=input_type,
                        embedding_types=["float"],
                        output_dimension=output_dimension,
                        texts=batch,
                    )
                    break
                except Exception as exc:
                    if (
                        not _is_cohere_rate_limit_error(exc)
                        or attempt == retry_attempts
                    ):
                        raise
                    _wait_for_cohere_retry(exc, attempt, window_seconds)

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
                heading_path=result.heading_path,
                retrieval_method=result.retrieval_method,
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


def _add_bm25_token(tokens: list[str], token: str) -> None:
    token = token.lower().strip("._-/:-")
    if not token:
        return
    if token in BM25_STOP_WORDS:
        return
    if len(token) == 1 and token not in {"c", "r"}:
        return
    tokens.append(token)


def tokenize_for_bm25(text: str) -> list[str]:
    tokens: list[str] = []
    for match in BM25_TOKEN_RE.finditer(text):
        raw_token = match.group(0)
        _add_bm25_token(tokens, raw_token)

        for part in re.split(r"[._/\-:]+", raw_token):
            _add_bm25_token(tokens, part)
            for camel_part in CAMEL_CASE_RE.findall(part):
                _add_bm25_token(tokens, camel_part)

    return tokens


def save_bm25_index(index: BM25Index, output_file: str) -> None:
    ensure_parent_dir(output_file)
    with open(output_file, "wb") as handle:
        pickle.dump(index, handle)


def load_bm25_index(path: str) -> BM25Index | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        index = pickle.load(handle)
    if isinstance(index, BM25Index):
        return index
    return None


def default_bm25_index_path(document_dict_path: str) -> str:
    path = Path(document_dict_path)
    name = path.name
    if name.startswith("document_dict_") and name.endswith(".pkl"):
        source = name.removeprefix("document_dict_").removesuffix(".pkl")
        return str(path.with_name(f"bm25_index_{source}.pkl"))
    return str(path.with_name("bm25_index.pkl"))


def result_dedupe_key(result: SearchResult) -> str:
    if result.heading_path:
        return f"{result.doc_id}::{result.heading_path}"
    return result.doc_id or result.chunk_id


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = DEFAULT_FUSION_TOP_K,
) -> list[SearchResult]:
    fused_scores: dict[str, float] = {}
    representatives: dict[str, SearchResult] = {}

    for ranked_results in ranked_lists:
        for rank, result in enumerate(ranked_results, start=1):
            key = result_dedupe_key(result)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

            current = representatives.get(key)
            if current is None:
                representatives[key] = result
            elif (
                result.retrieval_method == "bm25" and current.retrieval_method != "bm25"
            ):
                representatives[key] = result
            elif (
                result.retrieval_method == current.retrieval_method
                and result.score > current.score
            ):
                representatives[key] = result

    fused_results: list[SearchResult] = []
    for key, score in fused_scores.items():
        result = representatives[key]
        fused_results.append(
            SearchResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                title=result.title,
                url=result.url,
                source=result.source,
                retrieve_doc=result.retrieve_doc,
                tokens=result.tokens,
                score=score,
                content=result.content,
                chunk_content=result.chunk_content,
                heading_path=result.heading_path,
                retrieval_method="hybrid"
                if len(ranked_lists) > 1
                else result.retrieval_method,
            )
        )

    fused_results.sort(key=lambda result: result.score, reverse=True)
    return fused_results[: max(0, top_k)]


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
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        fusion_top_k: int = DEFAULT_FUSION_TOP_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        rrf_k: int = DEFAULT_RRF_K,
        bm25_index_path: str | None = None,
        answer_model_name: str | None = None,
        token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    ) -> None:
        self._db_path = db_path
        self._collection_name = collection_name
        self._document_dict_path = document_dict_path
        self._dense_top_k = dense_top_k
        self._bm25_top_k = bm25_top_k
        self._fusion_top_k = fusion_top_k
        self._rerank_top_k = rerank_top_k
        self._rrf_k = rrf_k
        self._token_budget = token_budget
        self._embed_model = embed_model
        self._rerank_model = rerank_model
        self._encoding = get_token_encoding(answer_model_name)
        self._bm25_index_path = bm25_index_path or default_bm25_index_path(
            document_dict_path
        )

        client = chromadb.PersistentClient(path=db_path)
        self._collection = client.get_or_create_collection(name=collection_name)
        with open(document_dict_path, "rb") as handle:
            self._document_dict: dict[str, dict[str, Any]] = pickle.load(handle)

        self._bm25_index = load_bm25_index(self._bm25_index_path)
        self._cohere = cohere.ClientV2(api_key=cohere_api_key)

    def search(
        self,
        query: str,
        *,
        allowed_sources: list[str] | None = None,
    ) -> list[SearchResult]:
        dense_hits = self._dense_search(query, allowed_sources=allowed_sources)
        bm25_hits = self._bm25_search(query, allowed_sources=allowed_sources)
        fused_hits = reciprocal_rank_fusion(
            [hits for hits in (dense_hits, bm25_hits) if hits],
            rrf_k=self._rrf_k,
            top_k=self._fusion_top_k,
        )
        if not fused_hits:
            return []

        reranked = rerank_results(
            self._cohere,
            query,
            fused_hits,
            model=self._rerank_model,
            top_n=self._rerank_top_k,
        )
        return self._apply_token_budget(reranked)

    def _dense_search(
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
        for chunk_id, chunk_text, metadata, distance in zip(
            chunk_ids, documents, metadatas, distances, strict=False
        ):
            if metadata is None:
                continue

            dense_hits.append(
                self._search_result_from_metadata(
                    chunk_id=str(chunk_id),
                    score=_distance_to_score(distance),
                    chunk_text=str(chunk_text),
                    metadata=dict(metadata),
                    retrieval_method="dense",
                )
            )
        return dense_hits

    def _bm25_search(
        self,
        query: str,
        *,
        allowed_sources: list[str] | None = None,
    ) -> list[SearchResult]:
        if self._bm25_index is None:
            return []

        return [
            self._search_result_from_metadata(
                chunk_id=record.chunk_id,
                score=score,
                chunk_text=format_chunk_for_retrieval(record.text, record.metadata),
                metadata=record.metadata,
                raw_chunk_text=record.text,
                retrieval_method="bm25",
            )
            for record, score in self._bm25_index.search(
                query,
                allowed_sources=allowed_sources,
                top_k=self._bm25_top_k,
            )
        ]

    def _search_result_from_metadata(
        self,
        *,
        chunk_id: str,
        score: float,
        chunk_text: str,
        metadata: dict[str, Any],
        retrieval_method: str,
        raw_chunk_text: str | None = None,
    ) -> SearchResult:
        doc_id = str(metadata["doc_id"])
        full_doc = self._document_dict.get(doc_id)
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
            else (full_doc.get("retrieve_doc") if isinstance(full_doc, dict) else False)
        )
        tokens_value = metadata.get("tokens")
        if _is_missing_metadata_value(tokens_value) and isinstance(full_doc, dict):
            tokens_value = full_doc.get("tokens")
        try:
            tokens = int(tokens_value)
        except (TypeError, ValueError):
            tokens = 0

        exact_chunk = _string_metadata_value(
            raw_chunk_text,
            metadata.get("raw_text"),
            default=str(chunk_text),
        )
        chunk_for_context = format_chunk_for_retrieval(exact_chunk, metadata)
        if retrieve_doc and full_doc is not None:
            content = get_full_doc_content(full_doc)
        else:
            content = chunk_for_context

        return SearchResult(
            chunk_id=chunk_id,
            doc_id=doc_id,
            title=title,
            url=url,
            source=source,
            retrieve_doc=retrieve_doc,
            tokens=tokens,
            score=score,
            content=content,
            chunk_content=exact_chunk,
            heading_path=_string_metadata_value(metadata.get("heading_path")),
            retrieval_method=retrieval_method,
        )

    def _apply_token_budget(self, results: list[SearchResult]) -> list[SearchResult]:
        filtered: list[SearchResult] = []
        total_tokens = 0
        for result in results:
            if result.score < 0.10:
                continue

            # disallowed_special=() so literal "<|endoftext|>" in chunks doesn't crash.
            result_tokens = len(
                self._encoding.encode(result.content, disallowed_special=())
            )
            if total_tokens + result_tokens > self._token_budget:
                break

            total_tokens += result_tokens
            filtered.append(result)
        return filtered


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_document_dict(
    document_dict: dict[str, dict[str, Any]], output_file: str
) -> None:
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
            logger.warning("Skipping malformed retrieval payload entry")
    return parsed
