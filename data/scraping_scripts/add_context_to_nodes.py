import asyncio
import json
import os
import pickle
import random
import re
import time
from collections import deque
from typing import List

import tiktoken
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from jinja2 import Template
from llama_index.core import Document
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt
from tqdm.asyncio import tqdm

from scripts.chroma_rag import (
    ChunkRecord,
    build_chunk_records,
    format_chunk_for_retrieval,
)

load_dotenv(".env")

CONTEXT_MODEL = os.getenv("GEMINI_CONTEXT_MODEL", "gemini-3.1-flash-lite")
CONTEXT_MAX_OUTPUT_TOKENS = 1000
CONTEXT_TPM_LIMIT = int(os.getenv("GEMINI_CONTEXT_TPM_LIMIT", "30000000"))
CONTEXT_TPM_SAFETY_MARGIN = float(os.getenv("GEMINI_CONTEXT_TPM_SAFETY_MARGIN", "0.8"))
CONTEXT_TPM_WINDOW_SECONDS = float(os.getenv("GEMINI_CONTEXT_TPM_WINDOW_SECONDS", "60"))
CONTEXT_RETRY_ATTEMPTS = int(os.getenv("GEMINI_CONTEXT_RETRY_ATTEMPTS", "8"))
DEFAULT_SEMAPHORE_LIMIT = int(os.getenv("GEMINI_CONTEXT_CONCURRENCY", "50"))
MAX_DOCUMENT_TOKENS = 120_000
RETRYABLE_GENAI_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_genai_client: genai.Client | None = None
_token_encoding = tiktoken.get_encoding("cl100k_base")


class AsyncTokenWindowLimiter:
    def __init__(self, tokens_per_window: int, window_seconds: float) -> None:
        self.tokens_per_window = max(1, tokens_per_window)
        self.window_seconds = window_seconds
        self._events: deque[tuple[float, int]] = deque()
        self._used_tokens = 0
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> None:
        tokens = min(max(1, tokens), self.tokens_per_window)

        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)

                if self._used_tokens + tokens <= self.tokens_per_window:
                    self._events.append((now, tokens))
                    self._used_tokens += tokens
                    return

                oldest_at, _ = self._events[0]
                delay = max(0.1, self.window_seconds - (now - oldest_at))

            await asyncio.sleep(delay + random.uniform(0.1, 0.75))

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= self.window_seconds:
            _, tokens = self._events.popleft()
            self._used_tokens -= tokens


_input_token_limiter = AsyncTokenWindowLimiter(
    tokens_per_window=int(CONTEXT_TPM_LIMIT * CONTEXT_TPM_SAFETY_MARGIN),
    window_seconds=CONTEXT_TPM_WINDOW_SECONDS,
)


def create_docs(input_file: str) -> List[Document]:
    with open(input_file, "r") as f:
        documents: list[Document] = []
        for line in f:
            data = json.loads(line)
            documents.append(
                Document(
                    doc_id=data["doc_id"],
                    text=data["content"],
                    metadata={  # type: ignore
                        "url": data["url"],
                        "title": data["name"],
                        "tokens": data["tokens"],
                        "retrieve_doc": data["retrieve_doc"],
                        "source": data["source"],
                    },
                    excluded_llm_metadata_keys=[
                        "title",
                        "tokens",
                        "retrieve_doc",
                        "source",
                    ],
                    excluded_embed_metadata_keys=[
                        "url",
                        "tokens",
                        "retrieve_doc",
                        "source",
                    ],
                )
            )
    return documents


class SituatedContext(BaseModel):
    title: str = Field(..., description="The title of the document.")
    context: str = Field(
        ..., description="The context to situate the chunk within the document."
    )


def get_genai_client() -> genai.Client:
    global _genai_client

    if _genai_client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY must be set to add chunk context."
            )
        _genai_client = genai.Client(api_key=api_key)

    return _genai_client


def is_retryable_genai_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, APIError)
        and getattr(exc, "code", None) in RETRYABLE_GENAI_STATUS_CODES
    )


def count_input_tokens(content: str) -> int:
    return len(_token_encoding.encode(content, disallowed_special=()))


def extract_retry_delay_seconds(exc: BaseException | None) -> float | None:
    if not isinstance(exc, APIError):
        return None

    retry_delay = _find_retry_delay(getattr(exc, "details", None))
    if retry_delay is not None:
        return retry_delay

    message = getattr(exc, "message", None) or str(exc)
    match = re.search(r"retry in ([0-9.]+)s", message, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None


def _find_retry_delay(value) -> float | None:
    if isinstance(value, dict):
        if value.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
            return _parse_duration_seconds(value.get("retryDelay"))
        for item in value.values():
            retry_delay = _find_retry_delay(item)
            if retry_delay is not None:
                return retry_delay
    elif isinstance(value, list):
        for item in value:
            retry_delay = _find_retry_delay(item)
            if retry_delay is not None:
                return retry_delay
    return None


def _parse_duration_seconds(value) -> float | None:
    if not isinstance(value, str):
        return None

    match = re.fullmatch(r"([0-9.]+)s", value)
    if match:
        return float(match.group(1))

    return None


def wait_for_genai_retry(retry_state) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    retry_delay = extract_retry_delay_seconds(exc)
    if retry_delay is not None:
        return retry_delay + random.uniform(1.0, 4.0)

    exponential_delay = min(60, max(4, 2**retry_state.attempt_number))
    return exponential_delay + random.uniform(0.5, 2.0)


@retry(
    retry=retry_if_exception(is_retryable_genai_error),
    stop=stop_after_attempt(CONTEXT_RETRY_ATTEMPTS),
    wait=wait_for_genai_retry,
    reraise=True,
)
async def situate_context(doc: str, chunk: str) -> str:
    template = Template(
        """
<document>
{{ doc }}
</document>

Here is the chunk we want to situate within the whole document above:

<chunk>
{{ chunk }}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Return a title for the document and the succinct context.
"""
    )

    content = template.render(doc=doc, chunk=chunk)
    await _input_token_limiter.acquire(count_input_tokens(content))

    response = await get_genai_client().aio.models.generate_content(
        model=CONTEXT_MODEL,
        contents=content,
        config=types.GenerateContentConfig(
            max_output_tokens=CONTEXT_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=SituatedContext,
        ),
    )
    if isinstance(response.parsed, SituatedContext):
        return response.parsed.context
    if isinstance(response.parsed, dict):
        return SituatedContext.model_validate(response.parsed).context
    if response.text:
        return SituatedContext.model_validate_json(response.text).context
    raise ValueError("Gemini returned no context response text.")


def document_to_row(document: Document) -> dict:
    return {
        "doc_id": document.doc_id,
        "content": document.get_content(),
        "name": document.metadata["title"],
        "url": document.metadata["url"],
        "source": document.metadata["source"],
        "retrieve_doc": document.metadata["retrieve_doc"],
        "tokens": document.metadata["tokens"],
    }


async def process_chunk(
    chunk_record: ChunkRecord,
    document_dict: dict[str, Document],
) -> ChunkRecord:
    doc_id = chunk_record.doc_id
    doc: Document = document_dict[doc_id]

    if doc.metadata["tokens"] > MAX_DOCUMENT_TOKENS:
        # Tokenize the document text
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(doc.get_content())

        # Trim to 120,000 tokens
        trimmed_tokens = tokens[:MAX_DOCUMENT_TOKENS]

        # Decode back to text
        trimmed_text = encoding.decode(trimmed_tokens)

        # Update the document with trimmed text
        doc = Document(text=trimmed_text, metadata=doc.metadata)
        doc.metadata["tokens"] = MAX_DOCUMENT_TOKENS

    context = await situate_context(doc.get_content(), chunk_record.text)
    metadata = dict(chunk_record.metadata)
    metadata["raw_text"] = chunk_record.text
    contextual_text = (
        f"{format_chunk_for_retrieval(chunk_record.text, metadata)}"
        f"\n\nContext: {context}"
    )
    return ChunkRecord(
        chunk_id=chunk_record.chunk_id,
        doc_id=chunk_record.doc_id,
        text=contextual_text,
        metadata=metadata,
    )


async def process(
    documents: List[Document], semaphore_limit: int = DEFAULT_SEMAPHORE_LIMIT
) -> List[ChunkRecord]:

    chunk_records = build_chunk_records([document_to_row(doc) for doc in documents])
    print(f"Number of chunks: {len(chunk_records)}")

    document_dict: dict[str, Document] = {doc.doc_id: doc for doc in documents}

    print(
        "Gemini context rate limits: "
        f"{_input_token_limiter.tokens_per_window:,} input tokens/"
        f"{CONTEXT_TPM_WINDOW_SECONDS:g}s, concurrency={semaphore_limit}"
    )

    semaphore = asyncio.Semaphore(semaphore_limit)

    async def process_with_semaphore(chunk_record: ChunkRecord):
        async with semaphore:
            result = await process_chunk(chunk_record, document_dict)
            await asyncio.sleep(0.1)
            return result

    tasks = [process_with_semaphore(chunk_record) for chunk_record in chunk_records]

    results: List[ChunkRecord] = await tqdm.gather(*tasks, desc="Processing chunks")

    return results


async def main():
    documents: List[Document] = create_docs("data/all_sources_data.jsonl")
    enhanced_nodes: List[ChunkRecord] = await process(documents)

    with open("data/all_sources_contextual_nodes.pkl", "wb") as f:
        pickle.dump(enhanced_nodes, f)

    with open("data/all_sources_contextual_nodes.pkl", "rb") as f:
        enhanced_nodes: list[ChunkRecord] = pickle.load(f)

    for i, node in enumerate(enhanced_nodes):
        print(f"Chunk {i + 1}:")
        print(f"Chunk ID: {node.chunk_id}")
        print(f"Document ID: {node.doc_id}")
        print(f"Text: {node.text}")
        print(f"Metadata: {node.metadata}")
        break


if __name__ == "__main__":
    asyncio.run(main())
