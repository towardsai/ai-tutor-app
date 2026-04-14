"""
Build local Chroma indexes from the JSONL corpus.

The script keeps the previous command surface so existing workflows can still call:

    uv run -m data.scraping_scripts.create_vector_stores all_sources
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pickle
import shutil
from typing import Any, Sequence

import chromadb
import cohere
from dotenv import load_dotenv
from tqdm.auto import tqdm

try:
    from data.scraping_scripts.add_context_to_nodes import create_docs, process
    from scripts.chroma_rag import (
        DEFAULT_EMBED_MODEL,
        build_document_dict,
        embed_texts,
        load_jsonl_documents,
        normalize_chunk_record,
        save_document_dict,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data.scraping_scripts.add_context_to_nodes import create_docs, process
    from scripts.chroma_rag import (
        DEFAULT_EMBED_MODEL,
        build_document_dict,
        embed_texts,
        load_jsonl_documents,
        normalize_chunk_record,
        save_document_dict,
    )

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SOURCE_CONFIGS = {
    "transformers": {
        "input_file": "data/transformers_data.jsonl",
        "db_name": "chroma-db-transformers",
        "document_dict_file": "document_dict_transformers.pkl",
    },
    "peft": {
        "input_file": "data/peft_data.jsonl",
        "db_name": "chroma-db-peft",
        "document_dict_file": "document_dict_peft.pkl",
    },
    "trl": {
        "input_file": "data/trl_data.jsonl",
        "db_name": "chroma-db-trl",
        "document_dict_file": "document_dict_trl.pkl",
    },
    "llama_index": {
        "input_file": "data/llama_index_data.jsonl",
        "db_name": "chroma-db-llama_index",
        "document_dict_file": "document_dict_llama_index.pkl",
    },
    "openai_cookbooks": {
        "input_file": "data/openai_cookbooks_data.jsonl",
        "db_name": "chroma-db-openai_cookbooks",
        "document_dict_file": "document_dict_openai_cookbooks.pkl",
    },
    "langchain": {
        "input_file": "data/langchain_data.jsonl",
        "db_name": "chroma-db-langchain",
        "document_dict_file": "document_dict_langchain.pkl",
    },
    "tai_blog": {
        "input_file": "data/tai_blog_data.jsonl",
        "db_name": "chroma-db-tai_blog",
        "document_dict_file": "document_dict_tai_blog.pkl",
    },
    "all_sources": {
        "input_file": "data/all_sources_data.jsonl",
        "db_name": "chroma-db-all_sources",
        "document_dict_file": "document_dict_all_sources.pkl",
    },
}


def load_or_create_chunk_records(source: str) -> list[Any]:
    config = SOURCE_CONFIGS[source]
    if source == "all_sources" and os.path.exists("data/all_sources_contextual_nodes.pkl"):
        with open("data/all_sources_contextual_nodes.pkl", "rb") as handle:
            return pickle.load(handle)

    documents = create_docs(config["input_file"])
    return asyncio.run(process(documents))


def iter_batches(items: Sequence[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def process_source(source: str) -> None:
    config = SOURCE_CONFIGS[source]
    document_rows = load_jsonl_documents(config["input_file"])
    if not document_rows:
        print(f"No documents found for {source}")
        return

    db_name = config["db_name"]
    db_path = f"data/{db_name}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    os.makedirs(db_path, exist_ok=True)

    chunk_records = [
        normalize_chunk_record(record)
        for record in load_or_create_chunk_records(source)
    ]
    if not chunk_records:
        logger.info("No chunk records found for %s", source)
        return

    chunk_ids = [record.chunk_id for record in chunk_records]
    chunk_texts = [record.text for record in chunk_records]
    chunk_metadatas = [record.metadata for record in chunk_records]

    logger.info(
        "Preparing %s chunks from %s documents for %s",
        len(chunk_records),
        len(document_rows),
        source,
    )

    cohere_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    logger.info("Generating embeddings for %s", source)
    embeddings = embed_texts(
        cohere_client,
        chunk_texts,
        input_type="search_document",
        model=DEFAULT_EMBED_MODEL,
        show_progress=True,
        progress_desc=f"Embedding {source}",
    )

    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name=db_name)
    max_batch_size = chroma_client.get_max_batch_size()
    logger.info(
        "Writing %s embeddings to Chroma for %s in batches of up to %s",
        len(chunk_ids),
        source,
        max_batch_size,
    )
    with tqdm(total=len(chunk_ids), desc=f"Upserting {source}", unit="chunk") as progress:
        for batch_ids, batch_embeddings, batch_texts, batch_metadatas in zip(
            iter_batches(chunk_ids, max_batch_size),
            iter_batches(embeddings, max_batch_size),
            iter_batches(chunk_texts, max_batch_size),
            iter_batches(chunk_metadatas, max_batch_size),
        ):
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
            )
            progress.update(len(batch_ids))

    document_dict = build_document_dict(document_rows)
    save_document_dict(
        document_dict,
        f"{db_path}/{config['document_dict_file']}",
    )

    print(
        f"Indexed {len(chunk_records)} chunks from {len(document_rows)} documents into {db_path}"
    )


def main(sources: list[str]) -> None:
    for source in sources:
        if source not in SOURCE_CONFIGS:
            print(f"Unknown source: {source}")
            continue
        process_source(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process sources and create local Chroma vector stores."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        choices=SOURCE_CONFIGS.keys(),
        help="Specify one or more sources to process",
    )
    args = parser.parse_args()
    main(args.sources)
