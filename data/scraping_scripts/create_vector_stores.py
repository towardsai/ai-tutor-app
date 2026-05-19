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
    from data.scraping_scripts.source_registry import (
        ACTIVE_SOURCE_KEYS,
        vector_store_source_configs,
    )
    from data.scraping_scripts.add_context_to_nodes import create_docs, process
    from scripts.chroma_rag import (
        DEFAULT_COHERE_EMBED_BATCH_SIZE,
        DEFAULT_EMBED_MODEL,
        BM25Index,
        build_chunk_records,
        build_document_dict,
        embed_texts,
        get_chunk_record_source,
        load_jsonl_documents,
        normalize_chunk_record,
        save_bm25_index,
        save_document_dict,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data.scraping_scripts.source_registry import (
        ACTIVE_SOURCE_KEYS,
        vector_store_source_configs,
    )
    from data.scraping_scripts.add_context_to_nodes import create_docs, process
    from scripts.chroma_rag import (
        DEFAULT_COHERE_EMBED_BATCH_SIZE,
        DEFAULT_EMBED_MODEL,
        BM25Index,
        build_chunk_records,
        build_document_dict,
        embed_texts,
        get_chunk_record_source,
        load_jsonl_documents,
        normalize_chunk_record,
        save_bm25_index,
        save_document_dict,
    )

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SOURCE_CONFIGS = vector_store_source_configs()


def load_or_create_chunk_records(source: str) -> list[Any]:
    config = SOURCE_CONFIGS[source]
    if source == "all_sources" and os.path.exists("data/all_sources_contextual_nodes.pkl"):
        with open("data/all_sources_contextual_nodes.pkl", "rb") as handle:
            records = pickle.load(handle)
        active_records = []
        skipped_count = 0
        for record in records:
            try:
                record_source = get_chunk_record_source(record)
            except Exception:
                active_records.append(record)
                continue
            if record_source in ACTIVE_SOURCE_KEYS:
                active_records.append(record)
            else:
                skipped_count += 1
        if skipped_count:
            logger.info(
                "Skipped %s inactive contextual chunks while building all_sources",
                skipped_count,
            )
        return active_records

    documents = create_docs(config["input_file"])
    return asyncio.run(process(documents))


def iter_batches(items: Sequence[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def get_collection_ids(collection, batch_size: int = 5000) -> set[str]:
    ids: set[str] = set()
    offset = 0

    while True:
        result = collection.get(limit=batch_size, offset=offset, include=[])
        batch_ids = result.get("ids", [])
        if not batch_ids:
            break

        ids.update(str(chunk_id) for chunk_id in batch_ids)
        offset += len(batch_ids)

    return ids


def write_retrieval_artifacts(
    *,
    config: dict[str, str],
    document_rows: list[dict[str, Any]],
    db_path: str,
) -> int:
    document_dict = build_document_dict(document_rows)
    save_document_dict(
        document_dict,
        f"{db_path}/{config['document_dict_file']}",
    )
    bm25_records = build_chunk_records(document_rows)
    save_bm25_index(
        BM25Index.build(bm25_records),
        f"{db_path}/{config['bm25_index_file']}",
    )
    return len(bm25_records)


def process_source(
    source: str,
    *,
    force_rebuild: bool = False,
    skip_dense_embeddings: bool = False,
    embed_batch_size: int = DEFAULT_COHERE_EMBED_BATCH_SIZE,
    cohere_embed_inputs_per_minute: int | None = None,
    cohere_embed_tpm_limit: int | None = None,
    cohere_embed_rpm_limit: int | None = None,
) -> None:
    config = SOURCE_CONFIGS[source]
    document_rows = load_jsonl_documents(config["input_file"])
    if not document_rows:
        print(f"No documents found for {source}")
        return

    db_name = config["db_name"]
    db_path = f"data/{db_name}"
    if force_rebuild and not skip_dense_embeddings and os.path.exists(db_path):
        shutil.rmtree(db_path)

    os.makedirs(db_path, exist_ok=True)

    if skip_dense_embeddings:
        bm25_count = write_retrieval_artifacts(
            config=config,
            document_rows=document_rows,
            db_path=db_path,
        )
        print(
            f"Indexed {bm25_count} BM25 chunks from {len(document_rows)} documents "
            f"into {db_path}; skipped dense embedding updates"
        )
        return

    chunk_records = [
        normalize_chunk_record(record)
        for record in load_or_create_chunk_records(source)
    ]
    if not chunk_records:
        logger.info("No chunk records found for %s", source)
        return

    chunk_ids = [record.chunk_id for record in chunk_records]
    logger.info(
        "Preparing %s chunks from %s documents for %s",
        len(chunk_records),
        len(document_rows),
        source,
    )

    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name=db_name)
    max_batch_size = chroma_client.get_max_batch_size()
    existing_ids = set() if force_rebuild else get_collection_ids(collection)
    desired_ids = set(chunk_ids)

    stale_ids = sorted(existing_ids - desired_ids)
    if stale_ids:
        logger.info("Deleting %s stale embeddings for %s", len(stale_ids), source)
        for batch_ids in iter_batches(stale_ids, max_batch_size):
            collection.delete(ids=batch_ids)

    reusable_ids = existing_ids & desired_ids
    records_to_embed = [
        record for record in chunk_records if record.chunk_id not in reusable_ids
    ]

    logger.info(
        "Reusing %s existing embeddings and generating %s embeddings for %s",
        len(reusable_ids),
        len(records_to_embed),
        source,
    )

    if records_to_embed:
        cohere_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        texts_to_embed = [record.text for record in records_to_embed]
        ids_to_embed = [record.chunk_id for record in records_to_embed]
        metadatas_to_embed = [record.metadata for record in records_to_embed]

        logger.info("Generating embeddings for %s", source)
        embeddings = embed_texts(
            cohere_client,
            texts_to_embed,
            input_type="search_document",
            model=DEFAULT_EMBED_MODEL,
            batch_size=embed_batch_size,
            max_inputs_per_minute=cohere_embed_inputs_per_minute,
            max_tokens_per_minute=cohere_embed_tpm_limit,
            max_requests_per_minute=cohere_embed_rpm_limit,
            show_progress=True,
            progress_desc=f"Embedding {source}",
        )

        logger.info(
            "Writing %s new embeddings to Chroma for %s in batches of up to %s",
            len(ids_to_embed),
            source,
            max_batch_size,
        )
        with tqdm(
            total=len(ids_to_embed), desc=f"Upserting {source}", unit="chunk"
        ) as progress:
            for batch_ids, batch_embeddings, batch_texts, batch_metadatas in zip(
                iter_batches(ids_to_embed, max_batch_size),
                iter_batches(embeddings, max_batch_size),
                iter_batches(texts_to_embed, max_batch_size),
                iter_batches(metadatas_to_embed, max_batch_size),
            ):
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                )
                progress.update(len(batch_ids))

    bm25_count = write_retrieval_artifacts(
        config=config,
        document_rows=document_rows,
        db_path=db_path,
    )

    print(
        f"Indexed {len(chunk_records)} dense chunks and {bm25_count} BM25 chunks "
        f"from {len(document_rows)} documents into {db_path}"
    )


def main(
    sources: list[str],
    *,
    force_rebuild: bool = False,
    skip_dense_embeddings: bool = False,
    embed_batch_size: int = DEFAULT_COHERE_EMBED_BATCH_SIZE,
    cohere_embed_inputs_per_minute: int | None = None,
    cohere_embed_tpm_limit: int | None = None,
    cohere_embed_rpm_limit: int | None = None,
) -> None:
    for source in sources:
        if source not in SOURCE_CONFIGS:
            print(f"Unknown source: {source}")
            continue
        process_source(
            source,
            force_rebuild=force_rebuild,
            skip_dense_embeddings=skip_dense_embeddings,
            embed_batch_size=embed_batch_size,
            cohere_embed_inputs_per_minute=cohere_embed_inputs_per_minute,
            cohere_embed_tpm_limit=cohere_embed_tpm_limit,
            cohere_embed_rpm_limit=cohere_embed_rpm_limit,
        )


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
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete the existing Chroma directory and regenerate all embeddings.",
    )
    parser.add_argument(
        "--skip-dense-embeddings",
        action="store_true",
        help=(
            "Only write retrieval artifacts that do not require Cohere embeddings "
            "(document dictionary and BM25 index). Existing Chroma data is left untouched."
        ),
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=DEFAULT_COHERE_EMBED_BATCH_SIZE,
        help="Maximum number of chunks to include in one Cohere embed request.",
    )
    parser.add_argument(
        "--cohere-embed-inputs-per-minute",
        type=int,
        default=None,
        help=(
            "Cohere embed input-per-minute limit before the safety margin. "
            "Defaults to COHERE_EMBED_INPUTS_PER_MINUTE or 2000; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--cohere-embed-tpm-limit",
        type=int,
        default=None,
        help=(
            "Cohere embed token-per-minute limit before the safety margin. "
            "Defaults to COHERE_EMBED_TPM_LIMIT or 0; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--cohere-embed-rpm-limit",
        type=int,
        default=None,
        help=(
            "Cohere embed request-per-minute limit before the safety margin. "
            "Defaults to COHERE_EMBED_RPM_LIMIT or 0; use 0 to disable."
        ),
    )
    args = parser.parse_args()
    main(
        args.sources,
        force_rebuild=args.force_rebuild,
        skip_dense_embeddings=args.skip_dense_embeddings,
        embed_batch_size=args.embed_batch_size,
        cohere_embed_inputs_per_minute=args.cohere_embed_inputs_per_minute,
        cohere_embed_tpm_limit=args.cohere_embed_tpm_limit,
        cohere_embed_rpm_limit=args.cohere_embed_rpm_limit,
    )
