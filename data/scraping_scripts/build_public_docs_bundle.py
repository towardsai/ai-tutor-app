"""Build (and optionally upload) the PUBLIC, docs-only vector-db/KB bundle.

The production bundle lives in the **private** dataset
``towardsai-tutors/ai-tutor-vector-db`` and contains every source, including
the course content (real student-facing material we keep gated). Anyone
without an ``HF_TOKEN`` that can read that repo cannot cold-start the app.

This script derives a **public** docs-only bundle from the already-built
production artifacts and pushes it to ``towardsai-tutors/ai-tutor-vector-db-public``
(public). The runtime (``app.config.ensure_local_vector_db``) falls back to it
when no usable ``HF_TOKEN`` is present. It contains only the 9 documentation /
reference sources (``DOC_SOURCE_KEYS``); the 5 course sources are stripped.

Why "derive", not "rebuild": the docs-only bundle is exactly what prod would
produce if the course sources had never existed, so we build each artifact the
same way prod does, minus the course rows:

* **Dense (Chroma)** - copy the prod collection and ``delete`` the course
  chunks. This reuses the existing Cohere embeddings, so the build costs **$0**
  and is identical to prod for every docs chunk.
* **BM25 + document dict** - rebuilt from a docs-only view of
  ``all_sources_data.jsonl`` with the same helpers prod uses
  (``build_document_dict`` / ``BM25Index.build``). Pure, no network.
* **KB** - copy ``data/kb`` and prune: drop ``raw/courses`` and ``wiki/courses``
  and filter the course rows out of the generated indexes. The rest of the
  docs wiki is kept verbatim.

Usage::

    # build into data/public_docs_bundle/ and upload to the public repo
    uv run -m data.scraping_scripts.build_public_docs_bundle

    # build only (inspect locally first), skip the upload
    uv run -m data.scraping_scripts.build_public_docs_bundle --skip-upload

Prerequisites: a complete local prod bundle (``data/chroma-db-all_sources/``,
``data/all_sources_data.jsonl``, ``data/kb/``). Build/refresh it with the docs
workflow first. Uploading additionally needs an ``HF_TOKEN`` with write access
to the ``towardsai-tutors`` org.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import shutil
import sqlite3
from pathlib import Path

import chromadb

try:
    from data.scraping_scripts.source_registry import (
        ALL_SOURCES_JSONL,
        CONTEXTUAL_NODES_PKL,
        COURSE_SOURCE_KEYS,
        DOC_SOURCE_KEYS,
    )
    from app.chroma_rag import (
        BM25Index,
        build_chunk_records,
        build_document_dict,
        get_chunk_record_source,
        load_jsonl_documents,
        save_bm25_index,
        save_document_dict,
    )
except ModuleNotFoundError:
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data.scraping_scripts.source_registry import (
        ALL_SOURCES_JSONL,
        CONTEXTUAL_NODES_PKL,
        COURSE_SOURCE_KEYS,
        DOC_SOURCE_KEYS,
    )
    from app.chroma_rag import (
        BM25Index,
        build_chunk_records,
        build_document_dict,
        get_chunk_record_source,
        load_jsonl_documents,
        save_bm25_index,
        save_document_dict,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Layout mirrors the prod bundle so the runtime needs no path changes: the
# public repo has the identical tree (chroma-db-all_sources/, kb/), just with
# the course rows removed. Keep these names in sync with app/config.py.
VECTOR_DB_DIR = "chroma-db-all_sources"
VECTOR_COLLECTION_NAME = "chroma-db-all_sources"
DOCUMENT_DICT_FILE = "document_dict_all_sources.pkl"
BM25_INDEX_FILE = "bm25_index_all_sources.pkl"
KB_DIR_NAME = "kb"

PUBLIC_REPO_ID = "towardsai-tutors/ai-tutor-vector-db-public"
DEFAULT_SOURCE_DIR = Path("data")
DEFAULT_STAGE_DIR = Path("data/public_docs_bundle")

_COURSE_KEYS = frozenset(COURSE_SOURCE_KEYS)


def _is_course_row(row: dict) -> bool:
    return str(row.get("source") or "") in _COURSE_KEYS


def stage_chroma(source_dir: Path, stage_dir: Path, *, vacuum: bool = True) -> dict:
    """Copy the prod Chroma collection and delete the course chunks.

    Reuses the existing docs embeddings (no Cohere call). Returns counts for
    the build summary.
    """
    src = source_dir / VECTOR_DB_DIR
    dst = stage_dir / VECTOR_DB_DIR
    if not src.is_dir():
        raise SystemExit(f"Missing prod Chroma collection: {src}")

    logger.info("Copying Chroma collection %s -> %s", src, dst)
    shutil.copytree(src, dst)

    client = chromadb.PersistentClient(path=str(dst))
    collection = client.get_collection(name=VECTOR_COLLECTION_NAME)
    before = collection.count()
    course_keys = sorted(_COURSE_KEYS)
    if course_keys:
        collection.delete(where={"source": {"$in": course_keys}})
    after = collection.count()
    logger.info(
        "Chroma: kept %s docs chunks, removed %s course chunks",
        after,
        before - after,
    )

    # Release the client so the sqlite file is unlocked before VACUUM.
    del collection
    del client
    try:
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:  # API location varies across chromadb versions
        pass
    gc.collect()

    if vacuum:
        sqlite_path = dst / "chroma.sqlite3"
        try:
            con = sqlite3.connect(sqlite_path)
            con.execute("VACUUM")
            con.close()
            logger.info("Vacuumed %s to reclaim space from deleted rows", sqlite_path)
        except sqlite3.Error as exc:  # best-effort; a larger file is harmless
            logger.warning("VACUUM skipped (%s); bundle will be larger", exc)

    return {"dense_kept": after, "dense_removed": before - after}


def rebuild_retrieval_pkls(source_dir: Path, stage_dir: Path) -> dict:
    """Rebuild the BM25 index and document dict from a docs-only JSONL.

    Mirrors ``create_vector_stores.write_retrieval_artifacts`` so the public
    pkls are byte-for-byte what prod would write without the course rows.
    """
    jsonl_path = source_dir / Path(ALL_SOURCES_JSONL).name
    if not jsonl_path.exists():
        raise SystemExit(f"Missing aggregate corpus: {jsonl_path}")

    all_rows = load_jsonl_documents(str(jsonl_path))
    docs_rows = [row for row in all_rows if not _is_course_row(row)]
    logger.info(
        "Corpus: %s docs documents (dropped %s course documents)",
        len(docs_rows),
        len(all_rows) - len(docs_rows),
    )

    dst = stage_dir / VECTOR_DB_DIR
    save_document_dict(build_document_dict(docs_rows), str(dst / DOCUMENT_DICT_FILE))
    bm25_records = build_chunk_records(docs_rows)
    save_bm25_index(BM25Index.build(bm25_records), str(dst / BM25_INDEX_FILE))
    logger.info(
        "Rebuilt document dict (%s docs) and BM25 index (%s chunks)",
        len(docs_rows),
        len(bm25_records),
    )
    return {"documents": len(docs_rows), "bm25_chunks": len(bm25_records)}


def stage_contextual_nodes(source_dir: Path, stage_dir: Path) -> int:
    """Write a docs-only copy of the contextual-nodes pickle (opt-in).

    Not needed at runtime; included only when ``--include-contextual`` is
    passed so others can rebuild the docs embeddings from scratch.
    """
    import pickle

    src = source_dir / Path(CONTEXTUAL_NODES_PKL).name
    if not src.exists():
        logger.warning("No contextual nodes pickle at %s; skipping", src)
        return 0
    with src.open("rb") as handle:
        records = pickle.load(handle)
    kept = []
    for record in records:
        try:
            record_source = get_chunk_record_source(record)
        except Exception:
            kept.append(record)  # keep records we cannot classify
            continue
        if record_source not in _COURSE_KEYS:
            kept.append(record)
    dst = stage_dir / Path(CONTEXTUAL_NODES_PKL).name
    with dst.open("wb") as handle:
        pickle.dump(kept, handle)
    logger.info("Contextual nodes: kept %s docs chunks", len(kept))
    return len(kept)


def _filter_jsonl(path: Path) -> int:
    """Drop course rows from a generated *.jsonl index in place."""
    if not path.exists():
        return 0
    kept = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("source") or "") not in _COURSE_KEYS:
                kept.append(line if line.endswith("\n") else line + "\n")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(kept)
    return len(kept)


def _filter_tsv(path: Path) -> int:
    """Drop course rows from symbols.tsv in place (keeps the header)."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        kept = [
            row for row in reader if str(row.get("source") or "") not in _COURSE_KEYS
        ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(kept)
    return len(kept)


def stage_kb(source_dir: Path, stage_dir: Path) -> dict:
    """Copy data/kb and prune course content (raw, wiki, generated indexes)."""
    src = source_dir / KB_DIR_NAME
    dst = stage_dir / KB_DIR_NAME
    if not src.is_dir():
        raise SystemExit(f"Missing KB directory: {src}")

    logger.info("Copying KB %s -> %s", src, dst)
    shutil.copytree(src, dst)

    for relative in ("raw/courses", "wiki/courses"):
        target = dst / relative
        if target.exists():
            shutil.rmtree(target)
            logger.info("Removed %s", target)

    generated = dst / "generated"
    manifest_rows = _filter_jsonl(generated / "corpus_manifest.jsonl")
    heading_rows = _filter_jsonl(generated / "headings.jsonl")
    symbol_rows = _filter_tsv(generated / "symbols.tsv")
    logger.info(
        "KB generated indexes: %s manifest rows, %s heading rows, %s symbol rows",
        manifest_rows,
        heading_rows,
        symbol_rows,
    )
    return {"manifest_rows": manifest_rows}


def build_bundle(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    stage_dir: Path = DEFAULT_STAGE_DIR,
    include_contextual: bool = False,
    vacuum: bool = True,
) -> dict:
    if stage_dir.exists():
        logger.info("Clearing existing staging dir %s", stage_dir)
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"sources": sorted(DOC_SOURCE_KEYS)}
    summary.update(stage_chroma(source_dir, stage_dir, vacuum=vacuum))
    summary.update(rebuild_retrieval_pkls(source_dir, stage_dir))
    summary.update(stage_kb(source_dir, stage_dir))
    if include_contextual:
        summary["contextual_chunks"] = stage_contextual_nodes(source_dir, stage_dir)
    return summary


def public_allow_patterns(*, include_contextual: bool) -> list[str]:
    patterns = [f"{VECTOR_DB_DIR}/**", f"{KB_DIR_NAME}/**"]
    if include_contextual:
        patterns.append(Path(CONTEXTUAL_NODES_PKL).name)
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and upload the public docs-only vector-db/KB bundle."
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE_DIR))
    parser.add_argument("--repo", default=PUBLIC_REPO_ID)
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Build the bundle locally but do not push it to Hugging Face.",
    )
    parser.add_argument(
        "--include-contextual",
        action="store_true",
        help="Also ship a docs-only all_sources_contextual_nodes.pkl so others "
        "can rebuild the docs embeddings (larger download; off by default).",
    )
    parser.add_argument(
        "--no-vacuum",
        action="store_true",
        help="Skip the sqlite VACUUM that reclaims space from deleted rows.",
    )
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir)
    summary = build_bundle(
        source_dir=Path(args.source_dir),
        stage_dir=stage_dir,
        include_contextual=args.include_contextual,
        vacuum=not args.no_vacuum,
    )

    print("\nBuilt public docs-only bundle:")
    print(f"  sources:        {', '.join(summary['sources'])}")
    print(f"  documents:      {summary['documents']}")
    print(
        f"  dense chunks:   {summary['dense_kept']} (removed {summary['dense_removed']})"
    )
    print(f"  bm25 chunks:    {summary['bm25_chunks']}")
    print(f"  KB manifest:    {summary['manifest_rows']} rows")
    if "contextual_chunks" in summary:
        print(f"  contextual:     {summary['contextual_chunks']} chunks")
    print(f"  staged at:      {stage_dir}")

    if args.skip_upload:
        print("\n--skip-upload set; not pushing to Hugging Face.")
        return

    from data.scraping_scripts.upload_dbs_to_hf import upload_bundle

    print(f"\nUploading {stage_dir} -> {args.repo} (public) ...")
    upload_bundle(
        args.repo,
        folder_path=str(stage_dir),
        allow_patterns=public_allow_patterns(
            include_contextual=args.include_contextual
        ),
        create_public=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
