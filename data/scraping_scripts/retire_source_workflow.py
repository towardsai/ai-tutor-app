#!/usr/bin/env python
"""
Retire one or more sources from the AI Tutor data pipeline.

This removes the source from:
1. data/all_sources_data.jsonl
2. data/all_sources_contextual_nodes.pkl
3. the rebuilt Chroma vector store
4. the private Hugging Face data repository's per-source JSONL file

Example:
    uv run -m data.scraping_scripts.retire_source_workflow --sources 8-hour_primer --yes
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from data.scraping_scripts.hf_auth import HuggingFaceAuthError, validate_hf_access
from scripts.chroma_rag import get_chunk_record_source

try:
    from data.scraping_scripts.process_md_files import SOURCE_CONFIGS
except Exception:
    SOURCE_CONFIGS = {}

load_dotenv()

DATA_REPO_ID = "towardsai-tutors/ai-tutor-data"
VECTOR_REPO_ID = "towardsai-tutors/ai-tutor-vector-db"
ALL_SOURCES_JSONL = Path("data/all_sources_data.jsonl")
CONTEXTUAL_NODES = Path("data/all_sources_contextual_nodes.pkl")


def run_module(module_name: str, *module_args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", module_name, *module_args])


def backup_file(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup_path = path.with_name(f"{path.name}.{timestamp}.bak")
    shutil.copy2(path, backup_path)
    return backup_path


def source_jsonl_filename(source: str) -> str:
    config = SOURCE_CONFIGS.get(source)
    if config:
        return Path(config["output_file"]).name
    return f"{source}_data.jsonl"


def ensure_safe_sources(sources: list[str]) -> None:
    for source in sources:
        if not source or source in {".", ".."} or "/" in source or "\\" in source:
            raise SystemExit(f"Unsafe source name: {source!r}")
        if source == "all_sources":
            raise SystemExit("Refusing to retire the aggregate source 'all_sources'.")


def ensure_required_files_exist(data_repo_id: str) -> None:
    required_files = {
        ALL_SOURCES_JSONL: "all_sources_data.jsonl",
        CONTEXTUAL_NODES: "all_sources_contextual_nodes.pkl",
    }

    for local_path, remote_filename in required_files.items():
        if local_path.exists():
            continue

        print(f"{remote_filename} not found locally. Downloading from {data_repo_id}...")
        hf_hub_download(
            token=os.getenv("HF_TOKEN"),
            repo_id=data_repo_id,
            filename=remote_filename,
            repo_type="dataset",
            local_dir="data",
        )


def filter_all_sources_jsonl(sources_to_retire: set[str], dry_run: bool) -> Counter[str]:
    if not ALL_SOURCES_JSONL.exists():
        raise SystemExit(f"Missing required file: {ALL_SOURCES_JSONL}")

    kept_rows: list[dict[str, Any]] = []
    removed_counts: Counter[str] = Counter()

    with ALL_SOURCES_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source")
            if source in sources_to_retire:
                removed_counts[source] += 1
            else:
                kept_rows.append(row)

    if not dry_run:
        backup_path = backup_file(ALL_SOURCES_JSONL)
        print(f"Backed up {ALL_SOURCES_JSONL} to {backup_path}")
        with ALL_SOURCES_JSONL.open("w", encoding="utf-8") as handle:
            for row in kept_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return removed_counts


def filter_contextual_nodes(sources_to_retire: set[str], dry_run: bool) -> Counter[str]:
    if not CONTEXTUAL_NODES.exists():
        raise SystemExit(f"Missing required file: {CONTEXTUAL_NODES}")

    with CONTEXTUAL_NODES.open("rb") as handle:
        nodes = pickle.load(handle)

    kept_nodes: list[Any] = []
    removed_counts: Counter[str] = Counter()
    unknown_source = 0

    for node in nodes:
        try:
            source = get_chunk_record_source(node)
        except Exception:
            unknown_source += 1
            kept_nodes.append(node)
            continue

        if source in sources_to_retire:
            removed_counts[str(source)] += 1
        else:
            kept_nodes.append(node)

    if not dry_run:
        backup_path = backup_file(CONTEXTUAL_NODES)
        print(f"Backed up {CONTEXTUAL_NODES} to {backup_path}")
        with CONTEXTUAL_NODES.open("wb") as handle:
            pickle.dump(kept_nodes, handle)

    if unknown_source:
        print(f"Kept {unknown_source} contextual nodes with undetermined source.")

    return removed_counts


def delete_local_source_files(sources: list[str], dry_run: bool) -> None:
    for source in sources:
        source_path = Path("data") / source_jsonl_filename(source)
        if not source_path.exists():
            continue
        if dry_run:
            print(f"Would delete local source file: {source_path}")
            continue
        backup_path = backup_file(source_path)
        source_path.unlink()
        print(f"Deleted local source file {source_path} after backup to {backup_path}")


def rebuild_vector_store() -> None:
    if not os.getenv("COHERE_API_KEY"):
        raise SystemExit("COHERE_API_KEY is required to rebuild the Chroma vector store.")

    print("Rebuilding Chroma vector store for all_sources...")
    result = run_module("data.scraping_scripts.create_vector_stores", "all_sources")
    if result.returncode != 0:
        raise SystemExit("Error rebuilding vector store. Check output above.")


def upload_data_files(data_repo_id: str) -> None:
    api = validate_hf_access(repo_id=data_repo_id)
    for local_path in (ALL_SOURCES_JSONL, CONTEXTUAL_NODES):
        print(f"Uploading {local_path.name} to {data_repo_id}...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=data_repo_id,
            repo_type="dataset",
        )


def delete_remote_source_files(
    data_repo_id: str,
    sources: list[str],
    remote_source_files: list[str] | None,
) -> None:
    api = validate_hf_access(repo_id=data_repo_id)
    filenames = (
        remote_source_files
        if remote_source_files is not None
        else sorted({source_jsonl_filename(source) for source in sources})
    )

    for filename in filenames:
        print(f"Deleting {filename} from {data_repo_id}...")
        try:
            api.delete_file(
                path_in_repo=filename,
                repo_id=data_repo_id,
                repo_type="dataset",
            )
        except Exception as exc:
            print(f"Warning: could not delete {filename}: {exc}")


def upload_vector_store(vector_repo_id: str) -> None:
    print(f"Uploading rebuilt vector store to {vector_repo_id}...")
    result = run_module("data.scraping_scripts.upload_dbs_to_hf", "--repo", vector_repo_id)
    if result.returncode != 0:
        raise SystemExit("Error uploading vector store. Check output above.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retire sources from all_sources data, contextual nodes, and Chroma."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source keys to retire, for example: 8-hour_primer transformers",
    )
    parser.add_argument(
        "--data-repo",
        default=DATA_REPO_ID,
        help=f"Hugging Face data dataset repo. Default: {DATA_REPO_ID}",
    )
    parser.add_argument(
        "--vector-repo",
        default=VECTOR_REPO_ID,
        help=f"Hugging Face vector dataset repo. Default: {VECTOR_REPO_ID}",
    )
    parser.add_argument(
        "--remote-source-files",
        nargs="*",
        help=(
            "Optional exact per-source JSONL filenames to delete from the data repo. "
            "Defaults to SOURCE_CONFIGS output files or <source>_data.jsonl."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download missing aggregate data files from Hugging Face.",
    )
    parser.add_argument(
        "--skip-vector-rebuild",
        action="store_true",
        help="Filter data files but do not rebuild or upload Chroma.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Do not upload updated data or vector files to Hugging Face.",
    )
    parser.add_argument(
        "--skip-data-upload",
        action="store_true",
        help="Do not upload all_sources_data.jsonl/all_sources_contextual_nodes.pkl.",
    )
    parser.add_argument(
        "--keep-remote-source-files",
        action="store_true",
        help="Do not delete retired per-source JSONL files from the data repo.",
    )
    parser.add_argument(
        "--delete-local-source-files",
        action="store_true",
        help="Delete matching local per-source JSONL files after backing them up.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without changing files or uploading.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the retirement operation. Required unless --dry-run is used.",
    )

    args = parser.parse_args()
    ensure_safe_sources(args.sources)

    if not args.dry_run and not args.yes:
        parser.error("Pass --yes to retire sources, or --dry-run to preview changes.")

    missing_required_files = [
        path for path in (ALL_SOURCES_JSONL, CONTEXTUAL_NODES) if not path.exists()
    ]
    needs_data_repo = (
        (not args.skip_download and bool(missing_required_files))
        or (not args.dry_run and not args.skip_upload and not args.skip_data_upload)
    )
    needs_vector_repo = (
        not args.dry_run and not args.skip_upload and not args.skip_vector_rebuild
    )

    try:
        if needs_data_repo:
            validate_hf_access(repo_id=args.data_repo)
        if needs_vector_repo:
            validate_hf_access(repo_id=args.vector_repo)
    except HuggingFaceAuthError as exc:
        raise SystemExit(str(exc)) from exc

    if not args.skip_download:
        ensure_required_files_exist(args.data_repo)

    sources_to_retire = set(args.sources)
    print(f"Retiring sources: {', '.join(sorted(sources_to_retire))}")

    removed_docs = filter_all_sources_jsonl(sources_to_retire, args.dry_run)
    removed_nodes = filter_contextual_nodes(sources_to_retire, args.dry_run)

    for source in sorted(sources_to_retire):
        print(
            f"{source}: removed {removed_docs[source]} raw docs and "
            f"{removed_nodes[source]} contextual chunks"
        )

    if args.delete_local_source_files:
        delete_local_source_files(args.sources, args.dry_run)

    if args.dry_run:
        print("Dry run complete. No files were changed.")
        return

    if not args.skip_vector_rebuild:
        rebuild_vector_store()

    if args.skip_upload:
        print("Skipping all Hugging Face uploads.")
        return

    if not args.skip_data_upload:
        upload_data_files(args.data_repo)
        if not args.keep_remote_source_files:
            delete_remote_source_files(
                args.data_repo,
                args.sources,
                args.remote_source_files,
            )

    if not args.skip_vector_rebuild:
        upload_vector_store(args.vector_repo)

    print("Source retirement completed successfully.")


if __name__ == "__main__":
    main()
