#!/usr/bin/env python
"""
AI Tutor App - Documentation Update Workflow

This script automates the process of updating documentation sources:
1. Download documentation from GitHub or official llms.txt indexes
2. Process markdown files to create JSONL data
3. Add contextual information to document nodes
4. Create vector stores
5. Upload databases to HuggingFace

This workflow is specific to updating library documentation (Transformers, PEFT, LlamaIndex, OpenAI docs, etc.).
For adding courses, use the add_course_workflow.py script instead.

Usage:
    python update_docs_workflow.py --sources [SOURCE1] [SOURCE2] ...

    Additional flags to run specific steps (if you want to restart from a specific point):
    --skip-download         Skip the GitHub download step
    --skip-process          Skip the markdown processing step
    --process-all-context   Regenerate context for every doc (default: only new or changed content)
    --skip-context          Skip the context addition step entirely
    --skip-vectors          Skip vector store creation
    --skip-upload           Skip uploading to HuggingFace
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
from typing import Dict, List, Set

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from data.scraping_scripts.contextual_node_pruning import (
    prune_contextual_nodes_to_active_sources,
)
from data.scraping_scripts.hf_auth import HuggingFaceAuthError, validate_hf_access
from data.scraping_scripts.source_registry import (
    DOC_SOURCE_KEYS,
    GITHUB_SOURCE_KEYS,
    LLMS_TXT_SOURCE_KEYS,
    required_data_files,
    source_output_files,
)
from app.chroma_rag import get_chunk_record_doc_id, get_chunk_record_metadata

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_module(module_name: str, *module_args: str) -> subprocess.CompletedProcess:
    """Run a repository script as a module with the current interpreter."""
    return subprocess.run([sys.executable, "-m", module_name, *module_args])


def ensure_hf_access() -> None:
    try:
        validate_hf_access(repo_id="towardsai-tutors/ai-tutor-data")
    except HuggingFaceAuthError as exc:
        logger.error(str(exc))
        sys.exit(1)


def ensure_required_files_exist(sources_to_regenerate: List[str] | None = None):
    """Download required data files from HuggingFace if they don't exist locally."""
    required_files = required_data_files()
    regenerated_source_files = source_output_files(sources_to_regenerate or [])

    # Critical files that must be downloaded
    critical_files = [
        "data/all_sources_data.jsonl",
        "data/all_sources_contextual_nodes.pkl",
    ]

    # Check and download each file
    for local_path, remote_filename in required_files.items():
        if local_path in regenerated_source_files:
            if not os.path.exists(local_path):
                logger.info(
                    "%s will be regenerated for this run; skipping HuggingFace download",
                    remote_filename,
                )
            continue

        if not os.path.exists(local_path):
            logger.info(
                f"{remote_filename} not found. Attempting to download from HuggingFace..."
            )
            try:
                hf_hub_download(
                    token=os.getenv("HF_TOKEN"),
                    repo_id="towardsai-tutors/ai-tutor-data",
                    filename=remote_filename,
                    repo_type="dataset",
                    local_dir="data",
                )
                logger.info(
                    f"Successfully downloaded {remote_filename} from HuggingFace"
                )
            except Exception as e:
                logger.warning(f"Could not download {remote_filename}: {e}")

                # Only create empty file for all_sources_data.jsonl if it's missing
                if local_path == "data/all_sources_data.jsonl":
                    logger.warning(
                        "Creating a new all_sources_data.jsonl file. This will not include previously existing data."
                    )
                    open(local_path, "w").close()

                # If critical file is missing, print a more serious warning
                if local_path in critical_files:
                    logger.warning(
                        f"Critical file {remote_filename} is missing. The workflow may not function correctly."
                    )

                    if local_path == "data/all_sources_contextual_nodes.pkl":
                        logger.warning(
                            "The context addition step will process all documents since no existing contexts were found."
                        )


# Documentation sources that can be updated automatically
DOC_SOURCES = list(DOC_SOURCE_KEYS)
GITHUB_SOURCES = list(GITHUB_SOURCE_KEYS)
LLMS_TXT_SOURCES = list(LLMS_TXT_SOURCE_KEYS)

# Node-metadata key stamped by add_context_to_nodes.process_chunk with the
# source document's JSONL ``content_hash``. Deliberately duplicated here (not
# imported) so this module keeps its light import footprint —
# add_context_to_nodes pulls in Gemini/tiktoken/llama_index, which this
# module's importers and tests should not need.
# tests/test_incremental_context.py asserts the two constants stay in sync.
DOC_CONTENT_HASH_METADATA_KEY = "doc_content_hash"


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str) -> None:
    """Save data to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def download_from_github(sources: List[str]) -> None:
    """Download documentation from GitHub repositories."""
    logger.info(f"Downloading documentation from GitHub for sources: {sources}")

    for source in sources:
        if source not in GITHUB_SOURCES:
            logger.warning(f"Source {source} is not a GitHub source, skipping download")
            continue

        logger.info(f"Downloading {source} documentation")
        result = run_module("data.scraping_scripts.github_to_markdown_ai_docs", source)

        if result.returncode != 0:
            logger.error(
                f"Error downloading {source} documentation. Stopping workflow to avoid overwriting source JSONL files with incomplete data."
            )
            sys.exit(1)

        logger.info(f"Successfully downloaded {source} documentation")


def download_from_llms_txt(sources: List[str]) -> None:
    """Download documentation from llms.txt indexes."""
    logger.info(
        f"Downloading documentation from llms.txt indexes for sources: {sources}"
    )

    for source in sources:
        if source not in LLMS_TXT_SOURCES:
            logger.warning(
                f"Source {source} is not an llms.txt source, skipping download"
            )
            continue

        logger.info(f"Downloading {source} documentation")
        result = run_module("data.scraping_scripts.llms_txt_to_markdown_docs", source)

        if result.returncode != 0:
            logger.error(
                f"Error downloading {source} documentation. Stopping workflow to avoid overwriting source JSONL files with incomplete data."
            )
            sys.exit(1)

        logger.info(f"Successfully downloaded {source} documentation")


def download_documentation(sources: List[str]) -> None:
    """Download docs with the right source-specific downloader."""
    github_sources = [source for source in sources if source in GITHUB_SOURCES]
    llms_txt_sources = [source for source in sources if source in LLMS_TXT_SOURCES]

    if github_sources:
        download_from_github(github_sources)
    if llms_txt_sources:
        download_from_llms_txt(llms_txt_sources)

    unsupported = sorted(set(sources) - set(github_sources) - set(llms_txt_sources))
    for source in unsupported:
        logger.warning(f"Source {source} is not a downloadable docs source, skipping")


def capture_source_versions(sources: List[str]) -> None:
    """Record latest release tag + SHA + indexed date per source."""
    logger.info(f"Capturing source versions for: {sources}")
    result = run_module(
        "data.scraping_scripts.capture_source_versions", "--sources", *sources
    )
    if result.returncode != 0:
        logger.warning(
            "Version capture finished with non-zero exit; continuing workflow."
        )
    else:
        logger.info("Source versions captured successfully")


def process_markdown_files(sources: List[str]) -> None:
    """Process markdown files for specific sources."""
    logger.info(f"Processing markdown files for sources: {sources}")

    result = run_module("data.scraping_scripts.process_md_files", *sources)

    if result.returncode != 0:
        logger.error("Error processing markdown files - check output above")
        sys.exit(1)

    if len(sources) == 1:
        from data.scraping_scripts.process_md_files import combine_all_sources

        logger.info("Rebuilding all_sources_data.jsonl after single-source update")
        combine_all_sources(sources)

    logger.info("Successfully processed markdown files")


def build_doc_hash_map(nodes: List) -> Dict[str, str | None]:
    """Map doc_id -> stored ``doc_content_hash`` for a list of contextual nodes.

    Membership means the doc already has contextual nodes; the value is the
    document-level content hash stamped at context-generation time, or ``None``
    for legacy nodes written before hashes existed. If a doc has a mix of
    hashed and unhashed nodes (shouldn't happen, but resolve it sanely), any
    stored hash for the doc counts. Nodes whose doc_id can't be determined are
    skipped.
    """
    doc_hashes: Dict[str, str | None] = {}
    for node in nodes:
        try:
            doc_id = get_chunk_record_doc_id(node)
        except Exception:
            continue
        try:
            metadata = get_chunk_record_metadata(node)
        except Exception:
            metadata = {}
        stored_hash = metadata.get(DOC_CONTENT_HASH_METADATA_KEY)
        if doc_id not in doc_hashes:
            doc_hashes[doc_id] = stored_hash
        elif doc_hashes[doc_id] is None and stored_hash is not None:
            doc_hashes[doc_id] = stored_hash
    return doc_hashes


def get_processed_doc_hashes() -> Dict[str, str | None]:
    """Get doc_id -> stored content hash for docs already processed with context.

    Key membership carries the old ``get_processed_doc_ids`` contract (the doc
    has contextual nodes in the PKL); the value adds the stored
    ``doc_content_hash`` (``None`` for legacy nodes), enabling changed-content
    detection. See ``build_doc_hash_map`` for resolution rules.
    """
    if not os.path.exists("data/all_sources_contextual_nodes.pkl"):
        return {}

    try:
        with open("data/all_sources_contextual_nodes.pkl", "rb") as f:
            nodes = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading processed doc hashes: {e}")
        return {}

    return build_doc_hash_map(nodes)


def select_docs_to_process(
    all_docs: List[Dict],
    stored_hashes: Dict[str, str | None],
) -> tuple[List[Dict], Dict[str, int]]:
    """Pick the docs needing (re)contextualization, with selection stats.

    A doc is selected when:

    - its doc_id has no contextual nodes yet (**new**), or
    - its stored ``doc_content_hash`` differs from the JSONL row's current
      ``content_hash`` (**changed** in place; doc_ids are path-based and stable
      across content edits, so id membership alone can never catch this).

    Docs whose pkl nodes lack the hash field (**legacy**, written before this
    fix) are treated as UNCHANGED so shipping hash-forward detection does not
    trigger a surprise full-corpus Gemini reprocess; a one-time
    ``--process-all-context`` run baselines hashes for the whole corpus, after
    which in-place edits are picked up automatically. Rows without a
    ``content_hash`` (older JSONLs) are likewise treated as unchanged.

    Returns ``(docs_to_process, stats)`` with stats keys ``new``, ``changed``,
    and ``legacy_unhashed``.
    """
    docs_to_process: List[Dict] = []
    stats = {"new": 0, "changed": 0, "legacy_unhashed": 0}

    for doc in all_docs:
        doc_id = doc["doc_id"]
        if doc_id not in stored_hashes:
            stats["new"] += 1
            docs_to_process.append(doc)
            continue

        stored_hash = stored_hashes[doc_id]
        if stored_hash is None:
            stats["legacy_unhashed"] += 1
            continue

        row_hash = doc.get("content_hash")
        if row_hash is not None and row_hash != stored_hash:
            stats["changed"] += 1
            docs_to_process.append(doc)

    return docs_to_process, stats


def merge_contextual_nodes(
    existing_nodes: List,
    new_nodes: List,
    reprocessed_doc_ids: Set[str],
) -> List:
    """Merge freshly contextualized nodes into an existing contextual-node list.

    Every existing node is preserved except those whose doc_id is in
    ``reprocessed_doc_ids`` (the docs being (re)processed in this run); their
    old nodes are superseded by ``new_nodes``, which are appended. Nodes whose
    doc_id cannot be determined are kept. Nodes are never removed by source:
    an incremental run that adds one new page to a source must keep every
    other already-indexed page of that source intact.
    """
    merged: List = []
    replaced_count = 0
    unknown_count = 0

    for node in existing_nodes:
        try:
            doc_id = get_chunk_record_doc_id(node)
        except Exception:
            # Keep nodes whose doc_id can't be determined
            merged.append(node)
            unknown_count += 1
            continue

        if doc_id in reprocessed_doc_ids:
            replaced_count += 1
        else:
            merged.append(node)

    logger.info(
        "Merging contextual nodes: kept %s existing nodes "
        "(%s with undetermined doc_id), replaced %s nodes for reprocessed "
        "doc_ids, appended %s new nodes",
        len(merged),
        unknown_count,
        replaced_count,
        len(new_nodes),
    )
    return merged + list(new_nodes)


def _add_context_for_new_docs(temp_file: str, reprocessed_doc_ids: Set[str]) -> None:
    """Generate context for the docs in ``temp_file`` and merge into the PKL."""
    # Imported lazily: pulls in Gemini/tiktoken/llama_index, which the rest of
    # this module (and its importers/tests) should not need.
    import asyncio

    from data.scraping_scripts.add_context_to_nodes import create_docs, process

    documents = create_docs(temp_file)
    enhanced_nodes = asyncio.run(process(documents))
    logger.info("Generated context for %s new nodes", len(enhanced_nodes))

    pkl_path = "data/all_sources_contextual_nodes.pkl"
    existing_nodes: List = []
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            existing_nodes = pickle.load(f)

    all_nodes = merge_contextual_nodes(
        existing_nodes, enhanced_nodes, reprocessed_doc_ids
    )

    with open(pkl_path, "wb") as f:
        pickle.dump(all_nodes, f)

    logger.info("Total nodes in updated file: %s", len(all_nodes))


def add_context_to_nodes(new_only: bool = False) -> None:
    """Add context to document nodes, optionally only new/changed content."""
    logger.info("Adding context to document nodes")

    if new_only:
        # Load all documents
        all_docs = load_jsonl("data/all_sources_data.jsonl")
        stored_hashes = get_processed_doc_hashes()

        # Select docs with no contextual nodes yet (new) plus docs whose
        # content hash differs from the one stamped in the pkl (changed).
        docs_to_process, stats = select_docs_to_process(all_docs, stored_hashes)
        logger.info(
            "Context selection: %s new docs, %s changed docs, %s legacy docs "
            "without a stored content hash (treated as unchanged; a one-time "
            "--process-all-context run baselines hashes for the whole corpus)",
            stats["new"],
            stats["changed"],
            stats["legacy_unhashed"],
        )

        if not docs_to_process:
            logger.info("No new or changed documents to process")
            return

        # Save temporary JSONL with only the documents to (re)process
        temp_file = "data/new_docs_temp.jsonl"
        save_jsonl(docs_to_process, temp_file)

        try:
            # merge_contextual_nodes replaces every existing node whose doc_id
            # is in this set, so a changed doc's old nodes are dropped in
            # favor of the freshly contextualized ones.
            _add_context_for_new_docs(
                temp_file, {doc["doc_id"] for doc in docs_to_process}
            )
        except Exception:
            logger.exception("Error adding context to nodes")
            sys.exit(1)

        logger.info("Successfully added context to nodes")

        # Clean up temp file (kept on failure to help debugging)
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return

    # Process all documents
    logger.info("Adding context to all nodes")
    result = subprocess.run(
        [sys.executable, "-m", "data.scraping_scripts.add_context_to_nodes"]
    )

    if result.returncode != 0:
        logger.error("Error adding context to nodes - check output above")
        sys.exit(1)

    logger.info("Successfully added context to nodes")


def create_vector_stores() -> None:
    """Create vector stores from processed documents."""
    logger.info("Creating vector stores")
    result = run_module("data.scraping_scripts.create_vector_stores", "all_sources")

    if result.returncode != 0:
        logger.error("Error creating vector stores - check output above")
        sys.exit(1)

    logger.info("Successfully created vector stores")


def build_kb_artifacts() -> None:
    """Build kb/raw, kb/generated, and refresh kb/wiki.

    update_kb_wiki auto-promotes to seed_defaults when wiki/ is empty, so the
    same invocation works for both fresh and incremental rebuilds. On
    incremental runs, maintainer-authored prose outside `<!-- AUTO-GENERATED -->`
    markers is preserved. Ends with lint_kb_wiki so a wiki page referencing a
    nonexistent raw/wiki/generated path fails the build before upload.
    """
    logger.info("Building KB artifacts")
    result = run_module("data.scraping_scripts.build_kb_artifacts")
    if result.returncode != 0:
        logger.error("Error building KB artifacts - check output above")
        sys.exit(1)

    result = run_module("data.scraping_scripts.update_kb_wiki")
    if result.returncode != 0:
        logger.error("Error updating KB wiki - check output above")
        sys.exit(1)

    result = run_module("data.scraping_scripts.lint_kb_wiki")
    if result.returncode != 0:
        logger.error(
            "KB wiki lint found broken file references - check output above. "
            "Fix the wiki (or the artifacts it points at) before uploading."
        )
        sys.exit(1)
    logger.info("Successfully built KB artifacts")


def upload_to_huggingface(upload_jsonl: bool = False) -> None:
    """Upload databases to HuggingFace."""
    logger.info("Uploading databases to HuggingFace")
    result = run_module("data.scraping_scripts.upload_dbs_to_hf")

    if result.returncode != 0:
        logger.error("Error uploading databases - check output above")
        sys.exit(1)

    logger.info("Successfully uploaded databases to HuggingFace")

    if upload_jsonl:
        logger.info("Uploading data files to HuggingFace")

        try:
            # Note: This uses a separate private repository
            result = run_module("data.scraping_scripts.upload_data_to_hf")

            if result.returncode != 0:
                logger.error("Error uploading data files - check output above")
                sys.exit(1)

            logger.info("Successfully uploaded data files to HuggingFace")
        except Exception as e:
            logger.error(f"Error uploading JSONL file: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="AI Tutor App Documentation Update Workflow"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=DOC_SOURCES,
        default=DOC_SOURCES,
        help="Documentation sources to update",
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip downloading source docs"
    )
    parser.add_argument(
        "--skip-process", action="store_true", help="Skip processing markdown files"
    )
    parser.add_argument(
        "--process-all-context",
        action="store_true",
        help=(
            "Process all content when adding context (default: only process "
            "new or changed content; also baselines doc content hashes for "
            "legacy nodes written before hashes were stamped)"
        ),
    )
    parser.add_argument(
        "--skip-context",
        action="store_true",
        help="Skip the context addition step entirely",
    )
    parser.add_argument(
        "--skip-vectors", action="store_true", help="Skip vector store creation"
    )
    parser.add_argument(
        "--skip-kb",
        action="store_true",
        help="Skip generated KB markdown/wiki artifact creation",
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="Skip uploading to HuggingFace"
    )
    parser.add_argument(
        "--skip-data-upload",
        action="store_true",
        help="Skip uploading data files (.jsonl and .pkl) to private HuggingFace repo (they are uploaded by default)",
    )

    args = parser.parse_args()

    ensure_hf_access()

    # Keep untouched source JSONLs by downloading them when needed, but don't
    # require first-time sources that this run is about to regenerate.
    sources_to_regenerate = [] if args.skip_process else args.sources
    ensure_required_files_exist(sources_to_regenerate=sources_to_regenerate)

    # Execute the workflow steps
    if not args.skip_download:
        download_documentation(args.sources)
        capture_source_versions(args.sources)

    if not args.skip_process:
        process_markdown_files(args.sources)

    if not args.skip_kb:
        build_kb_artifacts()

    if not args.skip_context:
        add_context_to_nodes(not args.process_all_context)

    prune_contextual_nodes_to_active_sources()

    if not args.skip_vectors:
        create_vector_stores()

    if not args.skip_upload:
        # By default, also upload the data files (JSONL and PKL) unless explicitly skipped
        upload_to_huggingface(not args.skip_data_upload)

    logger.info("Documentation update workflow completed successfully")


if __name__ == "__main__":
    main()
