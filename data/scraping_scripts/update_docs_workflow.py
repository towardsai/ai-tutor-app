#!/usr/bin/env python
"""
AI Tutor App - Documentation Update Workflow

This script automates the process of updating documentation from GitHub repositories:
1. Download documentation from GitHub using the API
2. Process markdown files to create JSONL data
3. Add contextual information to document nodes
4. Create vector stores
5. Upload databases to HuggingFace

This workflow is specific to updating library documentation (Transformers, PEFT, LlamaIndex, etc.).
For adding courses, use the add_course_workflow.py script instead.

Usage:
    python update_docs_workflow.py --sources [SOURCE1] [SOURCE2] ...

    Additional flags to run specific steps (if you want to restart from a specific point):
    --skip-download         Skip the GitHub download step
    --skip-process          Skip the markdown processing step
    --new-context-only      Only process new content when adding context
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

from data.scraping_scripts.hf_auth import HuggingFaceAuthError, validate_hf_access
from scripts.chroma_rag import get_chunk_record_doc_id

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


def ensure_required_files_exist():
    """Download required data files from HuggingFace if they don't exist locally."""
    # List of files to check and download
    required_files = {
        # Critical files
        "data/all_sources_data.jsonl": "all_sources_data.jsonl",
        "data/all_sources_contextual_nodes.pkl": "all_sources_contextual_nodes.pkl",
        # Documentation source files
        "data/transformers_data.jsonl": "transformers_data.jsonl",
        "data/peft_data.jsonl": "peft_data.jsonl",
        "data/trl_data.jsonl": "trl_data.jsonl",
        "data/llama_index_data.jsonl": "llama_index_data.jsonl",
        "data/langchain_data.jsonl": "langchain_data.jsonl",
        "data/openai_cookbooks_data.jsonl": "openai_cookbooks_data.jsonl",
        # Course files
        "data/tai_blog_data.jsonl": "tai_blog_data.jsonl",
        "data/8-hour_primer_data.jsonl": "8-hour_primer_data.jsonl",
        "data/master_ai_for_work_data.jsonl": "master_ai_for_work_data.jsonl",
        "data/agentic_ai_engineering_data.jsonl": "agentic_ai_engineering_data.jsonl",
        "data/full_stack_ai_engineering_data.jsonl": "full_stack_ai_engineering_data.jsonl",
        "data/beginner_python_for_ai_engineering_data.jsonl": "beginner_python_for_ai_engineering_data.jsonl",
    }

    # Critical files that must be downloaded
    critical_files = [
        "data/all_sources_data.jsonl",
        "data/all_sources_contextual_nodes.pkl",
    ]

    # Check and download each file
    for local_path, remote_filename in required_files.items():
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


# Documentation sources that can be updated via GitHub API
GITHUB_SOURCES = [
    "transformers",
    "peft",
    "trl",
    "llama_index",
    "openai_cookbooks",
    "langchain",
]


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

    logger.info("Successfully processed markdown files")


def get_processed_doc_ids() -> Set[str]:
    """Get set of doc_ids that have already been processed with context."""
    if not os.path.exists("data/all_sources_contextual_nodes.pkl"):
        return set()

    try:
        with open("data/all_sources_contextual_nodes.pkl", "rb") as f:
            nodes = pickle.load(f)
            return {get_chunk_record_doc_id(node) for node in nodes}
    except Exception as e:
        logger.error(f"Error loading processed doc_ids: {e}")
        return set()


def add_context_to_nodes(new_only: bool = False) -> None:
    """Add context to document nodes, optionally processing only new content."""
    logger.info("Adding context to document nodes")

    if new_only:
        # Load all documents
        all_docs = load_jsonl("data/all_sources_data.jsonl")
        processed_ids = get_processed_doc_ids()

        # Filter for unprocessed documents
        new_docs = [doc for doc in all_docs if doc["doc_id"] not in processed_ids]

        if not new_docs:
            logger.info("No new documents to process")
            return

        # Save temporary JSONL with only new documents
        temp_file = "data/new_docs_temp.jsonl"
        save_jsonl(new_docs, temp_file)

        # Temporarily modify the add_context_to_nodes.py script to use the temp file
        cmd = [
            sys.executable,
            "-c",
            f"""
import asyncio
import os
import pickle
import json
from data.scraping_scripts.add_context_to_nodes import create_docs, process
from scripts.chroma_rag import get_chunk_record_source

async def main():
    # First, get the list of sources being updated from the temp file
    updated_sources = set()
    with open("{temp_file}", "r") as f:
        for line in f:
            data = json.loads(line)
            updated_sources.add(data["source"])
    
    print(f"Updating nodes for sources: {{updated_sources}}")
    
    # Process new documents
    documents = create_docs("{temp_file}")
    enhanced_nodes = await process(documents)
    print(f"Generated context for {{len(enhanced_nodes)}} new nodes")
    
    # Load existing nodes if they exist
    existing_nodes = []
    if os.path.exists("data/all_sources_contextual_nodes.pkl"):
        with open("data/all_sources_contextual_nodes.pkl", "rb") as f:
            existing_nodes = pickle.load(f)
        
        # Filter out existing nodes for sources we're updating
        filtered_nodes = []
        removed_count = 0
        
        for node in existing_nodes:
            try:
                source = get_chunk_record_source(node)
                if source not in updated_sources:
                    filtered_nodes.append(node)
                else:
                    removed_count += 1
            except Exception:
                # Keep nodes where we can't determine the source
                filtered_nodes.append(node)
        
        print(f"Removed {{removed_count}} existing nodes for updated sources")
        existing_nodes = filtered_nodes
    
    # Combine filtered existing nodes with new nodes
    all_nodes = existing_nodes + enhanced_nodes
    
    # Save all nodes
    with open("data/all_sources_contextual_nodes.pkl", "wb") as f:
        pickle.dump(all_nodes, f)
    
    print(f"Total nodes in updated file: {{len(all_nodes)}}")

asyncio.run(main())
            """,
        ]
    else:
        # Process all documents
        logger.info("Adding context to all nodes")
        cmd = [sys.executable, "-m", "data.scraping_scripts.add_context_to_nodes"]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error("Error adding context to nodes - check output above")
        sys.exit(1)

    logger.info("Successfully added context to nodes")

    # Clean up temp file if it exists
    if new_only and os.path.exists("data/new_docs_temp.jsonl"):
        os.remove("data/new_docs_temp.jsonl")


def create_vector_stores() -> None:
    """Create vector stores from processed documents."""
    logger.info("Creating vector stores")
    result = run_module("data.scraping_scripts.create_vector_stores", "all_sources")

    if result.returncode != 0:
        logger.error("Error creating vector stores - check output above")
        sys.exit(1)

    logger.info("Successfully created vector stores")


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
        choices=GITHUB_SOURCES,
        default=GITHUB_SOURCES,
        help="GitHub documentation sources to update",
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip downloading from GitHub"
    )
    parser.add_argument(
        "--skip-process", action="store_true", help="Skip processing markdown files"
    )
    parser.add_argument(
        "--process-all-context",
        action="store_true",
        help="Process all content when adding context (default: only process new content)",
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
        "--skip-upload", action="store_true", help="Skip uploading to HuggingFace"
    )
    parser.add_argument(
        "--skip-data-upload",
        action="store_true",
        help="Skip uploading data files (.jsonl and .pkl) to private HuggingFace repo (they are uploaded by default)",
    )

    args = parser.parse_args()

    ensure_hf_access()

    # Ensure required data files exist before proceeding
    ensure_required_files_exist()

    # Execute the workflow steps
    if not args.skip_download:
        download_from_github(args.sources)
        capture_source_versions(args.sources)

    if not args.skip_process:
        process_markdown_files(args.sources)

    if not args.skip_context:
        add_context_to_nodes(not args.process_all_context)

    if not args.skip_vectors:
        create_vector_stores()

    if not args.skip_upload:
        # By default, also upload the data files (JSONL and PKL) unless explicitly skipped
        upload_to_huggingface(not args.skip_data_upload)

    logger.info("Documentation update workflow completed successfully")


if __name__ == "__main__":
    main()
