#!/usr/bin/env python
"""
AI Tutor App - Course Addition Workflow

This script guides you through adding or updating one or more courses in the AI Tutor App:

1. Process course markdown files to create per-course JSONL data
2. MANDATORY MANUAL STEP: Add URLs to each course JSONL
3. Rebuild all_sources_data.jsonl from the per-source JSONLs listed in SOURCE_CONFIGS
   (this naturally drops any retired sources no longer in SOURCE_CONFIGS)
4. Optionally purge retired sources from the contextual-nodes PKL
5. Add contextual information to document nodes (only new docs by default)
6. Create vector stores
7. Upload databases + data files to HuggingFace
8. Update UI configuration for each course

Usage:
    uv run python -m data.scraping_scripts.add_course_workflow --courses [COURSE_1] [COURSE_2] ...

    Additional flags:
    --purge-sources S1 S2   Remove nodes for these source names from the contextual-nodes PKL
                            (use after renaming/retiring a course, e.g. llm_developper, python_primer)
    --skip-process-md       Skip the markdown processing step
    --skip-merge            Skip rebuilding all_sources_data.jsonl
    --process-all-context   Regenerate context for all docs (default: only new docs)
    --skip-context          Skip the context addition step entirely
    --skip-vectors          Skip vector store creation
    --skip-upload           Skip uploading to HuggingFace
    --skip-data-upload      Skip uploading .jsonl/.pkl data files (they upload by default)
    --skip-ui-update        Skip updating the UI configuration
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path
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


def process_markdown_files(course_name: str) -> str:
    """Process markdown files for a specific course. Returns path to output JSONL."""
    logger.info(f"Processing markdown files for course: {course_name}")
    result = run_module("data.scraping_scripts.process_md_files", course_name)

    if result.returncode != 0:
        logger.error("Error processing markdown files - check output above")
        sys.exit(1)

    logger.info(f"Successfully processed markdown files for {course_name}")

    # Determine the output file path from process_md_files.py
    from data.scraping_scripts.process_md_files import SOURCE_CONFIGS

    if course_name not in SOURCE_CONFIGS:
        logger.error(f"Course {course_name} not found in SOURCE_CONFIGS")
        sys.exit(1)

    output_file = SOURCE_CONFIGS[course_name]["output_file"]
    return output_file


def manual_url_addition(jsonl_path: str) -> None:
    """Guide the user through manually adding URLs to the course JSONL."""
    logger.info("=== MANDATORY MANUAL STEP: URL ADDITION ===")
    logger.info(f"Please add the URLs to the course content in: {jsonl_path}")
    logger.info("For each document in the JSONL file:")
    logger.info("1. Open the file in a text editor")
    logger.info("2. Find the empty 'url' field for each document")
    logger.info("3. Add the appropriate URL from the live course platform")
    logger.info(
        "   Example URL format: https://academy.towardsai.net/courses/take/python-for-genai/multimedia/62515980-course-structure"
    )
    logger.info("4. Save the file when done")

    # Check if URLs are present
    data = load_jsonl(jsonl_path)
    missing_urls = sum(1 for item in data if not item.get("url"))

    if missing_urls > 0:
        logger.warning(f"Found {missing_urls} documents without URLs in {jsonl_path}")

        answer = input(
            f"\n{missing_urls} documents are missing URLs. Have you added all the URLs? (yes/no): "
        )
        if answer.lower() not in ["yes", "y"]:
            logger.info("Please add the URLs and run the script again.")
            sys.exit(0)
    else:
        logger.info("All documents have URLs. Continuing with the workflow.")


def rebuild_all_sources(courses: List[str]) -> None:
    """Rebuild all_sources_data.jsonl from the per-source JSONLs in SOURCE_CONFIGS.

    Unlike the previous append-style merge, this drops any source whose entry
    has been removed from SOURCE_CONFIGS (e.g. retired/renamed courses) and
    reloads every other source from its own JSONL. Safe against stale rows
    for sources like llm_developper / python_primer that were renamed.
    """
    from data.scraping_scripts.process_md_files import combine_all_sources

    logger.info("Rebuilding data/all_sources_data.jsonl from per-source JSONLs")
    combine_all_sources(courses)


def purge_sources_from_pkl(sources_to_purge: List[str]) -> None:
    """Remove nodes belonging to the given source names from the contextual-nodes PKL.

    Use this after renaming/retiring a course so its old chunks don't linger
    in the vector DB on the next rebuild.
    """
    from scripts.chroma_rag import get_chunk_record_source

    pkl_path = "data/all_sources_contextual_nodes.pkl"
    if not os.path.exists(pkl_path):
        logger.info(f"{pkl_path} does not exist; nothing to purge")
        return

    with open(pkl_path, "rb") as f:
        nodes = pickle.load(f)

    purge_set = set(sources_to_purge)
    kept: List = []
    removed = 0
    unknown_source = 0
    for node in nodes:
        try:
            source = get_chunk_record_source(node)
        except Exception:
            kept.append(node)
            unknown_source += 1
            continue
        if source in purge_set:
            removed += 1
        else:
            kept.append(node)

    with open(pkl_path, "wb") as f:
        pickle.dump(kept, f)

    logger.info(
        f"Purged {removed} nodes for sources {sorted(purge_set)} from {pkl_path}; "
        f"{len(kept)} remain ({unknown_source} retained with undetermined source)"
    )


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


def update_ui_files(course_name: str) -> None:
    """Update setup.py and the default selected UI sources for the new course."""
    logger.info(f"Updating UI files with new course: {course_name}")

    # Get the source configuration for display name
    from data.scraping_scripts.process_md_files import SOURCE_CONFIGS

    if course_name not in SOURCE_CONFIGS:
        logger.error(f"Course {course_name} not found in SOURCE_CONFIGS")
        return

    # Get a readable display name for the UI
    display_name = course_name.replace("_", " ").title()

    # Update setup.py - add to AVAILABLE_SOURCES, AVAILABLE_SOURCES_UI, and SOURCE_UI_TO_KEY
    setup_path = Path("scripts/setup.py")
    if setup_path.exists():
        setup_content = setup_path.read_text()

        # Check if already added
        if f'"{course_name}"' in setup_content:
            logger.info(f"Course {course_name} already in setup.py")
        else:
            # Add to AVAILABLE_SOURCES_UI
            ui_list_start = setup_content.find("AVAILABLE_SOURCES_UI = [")
            ui_list_end = setup_content.find("]", ui_list_start)
            new_ui_content = (
                setup_content[:ui_list_end]
                + f'    "{display_name}",\n'
                + setup_content[ui_list_end:]
            )

            # Add to AVAILABLE_SOURCES
            sources_list_start = new_ui_content.find("AVAILABLE_SOURCES = [")
            sources_list_end = new_ui_content.find("]", sources_list_start)
            new_content = (
                new_ui_content[:sources_list_end]
                + f'    "{course_name}",\n'
                + new_ui_content[sources_list_end:]
            )

            mapping_start = new_content.find("SOURCE_UI_TO_KEY = {")
            mapping_end = new_content.find("}", mapping_start)
            if f'"{display_name}": "{course_name}"' not in new_content:
                new_content = (
                    new_content[:mapping_end]
                    + f'    "{display_name}": "{course_name}",\n'
                    + new_content[mapping_end:]
                )

            # Write updated content
            setup_path.write_text(new_content)
            logger.info(f"Updated setup.py with {course_name}")
    else:
        logger.warning(f"setup.py not found at {setup_path}")

    # Update main.py - add to the default selected source list
    main_path = Path("scripts/main.py")
    if main_path.exists():
        main_content = main_path.read_text()

        # Check if already added
        if f'"{display_name}"' in main_content:
            logger.info(f"Course {course_name} already in main.py")
        else:
            value_start = main_content.find("value=[")
            value_end = main_content.find("]", value_start)
            new_main_content = main_content
            if value_start != -1 and value_end != -1:
                new_main_content = (
                    main_content[: value_start + 7]
                    + f'        "{display_name}",\n'
                    + main_content[value_start + 7 :]
                )

            # Write updated content
            main_path.write_text(new_main_content)
            logger.info(f"Updated main.py with {course_name}")
    else:
        logger.warning(f"main.py not found at {main_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Tutor App Course Addition Workflow"
    )
    parser.add_argument(
        "--courses",
        nargs="+",
        required=True,
        help="One or more course names to process (must match SOURCE_CONFIGS)",
    )
    parser.add_argument(
        "--purge-sources",
        nargs="+",
        default=[],
        help="Source names to remove from the contextual-nodes PKL before re-adding context "
        "(use for retired/renamed courses like llm_developper, python_primer)",
    )
    parser.add_argument(
        "--skip-process-md",
        action="store_true",
        help="Skip the markdown processing step",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip rebuilding all_sources_data.jsonl",
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
        "--skip-ui-update",
        action="store_true",
        help="Skip updating the UI configuration",
    )
    parser.add_argument(
        "--skip-data-upload",
        action="store_true",
        help="Skip uploading data files to private HuggingFace repo (they are uploaded by default)",
    )

    args = parser.parse_args()
    courses: List[str] = args.courses

    ensure_hf_access()

    # Ensure required data files exist before proceeding
    ensure_required_files_exist()

    from data.scraping_scripts.process_md_files import SOURCE_CONFIGS

    # Validate every course up front so we fail fast
    for course_name in courses:
        if course_name not in SOURCE_CONFIGS:
            logger.error(f"Course {course_name} not found in SOURCE_CONFIGS")
            sys.exit(1)

    # Per-course: process markdown + manual URL addition
    for course_name in courses:
        course_jsonl_path = SOURCE_CONFIGS[course_name]["output_file"]

        if not args.skip_process_md:
            if os.path.exists(course_jsonl_path):
                logger.info(
                    f"JSONL file {course_jsonl_path} already exists. Skipping markdown processing for {course_name}."
                )
            else:
                course_jsonl_path = process_markdown_files(course_name)

        # Manual URL addition is mandatory for each course
        manual_url_addition(course_jsonl_path)

    # Shared steps — run once across all courses
    if not args.skip_merge:
        rebuild_all_sources(courses)

    if args.purge_sources:
        purge_sources_from_pkl(args.purge_sources)

    if not args.skip_context:
        add_context_to_nodes(not args.process_all_context)

    if not args.skip_vectors:
        create_vector_stores()

    if not args.skip_upload:
        # By default, also upload the data files (JSONL and PKL) unless explicitly skipped
        upload_to_huggingface(not args.skip_data_upload)

    if not args.skip_ui_update:
        for course_name in courses:
            update_ui_files(course_name)

    logger.info("Course addition workflow completed successfully")


if __name__ == "__main__":
    main()
