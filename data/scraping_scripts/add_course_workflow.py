#!/usr/bin/env python
"""
AI Tutor App - Course Addition Workflow

This script guides you through the complete process of adding a new course to the AI Tutor App:

1. Process course markdown files to create JSONL data
2. MANDATORY MANUAL STEP: Add URLs to course content in the generated JSONL
3. Merge course JSONL into all_sources_data.jsonl
4. Add contextual information to document nodes
5. Create vector stores
6. Upload databases to HuggingFace
7. Update UI configuration

Usage:
    uv run python -m data.scraping_scripts.add_course_workflow --course [COURSE_NAME]

    Additional flags to run specific steps (if you want to restart from a specific point):
    --skip-process-md       Skip the markdown processing step
    --skip-merge            Skip merging into all_sources_data.jsonl
    --new-context-only      Only process new content when adding context
    --skip-context          Skip the context addition step entirely
    --skip-vectors          Skip vector store creation
    --skip-upload           Skip uploading to HuggingFace
    --skip-ui-update        Skip updating the UI configuration
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        "data/llm_developer_data.jsonl": "llm_developer_data.jsonl",
        "data/python_primer_data.jsonl": "python_primer_data.jsonl"
    }
    
    # Critical files that must be downloaded
    critical_files = [
        "data/all_sources_data.jsonl",
        "data/all_sources_contextual_nodes.pkl"
    ]
    
    # Check and download each file
    for local_path, remote_filename in required_files.items():
        if not os.path.exists(local_path):
            logger.info(f"{remote_filename} not found. Attempting to download from HuggingFace...")
            try:
                hf_hub_download(
                    token=os.getenv("HF_TOKEN"),
                    repo_id="towardsai-tutors/ai-tutor-data",
                    filename=remote_filename,
                    repo_type="dataset",
                    local_dir="data",
                )
                logger.info(f"Successfully downloaded {remote_filename} from HuggingFace")
            except Exception as e:
                logger.warning(f"Could not download {remote_filename}: {e}")
                
                # Only create empty file for all_sources_data.jsonl if it's missing
                if local_path == "data/all_sources_data.jsonl":
                    logger.warning("Creating a new all_sources_data.jsonl file. This will not include previously existing data.")
                    with open(local_path, "w") as f:
                        pass
                
                # If critical file is missing, print a more serious warning
                if local_path in critical_files:
                    logger.warning(f"Critical file {remote_filename} is missing. The workflow may not function correctly.")
                    
                    if local_path == "data/all_sources_contextual_nodes.pkl":
                        logger.warning("The context addition step will process all documents since no existing contexts were found.")


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
    cmd = ["python", "data/scraping_scripts/process_md_files.py", course_name]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"Error processing markdown files - check output above")
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
    logger.info(f"=== MANDATORY MANUAL STEP: URL ADDITION ===")
    logger.info(f"Please add the URLs to the course content in: {jsonl_path}")
    logger.info(f"For each document in the JSONL file:")
    logger.info(f"1. Open the file in a text editor")
    logger.info(f"2. Find the empty 'url' field for each document")
    logger.info(f"3. Add the appropriate URL from the live course platform")
    logger.info(f"   Example URL format: https://academy.towardsai.net/courses/take/python-for-genai/multimedia/62515980-course-structure")
    logger.info(f"4. Save the file when done")

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


def merge_into_all_sources(course_jsonl_path: str) -> None:
    """Merge the course JSONL into all_sources_data.jsonl."""
    all_sources_path = "data/all_sources_data.jsonl"
    logger.info(f"Merging {course_jsonl_path} into {all_sources_path}")

    # Load course data
    course_data = load_jsonl(course_jsonl_path)

    # Load existing all_sources data if it exists
    all_data = []
    if os.path.exists(all_sources_path):
        all_data = load_jsonl(all_sources_path)

    # Get doc_ids from existing data
    existing_ids = {item["doc_id"] for item in all_data}

    # Add new course data (avoiding duplicates)
    new_items = 0
    for item in course_data:
        if item["doc_id"] not in existing_ids:
            all_data.append(item)
            existing_ids.add(item["doc_id"])
            new_items += 1

    # Save the combined data
    save_jsonl(all_data, all_sources_path)
    logger.info(f"Added {new_items} new documents to {all_sources_path}")


def get_processed_doc_ids() -> Set[str]:
    """Get set of doc_ids that have already been processed with context."""
    if not os.path.exists("data/all_sources_contextual_nodes.pkl"):
        return set()

    try:
        with open("data/all_sources_contextual_nodes.pkl", "rb") as f:
            nodes = pickle.load(f)
            return {node.source_node.node_id for node in nodes}
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
            "python",
            "-c",
            f"""
import asyncio
import os
import pickle
import json
from data.scraping_scripts.add_context_to_nodes import create_docs, process

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
            # Try to extract source from node metadata
            try:
                source = None
                if hasattr(node, 'source_node') and hasattr(node.source_node, 'metadata'):
                    source = node.source_node.metadata.get("source")
                elif hasattr(node, 'metadata'):
                    source = node.metadata.get("source")
                
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
        cmd = ["python", "data/scraping_scripts/add_context_to_nodes.py"]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"Error adding context to nodes - check output above")
        sys.exit(1)

    logger.info("Successfully added context to nodes")

    # Clean up temp file if it exists
    if new_only and os.path.exists("data/new_docs_temp.jsonl"):
        os.remove("data/new_docs_temp.jsonl")


def create_vector_stores() -> None:
    """Create vector stores from processed documents."""
    logger.info("Creating vector stores")
    cmd = ["python", "data/scraping_scripts/create_vector_stores.py", "all_sources"]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"Error creating vector stores - check output above")
        sys.exit(1)

    logger.info("Successfully created vector stores")


def upload_to_huggingface(upload_jsonl: bool = False) -> None:
    """Upload databases to HuggingFace."""
    logger.info("Uploading databases to HuggingFace")
    cmd = ["python", "data/scraping_scripts/upload_dbs_to_hf.py"]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"Error uploading databases - check output above")
        sys.exit(1)

    logger.info("Successfully uploaded databases to HuggingFace")

    if upload_jsonl:
        logger.info("Uploading data files to HuggingFace")

        try:
            # Note: This uses a separate private repository
            cmd = ["python", "data/scraping_scripts/upload_data_to_hf.py"]
            result = subprocess.run(cmd)

            if result.returncode != 0:
                logger.error(f"Error uploading data files - check output above")
                sys.exit(1)

            logger.info("Successfully uploaded data files to HuggingFace")
        except Exception as e:
            logger.error(f"Error uploading JSONL file: {e}")
            sys.exit(1)


def update_ui_files(course_name: str) -> None:
    """Update main.py and setup.py with the new source."""
    logger.info(f"Updating UI files with new course: {course_name}")

    # Get the source configuration for display name
    from data.scraping_scripts.process_md_files import SOURCE_CONFIGS

    if course_name not in SOURCE_CONFIGS:
        logger.error(f"Course {course_name} not found in SOURCE_CONFIGS")
        return

    # Get a readable display name for the UI
    display_name = course_name.replace("_", " ").title()

    # Update setup.py - add to AVAILABLE_SOURCES and AVAILABLE_SOURCES_UI
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

            # Write updated content
            setup_path.write_text(new_content)
            logger.info(f"Updated setup.py with {course_name}")
    else:
        logger.warning(f"setup.py not found at {setup_path}")

    # Update main.py - add to source_mapping
    main_path = Path("scripts/main.py")
    if main_path.exists():
        main_content = main_path.read_text()

        # Check if already added
        if f'"{display_name}": "{course_name}"' in main_content:
            logger.info(f"Course {course_name} already in main.py")
        else:
            # Add to source_mapping
            mapping_start = main_content.find("source_mapping = {")
            mapping_end = main_content.find("}", mapping_start)
            new_main_content = (
                main_content[:mapping_end]
                + f'            "{display_name}": "{course_name}",\n'
                + main_content[mapping_end:]
            )

            # Add to default selected sources if not there
            value_start = new_main_content.find("value=[")
            value_end = new_main_content.find("]", value_start)

            if f'"{display_name}"' not in new_main_content[value_start:value_end]:
                new_main_content = (
                    new_main_content[: value_start + 7]
                    + f'        "{display_name}",\n'
                    + new_main_content[value_start + 7 :]
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
        "--course",
        required=True,
        help="Name of the course to process (must match SOURCE_CONFIGS)",
    )
    parser.add_argument(
        "--skip-process-md",
        action="store_true",
        help="Skip the markdown processing step",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging into all_sources_data.jsonl",
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
    course_name = args.course

    # Ensure required data files exist before proceeding
    ensure_required_files_exist()

    # Get the output file path
    from data.scraping_scripts.process_md_files import SOURCE_CONFIGS

    if course_name not in SOURCE_CONFIGS:
        logger.error(f"Course {course_name} not found in SOURCE_CONFIGS")
        sys.exit(1)

    course_jsonl_path = SOURCE_CONFIGS[course_name]["output_file"]

    # Execute the workflow steps
    if not args.skip_process_md:
        course_jsonl_path = process_markdown_files(course_name)

    # Always do the manual URL addition step for courses
    manual_url_addition(course_jsonl_path)

    if not args.skip_merge:
        merge_into_all_sources(course_jsonl_path)

    if not args.skip_context:
        add_context_to_nodes(not args.process_all_context)

    if not args.skip_vectors:
        create_vector_stores()

    if not args.skip_upload:
        # By default, also upload the data files (JSONL and PKL) unless explicitly skipped
        upload_to_huggingface(not args.skip_data_upload)

    if not args.skip_ui_update:
        update_ui_files(course_name)

    logger.info("Course addition workflow completed successfully")


if __name__ == "__main__":
    main()
