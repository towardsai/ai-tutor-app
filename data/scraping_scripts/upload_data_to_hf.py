#!/usr/bin/env python
"""
Upload Data Files to HuggingFace

This script uploads key data files to a private HuggingFace dataset repository:
1. all_sources_data.jsonl - The raw document data
2. all_sources_contextual_nodes.pkl - The contextualized chunk manifest used for embedding

This is useful for new team members who need the latest version of the data.

Usage:
    python upload_data_to_hf.py [--repo REPO_ID]

Arguments:
    --repo REPO_ID     HuggingFace dataset repository ID (default: towardsai-tutors/ai-tutor-data)
"""

import argparse
import os

from dotenv import load_dotenv

try:
    from data.scraping_scripts.hf_auth import HuggingFaceAuthError, validate_hf_access
except ModuleNotFoundError:
    from hf_auth import HuggingFaceAuthError, validate_hf_access

load_dotenv()


def upload_files_to_huggingface(repo_id="towardsai-tutors/ai-tutor-data"):
    """Upload data files to a private HuggingFace repository."""
    # Main files to upload
    files_to_upload = [
        # Combined data and vector store
        "data/all_sources_data.jsonl",
        "data/all_sources_contextual_nodes.pkl",
        # Individual source files
        "data/transformers_data.jsonl",
        "data/peft_data.jsonl",
        "data/trl_data.jsonl",
        "data/llama_index_data.jsonl",
        "data/langchain_data.jsonl",
        "data/openai_cookbooks_data.jsonl",
        # Course files
        "data/tai_blog_data.jsonl",
        "data/8-hour_primer_data.jsonl",
        "data/master_ai_for_work_data.jsonl",
        "data/agentic_ai_engineering_data.jsonl",
        "data/full_stack_ai_engineering_data.jsonl",
        "data/beginner_python_for_ai_engineering_data.jsonl",
    ]

    # Filter to only include files that exist
    existing_files = []
    missing_files = []

    for file_path in files_to_upload:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)

    # Critical files must exist
    critical_files = [
        "data/all_sources_data.jsonl",
        "data/all_sources_contextual_nodes.pkl",
    ]
    critical_missing = [f for f in critical_files if f in missing_files]

    if critical_missing:
        print(
            f"Error: The following critical files were not found: {', '.join(critical_missing)}"
        )
        # return False

    if missing_files:
        print(
            f"Warning: The following files were not found and will not be uploaded: {', '.join(missing_files)}"
        )
        print("This is normal if you're only updating certain sources.")

    try:
        api = validate_hf_access(repo_id=repo_id)

        # Check if repository exists, create if it doesn't
        print(f"Repository {repo_id} exists")

        # Upload all existing files
        for file_path in existing_files:
            try:
                file_name = os.path.basename(file_path)
                print(f"Uploading {file_name}...")

                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_name,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(
                    f"Successfully uploaded {file_name} to HuggingFace repository {repo_id}"
                )
            except Exception as e:
                print(f"Error uploading {file_path}: {e}")
                # Continue with other files even if one fails

        return True
    except HuggingFaceAuthError as e:
        print(e)
        return False
    except Exception as e:
        print(f"Error uploading files: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload Data Files to HuggingFace")
    parser.add_argument(
        "--repo",
        default="towardsai-tutors/ai-tutor-data",
        help="HuggingFace dataset repository ID",
    )

    args = parser.parse_args()
    success = upload_files_to_huggingface(args.repo)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
