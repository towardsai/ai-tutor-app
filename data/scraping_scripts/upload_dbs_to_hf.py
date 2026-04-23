"""Upload the Chroma vector database to a Hugging Face dataset repository."""

import argparse

from dotenv import load_dotenv

try:
    from data.scraping_scripts.hf_auth import HuggingFaceAuthError, validate_hf_access
except ModuleNotFoundError:
    from hf_auth import HuggingFaceAuthError, validate_hf_access

load_dotenv()

DEFAULT_REPO_ID = "towardsai-tutors/ai-tutor-vector-db"


def upload_vector_db(repo_id: str = DEFAULT_REPO_ID) -> None:
    try:
        api = validate_hf_access(repo_id=repo_id)
    except HuggingFaceAuthError as exc:
        print(exc)
        raise SystemExit(1) from exc

    api.upload_folder(
        folder_path="data",
        repo_id=repo_id,
        repo_type="dataset",
        # multi_commits=True,
        # multi_commits_verbose=True,
        delete_patterns=["*"],
        allow_patterns=[
            "chroma-db-all_sources/**",
            "all_sources_contextual_nodes.pkl",
        ],
        ignore_patterns=["*.jsonl", "*.py", "*.txt", "*.ipynb", "*.md", "*.pyc", "*.mdx"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the Chroma vector database to Hugging Face."
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo. Default: {DEFAULT_REPO_ID}",
    )
    args = parser.parse_args()
    upload_vector_db(args.repo)


if __name__ == "__main__":
    main()
