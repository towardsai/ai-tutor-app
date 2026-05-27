"""Upload the Chroma vector database to a Hugging Face dataset repository."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationDelete, HfApi
from huggingface_hub.utils import filter_repo_objects

try:
    from data.scraping_scripts.hf_auth import HuggingFaceAuthError, validate_hf_access
except ModuleNotFoundError:
    from hf_auth import HuggingFaceAuthError, validate_hf_access

load_dotenv()

DEFAULT_REPO_ID = "towardsai-tutors/ai-tutor-vector-db"

FOLDER_PATH = "data"
ALLOW_PATTERNS = [
    "chroma-db-all_sources/**",
    "all_sources_contextual_nodes.pkl",
    "kb/**",
]
IGNORE_PATTERNS = ["*.py", "*.ipynb", "*.pyc", "__pycache__/**"]


def _prune_stale_remote_files(
    api: HfApi,
    repo_id: str,
    *,
    folder_path: str,
    allow_patterns: list[str],
    ignore_patterns: list[str],
) -> None:
    """Restore the auto-prune behavior we lost when switching from
    `upload_folder(..., delete_patterns=["*"])` to `upload_large_folder`.

    Compute the set of remote files that match `allow_patterns` (and don't
    match `ignore_patterns`) but are no longer present locally, and delete
    them in a single commit before the upload runs. We deliberately do this
    in its own commit so the subsequent `upload_large_folder` stays
    additive and resumable.
    """
    local_root = Path(folder_path)
    local_files_all = [
        p.relative_to(local_root).as_posix()
        for p in local_root.rglob("*")
        if p.is_file()
    ]
    local_kept = set(
        filter_repo_objects(
            local_files_all,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    )

    remote_files_all = api.list_repo_files(repo_id, repo_type="dataset")
    remote_kept = list(
        filter_repo_objects(
            remote_files_all,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
    )

    stale = sorted(f for f in remote_kept if f not in local_kept)
    if not stale:
        print("Prune step: no stale remote files to delete.")
        return

    print(f"Prune step: deleting {len(stale)} stale remote file(s):")
    for f in stale:
        print(f"  - {f}")

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
        commit_message=f"Prune {len(stale)} stale file(s) before upload",
    )


def upload_vector_db(repo_id: str = DEFAULT_REPO_ID) -> None:
    try:
        api = validate_hf_access(repo_id=repo_id)
    except HuggingFaceAuthError as exc:
        print(exc)
        raise SystemExit(1) from exc

    # Step 1: prune. Restore the auto-cleanup we lost when moving off
    # `upload_folder(..., delete_patterns=["*"])`. See helper docstring.
    _prune_stale_remote_files(
        api,
        repo_id,
        folder_path=FOLDER_PATH,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )

    # Step 2: upload. `upload_large_folder` is the recommended path once the
    # total payload crosses ~1 GB — it commits in many small batches instead
    # of one big atomic commit, which sidesteps the `commit/main` 500s
    # `upload_folder` tends to hit at this size. It is resumable: re-running
    # after a failure skips already-uploaded blobs.
    api.upload_large_folder(
        folder_path=FOLDER_PATH,
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
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
