"""Publish the private vector database and archived KB to Hugging Face."""

import argparse
import gzip
import os
import tarfile
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
KB_DIR_NAME = "kb"
KB_ARCHIVE_NAME = "kb.tar.gz"
ALLOW_PATTERNS = [
    "chroma-db-all_sources/**",
    "all_sources_contextual_nodes.pkl",
    KB_ARCHIVE_NAME,
]
# The upload contains only the archive, but the prune pass also owns the
# unpacked tree left by older publishes. Keeping these scopes separate lets us
# delete kb/** without re-uploading it from the local build artifacts.
PRUNE_ALLOW_PATTERNS = [*ALLOW_PATTERNS, f"{KB_DIR_NAME}/**"]
IGNORE_PATTERNS = ["*.py", "*.ipynb", "*.pyc", "__pycache__/**"]


def _normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Remove host-specific metadata so unchanged KBs produce one stable blob."""
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    if info.isdir():
        info.mode = 0o755
    elif info.isfile():
        info.mode = 0o644
    return info


def create_kb_archive(folder_path: str = FOLDER_PATH) -> Path:
    """Build ``kb.tar.gz`` atomically without modifying the local KB tree."""
    root = Path(folder_path)
    kb_dir = root / KB_DIR_NAME
    required = (
        kb_dir / "generated" / "corpus_manifest.jsonl",
        kb_dir / "wiki" / "index.md",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot publish an incomplete KB; missing: " + ", ".join(missing)
        )

    archive_path = root / KB_ARCHIVE_NAME
    temp_path = archive_path.with_name(f"{archive_path.name}.tmp")
    temp_path.unlink(missing_ok=True)
    print(f"Archiving {kb_dir} -> {archive_path} ...")
    try:
        with temp_path.open("wb") as raw:
            # Fix the gzip timestamp/name and tar ownership/timestamps so an
            # unchanged KB hashes identically across machines and publishes.
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as tar:
                    tar.add(
                        kb_dir,
                        arcname=KB_DIR_NAME,
                        filter=_normalized_tar_info,
                    )
        os.replace(temp_path, archive_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    print(
        f"KB archive ready: {archive_path} ({archive_path.stat().st_size / 1e6:.1f} MB)"
    )
    return archive_path


def _prune_stale_remote_files(
    api: HfApi,
    repo_id: str,
    *,
    folder_path: str,
    allow_patterns: list[str],
    prune_allow_patterns: list[str] | None,
    ignore_patterns: list[str],
) -> None:
    """Verify the upload, then prune files owned by this publisher.

    ``allow_patterns`` describes files expected from the current upload.
    ``prune_allow_patterns`` may be wider, which is how the archive migration
    deletes legacy remote ``kb/**`` files without treating the local unpacked
    tree as publishable. Refuse to delete anything unless every expected local
    upload is visible remotely.
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

    remote_files_all = set(api.list_repo_files(repo_id, repo_type="dataset"))
    missing_remote = sorted(local_kept - remote_files_all)
    if missing_remote:
        preview = ", ".join(missing_remote[:10])
        suffix = "" if len(missing_remote) <= 10 else ", ..."
        raise RuntimeError(
            "Upload verification failed; refusing to prune because "
            f"{len(missing_remote)} expected remote file(s) are missing: "
            f"{preview}{suffix}"
        )

    prune_allow = prune_allow_patterns or allow_patterns
    remote_kept = list(
        filter_repo_objects(
            remote_files_all,
            allow_patterns=prune_allow,
            ignore_patterns=ignore_patterns,
        )
    )

    stale = sorted(f for f in remote_kept if f not in local_kept)
    if not stale:
        print("Prune step: no stale remote files to delete.")
        return

    print(f"Prune step: deleting {len(stale)} stale remote file(s).")
    for f in stale[:20]:
        print(f"  - {f}")
    if len(stale) > 20:
        print(f"  ... and {len(stale) - 20} more")

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
        commit_message=f"Prune {len(stale)} stale file(s) before upload",
    )


def upload_bundle(
    repo_id: str = DEFAULT_REPO_ID,
    *,
    folder_path: str = FOLDER_PATH,
    allow_patterns: list[str] | None = None,
    prune_allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    create_public: bool = False,
) -> None:
    """Upload ``folder_path``, verify it, then prune stale remote files.

    The default arguments reproduce the production upload (the private
    all-sources bundle). ``build_public_docs_bundle`` reuses this with a
    different folder/repo/patterns and ``create_public=True`` to publish the
    docs-only bundle to a public dataset repo it may need to create first.
    """
    allow = allow_patterns if allow_patterns is not None else ALLOW_PATTERNS
    ignore = ignore_patterns if ignore_patterns is not None else IGNORE_PATTERNS

    if create_public:
        # The public repo may not exist yet, so we cannot validate access to it
        # up front (validate_hf_access requires the repo to exist). Create it as
        # a public dataset (idempotent), which also proves the token can write.
        api = HfApi(token=os.getenv("HF_TOKEN"))
        api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    else:
        try:
            api = validate_hf_access(repo_id=repo_id)
        except HuggingFaceAuthError as exc:
            print(exc)
            raise SystemExit(1) from exc

    # Step 1: upload additively. Stale files remain available until the
    # verification and explicit prune below, so a failed archive migration
    # cannot strand the repo without either KB representation.
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow,
        ignore_patterns=ignore,
    )

    # Step 2: verify the expected files reached the repo, then prune. Uploading
    # first is load-bearing for archive migrations: the old unpacked KB stays
    # usable if the archive upload fails.
    _prune_stale_remote_files(
        api,
        repo_id,
        folder_path=folder_path,
        allow_patterns=allow,
        prune_allow_patterns=prune_allow_patterns,
        ignore_patterns=ignore,
    )


def upload_vector_db(repo_id: str = DEFAULT_REPO_ID) -> None:
    archive_path = create_kb_archive(FOLDER_PATH)
    try:
        upload_bundle(
            repo_id,
            allow_patterns=ALLOW_PATTERNS,
            prune_allow_patterns=PRUNE_ALLOW_PATTERNS,
        )
    finally:
        # The archive is a transient publishing artifact. The canonical local
        # KB stays unpacked for development and maintenance workflows.
        archive_path.unlink(missing_ok=True)


GRAPHRAG_LOCAL_DIR = "data/graphrag/output"
GRAPHRAG_REPO_PATH = "graphrag/output"


def upload_graphrag_index(repo_id: str = DEFAULT_REPO_ID) -> None:
    """Upload the local GraphRAG index to the same dataset repo (no prune).

    Lets anyone with read access to the (private) repo pull the prebuilt graph
    and re-run the GraphRAG-vs-RAG eval without rebuilding it (~$45 of Gemini
    indexing). Deliberately separate from `upload_vector_db` and **prune-free**:
    it must never delete the production bundle. The runtime cold-start download
    (`config.ensure_local_vector_db`) ignores `graphrag/**`, so prod Spaces do
    not pull this ~150 MB experiment artifact; pull it explicitly to run the eval
    (see evals/graphrag.md).
    """
    try:
        api = validate_hf_access(repo_id=repo_id)
    except HuggingFaceAuthError as exc:
        print(exc)
        raise SystemExit(1) from exc
    if not Path(GRAPHRAG_LOCAL_DIR).is_dir():
        raise SystemExit(f"No GraphRAG index at {GRAPHRAG_LOCAL_DIR}; build it first.")
    api.upload_folder(
        folder_path=GRAPHRAG_LOCAL_DIR,
        path_in_repo=GRAPHRAG_REPO_PATH,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {GRAPHRAG_LOCAL_DIR} -> {repo_id}:{GRAPHRAG_REPO_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the Chroma vector database to Hugging Face."
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--graphrag",
        action="store_true",
        help="Upload the GraphRAG index (data/graphrag/output) instead of the "
        "vector-db bundle. Prune-free; never touches the production bundle.",
    )
    args = parser.parse_args()
    if args.graphrag:
        upload_graphrag_index(args.repo)
    else:
        upload_vector_db(args.repo)


if __name__ == "__main__":
    main()
