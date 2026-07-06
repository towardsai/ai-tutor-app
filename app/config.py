import logging
import os
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from data.scraping_scripts.source_registry import (
    AVAILABLE_SOURCES,
    AVAILABLE_SOURCES_UI,
    COURSE_SOURCE_KEYS,
    DEFAULT_SELECTED_SOURCE_KEYS,
    DEFAULT_SELECTED_SOURCES_UI,
    SOURCE_DISPLAY_INFO,
    SOURCE_KEY_TO_LABEL,
    SOURCE_UI_TO_KEY,
)

from .agent_tracing import configure_langsmith_environment, langsmith_tracing_enabled

load_dotenv()
configure_langsmith_environment()

# Server logs go to stdout (captured by the HF Space logs); LangSmith handles agent traces.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# Drop benign per-query Gemini noise: schema-key warnings, and the AFC notice
# (filtered by message so other google-genai warnings still surface).
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").addFilter(
    lambda record: "AFC is disabled" not in record.getMessage()
)

logger = logging.getLogger(__name__)

if langsmith_tracing_enabled():
    logger.info(
        "LangSmith tracing enabled. project=%s",
        os.getenv("LANGSMITH_PROJECT", "default"),
    )

VECTOR_DB_DIR = "data/chroma-db-all_sources"
VECTOR_COLLECTION_NAME = "chroma-db-all_sources"
DOCUMENT_DICT_PATH = f"{VECTOR_DB_DIR}/document_dict_all_sources.pkl"
BM25_INDEX_PATH = f"{VECTOR_DB_DIR}/bm25_index_all_sources.pkl"
CHROMA_SQLITE_PATH = f"{VECTOR_DB_DIR}/chroma.sqlite3"
# The full bundle (all sources, incl. gated course content) lives in the
# private repo; cold-starting from it needs an HF_TOKEN with read access. When
# that token is missing or lacks access, we fall back to the public docs-only
# bundle (the 9 documentation sources, no courses) built by
# data/scraping_scripts/build_public_docs_bundle.py. Both repos share the same
# tree layout, so only the download source changes here.
VECTOR_DB_REPO_ID = "towardsai-tutors/ai-tutor-vector-db"
PUBLIC_VECTOR_DB_REPO_ID = "towardsai-tutors/ai-tutor-vector-db-public"
# KB paths honor AI_TUTOR_KB_DIR — the same env var app/kb_shell.py reads — so
# the browsing sandbox and citation resolution (kb_manifest) always see one
# tree. Note the cold-start HF bundle download only materializes data/kb, so a
# custom AI_TUTOR_KB_DIR must point at an already-populated KB tree. Path()
# round-trip keeps config's relative-string convention while normalizing
# trailing slashes, which the f"{KB_DIR}/..." prefix logic below and in
# kb_manifest relies on.
_KB_DIR_ENV = os.getenv("AI_TUTOR_KB_DIR", "").strip()
KB_DIR = str(Path(_KB_DIR_ENV)) if _KB_DIR_ENV else "data/kb"
KB_MANIFEST_PATH = f"{KB_DIR}/generated/corpus_manifest.jsonl"
KB_INDEX_PATH = f"{KB_DIR}/wiki/index.md"
KB_AGENTS_PATH = f"{KB_DIR}/AGENTS.md"
# In-git template, copied into data/kb/AGENTS.md by ensure_kb_agents_md().
KB_AGENTS_TEMPLATE_PATH = "data/scraping_scripts/kb_agents_template.md"
DEFAULT_MODEL_NAME = "google-genai:gemini-3.5-flash"

AVAILABLE_MODELS: tuple[dict[str, str], ...] = (
    {"id": "google-genai:gemini-3.5-flash", "label": "Gemini 3.5 Flash"},
    {"id": "anthropic:claude-haiku-4-5", "label": "Claude Haiku 4.5"},
)


_BUNDLE_LOCK = Lock()
_BUNDLE_READY = False
_KB_AGENTS_WRITE_LOCK = Lock()


def _bundle_complete() -> bool:
    required = (
        VECTOR_DB_DIR,
        CHROMA_SQLITE_PATH,
        DOCUMENT_DICT_PATH,
        BM25_INDEX_PATH,
        KB_MANIFEST_PATH,
        KB_INDEX_PATH,
    )
    return all(os.path.exists(path) for path in required)


def ensure_kb_agents_md() -> None:
    """Write data/kb/AGENTS.md from the in-git template.

    Atomic (temp file + os.replace, so a concurrent prompt build never reads
    a truncated file) and skipped when the content already matches.
    """
    template_path = Path(KB_AGENTS_TEMPLATE_PATH)
    if not template_path.exists():
        return
    content = template_path.read_text(encoding="utf-8")
    target = Path(KB_AGENTS_PATH)
    with _KB_AGENTS_WRITE_LOCK:
        try:
            if target.read_text(encoding="utf-8") == content:
                return
        except OSError:
            pass
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(target.name + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, target)


def _snapshot_bundle(repo_id: str, *, token: str | bool | None) -> None:
    """Download a vector-db/KB bundle snapshot into ``data/``.

    ``token=False`` means download anonymously: ``None`` would make
    huggingface_hub re-resolve and send the cached/env token, and a token the
    Hub rejects fails the request even against a public repo. Public-bundle
    downloads must always pass ``False``.

    Mutes httpx's per-file flood during the cold-start download only. The
    GraphRAG experiment index (~150 MB) lives in the private repo for
    reproducibility but prod does not use it; skip it on cold start. Pull it
    explicitly to run that eval. ``README.md`` is the public repo's dataset
    card, not runtime data. (The private bundle has neither a kb archive nor
    a README, and the public one has no graphrag/ tree, so unmatched patterns
    are no-ops.)
    """
    from huggingface_hub import snapshot_download

    httpx_logger = logging.getLogger("httpx")
    previous_level = httpx_logger.level
    httpx_logger.setLevel(logging.WARNING)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir="data",
            repo_type="dataset",
            token=token,
            ignore_patterns=["graphrag/**", "README.md"],
        )
    finally:
        httpx_logger.setLevel(previous_level)
    _extract_kb_archive()


def _extract_kb_archive(base_dir: str = "data") -> None:
    """Extract ``kb.tar.gz`` into ``data/kb`` and delete the archive.

    The public bundle ships the KB as one archive instead of ~3,000 files so
    an anonymous cold start is not throttled by HF's per-request rate limits
    (see build_public_docs_bundle.archive_kb). A no-op when no archive was
    downloaded (the private bundle ships the unpacked tree).
    """
    import tarfile

    archive = Path(base_dir) / "kb.tar.gz"
    if not archive.exists():
        return
    logger.info("Extracting %s", archive)
    with tarfile.open(archive, "r:gz") as tar:
        # filter="data" blocks absolute paths, traversal, and special files.
        tar.extractall(path=base_dir, filter="data")
    archive.unlink()


def _download_bundle() -> None:
    """Download the bundle, falling back to the public docs-only one.

    With a usable ``HF_TOKEN`` we pull the full private bundle (all sources,
    incl. gated course content). When the token is absent, or present but
    without access to the private repo, we fall back to the public docs-only
    bundle so the app can still cold-start (documentation sources only). Any
    non-access error (e.g. a network failure) propagates instead of silently
    degrading to the smaller bundle.
    """
    try:
        from huggingface_hub.errors import (
            GatedRepoError,
            RepositoryNotFoundError,
        )
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import (
            GatedRepoError,
            RepositoryNotFoundError,
        )

    # Resolve the token the way huggingface_hub itself does: the HF_TOKEN env
    # var first, then a cached `huggingface-cli login`. Keying off os.getenv
    # alone would skip a cached login that the old implicit-token download honored.
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:  # pragma: no cover - very old hub without get_token
        token = os.getenv("HF_TOKEN") or None

    if not token:
        logger.warning(
            "No Hugging Face token found; downloading the public docs-only "
            "bundle from %s (documentation sources only, no course content).",
            PUBLIC_VECTOR_DB_REPO_ID,
        )
        _snapshot_bundle(PUBLIC_VECTOR_DB_REPO_ID, token=False)
        return

    try:
        _snapshot_bundle(VECTOR_DB_REPO_ID, token=token)
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        logger.warning(
            "HF_TOKEN cannot access the private bundle %s (%s); falling back to "
            "the public docs-only bundle %s.",
            VECTOR_DB_REPO_ID,
            type(exc).__name__,
            PUBLIC_VECTOR_DB_REPO_ID,
        )
        _snapshot_bundle(PUBLIC_VECTOR_DB_REPO_ID, token=False)


def ensure_local_vector_db() -> None:
    """Make sure the vector-db/KB bundle and AGENTS.md exist locally.

    A flag check once the bundle has been verified, so per-tool-call use is
    effectively free; the verification and download run single-flight under
    a lock so parallel first-turn tool calls cannot race two downloads.
    """
    global _BUNDLE_READY
    if _BUNDLE_READY:
        return
    with _BUNDLE_LOCK:
        if _BUNDLE_READY:
            return
        if not _bundle_complete():
            logger.warning(
                "Vector database missing or incomplete locally, downloading "
                "from Hugging Face"
            )
            if KB_DIR != "data/kb":
                # The bundle extracts to fixed paths under data/; it cannot
                # populate a custom KB dir, which must be pre-populated.
                logger.warning(
                    "AI_TUTOR_KB_DIR=%s: the Hugging Face bundle only "
                    "materializes data/kb, so a custom KB dir must already "
                    "contain the KB tree.",
                    KB_DIR,
                )
            _download_bundle()
        ensure_kb_agents_md()
        _BUNDLE_READY = _bundle_complete() and os.path.exists(KB_AGENTS_PATH)


__all__ = [
    "AVAILABLE_MODELS",
    "AVAILABLE_SOURCES",
    "AVAILABLE_SOURCES_UI",
    "COURSE_SOURCE_KEYS",
    "DEFAULT_SELECTED_SOURCE_KEYS",
    "DEFAULT_SELECTED_SOURCES_UI",
    "DEFAULT_MODEL_NAME",
    "BM25_INDEX_PATH",
    "DOCUMENT_DICT_PATH",
    "KB_AGENTS_PATH",
    "KB_AGENTS_TEMPLATE_PATH",
    "KB_DIR",
    "KB_INDEX_PATH",
    "KB_MANIFEST_PATH",
    "SOURCE_DISPLAY_INFO",
    "SOURCE_KEY_TO_LABEL",
    "SOURCE_UI_TO_KEY",
    "VECTOR_COLLECTION_NAME",
    "VECTOR_DB_DIR",
    "ensure_kb_agents_md",
    "ensure_local_vector_db",
]
