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
    SOURCE_KEY_TO_LABEL,
    SOURCE_UI_TO_KEY,
)

from .agent_tracing import configure_langsmith_environment, langsmith_tracing_enabled

load_dotenv(override=True)
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
KB_DIR = "data/kb"
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
            from huggingface_hub import snapshot_download

            # Mute httpx's per-file flood during the cold-start download only.
            httpx_logger = logging.getLogger("httpx")
            previous_level = httpx_logger.level
            httpx_logger.setLevel(logging.WARNING)
            try:
                snapshot_download(
                    repo_id="towardsai-tutors/ai-tutor-vector-db",
                    local_dir="data",
                    repo_type="dataset",
                )
            finally:
                httpx_logger.setLevel(previous_level)
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
    "SOURCE_KEY_TO_LABEL",
    "SOURCE_UI_TO_KEY",
    "VECTOR_COLLECTION_NAME",
    "VECTOR_DB_DIR",
    "ensure_kb_agents_md",
    "ensure_local_vector_db",
]
