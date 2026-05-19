import os

import logfire
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
from .utils import init_mongo_db

load_dotenv(override=True)
configure_langsmith_environment()
try:
    logfire.configure()
except Exception:
    pass

if langsmith_tracing_enabled():
    logfire.info(
        "LangSmith tracing enabled.",
        project=os.getenv("LANGSMITH_PROJECT", "default"),
    )

VECTOR_DB_DIR = "data/chroma-db-all_sources"
VECTOR_COLLECTION_NAME = "chroma-db-all_sources"
DOCUMENT_DICT_PATH = f"{VECTOR_DB_DIR}/document_dict_all_sources.pkl"
BM25_INDEX_PATH = f"{VECTOR_DB_DIR}/bm25_index_all_sources.pkl"
DEFAULT_MODEL_NAME = "google-genai:gemini-3.5-flash"

AVAILABLE_MODELS: tuple[dict[str, str], ...] = (
    {"id": "google-genai:gemini-3.5-flash", "label": "Gemini 3.5 Flash"},
    {"id": "anthropic:claude-haiku-4-5", "label": "Claude Haiku 4.5"},
)

CONCURRENCY_COUNT = int(os.getenv("CONCURRENCY_COUNT", 64))
MONGODB_URI = os.getenv("MONGODB_URI")


def ensure_local_vector_db() -> None:
    if os.path.exists(VECTOR_DB_DIR) and os.path.exists(DOCUMENT_DICT_PATH):
        return

    logfire.warn(
        "Vector database does not exist locally, downloading from Hugging Face"
    )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="towardsai-tutors/ai-tutor-vector-db",
        local_dir="data",
        repo_type="dataset",
    )


mongo_db = (
    init_mongo_db(uri=MONGODB_URI, db_name="towardsai-buster")
    if MONGODB_URI
    else logfire.warn("No mongodb uri found, you will not be able to save data.")
)

__all__ = [
    "AVAILABLE_MODELS",
    "AVAILABLE_SOURCES",
    "AVAILABLE_SOURCES_UI",
    "COURSE_SOURCE_KEYS",
    "DEFAULT_SELECTED_SOURCE_KEYS",
    "DEFAULT_SELECTED_SOURCES_UI",
    "CONCURRENCY_COUNT",
    "DEFAULT_MODEL_NAME",
    "BM25_INDEX_PATH",
    "DOCUMENT_DICT_PATH",
    "SOURCE_KEY_TO_LABEL",
    "SOURCE_UI_TO_KEY",
    "VECTOR_COLLECTION_NAME",
    "VECTOR_DB_DIR",
    "ensure_local_vector_db",
    "mongo_db",
]
