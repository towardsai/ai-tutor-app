import os

import logfire
from dotenv import load_dotenv

from .utils import init_mongo_db

load_dotenv()
try:
    logfire.configure()
except Exception:
    pass

VECTOR_DB_DIR = "data/chroma-db-all_sources"
VECTOR_COLLECTION_NAME = "chroma-db-all_sources"
DOCUMENT_DICT_PATH = f"{VECTOR_DB_DIR}/document_dict_all_sources.pkl"
DEFAULT_MODEL_NAME = "google-genai:gemini-3-flash-preview"

AVAILABLE_SOURCES_UI = [
    "Transformers Docs",
    "PEFT Docs",
    "TRL Docs",
    "LlamaIndex Docs",
    "LangChain Docs",
    "OpenAI Cookbooks",
    "8 Hour Primer",
    "Advanced LLM Developer",
    "Python Primer",
    "Master AI For Work",
    "Agentic AI Engineering",
]

DEFAULT_SELECTED_SOURCES_UI = [
    "Agentic AI Engineering",
    "Master AI For Work",
    "Advanced LLM Developer",
    "8 Hour Primer",
    "Python Primer",
    "Transformers Docs",
    "PEFT Docs",
    "TRL Docs",
    "LlamaIndex Docs",
    "LangChain Docs",
    "OpenAI Cookbooks",
]

AVAILABLE_SOURCES = [
    "transformers",
    "peft",
    "trl",
    "llama_index",
    "langchain",
    "openai_cookbooks",
    "8-hour_primer",
    "llm_developer",
    "python_primer",
    "master_ai_for_work",
    "agentic_ai_engineering",
]

SOURCE_UI_TO_KEY = {
    "Transformers Docs": "transformers",
    "PEFT Docs": "peft",
    "TRL Docs": "trl",
    "LlamaIndex Docs": "llama_index",
    "LangChain Docs": "langchain",
    "OpenAI Cookbooks": "openai_cookbooks",
    "8 Hour Primer": "8-hour_primer",
    "Advanced LLM Developer": "llm_developer",
    "Python Primer": "python_primer",
    "Master AI For Work": "master_ai_for_work",
    "Agentic AI Engineering": "agentic_ai_engineering",
}

COURSE_SOURCE_KEYS = frozenset(
    {
        "8-hour_primer",
        "llm_developer",
        "python_primer",
        "master_ai_for_work",
        "agentic_ai_engineering",
    }
)

SOURCE_KEY_TO_LABEL = {value: key for key, value in SOURCE_UI_TO_KEY.items()}
DEFAULT_SELECTED_SOURCE_KEYS = tuple(
    SOURCE_UI_TO_KEY[label] for label in DEFAULT_SELECTED_SOURCES_UI
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
    "AVAILABLE_SOURCES",
    "AVAILABLE_SOURCES_UI",
    "COURSE_SOURCE_KEYS",
    "DEFAULT_SELECTED_SOURCE_KEYS",
    "DEFAULT_SELECTED_SOURCES_UI",
    "CONCURRENCY_COUNT",
    "DEFAULT_MODEL_NAME",
    "DOCUMENT_DICT_PATH",
    "SOURCE_KEY_TO_LABEL",
    "SOURCE_UI_TO_KEY",
    "VECTOR_COLLECTION_NAME",
    "VECTOR_DB_DIR",
    "ensure_local_vector_db",
    "mongo_db",
]
