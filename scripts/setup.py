import asyncio
import json
import logging
import os
import pickle

import chromadb
import logfire
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from .custom_retriever import CustomRetriever
from .utils import init_mongo_db

load_dotenv()

logfire.configure()

if not os.path.exists("data/chroma-db-all_sources"):
    # Download the vector database from the Hugging Face Hub if it doesn't exist locally
    # https://huggingface.co/datasets/towardsai-buster/ai-tutor-vector-db/tree/main
    logfire.warn(
        f"Vector database does not exist at 'data/chroma-db-all_sources', downloading from Hugging Face Hub"
    )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="towardsai-tutors/ai-tutor-vector-db",
        local_dir="data",
        repo_type="dataset",
    )
    logfire.info(f"Downloaded vector database to 'data/chroma-db-all_sources'")


def setup_database(db_collection, dict_file_name) -> CustomRetriever:
    db = chromadb.PersistentClient(path=f"data/{db_collection}")
    chroma_collection = db.get_or_create_collection(db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = CohereEmbedding(
        api_key=os.environ["COHERE_API_KEY"],
        model_name="embed-english-v3.0",
        input_type="search_query",
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        transformations=[SentenceSplitter(chunk_size=800, chunk_overlap=0)],
        show_progress=True,
        # use_async=True,
    )
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=15,
        embed_model=embed_model,
        # use_async=True,
    )
    with open(f"data/{db_collection}/{dict_file_name}", "rb") as f:
        document_dict = pickle.load(f)

    return CustomRetriever(vector_retriever, document_dict)


custom_retriever_all_sources: CustomRetriever = setup_database(
    "chroma-db-all_sources",
    "document_dict_all_sources.pkl",
)


CONCURRENCY_COUNT = int(os.getenv("CONCURRENCY_COUNT", 64))
MONGODB_URI = os.getenv("MONGODB_URI")

AVAILABLE_SOURCES_UI = [
    "Transformers Docs",
    "PEFT Docs",
    "TRL Docs",
    "LlamaIndex Docs",
    "LangChain Docs",
    "OpenAI Cookbooks",
    "Towards AI Blog",
    "8 Hour Primer",
    "Advanced LLM Developer",
    "Python Primer",
]

AVAILABLE_SOURCES = [
    "transformers",
    "peft",
    "trl",
    "llama_index",
    "langchain",
    "openai_cookbooks",
    "tai_blog",
    "8-hour_primer",
    "llm_developer",
    "python_primer",
]

mongo_db = (
    init_mongo_db(uri=MONGODB_URI, db_name="towardsai-buster")
    if MONGODB_URI
    else logfire.warn("No mongodb uri found, you will not be able to save data.")
)

__all__ = [
    "custom_retriever_all_sources",
    "mongo_db",
    "CONCURRENCY_COUNT",
    "AVAILABLE_SOURCES_UI",
    "AVAILABLE_SOURCES",
]
