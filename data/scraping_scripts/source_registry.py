"""Central source registry for the AI Tutor knowledge base.

Sources listed here are active in the KB pipeline. To retire a source, run
``retire_source_workflow.py``; confirmed retirements remove sources from this
file automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

ALL_SOURCES_JSONL = "data/all_sources_data.jsonl"
CONTEXTUAL_NODES_PKL = "data/all_sources_contextual_nodes.pkl"


SOURCE_CONFIGS: dict[str, dict[str, Any]] = {
    "transformers": {
        "base_url": "https://huggingface.co/docs/transformers/",
        "input_directory": "data/transformers_md_files",
        "output_file": "data/transformers_data.jsonl",
        "source_name": "transformers",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": ["internal"],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "peft": {
        "base_url": "https://huggingface.co/docs/peft/",
        "input_directory": "data/peft_md_files",
        "output_file": "data/peft_data.jsonl",
        "source_name": "peft",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "trl": {
        "base_url": "https://huggingface.co/docs/trl/",
        "input_directory": "data/trl_md_files",
        "output_file": "data/trl_data.jsonl",
        "source_name": "trl",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "llama_index": {
        "base_url": "https://docs.llamaindex.ai/en/stable/",
        "input_directory": "data/llama_index_md_files",
        "output_file": "data/llama_index_data.jsonl",
        "source_name": "llama_index",
        "use_include_list": True,
        "included_dirs": [
            "src/content/docs/framework/index.md",
            "src/content/docs/framework/getting_started",
            "src/content/docs/framework/understanding",
            "src/content/docs/framework/use_cases",
            "src/content/docs/framework/module_guides",
            "src/content/docs/framework/optimizing",
            "src/content/docs/framework/community/faq",
            "src/content/docs/framework/community/integrations",
            "src/content/docs/framework/llama_cloud",
            "examples",
        ],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "langchain": {
        "base_url": "https://docs.langchain.com/oss/python/",
        "input_directory": "data/langchain_md_files",
        "output_file": "data/langchain_data.jsonl",
        "source_name": "langchain",
        "use_include_list": True,
        "included_dirs": [
            "concepts",
            "langchain",
            "python/integrations/chat/",
            "python/integrations/document_loaders/",
            "python/integrations/document_transformers/",
            "python/integrations/embeddings/",
            "python/integrations/retrievers/",
            "python/integrations/splitters/",
            "python/integrations/stores/",
            "python/integrations/tools/",
            "python/integrations/vectorstores/",
            "python/migrate",
            "python/releases",
        ],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [
            "security-policy.mdx",
            "release-policy.mdx",
            "versioning.mdx",
        ],
        "url_extension": "",
    },
    "langgraph": {
        "base_url": "https://docs.langchain.com/oss/python/langgraph/",
        "input_directory": "data/langgraph_md_files",
        "output_file": "data/langgraph_data.jsonl",
        "source_name": "langgraph",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "deep_agents": {
        "base_url": "https://docs.langchain.com/oss/python/deepagents/",
        "input_directory": "data/deep_agents_md_files",
        "output_file": "data/deep_agents_data.jsonl",
        "source_name": "deep_agents",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "openai_docs": {
        "base_url": "https://developers.openai.com/",
        "input_directory": "data/openai_docs_md_files",
        "output_file": "data/openai_docs_data.jsonl",
        "source_name": "openai_docs",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
        "llms_txt_urls": [
            "https://developers.openai.com/api/docs/llms.txt",
            "https://developers.openai.com/codex/llms.txt",
        ],
        "llms_url_include_prefixes": [
            "https://developers.openai.com/api/docs/",
            "https://developers.openai.com/codex/",
        ],
    },
    "claude_code_docs": {
        "base_url": "https://code.claude.com/docs/",
        "input_directory": "data/claude_code_docs_md_files",
        "output_file": "data/claude_code_docs_data.jsonl",
        "source_name": "claude_code_docs",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
        "llms_txt_urls": [
            "https://code.claude.com/docs/llms.txt",
        ],
        "llms_url_include_prefixes": [
            "https://code.claude.com/docs/en/",
        ],
    },
    "full_stack_ai_engineering": {
        "base_url": "",
        "input_directory": "data/full_stack_ai_engineering",
        "output_file": "data/full_stack_ai_engineering_data.jsonl",
        "source_name": "full_stack_ai_engineering",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "beginner_python_for_ai_engineering": {
        "base_url": "",
        "input_directory": "data/beginner_python_for_ai_engineering",
        "output_file": "data/beginner_python_for_ai_engineering_data.jsonl",
        "source_name": "beginner_python_for_ai_engineering",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "master_ai_for_work": {
        "base_url": "",
        "input_directory": "data/master_ai_for_work",
        "output_file": "data/master_ai_for_work_data.jsonl",
        "source_name": "master_ai_for_work",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
    "agentic_ai_engineering": {
        "base_url": "",
        "input_directory": "data/agentic_ai_engineering",
        "output_file": "data/agentic_ai_engineering_data.jsonl",
        "source_name": "agentic_ai_engineering",
        "use_include_list": False,
        "included_dirs": [],
        "excluded_dirs": [],
        "excluded_root_files": [],
        "included_root_files": [],
        "url_extension": "",
    },
}

DOC_SOURCE_KEYS = (
    "transformers",
    "peft",
    "trl",
    "llama_index",
    "langchain",
    "langgraph",
    "deep_agents",
    "openai_docs",
    "claude_code_docs",
)
GITHUB_SOURCE_KEYS = (
    "transformers",
    "peft",
    "trl",
    "llama_index",
    "langchain",
    "langgraph",
    "deep_agents",
)
LLMS_TXT_SOURCE_KEYS = (
    "openai_docs",
    "claude_code_docs",
)
COURSE_SOURCE_KEYS = frozenset(
    {
        "full_stack_ai_engineering",
        "beginner_python_for_ai_engineering",
        "master_ai_for_work",
        "agentic_ai_engineering",
    }
)

ACTIVE_SOURCE_KEYS = frozenset(SOURCE_CONFIGS.keys())
AVAILABLE_SOURCES = list(SOURCE_CONFIGS.keys())

SOURCE_KEY_TO_LABEL = {
    "transformers": "Transformers Docs",
    "peft": "PEFT Docs",
    "trl": "TRL Docs",
    "llama_index": "LlamaIndex Docs",
    "langchain": "LangChain Docs",
    "langgraph": "LangGraph Docs",
    "deep_agents": "Deep Agents Docs",
    "openai_docs": "OpenAI Docs",
    "claude_code_docs": "Claude Code Docs",
    "full_stack_ai_engineering": "Full Stack AI Engineering",
    "beginner_python_for_ai_engineering": "Beginner Python for AI Engineering",
    "master_ai_for_work": "Master AI For Work",
    "agentic_ai_engineering": "Agentic AI Engineering",
}

UI_SOURCE_KEYS = (
    "openai_docs",
    "claude_code_docs",
    "langgraph",
    "deep_agents",
    "langchain",
    "llama_index",
    "transformers",
    "peft",
    "trl",
    "agentic_ai_engineering",
    "master_ai_for_work",
    "full_stack_ai_engineering",
    "beginner_python_for_ai_engineering",
)
SOURCE_UI_TO_KEY = {SOURCE_KEY_TO_LABEL[key]: key for key in UI_SOURCE_KEYS}
AVAILABLE_SOURCES_UI = list(SOURCE_UI_TO_KEY.keys())

DEFAULT_SELECTED_SOURCE_KEYS = (
    "agentic_ai_engineering",
    "master_ai_for_work",
    "full_stack_ai_engineering",
    "beginner_python_for_ai_engineering",
    "openai_docs",
    "claude_code_docs",
    "langgraph",
    "deep_agents",
    "transformers",
    "peft",
    "trl",
    "llama_index",
    "langchain",
)
DEFAULT_SELECTED_SOURCES_UI = [
    SOURCE_KEY_TO_LABEL[key] for key in DEFAULT_SELECTED_SOURCE_KEYS
]


def aggregate_data_files() -> dict[str, str]:
    return {
        ALL_SOURCES_JSONL: Path(ALL_SOURCES_JSONL).name,
        CONTEXTUAL_NODES_PKL: Path(CONTEXTUAL_NODES_PKL).name,
    }


def source_data_files() -> dict[str, str]:
    return {
        str(config["output_file"]): Path(str(config["output_file"])).name
        for config in SOURCE_CONFIGS.values()
    }


def source_output_files(sources: Iterable[str]) -> set[str]:
    return {
        str(SOURCE_CONFIGS[source]["output_file"])
        for source in sources
        if source in SOURCE_CONFIGS
    }


def required_data_files(*, include_source_files: bool = True) -> dict[str, str]:
    files = aggregate_data_files()
    if include_source_files:
        files.update(source_data_files())
    return files


def upload_data_file_paths() -> list[str]:
    return list(required_data_files().keys())


def vector_store_source_configs() -> dict[str, dict[str, str]]:
    configs = {
        source: {
            "input_file": str(config["output_file"]),
            "db_name": f"chroma-db-{source}",
            "document_dict_file": f"document_dict_{source}.pkl",
            "bm25_index_file": f"bm25_index_{source}.pkl",
        }
        for source, config in SOURCE_CONFIGS.items()
    }
    configs["all_sources"] = {
        "input_file": ALL_SOURCES_JSONL,
        "db_name": "chroma-db-all_sources",
        "document_dict_file": "document_dict_all_sources.pkl",
        "bm25_index_file": "bm25_index_all_sources.pkl",
    }
    return configs
