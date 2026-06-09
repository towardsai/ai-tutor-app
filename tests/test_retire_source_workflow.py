import ast

from data.scraping_scripts.retire_source_workflow import (
    remove_source_from_registry_text,
)


def test_remove_source_from_registry_text_removes_active_source_entries():
    registry = """SOURCE_CONFIGS = {
    "openai_cookbooks": {
        "output_file": "data/openai_cookbooks_data.jsonl",
        "nested": {"keep": "balanced"},
    },
    "langchain": {
        "output_file": "data/langchain_data.jsonl",
    },
}

DOC_SOURCE_KEYS = (
    "openai_cookbooks",
    "langchain",
)
GITHUB_SOURCE_KEYS = (
    "openai_cookbooks",
    "langchain",
)
SOURCE_KEY_TO_LABEL = {
    "openai_cookbooks": "OpenAI Cookbooks",
    "langchain": "LangChain Docs",
}
UI_SOURCE_KEYS = (
    "openai_cookbooks",
    "langchain",
)
"""

    updated, changed = remove_source_from_registry_text(
        registry,
        "openai_cookbooks",
    )

    assert changed is True
    assert "openai_cookbooks" not in updated
    assert '"langchain"' in updated
    ast.parse(updated)


def test_remove_source_from_registry_text_ignores_missing_source():
    registry = 'SOURCE_CONFIGS = {"langchain": {"output_file": "x"}}\n'

    updated, changed = remove_source_from_registry_text(registry, "missing")

    assert changed is False
    assert updated == registry
