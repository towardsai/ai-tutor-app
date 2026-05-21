from data.scraping_scripts.process_md_files import (
    content_sha256,
    extract_title,
    stable_doc_id,
)


def test_extract_title_prefers_frontmatter_title() -> None:
    content = """---
title: "Dappier integration"
description: "Integrate with the Dappier retriever using LangChain Python."
---

# DappierRetriever
"""

    assert extract_title(content) == "Dappier integration"


def test_extract_title_skips_frontmatter_delimiters() -> None:
    content = """---
description: A page without an explicit title
---

# Real page title
"""

    assert extract_title(content) == "Real page title"


def test_extract_title_uses_sidebar_title_when_title_missing() -> None:
    content = """---
sidebarTitle: Overview
description: A page without an explicit title
---

# Longer body heading
"""

    assert extract_title(content) == "Overview"


def test_extract_title_ignores_headings_inside_code_fences() -> None:
    content = """```python
# Not a page title
```

# Real page title
"""

    assert extract_title(content) == "Real page title"


def test_extract_title_keeps_first_line_fallback() -> None:
    content = """---
description: A page without heading syntax
---

First paragraph fallback
"""

    assert extract_title(content) == "First paragraph fallback"


def test_stable_doc_id_prefers_source_path() -> None:
    doc_id = stable_doc_id(
        source="transformers",
        source_path="main_classes/model.md",
        title="Model",
        url="https://example.com/ignored",
        content_hash=content_sha256("content"),
    )

    assert doc_id == "transformers:main-classes-model"


def test_stable_doc_id_uses_url_when_source_path_missing() -> None:
    doc_id = stable_doc_id(
        source="agentic_ai_engineering",
        source_path="",
        title="Lesson 18",
        url="https://academy.towardsai.net/courses/take/agent-engineering/multimedia/70289117-lesson-18-the-research-loop",
        content_hash=content_sha256("content"),
    )

    assert doc_id == (
        "agentic_ai_engineering:courses-take-agent-engineering-multimedia-"
        "70289117-lesson-18-the-research-loop"
    )


def test_content_sha256_is_stable_and_prefixed() -> None:
    assert content_sha256("same") == content_sha256("same")
    assert content_sha256("same").startswith("sha256:")
    assert content_sha256("same") != content_sha256("different")
