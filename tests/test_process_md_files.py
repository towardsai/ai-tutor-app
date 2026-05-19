from data.scraping_scripts.process_md_files import extract_title


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
