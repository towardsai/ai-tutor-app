import json
from pathlib import Path
from typing import Dict, List

import pytest

from data.scraping_scripts import process_md_files as process_md_files_module
from data.scraping_scripts.process_md_files import (
    combine_all_sources,
    content_sha256,
    extract_title,
    main,
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


AGGREGATE_PATH = Path("data/all_sources_data.jsonl")


def _row(source: str, doc_id: str) -> Dict:
    return {"doc_id": doc_id, "source": source, "content": f"content of {doc_id}"}


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_doc_ids(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["doc_id"] for line in lines}


@pytest.fixture
def fake_registry(tmp_path, monkeypatch) -> Dict[str, Dict]:
    """Run in a temp cwd with a two-source registry (alpha, beta)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    configs = {
        "alpha": {"output_file": "data/alpha_data.jsonl"},
        "beta": {"output_file": "data/beta_data.jsonl"},
    }
    monkeypatch.setattr(process_md_files_module, "SOURCE_CONFIGS", configs)
    return configs


def test_combine_single_source_keeps_other_sources_from_their_jsonl(
    fake_registry,
) -> None:
    _write_jsonl(Path("data/alpha_data.jsonl"), [_row("alpha", "alpha:new")])
    _write_jsonl(Path("data/beta_data.jsonl"), [_row("beta", "beta:doc")])
    _write_jsonl(
        AGGREGATE_PATH,
        [_row("alpha", "alpha:stale"), _row("beta", "beta:doc")],
    )

    combine_all_sources(["alpha"])

    assert _read_doc_ids(AGGREGATE_PATH) == {"alpha:new", "beta:doc"}


def test_combine_falls_back_to_aggregate_when_per_source_jsonl_missing(
    fake_registry,
) -> None:
    _write_jsonl(Path("data/alpha_data.jsonl"), [_row("alpha", "alpha:new")])
    # beta has no per-source JSONL on disk; its rows exist only in the aggregate.
    _write_jsonl(
        AGGREGATE_PATH,
        [_row("alpha", "alpha:stale"), _row("beta", "beta:doc")],
    )

    combine_all_sources(["alpha"])

    assert _read_doc_ids(AGGREGATE_PATH) == {"alpha:new", "beta:doc"}


def test_combine_drops_sources_retired_from_registry(fake_registry) -> None:
    _write_jsonl(Path("data/alpha_data.jsonl"), [_row("alpha", "alpha:new")])
    _write_jsonl(Path("data/beta_data.jsonl"), [_row("beta", "beta:doc")])
    _write_jsonl(
        AGGREGATE_PATH,
        [_row("beta", "beta:doc"), _row("retired", "retired:doc")],
    )

    combine_all_sources(["alpha"])

    assert _read_doc_ids(AGGREGATE_PATH) == {"alpha:new", "beta:doc"}


def test_main_single_source_run_refreshes_aggregate(fake_registry, monkeypatch) -> None:
    """Regression: a single-source CLI run must rebuild all_sources_data.jsonl."""
    _write_jsonl(Path("data/beta_data.jsonl"), [_row("beta", "beta:doc")])
    _write_jsonl(
        AGGREGATE_PATH,
        [_row("alpha", "alpha:stale"), _row("beta", "beta:doc")],
    )

    def fake_process_source(source: str) -> None:
        output_file = fake_registry[source]["output_file"]
        _write_jsonl(Path(output_file), [_row(source, f"{source}:new")])

    monkeypatch.setattr(process_md_files_module, "process_source", fake_process_source)

    main(["alpha"])

    assert _read_doc_ids(AGGREGATE_PATH) == {"alpha:new", "beta:doc"}
