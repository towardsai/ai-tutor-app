"""
Fetch Markdown documentation pages listed in llms.txt indexes.

This downloader is for official docs sites that expose AI-friendly Markdown
indexes, such as OpenAI's developer docs. It writes the linked Markdown pages
into a local directory so the existing process_md_files.py pipeline can create
JSONL, contextual nodes, and vector stores without needing an HTML crawler.

Usage:
    uv run -m data.scraping_scripts.llms_txt_to_markdown_docs openai_docs
    uv run -m data.scraping_scripts.llms_txt_to_markdown_docs openai_docs --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

try:
    from data.scraping_scripts.source_registry import (
        LLMS_TXT_SOURCE_KEYS,
        SOURCE_CONFIGS,
    )
except ModuleNotFoundError:
    from source_registry import LLMS_TXT_SOURCE_KEYS, SOURCE_CONFIGS

load_dotenv()

LINK_PATTERN = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)")
SOURCE_EXTENSIONS_FILENAME = "_source_extensions.json"
SOURCE_URLS_FILENAME = "_source_urls.json"
LLMS_TXT_MANIFEST_FILENAME = "_llms_txt_manifest.json"
REQUEST_TIMEOUT_SECONDS = 60


class LLMSTxtDownloadError(RuntimeError):
    """Raised when an llms.txt source cannot be downloaded safely."""


def fetch_text(url: str) -> str:
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def parse_markdown_links(index_text: str) -> list[str]:
    """Extract HTTP(S) Markdown links from an llms.txt index."""
    urls = []
    seen = set()
    for match in LINK_PATTERN.finditer(index_text):
        url = match.group(1).strip()
        if url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


def should_download_url(url: str, include_prefixes: Iterable[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.path.endswith((".md", ".mdx")):
        return False

    prefixes = tuple(include_prefixes)
    return not prefixes or any(url.startswith(prefix) for prefix in prefixes)


def local_relative_path(url: str) -> str:
    parsed = urlparse(url)
    relative_path = parsed.path.lstrip("/")
    if not relative_path:
        raise LLMSTxtDownloadError(f"Could not derive a local path from URL: {url}")
    normalized = os.path.normpath(relative_path).replace(os.sep, "/")
    if normalized == ".." or normalized.startswith("../"):
        raise LLMSTxtDownloadError(f"Refusing unsafe local path for URL: {url}")
    return normalized


def display_url(doc_url: str) -> str:
    for suffix in (".md", ".mdx"):
        if doc_url.endswith(suffix):
            return doc_url[: -len(suffix)]
    return doc_url


def write_text_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        if content and not content.endswith("\n"):
            f.write("\n")


def write_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def collect_doc_urls(config: dict) -> list[str]:
    index_urls = config.get("llms_txt_urls") or []
    if not index_urls:
        raise LLMSTxtDownloadError("Source config is missing llms_txt_urls")

    include_prefixes = config.get("llms_url_include_prefixes") or []
    docs: list[str] = []
    seen = set()

    for index_url in index_urls:
        print(f"Fetching index: {index_url}")
        index_text = fetch_text(str(index_url))
        for url in parse_markdown_links(index_text):
            if not should_download_url(url, include_prefixes):
                continue
            if url in seen:
                continue
            docs.append(url)
            seen.add(url)

    return docs


def process_source(source: str, *, dry_run: bool = False) -> None:
    if source not in LLMS_TXT_SOURCE_KEYS:
        available = ", ".join(LLMS_TXT_SOURCE_KEYS)
        raise LLMSTxtDownloadError(
            f"Unknown llms.txt source '{source}'. Available sources: {available}"
        )

    config = SOURCE_CONFIGS[source]
    doc_urls = collect_doc_urls(config)

    if not doc_urls:
        raise LLMSTxtDownloadError(f"No Markdown docs found for source: {source}")

    if dry_run:
        print(f"Found {len(doc_urls)} Markdown docs for {source}")
        return

    local_dir = str(config["input_directory"])
    shutil.rmtree(local_dir, ignore_errors=True)
    os.makedirs(local_dir, exist_ok=True)

    source_extensions: dict[str, str] = {}
    source_urls: dict[str, str] = {}

    print(f"Downloading {len(doc_urls)} Markdown docs for {source}")
    for doc_url in doc_urls:
        relative_path = local_relative_path(doc_url)
        local_path = os.path.join(local_dir, relative_path)
        print(f"Downloading {doc_url}")
        content = fetch_text(doc_url)
        write_text_file(local_path, content)
        source_extensions[relative_path] = os.path.splitext(relative_path)[1]
        source_urls[relative_path] = display_url(doc_url)

    write_json(os.path.join(local_dir, SOURCE_EXTENSIONS_FILENAME), source_extensions)
    write_json(os.path.join(local_dir, SOURCE_URLS_FILENAME), source_urls)
    write_json(
        os.path.join(local_dir, LLMS_TXT_MANIFEST_FILENAME),
        {
            "downloadedAt": datetime.now(timezone.utc).isoformat(),
            "indexUrls": config.get("llms_txt_urls") or [],
            "documentCount": len(doc_urls),
        },
    )
    print(f"Finished processing {source}")


def main(sources: list[str], *, dry_run: bool = False) -> None:
    for source in sources:
        process_source(source, dry_run=dry_run)
    print("All specified llms.txt sources have been processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="+",
        choices=LLMS_TXT_SOURCE_KEYS,
        help="Specify one or more llms.txt-backed sources to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch indexes and report document counts without writing files",
    )
    args = parser.parse_args()

    main(args.sources, dry_run=args.dry_run)
