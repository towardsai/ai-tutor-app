"""
Capture the latest release tag + commit SHA or docs timestamp for each source and
write them to `data/source_versions.json`, so the frontend can surface which
library version is represented in the knowledge base (and how fresh it is).

Usage:
    uv run -m data.scraping_scripts.capture_source_versions
    uv run -m data.scraping_scripts.capture_source_versions --sources langchain peft

This is called automatically from `update_docs_workflow.py` after the download
step, but can also be run standalone to refresh the JSON.

Output shape (per source):
    {
      "version": "v4.46.3" | null,      # latest release tag; null if no releases
      "sha": "abc1234" | null,           # short SHA of default branch HEAD
      "indexedAt": "2026-04-17"          # UTC date of this capture
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

try:
    from data.scraping_scripts.source_registry import DOC_SOURCE_KEYS, SOURCE_CONFIGS
except ModuleNotFoundError:
    from source_registry import DOC_SOURCE_KEYS, SOURCE_CONFIGS

load_dotenv()

OUTPUT_PATH = Path("data/source_versions.json")

# Source key -> GitHub repo used for version lookup. For langchain we query the
# framework repo (where releases track the library), not the docs repo.
VERSION_REPOS: dict[str, tuple[str, str]] = {
    "transformers": ("huggingface", "transformers"),
    "peft": ("huggingface", "peft"),
    "trl": ("huggingface", "trl"),
    "llama_index": ("run-llama", "llama_index"),
    "langchain": ("langchain-ai", "langchain"),
    "langgraph": ("langchain-ai", "langgraph"),
    "deep_agents": ("langchain-ai", "deepagents"),
}

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"


def _get(url: str) -> Optional[dict]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException as exc:
        print(f"  ! network error: {exc}", file=sys.stderr)
        return None
    if response.status_code == 404:
        return None
    if not response.ok:
        print(
            f"  ! GitHub API {response.status_code} for {url}: {response.text[:160]}",
            file=sys.stderr,
        )
        return None
    return response.json()


def fetch_latest_release(owner: str, repo: str) -> Optional[str]:
    data = _get(f"https://api.github.com/repos/{owner}/{repo}/releases/latest")
    if not data:
        return None
    tag = data.get("tag_name")
    return tag if isinstance(tag, str) and tag else None


def fetch_default_branch_sha(owner: str, repo: str) -> Optional[str]:
    repo_data = _get(f"https://api.github.com/repos/{owner}/{repo}")
    if not repo_data:
        return None
    default_branch = repo_data.get("default_branch") or "main"
    branch_data = _get(
        f"https://api.github.com/repos/{owner}/{repo}/branches/{default_branch}"
    )
    if not branch_data:
        return None
    sha = branch_data.get("commit", {}).get("sha")
    return sha[:7] if isinstance(sha, str) else None


def capture_github_source(source: str) -> dict:
    owner, repo = VERSION_REPOS[source]
    print(f"- {source}: querying {owner}/{repo}")
    return {
        "version": fetch_latest_release(owner, repo),
        "sha": fetch_default_branch_sha(owner, repo),
        "indexedAt": datetime.now(timezone.utc).date().isoformat(),
    }


def capture_llms_txt_source(source: str) -> dict:
    print(f"- {source}: llms.txt source, no library version")
    # llms.txt docs sites have no release version; freshness is conveyed by
    # indexedAt alone (the index's Last-Modified is ~always the download date).
    return {
        "version": None,
        "sha": None,
        "indexedAt": datetime.now(timezone.utc).date().isoformat(),
    }


def capture_for_source(source: str) -> dict | None:
    if source in VERSION_REPOS:
        return capture_github_source(source)
    if SOURCE_CONFIGS.get(source, {}).get("llms_txt_urls"):
        return capture_llms_txt_source(source)
    return None


def load_existing() -> dict:
    if not OUTPUT_PATH.exists():
        return {}
    try:
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def capture(sources: list[str]) -> dict:
    versions = load_existing()
    for source in sources:
        metadata = capture_for_source(source)
        if metadata is None:
            print(f"  ! skipping unknown source: {source}", file=sys.stderr)
            continue
        versions[source] = metadata

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {OUTPUT_PATH}")
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(DOC_SOURCE_KEYS),
        default=list(DOC_SOURCE_KEYS),
        help="Subset of sources to refresh (default: all).",
    )
    args = parser.parse_args()
    capture(args.sources)


if __name__ == "__main__":
    main()
