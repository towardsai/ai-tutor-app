"""
Fetch Markdown files from specified GitHub repositories.

This script fetches Markdown (.md), MDX (.mdx), and Jupyter Notebook (.ipynb) files
from specified GitHub repositories, particularly focusing on documentation sources
for various AI and machine learning libraries.

Key features:
1. Configurable for multiple documentation sources (e.g., Hugging Face Transformers, PEFT, TRL)
2. Command-line interface for specifying one or more sources to process
3. Automatic conversion of Jupyter Notebooks to Markdown
4. Rate limiting handling to comply with GitHub API restrictions
5. Retry mechanism for resilience against network issues

Usage:
    python github_to_markdown_ai_docs.py <source1> [<source2> ...]

Where <sourceN> is one of the predefined sources in SOURCE_CONFIGS (e.g., 'transformers', 'peft', 'trl').

Example:
    python github_to_markdown_ai_docs.py trl peft

This will download and process the documentation files for both TRL and PEFT libraries.

Note: 
- Ensure you have set the GITHUB_TOKEN variable with your GitHub Personal Access Token.
- The script creates a 'data' directory in the current working directory to store the downloaded files.
- Each source's files are stored in a subdirectory named '<repo>_md_files'.

"""

import argparse
import json
import os
import random
import shutil
import time
from typing import List

import nbformat
import requests
from dotenv import load_dotenv
from nbconvert import MarkdownExporter

load_dotenv()

# Configuration for different sources
SOURCE_CONFIGS = {
    "transformers": {
        "owner": "huggingface",
        "repo": "transformers",
        "path": "docs/source/en",
    },
    "peft": {
        "owner": "huggingface",
        "repo": "peft",
        "path": "docs/source",
    },
    "trl": {
        "owner": "huggingface",
        "repo": "trl",
        "path": "docs/source",
    },
    "llama_index": {
        "owner": "run-llama",
        "repo": "llama_index",
        "paths": [
            {
                "remote_path": "docs/src/content/docs/framework",
                "local_subdir": "src/content/docs/framework",
            },
            {
                "remote_path": "docs/examples",
                "local_subdir": "examples",
            },
        ],
    },
    "openai_cookbooks": {
        "owner": "openai",
        "repo": "openai-cookbook",
        "path": "examples",
    },
    "langchain": {
        "owner": "langchain-ai",
        "repo": "docs",
        "paths": [
            {"remote_path": "src/oss/concepts", "local_subdir": "concepts"},
            {"remote_path": "src/oss/langchain", "local_subdir": "langchain"},
            {
                "remote_path": "src/oss/python/integrations",
                "local_subdir": "python/integrations",
            },
            {
                "remote_path": "src/oss/python/migrate",
                "local_subdir": "python/migrate",
            },
            {
                "remote_path": "src/oss/python/releases",
                "local_subdir": "python/releases",
            },
            {
                "remote_path": "src/oss/security-policy.mdx",
                "local_subdir": "",
            },
            {
                "remote_path": "src/oss/release-policy.mdx",
                "local_subdir": "",
            },
            {
                "remote_path": "src/oss/versioning.mdx",
                "local_subdir": "",
            },
        ],
        "local_dir": "data/langchain_md_files",
    },
}

# GitHub Personal Access Token (replace with your own token)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Headers for authenticated requests
HEADERS = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

# Maximum number of retries
MAX_RETRIES = 5


class GitHubAPIError(RuntimeError):
    """Raised when the GitHub API returns an unexpected error."""


class GitHubAuthError(GitHubAPIError):
    """Raised when GitHub credentials are missing, invalid, or unauthorized."""


def _raise_for_github_error(response: requests.Response) -> None:
    if response.ok:
        return

    message = ""
    try:
        payload = response.json()
        message = str(payload.get("message", ""))
    except Exception:
        message = response.text.strip()

    if response.status_code == 401:
        raise GitHubAuthError(
            "GitHub authentication failed: GITHUB_TOKEN is invalid, expired, or revoked."
        )

    if response.status_code == 403:
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            reset_at = response.headers.get("X-RateLimit-Reset")
            raise GitHubAPIError(
                f"GitHub API rate limit exceeded. Reset timestamp: {reset_at or 'unknown'}."
            )
        raise GitHubAuthError(
            "GitHub authentication failed: access is forbidden. "
            "Check token scopes and any required SSO authorization."
        )

    raise GitHubAPIError(
        f"GitHub API request failed with status {response.status_code}: {message or response.reason}"
    )


def check_rate_limit():
    rate_limit_url = "https://api.github.com/rate_limit"
    response = requests.get(rate_limit_url, headers=HEADERS, timeout=30)
    _raise_for_github_error(response)
    data = response.json()
    resources = data.get("resources", {})
    core = resources.get("core")
    if core is None:
        raise GitHubAPIError(
            f"Unexpected rate limit response from GitHub: {json.dumps(data)[:300]}"
        )
    remaining = core["remaining"]
    reset_time = core["reset"]

    if remaining < 10:  # Adjust this threshold as needed
        wait_time = reset_time - time.time()
        print(f"Rate limit nearly exceeded. Waiting for {wait_time:.2f} seconds.")
        time.sleep(wait_time + 1)  # Add 1 second buffer


def get_files_in_directory(api_url: str, retries: int = 0):
    try:
        check_rate_limit()
        response = requests.get(api_url, headers=HEADERS, timeout=30)
        _raise_for_github_error(response)
        return response.json()
    except GitHubAPIError:
        raise
    except requests.exceptions.RequestException as e:
        if retries < MAX_RETRIES:
            wait_time = (2**retries) + random.random()
            print(
                f"Error fetching directory contents: {e}. Retrying in {wait_time:.2f} seconds..."
            )
            time.sleep(wait_time)
            return get_files_in_directory(api_url, retries + 1)
        else:
            print(
                f"Failed to fetch directory contents after {MAX_RETRIES} retries: {e}"
            )
            return []


def download_file(file_url: str, file_path: str, retries: int = 0):
    try:
        check_rate_limit()
        response = requests.get(file_url, headers=HEADERS, timeout=60)
        _raise_for_github_error(response)
        with open(file_path, "wb") as file:
            file.write(response.content)
    except GitHubAPIError:
        raise
    except requests.exceptions.RequestException as e:
        if retries < MAX_RETRIES:
            wait_time = (2**retries) + random.random()
            print(
                f"Error downloading file: {e}. Retrying in {wait_time:.2f} seconds..."
            )
            time.sleep(wait_time)
            download_file(file_url, file_path, retries + 1)
        else:
            print(f"Failed to download file after {MAX_RETRIES} retries: {e}")

    # def convert_ipynb_to_md(ipynb_path: str, md_path: str):
    #     with open(ipynb_path, "r", encoding="utf-8") as f:
    #         notebook = nbformat.read(f, as_version=4)

    #     exporter = MarkdownExporter()
    #     markdown, _ = exporter.from_notebook_node(notebook)

    #     with open(md_path, "w", encoding="utf-8") as f:
    #         f.write(markdown)


def convert_ipynb_to_md(ipynb_path: str, md_path: str):
    try:
        with open(ipynb_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        exporter = MarkdownExporter()
        markdown, _ = exporter.from_notebook_node(notebook)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
    except (json.JSONDecodeError, nbformat.reader.NotJSONError) as e:
        print(f"Error converting notebook {ipynb_path}: {str(e)}")
        print("Skipping this file and continuing with others...")
    except Exception as e:
        print(f"Unexpected error converting notebook {ipynb_path}: {str(e)}")
        print("Skipping this file and continuing with others...")


def fetch_files(api_url: str, local_dir: str):
    files = get_files_in_directory(api_url)
    if isinstance(files, dict):
        files = [files]
    for file in files:
        if file["type"] == "file" and file["name"].endswith((".md", ".mdx", ".ipynb")):
            file_url = file["download_url"]
            file_name = file["name"]
            file_path = os.path.join(local_dir, file_name)
            print(f"Downloading {file_name}...")
            download_file(file_url, file_path)

            if file_name.endswith(".ipynb"):
                md_file_name = file_name.replace(".ipynb", ".md")
                md_file_path = os.path.join(local_dir, md_file_name)
                print(f"Converting {file_name} to markdown...")
                convert_ipynb_to_md(file_path, md_file_path)
                os.remove(file_path)  # Remove the .ipynb file after conversion
        elif file["type"] == "dir":
            subdir = os.path.join(local_dir, file["name"])
            os.makedirs(subdir, exist_ok=True)
            fetch_files(file["url"], subdir)


def process_source(source: str):
    if source not in SOURCE_CONFIGS:
        print(
            f"Error: Unknown source '{source}'. Available sources: {', '.join(SOURCE_CONFIGS.keys())}"
        )
        return

    config = SOURCE_CONFIGS[source]
    local_dir = config.get("local_dir", f"data/{config['repo']}_md_files")
    shutil.rmtree(local_dir, ignore_errors=True)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Processing source: {source}")

    if "paths" in config:
        for path_config in config["paths"]:
            api_url = (
                f"https://api.github.com/repos/{config['owner']}/{config['repo']}/contents/"
                f"{path_config['remote_path']}"
            )
            target_dir = os.path.join(local_dir, path_config.get("local_subdir", ""))
            os.makedirs(target_dir, exist_ok=True)
            fetch_files(api_url, target_dir)
    else:
        api_url = (
            f"https://api.github.com/repos/{config['owner']}/{config['repo']}/contents/{config['path']}"
        )
        fetch_files(api_url, local_dir)

    print(f"Finished processing {source}")


def main(sources: List[str]):
    try:
        for source in sources:
            process_source(source)
        print("All specified sources have been processed.")
    except GitHubAuthError as exc:
        print(f"GitHub auth error: {exc}")
        raise SystemExit(1) from exc
    except GitHubAPIError as exc:
        print(f"GitHub API error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Markdown files from specified GitHub repositories."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        choices=SOURCE_CONFIGS.keys(),
        help="Specify one or more sources to process",
    )
    args = parser.parse_args()

    main(args.sources)
