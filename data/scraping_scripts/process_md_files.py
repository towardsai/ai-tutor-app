"""
Markdown Document Processor for Documentation Sources

This script processes Markdown (.md) and MDX (.mdx) files from various documentation sources
(such as Hugging Face Transformers, PEFT, TRL, LlamaIndex, and OpenAI Cookbook) and converts
them into a standardized JSONL format for further processing or indexing.

Key features:
1. Configurable for multiple documentation sources
2. Extracts titles, generates URLs, and counts tokens for each document
3. Supports inclusion/exclusion of specific directories and root files
4. Removes copyright headers from content
5. Generates a unique ID for each document
6. Determines if a whole document should be retrieved based on token count
7. Handles special cases like openai-cookbook repo by adding .ipynb extensions
8. Processes multiple sources in a single run

Usage:
    python process_md_files.py <source1> <source2> ...

Where <source1>, <source2>, etc. are one or more of the active sources in
source_registry.py
(e.g., 'transformers', 'llama_index', 'openai_cookbooks').

The script processes all Markdown files in the specified input directories (and their subdirectories),
applies the configured filters, and saves the results in JSONL files. Each line in the output
files represents a single document with metadata and content.

To add, modify, or retire sources, update data/scraping_scripts/source_registry.py.
"""

import argparse
import hashlib
import json
import logging
import os
import re
from typing import Dict, List
from urllib.parse import urlparse

import tiktoken

try:
    from data.scraping_scripts.source_registry import SOURCE_CONFIGS
except ModuleNotFoundError:
    from source_registry import SOURCE_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_frontmatter(content: str) -> tuple[str | None, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, content

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() in {"---", "..."}:
            frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            return frontmatter, body

    return None, content


def clean_frontmatter_scalar(value: str) -> str | None:
    value = value.strip()
    if not value or value in {"|", ">"}:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        quote = value[0]
        value = value[1:-1]
        if quote == '"':
            value = value.replace(r"\"", '"').replace(r"\\", "\\")
        else:
            value = value.replace("''", "'")

    value = value.strip()
    return value or None


def extract_frontmatter_title(frontmatter: str) -> str | None:
    for key in ("title", "sidebarTitle"):
        title_match = re.search(rf"(?m)^\s*{key}\s*:\s*(.+?)\s*$", frontmatter)
        if title_match:
            title = clean_frontmatter_scalar(title_match.group(1))
            if title:
                return title

    return None


def iter_markdown_lines_outside_code_fences(content: str):
    in_code_fence = False
    fence_char = None

    for line in content.splitlines():
        fence_match = re.match(r"^\s{0,3}(```+|~~~+)", line)
        if fence_match:
            current_fence_char = fence_match.group(1)[0]
            if not in_code_fence:
                in_code_fence = True
                fence_char = current_fence_char
            elif current_fence_char == fence_char:
                in_code_fence = False
                fence_char = None
            continue

        if not in_code_fence:
            yield line


def extract_title(content: str):
    frontmatter, body = split_frontmatter(content)
    if frontmatter:
        title = extract_frontmatter_title(frontmatter)
        if title:
            return title

    for line in iter_markdown_lines_outside_code_fences(body):
        title_match = re.match(r"^\s{0,3}#\s+(.+)$", line)
        if title_match:
            return title_match.group(1).strip()

    lines = body.split("\n")
    for line in lines:
        if line.strip():
            return line.strip()

    return None


def load_source_extension_manifest(directory: str) -> Dict[str, str]:
    manifest_path = os.path.join(directory, "_source_extensions.json")
    if not os.path.exists(manifest_path):
        return {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not load source extension manifest: %s", manifest_path)
        return {}

    return {
        str(path): str(extension)
        for path, extension in manifest.items()
        if isinstance(path, str) and isinstance(extension, str)
    }


def load_source_url_manifest(directory: str) -> Dict[str, str]:
    manifest_path = os.path.join(directory, "_source_urls.json")
    if not os.path.exists(manifest_path):
        return {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not load source URL manifest: %s", manifest_path)
        return {}

    return {
        str(path): str(url)
        for path, url in manifest.items()
        if isinstance(path, str) and isinstance(url, str)
    }


def generate_url(
    file_path: str,
    config: Dict,
    source_extension: str | None = None,
    source_url: str | None = None,
) -> str:
    """
    Return an empty string if base_url is empty;
    otherwise return the constructed URL as before.
    """
    if source_url:
        return source_url

    if not config["base_url"]:
        return ""

    source_name = config["source_name"]
    path_with_forward_slashes = file_path.replace("\\", "/")

    if config.get("preserve_file_extension_in_url"):
        if source_extension:
            path_without_extension = os.path.splitext(path_with_forward_slashes)[0]
            return config["base_url"] + path_without_extension + source_extension
        return config["base_url"] + path_with_forward_slashes

    path_without_extension = os.path.splitext(file_path)[0]

    if source_name == "llama_index":
        framework_prefix = "src/content/docs/framework/"
        if path_without_extension.startswith(framework_prefix):
            path_without_extension = path_without_extension[len(framework_prefix) :]
        if path_without_extension == "index":
            path_without_extension = ""
        elif path_without_extension.endswith("/index"):
            path_without_extension = path_without_extension[: -len("/index")]

    if source_name == "langchain" and path_without_extension.startswith("python/"):
        path_without_extension = path_without_extension[len("python/") :]

    path_with_forward_slashes = path_without_extension.replace("\\", "/")
    return config["base_url"] + path_with_forward_slashes + config["url_extension"]


def should_include_file(file_path: str, config: Dict) -> bool:
    if os.path.dirname(file_path) == "":
        if config["use_include_list"]:
            return os.path.basename(file_path) in config["included_root_files"]
        else:
            return os.path.basename(file_path) not in config["excluded_root_files"]

    if config["use_include_list"]:
        return any(file_path.startswith(dir) for dir in config["included_dirs"])
    else:
        return not any(file_path.startswith(dir) for dir in config["excluded_dirs"])


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def remove_copyright_header(content: str) -> str:
    header_pattern = re.compile(r"<!--Copyright.*?-->\s*", re.DOTALL)
    cleaned_content = header_pattern.sub("", content, count=1)
    return cleaned_content.strip()


def remove_inline_base64_images(content: str) -> str:
    content = re.sub(
        r"!\[[^\]]*\]\(\s*data:image/[^,\s)]+;base64,[A-Za-z0-9+/=]+(?:\s+\"[^\"]*\")?\s*\)",
        "[inline image omitted]",
        content,
    )
    content = re.sub(
        r"<img\b[^>]*\bsrc=[\"']data:image/[^,\s\"']+;base64,[A-Za-z0-9+/=]+[\"'][^>]*>",
        "[inline image omitted]",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(
        r"data:image/[^,\s)\"']+;base64,[A-Za-z0-9+/=]+",
        "[inline image omitted]",
        content,
    )
    return content


def clean_document_content(content: str) -> str:
    content = remove_copyright_header(content)
    content = remove_inline_base64_images(content)
    return content.strip()


def content_sha256(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def slugify_identifier(value: str) -> str:
    value = os.path.splitext(value.replace("\\", "/"))[0]
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "untitled"


def stable_doc_id(
    *,
    source: str,
    source_path: str,
    title: str | None,
    url: str,
    content_hash: str,
) -> str:
    if source_path:
        basis = source_path
    else:
        parsed_path = urlparse(url).path.strip("/") if url else ""
        basis = parsed_path or title or content_hash.removeprefix("sha256:")[:12]
    return f"{source}:{slugify_identifier(basis)}"


def process_md_files(directory: str, config: Dict) -> List[Dict]:
    jsonl_data = []
    source_extension_manifest = load_source_extension_manifest(directory)
    source_url_manifest = load_source_url_manifest(directory)

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md") or file.endswith(".mdx"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)

                if should_include_file(relative_path, config):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    cleaned_content = clean_document_content(content)
                    title = extract_title(cleaned_content)
                    token_count = num_tokens_from_string(cleaned_content, "cl100k_base")
                    source_path = relative_path.replace("\\", "/")
                    url = generate_url(
                        relative_path,
                        config,
                        source_extension_manifest.get(source_path),
                        source_url_manifest.get(source_path),
                    )
                    hash_value = content_sha256(cleaned_content)

                    # Skip very small or extremely large files
                    if token_count < 100 or token_count > 200_000:
                        logger.info(
                            f"Skipping {relative_path} due to token count {token_count}"
                        )
                        continue

                    json_object = {
                        "tokens": token_count,
                        "doc_id": stable_doc_id(
                            source=config["source_name"],
                            source_path=source_path,
                            title=title,
                            url=url,
                            content_hash=hash_value,
                        ),
                        "name": (title if title else file),
                        "url": url,
                        "retrieve_doc": (token_count <= 8000),
                        "source": config["source_name"],
                        "source_path": source_path,
                        "content_hash": hash_value,
                        "content": cleaned_content,
                    }

                    jsonl_data.append(json_object)

    return jsonl_data


def save_jsonl(data: List[Dict], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def combine_all_sources(sources: List[str]) -> None:
    """
    Combine JSONL files from multiple sources, preserving existing sources not being processed.

    For example, if sources = ['transformers'], this will:
    1. Load data from transformers_data.jsonl
    2. Load data from all other source JSONL files that exist (course files, etc.)
    3. Combine them all into all_sources_data.jsonl
    """
    all_data = []
    output_file = "data/all_sources_data.jsonl"

    # Track which sources we're processing
    processed_sources = set()

    # First, add data from sources we're explicitly processing
    for source in sources:
        if source not in SOURCE_CONFIGS:
            logger.error(f"Unknown source '{source}'. Skipping.")
            continue

        processed_sources.add(source)
        input_file = SOURCE_CONFIGS[source]["output_file"]
        logger.info(f"Processing updated source: {source} from {input_file}")

        try:
            source_data = []
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    source_data.append(json.loads(line))

            logger.info(f"Added {len(source_data)} documents from {source}")
            all_data.extend(source_data)
        except Exception as e:
            logger.error(f"Error loading {input_file}: {e}")

    # Now add data from all other sources not being processed
    for source_name, config in SOURCE_CONFIGS.items():
        # Skip sources we already processed
        if source_name in processed_sources:
            continue

        # Try to load the individual source file
        source_file = config["output_file"]
        if os.path.exists(source_file):
            logger.info(f"Preserving existing source: {source_name} from {source_file}")
            try:
                source_data = []
                with open(source_file, "r", encoding="utf-8") as f:
                    for line in f:
                        source_data.append(json.loads(line))

                logger.info(
                    f"Preserved {len(source_data)} documents from {source_name}"
                )
                all_data.extend(source_data)
            except Exception as e:
                logger.error(f"Error loading {source_file}: {e}")

    logger.info(f"Total documents combined: {len(all_data)}")
    save_jsonl(all_data, output_file)
    logger.info(f"Combined data saved to {output_file}")


def process_source(source: str) -> None:
    if source not in SOURCE_CONFIGS:
        logger.error(f"Unknown source '{source}'. Skipping.")
        return

    config = SOURCE_CONFIGS[source]
    logger.info(f"\n\nProcessing source: {source}")
    jsonl_data = process_md_files(config["input_directory"], config)
    save_jsonl(jsonl_data, config["output_file"])
    logger.info(
        f"Processed {len(jsonl_data)} files and saved to {config['output_file']}"
    )


def main(sources: List[str]) -> None:
    for source in sources:
        process_source(source)

    if len(sources) > 1:
        combine_all_sources(sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Markdown files from specified sources."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        choices=SOURCE_CONFIGS.keys(),
        help="Specify one or more sources to process",
    )
    args = parser.parse_args()

    main(args.sources)
