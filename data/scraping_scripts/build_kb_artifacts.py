"""Build agent-browsable KB artifacts from the normalized JSONL corpus."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from data.scraping_scripts.process_md_files import (
        clean_document_content,
        content_sha256,
        extract_title,
        generate_url,
        load_source_extension_manifest,
        load_source_url_manifest,
        should_include_file,
        split_frontmatter,
        slugify_identifier,
        stable_doc_id,
    )
    from data.scraping_scripts.source_registry import COURSE_SOURCE_KEYS, SOURCE_CONFIGS
except ModuleNotFoundError:
    from process_md_files import (
        clean_document_content,
        content_sha256,
        extract_title,
        generate_url,
        load_source_extension_manifest,
        load_source_url_manifest,
        should_include_file,
        split_frontmatter,
        slugify_identifier,
        stable_doc_id,
    )
    from source_registry import COURSE_SOURCE_KEYS, SOURCE_CONFIGS


DEFAULT_INPUT_FILE = Path("data/all_sources_data.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/kb")
RAW_DIR_NAME = "raw"
GENERATED_DIR_NAME = "generated"
MARKDOWN_EXTENSIONS = {".md", ".mdx"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"`([^`\n]{2,160})`")
SYMBOL_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b|"
    r"\b[a-z_]+_[a-z0-9_]+\b|"
    r"\b[A-Z][A-Za-z0-9_]{2,}\b"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusRecord:
    doc_id: str
    legacy_doc_id: str
    title: str
    source: str
    source_group: str
    source_path: str
    url: str
    tokens: int
    content_hash: str
    content: str


@dataclass(frozen=True)
class OriginalMarkdown:
    source_path: str
    path: Path
    url: str
    title: str
    content_hash: str
    content: str


@dataclass
class OriginalMarkdownIndex:
    by_source_path: dict[str, OriginalMarkdown]
    by_url: dict[str, list[OriginalMarkdown]]
    by_content_hash: dict[str, list[OriginalMarkdown]]
    by_title: dict[str, list[OriginalMarkdown]]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def looks_like_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except (TypeError, ValueError):
        return False
    return True


def normalize_source_path(row: dict[str, Any]) -> str:
    source_path = str(row.get("source_path") or "").replace("\\", "/").strip("/")
    path = Path(source_path)
    if path.is_absolute() or ".." in path.parts:
        return ""
    return source_path


def normalize_record(row: dict[str, Any]) -> CorpusRecord:
    content = str(row.get("content") or "")
    source = str(row.get("source") or "unknown")
    title = str(row.get("name") or row.get("title") or "Untitled")
    url = str(row.get("url") or "")
    source_path = normalize_source_path(row)
    if source in COURSE_SOURCE_KEYS:
        source_path = ""
    content_hash = str(row.get("content_hash") or content_sha256(content))
    raw_doc_id = str(row.get("doc_id") or "")
    if raw_doc_id and not looks_like_uuid(raw_doc_id) and ":" in raw_doc_id:
        doc_id = raw_doc_id
    else:
        doc_id = stable_doc_id(
            source=source,
            source_path=source_path,
            title=title,
            url=url,
            content_hash=content_hash,
        )
    try:
        tokens = int(row.get("tokens") or 0)
    except (TypeError, ValueError):
        tokens = 0
    return CorpusRecord(
        doc_id=doc_id,
        legacy_doc_id=raw_doc_id if raw_doc_id != doc_id else "",
        title=title,
        source=source,
        source_group="courses" if source in COURSE_SOURCE_KEYS else "docs",
        source_path=source_path,
        url=url,
        tokens=tokens,
        content_hash=content_hash,
        content=content,
    )


def quote_frontmatter(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def strip_existing_frontmatter(content: str) -> str:
    frontmatter, body = split_frontmatter(content)
    return body if frontmatter is not None else content


def markdown_with_frontmatter(
    record: CorpusRecord,
    *,
    generated_from: Path,
    content: str,
    source_path: str = "",
    original_path: str = "",
) -> str:
    frontmatter_source_path = source_path or record.source_path
    lines = [
        "---",
        f"doc_id: {quote_frontmatter(record.doc_id)}",
        f"legacy_doc_id: {quote_frontmatter(record.legacy_doc_id)}",
        f"source: {quote_frontmatter(record.source)}",
        f"source_group: {quote_frontmatter(record.source_group)}",
        f"title: {quote_frontmatter(record.title)}",
        f"url: {quote_frontmatter(record.url)}",
        f"source_path: {quote_frontmatter(frontmatter_source_path)}",
        f"tokens: {record.tokens}",
        f"content_hash: {quote_frontmatter(record.content_hash)}",
        f"generated_from: {quote_frontmatter(str(generated_from))}",
    ]
    if original_path:
        lines.append(f"original_path: {quote_frontmatter(original_path)}")
    lines.extend(["---", ""])
    body = strip_existing_frontmatter(content).strip()
    return "\n".join(lines) + body + "\n"


# Cap slug length so the generated filename stays well under the 255-byte
# per-component limit on macOS/most filesystems. Headroom is needed beyond the
# ".md" because `huggingface_hub.upload_large_folder` mirrors each file under
# data/.cache/huggingface/upload/ and appends ".lock"/".metadata"; an unbounded
# slug (e.g. a llama_index page slugified from a full colab-badge <a href> link)
# overflowed that limit and aborted the upload. Collisions from truncation are
# handled downstream by the content-hash + counter suffix in unique_output_path.
MAX_SLUG_LEN = 150


def record_slug(record: CorpusRecord) -> str:
    basis = record.source_path or record.title or record.doc_id
    slug = slugify_identifier(basis)
    if slug == "untitled":
        slug = record.content_hash.removeprefix("sha256:")[:12]
    if len(slug) > MAX_SLUG_LEN:
        slug = slug[:MAX_SLUG_LEN].rstrip("-") or slug[:MAX_SLUG_LEN]
    return slug


def safe_relative_path(value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe source path: {value}")
    return path


def unique_path(candidate: Path, used_paths: set[Path], content_hash: str) -> Path:
    if candidate not in used_paths:
        used_paths.add(candidate)
        return candidate

    suffix = content_hash.removeprefix("sha256:")[:8]
    stem = candidate.stem
    candidate = candidate.with_name(f"{stem}-{suffix}{candidate.suffix}")
    counter = 2
    while candidate in used_paths:
        candidate = candidate.with_name(f"{stem}-{suffix}-{counter}{candidate.suffix}")
        counter += 1
    used_paths.add(candidate)
    return candidate


def unique_output_path(
    record: CorpusRecord,
    used_paths: set[Path],
    raw_dir: Path,
) -> Path:
    slug = record_slug(record)
    base_dir = raw_dir / record.source_group / record.source
    candidate = base_dir / f"{slug}.md"
    if candidate not in used_paths:
        used_paths.add(candidate)
        return candidate

    suffix = record.content_hash.removeprefix("sha256:")[:8]
    candidate = base_dir / f"{slug}-{suffix}.md"
    counter = 2
    while candidate in used_paths:
        candidate = base_dir / f"{slug}-{suffix}-{counter}.md"
        counter += 1
    used_paths.add(candidate)
    return candidate


def add_index_value(
    index: dict[str, list[OriginalMarkdown]],
    key: str,
    item: OriginalMarkdown,
) -> None:
    if key:
        index.setdefault(key, []).append(item)


def build_original_markdown_index(source: str) -> OriginalMarkdownIndex | None:
    config = SOURCE_CONFIGS.get(source)
    if not config:
        return None

    directory = Path(str(config.get("input_directory") or ""))
    if not directory.exists() or not directory.is_dir():
        return None

    source_extension_manifest = load_source_extension_manifest(str(directory))
    source_url_manifest = load_source_url_manifest(str(directory))
    by_source_path: dict[str, OriginalMarkdown] = {}
    by_url: dict[str, list[OriginalMarkdown]] = {}
    by_content_hash: dict[str, list[OriginalMarkdown]] = {}
    by_title: dict[str, list[OriginalMarkdown]] = {}

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix not in MARKDOWN_EXTENSIONS:
            continue
        source_path = path.relative_to(directory).as_posix()
        if not should_include_file(source_path, config):
            continue
        raw_content = path.read_text(encoding="utf-8")
        content = clean_document_content(raw_content)
        item = OriginalMarkdown(
            source_path=source_path,
            path=path,
            url=generate_url(
                source_path,
                config,
                source_extension_manifest.get(source_path),
                source_url_manifest.get(source_path),
            ),
            title=extract_title(content) or path.name,
            content_hash=content_sha256(content),
            content=content,
        )
        by_source_path[source_path] = item
        add_index_value(by_url, item.url, item)
        add_index_value(by_content_hash, item.content_hash, item)
        add_index_value(by_title, item.title, item)

    return OriginalMarkdownIndex(
        by_source_path=by_source_path,
        by_url=by_url,
        by_content_hash=by_content_hash,
        by_title=by_title,
    )


def unique_match(items: list[OriginalMarkdown] | None) -> OriginalMarkdown | None:
    return items[0] if items and len(items) == 1 else None


def find_original_markdown(
    record: CorpusRecord,
    indexes: dict[str, OriginalMarkdownIndex | None],
) -> OriginalMarkdown | None:
    if record.source_group != "docs":
        return None
    if record.source not in indexes:
        indexes[record.source] = build_original_markdown_index(record.source)
    index = indexes.get(record.source)
    if index is None:
        return None

    if record.source_path and (match := index.by_source_path.get(record.source_path)):
        return match
    if match := unique_match(index.by_url.get(record.url)):
        return match
    if match := unique_match(index.by_content_hash.get(record.content_hash)):
        return match
    if match := unique_match(index.by_title.get(record.title)):
        return match
    return None


def manifest_row(
    record: CorpusRecord,
    *,
    output_path: Path,
    source_path: str = "",
    original_path: str = "",
) -> dict[str, Any]:
    return {
        "doc_id": record.doc_id,
        "legacy_doc_id": record.legacy_doc_id,
        "title": record.title,
        "source": record.source,
        "source_group": record.source_group,
        "source_path": source_path or record.source_path,
        "url": record.url,
        "tokens": record.tokens,
        "content_hash": record.content_hash,
        "path": output_path.as_posix(),
        "original_path": original_path,
    }


def write_raw_markdown(
    records: list[CorpusRecord],
    *,
    input_file: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    raw_dir = output_dir / RAW_DIR_NAME
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    used_paths: set[Path] = set()
    original_indexes: dict[str, OriginalMarkdownIndex | None] = {}
    fallback_counts: dict[str, int] = {}
    manifest: list[dict[str, Any]] = []
    for record in records:
        original = find_original_markdown(record, original_indexes)
        if original:
            output_path = (
                raw_dir
                / "docs"
                / record.source
                / safe_relative_path(original.source_path)
            )
            output_path = unique_path(output_path, used_paths, record.content_hash)
            source_path = original.source_path
            original_path = original.path.as_posix()
            content = original.content
        else:
            output_path = unique_output_path(record, used_paths, raw_dir)
            source_path = record.source_path
            original_path = ""
            content = record.content
            if record.source_group == "docs":
                fallback_counts[record.source] = (
                    fallback_counts.get(record.source, 0) + 1
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown = markdown_with_frontmatter(
            record,
            generated_from=input_file,
            content=content,
            source_path=source_path,
            original_path=original_path,
        )
        output_path.write_text(markdown, encoding="utf-8")
        manifest.append(
            manifest_row(
                record,
                output_path=output_path,
                source_path=source_path,
                original_path=original_path,
            )
        )
    for source, count in sorted(fallback_counts.items()):
        logger.warning(
            "Generated %s docs KB page(s) from JSONL because original markdown was not found.",
            count,
            extra={"source": source},
        )
    return manifest


def iter_markdown_headings(path: Path) -> Iterable[dict[str, Any]]:
    in_fence = False
    fence_char = ""
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            current = fence_match.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_char = current
            elif current == fence_char:
                in_fence = False
                fence_char = ""
            continue
        if in_fence:
            continue
        match = HEADING_RE.match(line)
        if match:
            yield {
                "line": line_number,
                "level": len(match.group(1)),
                "heading": match.group(2).strip(),
            }


def heading_path_for_line(headings: list[dict[str, Any]], line_number: int) -> str:
    stack: list[dict[str, Any]] = []
    for heading in headings:
        if int(heading["line"]) > line_number:
            break
        level = int(heading["level"])
        stack = [item for item in stack if int(item["level"]) < level]
        stack.append(heading)
    return " > ".join(str(item["heading"]) for item in stack)


def extract_symbols(text: str) -> set[str]:
    symbols: set[str] = set()
    for inline_match in INLINE_CODE_RE.finditer(text):
        for match in SYMBOL_RE.finditer(inline_match.group(1)):
            symbols.add(match.group(0))
    for match in SYMBOL_RE.finditer(text):
        symbols.add(match.group(0))
    return {
        symbol
        for symbol in symbols
        if len(symbol) >= 3 and not symbol.lower().startswith(("http", "www"))
    }


def write_generated_indexes(
    manifest: list[dict[str, Any]], generated_dir: Path
) -> None:
    if generated_dir.exists():
        shutil.rmtree(generated_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = generated_dir / "corpus_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    heading_rows: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, str]] = []
    for item in manifest:
        path = Path(str(item["path"]))
        text = path.read_text(encoding="utf-8")
        headings = list(iter_markdown_headings(path))
        for heading in headings:
            heading_rows.append(
                {
                    "doc_id": item["doc_id"],
                    "source": item["source"],
                    "title": item["title"],
                    "path": item["path"],
                    **heading,
                }
            )
        for symbol in sorted(extract_symbols(text)):
            first_line = 1
            for line_number, line in enumerate(text.splitlines(), 1):
                if symbol in line:
                    first_line = line_number
                    break
            symbol_rows.append(
                {
                    "symbol": symbol,
                    "source": str(item["source"]),
                    "title": str(item["title"]),
                    "path": str(item["path"]),
                    "heading": heading_path_for_line(headings, first_line),
                    "doc_id": str(item["doc_id"]),
                }
            )

    with (generated_dir / "headings.jsonl").open("w", encoding="utf-8") as handle:
        for row in heading_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    with (generated_dir / "symbols.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["symbol", "source", "title", "path", "heading", "doc_id"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(symbol_rows)


def build_kb_artifacts(input_file: Path, output_dir: Path) -> dict[str, int]:
    rows = load_jsonl(input_file)
    records = [
        normalize_record(row) for row in rows if str(row.get("content") or "").strip()
    ]
    manifest = write_raw_markdown(records, input_file=input_file, output_dir=output_dir)
    write_generated_indexes(manifest, output_dir / GENERATED_DIR_NAME)
    return {"documents": len(records), "manifest_rows": len(manifest)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local KB markdown artifacts.")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    summary = build_kb_artifacts(Path(args.input_file), Path(args.output_dir))
    print(
        "Built KB artifacts: "
        f"{summary['documents']} documents, {summary['manifest_rows']} manifest rows"
    )


if __name__ == "__main__":
    main()
