from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .chat_types import SourceMatch
from .setup import COURSE_SOURCE_KEYS, KB_DIR, SOURCE_KEY_TO_LABEL

KB_DOC_SCHEME_RE = re.compile(r"^kb://doc/(?P<doc_id>[^)\]\s]+)$")
RAW_PATH_RE = re.compile(r"(?:data/kb/)?raw/[^\s)\]>,:]+?\.(?:mdx|md)")


@dataclass(frozen=True, slots=True)
class KbManifestEntry:
    doc_id: str
    title: str
    url: str
    source: str
    source_group: str
    path: str


def _normalize_path(value: str) -> str:
    value = value.strip().strip("<>`'\"")
    if value.startswith("./"):
        value = value[2:]
    if value.startswith(f"{KB_DIR}/"):
        return value
    if value.startswith("raw/"):
        return f"{KB_DIR}/{value}"
    return value


@lru_cache(maxsize=4)
def load_manifest_entries(kb_dir: str = KB_DIR) -> tuple[KbManifestEntry, ...]:
    manifest_path = Path(kb_dir) / "generated" / "corpus_manifest.jsonl"
    if not manifest_path.exists():
        return ()

    entries: list[KbManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            entries.append(
                KbManifestEntry(
                    doc_id=str(row.get("doc_id") or ""),
                    title=str(row.get("title") or ""),
                    url=str(row.get("url") or ""),
                    source=str(row.get("source") or ""),
                    source_group=str(row.get("source_group") or ""),
                    path=str(row.get("path") or ""),
                )
            )
    return tuple(entries)


def manifest_indexes(
    kb_dir: str = KB_DIR,
) -> tuple[dict[str, KbManifestEntry], dict[str, KbManifestEntry], dict[str, KbManifestEntry], dict[str, KbManifestEntry]]:
    by_doc_id: dict[str, KbManifestEntry] = {}
    by_url: dict[str, KbManifestEntry] = {}
    by_path: dict[str, KbManifestEntry] = {}
    by_title: dict[str, KbManifestEntry] = {}
    for entry in load_manifest_entries(kb_dir):
        if entry.doc_id:
            by_doc_id[entry.doc_id] = entry
        if entry.url:
            by_url[entry.url] = entry
        if entry.path:
            by_path[_normalize_path(entry.path)] = entry
            if entry.path.startswith(f"{KB_DIR}/"):
                by_path[entry.path[len(KB_DIR) + 1 :]] = entry
        if entry.title:
            by_title[entry.title.strip().lower()] = entry
    return by_doc_id, by_url, by_path, by_title


def source_match_from_manifest(entry: KbManifestEntry, *, score: float = 1.0) -> SourceMatch:
    return SourceMatch(
        doc_id=entry.doc_id,
        title=entry.title,
        url=entry.url or entry.path,
        source_key=entry.source,
        source_label=SOURCE_KEY_TO_LABEL.get(entry.source, entry.source),
        score=score,
        group="courses" if entry.source in COURSE_SOURCE_KEYS else entry.source_group or "docs",
    )


def resolve_manifest_reference(
    reference: str,
    *,
    label: str = "",
    kb_dir: str = KB_DIR,
) -> SourceMatch | None:
    value = reference.strip()
    by_doc_id, by_url, by_path, by_title = manifest_indexes(kb_dir)

    if match := KB_DOC_SCHEME_RE.match(value):
        entry = by_doc_id.get(match.group("doc_id"))
        return source_match_from_manifest(entry) if entry else None

    entry = by_url.get(value)
    if entry:
        return source_match_from_manifest(entry)

    entry = by_path.get(_normalize_path(value))
    if entry:
        return source_match_from_manifest(entry)

    if label:
        entry = by_title.get(label.strip().lower())
        if entry:
            return source_match_from_manifest(entry)
    return None


def extract_raw_paths(text: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for match in RAW_PATH_RE.finditer(text):
        value = match.group(0).rstrip(":.,;")
        key = _normalize_path(value)
        if key in seen:
            continue
        seen.add(key)
        paths.append(value)
    return paths


def parse_markdown_citations(text: str) -> list[tuple[str, str]]:
    citations: list[tuple[str, str]] = []
    link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    for match in link_re.finditer(text):
        citations.append((match.group(1).strip(), match.group(2).strip()))

    linked_refs = {ref for _label, ref in citations}
    bare_re = re.compile(r"(?<!\()(?P<ref>https?://[^\s<>()]+|kb://doc/[^\s<>()]+)")
    for match in bare_re.finditer(text):
        ref = match.group("ref").rstrip(".,;")
        if ref not in linked_refs:
            citations.append(("", ref))
    for path in extract_raw_paths(text):
        if path not in linked_refs:
            citations.append(("", path))
    return citations


def source_match_key(match: SourceMatch) -> str:
    return match.doc_id or match.url or match.title


def normalize_url(url: str) -> str:
    """Drop the fragment and trailing slash so one page dedupes to one card."""
    value = url.strip().split("#", 1)[0]
    return value.rstrip("/")


def citation_dedupe_key(match: SourceMatch) -> str:
    """Dedupe resolved citations by URL (one card per page); fall back to id/title."""
    return normalize_url(match.url) or match.doc_id or match.title


def source_match_payload(match: SourceMatch, *, message_id: str, call_id: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message_id": message_id,
        "doc_id": match.doc_id,
        "title": match.title,
        "url": match.url,
        "source_key": match.source_key,
        "source_label": match.source_label,
        "score": match.score,
        "group": match.group,
    }
    if call_id:
        payload["call_id"] = call_id
    return payload
