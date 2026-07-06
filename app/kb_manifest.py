from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chat_types import SourceMatch
from .config import COURSE_SOURCE_KEYS, KB_DIR, SOURCE_KEY_TO_LABEL

logger = logging.getLogger(__name__)

KB_DOC_SCHEME_RE = re.compile(r"^kb://doc/(?P<doc_id>[^)\]\s]+)$")
RAW_PATH_RE = re.compile(r"(?:data/kb/)?raw/[^\s)\]>,:]+?\.(?:mdx|md)")

# Strip fenced and inline code before harvesting citations: a URL or path shown
# as a code example (e.g. ``git clone https://…`` in a fence, or an inline span
# explaining "the part starting with `https://`") is documentation, not a
# citation. Mirrors the frontend's stripCodeSegments (frontend/lib/chat-ui.ts)
# so the server's chip set matches the references the client is willing to
# number; otherwise such URLs surface as orphan, sometimes malformed, chips.
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)")
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def _strip_code_segments(text: str) -> str:
    text = _FENCED_CODE_RE.sub(" ", text)
    return _INLINE_CODE_RE.sub(" ", text)


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


def kb_root_path(value: str) -> str:
    """KB-root-relative form of a manifest path ("raw/docs/...") — the shape
    the model is instructed to cite and the client matches against."""
    value = value.strip()
    if value.startswith("./"):
        value = value[2:]
    if value.startswith(f"{KB_DIR}/"):
        value = value[len(KB_DIR) + 1 :]
    return value


# Cache parsed manifests per kb_dir, but only once the file actually exists, so
# a lookup during the first-start download window does not pin an empty result.
_MANIFEST_CACHE: dict[str, tuple[KbManifestEntry, ...]] = {}


def load_manifest_entries(kb_dir: str = KB_DIR) -> tuple[KbManifestEntry, ...]:
    cached = _MANIFEST_CACHE.get(kb_dir)
    if cached is not None:
        return cached

    manifest_path = Path(kb_dir) / "generated" / "corpus_manifest.jsonl"
    if not manifest_path.exists():
        # Do not cache the missing-file case: the KB bundle may still be
        # downloading on first start, so retry on the next call instead of
        # pinning an empty manifest (and thus never resolving citations) for
        # the process lifetime.
        return ()

    entries: list[KbManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed manifest line %d in %s: %s",
                    line_number,
                    manifest_path,
                    exc,
                )
                continue
            if not isinstance(row, dict):
                logger.warning(
                    "Skipping non-object manifest line %d in %s",
                    line_number,
                    manifest_path,
                )
                continue
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
    result = tuple(entries)
    _MANIFEST_CACHE[kb_dir] = result
    return result


def available_source_keys(kb_dir: str = KB_DIR) -> frozenset[str] | None:
    """Source keys actually present in the downloaded KB bundle.

    Returns ``None`` when the manifest is not available yet (the bundle may
    still be downloading on first start), signalling callers to not filter.
    Otherwise it is the distinct set of sources shipped in the bundle. On the
    public docs-only bundle this excludes the course sources, so the UI source
    picker can hide sources that have no data to query.
    """
    entries = load_manifest_entries(kb_dir)
    if not entries:
        return None
    return frozenset(entry.source for entry in entries if entry.source)


_ManifestIndexes = tuple[
    dict[str, KbManifestEntry],
    dict[str, KbManifestEntry],
    dict[str, KbManifestEntry],
    dict[str, KbManifestEntry],
]

# Cache built indexes per kb_dir, next to the entries tuple they were derived
# from. Like _MANIFEST_CACHE, nothing is cached while the manifest file is
# absent (load_manifest_entries returned an uncached empty tuple), so a lookup
# during the first-start download window does not pin empty indexes. The
# identity check against the cached entries keeps the indexes valid for the
# process lifetime and rebuilds them whenever _MANIFEST_CACHE is repopulated
# (e.g. popped in tests). resolve_manifest_reference runs once per citation or
# shell-printed path on the event loop, so repeated calls must be O(1) instead
# of re-scanning all manifest entries.
_INDEX_CACHE: dict[str, tuple[tuple[KbManifestEntry, ...], _ManifestIndexes]] = {}


def manifest_indexes(kb_dir: str = KB_DIR) -> _ManifestIndexes:
    entries = load_manifest_entries(kb_dir)
    cached = _INDEX_CACHE.get(kb_dir)
    if cached is not None and cached[0] is entries:
        return cached[1]

    by_doc_id: dict[str, KbManifestEntry] = {}
    by_url: dict[str, KbManifestEntry] = {}
    by_path: dict[str, KbManifestEntry] = {}
    by_title: dict[str, KbManifestEntry] = {}
    # Titles like "Introduction"/"Quickstart" recur across docs; a plain dict
    # would resolve a bare-label citation to whichever doc was ingested last and
    # surface a misleading source card. Track collisions and refuse to resolve
    # an ambiguous title to any single doc.
    ambiguous_titles: set[str] = set()
    for entry in entries:
        if entry.doc_id:
            by_doc_id[entry.doc_id] = entry
        if entry.url:
            by_url[entry.url] = entry
        if entry.path:
            by_path[_normalize_path(entry.path)] = entry
            if entry.path.startswith(f"{KB_DIR}/"):
                by_path[entry.path[len(KB_DIR) + 1 :]] = entry
        if entry.title:
            title_key = entry.title.strip().lower()
            if title_key in ambiguous_titles:
                continue
            existing = by_title.get(title_key)
            if existing is not None and existing.doc_id != entry.doc_id:
                ambiguous_titles.add(title_key)
                del by_title[title_key]
            else:
                by_title[title_key] = entry
    indexes = (by_doc_id, by_url, by_path, by_title)
    if entries:
        _INDEX_CACHE[kb_dir] = (entries, indexes)
    return indexes


def source_match_from_manifest(
    entry: KbManifestEntry, *, score: float = 1.0
) -> SourceMatch:
    return SourceMatch(
        doc_id=entry.doc_id,
        title=entry.title,
        url=entry.url or entry.path,
        source_key=entry.source,
        source_label=SOURCE_KEY_TO_LABEL.get(entry.source, entry.source),
        score=score,
        group="courses"
        if entry.source in COURSE_SOURCE_KEYS
        else entry.source_group or "docs",
        path=kb_root_path(entry.path),
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
    # Scan code-stripped text so URLs/paths that only appear inside code
    # examples are never harvested as citations (see _strip_code_segments).
    scannable = _strip_code_segments(text)
    citations: list[tuple[str, str]] = []
    # The destination must be whitespace-free: with [^)]+ a malformed link the
    # model closed with "]" instead of ")" matched across newlines up to the
    # next real citation's closing paren, swallowing that citation into one
    # garbage reference.
    link_re = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
    for match in link_re.finditer(scannable):
        citations.append((match.group(1).strip(), match.group(2).strip()))

    linked_refs = {ref for _label, ref in citations}
    # Backticks are excluded from the URL body: even past code stripping, a
    # stray unbalanced backtick must never become part of a captured URL (which
    # is how "https://`" leaked through before). No opening-paren guard: a URL
    # inside a well-formed link dedupes via linked_refs, and this pass is what
    # recovers the URL of a malformed link (trailing "]"/":" junk stripped) so
    # the citation still resolves to a source card.
    bare_re = re.compile(r"(?P<ref>https?://[^\s<>()`]+|kb://doc/[^\s<>()`]+)")
    for match in bare_re.finditer(scannable):
        ref = match.group("ref").rstrip(".,;:]")
        if ref and ref not in linked_refs:
            citations.append(("", ref))
    for path in extract_raw_paths(scannable):
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


def source_match_payload(
    match: SourceMatch, *, message_id: str, call_id: str = ""
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message_id": message_id,
        "doc_id": match.doc_id,
        "title": match.title,
        "url": match.url,
        "source_key": match.source_key,
        "source_label": match.source_label,
        "score": match.score,
        "group": match.group,
        "path": match.path,
    }
    if call_id:
        payload["call_id"] = call_id
    return payload
