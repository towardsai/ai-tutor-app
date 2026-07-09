"""Build (and optionally upload) the PUBLIC, docs-only vector-db/KB bundle.

The production bundle lives in the **private** dataset
``towardsai-tutors/ai-tutor-vector-db`` and contains every source, including
the course content (real student-facing material we keep gated). Anyone
without an ``HF_TOKEN`` that can read that repo cannot cold-start the app.

This script derives a **public** docs-only bundle from the already-built
production artifacts and pushes it to ``towardsai-tutors/ai-tutor-vector-db-public``
(public). The runtime (``app.config.ensure_local_vector_db``) falls back to it
when no usable ``HF_TOKEN`` is present. It contains only the documentation /
reference sources (``DOC_SOURCE_KEYS``); everything else is stripped.

Filtering is an **allowlist** (keep ``source in DOC_SOURCE_KEYS``), never a
denylist: a source that is missing from the registry groupings, or a chunk
with missing/unknown ``source`` metadata, is dropped, not published. A
misclassified source must fail closed here, because whatever survives this
script becomes world-readable.

Why "derive", not "rebuild": the docs-only bundle is exactly what prod would
produce if the course sources had never existed, so we build each artifact the
same way prod does, minus the non-doc rows:

* **Dense (Chroma)** - copy the prod collection, scan every chunk's metadata,
  and delete everything whose ``source`` is not an allowed doc key. This
  reuses the existing Cohere embeddings, so the build costs **$0** and is
  identical to prod for every docs chunk.
* **BM25 + document dict** - rebuilt from a docs-only view of
  ``all_sources_data.jsonl`` with the same helpers prod uses
  (``build_document_dict`` / ``BM25Index.build``). Pure, no network.
* **KB** - copy ``data/kb``, drop ``raw/courses`` and ``wiki/courses``, prune
  every remaining ``raw/`` source directory and ``wiki/frameworks/`` page
  whose key is not allowlisted (rows with missing/unknown ``source`` and
  retired courses are normalized into the docs group upstream, so they land
  there), and filter the generated indexes down to doc rows. Then the wiki is
  **publicized**: the scaffolder (``update_kb_wiki``) regenerates every
  AUTO-GENERATED marker block from the filtered manifest (so scaffolded link
  lists lose their course entries the same way they gained them), and course
  references in maintainer prose are pruned deterministically (course bullets
  dropped, course sentences stripped). A ``(maintainer)`` entry is appended to
  ``wiki/log.md`` per ``data/kb/MAINTAINER.md``.

Every build ends with a **leak audit** that fails the build if any course
path, course source key, non-allowlisted row, non-allowlisted ``raw/`` file,
or non-allowlisted source-keyed wiki page survives in the staged KB.

After the audit, the staged ``kb/`` tree is packed into a single
``kb.tar.gz`` and the tree is removed from staging. The public repo ships the
archive, not the ~3,000 individual KB files: anonymous downloads are
rate-limited per request, so the file-per-page layout stretched a token-free
cold start to ~15 minutes. The runtime extracts the archive after download
(``app.config._extract_kb_archive``). ``kb/**`` stays in the upload allow
patterns so the prune step clears the unpacked tree that earlier publishes
left in the repo.

Usage::

    # build into data/public_docs_bundle/ and upload to the public repo
    uv run -m data.scraping_scripts.build_public_docs_bundle

    # build only (inspect locally first), skip the upload
    uv run -m data.scraping_scripts.build_public_docs_bundle --skip-upload

Prerequisites: a complete local prod bundle (``data/chroma-db-all_sources/``,
``data/all_sources_data.jsonl``, ``data/kb/``). Build/refresh it with the docs
workflow first. Uploading additionally needs an ``HF_TOKEN`` with write access
to the ``towardsai-tutors`` org.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import re
import shutil
import sqlite3
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import chromadb

try:
    from data.scraping_scripts.source_registry import (
        ACTIVE_SOURCE_KEYS,
        ALL_SOURCES_JSONL,
        CONTEXTUAL_NODES_PKL,
        DOC_SOURCE_KEYS,
    )
    from data.scraping_scripts.update_kb_wiki import (
        AUTO_END,
        AUTO_START,
        update_kb_wiki,
    )
    from app.chroma_rag import (
        BM25Index,
        build_chunk_records,
        build_document_dict,
        get_chunk_record_source,
        load_jsonl_documents,
        save_bm25_index,
        save_document_dict,
    )
except ModuleNotFoundError:
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data.scraping_scripts.source_registry import (
        ACTIVE_SOURCE_KEYS,
        ALL_SOURCES_JSONL,
        CONTEXTUAL_NODES_PKL,
        DOC_SOURCE_KEYS,
    )
    from data.scraping_scripts.update_kb_wiki import (
        AUTO_END,
        AUTO_START,
        update_kb_wiki,
    )
    from app.chroma_rag import (
        BM25Index,
        build_chunk_records,
        build_document_dict,
        get_chunk_record_source,
        load_jsonl_documents,
        save_bm25_index,
        save_document_dict,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Layout mirrors the prod bundle so the runtime needs no path changes: the
# public repo has the identical tree (chroma-db-all_sources/, kb/), just with
# the course rows removed. Keep these names in sync with app/config.py.
VECTOR_DB_DIR = "chroma-db-all_sources"
VECTOR_COLLECTION_NAME = "chroma-db-all_sources"
DOCUMENT_DICT_FILE = "document_dict_all_sources.pkl"
BM25_INDEX_FILE = "bm25_index_all_sources.json.gz"
KB_DIR_NAME = "kb"
KB_ARCHIVE_NAME = "kb.tar.gz"

PUBLIC_REPO_ID = "towardsai-tutors/ai-tutor-vector-db-public"
DEFAULT_SOURCE_DIR = Path("data")
DEFAULT_STAGE_DIR = Path("data/public_docs_bundle")

# Allowlist: only these sources may appear in the public bundle. Anything
# else — course sources, sources someone forgot to classify, chunks with
# missing metadata — is dropped (fail closed).
_PUBLIC_KEYS = frozenset(DOC_SOURCE_KEYS)
# Registry sources that must never be named in public wiki prose (used by the
# prose pruner and the leak audit). Derived from the allowlist, not from
# COURSE_SOURCE_KEYS, so an unclassified source is treated as private too.
_NON_PUBLIC_KEYS = frozenset(ACTIVE_SOURCE_KEYS) - _PUBLIC_KEYS


def _is_public_row(row: dict) -> bool:
    return str(row.get("source") or "") in _PUBLIC_KEYS


def stage_chroma(source_dir: Path, stage_dir: Path, *, vacuum: bool = True) -> dict:
    """Copy the prod Chroma collection and keep only allowlisted doc chunks.

    Scans every chunk's metadata and deletes anything whose ``source`` is not
    in ``DOC_SOURCE_KEYS`` — including chunks with missing or unrecognized
    metadata, which a ``where`` denylist would silently keep. Reuses the
    existing docs embeddings (no Cohere call). Returns counts for the build
    summary.
    """
    src = source_dir / VECTOR_DB_DIR
    dst = stage_dir / VECTOR_DB_DIR
    if not src.is_dir():
        raise SystemExit(f"Missing prod Chroma collection: {src}")

    logger.info("Copying Chroma collection %s -> %s", src, dst)
    shutil.copytree(src, dst)

    client = chromadb.PersistentClient(path=str(dst))
    collection = client.get_collection(name=VECTOR_COLLECTION_NAME)
    before = collection.count()

    batch = 5000
    drop_ids: list[str] = []
    kept_by_source: dict[str, int] = {}
    offset = 0
    while True:
        page = collection.get(limit=batch, offset=offset, include=["metadatas"])
        ids = page.get("ids") or []
        if not ids:
            break
        metadatas = page.get("metadatas") or [None] * len(ids)
        for chunk_id, metadata in zip(ids, metadatas):
            source = str((metadata or {}).get("source") or "")
            if source in _PUBLIC_KEYS:
                kept_by_source[source] = kept_by_source.get(source, 0) + 1
            else:
                drop_ids.append(chunk_id)
        offset += len(ids)

    for index in range(0, len(drop_ids), batch):
        collection.delete(ids=drop_ids[index : index + batch])
    after = collection.count()
    logger.info(
        "Chroma: kept %s docs chunks, removed %s non-doc chunks",
        after,
        before - after,
    )
    for source in sorted(kept_by_source):
        logger.info("  %s: %s chunks", source, kept_by_source[source])

    # Release the client so the sqlite file is unlocked before VACUUM.
    del collection
    del client
    try:
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:  # API location varies across chromadb versions
        pass
    gc.collect()

    if after != sum(kept_by_source.values()):
        raise SystemExit(
            f"Chroma post-delete count mismatch: {after} chunks remain but "
            f"{sum(kept_by_source.values())} were classified as public."
        )

    if vacuum:
        sqlite_path = dst / "chroma.sqlite3"
        try:
            con = sqlite3.connect(sqlite_path)
            con.execute("VACUUM")
            con.close()
            logger.info("Vacuumed %s to reclaim space from deleted rows", sqlite_path)
        except sqlite3.Error as exc:  # best-effort; a larger file is harmless
            logger.warning("VACUUM skipped (%s); bundle will be larger", exc)

    return {
        "dense_kept": after,
        "dense_removed": before - after,
        "dense_by_source": kept_by_source,
    }


def rebuild_retrieval_artifacts(source_dir: Path, stage_dir: Path) -> dict:
    """Rebuild the BM25 index and document dict from a docs-only JSONL.

    Mirrors ``create_vector_stores.write_retrieval_artifacts`` so the public
    artifacts are byte-for-byte what prod would write without the course rows.
    """
    jsonl_path = source_dir / Path(ALL_SOURCES_JSONL).name
    if not jsonl_path.exists():
        raise SystemExit(f"Missing aggregate corpus: {jsonl_path}")

    all_rows = load_jsonl_documents(str(jsonl_path))
    docs_rows = [row for row in all_rows if _is_public_row(row)]
    logger.info(
        "Corpus: %s docs documents (dropped %s non-doc documents)",
        len(docs_rows),
        len(all_rows) - len(docs_rows),
    )

    dst = stage_dir / VECTOR_DB_DIR
    save_document_dict(build_document_dict(docs_rows), str(dst / DOCUMENT_DICT_FILE))
    bm25_records = build_chunk_records(docs_rows)
    save_bm25_index(BM25Index.build(bm25_records), str(dst / BM25_INDEX_FILE))
    logger.info(
        "Rebuilt document dict (%s docs) and BM25 index (%s chunks)",
        len(docs_rows),
        len(bm25_records),
    )
    return {"documents": len(docs_rows), "bm25_chunks": len(bm25_records)}


def stage_contextual_nodes(source_dir: Path, stage_dir: Path) -> int:
    """Write a docs-only copy of the contextual-nodes pickle (opt-in).

    Not needed at runtime; included only when ``--include-contextual`` is
    passed so others can rebuild the docs embeddings from scratch. Records
    that cannot be classified are dropped (fail closed), never published.
    """
    import pickle

    src = source_dir / Path(CONTEXTUAL_NODES_PKL).name
    if not src.exists():
        logger.warning("No contextual nodes pickle at %s; skipping", src)
        return 0
    with src.open("rb") as handle:
        records = pickle.load(handle)
    kept = []
    dropped_unclassified = 0
    for record in records:
        try:
            record_source = get_chunk_record_source(record)
        except Exception:
            dropped_unclassified += 1
            continue
        if record_source in _PUBLIC_KEYS:
            kept.append(record)
    dst = stage_dir / Path(CONTEXTUAL_NODES_PKL).name
    with dst.open("wb") as handle:
        pickle.dump(kept, handle)
    logger.info("Contextual nodes: kept %s docs chunks", len(kept))
    if dropped_unclassified:
        logger.warning(
            "Contextual nodes: dropped %s unclassifiable records (fail closed)",
            dropped_unclassified,
        )
    return len(kept)


def _filter_jsonl(path: Path) -> int:
    """Keep only allowlisted doc rows in a generated *.jsonl index, in place."""
    if not path.exists():
        return 0
    kept = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if _is_public_row(row):
                kept.append(line if line.endswith("\n") else line + "\n")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(kept)
    return len(kept)


def _filter_tsv(path: Path) -> int:
    """Keep only allowlisted doc rows in symbols.tsv, in place (keeps header)."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        kept = [row for row in reader if _is_public_row(row)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(kept)
    return len(kept)


def _remove_tree_or_file(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def stage_kb(source_dir: Path, stage_dir: Path) -> dict:
    """Copy data/kb and prune non-public content (raw, wiki, generated indexes).

    Beyond dropping the course trees wholesale, the raw/ mirrors and the
    source-keyed wiki pages are filtered by the same allowlist as everything
    else: ``build_kb_artifacts.normalize_record`` maps a row with a
    missing/unknown ``source`` — or a retired course whose rows linger in
    ``all_sources_data.jsonl`` after its key left the registry — to
    ``source_group="docs"``, so its markdown lands under ``raw/docs/<key>/``
    and its wiki page under ``wiki/frameworks/<key>.md``, outside the
    ``courses/`` prunes. Anything whose key is not in the allowlist is
    removed, not published (fail closed).
    """
    src = source_dir / KB_DIR_NAME
    dst = stage_dir / KB_DIR_NAME
    if not src.is_dir():
        raise SystemExit(f"Missing KB directory: {src}")

    logger.info("Copying KB %s -> %s", src, dst)
    shutil.copytree(src, dst)

    for relative in ("raw/courses", "wiki/courses"):
        target = dst / relative
        if target.exists():
            shutil.rmtree(target)
            logger.info("Removed %s", target)

    # Allowlist the raw tree: keep only raw/docs/<key>/ for public keys.
    # Every other raw/ entry — an unknown/typo'd source dir, a retired
    # course's dir, an unexpected group dir, a stray file — is removed.
    raw_root = dst / "raw"
    if raw_root.is_dir():
        for group_dir in sorted(raw_root.iterdir()):
            if not group_dir.is_dir() or group_dir.name != "docs":
                _remove_tree_or_file(group_dir)
                logger.info("Removed unexpected raw entry %s (fail closed)", group_dir)
                continue
            for source_entry in sorted(group_dir.iterdir()):
                if source_entry.is_dir() and source_entry.name in _PUBLIC_KEYS:
                    continue
                _remove_tree_or_file(source_entry)
                logger.info(
                    "Removed non-allowlisted raw source %s (fail closed)",
                    source_entry,
                )

    # wiki/frameworks/ pages are source-keyed the same way; a retired course's
    # page lands here (its key is not in COURSE_SOURCE_KEYS anymore) and its
    # name is unknown to the prose pruner (which derives from ACTIVE keys),
    # so filter structurally: keep only <key>.md pages for allowlisted keys.
    frameworks_dir = dst / "wiki" / "frameworks"
    if frameworks_dir.is_dir():
        for page in sorted(frameworks_dir.iterdir()):
            if page.is_file() and page.suffix == ".md" and page.stem in _PUBLIC_KEYS:
                continue
            _remove_tree_or_file(page)
            logger.info(
                "Removed non-allowlisted wiki framework page %s (fail closed)", page
            )

    # MAINTAINER.md is the maintainer-agent manual: not needed at runtime, not
    # in git, ships only in the private bundle — and its worked examples quote
    # course paths, so it must not enter the public bundle.
    maintainer = dst / "MAINTAINER.md"
    if maintainer.exists():
        maintainer.unlink()
        logger.info("Removed %s", maintainer)

    generated = dst / "generated"
    manifest_rows = _filter_jsonl(generated / "corpus_manifest.jsonl")
    heading_rows = _filter_jsonl(generated / "headings.jsonl")
    symbol_rows = _filter_tsv(generated / "symbols.tsv")
    logger.info(
        "KB generated indexes: %s manifest rows, %s heading rows, %s symbol rows",
        manifest_rows,
        heading_rows,
        symbol_rows,
    )
    return {"manifest_rows": manifest_rows}


# Course paths, plus the `lesson-…` shorthand maintainer prose uses when a
# bullet refers back to a course lesson cited earlier on the page (those
# earlier citations get pruned, which would leave this shorthand dangling).
# Verified to not occur anywhere in the docs sources or docs wiki pages.
_COURSE_PATH_TOKENS = ("raw/courses/", "wiki/courses/", "../courses/", "`lesson-")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _prose_prune_tokens() -> tuple[str, ...]:
    """Substrings that must not survive in public wiki prose.

    Course paths plus the names of every non-public source (derived from the
    allowlist so an unclassified source counts as private).
    """
    return _COURSE_PATH_TOKENS + tuple(sorted(_NON_PUBLIC_KEYS))


def prune_course_prose(text: str, tokens: tuple[str, ...]) -> tuple[str, int]:
    """Remove course references from maintainer prose, deterministically.

    Operates line-wise, leaving AUTO-GENERATED marker blocks untouched (the
    scaffolder owns those and has already regenerated them from the filtered
    manifest). A list item mentioning a course is dropped whole; a prose line
    keeps only the sentences that do not mention one. Returns the new text and
    the number of lines that were dropped or trimmed.
    """
    out_lines: list[str] = []
    removed = 0
    in_marker = False
    for line in text.splitlines():
        if AUTO_START in line:
            in_marker = True
        elif AUTO_END in line:
            in_marker = False
            out_lines.append(line)
            continue
        if in_marker or not any(token in line for token in tokens):
            out_lines.append(line)
            continue
        removed += 1
        if _LIST_ITEM_RE.match(line):
            continue  # drop the whole list item
        kept_sentences = [
            sentence
            for sentence in _SENTENCE_SPLIT_RE.split(line)
            if not any(token in sentence for token in tokens)
        ]
        if kept_sentences:
            out_lines.append(" ".join(kept_sentences))

    # Dropping lines can leave doubled blank lines; collapse them (outside
    # marker blocks nothing relies on multiple consecutive blanks).
    collapsed: list[str] = []
    for line in out_lines:
        if not line.strip() and collapsed and not collapsed[-1].strip():
            continue
        collapsed.append(line)
    return "\n".join(collapsed).rstrip() + "\n", removed


def publicize_wiki(stage_dir: Path) -> dict:
    """Make the staged wiki consistent with the docs-only corpus.

    Follows the ownership model in ``data/kb/MAINTAINER.md``: the scaffolder
    regenerates everything inside AUTO-GENERATED markers from the (already
    filtered) manifest, then course references in maintainer prose are pruned,
    and a ``(maintainer)`` entry is appended to ``wiki/log.md``.
    """
    kb_dir = stage_dir / KB_DIR_NAME

    # 1. Scaffolder pass: marker blocks lose their course entries the same way
    #    they gained them; AGENTS.md is rewritten from the template; a
    #    scaffolder entry lands in log.md.
    update_kb_wiki(kb_dir, seed_defaults=False)

    # 2. Prune maintainer prose (outside markers), including old log entries.
    tokens = _prose_prune_tokens()
    pruned_lines = 0
    pruned_files = 0
    for md_path in sorted((kb_dir / "wiki").rglob("*.md")):
        original = md_path.read_text(encoding="utf-8")
        updated, removed = prune_course_prose(original, tokens)
        if removed:
            md_path.write_text(updated, encoding="utf-8")
            pruned_files += 1
            pruned_lines += removed
    logger.info(
        "Wiki prose: pruned %s course-referencing lines across %s pages",
        pruned_lines,
        pruned_files,
    )

    # 3. Log the pass per MAINTAINER.md's logging convention (appended after
    #    pruning; the entry itself names no pruned source).
    now = datetime.now(UTC).isoformat(timespec="seconds")
    entry = (
        f"## {now} (maintainer)\n\n"
        "- Derived the public docs-only bundle: filtered the corpus and\n"
        "  generated indexes to documentation sources, regenerated scaffolded\n"
        f"  wiki blocks from the filtered manifest, and pruned {pruned_lines}\n"
        f"  prose lines referencing removed sources across {pruned_files} pages."
    )
    log_path = kb_dir / "wiki" / "log.md"
    existing = (
        log_path.read_text(encoding="utf-8").rstrip()
        if log_path.exists()
        else "# KB Log"
    )
    log_path.write_text(f"{existing}\n\n{entry}\n", encoding="utf-8")
    return {"wiki_pruned_lines": pruned_lines, "wiki_pruned_files": pruned_files}


def audit_staged_kb(stage_dir: Path) -> None:
    """Fail the build if anything non-public survives in the staged KB.

    Last line of defense before publishing: checks course directories are
    gone, every raw/ file and source-keyed wiki page sits under an
    allowlisted source key, generated indexes contain only allowlisted rows,
    and no wiki page mentions a course path or a non-public source name.
    """
    kb_dir = stage_dir / KB_DIR_NAME
    problems: list[str] = []

    for relative in ("raw/courses", "wiki/courses"):
        if (kb_dir / relative).exists():
            problems.append(f"{relative}/ still exists")

    # Structural allowlist over raw/: every file must live at
    # raw/docs/<key>/... with an allowlisted key. raw/ is exempt from the
    # prose-token scan below (upstream doc mirrors may quote anything), so
    # without this check a non-allowlisted source directory — unknown/typo'd
    # source metadata, or a retired course's leftover rows, both of which
    # build_kb_artifacts files under raw/docs/ — would ship silently.
    raw_root = kb_dir / "raw"
    if raw_root.exists():
        for path in sorted(p for p in raw_root.rglob("*") if p.is_file()):
            parts = path.relative_to(kb_dir).parts
            if len(parts) < 4 or parts[1] != "docs" or parts[2] not in _PUBLIC_KEYS:
                problems.append(
                    f"{path.relative_to(kb_dir)}: raw file outside an "
                    "allowlisted raw/docs/<source>/ directory"
                )

    # wiki/frameworks/ pages are keyed by source; a retired course's page
    # lands here and its name is not a prose token (tokens derive from
    # ACTIVE keys), so require every page to belong to an allowlisted key.
    frameworks_dir = kb_dir / "wiki" / "frameworks"
    if frameworks_dir.exists():
        for path in sorted(p for p in frameworks_dir.rglob("*") if p.is_file()):
            if (
                path.parent != frameworks_dir
                or path.suffix != ".md"
                or path.stem not in _PUBLIC_KEYS
            ):
                problems.append(
                    f"{path.relative_to(kb_dir)}: wiki page for a "
                    "non-allowlisted source key"
                )

    generated = kb_dir / "generated"
    for name in ("corpus_manifest.jsonl", "headings.jsonl"):
        path = generated / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not _is_public_row(row):
                problems.append(f"{name}: non-public row source={row.get('source')!r}")
                break
    symbols = generated / "symbols.tsv"
    if symbols.exists():
        with symbols.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if not _is_public_row(row):
                    problems.append(
                        f"symbols.tsv: non-public row source={row.get('source')!r}"
                    )
                    break

    # Scan every markdown file we author or generate (wiki, kb root, anything
    # new) — only raw/ (upstream doc mirrors) and AGENTS.md are exempt.
    # AGENTS.md is regenerated from the repo template, which legitimately
    # names the wiki/courses/ *layout* (not course content) for the private
    # bundle's benefit.
    tokens = _prose_prune_tokens()
    for md_path in sorted(kb_dir.rglob("*.md")):
        relative = md_path.relative_to(kb_dir)
        if relative.parts[0] == "raw" or relative == Path("AGENTS.md"):
            continue
        text = md_path.read_text(encoding="utf-8")
        hits = sorted({token for token in tokens if token in text})
        if hits:
            problems.append(f"{relative}: mentions {', '.join(hits)}")

    if problems:
        for problem in problems:
            logger.error("Leak audit: %s", problem)
        raise SystemExit(
            f"Public bundle leak audit failed with {len(problems)} problem(s); "
            "nothing was uploaded."
        )
    logger.info("Leak audit passed: staged KB contains no course content.")


def write_dataset_card(stage_dir: Path) -> None:
    """Write the HF dataset card (README.md) for the public repo.

    Script-owned so it stays accurate across refreshes; uploaded because
    ``public_allow_patterns`` includes ``README.md``.
    """
    doc_sources = ", ".join(f"`{key}`" for key in sorted(_PUBLIC_KEYS))
    card = f"""---
pretty_name: AI Tutor Vector DB (public docs-only bundle)
viewer: false
---

# AI Tutor vector DB — public docs-only bundle

Prebuilt retrieval artifacts for the [Towards AI tutor chatbot](https://github.com/towardsai/ai-tutor-app):
a ChromaDB collection (Cohere `embed-v4.0` embeddings), BM25 index and
document dictionary, plus the file-based knowledge base (`kb/`) the agent
browses at runtime.

This is the **public fallback** for the private
`towardsai-tutors/ai-tutor-vector-db` bundle. It contains **documentation
sources only** ({doc_sources}); the Towards AI course content that ships in
the private bundle is not included. The app downloads this bundle
automatically on first start when no `HF_TOKEN` (or one without access to the
private dataset) is available.

Derived from the production artifacts by
`data/scraping_scripts/build_public_docs_bundle.py` — an allowlist filter over
the prod bundle (same embeddings, same tree layout), followed by a leak audit.
Documentation content belongs to its upstream projects and keeps their
respective licenses.
"""
    (stage_dir / "README.md").write_text(card, encoding="utf-8")


def archive_kb(stage_dir: Path) -> dict:
    """Pack the audited ``kb/`` tree into ``kb.tar.gz`` and drop the tree.

    One archive instead of ~3,000 files keeps a token-free cold start out of
    HF's anonymous per-request rate limits. Removing the tree afterwards makes
    the prune step delete the unpacked ``kb/**`` files that earlier publishes
    left in the repo (prune deletes remote files that match the allow patterns
    but no longer exist locally).
    """
    kb_dir = stage_dir / KB_DIR_NAME
    archive_path = stage_dir / KB_ARCHIVE_NAME
    logger.info("Archiving %s -> %s", kb_dir, archive_path)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(kb_dir, arcname=KB_DIR_NAME)
    shutil.rmtree(kb_dir)
    size_mb = archive_path.stat().st_size / 1e6
    logger.info("KB archive: %.1f MB (kb/ tree removed from staging)", size_mb)
    return {"kb_archive_mb": round(size_mb, 1)}


def build_bundle(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    stage_dir: Path = DEFAULT_STAGE_DIR,
    include_contextual: bool = False,
    vacuum: bool = True,
) -> dict:
    if stage_dir.exists():
        logger.info("Clearing existing staging dir %s", stage_dir)
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"sources": sorted(DOC_SOURCE_KEYS)}
    summary.update(stage_chroma(source_dir, stage_dir, vacuum=vacuum))
    summary.update(rebuild_retrieval_artifacts(source_dir, stage_dir))
    summary.update(stage_kb(source_dir, stage_dir))
    summary.update(publicize_wiki(stage_dir))
    write_dataset_card(stage_dir)
    if include_contextual:
        summary["contextual_chunks"] = stage_contextual_nodes(source_dir, stage_dir)
    audit_staged_kb(stage_dir)
    summary.update(archive_kb(stage_dir))
    return summary


def public_allow_patterns(*, include_contextual: bool) -> list[str]:
    # kb/** stays in the list even though staging only holds kb.tar.gz: the
    # prune step deletes remote files that match the patterns but are absent
    # locally, which clears the unpacked kb tree from pre-archive publishes.
    patterns = [
        f"{VECTOR_DB_DIR}/**",
        f"{KB_DIR_NAME}/**",
        KB_ARCHIVE_NAME,
        "README.md",
    ]
    if include_contextual:
        patterns.append(Path(CONTEXTUAL_NODES_PKL).name)
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and upload the public docs-only vector-db/KB bundle."
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--stage-dir", default=str(DEFAULT_STAGE_DIR))
    parser.add_argument("--repo", default=PUBLIC_REPO_ID)
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Build the bundle locally but do not push it to Hugging Face.",
    )
    parser.add_argument(
        "--include-contextual",
        action="store_true",
        help="Also ship a docs-only all_sources_contextual_nodes.pkl so others "
        "can rebuild the docs embeddings (larger download; off by default).",
    )
    parser.add_argument(
        "--no-vacuum",
        action="store_true",
        help="Skip the sqlite VACUUM that reclaims space from deleted rows.",
    )
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir)
    summary = build_bundle(
        source_dir=Path(args.source_dir),
        stage_dir=stage_dir,
        include_contextual=args.include_contextual,
        vacuum=not args.no_vacuum,
    )

    print("\nBuilt public docs-only bundle:")
    print(f"  sources:        {', '.join(summary['sources'])}")
    print(f"  documents:      {summary['documents']}")
    print(
        f"  dense chunks:   {summary['dense_kept']} (removed {summary['dense_removed']})"
    )
    print(f"  bm25 chunks:    {summary['bm25_chunks']}")
    print(f"  KB manifest:    {summary['manifest_rows']} rows")
    print(
        f"  wiki pruning:   {summary['wiki_pruned_lines']} course-referencing "
        f"lines removed across {summary['wiki_pruned_files']} pages"
    )
    if "contextual_chunks" in summary:
        print(f"  contextual:     {summary['contextual_chunks']} chunks")
    print(f"  KB archive:     {summary['kb_archive_mb']} MB (kb.tar.gz)")
    print(f"  staged at:      {stage_dir}")
    print("  leak audit:     passed")

    if args.skip_upload:
        print("\n--skip-upload set; not pushing to Hugging Face.")
        return

    from data.scraping_scripts.upload_dbs_to_hf import upload_bundle

    print(f"\nUploading {stage_dir} -> {args.repo} (public) ...")
    upload_bundle(
        args.repo,
        folder_path=str(stage_dir),
        allow_patterns=public_allow_patterns(
            include_contextual=args.include_contextual
        ),
        create_public=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
