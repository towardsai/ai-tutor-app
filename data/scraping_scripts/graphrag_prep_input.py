"""Prepare GraphRAG input from the existing corpus.

Emits one JSONL row per corpus document into the GraphRAG workspace input dir,
using the canonical raw markdown mirrors as text and the KB manifest as the
metadata source of truth. The row ``id`` is set to our ``doc_id`` so GraphRAG's
``text_units.document_ids`` map straight back to ``corpus_manifest.jsonl``
(source / url / lesson) at retrieval time -- no GraphRAG metadata propagation
needed.

This is part of the GraphRAG-vs-RAG experiment (branch experiment/graphrag-vs-rag).
The workspace under data/graphrag/ is local-only and never uploaded to the HF
vector-db bundle.

  uv run -m data.scraping_scripts.graphrag_prep_input            # full corpus
  uv run -m data.scraping_scripts.graphrag_prep_input --limit 10 # smoke subset
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("graphrag_prep_input")

MANIFEST_PATH = "data/kb/generated/corpus_manifest.jsonl"
DEFAULT_OUT = "data/graphrag/input/corpus.jsonl"


def load_manifest(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_doc_text(record: dict) -> str:
    """Canonical raw markdown for a manifest record, or '' if unreadable."""
    path = record.get("path")
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning(
            "Skipping %s: cannot read %s (%s)", record.get("doc_id"), path, exc
        )
        return ""


def build_rows(records: list[dict], sources: set[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        if sources and record.get("source") not in sources:
            continue
        text = read_doc_text(record)
        if not text:
            continue
        rows.append(
            {
                "id": record["doc_id"],
                "title": record.get("title") or record["doc_id"],
                "text": text,
                "source": record.get("source", ""),
                "url": record.get("url", ""),
                "source_group": record.get("source_group", ""),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="First N documents only (for a cheap smoke index); 0 = all.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=[],
        help="Restrict to these source keys (e.g. full_stack_ai_engineering); "
        "empty = all sources. Used to scope the GraphRAG index to the eval "
        "battery's sources and keep indexing within budget.",
    )
    parser.add_argument(
        "--doc-ids",
        nargs="*",
        default=[],
        help="Restrict to these exact doc_ids (e.g. a single long lesson for the "
        "knowledge-compaction study).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    records = load_manifest(args.manifest)
    if args.sources:
        wanted = set(args.sources)
        records = [r for r in records if r.get("source") in wanted]
    if args.doc_ids:
        wanted_ids = set(args.doc_ids)
        records = [r for r in records if r.get("doc_id") in wanted_ids]
    if args.limit:
        records = records[: args.limit]
    rows = build_rows(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    total_tokens = sum(r.get("tokens", 0) for r in records)
    logger.info(
        "Wrote %d/%d docs (%.1fM manifest tokens) -> %s",
        len(rows),
        len(records),
        total_tokens / 1e6,
        out_path,
    )


if __name__ == "__main__":
    main()
