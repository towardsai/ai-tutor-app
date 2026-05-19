"""Helpers for keeping contextual-node pickles aligned with active sources."""

from __future__ import annotations

import logging
import os
import pickle
from collections import Counter
from typing import Any

from data.scraping_scripts.source_registry import (
    ACTIVE_SOURCE_KEYS,
    CONTEXTUAL_NODES_PKL,
)
from scripts.chroma_rag import get_chunk_record_source

logger = logging.getLogger(__name__)


def prune_contextual_nodes_to_active_sources(
    pkl_path: str = CONTEXTUAL_NODES_PKL,
) -> Counter[str]:
    """Remove contextual nodes whose source is not active in source_registry.py."""
    if not os.path.exists(pkl_path):
        logger.info("%s does not exist; no contextual nodes to prune", pkl_path)
        return Counter()

    with open(pkl_path, "rb") as handle:
        nodes = pickle.load(handle)

    kept_nodes: list[Any] = []
    removed_counts: Counter[str] = Counter()
    unknown_source = 0

    for node in nodes:
        try:
            source = get_chunk_record_source(node)
        except Exception:
            unknown_source += 1
            kept_nodes.append(node)
            continue

        if source in ACTIVE_SOURCE_KEYS:
            kept_nodes.append(node)
        else:
            removed_counts[str(source)] += 1

    if not removed_counts:
        if unknown_source:
            logger.info(
                "No inactive contextual nodes pruned; kept %s nodes with unknown source",
                unknown_source,
            )
        return removed_counts

    with open(pkl_path, "wb") as handle:
        pickle.dump(kept_nodes, handle)

    logger.info(
        "Pruned %s inactive contextual nodes from %s: %s",
        sum(removed_counts.values()),
        pkl_path,
        dict(sorted(removed_counts.items())),
    )
    if unknown_source:
        logger.info("Kept %s contextual nodes with unknown source", unknown_source)
    return removed_counts
