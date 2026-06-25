"""Seed and refresh the agent-facing KB wiki."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_KB_DIR = Path("data/kb")
# Edit this template to change KB agent guidance; gets copied to data/kb/AGENTS.md.
AGENTS_TEMPLATE_PATH = Path(__file__).resolve().parent / "kb_agents_template.md"
AUTO_START = "<!-- AUTO-GENERATED:START -->"
AUTO_END = "<!-- AUTO-GENERATED:END -->"
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "rag": ("rag", "retrieval", "vector", "embedding", "chunk"),
    "agents": ("agent", "tool", "langgraph", "workflow", "mcp"),
    "retrieval": ("retrieval", "retriever", "bm25", "rerank", "search"),
    "lora": ("lora", "qlora", "adapter", "peft", "loraconfig"),
    "quantization": ("quantization", "quantized", "bitsandbytes", "nf4", "4-bit"),
    "fine-tuning": ("fine-tuning", "finetuning", "sft", "trainer", "trl"),
    "model-loading": ("from_pretrained", "automodel", "tokenizer", "device_map"),
    "tool-calling": ("tool calling", "function calling", "tools", "schema"),
    "evaluation": ("evaluation", "eval", "benchmark", "metrics"),
    "guardrails": ("guardrail", "jailbreak", "prompt injection", "moderation"),
    "serving": ("vllm", "sglang", "pagedattention", "kv cache", "serving"),
}


def load_manifest(kb_dir: Path) -> list[dict[str, Any]]:
    manifest_path = kb_dir / "generated" / "corpus_manifest.jsonl"
    if not manifest_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def wiki_path(kb_dir: Path, *parts: str) -> Path:
    return kb_dir / "wiki" / Path(*parts)


def shell_path(row: dict[str, Any], kb_dir: Path) -> str:
    path = str(row.get("path") or "")
    if not path:
        return ""
    try:
        return Path(path).resolve().relative_to(kb_dir.resolve()).as_posix()
    except ValueError:
        pass
    prefix = "data/kb/"
    if path.startswith(prefix):
        return path[len(prefix) :]
    parts = Path(path).parts
    if "raw" in parts:
        return "/".join(parts[parts.index("raw") :])
    return path


def generated_block(content: str) -> str:
    return f"{AUTO_START}\n{content.rstrip()}\n{AUTO_END}"


def write_generated_section(
    path: Path, header: str, generated: str, *, overwrite: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    block = generated_block(generated)
    if not path.exists() or overwrite:
        path.write_text(f"{header.rstrip()}\n\n{block}\n", encoding="utf-8")
        return

    existing = path.read_text(encoding="utf-8")
    start = existing.find(AUTO_START)
    end = existing.find(AUTO_END, start)
    if start != -1 and end != -1:
        updated = existing[:start] + block + existing[end + len(AUTO_END) :]
        path.write_text(updated.rstrip() + "\n", encoding="utf-8")
        return
    path.write_text(existing.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def top_sources(manifest: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counts = Counter(str(row.get("source") or "unknown") for row in manifest)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def seed_index(
    kb_dir: Path, manifest: list[dict[str, Any]], *, overwrite: bool
) -> None:
    group_by_source = {
        str(row.get("source") or "unknown"): str(row.get("source_group") or "docs")
        for row in manifest
    }
    source_lines = []
    for source, count in top_sources(manifest):
        folder = "courses" if group_by_source.get(source) == "courses" else "frameworks"
        source_lines.append(
            f"- {source}: `wiki/{folder}/{source}.md` - {count} corpus pages"
        )
    topic_lines = [f"- {topic}: `wiki/topics/{topic}.md`" for topic in TOPIC_KEYWORDS]
    header = """# AI Tutor KB Index

This wiki is a navigation layer over the generated corpus markdown.
"""
    generated = f"""## Sources

{chr(10).join(source_lines) if source_lines else "- No corpus sources indexed yet."}

## Topics

{chr(10).join(topic_lines)}

## Generated Indexes

- Corpus manifest: `generated/corpus_manifest.jsonl`
- Headings: `generated/headings.jsonl`
- Symbols: `generated/symbols.tsv`
"""
    path = wiki_path(kb_dir, "index.md")
    if path.exists() and AUTO_START not in path.read_text(encoding="utf-8"):
        overwrite = True
    write_generated_section(path, header, generated, overwrite=overwrite)


def seed_log(kb_dir: Path, manifest: list[dict[str, Any]], *, overwrite: bool) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    path = wiki_path(kb_dir, "log.md")
    entry = f"""## {now}

- Generated or refreshed the KB wiki scaffold/source maps.
- Corpus documents indexed: {len(manifest)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(f"# KB Log\n\n{entry.rstrip()}\n", encoding="utf-8")
        return
    existing = path.read_text(encoding="utf-8").rstrip()
    path.write_text(f"{existing}\n\n{entry.rstrip()}\n", encoding="utf-8")


def seed_source_pages(
    kb_dir: Path, manifest: list[dict[str, Any]], *, overwrite: bool
) -> None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_source[str(row.get("source") or "unknown")].append(row)

    for source, rows in sorted(by_source.items()):
        rows = sorted(rows, key=lambda item: str(item.get("title") or ""))
        sample = rows[:20]
        page_links = [
            f"- {row.get('title')}: `{shell_path(row, kb_dir)}`" for row in sample
        ]
        group = str(rows[0].get("source_group") or "docs")
        folder = "courses" if group == "courses" else "frameworks"
        header = f"""# {source}

Use this page to orient inside the `{source}` corpus before reading source pages directly.
"""
        generated = f"""## Corpus Pages

{chr(10).join(page_links) if page_links else "- No pages indexed."}

## Verification

Treat generated corpus markdown as the source of truth for this source. Use exact search for symbols, error strings, and code identifiers.
"""
        path = wiki_path(kb_dir, folder, f"{source}.md")
        page_overwrite = overwrite
        if path.exists() and AUTO_START not in path.read_text(encoding="utf-8"):
            page_overwrite = True
        write_generated_section(path, header, generated, overwrite=page_overwrite)


def matching_topic_rows(
    manifest: list[dict[str, Any]],
    keywords: tuple[str, ...],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    scored: list[tuple[int, dict[str, Any]]] = []
    lowered = tuple(keyword.lower() for keyword in keywords)
    for row in manifest:
        title = str(row.get("title") or "")
        path = Path(str(row.get("path") or ""))
        text = title.lower()
        if path.exists():
            text += "\n" + path.read_text(encoding="utf-8").lower()[:20_000]
        score = sum(text.count(keyword) for keyword in lowered)
        if score:
            scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("title") or "")))
    return [row for _score, row in scored[:limit]]


def seed_topic_pages(
    kb_dir: Path, manifest: list[dict[str, Any]], *, overwrite: bool
) -> None:
    for topic, keywords in TOPIC_KEYWORDS.items():
        rows = matching_topic_rows(manifest, keywords)
        links = [f"- {row.get('title')}: `{shell_path(row, kb_dir)}`" for row in rows]
        header = f"""# {topic.replace("-", " ").title()}

Use this page as a starting map for questions involving: {", ".join(keywords)}.
"""
        generated = f"""## Starting Corpus Pages

{chr(10).join(links) if links else "- No strong corpus matches found yet."}

## Agent Notes

- Start here for orientation.
- Verify exact factual claims in raw corpus markdown pages.
- Use `rg` over `raw/` and `generated/symbols.tsv` for API/class/function names.
"""
        write_generated_section(
            wiki_path(kb_dir, "topics", f"{topic}.md"),
            header,
            generated,
            overwrite=overwrite,
        )


def seed_index_pages(kb_dir: Path, *, overwrite: bool) -> None:
    for folder, title in (("recipes", "Recipes"), ("errors", "Errors")):
        header = f"""# {title}

This index starts empty. Add pages here when repeated recipes or recurring errors emerge from the corpus.
"""
        write_generated_section(
            wiki_path(kb_dir, folder, "index.md"),
            header,
            "- No generated entries yet.",
            overwrite=overwrite,
        )


def write_agents_md(kb_dir: Path) -> None:
    if not AGENTS_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template missing: {AGENTS_TEMPLATE_PATH}")
    (kb_dir / "AGENTS.md").parent.mkdir(parents=True, exist_ok=True)
    (kb_dir / "AGENTS.md").write_text(
        AGENTS_TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )


def wiki_is_empty(kb_dir: Path) -> bool:
    """True when wiki/ is missing or has no .md files (fresh build)."""
    wiki_dir = kb_dir / "wiki"
    if not wiki_dir.exists():
        return True
    return not any(wiki_dir.rglob("*.md"))


def update_kb_wiki(kb_dir: Path, *, seed_defaults: bool) -> None:
    manifest = load_manifest(kb_dir)
    # Empty wiki/ promotes to a full seed so topic/recipe/error pages get written.
    if not seed_defaults and wiki_is_empty(kb_dir):
        seed_defaults = True
    write_agents_md(kb_dir)
    seed_index(kb_dir, manifest, overwrite=seed_defaults)
    seed_log(kb_dir, manifest, overwrite=seed_defaults)
    seed_source_pages(kb_dir, manifest, overwrite=seed_defaults)
    seed_topic_pages(kb_dir, manifest, overwrite=seed_defaults)
    seed_index_pages(kb_dir, overwrite=seed_defaults)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed or refresh the KB wiki.")
    parser.add_argument("--kb-dir", default=str(DEFAULT_KB_DIR))
    parser.add_argument("--seed-defaults", action="store_true")
    args = parser.parse_args()

    update_kb_wiki(
        Path(args.kb_dir),
        seed_defaults=bool(args.seed_defaults),
    )
    print(f"Updated KB wiki in {args.kb_dir}")


if __name__ == "__main__":
    main()
