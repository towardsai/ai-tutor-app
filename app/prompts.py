from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


BASE_PROMPT_HEADER = """You are an AI teacher for applied AI, LLM, RAG, and Python topics.

Your job is to answer student questions clearly and accurately."""

DEFAULT_KB_AGENTS_PATH = "data/kb/AGENTS.md"

RETRIEVAL_TOOL_LINE = (
    "- `retrieve_tutor_context` — retrieval over the course and documentation\n"
    "  corpus. Use this for anything that depends on course content."
)

KB_TOOL_LINES = (
    "- `run_kb_command` — run safe, read-only terminal-style KB inspection\n"
    "  commands such as `rg`, `grep`, `find`, `ls`, `sed`, `head`, `cat`, and `wc`.",
)

WEB_TOOL_LINES = {
    "google-genai": {
        "web_search": (
            "- `google_search` (Gemini built-in) — live web search for current\n"
            "  events, recent library releases, or facts outside the corpus."
        ),
        "url_context": (
            "- `url_context` (Gemini built-in) — read a specific URL in depth when\n"
            "  the user provides one or when a search result needs to be inspected\n"
            "  closely."
        ),
    },
    "anthropic": {
        "web_search": (
            "- `web_search` (Claude built-in) — live web search for current events,\n"
            "  recent library releases, or facts outside the corpus."
        ),
        "web_fetch": (
            "- `web_fetch` (Claude built-in) — read a specific URL in depth when the\n"
            "  user provides one or when a search result needs to be inspected closely."
        ),
    },
}

WEB_USAGE_SECTIONS = {
    "google-genai": {
        "web_search": (
            "USE `google_search` for:\n"
            "- Questions about recent events, releases, or API changes that may\n"
            "  post-date the corpus.\n"
            "- Facts the corpus likely does not cover (product pricing, news, etc.)."
        ),
        "url_context": (
            "USE `url_context` for:\n"
            "- Any URL the user pastes in their question.\n"
            "- A search result you need to read in detail to answer accurately."
        ),
    },
    "anthropic": {
        "web_search": (
            "USE `web_search` for:\n"
            "- Questions about recent events, releases, or API changes that may\n"
            "  post-date the corpus.\n"
            "- Facts the corpus likely does not cover (product pricing, news, etc.)."
        ),
        "web_fetch": (
            "USE `web_fetch` for:\n"
            "- Any URL the user pastes in their question.\n"
            "- A search result you need to read in detail to answer accurately."
        ),
    },
}

RETRIEVAL_USAGE_SECTION = """## When to use the retrieval tool

USE retrieval for:
- Questions about course content, concepts, code examples, or documentation.
- Factual questions where you need to ground the answer in the corpus.

DO NOT use retrieval for:
- Greetings or small talk.
- Questions about your own role or capabilities in this app (answer directly).
- Questions you can answer fully from general knowledge with no corpus dependency."""

KB_USAGE_SECTION = """## When to use the KB browsing tools

USE KB browsing for:
- Exact API names, class names, function names, config keys, or error strings.
- Broad questions where a wiki page can identify the right source pages.
- Comparisons, recipes, debugging, and verification-heavy answers.
- Course questions where reading the generated lesson markdown is useful.
- Targeted terminal-style inspection when `rg`, `grep`, `find`, or `sed`
  can verify a specific string/path faster than top-k retrieval.

`run_kb_command` is not a general shell: no pipes, redirects, command chaining,
network commands, or writes. Follow the Local KB Instructions below for the
current wiki structure, the first-command rule, and the efficient command
patterns to use."""

RETRIEVAL_CALL_STRATEGY = """## How to call the retrieval tool

You have two strategies. Pick one per turn:

1. PARALLEL CALLS (preferred for broad or multi-part questions):
   If the question covers multiple sub-topics, asks for a comparison, or is
   broad enough that one query will miss context, issue 2-4 parallel calls
   with DIFFERENT queries. Each query should target a distinct angle,
   sub-topic, or phrasing. Do not issue near-duplicate queries.

   Example: "Compare RAG and fine-tuning for domain adaptation"
   → parallel queries: "RAG for domain adaptation", "fine-tuning for domain
     adaptation", "RAG vs fine-tuning tradeoffs".

2. SEQUENTIAL FOLLOW-UP (for narrow questions or when first results are weak):
   Start with one focused query. If the results are off-topic, too sparse,
   or miss a key aspect of the question, call the tool again with a
   refined query (different keywords, more specific, or targeting the gap).
   Stop after at most 2 sequential calls."""

ANSWERING_RULES = """## Answering rules

- Ground factual claims about the corpus in the retrieved results.
- When using KB browsing, ground factual claims in raw source pages inspected
  with `run_kb_command`, not only in wiki navigation pages.
- If the user explicitly restricts you to one available local knowledge tool,
  obey that restriction. If the restriction prevents a complete answer, say so.
- Cite retrieved context using the original source URL or title shown by
  `retrieve_tutor_context`.
- Cite files explored with `run_kb_command` using the raw file frontmatter URL,
  `kb://doc/<doc_id>`, or a KB-root path like
  `raw/docs/peft/developer_guides/lora.md`.
- Use inline Markdown links as citations sentence-by-sentence: place a
  `[short source title](url-or-kb-reference)` citation in the same sentence,
  immediately after the claim it supports.
- Do not use bare bracketed URLs or paths like `[https://...]` or
  `[raw/docs/...]`; the bracket text should be a short source title.
- Do not place citations inside code blocks. Cite the prose sentence that
  introduces or explains the code instead.
- Do not put all citations only in a final "Sources" section. A short sources
  recap is okay only if the answer already has inline citations.
- Never cite `wiki/` or `generated/` KB paths: they are navigation indexes,
  not sources, and they do not resolve to citable source cards. When a wiki
  page led you to a claim, cite the underlying `raw/` page it references
  (open it with `run_kb_command` if you have not already).
- If retrieval results are weak or missing, say the topic is not well
  covered by the current knowledge base rather than guessing.
- Synthesize retrieved material into a clear teaching explanation. Do not
  paste tool output verbatim, except for code blocks that should be
  preserved as-is.
- Prefer a few solid paragraphs over shallow bullet lists.
- Include complete, runnable code blocks when code is relevant.
- End with a short invitation for a follow-up question.

The retrieval tool returns JSON with matched passages and source metadata."""

NO_KB_NOTE = """## Knowledge base disabled

The user deselected every knowledge-base source for this conversation, so the
corpus retrieval and KB browsing tools are unavailable. Answer from general
knowledge (and the web tools above, when available). Do not claim to have
searched the course corpus; mention briefly when an answer would be stronger
with course sources enabled."""

NO_KB_ANSWERING_RULES = """## Answering rules

- When you use web tools, cite sources with inline Markdown links: place a
  `[short source title](url)` citation in the same sentence, immediately
  after the claim it supports.
- Synthesize a clear teaching explanation. Prefer a few solid paragraphs over
  shallow bullet lists.
- Include complete, runnable code blocks when code is relevant.
- End with a short invitation for a follow-up question."""


def _provider_key(model_name: str) -> str:
    normalized = (model_name or "").strip()
    if ":" in normalized:
        return normalized.split(":", 1)[0]
    if normalized.startswith("gpt-"):
        return "openai"
    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith("gemini"):
        return "google-genai"
    return ""


def kb_agents_path() -> Path:
    return Path(os.getenv("AI_TUTOR_KB_AGENTS_PATH", DEFAULT_KB_AGENTS_PATH))


def load_kb_agents_instructions(path: Path | None = None) -> str:
    resolved = path or kb_agents_path()
    try:
        if not resolved.exists() or not resolved.is_file():
            return ""
        return resolved.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def ensure_kb_agents_instructions() -> str:
    """Load the Local KB Instructions, materializing the file if missing.

    `data/kb/AGENTS.md` is generated from an in-git template, so it can be
    written without the KB bundle download. Returns "" (with a loud log)
    only when even the template is unavailable.
    """
    instructions = load_kb_agents_instructions()
    if instructions:
        return instructions
    try:
        from .config import ensure_kb_agents_md

        ensure_kb_agents_md()
    except OSError as exc:
        logger.warning("Could not materialize KB agent instructions: %s", exc)
    instructions = load_kb_agents_instructions()
    if not instructions:
        logger.warning(
            "KB agent instructions unavailable; the system prompt will omit "
            "the Local KB section for this request."
        )
    return instructions


def build_system_prompt(
    model_name: str,
    enabled_tools: tuple[str, ...],
    kb_agents_instructions: str | None = None,
    include_local_tools: bool = True,
) -> str:
    provider = _provider_key(model_name)
    enabled = set(enabled_tools)
    tool_lines = [RETRIEVAL_TOOL_LINE, *KB_TOOL_LINES] if include_local_tools else []
    usage_sections: list[str] = []
    provider_web_tools = WEB_TOOL_LINES.get(provider, {})
    provider_web_usage = WEB_USAGE_SECTIONS.get(provider, {})
    for key in ("web_search", "url_context", "web_fetch"):
        if key in enabled and key in provider_web_tools:
            tool_lines.append(provider_web_tools[key])
            usage_sections.append(provider_web_usage[key])

    parts = [BASE_PROMPT_HEADER]
    if tool_lines:
        if len(tool_lines) == 1:
            intro = "You have one tool available:"
        else:
            intro = f"You have {len(tool_lines)} tools available:"
        parts.append(f"{intro}\n\n" + "\n".join(tool_lines))

    if include_local_tools:
        parts.append(RETRIEVAL_USAGE_SECTION)
        parts.append(KB_USAGE_SECTION)
    if usage_sections:
        parts.append(
            "## When to use web search / URL reading\n\n" + "\n\n".join(usage_sections)
        )
        if include_local_tools:
            parts.append(
                "Prefer `retrieve_tutor_context` first when the question is clearly about\n"
                "course material. Combine tools when it helps (e.g. retrieve corpus\n"
                "context, then search the web for the latest update)."
            )
    if not include_local_tools:
        # The user deselected every source: the prompt must not describe or
        # instruct tools the agent does not have this turn.
        parts.append(NO_KB_NOTE)
        parts.append(NO_KB_ANSWERING_RULES)
        return "\n\n".join(parts) + "\n"
    if kb_agents_instructions is None:
        kb_agents_instructions = load_kb_agents_instructions()
    if kb_agents_instructions:
        parts.append(
            "## Local KB Instructions\n\n"
            "The following instructions are loaded from `data/kb/AGENTS.md` and "
            "describe the current local KB schema and workflow.\n\n"
            f"{kb_agents_instructions}"
        )
    parts.append(RETRIEVAL_CALL_STRATEGY)
    parts.append(ANSWERING_RULES)
    return "\n\n".join(parts) + "\n"
