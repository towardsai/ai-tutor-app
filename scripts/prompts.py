from __future__ import annotations


BASE_PROMPT_HEADER = """You are an AI teacher for applied AI, LLM, RAG, and Python topics.

Your job is to answer student questions clearly and accurately."""

RETRIEVAL_TOOL_LINE = (
    "- `retrieve_tutor_context` — retrieval over the course and documentation\n"
    "  corpus. Use this for anything that depends on course content."
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
- If retrieval results are weak or missing, say the topic is not well
  covered by the current knowledge base rather than guessing.
- Synthesize retrieved material into a clear teaching explanation. Do not
  paste tool output verbatim, except for code blocks that should be
  preserved as-is.
- Prefer a few solid paragraphs over shallow bullet lists.
- Include complete, runnable code blocks when code is relevant.
- End with a short invitation for a follow-up question.

The retrieval tool returns JSON with matched passages and source metadata."""


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


def build_system_prompt(model_name: str, enabled_tools: tuple[str, ...]) -> str:
    provider = _provider_key(model_name)
    enabled = set(enabled_tools)
    tool_lines = [RETRIEVAL_TOOL_LINE]
    usage_sections: list[str] = []
    provider_web_tools = WEB_TOOL_LINES.get(provider, {})
    provider_web_usage = WEB_USAGE_SECTIONS.get(provider, {})
    for key in ("web_search", "url_context", "web_fetch"):
        if key in enabled and key in provider_web_tools:
            tool_lines.append(provider_web_tools[key])
            usage_sections.append(provider_web_usage[key])

    if len(tool_lines) == 1:
        intro = "You have one tool available:"
    else:
        intro = f"You have {len(tool_lines)} tools available:"

    parts = [
        BASE_PROMPT_HEADER,
        f"{intro}\n\n" + "\n".join(tool_lines),
        RETRIEVAL_USAGE_SECTION,
    ]
    if usage_sections:
        parts.append("## When to use web search / URL reading\n\n" + "\n\n".join(usage_sections))
        parts.append(
            "Prefer `retrieve_tutor_context` first when the question is clearly about\n"
            "course material. Combine tools when it helps (e.g. retrieve corpus\n"
            "context, then search the web for the latest update)."
        )
    parts.append(RETRIEVAL_CALL_STRATEGY)
    parts.append(ANSWERING_RULES)
    return "\n\n".join(parts) + "\n"
