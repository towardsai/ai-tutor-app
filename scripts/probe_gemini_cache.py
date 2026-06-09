"""Send the same prefix to Gemini twice and report cached_content_token_count.

Run with:
    uv run dotenv -f .env run -- python -m scripts.probe_gemini_cache
"""

from __future__ import annotations

import os
import time

from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash"

# Build a stable prefix that comfortably exceeds the 1024-token Flash minimum
# for implicit caching. Repeat a paragraph until we have ~3k tokens worth.
SYSTEM = (
    "You are an AI teacher for applied AI, LLM, RAG, and Python topics. "
    "Your job is to answer student questions clearly and accurately. "
    "Ground claims in the provided context and cite sources inline."
)
PARAGRAPH = (
    "Retrieval-Augmented Generation (RAG) systems combine a retrieval step "
    "over a corpus with a generation step using a language model. The "
    "retrieval step selects the most relevant passages from a vector index, "
    "and the generation step conditions on those passages to produce a "
    "grounded answer. Common components include a chunker, an embedding "
    "model, a vector store, a retriever, a reranker, and a generation model. "
    "Evaluation typically considers faithfulness, answer relevance, and "
    "context precision and recall.\n\n"
)
LONG_CONTEXT = PARAGRAPH * 12  # ~3-4k tokens


def call(client: genai.Client, user_text: str) -> dict:
    start = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        config=types.GenerateContentConfig(system_instruction=SYSTEM),
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": LONG_CONTEXT},
                    {"text": user_text},
                ],
            }
        ],
    )
    elapsed = time.perf_counter() - start
    usage = response.usage_metadata
    return {
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "cached_tokens": getattr(usage, "cached_content_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "text_preview": (response.text or "")[:120],
    }


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in .env")

    client = genai.Client(api_key=api_key)

    print(f"Model: {MODEL}")
    print(
        "Stable prefix (system + LONG_CONTEXT) repeats. Only the final user_text differs."
    )
    print()

    for i in range(3):
        result = call(
            client, user_text=f"Question {i + 1}: what is RAG in one sentence?"
        )
        print(f"Call {i + 1}: {result}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
