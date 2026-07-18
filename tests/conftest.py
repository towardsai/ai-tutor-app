from __future__ import annotations

import os


# app.config loads the repository's .env during test collection. Without an
# explicit override, ordinary unit tests upload synthetic LangGraph and
# retriever runs into the production LangSmith project. Keep normal pytest
# hermetic; the opt-in live E2E mode deliberately retains tracing.
if os.getenv("RUN_LIVE_API_E2E") != "1":
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
