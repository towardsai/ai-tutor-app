---
title: AI Tutor
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

### AI Tutor Chatbot

An agentic RAG tutor for applied AI, LLMs, RAG, and Python: a Next.js frontend served by a FastAPI backend, with a LangChain/LangGraph agent core. See [AGENTS.md](./AGENTS.md) for the architecture map.

The live app is deployed on Hugging Face Spaces at: [AI Tutor Chatbot on Hugging Face](https://huggingface.co/spaces/towardsai-tutors/ai-tutor-chatbot) (prod).

**Deployment flow:** every push to `main` auto-deploys to the private dev Space ([ai-tutor](https://huggingface.co/spaces/towardsai-tutors/ai-tutor)) for verification; the prod Space is promoted manually via the "Deploy prod to Hugging Face" workflow in the Actions tab.

### Backend — Quick Start

1. Install dependencies (requires [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)):

   ```bash
   uv sync
   ```

2. Configure environment variables:

   ```bash
   cp .env.example .env  # then edit values
   ```

   The chat model is provider-agnostic, configured in `provider:model` format, for example `google-genai:gemini-3.5-flash`. Optional provider keys include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY`. Anthropic support is wired through the `anthropic` SDK, and Gemini support is wired through Google’s `google-genai` SDK.
   To trace requests in LangSmith, set `LANGSMITH_API_KEY`. The app enables tracing automatically when that key is present unless `LANGSMITH_TRACING=false` is set.

### LangSmith Agent Tracing

The chatbot is built with `langchain.agents.create_agent()`, so LangSmith can trace the LangGraph/LangChain run tree without extra dependencies. Add these values to `.env`:

```bash
LANGSMITH_API_KEY=ls_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ai-tutor-app
```

Each chat turn is traced as `ai-tutor-agent-turn` with metadata for the backend thread id, message id, selected sources, requested model, and effective tool list. Child runs capture the model calls and `retrieve_tutor_context` tool executions, including tool inputs and outputs. Provider-side tools such as Gemini `google_search` / `url_context` and Claude `web_search` / `web_fetch` are visible through the model request/response metadata that LangChain receives from those providers.

Tracing sends prompts, retrieved snippets, tool inputs, tool outputs, and model responses to LangSmith. Set `LANGSMITH_TRACING=false` to keep tracing disabled while leaving the key in your environment.

### Next.js Frontend — Quick Start

The Next.js frontend in [frontend](./frontend) talks to the FastAPI backend.

1. Start the Python API:

   ```bash
   uv run -m scripts.api
   ```

   If `8000` is already taken, bind another port instead:

   ```bash
   AI_TUTOR_API_PORT=8001 uv run -m scripts.api
   ```

2. In a second terminal, install the frontend dependencies:

   ```bash
   cd frontend
   npm install
   ```

3. Configure the frontend API target:

   ```bash
   cp .env.example .env.local
   ```

   The default points at `http://127.0.0.1:8000`, which matches the local FastAPI app.
   If you override the backend port, update `NEXT_PUBLIC_AI_TUTOR_API_BASE_URL` to match.

4. Run the frontend:

   ```bash
   npm run dev
   ```

5. Open [http://localhost:3000](http://localhost:3000).

This frontend consumes:

- `GET /api/tools` — available models, tools, and the source picker (sources are nested in the response)
- `POST /api/chat` — the streaming chat endpoint (SSE, Vercel AI SDK UI-message protocol)

and renders sources, tool activity, and reasoning as separate UI elements rather than a single markdown block.

API notes:

- API clients should usually send only the new user message and continue the conversation with `threadId` (the `data-thread` part of the stream carries it; send it back on the next request).
- The source filter is request-scoped, so you can keep the same `threadId` while changing sources between turns.
- Sending an empty `threadId` starts a new backend conversation.

### Knowledge Base (file-based)

Alongside vector retrieval, the agent can browse a local, file-based knowledge base under `data/kb/` like a filesystem (read-only `rg`/`grep`/`cat`/… via the `run_kb_command` tool). It has three layers: `raw/` (immutable corpus mirrors), `wiki/` (an LLM-maintained synthesis/navigation layer), and `generated/` (machine indexes for manifests, headings, and symbols).

This is a deliberate take on [Andrej Karpathy's "LLM wiki" idea](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — a persistent, compounding wiki an LLM maintains over immutable sources, rather than re-deriving knowledge from scratch on every query. See [data/kb/MAINTAINER.md](./data/kb/MAINTAINER.md) for the design and the wiki-maintainer workflow, and [AGENTS.md](./AGENTS.md) for the overall app architecture.

### Rebuild Local Retrieval Index

After updating the JSONL corpus, rebuild the local Chroma index with:

```bash
uv run -m data.scraping_scripts.build_kb_artifacts
uv run -m data.scraping_scripts.update_kb_wiki
uv run -m data.scraping_scripts.add_context_to_nodes
uv run -m data.scraping_scripts.create_vector_stores all_sources
```

The KB commands generate browseable markdown, indexes, and wiki navigation pages for the agent. The context command builds the chunk manifest used by the workflows, and the vector command writes dense embeddings into the local Chroma database.
The vector-store build now shows embedding and Chroma upsert progress in the terminal.

### Updating Data Sources

For adding new courses or updating documentation:

- See the detailed instructions in [data/scraping_scripts/README.md](./data/scraping_scripts/README.md)
