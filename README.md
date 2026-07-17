---
title: Towards AI's Chatbot Tutor
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

### Towards AI's Chatbot Tutor

An agentic RAG tutor for applied AI, LLMs, RAG, and Python: a Next.js frontend served by a FastAPI backend, with a LangChain/LangGraph agent core. See [AGENTS.md](./AGENTS.md) for the architecture map.

Built by [Louis-François Bouchard](https://www.linkedin.com/in/whats-ai/) ([X](https://x.com/Whats_AI) · [LinkedIn](https://www.linkedin.com/in/whats-ai/)), Omar Solano, and Samridhi Vaid at [Towards AI](https://towardsai.net).

The live app is deployed on Hugging Face Spaces at: [AI Tutor Chatbot on Hugging Face](https://huggingface.co/spaces/towardsai-tutors/ai-tutor-chatbot) (prod).

**Deployment flow:** every push to `main` (except docs/markdown-only and scraping-script-only changes) auto-deploys to the private dev Space ([ai-tutor](https://huggingface.co/spaces/towardsai-tutors/ai-tutor)) for verification; the prod Space is promoted manually via the "Deploy prod to Hugging Face" workflow in the Actions tab.

### Workshop, slides, and going deeper

This repo also backs our **AI Engineer (AIE)** workshop on building a production AI tutor, presented by Towards AI.

- Experiments and live demo: [Context engineering experiments](https://huggingface.co/spaces/towardsai-tutors/context-engineering-experiments), see the experiment results and try the chatbot.
- Slides: [Google Slide Deck](https://docs.google.com/presentation/d/1BVqX1h2DPyIDCEWUSNXSVFIPTQgbhM_ik4JBVaQ2f3k/edit?usp=sharing)
- Workshop recording: TBD (coming soon)

Want to build this AI tutor yourself, end to end? Our [**Full Stack AI Engineer course**](https://academy.towardsai.net/courses/beginner-to-advanced-llm-dev), by Towards AI, walks you through building this exact AI tutor from scratch, plus the broader skills the AI engineering role demands: prompt and context engineering, data pipelines, RAG from scratch then at scale, fine-tuning, agents, observability, and production deployment to Hugging Face Spaces.

92 lessons. Hands-on capstone projects. A certificate. And an active Discord community. The first 6 lessons are free.

[**Start here →**](https://academy.towardsai.net/courses/beginner-to-advanced-llm-dev)

<a href="https://academy.towardsai.net/courses/beginner-to-advanced-llm-dev"><img src="./assets/full-stack-ai-engineer-course.png" alt="Become a Certified AI Engineer — Full Stack AI Engineer course by Towards AI" width="400"></a>

### Backend — Quick Start

1. Install dependencies (requires [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)):

   ```bash
   uv sync
   ```

2. Configure environment variables:

   ```bash
   cp .env.example .env  # then edit values
   ```

   The chat model is provider-agnostic, configured in `provider:model` format. The default is `deepseek:deepseek-v4-flash`, which uses DeepSeek's first-party API and falls back in-app to `google-genai:gemini-2.5-flash` when a Gemini key is configured. Set `DEEPSEEK_API_KEY` plus `GEMINI_API_KEY` or `GOOGLE_API_KEY` for the default path. Optional provider keys include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `OPENROUTER_API_KEY` for non-default provider paths.
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
   uv run -m app.api
   ```

   If `8000` is already taken, bind another port instead:

   ```bash
   AI_TUTOR_API_PORT=8001 uv run -m app.api
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
- `memoryPreset` is an optional direct-API/eval override. The normal frontend omits it, so the server selects the production preset from the requested model: DeepSeek and Gemini use `prod_v2`; other providers use the immutable legacy `prod` preset.
- Memory-preset precedence is `memoryPreset` request value → `AI_TUTOR_MEMORY_PRESET` environment override → model-aware production selection. Leave the environment variable unset for normal production behavior; incompatible preset/model combinations return HTTP 422.

### Knowledge Base (file-based)

Alongside vector retrieval, the agent can browse a local, file-based knowledge base under `data/kb/` like a filesystem (read-only `rg`/`grep`/`cat`/… via the `run_kb_command` tool). It has three layers: `raw/` (immutable corpus mirrors), `wiki/` (an LLM-maintained synthesis/navigation layer), and `generated/` (machine indexes for manifests, headings, and symbols).

This is a deliberate take on [Andrej Karpathy's "LLM wiki" idea](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — a persistent, compounding wiki an LLM maintains over immutable sources, rather than re-deriving knowledge from scratch on every query. The design and wiki-maintainer workflow live in `data/kb/MAINTAINER.md` (not in git — it ships with the private HF KB bundle and is present locally after first start with `HF_TOKEN`); see [AGENTS.md](./AGENTS.md) for the overall app architecture.

### Rebuild Local Retrieval Index

After updating the JSONL corpus, rebuild the local Chroma index with:

```bash
uv run -m data.scraping_scripts.build_kb_artifacts
uv run -m data.scraping_scripts.update_kb_wiki
uv run -m data.scraping_scripts.add_context_to_nodes
uv run -m data.scraping_scripts.create_vector_stores all_sources
```

The KB commands generate browseable markdown, indexes, and wiki navigation pages for the agent. The context command adds Gemini-generated context to each chunk and writes `all_sources_contextual_nodes.pkl` (consumed by the vector build), and the vector command writes dense embeddings into the local Chroma database.
The vector-store build now shows embedding and Chroma upsert progress in the terminal.

### Updating Data Sources

For adding new courses or updating documentation:

- See the detailed instructions in [data/scraping_scripts/README.md](./data/scraping_scripts/README.md)
