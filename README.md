---
title: AI Tutor
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

### Gradio UI Chatbot

A Gradio UI for the chatbot is available in [scripts/main.py](./scripts/main.py).

The Gradio demo is deployed on Hugging Face Spaces at: [AI Tutor Chatbot on Hugging Face](https://huggingface.co/spaces/towardsai-tutors/ai-tutor-chatbot).

**Note:** A GitHub Action automatically deploys the Gradio demo when changes are pushed to the main branch (excluding documentation and scripts in the `data/scraping_scripts` directory).

### Gradio UI — Quick Start

1. Install dependencies (requires [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)):

   ```bash
   uv sync
   ```

2. Configure environment variables:

   ```bash
   cp .env.example .env  # then edit values
   ```

   The chat model is provider-agnostic. Use the UI field in `provider:model` format, for example `openai:gpt-5.4-mini`. Optional provider keys include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY`. Anthropic support is wired through the `anthropic` SDK, and Gemini support is wired through Google’s `google-genai` SDK.
   To trace requests in LangSmith, set `LANGSMITH_API_KEY`. The app enables tracing automatically when that key is present unless `LANGSMITH_TRACING=false` is set.

3. Run:

   ```bash
   uv run -m scripts.main
   ```

   Starts the Gradio AI Tutor interface.

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

The repo now also includes a separate Next.js frontend in [frontend](./frontend) that talks to the FastAPI backend instead of the Gradio transport.

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

- `GET /api/sources`
- `POST /api/chat`

and renders sources, tool activity, and reasoning as separate UI elements rather than a single markdown block.

### Gradio API

The chat endpoint is exposed as `chat`, so the API flow is:

```bash
POST /gradio_api/call/chat
GET /gradio_api/call/chat/{event_id}
```

The `POST` body must send `data` in this exact order:

1. User message: string
2. History: array
3. Sources: array of source labels exactly as shown in the UI
4. Model: `provider:model` string
5. Show Gemini thoughts: boolean
6. Thread ID: string, empty for a new conversation

Example first turn:

```bash
curl -s http://127.0.0.1:7860/gradio_api/call/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "data": [
      "What is LoRA?",
      [],
      ["PEFT Docs", "Transformers Docs"],
      "openai:gpt-4o-mini",
      false,
      ""
    ]
  }'
```

That returns an `event_id`. Open the server-sent event stream with:

```bash
curl -N http://127.0.0.1:7860/gradio_api/call/chat/<event_id>
```

The streamed payloads look like this:

```json
["Partial assistant text", null, "thread-id"]
```

- `data[0]`: current streamed assistant text
- `data[1]`: Gradio's hidden state placeholder, ignore this
- `data[2]`: `thread_id` to reuse on the next turn

Example follow-up turn on the same conversation:

```bash
curl -s http://127.0.0.1:7860/gradio_api/call/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "data": [
      "How is it different from adapters?",
      [],
      ["PEFT Docs", "Transformers Docs"],
      "openai:gpt-4o-mini",
      false,
      "thread-id-from-previous-response"
    ]
  }'
```

Notes:

- API clients should usually send `[]` for history and continue the conversation with `thread_id`.
- The source filter is request-scoped, so you can keep the same `thread_id` while changing sources between turns.
- Sending an empty `thread_id` starts a new backend conversation.

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
