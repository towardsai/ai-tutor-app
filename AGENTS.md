# AI Tutor App — Agent Instructions

This is the **canonical, tool-agnostic** instruction file for the repo. `CLAUDE.md` pulls it in via `@AGENTS.md`; Codex and other agents read it directly. **Put new repo-wide guidance here** so every agent stays in sync. Keep this file a concise map — deep procedures live in the docs it points to.

## Project Overview

AI tutor for applied AI, LLMs, RAG, and Python. **Agentic RAG**: a LangChain/LangGraph agent grounds answers in a curated corpus of course + library docs, can browse a local file-based knowledge base, and (optionally) search the live web. Two frontends share one agent core:

- **Gradio** UI — `scripts/main.py` (the original chatbot).
- **Next.js** UI — `frontend/`, served by a **FastAPI** backend (`scripts/api.py`) that streams in the Vercel AI SDK UI-message protocol.

ChromaDB for vectors; Cohere for embeddings/rerank; chat model is provider-configurable (Gemini default, Anthropic, OpenAI). Python ≥3.13, managed with `uv`.

## Key URLs

- [GitHub repo](https://github.com/towardsai/ai-tutor-app)
- [Live demo (Gradio)](https://huggingface.co/spaces/towardsai-tutors/ai-tutor-chatbot)
- [Vector DB + KB bundle](https://huggingface.co/datasets/towardsai-tutors/ai-tutor-vector-db) · [Private raw JSONL data](https://huggingface.co/datasets/towardsai-tutors/ai-tutor-data)

## Where things live

| Concern | File(s) |
|---|---|
| Agent core (`build_agent`, `stream_chat`) | `scripts/chat_service.py` |
| System prompt assembly | `scripts/prompts.py` |
| Hybrid retrieval | `scripts/chroma_rag.py` |
| KB browsing sandbox + citation resolution | `scripts/kb_shell.py`, `scripts/kb_manifest.py` |
| FastAPI server (`/api/chat`, `/api/tools`, `/healthz`) | `scripts/api.py` |
| Gradio app + renderer | `scripts/main.py`, `scripts/gradio_presenter.py` |
| Paths, models, startup downloads | `scripts/setup.py` |
| **Sources — single source of truth** | `data/scraping_scripts/source_registry.py` |
| Tracing (LangSmith + Logfire) | `scripts/agent_tracing.py`, `scripts/setup.py` |
| Data pipeline / workflows (deep guide) | `data/scraping_scripts/README.md` |
| KB design + wiki maintainer workflow (deep guide) | `data/kb/MAINTAINER.md` |

## Architecture in brief

The agent is built with `langchain.agents.create_agent()` (LangGraph), an `InMemorySaver` checkpointer keyed by `thread_id`, and middlewares for context-editing, summarization, and source preference. `stream_chat()` is the single entry point both frontends call; it yields typed `ChatEvent`s. It always exposes two custom tools, plus provider-native web tools when enabled:

- **`retrieve_tutor_context(query)`** — hybrid RAG over the corpus, scoped to the user's selected sources.
- **`run_kb_command(...)`** — read-only KB file browsing (see below).
- **Gemini**: `google_search`, `url_context`. **Anthropic**: `web_search`, `web_fetch`.

Final-answer inline citations are resolved against current-turn evidence + the KB manifest into trusted source cards (`scripts/kb_manifest.py`).

**Retrieval** (`scripts/chroma_rag.py`): dense (Cohere `embed-v4.0`) + BM25 → Reciprocal Rank Fusion → Cohere rerank → token budget. See the file for the exact top-k / score constants.

**Corpus → searchable** lifecycle: markdown → `process_md_files.py` → per-source JSONL → `all_sources_data.jsonl` → (`add_context_to_nodes.py`, **Gemini**) → `*_contextual_nodes.pkl` → `create_vector_stores.py` → ChromaDB. Separately, `all_sources_data.jsonl` → `build_kb_artifacts.py` + `update_kb_wiki.py` → `data/kb/`.

## Knowledge base (KB) browsing tool

The agent's second grounding mechanism: instead of only top-k retrieval, it browses the corpus like a filesystem via `run_kb_command` — a sandboxed, **read-only** shell over `data/kb/` (allowed: `rg grep find ls sed head cat wc`; no pipes/redirects/network/writes; path-jailed to `data/kb/`; per-turn command budget). Sandbox lives in `scripts/kb_shell.py`.

`data/kb/` (a gitignored build artifact, downloaded on first start) has three layers:

- `raw/` — read-only markdown mirrors of the corpus (`docs/<source>/…`, `courses/<source>/…`).
- `wiki/` — LLM-maintained synthesis/navigation (`index.md`, `frameworks/`, `courses/`, `topics/`, `recipes/`, `errors/`, `log.md`).
- `generated/` — machine indexes (`corpus_manifest.jsonl`, `headings.jsonl`, `symbols.tsv`).

This deliberately implements **[Karpathy's "LLM wiki"](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)** idea — a persistent, compounding wiki an LLM maintains over immutable sources, instead of re-deriving knowledge per query. **`data/kb/MAINTAINER.md` owns the full design and the maintainer (ingest/curate/lint) workflow — read it before touching KB structure.**

Runtime guidance the agent follows is in `data/kb/AGENTS.md` (injected into the system prompt). It is **generated** — edit the template at `data/scraping_scripts/kb_agents_template.md`, never `data/kb/AGENTS.md` directly. Don't confuse `data/kb/AGENTS.md` (runtime KB rules) with this root file (repo dev guidance).

## Sources & config

`data/scraping_scripts/source_registry.py` is the **single source of truth** for sources (`SOURCE_CONFIGS`, key groupings, UI labels, defaults); `scripts/setup.py` re-exports them and both frontends derive the picker from it. Docs sources ingest via the GitHub API or `llms.txt`; course sources are Notion exports. To add a source: add it to the registry (+ the relevant grouping tuples), then run the matching workflow — no separate UI edit needed. Models live in `setup.AVAILABLE_MODELS` (default `google-genai:gemini-3.5-flash`; also Claude Haiku 4.5; OpenAI supported in code).

## Running locally

```bash
uv sync && cp .env.example .env      # then fill in keys
uv run -m scripts.main               # Gradio UI (:7860)
uv run -m scripts.api                # FastAPI backend (:8000; override AI_TUTOR_API_PORT/PORT)
# Next.js frontend (needs the API running):
cd frontend && npm install && cp .env.example .env.local && npm run dev   # :3000
```

First start downloads the vector-db/KB bundle from HF if missing (`HF_TOKEN`). The frontend is a static export (`output: 'export'`); `npm run build` emits `frontend/out`, which `scripts/api.py` mounts at `/`.

Test & lint: `uv run pytest` · `uv run ruff check .`

## Data update workflows

Run from the repo root; by default they rebuild KB artifacts and upload to HF. **Full guide: `data/scraping_scripts/README.md`.**

```bash
uv run -m data.scraping_scripts.add_course_workflow --courses NAME [NAME ...]   # note: --courses (plural)
uv run -m data.scraping_scripts.update_docs_workflow [--sources transformers peft ...]
uv run -m data.scraping_scripts.retire_source_workflow --sources KEY [--dry-run | --yes]
```

`update_docs_workflow` also performs a from-scratch rebuild if `data/kb` / `data/chroma-db-all_sources` are deleted. Common flags: `--process-all-context` (default is new-content-only), `--skip-kb`, `--skip-vectors`, `--skip-upload`, `--skip-data-upload`.

## Environment variables

Chat runtime: `COHERE_API_KEY` (retrieval), one chat-model provider key (`GEMINI_API_KEY`/`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`), `HF_TOKEN` (first-start download). Optional: `MONGODB_URI` (logging), `LANGSMITH_*` (tracing), `AI_TUTOR_API_PORT`/`HOST`/`CORS_ALLOW_ORIGINS`, `AI_TUTOR_KB_DIR`, `NEXT_PUBLIC_AI_TUTOR_API_BASE_URL`. Data workflows also need `GITHUB_TOKEN` and `GEMINI_API_KEY`/`GOOGLE_API_KEY` (context generation). See `.env.example`.

## Deployment

`.github/workflows/sync-to-hf.yml` force-pushes to two HF Spaces on every push to `main`: **`ai-tutor`** (`Dockerfile`: FastAPI + Next.js export) and **`ai-tutor-chatbot`** (`Dockerfile.gradio`). Both install `ripgrep` for `run_kb_command` and run on :7860.

## Conventions

- **No em-dashes in frontend user-facing text.** Use a comma, parentheses, a colon, or two sentences instead. This covers every string the UI renders (Next.js components and Gradio): tool descriptions, popovers, labels, `title` tooltips, placeholders, empty states. This file and other docs are exempt.
- **Decide frontend vs. backend ownership before changing behavior, and fix it on the side that owns the data.** The backend is the single source of truth for data shape and meaning; the frontend renders what it receives instead of reshaping it. If a field is empty, that should be because the server wrote it empty, not because the UI stripped it. For example, a source with no library version must get `version: null` from `capture_source_versions.py`; it should never be filtered out client-side. Reshaping data in the frontend causes drift between the two frontends and ambiguity about why a value is missing (did the server omit it, or did the client hide it?).

## Gotchas

- **Context generation uses Gemini**; embeddings/rerank use **Cohere**; the chat model is provider-configurable. OpenAI is required only when explicitly selected.
- `data/kb/` and `data/chroma-db-all_sources/` are build artifacts — never commit or hand-edit; regenerate or re-download.
- Two `AGENTS.md` files: this root one (repo dev guidance) vs `data/kb/AGENTS.md` (generated runtime KB rules).
- Source config lives in `source_registry.py`, not `setup.py` / `process_md_files.py` (older docs were wrong).
