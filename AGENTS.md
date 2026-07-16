# AI Tutor App — Agent Instructions

This is the **canonical, tool-agnostic** instruction file for the repo. `CLAUDE.md` pulls it in via `@AGENTS.md`; Codex and other agents read it directly. **Put new repo-wide guidance here** so every agent stays in sync. Keep this file a concise map — deep procedures live in the docs it points to.

## Project Overview

AI tutor for applied AI, LLMs, RAG, and Python. **Agentic RAG**: a LangChain/LangGraph agent grounds answers in a curated corpus of course + library docs, can browse a local file-based knowledge base, and (optionally) search the live web. One frontend: a **Next.js** UI (`frontend/`), served by a **FastAPI** backend (`app/api.py`) that streams in the Vercel AI SDK UI-message protocol. (A Gradio UI existed historically; it was removed to keep one rendering path.)

ChromaDB for vectors; Cohere for embeddings/rerank; chat model is provider-configurable (DeepSeek V4 Flash through the first-party API by default, with a rescue-only in-app fallback to Gemini 2.5 Flash when a Gemini key is set; Claude Haiku 4.5 is also selectable; OpenAI and OpenRouter-compatible models are supported in code). **A conversation cannot change model mid-thread** — checkpoints store provider-native message blocks that no other provider can replay; see `build_chat_model` in `app/chat_service.py`. Python ≥3.13, managed with `uv`.

## Key URLs

- [GitHub repo](https://github.com/towardsai/ai-tutor-app)
- [Live demo — prod Space](https://huggingface.co/spaces/towardsai-tutors/ai-tutor-chatbot) · [Dev Space (private)](https://huggingface.co/spaces/towardsai-tutors/ai-tutor)
- [Vector DB + KB bundle](https://huggingface.co/datasets/towardsai-tutors/ai-tutor-vector-db) (private; full corpus incl. courses) · [Public docs-only bundle](https://huggingface.co/datasets/towardsai-tutors/ai-tutor-vector-db-public) (no token needed; cold-start fallback) · [Private raw JSONL data](https://huggingface.co/datasets/towardsai-tutors/ai-tutor-data)

## Where things live

| Concern | File(s) |
|---|---|
| Agent core (`build_agent`, `stream_chat`) | `app/chat_service.py` |
| System prompt assembly | `app/prompts.py` |
| Hybrid retrieval | `app/chroma_rag.py` |
| KB browsing sandbox + citation resolution | `app/kb_shell.py`, `app/kb_manifest.py` |
| FastAPI server (`/api/chat`, `/api/tools`, `/healthz`) | `app/api.py` |
| Paths, models, startup downloads | `app/config.py` |
| **Sources — single source of truth** | `data/scraping_scripts/source_registry.py` |
| Agent tracing (LangSmith) + server logging (stdlib `logging` → stdout) | `app/agent_tracing.py`, `app/config.py` |
| Memory/context presets + per-turn telemetry (`context_stats`) | `app/memory_presets.py`, `app/telemetry.py` |
| Eval harness (run/grade/report batteries) | `evals/`, entry doc `evals.md` |
| Eval dataset schemas + glossary | `data/eval/README.md` |
| Data pipeline / workflows (deep guide) | `data/scraping_scripts/README.md` |
| KB design + wiki maintainer workflow (deep guide) | `data/kb/MAINTAINER.md` (not in git — ships with the private KB bundle, present locally after first start with `HF_TOKEN`) |

## Architecture in brief

The agent is built with `langchain.agents.create_agent()` (LangGraph), an `InMemorySaver` checkpointer keyed by `thread_id`, and middlewares assembled from the selected **memory preset** (`app/memory_presets.py`: context-editing, summarization, optional long-term student-profile memory) plus source preference. `stream_chat()` is the single entry point the API calls; it yields typed `ChatEvent`s that `app/api.py` encodes into the AI SDK UI-message stream, ending each turn with a `context_stats` telemetry event (tokens incl. cache buckets, est. cost, TTFT, compaction-trigger counts — independent of LangSmith). It always exposes two custom tools, plus provider-native web tools when enabled:

- **`retrieve_tutor_context(query)`** — hybrid RAG over the corpus, scoped to the user's selected sources.
- **`run_kb_command(...)`** — read-only KB file browsing (see below).
- **Gemini**: `google_search`, `url_context`. **Anthropic**: `web_search`, `web_fetch`.

Final-answer inline citations are resolved against current-turn evidence + the KB manifest into trusted source cards (`app/kb_manifest.py`).

**Retrieval** (`app/chroma_rag.py`): dense (Cohere `embed-v4.0`) + BM25 → Reciprocal Rank Fusion → Cohere rerank → token budget. See the file for the exact top-k / score constants.

**Corpus → searchable** lifecycle: markdown → `process_md_files.py` → per-source JSONL → `all_sources_data.jsonl` → (`add_context_to_nodes.py`, **Gemini**) → `*_contextual_nodes.pkl` → `create_vector_stores.py` → ChromaDB. Separately, `all_sources_data.jsonl` → `build_kb_artifacts.py` + `update_kb_wiki.py` → `data/kb/`.

## Knowledge base (KB) browsing tool

The agent's second grounding mechanism: instead of only top-k retrieval, it browses the corpus like a filesystem via `run_kb_command` — a sandboxed, **read-only** shell over `data/kb/` (allowed: `rg grep find ls sed head cat wc`; no pipes/redirects/network/writes; path-jailed to `data/kb/`; per-turn command budget). Sandbox lives in `app/kb_shell.py`.

`data/kb/` (a gitignored build artifact, downloaded on first start) has three layers:

- `raw/` — read-only markdown mirrors of the corpus (`docs/<source>/…`, `courses/<source>/…`).
- `wiki/` — LLM-maintained synthesis/navigation (`index.md`, `frameworks/`, `courses/`, `topics/`, `recipes/`, `errors/`, `log.md`).
- `generated/` — machine indexes (`corpus_manifest.jsonl`, `headings.jsonl`, `symbols.tsv`).

This deliberately implements **[Karpathy's "LLM wiki"](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)** idea — a persistent, compounding wiki an LLM maintains over immutable sources, instead of re-deriving knowledge per query. **`data/kb/MAINTAINER.md` owns the full design and the maintainer (ingest/curate/lint) workflow — read it before touching KB structure** (not in git: it ships with the private KB bundle, so it is on disk after first start with `HF_TOKEN`).

Runtime guidance the agent follows is in `data/kb/AGENTS.md` (injected into the system prompt). It is **generated** — edit the template at `data/scraping_scripts/kb_agents_template.md`, never `data/kb/AGENTS.md` directly. Don't confuse `data/kb/AGENTS.md` (runtime KB rules) with this root file (repo dev guidance).

## Sources & config

`data/scraping_scripts/source_registry.py` is the **single source of truth** for sources (`SOURCE_CONFIGS`, key groupings, UI labels, defaults); `app/config.py` re-exports them and the frontend derives the picker from it (via `/api/tools`). Docs sources ingest via the GitHub API or `llms.txt`; course sources are Notion exports. To add a source: add it to the registry (+ the relevant grouping tuples), then run the matching workflow — no separate UI edit needed. Models live in `config.AVAILABLE_MODELS` (default `deepseek:deepseek-v4-flash`; Claude Haiku 4.5 is also selectable; OpenAI is supported in code). **`GEMINI_FALLBACK_MODEL_NAME` (`google-genai:gemini-2.5-flash`) is deliberately NOT in `AVAILABLE_MODELS`**: it serves real traffic as the DeepSeek rescue path but must never be user-selectable, because pre-Gemini-3 models cannot combine Gemini's built-in web tools with our two custom tools (that needs Gemini 3+ "tool context circulation"), so selecting it would be one web-search toggle away from a 400. The fallback is safe only because it never receives web tools. `build_chat_model` in `app/chat_service.py` accepts `deepseek:` for the first-party API, `openrouter:` for compatible experiment models, and `ollama:` for local SLM experiments, with pricing in `app/telemetry.MODEL_PRICING`; for the default DeepSeek model it also wires an in-app fallback to `google-genai:gemini-2.5-flash` whenever a Gemini key is configured (and substitutes it outright if `DEEPSEEK_API_KEY` is missing).

## Running locally

```bash
uv sync && cp .env.example .env      # then fill in keys
uv run -m app.api                # FastAPI backend (:8000; override AI_TUTOR_API_PORT/PORT)
# Next.js frontend (needs the API running):
cd frontend && npm install && cp .env.example .env.local && npm run dev   # :3000
```

First start downloads the vector-db/KB bundle from HF if missing (`HF_TOKEN`). The frontend is a static export (`output: 'export'`); `npm run build` emits `frontend/out`, which `app/api.py` mounts at `/`.

Test, lint & format: `uv run pytest` · `uv run ruff check .` · `uv run ruff format .`. CI (`.github/workflows/ci.yml`) enforces all three on PRs/pushes to `main`; for local auto-fix on commit, run `uv run pre-commit install` once.

## Data update workflows

Run from the repo root; by default they rebuild KB artifacts and upload to HF. **Full guide: `data/scraping_scripts/README.md`.**

```bash
uv run -m data.scraping_scripts.add_course_workflow --courses NAME [NAME ...]   # note: --courses (plural)
uv run -m data.scraping_scripts.update_docs_workflow [--sources transformers peft ...]
uv run -m data.scraping_scripts.retire_source_workflow --sources KEY [--dry-run | --yes]
```

`update_docs_workflow` also performs a from-scratch rebuild if `data/kb` / `data/chroma-db-all_sources` are deleted. Common flags: `--process-all-context` (default is new-content-only), `--skip-kb`, `--skip-vectors`, `--skip-upload`, `--skip-data-upload`.

## Evaluation

**`evals.md` is the entry point** (what we evaluate, the datasets, results, remaining work); `evals/` is the harness (`run_battery` → `grade` → `report`, plus `check_triggers`, the blinded `handgrade_workbook`, the paired lockstep runner `run_compaction_experiment`, and the subagent-judge grading pipeline `grading_prep` → `grade_workflow.js`/`run_subscription_grading` → `grading_merge`). Eval datasets and run results contain **real student text**: they are gitignored and ship via the private `ai-tutor-data` HF dataset (`eval/`, `eval_runs/`) — download snippet in `evals.md`. Runs cost real API money (a 4-preset bake-off ≈ $323; the full eval program ≈ $590 across all Gemini 3.5 Flash runs); grading and reporting re-run offline from saved bundles for free. **Running or contributing an experiment? Follow `evals/contributing.md`** (run → upload data to HF → record an `F<N>` finding → merge; with a definition-of-done gate).

## Environment variables

Chat runtime: `COHERE_API_KEY` (retrieval; always required), `DEEPSEEK_API_KEY` for the default model plus `GEMINI_API_KEY`/`GOOGLE_API_KEY` to enable its in-app Gemini 2.5 Flash fallback, and the corresponding provider key when selecting another model (`GEMINI_API_KEY`/`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`; `OPENROUTER_API_KEY` is used by OpenRouter experiment models), plus `HF_TOKEN` for first-start download of the full private bundle. Without `HF_TOKEN`, cold start falls back to the public docs-only bundle (documentation sources only, course content hidden). Optional: `LANGSMITH_*` (tracing), `AI_TUTOR_API_PORT`/`HOST`/`CORS_ALLOW_ORIGINS`, `AI_TUTOR_KB_DIR`, `AI_TUTOR_MEMORY_PRESET` (default memory preset; see `app/memory_presets.py`), `NEXT_PUBLIC_AI_TUTOR_API_BASE_URL`. Data workflows also need `GITHUB_TOKEN` and `GEMINI_API_KEY`/`GOOGLE_API_KEY` (context generation). See `.env.example`.

## Deployment

Both HF Spaces run the same image (`Dockerfile`: FastAPI + Next.js static export, `ripgrep` installed for `run_kb_command`, port :7860), in a dev → prod flow:

- **Dev — `ai-tutor`** (private): `.github/workflows/sync-to-hf.yml` force-pushes on every push to `main` (docs/markdown-only and scraping-script-only pushes are skipped via `paths-ignore`). Verify changes here first.
- **Prod — `ai-tutor-chatbot`** (public): `.github/workflows/deploy-prod-to-hf.yml`, **manual trigger only** (Actions tab → "Deploy prod to Hugging Face" → Run workflow).

Both Spaces need the same runtime secrets (`COHERE_API_KEY`, model provider key, `HF_TOKEN`, optional `LANGSMITH_*`) configured in their HF settings.

## Conventions

- **No em-dashes in frontend user-facing text.** Use a comma, parentheses, a colon, or two sentences instead. This covers every string the UI renders (Next.js components): tool descriptions, popovers, labels, `title` tooltips, placeholders, empty states. This file and other docs are exempt.
- **Decide frontend vs. backend ownership before changing behavior, and fix it on the side that owns the data.** The backend is the single source of truth for data shape and meaning; the frontend renders what it receives instead of reshaping it. If a field is empty, that should be because the server wrote it empty, not because the UI stripped it. For example, a source with no library version must get `version: null` from `capture_source_versions.py`; it should never be filtered out client-side. Reshaping data in the frontend causes ambiguity about why a value is missing (did the server omit it, or did the client hide it?).

## Gotchas

- **One thread is one provider — a conversation cannot mix models while keeping tool outputs and thought signatures.** A thread's checkpoint stores *provider-native* messages: Gemini reasoning parts carry thought signatures, Anthropic thinking blocks carry a required cryptographic `signature`, and each provider's server-side tool calls are its own block types. Replaying one provider's history to another is a hard 400, not a degraded answer — verified both ways (Gemini history → DeepSeek 400s; Gemini history → Anthropic 400s with `messages.1.content.0.thinking.signature: Field required`). So you cannot "just send the payload to another provider": a fallback, a retry-on-another-provider, or mid-conversation switching **must not** be wired at the model/client layer. The only safe path is `sync_thread_with_history`'s branch to a fresh thread seeded with plain text, which drops the unportable state on purpose. The frontend model picker locks after the first message for this reason. The backend enforces it independently: `sync_thread_with_history` records which provider actually served each turn (`_THREAD_PROVIDERS`, taken from the served model, since the DeepSeek fallback means served ≠ requested) and branches to a fresh plain-text thread when the next turn's provider differs. That contains the fallback's one wart — a rescued turn checkpoints Gemini-shaped messages, which would otherwise strand the thread on Gemini at ~10x the price forever.
- **Provider-native web tools are bound from the requested model, and not every model can mix them with our custom tools.** Gemini needs "tool context circulation" (Gemini 3+) to combine `google_search`/`url_context` with function tools, so pre-3 Gemini gets no web toggles (`supports_gemini_tool_combination`). Anthropic combines them natively, but `allowed_callers: ["direct"]` is load-bearing — without it Haiku 4.5 400s on programmatic tool calling.
- **Context generation uses Gemini**; embeddings/rerank use **Cohere**; the chat model is provider-configurable. OpenAI is required only when explicitly selected.
- `data/kb/` and `data/chroma-db-all_sources/` are build artifacts — never commit or hand-edit; regenerate or re-download.
- Two `AGENTS.md` files: this root one (repo dev guidance) vs `data/kb/AGENTS.md` (generated runtime KB rules).
- Source config lives in `source_registry.py`, not `app/config.py` / `process_md_files.py` (older docs were wrong).
- **Eval data must never enter git.** `data/eval/` and `runs/` contain real student text; pushes to `main` get force-pushed to HF Spaces (prod is **public**), so a committed data file becomes world-readable on the next deploy. The gitignore rules covering `*.jsonl`, `data/eval/review_batches/`, `data/eval/review_log_v1.md`, and `runs/` are load-bearing — never weaken them; share via the private `ai-tutor-data` dataset instead.
