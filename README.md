---
title: AI Tutor Chatbot
emoji: 🧑🏻‍🏫
colorFrom: gray
colorTo: pink
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

3. Run:

   ```bash
   uv run -m scripts.main
   ```

   Starts the Gradio AI Tutor interface.

### Rebuild Local Retrieval Index

After updating the JSONL corpus, rebuild the local Chroma index with:

```bash
uv run -m data.scraping_scripts.add_context_to_nodes
uv run -m data.scraping_scripts.create_vector_stores all_sources
```

The first command now builds the chunk manifest used by the workflows, and the second command writes dense embeddings into the local Chroma database.
The vector-store build now shows embedding and Chroma upsert progress in the terminal.

### Updating Data Sources

For adding new courses or updating documentation:

- See the detailed instructions in [data/scraping_scripts/README.md](./data/scraping_scripts/README.md)
