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

3. Run:

   ```bash
   uv run -m scripts.main
   ```

   Starts the Gradio AI Tutor interface.

### Updating Data Sources

For adding new courses or updating documentation:

- See the detailed instructions in [data/scraping_scripts/README.md](./data/scraping_scripts/README.md)
