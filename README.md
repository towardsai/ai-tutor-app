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

### Installation (for Gradio UI)

1. **Install dependencies with [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods):**

   ```bash
   uv sync
   ```

### Usage (for Gradio UI)

1. **Set environment variables:**

- copy .env.example to a new .env file and fill in the values

2. **Run the application:**

   ```bash
   uv run scripts/main.py
   ```

   This command starts the Gradio interface for the AI Tutor chatbot.

### Updating Data Sources

This application uses a RAG (Retrieval Augmented Generation) system with multiple data sources, including documentation and courses. To update these sources:

1. **For adding new courses or updating documentation:**
   - See the detailed instructions in [data/scraping_scripts/README.md](./data/scraping_scripts/README.md)
   - Python scripts are available for both course addition and documentation updates

