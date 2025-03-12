---
title: AI Tutor Chatbot
emoji: 🧑🏻‍🏫
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 5.20.1
app_file: scripts/main.py
pinned: false
---

### Gradio UI Chatbot

A Gradio UI for the chatbot is available in [scripts/main.py](./scripts/main.py).

The Gradio demo is deployed on Hugging Face Spaces at: [AI Tutor Chatbot on Hugging Face](https://huggingface.co/spaces/towardsai-tutors/ai-tutor-chatbot).

**Note:** A GitHub Action automatically deploys the Gradio demo when changes are pushed to the main branch (excluding documentation and scripts in the `data/scraping_scripts` directory).

### Installation (for Gradio UI)

1. **Create a new Python environment:**

   ```bash
   python -m venv .venv
   ```

2. **Activate the environment:**

   For macOS and Linux:

   ```bash
   source .venv/bin/activate
   ```

   For Windows:

   ```bash
   .venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Usage (for Gradio UI)

1. **Set environment variables:**

   Before running the application, set up the required API keys:

   For macOS and Linux:

   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   export COHERE_API_KEY=your_cohere_api_key_here
   ```

   For Windows:

   ```bash
   set OPENAI_API_KEY=your_openai_api_key_here
   set COHERE_API_KEY=your_cohere_api_key_here
   ```

2. **Run the application:**

   ```bash
   python scripts/main.py
   ```

   This command starts the Gradio interface for the AI Tutor chatbot.

### Updating Data Sources

This application uses a RAG (Retrieval Augmented Generation) system with multiple data sources, including documentation and courses. To update these sources:

1. **For adding new courses or updating documentation:**
   - See the detailed instructions in [data/scraping_scripts/README.md](./data/scraping_scripts/README.md)
   - Automated workflows are available for both course addition and documentation updates
   
2. **Available workflows:**
   - `add_course_workflow.py` - For adding new course content
   - `update_docs_workflow.py` - For updating documentation from GitHub repositories
   - `upload_data_to_hf.py` - For uploading data files to HuggingFace

These scripts streamline the process of adding new content to the AI Tutor and ensure consistency across team members.
