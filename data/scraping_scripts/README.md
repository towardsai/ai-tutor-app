# AI Tutor App Data Workflows

This directory contains scripts for managing the AI Tutor App's data pipeline.

## Workflow Scripts

### 1. Adding a New Course

To add a new course to the AI Tutor:

```bash
python add_course_workflow.py --course [COURSE_NAME]
```

This will guide you through the complete process:

1. Process markdown files from the Notion export
2. Prompt you to manually add URLs to the course content
3. Merge the course data into the main dataset
4. Add contextual information to document nodes
5. Create vector stores
6. Upload databases to HuggingFace
7. Update UI configuration

**Requirements before running:**

- The course name must be properly configured in `process_md_files.py` under `SOURCE_CONFIGS`
- Course markdown files must be placed in the directory specified in the configuration
- You must have access to the live course platform to add URLs `https://academy.towardsai.net/enrollments`

### 2. Updating Documentation via GitHub API

To update library documentation from GitHub repositories:

```bash
python update_docs_workflow.py
```

This will update all supported documentation sources. You can also specify specific sources:

```bash
python update_docs_workflow.py --sources transformers peft
```

The workflow includes:

1. Downloading documentation from GitHub using the API
2. Processing markdown files to create JSONL data
3. Adding contextual information to document nodes
4. Creating vector stores
5. Uploading vector db and new JSONL files to HuggingFace

## Individual Components

If you need to run specific steps individually:

- **GitHub to Markdown**: `github_to_markdown_ai_docs.py`
- **Process Markdown**: `process_md_files.py`
- **Add Context**: `add_context_to_nodes.py`
- **Create Vector Stores**: `create_vector_stores.py`
- **Upload to Chroma Vector Store to HuggingFace**: `upload_dbs_to_hf.py`
- **Upload JSONL files to HuggingFace**: `upload_jsonl_to_hf.py`

## Tips for New Team Members

1. To update the AI Tutor with new content:
   - For new courses, use `add_course_workflow.py`
   - For updated documentation, use `update_docs_workflow.py`

2. When adding URLs to course content:
   - Get the URLs from the live course platform
   - Add them to the generated JSONL file in the `url` field
   - Example URL format: `https://academy.towardsai.net/courses/take/python-for-genai/multimedia/62515980-course-structure`
   - Make sure every document has a valid URL

3. By default, only new content will have context added to save time and resources. Use `--process-all-context` only if you need to regenerate context for all documents. Use `--skip-data-upload` if you don't want to upload data files to the private HuggingFace repo (they're uploaded by default).

4. When adding a new course, verify that it appears in the Gradio UI:
   - The workflow automatically updates `main.py` and `setup.py` to include the new source
   - Check that the new source appears in the dropdown menu in the UI
   - Make sure it's properly included in the default selected sources
   - Restart the Gradio app to see the changes

5. First time setup or missing files:
   - Both workflows automatically check for and download required data files:
     - `all_sources_data.jsonl` - Contains the raw document data
     - `all_sources_contextual_nodes.pkl` - Contains the processed nodes with added context
   - If the PKL file exists, the `--new-context-only` flag will only process new content
   - You must have proper HuggingFace credentials with access to the private repository

6. Make sure you have the required environment variables set:
   - `OPENAI_API_KEY` for LLM processing
   - `COHERE_API_KEY` for embeddings
   - `HF_TOKEN` for HuggingFace uploads
   - `GITHUB_TOKEN` for accessing documentation via the GitHub API