# Python scripts for adding new courses and updating documentation

## First workflow: Adding a New Course

Make sure you have the required environment variables set:

- `OPENAI_API_KEY` for the LLM
- `COHERE_API_KEY` for embeddings
- `HF_TOKEN` for HuggingFace uploads and downloads - [access to the private HuggingFace dataset repo](https://huggingface.co/datasets/towardsai-tutors/ai-tutor-data/tree/main)
- `GITHUB_TOKEN` for accessing files via the GitHub API

## 1. Prepare the course data

0. Make sure you have access to:

   - the live course you want to add:
     - [https://academy.towardsai.net/courses/...](https://academy.towardsai.net/courses/take/ai-business-professionals/multimedia/64071930-introduction)
   - you are part of the HuggingFace towards-ai team, to access the private spaces:
     - [towardsai-tutors](https://huggingface.co/towardsai-tutors)

1. In Notion, navigate to the main course page that contains the live lessons:

   - e.g. [Notion page](https://www.notion.so/seldonia/AI-for-Business-Professionals-190f9b6f42708087863df100e9b4b556)

2. Click on the three dots in the top right corner and select "Export"

3. Select these options:
   - Export format: "Markdown & CSV"
   - Include databases: current view
   - Include content: Everything
   - Include subpages: yes
   - Create folders for subpages: yes

4. Click on "Export"

5. Once the export is complete, unzip file.

6. Move the unzipped folder into the `ai-tutor-app/data` directory

7. Rename the folder to the course name.
   - e.g. `master_ai_for_work`

8. Open the `data/scraping_scripts/process_md_files.py` python file and locate the `SOURCE_CONFIGS` dictionary.

9. Add the new course to the `SOURCE_CONFIGS` dictionary.

   example:

   ```python
      "master_ai_for_work": {
         "base_url": "",
         "input_directory": "data/master_ai_for_work",  # Relative path to the directory that contains the Markdown files
         "output_file": "data/master_ai_for_work_data.jsonl", # The output file that will be created by the script
         "source_name": "master_ai_for_work",
         "use_include_list": False,
         "included_dirs": [],
         "excluded_dirs": [],
         "excluded_root_files": [],
         "included_root_files": [],
         "url_extension": "",
      },
   ```

   - The most important fields are:
      - input_directory: the relative path to the directory that contains the Markdown files
      - output_file: the name of the output file that will be created by the script
      - source_name: the name of the course (keep underscores, no spaces)
   - The other fields can stay as empty lists or empty strings.

## 2. Run the add_course_workflow.py script

```bash
uv run -m data.scraping_scripts.add_course_workflow --course [COURSE_NAME]
```

example:

```bash
uv run -m data.scraping_scripts.add_course_workflow --course master_ai_for_work
```

This script will guide you through the complete process, it will:

   1. Extract the markdown content from each of the lessons and create a new JSONL file for the course
   2. Download the JSONL files from the other courses
   3. Prompt you to manually add URLs to the course content, inside the newly created JSONL file (more details in step 3 below)
   4. Merge the course data into the main dataset
   5. Add contextual information to document chunks before embedding
   6. Create vector stores
   7. Upload databases to HuggingFace
   8. Update UI configuration

## 3. Add URLs to the course content + Manual Dataset Cleaning (Most important step)

- After the script has processed the markdown files, it will prompt you to manually add URLs to the course content.
- Answer "no" to the question "Have you added all the URLs?"
- Open the newly created `data/master_ai_for_work_data.jsonl` file in a text editor and open the live course page in the browser.
- If the JSON looks split into multiple wrapped lines in VS Code / Cursor, you can toggle word wrap off.
  - macOS: press ⌥ Option + Z
  - Windows/Linux: press Alt + Z

- For each lesson in the course, starting at the beginning [academy.towardsai.net](https://academy.towardsai.net/courses/take/ai-business-professionals/multimedia/64071930-introduction), copy the URL, and add it to the `url` field in the JSONL file, in the corresponding line/lesson. Read the note below to know what urls to add.

**Note:** While you do this, now is the time to clean up the .jsonl file, remove any lines/lessons that should not be added to the RAG chatbot.
example: "Course Admin and Syllabus", "Course Structure/Overview", "Course Outline", "Quiz", "Assigments", "Introduction to Module X" etc. You can also remove lines that are videos. Only actual lessons should be kept in the JSONL file, with the `url` field filled in.

- What you can do is add the URLs for all the lessons you want to add to the RAG chatbot, and when done, remove all the json lines that have an empty `url` field.

## 4. Once done, run the script again and answer "yes" to the question "Have you added all the URLs?"

```bash
uv run -m data.scraping_scripts.add_course_workflow --course master_ai_for_work
```

## 5. Its done

Run the chatbot locally to test if the course has been added correctly.

```bash
uv run -m scripts.main
```

----

## Second workflow: Updating Documentation via GitHub API

To update library documentation from GitHub repositories:

```bash
uv run -m data.scraping_scripts.update_docs_workflow
```

This will update all supported documentation sources. You can also specify specific sources:

```bash
uv run -m data.scraping_scripts.update_docs_workflow --sources transformers peft langchain
```

The workflow includes:

1. Downloading documentation from GitHub using the API
2. Processing markdown files to create JSONL data
3. Adding contextual information to document chunks
4. Creating vector stores
5. Uploading vector db and new JSONL files to HuggingFace

Both workflows validate Hugging Face access up front, and the docs workflow now stops immediately on GitHub auth/API failures so it does not overwrite source JSONL files with incomplete data.

## Individual Components

If you need to run specific steps individually:

- **GitHub to Markdown**: `github_to_markdown_ai_docs.py`
- **Process Markdown**: `process_md_files.py`
- **Add Context**: `add_context_to_nodes.py`
- **Create Vector Stores**: `create_vector_stores.py`
- **Upload to Chroma Vector Store to HuggingFace**: `upload_dbs_to_hf.py`
- **Upload JSONL files to HuggingFace**: `upload_data_to_hf.py`

## Tips for New Team Members

1. To update the AI Tutor with new content:
   - For new courses, use `add_course_workflow.py`
   - For updated documentation, use `update_docs_workflow.py`

2. When adding URLs to course content:
   - Get the URLs from the live course platform
   - Add them to the generated JSONL file in the `url` field
   - Example URL format: `https://academy.towardsai.net/courses/take/ai-business-professionals/multimedia/64072092-introduction-to-chatgpt-claude-and-gemini`
   - Make sure every document has a valid URL

3. By default, only new content will have context added to save time and resources. Use `--process-all-context` only if you need to regenerate context for all documents. Use `--skip-data-upload` if you don't want to upload data files to the private HuggingFace repo (they're uploaded by default).

4. When adding a new course, verify that it appears in the Gradio UI:
   - The workflow automatically updates `scripts/setup.py` to include the new source and `scripts/main.py` to preselect it by default
   - Check that the new source appears in the dropdown menu in the UI
   - Make sure it's properly included in the default selected sources
   - Restart the Gradio app to see the changes

5. First time setup or missing files:
   - Both workflows automatically check for and download required data files:
     - `all_sources_data.jsonl` - Contains the raw document data
     - `all_sources_contextual_nodes.pkl` - Contains the context-enriched chunk manifest used for embedding
   - If the PKL file exists, the default behavior only processes new content; use `--process-all-context` to regenerate context for every document
   - You must have proper HuggingFace credentials with access to the private repository

6. Vector-store creation:
   - `create_vector_stores.py` now shows terminal progress for embedding generation and Chroma upserts
   - `all_sources_contextual_nodes.pkl` stores chunk records for embedding; legacy pickles containing LlamaIndex nodes are still accepted by the runtime compatibility helpers
