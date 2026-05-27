# Python scripts for adding new courses and updating documentation

## First workflow: Adding a New Course

Make sure you have the required environment variables set:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` for context generation with Gemini
- `COHERE_API_KEY` for embeddings
- `HF_TOKEN` for HuggingFace uploads and downloads - [access to the private HuggingFace dataset repo](https://huggingface.co/datasets/towardsai-tutors/ai-tutor-data/tree/main)
- `GITHUB_TOKEN` for accessing files via the GitHub API

Optional Gemini context-generation tuning:

- `GEMINI_CONTEXT_TPM_LIMIT` - input token-per-minute quota to throttle against (defaults to `30000000`)
- `GEMINI_CONTEXT_TPM_SAFETY_MARGIN` - fraction of that quota to use before pausing (defaults to `0.8`)
- `GEMINI_CONTEXT_CONCURRENCY` - max concurrent context requests (defaults to `50`)
- `GEMINI_CONTEXT_RETRY_ATTEMPTS` - max tenacity attempts for transient Gemini API errors (defaults to `8`)

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

8. Open `data/scraping_scripts/source_registry.py`.

9. Add the new course to the `SOURCE_CONFIGS` dictionary. Sources listed in
   this registry are active in the knowledge base.

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
uv run -m data.scraping_scripts.add_course_workflow --courses [COURSE_NAME]
```

example:

```bash
uv run -m data.scraping_scripts.add_course_workflow --courses master_ai_for_work
```

This script will guide you through the complete process, it will:

   1. Extract the markdown content from each of the lessons and create a new JSONL file for the course
   2. Download the JSONL files from the other courses
   3. Prompt you to manually add URLs to the course content, inside the newly created JSONL file (more details in step 3 below)
   4. Merge the course data into the main dataset
   5. Add contextual information to document chunks before embedding
   6. Create vector stores
   7. Upload databases to HuggingFace
   8. Confirm the course is configured in the central source registry

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
uv run -m data.scraping_scripts.add_course_workflow --courses master_ai_for_work
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

### Rebuilding the KB from scratch

If you've blown away local KB state (e.g. `rm -rf data/kb data/chroma-db-all_sources`)
and want everything back, **just run the docs workflow** — it now handles
fresh-build cases automatically.

```bash
uv run -m data.scraping_scripts.update_docs_workflow
```

What happens under the hood when local state is missing:

1. `ensure_required_files_exist` downloads `all_sources_data.jsonl`,
   `all_sources_contextual_nodes.pkl`, and every per-source JSONL (including
   courses) from the private HF dataset `towardsai-tutors/ai-tutor-data`.
2. Fresh docs are re-fetched from GitHub / llms.txt indexes.
3. `build_kb_artifacts.py` regenerates `data/kb/raw/` and `data/kb/generated/`
   from the JSONL. Course raw pages are reconstructed from the JSONL too —
   you do **not** need the original Notion exports in `data/<course_name>/`.
4. `update_kb_wiki.py` rebuilds `data/kb/wiki/`. When it detects an empty
   `wiki/` it auto-promotes to a full seed (topic pages, recipes/errors index
   pages), so a clean rebuild produces the full wiki. On incremental runs,
   maintainer-authored prose outside `<!-- AUTO-GENERATED -->` markers is
   preserved; pass `--seed-defaults` to wipe and reseed everything.
5. `data/kb/AGENTS.md` is overwritten from
   `data/scraping_scripts/kb_agents_template.md` (the canonical template,
   tracked in git so it survives any `rm -rf data/`).
6. Chroma vector store is rebuilt by `create_vector_stores.py`.

Prerequisites for a from-scratch rebuild:

- `HF_TOKEN` with access to `towardsai-tutors/ai-tutor-data` (so course
  JSONLs and the contextual nodes PKL can be downloaded).
- `GITHUB_TOKEN` (rate-limit headroom for docs downloads).
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` (used by `add_context_to_nodes.py`).
- `COHERE_API_KEY` (used by `create_vector_stores.py`).

What this workflow **cannot** restore: the original Notion-exported course
markdown in `data/<course_name>/`. You don't need it for the runtime — the
chunked content lives in the JSONL on HF — but if you want to re-run
`add_course_workflow.py` against fresh course content, you'd need a new
Notion export.

### Editing KB agent guidance permanently

Do **not** edit `data/kb/AGENTS.md` directly. The file is overwritten in two
places, both reading from the canonical template at
`data/scraping_scripts/kb_agents_template.md`:

- `data.scraping_scripts.update_kb_wiki.write_agents_md` — rewrites it during
  every `update_docs_workflow.py` run, before uploading to HuggingFace.
- `scripts.setup.ensure_kb_agents_md` — rewrites it on every runtime startup,
  after `ensure_local_vector_db` downloads the snapshot. This catches the
  case where the HF snapshot has a stale AGENTS.md (e.g. uploaded before a
  template change landed in git) and ensures the live file always matches
  the version checked into the current branch.

Edit the template, commit it, and either run the docs workflow (to refresh
the HF snapshot) or just restart the chatbot (the template gets copied into
place on startup either way).

### Where `data/kb/` actually lives

`data/kb/` is gitignored (it's a 100MB+ build artifact). It's:

- **Built** by `data.scraping_scripts.build_kb_artifacts` (from
  `data/all_sources_data.jsonl`) and `data.scraping_scripts.update_kb_wiki`.
- **Uploaded** to `towardsai-tutors/ai-tutor-vector-db` by
  `data.scraping_scripts.upload_dbs_to_hf` (already includes `kb/**`).
- **Downloaded** by `scripts.setup.ensure_local_vector_db` on the first
  chatbot start (or any start where the local KB is missing).

Treat it the same way as `data/chroma-db-all_sources/`: never commit it,
never edit it by hand, regenerate or re-download when you need it.

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
- **Retire a source from data and Chroma**: `retire_source_workflow.py`

## Retiring a Course or Documentation Source

Use the retirement workflow when a source should be removed from retrieval and
from the Hugging Face data repositories:

```bash
uv run -m data.scraping_scripts.retire_source_workflow --sources 8-hour_primer --yes
```

The workflow:

1. Downloads `all_sources_data.jsonl` and `all_sources_contextual_nodes.pkl` if
   they are missing locally.
2. Removes rows/chunks whose `source` matches the retired source key.
3. Removes the retired source from `source_registry.py`, so it is no longer
   active in future workflow runs or the UI source picker.
4. Rebuilds `data/chroma-db-all_sources`.
5. Uploads the rebuilt vector DB and updated aggregate data files.
6. Deletes the retired per-source JSONL from `towardsai-tutors/ai-tutor-data`.

Run with `--dry-run` first to preview counts without changing files:

```bash
uv run -m data.scraping_scripts.retire_source_workflow --sources 8-hour_primer --dry-run
```

Use `--keep-source-registry` only if you want to remove existing chunks while
leaving the source configured as active for a future rebuild.

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
   - Add the source label and default-selection metadata in `source_registry.py`
   - Check that the new source appears in the dropdown menu in the UI
   - Make sure it's properly included in the default selected sources if desired
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

7. KB wiki artifact creation:
   - `build_kb_artifacts.py` converts `all_sources_data.jsonl` into generated markdown under `data/kb/raw/`
   - `update_kb_wiki.py` refreshes wiki navigation pages under `data/kb/wiki/`, preserving maintainer prose outside `<!-- AUTO-GENERATED -->` markers
   - The docs and course workflows run these steps automatically unless `--skip-kb` is passed
   - `upload_dbs_to_hf.py` uploads `data/kb/**` with the vector database so the runtime can download one KB bundle
