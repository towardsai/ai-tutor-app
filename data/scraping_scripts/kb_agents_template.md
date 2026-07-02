# AI Tutor KB Instructions

## Ground Truth

- `raw/` contains generated markdown mirrors of the normalized JSONL corpus. Treat these pages as the local source authority.
- `wiki/` contains navigation and synthesis pages. Use them to orient, not as final authority.
- `generated/` contains machine-generated indexes for manifests, headings, and symbols.

## First Command Rule

Unless the user gives you a specific file path or a known symbol/class name
to find, your FIRST `run_kb_command` must be one of:

- `cat wiki/index.md` — for any "tell me about X" / "what is X" / comparison
  / recipe question where X is a broad topic.
- `cat wiki/frameworks/<source>.md` (or `cat wiki/courses/<source>.md`, when
  course sources are present in this KB) — when the question is already
  scoped to one known source. `wiki/index.md` lists which sources exist; do
  not assume a source page is there if the index does not name it.

Do NOT start with `rg`, `grep`, `find`, or `ls` over `raw/` as the first
command. The wiki pages already give you the source map, so doing your own
discovery first wastes budget and pulls noisy hits into context.

## After Orientation

- Allowed `run_kb_command` programs: `rg`, `grep`, `find`, `ls`, `sed`, `head`,
  `cat`, `wc`. No pipes, no redirects, no command chaining.
- For exact class/function/method names, search `generated/symbols.tsv` before
  scanning raw files.
- Avoid broad searches that return thousands of lines. Add `-m 20` or narrow
  the path before searching `raw/`.
- Mention uncertainty when the knowledge base does not cover the answer.

## Answering

- Default to a concise synthesis rather than rewriting the markdown content verbatim.
- For "how do I build X?" questions, give the minimal architecture and a small runnable skeleton; expand only if the user asks for a full implementation.
- Cite the raw/source page inline for key claims. Do not cite every code comment or repeat the same citation after every sentence.
