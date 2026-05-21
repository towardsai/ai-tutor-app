# AI Tutor KB Instructions

## Ground Truth

- `raw/` contains generated markdown mirrors of the normalized JSONL corpus. Treat these pages as the local source authority.
- `wiki/` contains navigation and synthesis pages. Use them to orient, not as final authority.
- `generated/` contains machine-generated indexes for manifests, headings, and symbols.

## Standard Workflow

**First command rule.** Unless the user gives you a specific file path or a
known symbol/class name to find, your FIRST `run_kb_command` must be one of:

- `cat wiki/index.md` — for any "tell me about X" / "what is X" / comparison
  / recipe question where X is a broad topic.
- `cat wiki/frameworks/<source>.md` or `cat wiki/courses/<source>.md` — when
  the question is already scoped to one known source.

Do NOT start with `rg`, `grep`, `find`, or `ls` over `raw/` as the first
command. The wiki pages already give you the source map, so doing your own
discovery first wastes budget and pulls noisy hits into context.

After the wiki orientation step:

1. Use `rg`, `find`, `sed`, `head`, and `cat` to verify and read raw pages.
2. Use generated indexes for exact symbols, headings, source paths, and URLs.
3. Read `raw/` pages to verify exact claims, code, and version-sensitive behavior.
4. Mention uncertainty when the knowledge base does not cover the answer.

## Exploration Budget

- Prefer 3-6 KB commands for normal answers. Stop browsing once you have one clearly relevant source page and enough detail to answer.
- Use more commands only when the user asks for exhaustive coverage, a comparison across sources, or debugging that requires multiple files.
- Do not read a long file from top to bottom. Scan headings first, then read only targeted ranges.
- Avoid broad searches that return thousands of lines. Add `-m 20` or narrow the path before searching `raw/`.

## Efficient Command Patterns

- Orientation (always first for broad questions): `cat wiki/index.md`
- Source map: `cat wiki/courses/<source>.md` or `cat wiki/frameworks/<source>.md`
- Follow-up search (after orientation): `rg -n -m 20 "query terms" raw/<scoped-path>`
- Heading scan: `rg -n "^#{1,3} " raw/.../<page>.md`
- Targeted read: `sed -n 'START,ENDp' raw/.../<page>.md`
- Exact symbols: `rg -n -m 20 "ClassName|function_name" generated/symbols.tsv raw/`

## Answering From KB Browsing

- Default to a concise synthesis rather than rewriting the markdown content verbatim.
- For "how do I build X?" questions, give the minimal architecture and a small runnable skeleton; expand only if the user asks for a full implementation.
- Cite the raw/source page inline for key claims. Do not cite every code comment or repeat the same citation after every sentence.
