"""Lint KB wiki pages for broken file references.

Runs over every ``wiki/**/*.md`` page in a KB directory and verifies:

1. Every referenced KB-root-relative path (a token starting with ``raw/``,
   ``wiki/``, or ``generated/``) exists on disk under the KB dir.
2. Every referenced ``raw/`` *file* also appears in
   ``generated/corpus_manifest.jsonl`` (matched against the manifest's
   ``path`` field, normalized to a KB-root-relative path). Directory
   references such as ``raw/courses/llm_primer/`` only need to exist.
3. Every relative markdown link between wiki pages resolves. Targets are
   resolved against the containing file's directory, with ``#anchors``
   stripped.

Extraction policy (deliberately conservative, tuned so the current wiki
lints clean; when in doubt a token is *ignored* rather than guessed at):

- Only backticked inline code spans, markdown link targets, and fenced code
  blocks are scanned. Plain prose is not: phrases like "no raw/ tree" must
  not become findings.
- Fenced code blocks ARE scanned (with the same token rules as inline code)
  because shell commands in fences cite real KB paths, and a stale path in a
  suggested ``rg``/``sed`` command is exactly the drift this lint guards
  against. The current wiki has no fenced blocks, so this costs nothing
  today and protects future pages.
- Inline-code / fenced content is split on whitespace and only tokens that
  start with one of the three layer prefixes count as path references, so
  command spans like ``rg -n "LoraConfig" raw/docs/peft/`` contribute just
  the path argument.
- Tokens containing glob or placeholder characters (``* ? { } [ ] < > | $``)
  or an ellipsis are skipped: the wiki legitimately uses templates like
  ``wiki/topics/*.md`` and ``raw/docs/transformers/model_doc/{name}.md``.
- URLs (anything with ``://``), ``mailto:`` targets, pure ``#anchor`` links,
  and absolute paths are ignored.
- Markdown link targets starting with a layer prefix are resolved against
  the KB root (the ``run_kb_command`` shell convention); other targets are
  only checked when they end in ``.md`` (after anchor stripping) and are
  resolved against the containing page's directory.

Exit status is nonzero when any finding is reported, so the KB build
workflows can fail before uploading a broken bundle.

Usage:
    uv run -m data.scraping_scripts.lint_kb_wiki [--kb-dir data/kb]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from data.scraping_scripts.update_kb_wiki import load_manifest, shell_path
except ModuleNotFoundError:
    from update_kb_wiki import load_manifest, shell_path

DEFAULT_KB_DIR = Path("data/kb")
LAYER_PREFIXES = ("raw/", "wiki/", "generated/")
MANIFEST_RELATIVE_PATH = "generated/corpus_manifest.jsonl"
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(([^()]+)\)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
# Glob/placeholder/shell characters: a token containing any of these is a
# pattern or template, not a concrete path reference.
NON_PATH_CHARS = set("*?{}[]<>|$")
STRIP_CHARS = "'\"()., ;:!"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Finding:
    page: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.page}:{self.line}: {self.message}"


def clean_token(token: str) -> str:
    """Strip wrapping punctuation that prose attaches to a path token."""
    return token.strip(STRIP_CHARS)


def is_kb_path_token(token: str) -> bool:
    if not token.startswith(LAYER_PREFIXES):
        return False
    if NON_PATH_CHARS.intersection(token) or "..." in token or "://" in token:
        return False
    return True


def extract_kb_path_tokens(text: str) -> list[str]:
    """KB-root-relative path tokens in inline-code or fenced-code text."""
    tokens = []
    for raw_token in text.split():
        token = clean_token(raw_token)
        if token and is_kb_path_token(token):
            tokens.append(token)
    return tokens


def extract_link_targets(line: str) -> list[str]:
    """Markdown link targets on a line, minus titles, anchors, and ``<>``."""
    targets = []
    for match in MARKDOWN_LINK_RE.finditer(line):
        target = match.group(1).strip().strip("<>")
        # Drop an optional link title: [text](path "title")
        target = target.split()[0] if target.split() else ""
        target = target.split("#", 1)[0]
        if target:
            targets.append(target)
    return targets


def is_checkable_relative_link(target: str) -> bool:
    if "://" in target or target.startswith(("mailto:", "/", "#")):
        return False
    if NON_PATH_CHARS.intersection(target) or "..." in target:
        return False
    return target.endswith(".md")


def relative_to_kb(path: Path, kb_dir: Path) -> str:
    try:
        return path.resolve().relative_to(kb_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def check_kb_path(
    token: str,
    *,
    page: str,
    line: int,
    kb_dir: Path,
    manifest_paths: set[str],
) -> list[Finding]:
    candidate = kb_dir / token
    if not candidate.exists():
        return [Finding(page, line, f"missing KB path: {token}")]
    if token.endswith("/") or candidate.is_dir():
        return []
    if token.startswith("raw/") and token not in manifest_paths:
        return [
            Finding(page, line, f"raw file not in {MANIFEST_RELATIVE_PATH}: {token}")
        ]
    return []


def check_relative_link(
    target: str,
    *,
    page: str,
    page_path: Path,
    line: int,
    kb_dir: Path,
    manifest_paths: set[str],
) -> list[Finding]:
    resolved = page_path.parent / target
    if not resolved.exists():
        return [
            Finding(
                page,
                line,
                f"broken relative link: {target} "
                f"(resolves to {relative_to_kb(resolved, kb_dir)})",
            )
        ]
    kb_relative = relative_to_kb(resolved, kb_dir)
    if kb_relative.startswith("raw/") and resolved.is_file():
        if kb_relative not in manifest_paths:
            return [
                Finding(
                    page,
                    line,
                    f"raw file not in {MANIFEST_RELATIVE_PATH}: {kb_relative}",
                )
            ]
    return []


def lint_wiki_page(
    page_path: Path, kb_dir: Path, manifest_paths: set[str]
) -> list[Finding]:
    page = relative_to_kb(page_path, kb_dir)
    findings: list[Finding] = []
    seen: set[tuple[int, str]] = set()

    def add(new_findings: list[Finding]) -> None:
        for finding in new_findings:
            key = (finding.line, finding.message)
            if key not in seen:
                seen.add(key)
                findings.append(finding)

    in_fence = False
    fence_char = ""
    for line_number, line in enumerate(
        page_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            current = fence_match.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_char = current
            elif current == fence_char:
                in_fence = False
                fence_char = ""
            continue

        if in_fence:
            snippets = [line]
            link_targets: list[str] = []
        else:
            snippets = INLINE_CODE_RE.findall(line)
            link_targets = extract_link_targets(line)

        for snippet in snippets:
            for token in extract_kb_path_tokens(snippet):
                add(
                    check_kb_path(
                        token,
                        page=page,
                        line=line_number,
                        kb_dir=kb_dir,
                        manifest_paths=manifest_paths,
                    )
                )

        for target in link_targets:
            if is_kb_path_token(target):
                add(
                    check_kb_path(
                        target,
                        page=page,
                        line=line_number,
                        kb_dir=kb_dir,
                        manifest_paths=manifest_paths,
                    )
                )
            elif is_checkable_relative_link(target):
                add(
                    check_relative_link(
                        target,
                        page=page,
                        page_path=page_path,
                        line=line_number,
                        kb_dir=kb_dir,
                        manifest_paths=manifest_paths,
                    )
                )

    return findings


def lint_kb_wiki(kb_dir: Path) -> list[Finding]:
    wiki_dir = kb_dir / "wiki"
    if not wiki_dir.is_dir():
        return [Finding(str(kb_dir), 0, "wiki/ directory not found")]

    manifest_file = kb_dir / MANIFEST_RELATIVE_PATH
    if not manifest_file.is_file():
        return [Finding(str(kb_dir), 0, f"{MANIFEST_RELATIVE_PATH} not found")]
    manifest_paths = {
        path for row in load_manifest(kb_dir) if (path := shell_path(row, kb_dir))
    }

    findings: list[Finding] = []
    for page_path in sorted(wiki_dir.rglob("*.md")):
        findings.extend(lint_wiki_page(page_path, kb_dir, manifest_paths))
    return findings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(
        description="Verify KB wiki file references (paths, manifest, links)."
    )
    parser.add_argument("--kb-dir", default=str(DEFAULT_KB_DIR))
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    findings = lint_kb_wiki(kb_dir)
    if findings:
        for finding in findings:
            logger.error(str(finding))
        logger.error("KB wiki lint failed: %s broken reference(s)", len(findings))
        sys.exit(1)
    print(f"KB wiki lint passed: no broken references in {kb_dir / 'wiki'}")


if __name__ == "__main__":
    main()
