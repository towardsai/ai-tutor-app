import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.chat_types import SourceMatch
from app.kb_manifest import (
    kb_root_path,
    load_manifest_entries,
    manifest_indexes,
    parse_markdown_citations,
    resolve_manifest_reference,
    source_match_payload,
)


def _write_manifest(kb_dir: Path, rows: list[dict]) -> None:
    generated = kb_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row) for row in rows]
    (generated / "corpus_manifest.jsonl").write_text("\n".join(lines))


class KbRootPathTestCase(unittest.TestCase):
    def test_strips_kb_dir_prefix(self) -> None:
        self.assertEqual(
            kb_root_path("data/kb/raw/docs/peft/lora.md"),
            "raw/docs/peft/lora.md",
        )

    def test_strips_leading_dot_slash(self) -> None:
        self.assertEqual(
            kb_root_path("./raw/docs/peft/lora.md"),
            "raw/docs/peft/lora.md",
        )

    def test_keeps_root_relative_path(self) -> None:
        self.assertEqual(
            kb_root_path("raw/docs/peft/lora.md"),
            "raw/docs/peft/lora.md",
        )


class ManifestPathResolutionTestCase(unittest.TestCase):
    def test_manifest_match_carries_kb_root_path(self) -> None:
        # The client maps inline `raw/...` citations to the resolved URL via
        # this path, so it must be KB-root-relative regardless of how the
        # manifest spells it.
        with TemporaryDirectory() as tmp:
            kb_dir = Path(tmp)
            _write_manifest(
                kb_dir,
                [
                    {
                        "doc_id": "peft:lora",
                        "title": "LoRA",
                        "url": "https://example.com/lora",
                        "source": "peft",
                        "source_group": "docs",
                        "path": "data/kb/raw/docs/peft/lora.md",
                    }
                ],
            )

            match = resolve_manifest_reference(
                "raw/docs/peft/lora.md", kb_dir=str(kb_dir)
            )

            self.assertIsNotNone(match)
            assert match is not None
            self.assertEqual(match.url, "https://example.com/lora")
            self.assertEqual(match.path, "raw/docs/peft/lora.md")

            by_doc_scheme = resolve_manifest_reference(
                "kb://doc/peft:lora", kb_dir=str(kb_dir)
            )
            self.assertIsNotNone(by_doc_scheme)
            assert by_doc_scheme is not None
            self.assertEqual(by_doc_scheme.path, "raw/docs/peft/lora.md")


class AmbiguousTitleResolutionTestCase(unittest.TestCase):
    def test_duplicate_title_does_not_resolve_to_arbitrary_doc(self) -> None:
        # Two distinct docs share the title "Introduction"; a bare-label
        # citation must not silently resolve to whichever was ingested last.
        with TemporaryDirectory() as tmp:
            kb_dir = Path(tmp)
            _write_manifest(
                kb_dir,
                [
                    {
                        "doc_id": "peft:intro",
                        "title": "Introduction",
                        "url": "https://example.com/peft/intro",
                        "source": "peft",
                        "source_group": "docs",
                        "path": "data/kb/raw/docs/peft/intro.md",
                    },
                    {
                        "doc_id": "trl:intro",
                        "title": "Introduction",
                        "url": "https://example.com/trl/intro",
                        "source": "trl",
                        "source_group": "docs",
                        "path": "data/kb/raw/docs/trl/intro.md",
                    },
                    {
                        "doc_id": "peft:lora",
                        "title": "LoRA",
                        "url": "https://example.com/peft/lora",
                        "source": "peft",
                        "source_group": "docs",
                        "path": "data/kb/raw/docs/peft/lora.md",
                    },
                ],
            )

            self.assertIsNone(
                resolve_manifest_reference(
                    "Introduction", label="Introduction", kb_dir=str(kb_dir)
                )
            )

            # A unique title still resolves by label.
            lora = resolve_manifest_reference("LoRA", label="LoRA", kb_dir=str(kb_dir))
            self.assertIsNotNone(lora)
            assert lora is not None
            self.assertEqual(lora.doc_id, "peft:lora")

            # Both same-title docs remain resolvable by their unambiguous keys.
            by_url = resolve_manifest_reference(
                "https://example.com/trl/intro", kb_dir=str(kb_dir)
            )
            self.assertIsNotNone(by_url)
            assert by_url is not None
            self.assertEqual(by_url.doc_id, "trl:intro")


class ManifestLoaderTestCase(unittest.TestCase):
    def test_missing_manifest_is_not_cached_empty(self) -> None:
        # A lookup before the first-start bundle download must not pin an empty
        # manifest for the process lifetime; once the file appears it loads.
        with TemporaryDirectory() as tmp:
            kb_dir = Path(tmp)
            self.assertEqual(load_manifest_entries(str(kb_dir)), ())

            _write_manifest(
                kb_dir,
                [
                    {
                        "doc_id": "peft:lora",
                        "title": "LoRA",
                        "url": "https://example.com/lora",
                        "source": "peft",
                        "source_group": "docs",
                        "path": "data/kb/raw/docs/peft/lora.md",
                    }
                ],
            )

            entries = load_manifest_entries(str(kb_dir))
            self.assertEqual([entry.doc_id for entry in entries], ["peft:lora"])

    def test_malformed_line_is_skipped_not_fatal(self) -> None:
        with TemporaryDirectory() as tmp:
            kb_dir = Path(tmp)
            generated = kb_dir / "generated"
            generated.mkdir(parents=True, exist_ok=True)
            good_a = json.dumps(
                {
                    "doc_id": "peft:lora",
                    "title": "LoRA",
                    "url": "https://example.com/lora",
                    "source": "peft",
                    "source_group": "docs",
                    "path": "data/kb/raw/docs/peft/lora.md",
                }
            )
            good_b = json.dumps(
                {
                    "doc_id": "trl:intro",
                    "title": "Intro",
                    "url": "https://example.com/intro",
                    "source": "trl",
                    "source_group": "docs",
                    "path": "data/kb/raw/docs/trl/intro.md",
                }
            )
            (generated / "corpus_manifest.jsonl").write_text(
                f"{good_a}\n{{not valid json\n[1, 2, 3]\n{good_b}\n"
            )

            entries = load_manifest_entries(str(kb_dir))
            self.assertEqual(
                [entry.doc_id for entry in entries], ["peft:lora", "trl:intro"]
            )


class ManifestIndexCacheTestCase(unittest.TestCase):
    LORA_ROW = {
        "doc_id": "peft:lora",
        "title": "LoRA",
        "url": "https://example.com/lora",
        "source": "peft",
        "source_group": "docs",
        "path": "data/kb/raw/docs/peft/lora.md",
    }

    def test_indexes_are_cached_once_manifest_exists(self) -> None:
        # resolve_manifest_reference runs once per citation/shell path, so the
        # indexes must not be rebuilt over all entries on every call.
        with TemporaryDirectory() as tmp:
            kb_dir = Path(tmp)
            _write_manifest(kb_dir, [self.LORA_ROW])

            first = manifest_indexes(str(kb_dir))
            second = manifest_indexes(str(kb_dir))

            self.assertIs(first, second)

    def test_absent_manifest_does_not_pin_empty_indexes(self) -> None:
        # Mirrors load_manifest_entries: an index lookup before the first-start
        # bundle download must not cache empty indexes for the process
        # lifetime; once the file appears the indexes pick it up.
        with TemporaryDirectory() as tmp:
            kb_dir = Path(tmp)
            by_doc_id, _by_url, _by_path, _by_title = manifest_indexes(str(kb_dir))
            self.assertEqual(by_doc_id, {})
            self.assertIsNone(
                resolve_manifest_reference("kb://doc/peft:lora", kb_dir=str(kb_dir))
            )

            _write_manifest(kb_dir, [self.LORA_ROW])

            by_doc_id, _by_url, _by_path, _by_title = manifest_indexes(str(kb_dir))
            self.assertIn("peft:lora", by_doc_id)
            match = resolve_manifest_reference("kb://doc/peft:lora", kb_dir=str(kb_dir))
            self.assertIsNotNone(match)
            assert match is not None
            self.assertEqual(match.url, "https://example.com/lora")


class SourceMatchPayloadTestCase(unittest.TestCase):
    def test_payload_includes_path(self) -> None:
        match = SourceMatch(
            doc_id="peft:lora",
            title="LoRA",
            url="https://example.com/lora",
            source_key="peft",
            source_label="PEFT Docs",
            score=1.0,
            group="docs",
            path="raw/docs/peft/lora.md",
        )

        payload = source_match_payload(match, message_id="m1")

        self.assertEqual(payload["path"], "raw/docs/peft/lora.md")
        self.assertNotIn("call_id", payload)


class ParseMarkdownCitationsTestCase(unittest.TestCase):
    # Faithful excerpt of the answer from LangSmith trace
    # 019f004f-af13-7e42-ae3d-9ffd4a1de6d9 ("I'm having trouble cloning the
    # course GitHub repository..."), which produced an orphan "https://`" chip.
    GIT_CLONE_ANSWER = (
        "The official course repository is hosted on GitHub "
        "[Lesson 1, Part 2: Course Admin](https://academy.towardsai.net/lesson-1). "
        "To clone it locally, run:\n"
        "```bash\n"
        "git clone https://github.com/towardsai/agentic-ai-engineering-course.git\n"
        "```\n"
        "or for the RAG system:\n"
        "```bash\n"
        "git clone https://github.com/towardsai/ai-tutor-rag-system.git\n"
        "```\n"
        "**Windows:** Download Git from the "
        "[Git for Windows website](https://git-scm.com/download/win).\n"
        "Always default to the **HTTPS** URLs (the ones starting with "
        "`https://`). HTTPS does not require SSH keys.\n"
    )

    def test_does_not_harvest_inline_code_url(self) -> None:
        refs = [ref for _label, ref in parse_markdown_citations(self.GIT_CLONE_ANSWER)]
        # The bug: the inline span `https://` was captured (with the trailing
        # backtick) as a bare-URL citation and surfaced as a broken chip.
        self.assertNotIn("https://`", refs)
        self.assertNotIn("https://", refs)

    def test_does_not_harvest_code_fence_urls(self) -> None:
        refs = [ref for _label, ref in parse_markdown_citations(self.GIT_CLONE_ANSWER)]
        # `git clone` URLs are code examples, not citations.
        self.assertNotIn(
            "https://github.com/towardsai/agentic-ai-engineering-course.git", refs
        )
        self.assertNotIn("https://github.com/towardsai/ai-tutor-rag-system.git", refs)

    def test_keeps_genuine_markdown_links(self) -> None:
        citations = parse_markdown_citations(self.GIT_CLONE_ANSWER)
        refs = [ref for _label, ref in citations]
        self.assertIn("https://academy.towardsai.net/lesson-1", refs)
        self.assertIn("https://git-scm.com/download/win", refs)
        # Only the two real inline links survive; no code-derived noise.
        self.assertEqual(len(citations), 2)

    def test_still_harvests_bare_url_in_prose(self) -> None:
        # Regression guard: a bare URL in actual prose (not code) is still a
        # valid citation and must keep being harvested.
        citations = parse_markdown_citations(
            "See the docs at https://example.com/guide for details."
        )
        self.assertEqual(citations, [("", "https://example.com/guide")])

    # Faithful excerpt of the answer from LangSmith trace
    # 019f29f4-4e21-7a01-86b8-053161eb08d6 ("How do I evaluate and iterate on
    # my prompts..."): the model closed two citations with "]" instead of ")".
    LESSON_URL = (
        "https://academy.towardsai.net/courses/take/agent-engineering/multimedia/"
        "71589897-lesson-29-defining-the-evaluation-processes-and-metrics-theory"
    )
    OPENAI_URL = (
        "https://developers.openai.com/api/docs/guides/evaluation-best-practices"
    )
    MALFORMED_CLOSER_ANSWER = (
        f"You must adopt EDD [Lesson 29: Evaluation Processes and Metrics]({LESSON_URL}).\n\n"
        "### 1. The Optimization Flywheel Process\n"
        "Set up a loop known as the **Optimization Flywheel** "
        f"[Lesson 29: Evaluation Processes and Metrics]({LESSON_URL}]:\n\n"
        "1. **Gather a Static Dataset:** Compile 20 to 100 diverse inputs "
        f"[OpenAI Evaluation Best Practices]({OPENAI_URL}). Mine failures from "
        f"production logs [OpenAI Evaluation Best Practices]({OPENAI_URL}).\n"
    )

    def test_malformed_link_does_not_swallow_following_citations(self) -> None:
        # The bug: with a destination pattern of [^)]+ the "]"-closed link
        # matched across newlines up to the next citation's closing paren,
        # swallowing the following OpenAI citation into one garbage reference
        # that resolved to nothing. The malformed reference itself dedupes
        # against the well-formed Lesson 29 citation.
        refs = [
            ref
            for _label, ref in parse_markdown_citations(self.MALFORMED_CLOSER_ANSWER)
        ]
        self.assertEqual(refs, [self.LESSON_URL, self.OPENAI_URL, self.OPENAI_URL])

    def test_malformed_link_url_is_recovered(self) -> None:
        # A citation the model closed with "]" instead of ")" still yields its
        # URL (as a bare, label-less reference) so it can resolve to a card
        # even when no well-formed citation of the same doc exists.
        citations = parse_markdown_citations(
            "See [Lesson 29](https://example.com/lesson-29]: for the flywheel."
        )
        self.assertEqual(citations, [("", "https://example.com/lesson-29")])


if __name__ == "__main__":
    unittest.main()
