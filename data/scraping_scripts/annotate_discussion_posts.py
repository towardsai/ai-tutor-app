"""Annotate academy discussion posts into an eval-ready dataset.

Reads ``data/academy_discussion_posts.jsonl`` (scraped course discussion
threads) and emits ``data/academy_discussion_eval.jsonl``, one annotated
record per thread, combining:

- deterministic fields extracted in code: the cleaned question, the earliest
  reply written by course staff (the reference answer), the full thread in
  chronological order, and the corpus source key for the course; and
- LLM annotations (Gemini structured output): category, expected tutor
  behavior, a self-contained rewrite of the question, evergreen key points
  distilled from the staff answer, and an overall eval-quality grade.

The output is the raw material for eval datasets: filter on
``annotation.eval_quality`` / ``annotation.category`` to build the Q&A set
(judge against ``key_points``), the out-of-scope behavior set
(``expected_behavior == "redirect_to_support"``), and retrieval ground truth
(``source_key`` + ``lesson_url``).

Resumable: post_ids already present in the output file are skipped unless
``--force`` is passed.

Usage:
    uv run -m data.scraping_scripts.annotate_discussion_posts [--limit N] [--force]
"""

import argparse
import asyncio
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.asyncio import tqdm

load_dotenv(".env")

DEFAULT_INPUT = Path("data/academy_discussion_posts.jsonl")
DEFAULT_OUTPUT = Path("data/academy_discussion_eval.jsonl")
ANNOTATION_MODEL = os.getenv("ANNOTATION_MODEL", "gemini-3.5-flash")
ANNOTATION_CONCURRENCY = int(os.getenv("ANNOTATION_CONCURRENCY", "8"))
RETRYABLE_GENAI_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# People who answer threads in an official capacity. Replies from anyone else
# (students helping each other, the original poster following up) are kept in
# the thread but never used as the reference answer. Derived from responders
# who reply in threads they did not start; edit as the team changes.
STAFF_RESPONDERS = {
    "Jaiganesan N",
    "Louis-François Bouchard",
    "Paul Iusztin",
    "Samridhi",
    "Louie Peters",
    "Omar Solano",
}

# Dataset course_name -> source key in source_registry.SOURCE_CONFIGS.
# Courses absent from the corpus map to None (the tutor has no material for
# them, which the annotator is told about).
COURSE_TO_SOURCE_KEY: dict[str, str | None] = {
    "Full Stack AI Engineering": "full_stack_ai_engineering",
    "Agent Engineering: Building Multi-Agent Systems": "agentic_ai_engineering",
    "Beginner Python for AI Engineering": "beginner_python_for_ai_engineering",
    "Master AI For Work": "master_ai_for_work",
    "8-hour Generative AI Primer": None,
}


class PostAnnotation(BaseModel):
    """LLM-produced annotation for one discussion thread."""

    category: Literal[
        "conceptual", "debugging", "course_feedback", "platform_issue", "other"
    ] = Field(
        description=(
            "conceptual: technical/theory question about course material. "
            "debugging: an error or unexpected result in the student's code, "
            "notebook, or environment. course_feedback: typo/errata reports or "
            "suggestions about course content. platform_issue: academy website, "
            "videos, quizzes, certificates, accounts, billing, community access. "
            "other: greetings, kudos, or unintelligible posts."
        )
    )
    expected_behavior: Literal[
        "answer_from_corpus",
        "answer_general",
        "redirect_to_support",
        "acknowledge_feedback",
    ] = Field(
        description=(
            "What an ideal tutor response does. answer_from_corpus: answer "
            "grounded in course/docs material. answer_general: sound technical "
            "answer that likely goes beyond the corpus. redirect_to_support: "
            "the tutor cannot act on this; it should say so briefly and point "
            "to the support team, without inventing troubleshooting steps. "
            "acknowledge_feedback: thank the student and suggest reporting the "
            "issue to the team."
        )
    )
    self_contained: bool = Field(
        description=(
            "True if the question can be fully understood without seeing the "
            "lesson page, a screenshot, a quiz, or other context the tutor "
            "does not have."
        )
    )
    standalone_question: str | None = Field(
        description=(
            "A minimally rewritten, self-contained version of the question "
            "suitable to send to the tutor: resolve references like 'this "
            "lesson', 'Image 8', or 'the notebook' using the course and "
            "lesson names; keep the student's wording and code verbatim "
            "otherwise. Null if the question cannot be made self-contained."
        )
    )
    key_points: list[str] = Field(
        description=(
            "1-4 evergreen technical claims distilled from STAFF replies, "
            "each independently checkable against a candidate answer (the "
            "diagnosis, the fix, the concept). Exclude insider or time-bound "
            "content: notebook-update promises, platform actions, version "
            "numbers true only at the time. Empty if there is no staff reply "
            "or it contains nothing technical."
        )
    )
    excluded_from_key_points: str | None = Field(
        description=(
            "One short note on staff-reply content deliberately left out of "
            "key_points (e.g. 'promise to update the notebook'). Null if "
            "nothing was excluded."
        )
    )
    time_bound: bool = Field(
        description=(
            "True if the correct answer depends on a point-in-time state "
            "(package version, notebook revision, platform status) that may "
            "have changed since."
        )
    )
    eval_quality: Literal["gold", "usable", "weak", "exclude"] = Field(
        description=(
            "Suitability as an eval case for the eval its expected_behavior "
            "implies. gold: clear self-contained question with a solid staff "
            "answer yielding key points, or an unambiguous redirect case. "
            "usable: good question with a partial/thin reference or needing "
            "the rewrite. weak: marginal; heavily time-bound or barely "
            "intelligible. exclude: no eval value (greetings, kudos, empty)."
        )
    )
    notes: str = Field(description="One-line rationale for the grades above.")


ANNOTATION_PROMPT = """\
You are annotating a student discussion thread from an online AI course so it \
can be turned into an evaluation case for an AI tutor chatbot.

About the tutor being evaluated:
- It answers questions grounded in a corpus of course materials and AI library \
docs (Transformers, PEFT, TRL, LlamaIndex, LangChain, LangGraph, OpenAI docs, \
and the courses themselves), and can optionally search the web.
- It CANNOT: access the academy platform (videos, quizzes, assignments, \
certificates, accounts, billing, Slack/Discord), see screenshots or images, \
run or update course notebooks, or change course content.

Thread metadata:
- Course: {course_name} ({corpus_note})
- Lesson: {lesson_name}
- The original post {media_note}.

Thread (chronological; STAFF = official course staff, STUDENT = learner):

{thread_text}

Annotate the thread per the response schema. Additional rules:
- key_points must come only from STAFF replies. If there is no staff reply, \
key_points must be empty and eval_quality is at most "usable".
- standalone_question preserves the student's voice and any code verbatim; it \
only resolves references the tutor could not see. If the post fundamentally \
depends on unseen content (a screenshot, quiz option text, a specific page), \
set self_contained to false and standalone_question to null.
- A clear platform_issue is still a valuable eval case (the tutor should \
redirect gracefully), so grade it on that basis rather than excluding it.
"""

_genai_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY must be set.")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def is_retryable_genai_error(exc: BaseException) -> bool:
    return isinstance(exc, APIError) and exc.code in RETRYABLE_GENAI_STATUS_CODES


def parse_post_date(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = " ".join(text.split())
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_byline(byline: str | None) -> tuple[str | None, str | None]:
    """Split 'Posted by NAME on\\nDATE' into (name, ISO date)."""
    if not byline:
        return None, None
    match = re.search(r"Posted by\s+(.+?)\s+on\b\s*(.*)$", byline, re.DOTALL)
    if not match:
        return " ".join(byline.split()) or None, None
    name = " ".join(match.group(1).split())
    return name or None, parse_post_date(match.group(2))


def build_thread(record: dict) -> list[dict]:
    """Replies in chronological order (the source lists newest first)."""
    thread = []
    for resp in reversed(record.get("responses") or []):
        author = (resp.get("responded_by") or "").strip()
        content = (resp.get("content") or "").strip()
        if not content:
            continue
        thread.append(
            {
                "role": "staff" if author in STAFF_RESPONDERS else "student",
                "author": author,
                "date": parse_post_date(resp.get("responded_at")),
                "content": content,
                "links": resp.get("links") or [],
            }
        )
    return thread


def first_staff_reply(thread: list[dict]) -> dict | None:
    return next((r for r in thread if r["role"] == "staff"), None)


def render_thread(
    question: str, asked_by: str | None, asked_at: str | None, thread: list[dict]
) -> str:
    parts = [
        f"[1] STUDENT {asked_by or 'Unknown'} ({asked_at or 'unknown date'}) — original post:\n{question}"
    ]
    for i, reply in enumerate(thread, start=2):
        parts.append(
            f"[{i}] {reply['role'].upper()} {reply['author']} "
            f"({reply['date'] or 'unknown date'}):\n{reply['content']}"
        )
    return "\n\n".join(parts)


def build_record(raw: dict) -> dict:
    """Deterministic part of the output record (everything but `annotation`)."""
    asked_by, asked_at = parse_byline((raw.get("original_post") or {}).get("byline"))
    title = (raw.get("discussion_title") or "").strip()
    body = ((raw.get("original_post") or {}).get("content") or "").strip()
    question = f"{title}\n\n{body}".strip() if title else body
    thread = build_thread(raw)
    reference = first_staff_reply(thread)
    course_name = raw.get("course_name") or ""

    return {
        "post_id": raw.get("post_id"),
        "discussion_url": raw.get("discussion_url"),
        "discussion_title": title,
        "course_name": course_name,
        "source_key": COURSE_TO_SOURCE_KEY.get(course_name),
        "lesson_name": raw.get("lesson_name"),
        "lesson_url": raw.get("lesson_url"),
        "asked_by": asked_by,
        "asked_at": asked_at,
        "question": question,
        "question_has_media": bool((raw.get("original_post") or {}).get("media")),
        "reference_answer": reference["content"] if reference else None,
        "reference_answer_by": reference["author"] if reference else None,
        "reference_answer_at": reference["date"] if reference else None,
        "reference_links": reference["links"] if reference else [],
        "thread": thread,
    }


@retry(
    retry=retry_if_exception(is_retryable_genai_error),
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(multiplier=1, max=60),
    reraise=True,
)
async def annotate_record(record: dict, model: str) -> PostAnnotation:
    source_key = record["source_key"]
    corpus_note = (
        f"in the tutor's corpus as source '{source_key}'"
        if source_key
        else "NOT in the tutor's corpus"
    )
    media_note = (
        "includes an image/screenshot the tutor cannot see"
        if record["question_has_media"]
        else "has no attached media"
    )
    prompt = ANNOTATION_PROMPT.format(
        course_name=record["course_name"],
        corpus_note=corpus_note,
        lesson_name=record["lesson_name"] or "unknown",
        media_note=media_note,
        thread_text=render_thread(
            record["question"],
            record["asked_by"],
            record["asked_at"],
            record["thread"],
        ),
    )
    response = await get_genai_client().aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=PostAnnotation,
        ),
    )
    if isinstance(response.parsed, PostAnnotation):
        return response.parsed
    if isinstance(response.parsed, dict):
        return PostAnnotation.model_validate(response.parsed)
    if response.text:
        return PostAnnotation.model_validate_json(response.text)
    raise ValueError(f"Gemini returned no annotation for post {record['post_id']}.")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


async def annotate_all(records: list[dict], model: str) -> list[dict]:
    semaphore = asyncio.Semaphore(ANNOTATION_CONCURRENCY)
    annotated_at = datetime.now(UTC).isoformat(timespec="seconds")

    async def worker(record: dict) -> dict:
        async with semaphore:
            annotation = await annotate_record(record, model)
        return {
            **record,
            "annotation": annotation.model_dump(),
            "annotation_model": model,
            "annotated_at": annotated_at,
        }

    return await tqdm.gather(*(worker(r) for r in records), desc="Annotating")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=ANNOTATION_MODEL)
    parser.add_argument(
        "--limit", type=int, default=None, help="Annotate at most N pending posts."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-annotate posts already present in the output file.",
    )
    args = parser.parse_args()

    raw_records = load_jsonl(args.input)
    if not raw_records:
        raise SystemExit(f"No records found in {args.input}.")

    existing = {r["post_id"]: r for r in load_jsonl(args.output)}
    records = [build_record(raw) for raw in raw_records]
    pending = [r for r in records if args.force or r["post_id"] not in existing]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"{len(records)} threads in input, {len(existing)} already annotated, "
        f"{len(pending)} to annotate with {args.model}."
    )
    if pending:
        for record in asyncio.run(annotate_all(pending, args.model)):
            existing[record["post_id"]] = record

    merged = sorted(existing.values(), key=lambda r: int(r["post_id"]))
    write_jsonl(args.output, merged)

    counts: dict[str, int] = {}
    for record in merged:
        quality = record["annotation"]["eval_quality"]
        category = record["annotation"]["category"]
        counts[f"quality={quality}"] = counts.get(f"quality={quality}", 0) + 1
        counts[f"category={category}"] = counts.get(f"category={category}", 0) + 1
    print(f"Wrote {len(merged)} annotated threads to {args.output}.")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")


if __name__ == "__main__":
    main()
