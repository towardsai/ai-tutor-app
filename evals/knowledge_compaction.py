"""Knowledge-compaction study: how to fit one long lesson into context.

The workshop question this answers: filling the context window is where the
tokens/cost are -- so given a long document, what is the best way to *compact*
its knowledge into context to answer questions? We take the corpus's largest
course lesson, generate a question set over it, and answer every question under
several strategies, measuring **tokens, $/answer, latency, and answer quality**.

Strategies (each turns the lesson into the answering context):
  full_context          - the whole lesson ("shove it all"), the cost baseline
  trim                  - head+tail truncation to a token budget
  summary               - one LLM summary of the lesson (precomputed once)
  hierarchical_summary  - map-reduce: summarize chunks, then summarize summaries
  rag                   - chunk + embed + retrieve top-k for the question
  graphrag              - GraphRAG retriever over a per-lesson index
  selective             - structural skeleton (headings + lead sentences) to budget

Quality is graded by an LLM judge that sees the FULL lesson (the real source)
and checks whether the answer is correct and supported by it -- we never write
reference answers (repo rule). Everything runs on Gemini 2.5 Flash via the
OpenAI-compatible endpoint (stable under load), to keep cost low.

  uv run --env-file .env -m evals.knowledge_compaction --questions 24
  uv run --env-file .env -m evals.knowledge_compaction --questions 2 --smoke   # cheap check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .common import percentile

logger = logging.getLogger("evals.knowledge_compaction")

LESSON_PATH = (
    "data/kb/raw/courses/master_ai_for_work/"
    "case-study-pulling-together-our-llm-uses-organising-a-major-conference-end-to-end.md"
)
OUT_DIR = "data/compaction"
GRAPHRAG_LESSON_OUTPUT = "data/graphrag_lesson/output"

# Gemini 2.5 Flash via the OpenAI-compatible endpoint (stable under load).
MODEL = "openai/gemini-2.5-flash"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
PRICE_IN_PER_MTOK = 0.30
PRICE_OUT_PER_MTOK = 2.50

TRIM_BUDGET_TOKENS = 4000
SELECTIVE_BUDGET_TOKENS = 4000
RAG_TOP_K = 6
STRATEGIES = [
    "full_context",
    "trim",
    "summary",
    "hierarchical_summary",
    "rag",
    "graphrag",
    "selective",
]


def _gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set (run with: uv run --env-file .env ...)")
    return key


def complete(
    messages: list[dict], *, max_tokens: int | None = None, temperature: float = 0
) -> tuple[str, int, int, float]:
    """One Gemini 2.5 call -> (text, in_tokens, out_tokens, latency_s)."""
    import litellm

    started = time.monotonic()
    kwargs: dict = {
        "model": MODEL,
        "api_base": API_BASE,
        "api_key": _gemini_key(),
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    resp = litellm.completion(**kwargs)
    latency = time.monotonic() - started
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return text, int(usage.prompt_tokens), int(usage.completion_tokens), latency


def cost_usd(in_tok: int, out_tok: int) -> float:
    return in_tok / 1e6 * PRICE_IN_PER_MTOK + out_tok / 1e6 * PRICE_OUT_PER_MTOK


# --- context builders -----------------------------------------------------


def load_lesson(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def chunk_lesson(lesson: str) -> list[str]:
    from app.chroma_rag import heading_aware_markdown_chunks

    return [c.text for c in heading_aware_markdown_chunks(lesson, chunk_size=800)]


def trim_to_budget(text: str, enc, budget: int) -> str:
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= budget:
        return text
    half = budget // 2
    head = enc.decode(tokens[:half])
    tail = enc.decode(tokens[-half:])
    return f"{head}\n\n...[middle truncated]...\n\n{tail}"


def build_summary(lesson: str) -> str:
    text, *_ = complete(
        [
            {
                "role": "user",
                "content": (
                    "Summarize the following lesson into a thorough study summary "
                    "that preserves the concrete facts, steps, names, and examples a "
                    "student might be asked about. Use compact prose/bullets.\n\n"
                    f"{lesson}"
                ),
            }
        ],
        max_tokens=4000,
    )
    return text.strip()


def build_hierarchical_summary(chunks: list[str]) -> str:
    """Map-reduce summary: summarize each chunk, then summarize the summaries."""
    partials: list[str] = []
    for chunk in chunks:
        text, *_ = complete(
            [
                {
                    "role": "user",
                    "content": "Summarize this lesson section, keeping concrete "
                    f"facts, steps, and names:\n\n{chunk}",
                }
            ],
            max_tokens=600,
        )
        partials.append(text.strip())
    combined = "\n\n".join(partials)
    text, *_ = complete(
        [
            {
                "role": "user",
                "content": "Combine these section summaries into one coherent, "
                "non-redundant study summary that keeps the concrete details:\n\n"
                f"{combined}",
            }
        ],
        max_tokens=4000,
    )
    return text.strip()


def build_selective_skeleton(lesson: str, enc, budget: int) -> str:
    """Query-independent 'selective retention': every heading plus the lead
    sentence of each paragraph, truncated to budget. Keeps structure + topic
    sentences, drops the bulk."""
    lines = lesson.splitlines()
    kept: list[str] = []
    in_para = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            in_para = False
            continue
        if stripped.startswith("#"):
            kept.append(stripped)
            in_para = False
        elif not in_para:
            # first (lead) sentence of this paragraph
            lead = re.split(r"(?<=[.!?])\s", stripped, maxsplit=1)[0]
            kept.append(lead)
            in_para = True
    skeleton = "\n".join(kept)
    return trim_to_budget(skeleton, enc, budget)


class LessonRagIndex:
    """Tiny in-memory hybrid-free RAG over the lesson's chunks (Cohere embed +
    rerank), matching production's embed/rerank models for a fair cost shape."""

    def __init__(self, chunks: list[str]) -> None:
        import cohere

        from app.chroma_rag import DEFAULT_EMBED_MODEL, embed_texts

        self._chunks = chunks
        self._cohere = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        self._vecs = embed_texts(
            self._cohere, chunks, input_type="search_document", model=DEFAULT_EMBED_MODEL
        )

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> str:
        from app.chroma_rag import DEFAULT_EMBED_MODEL, embed_texts

        qv = embed_texts(
            self._cohere, [query], input_type="search_query", model=DEFAULT_EMBED_MODEL
        )[0]

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(y * y for y in b) ** 0.5
            return dot / (na * nb) if na and nb else 0.0

        ranked = sorted(
            range(len(self._chunks)), key=lambda i: cos(qv, self._vecs[i]), reverse=True
        )
        top = [self._chunks[i] for i in ranked[: max(top_k * 2, top_k)]]
        rr = self._cohere.rerank(
            model="rerank-v4.0-fast", query=query, documents=top, top_n=top_k
        )
        return "\n\n".join(top[r.index] for r in rr.results)


# --- run -------------------------------------------------------------------


def generate_questions(lesson: str, n: int, out_path: Path) -> list[str]:
    if out_path.exists():
        qs = [json.loads(line)["question"] for line in out_path.read_text().splitlines() if line.strip()]
        if len(qs) >= n:
            return qs[:n]
    text, *_ = complete(
        [
            {
                "role": "user",
                "content": (
                    f"Generate exactly {n} diverse questions a student might ask "
                    "about the lesson below. Mix specific-detail questions (a single "
                    "fact/step buried in the text) with cross-section synthesis "
                    "questions (require combining multiple parts). Return ONLY a JSON "
                    'array of strings.\n\n'
                    f"{lesson}"
                ),
            }
        ],
        max_tokens=3000,
    )
    raw = text.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    questions = json.loads(match.group(0) if match else raw)
    questions = [str(q).strip() for q in questions if str(q).strip()][:n]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps({"question": q}) for q in questions))
    return questions


def answer(question: str, context: str) -> tuple[str, int, int, float]:
    return complete(
        [
            {
                "role": "system",
                "content": "You are an AI tutor. Answer the student's question using "
                "ONLY the provided lesson context. If the context does not contain "
                "the answer, say you don't have enough information.",
            },
            {"role": "user", "content": f"Lesson context:\n{context}\n\nQuestion: {question}"},
        ],
        max_tokens=800,
    )


JUDGE_RE = re.compile(r"\{.*\}", re.DOTALL)


def judge(question: str, ans: str, lesson: str) -> tuple[bool, str]:
    text, *_ = complete(
        [
            {
                "role": "system",
                "content": "You grade an AI tutor's answer against the SOURCE LESSON "
                "(ground truth). Pass only if the answer is correct and supported by "
                "the lesson. Output ONLY JSON: {\"pass\": true/false, \"reason\": \"...\"}.",
            },
            {
                "role": "user",
                "content": f"SOURCE LESSON:\n{lesson}\n\nQUESTION: {question}\n\n"
                f"TUTOR ANSWER: {ans}",
            },
        ],
        max_tokens=300,
    )
    match = JUDGE_RE.search(text)
    if not match:
        return False, f"unparseable judge output: {text[:120]}"
    try:
        verdict = json.loads(match.group(0))
        return bool(verdict.get("pass")), str(verdict.get("reason", ""))
    except json.JSONDecodeError:
        return False, f"bad judge json: {text[:120]}"


@dataclass
class Row:
    question: str
    strategy: str
    answer: str
    context_tokens: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    judge_pass: bool
    judge_reason: str


def build_contexts(lesson: str, chunks: list[str], enc) -> dict:
    """Precompute the query-independent contexts once (the expensive part)."""
    logger.info("Precomputing summary / hierarchical summary / skeleton ...")
    ctx = {
        "full_context": lesson,
        "trim": trim_to_budget(lesson, enc, TRIM_BUDGET_TOKENS),
        "summary": build_summary(lesson),
        "hierarchical_summary": build_hierarchical_summary(chunks),
        "selective": build_selective_skeleton(lesson, enc, SELECTIVE_BUDGET_TOKENS),
    }
    return ctx


def report(rows: list[Row], out_dir: Path) -> str:
    by_strategy: dict[str, list[Row]] = {}
    for row in rows:
        by_strategy.setdefault(row.strategy, []).append(row)
    lines = [
        "# Knowledge-compaction report",
        "",
        f"Lesson: `{LESSON_PATH}` | model: gemini-2.5-flash | n questions: "
        f"{len({r.question for r in rows})}",
        "",
        "| strategy | judge pass | ctx tok | in tok/turn | $/turn | latency s p50/p95 |",
        "|---|---|---|---|---|---|",
    ]
    order = sorted(
        by_strategy,
        key=lambda s: -sum(1 for r in by_strategy[s] if r.judge_pass) / len(by_strategy[s]),
    )
    for s in order:
        rs = by_strategy[s]
        passed = sum(1 for r in rs if r.judge_pass)
        lat = [r.latency_s for r in rs]
        lines.append(
            f"| {s} | {passed}/{len(rs)} ({passed / len(rs):.0%}) | "
            f"{mean(r.context_tokens for r in rs):.0f} | "
            f"{mean(r.input_tokens for r in rs):.0f} | "
            f"${mean(r.cost_usd for r in rs):.4f} | "
            f"{percentile(lat, 50):.1f}/{percentile(lat, 95):.1f} |"
        )
    out = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", default=LESSON_PATH)
    parser.add_argument("--questions", type=int, default=24)
    parser.add_argument("--strategies", nargs="*", default=STRATEGIES)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--smoke", action="store_true", help="tiny run, skip nothing")
    args = parser.parse_args()

    from app.chroma_rag import get_token_encoding

    enc = get_token_encoding()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lesson = load_lesson(args.lesson)
    chunks = chunk_lesson(lesson)
    logger.info("Lesson %d tokens, %d chunks", len(enc.encode(lesson, disallowed_special=())), len(chunks))

    questions = generate_questions(lesson, args.questions, out_dir / "questions.jsonl")
    logger.info("Using %d questions", len(questions))

    strategies = list(args.strategies)
    contexts = build_contexts(lesson, chunks, enc)

    rag = LessonRagIndex(chunks) if "rag" in strategies else None
    graphrag = None
    if "graphrag" in strategies:
        try:
            from app.graph_rag import GraphRAGRetriever

            graphrag = GraphRAGRetriever(
                cohere_api_key=os.environ["COHERE_API_KEY"],
                output_dir=GRAPHRAG_LESSON_OUTPUT,
                lancedb_dir=f"{GRAPHRAG_LESSON_OUTPUT}/lancedb",
            )
        except Exception as exc:
            logger.warning("graphrag arm unavailable (%s); skipping it.", exc)
            strategies = [s for s in strategies if s != "graphrag"]

    def context_for(strategy: str, question: str) -> str:
        if strategy in contexts:
            return contexts[strategy]
        if strategy == "rag":
            return rag.retrieve(question)
        if strategy == "graphrag":
            return "\n\n".join(r.content for r in graphrag.search(question))
        raise ValueError(f"unknown strategy {strategy}")

    rows: list[Row] = []
    bundle = out_dir / "bundles.jsonl"
    bundle.write_text("")
    total_cost = 0.0
    for qi, question in enumerate(questions):
        for strategy in strategies:
            ctx = context_for(strategy, question)
            ctx_tok = len(enc.encode(ctx, disallowed_special=()))
            ans, in_tok, out_tok, latency = answer(question, ctx)
            c = cost_usd(in_tok, out_tok)
            jp, jr = judge(question, ans, lesson)
            total_cost += c
            row = Row(
                question=question,
                strategy=strategy,
                answer=ans,
                context_tokens=ctx_tok,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=c,
                latency_s=latency,
                judge_pass=jp,
                judge_reason=jr,
            )
            rows.append(row)
            with open(bundle, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")
        logger.info("q%d/%d done (running answer cost $%.3f)", qi + 1, len(questions), total_cost)

    print(report(rows, out_dir))
    print(f"\nBundles: {bundle}  | answer-call cost ~= ${total_cost:.2f} (excl. judge/precompute)")


if __name__ == "__main__":
    main()
