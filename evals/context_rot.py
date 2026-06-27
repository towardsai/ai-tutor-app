"""Context-rot sweep: does "keep everything" lose on QUALITY before capacity?

The one axis where ``full_history`` (the arm our other batteries keep crowning)
can actually fail: it FITS, it's cheap (cached), and it STILL answers worse
because the relevant fact is buried in a big window (lost-in-the-middle /
attention dilution). To isolate that we hold the fact, the question, the model,
and the task constant and vary only **context length** and the **needle's
position** -- a domain-adapted NIAH/RULER grid.

Design:
  * Substrate: a single STUFFED message = filler + one planted needle, retrieval
    OFF (tools off) -- the only path to the answer is reading the buried fact.
  * Needle: a synthetic fact ABSENT from the corpus, so retrieval/world-knowledge
    can't shortcut it. Probe question's only answer is the needle.
  * Distractors: near-miss facts planted in the filler so the model must
    discriminate, not just locate.
  * Calibration: the same probe at a tiny fill must pass ~100% -- proving any
    drop at length is ROT, not a hard question.
  * Vary: fill length (expressed as % of the model's window, e.g. 20/30/40 ...)
    x needle position (start / middle / end). Grade semantically.

  # cloud big-window smoke (Gemini), cheap mechanics check
  uv run --env-file .env -m evals.context_rot --provider gemini --smoke

  # local SLM, 32k window: sweep within-window rot AND over-the-wall truncation
  uv run --env-file .env -m evals.context_rot \
      --provider ollama --model ollama_chat/qwen2.5:32b-instruct \
      --num-ctx 32768 --window 32768 --fill-pcts 20 40 60 80 110 \
      --positions start middle end --trials 3 --out data/context_rot_qwen32b
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean

from .common import percentile
from .knowledge_compaction import PROVIDERS as _BASE_PROVIDERS
from .knowledge_compaction import ModelConfig, complete

logger = logging.getLogger("evals.context_rot")

# Rot is measured on the model under test only; cost is irrelevant here (we grade
# QUALITY, not dollars), so OpenRouter's fragmented caching -- which biased the
# cost runs -- does not matter for this study. DeepSeek-V4-Flash gives the
# big-window curve the local SLM (32k) can't reach.
PROVIDERS: dict[str, ModelConfig] = {
    **_BASE_PROVIDERS,
    "deepseek_or": ModelConfig(
        model="openrouter/deepseek/deepseek-v4-flash",
        api_base=None,  # litellm routes "openrouter/" itself
        api_key_env="OPENROUTER_API_KEY",
        price_in=0.0,  # unused: rot grades quality, not cost
        price_out=0.0,
    ),
}

# Synthetic needles: distinctive, plausible, and ABSENT from the course corpus,
# so the only way to answer the probe is to read the planted fact (not retrieve
# it, not guess it). Each ships its own near-miss DISTRACTORS, scattered in the
# filler so the model must discriminate the right fact, not merely locate a
# pattern. ``answer`` is the ground truth the judge grades against.
@dataclass(frozen=True)
class Needle:
    fact: str  # the sentence planted in context
    probe: str  # the question whose ONLY answer is the needle
    answer: str  # ground truth handed to the judge
    distractors: tuple[str, ...]  # near-misses scattered in the filler


NEEDLES: tuple[Needle, ...] = (
    Needle(
        fact="Operational note: the internal codename for the conference's "
        "logistics-scheduling AI was 'Project Lighthouse-7'.",
        probe="What was the internal codename for the conference's "
        "logistics-scheduling AI?",
        answer="Project Lighthouse-7",
        distractors=(
            "Operational note: the catering vendor's booking tool was nicknamed "
            "'Project Harbor-3' by the events team.",
            "Operational note: the badge-printing pipeline was informally called "
            "'Project Beacon-2' in early drafts.",
        ),
    ),
    Needle(
        fact="Per the runbook, the maximum allowed retry budget for the overnight "
        "ingestion job is exactly 47 attempts.",
        probe="What is the maximum allowed retry budget for the overnight "
        "ingestion job?",
        answer="47 attempts",
        distractors=(
            "Per the runbook, the alerting cooldown for the ingestion job is 12 "
            "minutes.",
            "Per the runbook, the nightly export job is capped at 30 attempts.",
        ),
    ),
    Needle(
        fact="Decision log: the fallback embedding model approved by the review "
        "board was code-named 'Cinnabar-9'.",
        probe="What was the code name of the fallback embedding model approved by "
        "the review board?",
        answer="Cinnabar-9",
        distractors=(
            "Decision log: the rejected reranker prototype was code-named "
            "'Cobalt-4'.",
            "Decision log: the primary embedding model in production is called "
            "'Marigold-2'.",
        ),
    ),
    Needle(
        fact="Scheduling note: the database migration window was fixed for the "
        "night of March 14th, 2027, starting at 02:00 UTC.",
        probe="On what date and time (UTC) was the database migration window "
        "scheduled to start?",
        answer="March 14th, 2027, at 02:00 UTC",
        distractors=(
            "Scheduling note: the load test was penciled in for February 9th, 2027.",
            "Scheduling note: the staging refresh runs every Sunday at 04:00 UTC.",
        ),
    ),
    Needle(
        fact="Budget memo: the contingency reserve set aside for the keynote "
        "stage production was 18,500 euros.",
        probe="How large was the contingency reserve set aside for the keynote "
        "stage production?",
        answer="18,500 euros",
        distractors=(
            "Budget memo: the travel reimbursement cap for speakers was 1,200 "
            "euros each.",
            "Budget memo: the total catering spend came to 42,000 euros.",
        ),
    ),
)

POSITION_FRACTIONS = {"start": 0.04, "middle": 0.5, "end": 0.96}


def load_filler(enc, max_tokens: int) -> list[int]:
    """Concatenate corpus markdown into one token list (>= max_tokens).

    Real lesson prose makes the hardest, most realistic distractor sea -- the
    needle has to survive among on-topic text, not lorem ipsum.
    """
    roots = [Path("data/kb/raw"), Path("data/kb/wiki")]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.rglob("*.md"), key=lambda p: -p.stat().st_size))
    if not files:
        raise SystemExit("no filler markdown found under data/kb/raw or data/kb/wiki")
    toks: list[int] = []
    for f in files:
        toks.extend(enc.encode(f.read_text(encoding="utf-8"), disallowed_special=()))
        if len(toks) >= max_tokens:
            break
    if len(toks) < max_tokens:
        # Loop the corpus until we reach the target (big windows need more than
        # the corpus holds); rot doesn't care that filler repeats.
        base = list(toks)
        while len(toks) < max_tokens:
            toks.extend(base)
    return toks[:max_tokens]


def build_context(
    enc, filler_tokens: list[int], fill_tokens: int, needle: Needle, position: str
) -> str:
    """Stuff `fill_tokens` of filler, sprinkle distractors, insert the needle at
    the requested position, and return the decoded reference block."""
    body = filler_tokens[:fill_tokens]
    insert_at = int(POSITION_FRACTIONS[position] * len(body))
    head = enc.decode(body[:insert_at])
    tail = enc.decode(body[insert_at:])
    # Scatter distractors a few paragraphs before/after so discrimination is
    # required; place them away from the needle's exact position.
    d = list(needle.distractors)
    dist_head = ("\n\n" + d[0] + "\n\n") if d else "\n\n"
    dist_tail = ("\n\n" + d[1] + "\n\n") if len(d) > 1 else "\n\n"
    return (
        head
        + dist_head
        + "\n\n"
        + needle.fact
        + "\n\n"
        + dist_tail
        + tail
    )


_WORD_RE = re.compile(r"[a-z0-9][a-z0-9-]+")
_STOP = frozenset(
    "the a an of to in on for and or is was were what which how when who that this "
    "with as at by from into per its it be are list note the".split()
)


def lexical_retrieve(chunks: list[str], query: str, top_k: int) -> str:
    """No-API top-k: rank chunks by rare-term overlap with the query (BM25-lite).

    A free stand-in for Cohere embed+rerank when the trial key is exhausted. Rare
    query terms (the distinctive needle words like 'codename', 'logistics-scheduling')
    are weighted up via inverse chunk-frequency, so the needle chunk -- not a
    generic-vocabulary distractor -- ranks top. A lower bound on retrieval quality:
    semantic retrieval would do at least as well at surfacing the buried fact.
    """
    import math

    q_terms = {t for t in _WORD_RE.findall(query.lower()) if t not in _STOP}
    n = len(chunks)
    chunk_terms = [set(_WORD_RE.findall(c.lower())) for c in chunks]
    df = {t: sum(1 for ct in chunk_terms if t in ct) for t in q_terms}
    idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in q_terms}
    scored = sorted(
        range(n),
        key=lambda i: sum(idf[t] for t in q_terms if t in chunk_terms[i]),
        reverse=True,
    )
    return "\n\n".join(chunks[i] for i in scored[:top_k])


def _conversation_messages(context: str, probe: str) -> list[dict]:
    """Reshape the SAME stuffed context (filler + distractors + needle) into a
    conversation-shaped sequence of messages: split on blank lines, group into turns, and
    alternate user/assistant so the model sees genuine chat structure instead of
    one flat document. The needle keeps its position; the needle text,
    distractors, filler, and judge are all identical to the document run -- only
    the message shape changes, so shape is the single variable."""
    import math

    paras = [s.strip() for s in context.split("\n\n") if s.strip()]
    max_turns = 80
    if len(paras) > max_turns:
        per = math.ceil(len(paras) / max_turns)
        paras = ["\n\n".join(paras[i : i + per]) for i in range(0, len(paras), per)]
    msgs: list[dict] = [
        {
            "role": "system",
            "content": "You are a tutor in a long multi-turn conversation. The "
            "answer to the final question is stated somewhere earlier in this "
            "conversation. Answer concisely. If it is truly not present, say you "
            "cannot find it.",
        }
    ]
    for i, body in enumerate(paras):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": body})
    msgs.append({"role": "user", "content": f"Question: {probe}"})
    return msgs


# Answer-token budget for the model under test. The original 300 was the single
# biggest confound in the document rot grid: DeepSeek-V4-Flash is verbose /
# reasoning-style, and a buried fact induces LONGER reasoning, so the 300 cap was
# hit before the answer was emitted -> an EMPTY response the judge scored as a
# miss. That is output truncation masquerading as lost-in-the-middle (it even
# correlated with the middle position). 2000 leaves room for reasoning + answer;
# any cell that still hits the cap is flagged ``truncated`` and treated as INVALID.
ANSWER_MAX_TOKENS = 2000


def answer(
    probe: str, context: str, cfg, fmt: str = "document"
) -> tuple[str, int, int, float]:
    """Single stuffed-context turn, retrieval OFF -- the model must read the
    buried needle; there is no tool to route around it. fmt='conversation'
    reshapes the identical context into multi-turn chat messages, to test whether
    the rot transfers to a conversation-shaped context."""
    if fmt == "conversation":
        return complete(
            _conversation_messages(context, probe),
            max_tokens=ANSWER_MAX_TOKENS,
            cfg=cfg,
        )
    return complete(
        [
            {
                "role": "system",
                "content": "You are an assistant answering from the provided "
                "reference material ONLY. The answer to the question is stated "
                "somewhere in the material. Answer concisely. If it is truly not "
                "present, say you cannot find it.",
            },
            {
                "role": "user",
                "content": f"Reference material:\n{context}\n\nQuestion: {probe}",
            },
        ],
        max_tokens=ANSWER_MAX_TOKENS,
        cfg=cfg,
    )


JUDGE_RE = re.compile(r"\{.*\}", re.DOTALL)
JUDGE_PASS_RE = re.compile(r'"pass"\s*:\s*(true|false)', re.IGNORECASE)


def judge(probe: str, ans: str, needle: Needle, cfg) -> tuple[bool, str]:
    """Grade against the NEEDLE fact only (not the whole filler): did the answer
    recover the planted fact? Semantic, so paraphrases pass and a distractor
    fails."""
    text, *_ = complete(
        [
            {
                "role": "system",
                "content": "You grade whether an answer correctly states a specific "
                "GROUND-TRUTH FACT. Pass only if the answer conveys that fact "
                "(semantic match; paraphrase is fine). Fail if it states a "
                "different value, a near-miss, or says it cannot find it. Output "
                'ONLY JSON: {"pass": true/false, "reason": "<=20 words"}.',
            },
            {
                "role": "user",
                "content": f"GROUND-TRUTH FACT: {needle.answer}\n\nQUESTION: "
                f"{probe}\n\nANSWER: {ans}",
            },
        ],
        max_tokens=200,
        cfg=cfg,
    )
    match = JUDGE_RE.search(text)
    if match:
        try:
            v = json.loads(match.group(0))
            return bool(v.get("pass")), str(v.get("reason", ""))
        except json.JSONDecodeError:
            pass
    pm = JUDGE_PASS_RE.search(text)
    if pm:
        return pm.group(1).lower() == "true", text[:160].strip()
    return False, f"unparseable judge: {text[:120]}"


@dataclass
class Cell:
    fill_tokens: int
    fill_pct: int | None
    position: str
    trial: int
    needle_answer: str
    context_tokens: int
    input_tokens: int
    output_tokens: int
    latency_s: float
    judge_pass: bool
    judge_reason: str
    answer_text: str
    ctx_overflow: bool
    mode: str = "stuff"  # "stuff" = full haystack in prompt; "rag" = retrieved top-k
    seen_tokens: int = 0  # tokens the model actually answered from (= ctx for stuff)
    format: str = "document"  # "document" = one stuffed message; "conversation" = multi-turn
    truncated: bool = False  # answer hit the output-token cap or came back empty ->
    # the cell is INVALID for rot (truncation is not an attention miss); flagged so it
    # can never again be silently counted as a buried-fact failure.


def report(cells: list[Cell], out_dir: Path, model: str, window: int | None) -> str:
    fills = sorted({c.fill_tokens for c in cells})
    positions = [p for p in ("start", "middle", "end") if any(c.position == p for c in cells)]
    fmt = cells[0].format if cells else "document"
    rmode = cells[0].mode if cells else "stuff"
    shape = (
        "conversation-shaped (multi-turn messages)"
        if fmt == "conversation"
        else "stuffed single turn"
    )
    retr = "retrieval ON (top-k)" if rmode.startswith("rag") else "retrieval OFF"
    substrate = f"Substrate: {shape}, {retr}, synthetic needle + distractors, semantic judge."

    def rate(subset: list[Cell]) -> str:
        if not subset:
            return "-"
        p = sum(1 for c in subset if c.judge_pass)
        return f"{p}/{len(subset)} ({p / len(subset):.0%})"

    lines = [
        "# Context-rot sweep",
        "",
        f"Model: {model}" + (f" | window {window // 1000}k" if window else "") + "",
        substrate,
        "",
        "## Pass rate by fill length x needle position",
        "",
        "| fill | " + " | ".join(positions) + " | overall |",
        "|---" * (len(positions) + 2) + "|",
    ]
    for ft in fills:
        row_cells = [c for c in cells if c.fill_tokens == ft]
        pct = next((c.fill_pct for c in row_cells if c.fill_pct is not None), None)
        ctx = int(mean(c.context_tokens for c in row_cells))
        ov = any(c.ctx_overflow for c in row_cells)
        label = f"{ctx // 1000}k" + (f" ({pct}%)" if pct is not None else "")
        if ov:
            label += " ⚠over"
        cells_by_pos = [rate([c for c in row_cells if c.position == p]) for p in positions]
        lines.append(
            f"| {label} | " + " | ".join(cells_by_pos) + f" | {rate(row_cells)} |"
        )
    lat = [c.latency_s for c in cells]
    ntrunc = sum(1 for c in cells if getattr(c, "truncated", False))
    lines += [
        "",
        f"Latency p50/p95: {percentile(lat, 50):.1f}s / {percentile(lat, 95):.1f}s | "
        f"cells: {len(cells)} | truncated (INVALID, output-cap hit): {ntrunc}",
        "",
        "Rot signature: pass rate falls as fill grows, and/or a U-shape across "
        "position (worst in the middle). Flat-to-large = keep-everything is robust.",
    ]
    out = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=sorted(PROVIDERS), default="gemini")
    p.add_argument("--model", help="Override the litellm model id.")
    p.add_argument("--api-base")
    p.add_argument("--num-ctx", type=int, help="Ollama context window.")
    p.add_argument("--judge-provider", choices=sorted(PROVIDERS), default="gemini")
    p.add_argument(
        "--window",
        type=int,
        help="Model window size, for expressing fills as %% and flagging overflow.",
    )
    p.add_argument(
        "--fills",
        type=int,
        nargs="*",
        help="Absolute fill lengths in tokens (e.g. 8000 50000 100000).",
    )
    p.add_argument(
        "--fill-pcts",
        type=int,
        nargs="*",
        help="Fills as %% of --window (Louis's framing: 20 30 40 ...). "
        "Needs --window.",
    )
    p.add_argument("--positions", nargs="*", default=["start", "middle", "end"])
    p.add_argument("--trials", type=int, default=3, help="Needles per cell (n).")
    p.add_argument(
        "--calibration",
        type=int,
        default=4000,
        help="Tiny-fill calibration cell (tokens); 0 to skip.",
    )
    p.add_argument("--out", default="data/context_rot")
    p.add_argument("--smoke", action="store_true", help="2 fills x start/end, 1 trial.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Keep cells already in <out>/bundles.jsonl and skip re-running them "
        "(continue an interrupted run instead of starting over).",
    )
    p.add_argument(
        "--retrieve",
        type=int,
        default=0,
        help="RAG mode: chunk the haystack, retrieve top-K for the probe, answer "
        "from those instead of stuffing it all. 0 = off (stuffed baseline). This is "
        "the compaction contrast -- does surfacing the buried fact restore recall?",
    )
    p.add_argument(
        "--retrieve-mode",
        choices=["cohere", "lexical"],
        default="cohere",
        help="cohere = embed+rerank (production-like); lexical = no-API BM25-lite "
        "(use when the Cohere trial key is exhausted).",
    )
    p.add_argument(
        "--format",
        choices=["document", "conversation"],
        default="document",
        help="document = one stuffed message (original NIAH). conversation = the "
        "SAME needle/filler/distractors reshaped into multi-turn chat messages, to "
        "test whether the rot transfers to a conversation-shaped context.",
    )
    args = p.parse_args()

    cfg = PROVIDERS[args.provider]
    if args.model:
        cfg = replace(cfg, model=args.model)
    if args.api_base is not None:
        cfg = replace(cfg, api_base=args.api_base)
    if args.num_ctx is not None:
        cfg = replace(cfg, num_ctx=args.num_ctx)
    judge_cfg = PROVIDERS[args.judge_provider]

    window = args.window or cfg.num_ctx
    # Resolve fill lengths.
    if args.fill_pcts:
        if not window:
            raise SystemExit("--fill-pcts needs --window (or an ollama --num-ctx).")
        fills = [(int(window * pct / 100), pct) for pct in args.fill_pcts]
    elif args.fills:
        fills = [(f, (int(100 * f / window) if window else None)) for f in args.fills]
    else:
        fills = [(8000, None), (40000, None)]
    positions = list(args.positions)
    trials = args.trials
    if args.smoke:
        fills = fills[:2]
        positions = ["start", "end"]
        trials = 1

    from app.chroma_rag import get_token_encoding

    enc = get_token_encoding()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_fill = max(f for f, _ in fills)
    logger.info("model=%s judge=%s | building filler up to %d tokens ...", cfg.model, judge_cfg.model, max_fill)
    filler = load_filler(enc, max_fill)

    meta = {
        "model": cfg.model,
        "format": args.format,
        "window": window,
        "judge": judge_cfg.model,
        "fills": [f for f, _ in fills],
        "positions": positions,
        "trials": trials,
        "calibration_tokens": args.calibration,
        "needles": [n.answer for n in NEEDLES],
    }
    meta_path = out_dir / "meta.json"
    # On resume, validate the EXISTING meta against the current args BEFORE
    # overwriting it -- otherwise a resume with a different model/window/judge/
    # format silently rewrites provenance and mixes incompatible cells. The grid
    # params (fills/positions/trials) may legitimately extend, so they are not
    # part of the invariant check.
    if args.resume and meta_path.exists():
        old = json.loads(meta_path.read_text())
        invariants = ("model", "format", "window", "judge")
        mismatch = {
            k: (old.get(k), meta[k]) for k in invariants if old.get(k) != meta[k]
        }
        if mismatch:
            raise SystemExit(
                f"--resume mismatch in {meta_path}: "
                + ", ".join(f"{k} {o!r}->{n!r}" for k, (o, n) in mismatch.items())
                + ". The existing run used different settings; use a fresh --out."
            )
    meta_path.write_text(json.dumps(meta, indent=2))

    # Build the work list: optional calibration cell first (proves probes are
    # answerable at small fill), then the length x position x trial grid.
    cells: list[Cell] = []
    bundle = out_dir / "bundles.jsonl"
    # Resume: keep cells already on disk and skip re-running them; otherwise start
    # fresh (truncate). Cells are keyed by (fill_tokens, position, trial), so a
    # killed run (laptop sleep/shutdown) continues from where it stopped.
    done_keys: set[tuple[int, str, int]] = set()
    if args.resume and bundle.exists():
        for line in bundle.read_text().splitlines():
            if not line.strip():
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a half-written final line (killed mid-write)
            cells.append(Cell(**c))
            done_keys.add((c["fill_tokens"], c["position"], c["trial"]))
        logger.info("resume: %d cells already done, skipping them", len(done_keys))
    else:
        bundle.write_text("")

    retrieve_k = args.retrieve
    mode = (f"rag-{args.retrieve_mode}" if retrieve_k else "stuff")
    if args.resume and cells:
        bad = next(
            (c for c in cells if c.format != args.format or c.mode != mode), None
        )
        if bad:
            raise SystemExit(
                f"--resume mismatch: {bundle} holds format={bad.format!r} "
                f"mode={bad.mode!r} cells, but you asked for format={args.format!r} "
                f"mode={mode!r}. Use a different --out."
            )
    if retrieve_k:
        from .knowledge_compaction import LessonRagIndex, chunk_lesson

    def run_cell(fill_tokens: int, fill_pct: int | None, position: str, trial: int) -> None:
        if (fill_tokens, position, trial) in done_keys:
            return  # already on disk (resume)
        needle = NEEDLES[trial % len(NEEDLES)]
        ctx = build_context(enc, filler, fill_tokens, needle, position)
        ctx_tok = len(enc.encode(ctx, disallowed_special=()))
        overflow = bool(window) and ctx_tok > window
        # RAG mode: retrieve top-K chunks for the probe and answer from those --
        # the compaction contrast. The haystack size (ctx_tok) stays the headline
        # "how big a context we're fighting"; seen_tok is what the model read.
        if retrieve_k:
            chunks = chunk_lesson(ctx)
            if args.retrieve_mode == "lexical":
                seen_ctx = lexical_retrieve(chunks, needle.probe, retrieve_k)
            else:
                seen_ctx = LessonRagIndex(chunks).retrieve(
                    needle.probe, top_k=retrieve_k
                )
        else:
            seen_ctx = ctx
        seen_tok = len(enc.encode(seen_ctx, disallowed_special=()))
        # Retry transient API/network errors (e.g. SSL bad-record-mac on the big
        # calls) instead of crashing the whole sweep; skip the cell after 3 tries
        # so --resume can pick it up later rather than recording a false miss.
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                ans, in_tok, out_tok, lat = answer(
                    needle.probe, seen_ctx, cfg, args.format
                )
                jp, jr = judge(needle.probe, ans, needle, judge_cfg)
                break
            except Exception as e:  # noqa: BLE001 - transient provider/network errors
                last_err = e
                logger.warning(
                    "fill=%dk pos=%s t%d attempt %d failed: %s",
                    fill_tokens // 1000,
                    position,
                    trial,
                    attempt + 1,
                    str(e)[:120],
                )
                time.sleep(5 * (attempt + 1))
        else:
            logger.error(
                "fill=%dk pos=%s t%d SKIPPED after retries: %s",
                fill_tokens // 1000,
                position,
                trial,
                str(last_err)[:120],
            )
            return
        # Output-truncation guard: a hit cap (or an empty answer at the cap) means
        # the model never got to state the fact -> this is NOT a clean rot miss.
        # Flag it so it is auditable and never silently counted as lost-in-the-middle.
        truncated = out_tok >= ANSWER_MAX_TOKENS or not ans.strip()
        if truncated:
            logger.warning(
                "fill=%dk pos=%s t%d TRUNCATED (out=%d, empty=%s) -- INVALID rot cell, "
                "raise ANSWER_MAX_TOKENS or disable thinking",
                fill_tokens // 1000,
                position,
                trial,
                out_tok,
                not ans.strip(),
            )
        cell = Cell(
            fill_tokens=fill_tokens,
            fill_pct=fill_pct,
            position=position,
            trial=trial,
            needle_answer=needle.answer,
            context_tokens=ctx_tok,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=lat,
            judge_pass=jp,
            judge_reason=jr,
            answer_text=ans[:500],
            ctx_overflow=overflow,
            mode=mode,
            seen_tokens=seen_tok,
            format=args.format,
            truncated=truncated,
        )
        cells.append(cell)
        with open(bundle, "a", encoding="utf-8") as h:
            h.write(json.dumps(asdict(cell), ensure_ascii=False) + "\n")
        logger.info(
            "fill=%dk pos=%s t%d -> %s (ctx %dk%s, %.1fs)",
            fill_tokens // 1000,
            position,
            trial,
            "PASS" if jp else "FAIL",
            ctx_tok // 1000,
            " OVERFLOW" if overflow else "",
            lat,
        )

    if args.calibration and not args.smoke:
        for trial in range(trials):
            run_cell(args.calibration, None, "middle", trial)

    for fill_tokens, fill_pct in fills:
        for position in positions:
            for trial in range(trials):
                run_cell(fill_tokens, fill_pct, position, trial)

    print(report(cells, out_dir, cfg.model, window))
    print(f"\nBundles: {bundle} | {len(cells)} cells")


if __name__ == "__main__":
    main()
