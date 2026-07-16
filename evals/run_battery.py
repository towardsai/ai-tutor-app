"""Run a test battery against the tutor and save one bundle per turn.

A battery is a JSONL file of related tests, such as one-question chats or
multi-turn sessions. A bundle is the saved JSON record for one tutor turn:
the question, answer, tool calls, source matches, timing, token usage, and
any error. Later commands grade and report from these bundles without calling
the tutor again.

Examples:
  uv run -m evals.run_battery --battery data/eval/battery_singleturn_v1.jsonl \
      --preset prod --out runs/bake1_singleturn_prod
  uv run -m evals.run_battery --battery data/eval/battery_sessions_v1.jsonl \
      --preset full_history --ids s01_fullstack_beginner_13t --trials 2

Notes:
- LangSmith tracing is OFF by default (free-plan quota); --langsmith enables it.
- Web tools are OFF by default for reproducibility; --enable-tools to add.
- Re-running with the same --out resumes: completed cases (all trials of all
  turns) are kept; incomplete sessions are re-run whole, because thread state
  lives in process memory and cannot be resumed across runs.
- Bundles are the durable artifact: grade/report re-run offline against them.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .common import (
    append_jsonl,
    detect_battery_type,
    ensure_battery_not_deprecated,
    load_jsonl,
    write_jsonl,
)

logger = logging.getLogger("evals.run_battery")

# Capture the full tool output the agent saw: the KB shell returns up to
# DEFAULT_MAX_OUTPUT_CHARS = 40k chars/call (app/kb_shell.py), so 40k preserves
# everything the model grounded on. This matters for offline faithfulness
# grading -- under the old 6k cap, KB-browse evidence was truncated, so a
# faithfulness judge scored capture-completeness, not grounding (evals.md
# F23/F24 class). output_chars still records the true length, so any rare
# overflow stays visible. Bundles are bigger (KB-heavy turns ~40k x several
# calls), which is an acceptable cost for local-only runs.
TOOL_OUTPUT_MAX_CHARS = 40_000
# Slowest observed turn is ~2.5 min; anything past this is a wedged stream
# (e.g. laptop sleep killed the connection mid-turn). The turn records a
# TimeoutError and the unit re-runs on resume instead of hanging forever.
# Overridable for slow local models whose multi-call middlewares (e.g.
# hierarchical summarization's map-reduce, ~30 sequential calls) legitimately
# need longer than the cloud default.
TURN_TIMEOUT_SECONDS = int(os.environ.get("AI_TUTOR_EVAL_TURN_TIMEOUT", "600"))
EXPERIMENT_SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery", required=True, help="Path to a battery JSONL.")
    parser.add_argument(
        "--allow-deprecated-battery",
        action="store_true",
        help="Run a battery deprecated after a validity audit (reproduction only).",
    )
    parser.add_argument("--preset", default="prod", help="Memory preset name.")
    parser.add_argument("--model", default="", help="Model id (default: app default).")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--out", default="", help="Output dir (default: derived).")
    parser.add_argument("--limit", type=int, default=0, help="First N cases only.")
    parser.add_argument(
        "--ids", nargs="*", default=[], help="Run only these case/session/persona ids."
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help=(
            "Run only records whose tags or tier match any value here "
            "(useful for tiered v2 batteries)."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--scope-sources",
        action="store_true",
        help="Restrict the tutor to the case's own course (default: all sources, "
        "like production).",
    )
    parser.add_argument(
        "--enable-tools", nargs="*", default=[], help="e.g. web_search url_context"
    )
    parser.add_argument(
        "--disable-kb",
        action="store_true",
        help="Drop the run_kb_command tool + KB prompt (KB on/off ablation).",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Give the agent NO retrieval/KB tools (empty sources): it must "
        "answer purely from the in-context conversation. Used by the "
        "knowledge-compaction study to isolate compaction vs keeping context.",
    )
    parser.add_argument(
        "--retrieval-budget",
        type=int,
        default=0,
        help="Per-request retrieval token budget (Axis B sweep, e.g. 30000); "
        "0 keeps the default 100k.",
    )
    parser.add_argument(
        "--retriever",
        default="",
        choices=["", "classical", "graphrag"],
        help="Retrieval backend (GraphRAG experiment): '' / classical = hybrid "
        "RAG (default); graphrag = GraphRAG over the prebuilt local index.",
    )
    parser.add_argument("--langsmith", action="store_true", help="Enable tracing.")
    return parser.parse_args()


def record_id(record: dict[str, Any]) -> str:
    for key in ("case_id", "session_id", "persona_id", "replay_id"):
        if key in record:
            return str(record[key])
    raise KeyError("record has no id")


def record_tags(record: dict[str, Any]) -> set[str]:
    tags = set(str(tag) for tag in (record.get("tags") or []))
    if record.get("tier"):
        tags.add(str(record["tier"]))
    return tags


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(repo_root: Path) -> str:
    """Hash executable harness/app sources, including untracked Python files."""
    digest = hashlib.sha256()
    paths = [
        path
        for folder in (repo_root / "app", repo_root / "evals")
        for path in folder.rglob("*.py")
    ]
    for path in sorted(paths):
        digest.update(str(path.relative_to(repo_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_value(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def build_run_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """Build the immutable scientific configuration for one battery run."""
    from app.config import DEFAULT_MODEL_NAME
    from app.memory_presets import resolve_memory_preset
    from app.telemetry import MODEL_PRICING

    repo_root = Path(__file__).resolve().parents[1]
    battery_path = Path(args.battery).resolve()
    lock_path = repo_root / "uv.lock"
    semantic_args = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {
            "out",
            "run_fingerprint",
            # Staging controls affect only how much work one paired-runner
            # process performs, not the scientific configuration or rows.
            "max_pairs_this_invocation",
            "first_pair_id",
            "import_completed_from",
            # Operational escape hatch, not scientific configuration.
            "allow_deprecated_battery",
        }
    }
    manifest = {
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "git_sha": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_status": _git_value(repo_root, "status", "--short"),
        "source_tree_sha256": _source_tree_sha256(repo_root),
        "uv_lock_sha256": _sha256_file(lock_path),
        "battery_path": str(battery_path),
        "battery_sha256": _sha256_file(battery_path),
        "memory_config": asdict(resolve_memory_preset(args.preset)),
        "requested_model": args.model or DEFAULT_MODEL_NAME,
        "provider_credentials_present": {
            "deepseek": bool(os.environ.get("DEEPSEEK_API_KEY")),
            "google": bool(
                os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            ),
            "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
        },
        "pricing_snapshot_usd_per_million": {
            key: asdict(value) for key, value in sorted(MODEL_PRICING.items())
        },
        "runner_args": semantic_args,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return manifest, hashlib.sha256(canonical.encode()).hexdigest()


def write_or_validate_run_config(out_dir: Path, args: argparse.Namespace) -> str:
    """Write a manifest once; refuse to mix stale bundles on resume."""
    manifest, fingerprint = build_run_manifest(args)
    config_path = out_dir / "run_config.json"
    bundles_path = out_dir / "bundles.jsonl"
    if config_path.exists() and bundles_path.exists() and bundles_path.stat().st_size:
        existing = json.loads(config_path.read_text())
        existing_fingerprint = existing.get("_fingerprint")
        if existing_fingerprint != fingerprint:
            raise SystemExit(
                "Refusing to resume with configuration drift: "
                f"stored fingerprint={existing_fingerprint!r}, "
                f"current={fingerprint!r}. Use a new --out directory."
            )
        return fingerprint

    payload = {
        **vars(args),
        "_fingerprint": fingerprint,
        "_manifest": manifest,
    }
    config_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return fingerprint


class BundleSink:
    """Append-only bundle store with resume bookkeeping."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = asyncio.Lock()

    async def write(self, rows: list[dict[str, Any]]) -> None:
        async with self.lock:
            await asyncio.to_thread(append_jsonl, self.path, rows)


def prune_incomplete(
    path: Path, expected_turns: dict[str, int]
) -> set[tuple[str, int]]:
    """Drop partial work units from a previous run; return completed (id, trial).

    A work unit is one trial of one case (or one whole session). Sessions with
    missing turns are pruned entirely: their thread state died with the old
    process, so they must re-run from turn 0.
    """
    if not path.exists():
        return set()
    rows = load_jsonl(path)
    by_unit: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_unit.setdefault((row["unit_id"], row["trial"]), []).append(row)
    completed = set()
    for (unit_id, trial), unit_rows in by_unit.items():
        wanted = expected_turns.get(unit_id, 1)
        if len(unit_rows) >= wanted and not any(r.get("error") for r in unit_rows):
            completed.add((unit_id, trial))
    kept = [r for r in rows if (r["unit_id"], r["trial"]) in completed]
    if len(kept) != len(rows):
        write_jsonl(path, kept)
        logger.info(
            "Resume: kept %d completed unit(s), pruned %d partial row(s).",
            len(completed),
            len(rows) - len(kept),
        )
    return completed


def make_bundle(
    *,
    args: argparse.Namespace,
    battery_type: str,
    unit_id: str,
    trial: int,
    turn_index: int | None,
    query: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": f"{unit_id}|turn{turn_index if turn_index is not None else 0}"
        f"|t{trial}",
        "run_fingerprint": getattr(args, "run_fingerprint", ""),
        "unit_id": unit_id,
        "battery_path": args.battery,
        "battery_type": battery_type,
        "preset": args.preset,
        "model": result["model"],
        "trial": trial,
        "turn_index": turn_index,
        "started_at": result["started_at"],
        "duration_ms": result["duration_ms"],
        "query": query,
        "answer": result["answer"],
        "thread_id": result["thread_id"],
        "tool_calls": result["tool_calls"],
        "resolved_sources": result["resolved_sources"],
        "context_stats": result["context_stats"],
        "turn_retry_attempts": int(result.get("turn_retry_attempts") or 0),
        "turn_attempt_failures": list(result.get("turn_attempt_failures") or []),
        "failed_attempt_usage_unavailable": bool(
            result.get("failed_attempt_usage_unavailable")
        ),
        "error": result["error"],
    }


def validate_experiment_result(
    args: argparse.Namespace, result: dict[str, Any]
) -> None:
    """Turn silent instrumentation/provider drift into a recorded run error."""
    from app.config import DEFAULT_MODEL_NAME
    from app.chat_service import model_provider_and_name
    from app.memory_presets import resolve_memory_preset

    config = resolve_memory_preset(args.preset)
    if not config.experiment_mode or result.get("error"):
        return
    stats = result.get("context_stats") or {}
    calls = stats.get("model_calls") or []
    problems: list[str] = []
    if not stats:
        problems.append("missing context_stats")
    if not calls:
        problems.append("missing per-call model telemetry")
    if len(calls) != int(stats.get("llm_calls") or 0):
        problems.append("per-call telemetry count does not match llm_calls")

    provider, expected_model = model_provider_and_name(args.model or DEFAULT_MODEL_NAME)
    actual_models = {str(call.get("model") or "") for call in calls}
    if actual_models and actual_models != {expected_model}:
        problems.append(
            f"expected only {expected_model}, observed {sorted(actual_models)}"
        )
    for call in calls:
        if not call.get("usage_reported"):
            problems.append(f"call {call.get('sequence')} missing token usage")
        if provider == "deepseek" and not call.get("cache_details_reported"):
            problems.append(
                f"call {call.get('sequence')} missing DeepSeek cache telemetry"
            )
    if stats.get("cost_breakdown") is None:
        problems.append("missing cost breakdown")
    for event in stats.get("compaction_events") or []:
        configured_trigger = int(event.get("configured_trigger_tokens") or 0)
        if configured_trigger != config.summarization_trigger_tokens:
            problems.append(
                "compaction event trigger does not match resolved preset "
                f"({configured_trigger} != {config.summarization_trigger_tokens})"
            )
        approximate_tokens = int(event.get("pre_compaction_tokens_approx") or 0)
        reported_tokens = int(event.get("trigger_reported_tokens") or 0)
        if max(approximate_tokens, reported_tokens) < configured_trigger:
            problems.append(
                "neither approximate nor provider-reported tokens reached the "
                "configured compaction trigger"
            )
        if not event.get("summary_input_untrimmed"):
            problems.append("summarizer input was trimmed")
        if int(event.get("summary_input_tokens_approx") or 0) <= 4_000:
            problems.append("summarizer input did not exceed the historical 4k cap")
        if config.summarization_strategy == "structured_prefix":
            if event.get("summary_strategy") != "structured_prefix":
                problems.append("structured-prefix compaction strategy not recorded")
            if not event.get("summary_system_message_present"):
                problems.append("structured-prefix summary omitted the system message")
            if not event.get("summary_cache_user_id_preserved"):
                problems.append("structured-prefix summary omitted the cache user_id")
            if not event.get("summary_provider_usage_reported"):
                problems.append("structured-prefix summary missing provider usage")
            if provider == "deepseek" and not event.get(
                "summary_provider_cache_details_reported"
            ):
                problems.append(
                    "structured-prefix summary missing DeepSeek cache telemetry"
                )
    if problems:
        result["error"] = "HarnessValidationError: " + "; ".join(problems)


async def run_turn(request: Any) -> dict[str, Any]:
    """Drive one stream_chat turn and collect everything the graders need."""
    from app.chat_service import stream_chat
    from app.config import DEFAULT_MODEL_NAME

    started_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    tool_calls: list[dict[str, Any]] = []
    resolved_sources: list[dict[str, Any]] = []
    answer, thread_id, error = "", "", None
    context_stats: dict[str, Any] | None = None
    try:
        async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
            async for event in stream_chat(request):
                if event.type == "thread_started":
                    thread_id = str(event.data.get("thread_id", ""))
                elif event.type == "tool_call_completed":
                    data = event.data
                    output_text = str(data.get("output_text") or "")
                    original_chars = data.get("output_original_chars")
                    tool_calls.append(
                        {
                            "tool_name": data.get("tool_name"),
                            "args_text": data.get("args_text", ""),
                            "output_text": output_text[:TOOL_OUTPUT_MAX_CHARS],
                            # Completeness is relative to what the model saw.
                            # The stable cap is part of the treatment, not a
                            # bundle-capture failure; retain pre-cap size below
                            # as separate diagnostic metadata.
                            "output_chars": len(output_text),
                            "output_original_chars": (
                                int(original_chars)
                                if isinstance(original_chars, int)
                                else len(output_text)
                            ),
                            "output_was_capped": bool(data.get("output_was_capped")),
                            "output_original_bytes": data.get("output_original_bytes"),
                            "output_retained_bytes": data.get("output_retained_bytes"),
                            "output_sha256": data.get("output_sha256"),
                            "matches": [
                                {
                                    key: match.get(key)
                                    for key in ("source_key", "url", "title", "score")
                                }
                                for match in data.get("matches") or []
                            ],
                        }
                    )
                elif event.type == "source_match":
                    resolved_sources.append(
                        {
                            key: event.data.get(key)
                            for key in ("source_key", "url", "title", "group")
                        }
                    )
                elif event.type == "context_stats":
                    context_stats = dict(event.data)
                elif event.type == "message_completed":
                    answer = str(event.data.get("answer", ""))
    except Exception as exc:  # noqa: BLE001 - record and continue the battery
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("Turn failed: %s", error)
    return {
        "model": request.model_name or DEFAULT_MODEL_NAME,
        "started_at": started_at,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "answer": answer,
        "thread_id": thread_id,
        "tool_calls": tool_calls,
        "resolved_sources": resolved_sources,
        "context_stats": context_stats,
        "error": error,
    }


def build_request(
    args: argparse.Namespace,
    *,
    query: str,
    source_key: str | None,
    history: tuple[Any, ...] = (),
    thread_id: str = "",
    student_id: str = "",
    cache_user_id: str = "",
) -> Any:
    from app.chat_types import ChatRequest
    from app.config import DEFAULT_MODEL_NAME, DEFAULT_SELECTED_SOURCE_KEYS

    if getattr(args, "no_tools", False):
        # Empty sources -> include_local_tools is False in the app, so the agent
        # gets no retrieval/KB tools and answers only from conversation context.
        source_keys: tuple[str, ...] = ()
    elif args.scope_sources and source_key:
        source_keys = (source_key,)
    else:
        source_keys = tuple(DEFAULT_SELECTED_SOURCE_KEYS)
    return ChatRequest(
        query=query,
        history=history,
        source_keys=source_keys,
        model_name=args.model or DEFAULT_MODEL_NAME,
        include_reasoning=False,
        thread_id=thread_id,
        enabled_tools=tuple(args.enable_tools),
        memory_preset=args.preset,
        student_id=student_id,
        cache_user_id=cache_user_id,
        disable_kb=args.disable_kb,
        retrieval_budget=args.retrieval_budget or None,
        retriever=args.retriever,
    )


def experiment_cache_user_id(
    preset: str, unit_id: str, trial: int, *, namespace: str = ""
) -> str:
    """Opaque stable DeepSeek KV-cache namespace for one trajectory."""
    digest = hashlib.sha256(
        f"{namespace}|{preset}|{unit_id}|{trial}".encode()
    ).hexdigest()[:24]
    return f"eval_{digest}"


async def run_single_case(
    args: argparse.Namespace,
    battery_type: str,
    record: dict[str, Any],
    trial: int,
    sink: BundleSink,
) -> None:
    """singleturn / personas-question / replay: one independent turn."""
    from app.chat_types import ChatTurn

    if record.get("_profile_seed"):
        # Re-seed right before every persona question: under profile_memory
        # the post-turn write-back would otherwise drift the profile between
        # questions (and across concurrent trials), making results
        # order-dependent. Each question grades the canonical seeded profile.
        from app.chat_service import set_student_profile

        set_student_profile(record["_student_id"], record["_profile_seed"])

    unit_id = record["_unit_id"]
    query = record["_query"]
    history = tuple(
        ChatTurn(role=turn["role"], content=turn["content"])
        for turn in record.get("_history", [])
    )
    request = build_request(
        args,
        query=query,
        source_key=record.get("source_key"),
        history=history,
        student_id=record.get("_student_id", ""),
        cache_user_id=experiment_cache_user_id(
            args.preset,
            unit_id,
            trial,
            namespace=getattr(args, "run_fingerprint", ""),
        ),
    )
    result = await run_turn(request)
    validate_experiment_result(args, result)
    await sink.write(
        [
            make_bundle(
                args=args,
                battery_type=battery_type,
                unit_id=unit_id,
                trial=trial,
                turn_index=None,
                query=query,
                result=result,
            )
        ]
    )


async def run_session(
    args: argparse.Namespace,
    session: dict[str, Any],
    trial: int,
    sink: BundleSink,
) -> None:
    """All turns sequentially on one thread, passing the visible transcript
    back each turn exactly like the real frontend does."""
    from app.chat_types import ChatTurn
    from app.chat_service import set_student_profile
    from app.memory_presets import resolve_memory_preset

    history: list[ChatTurn] = []
    thread_id = ""
    rows = []
    student_id = ""
    if resolve_memory_preset(args.preset).longterm_memory:
        # Engage long-term memory on session batteries. Without a student_id,
        # profile_memory's system-prompt injection and write-back both no-op,
        # making it indistinguishable from prod on session probes.
        student_id = f"{session['session_id']}|t{trial}"
        set_student_profile(student_id, "")
    for turn_index, query in enumerate(session["turns"]):
        request = build_request(
            args,
            query=query,
            source_key=session.get("source_key"),
            history=tuple(history),
            thread_id=thread_id,
            student_id=student_id,
            cache_user_id=experiment_cache_user_id(
                args.preset,
                session["session_id"],
                trial,
                namespace=getattr(args, "run_fingerprint", ""),
            ),
        )
        result = await run_turn(request)
        validate_experiment_result(args, result)
        thread_id = result["thread_id"] or thread_id
        rows.append(
            make_bundle(
                args=args,
                battery_type="sessions",
                unit_id=session["session_id"],
                trial=trial,
                turn_index=turn_index,
                query=query,
                result=result,
            )
        )
        if result["error"]:
            logger.warning(
                "Session %s trial %d aborted at turn %d.",
                session["session_id"],
                trial,
                turn_index,
            )
            break
        history.append(ChatTurn("user", query.strip()))
        history.append(ChatTurn("assistant", result["answer"]))
    await sink.write(rows)


def prepare_units(
    battery_type: str, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Flatten batteries into uniform work units with _unit_id/_query fields."""
    if battery_type == "singleturn":
        for record in records:
            record["_unit_id"] = record["case_id"]
            record["_query"] = record["question"]
        return records
    if battery_type == "replay":
        for record in records:
            record["_unit_id"] = record["replay_id"]
            record["_query"] = record["history"][-1]["content"]
            record["_history"] = record["history"][:-1]
        return records
    if battery_type == "personas":
        units = []
        for persona in records:
            for question in persona["questions"]:
                units.append(
                    {
                        "_unit_id": question["question_id"],
                        "_query": question["question"],
                        "_student_id": persona["persona_id"],
                        "_profile_seed": persona["profile_seed"],
                        "source_key": persona.get("source_key"),
                    }
                )
        return units
    return records  # sessions keep their own shape


async def run_all(args: argparse.Namespace) -> Path:
    ensure_battery_not_deprecated(args.battery, override=args.allow_deprecated_battery)
    records = load_jsonl(args.battery)
    battery_type = detect_battery_type(records)
    if args.ids:
        wanted = set(args.ids)
        key = {"personas": "persona_id"}.get(battery_type)
        records = [
            r
            for r in records
            if record_id(r) in wanted or (key and r.get(key) in wanted)
        ]
    if args.tags:
        wanted_tags = set(args.tags)
        records = [r for r in records if record_tags(r) & wanted_tags]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No records selected.")

    units = prepare_units(battery_type, records)
    out_dir = Path(
        args.out or f"runs/{datetime.date.today():%Y%m%d}_{battery_type}_{args.preset}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    args.run_fingerprint = write_or_validate_run_config(out_dir, args)
    sink = BundleSink(out_dir / "bundles.jsonl")

    expected_turns = {
        unit["session_id"] if battery_type == "sessions" else unit["_unit_id"]: (
            len(unit["turns"]) if battery_type == "sessions" else 1
        )
        for unit in units
    }
    completed = prune_incomplete(sink.path, expected_turns)

    semaphore = asyncio.Semaphore(args.concurrency)
    # Same-persona questions are serialized: each one re-seeds the canonical
    # profile (run_single_case) and profile_memory's post-turn write-back must
    # not land mid-way through a sibling question's turn. Different personas
    # still run concurrently.
    student_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    pending = 0

    async def guarded(unit: dict[str, Any], trial: int) -> None:
        async with semaphore:
            if battery_type == "sessions":
                await run_session(args, unit, trial, sink)
            elif unit.get("_student_id"):
                async with student_locks[unit["_student_id"]]:
                    await run_single_case(args, battery_type, unit, trial, sink)
            else:
                await run_single_case(args, battery_type, unit, trial, sink)

    tasks = []
    for unit in units:
        unit_id = unit["session_id"] if battery_type == "sessions" else unit["_unit_id"]
        for trial in range(1, args.trials + 1):
            if (unit_id, trial) in completed:
                continue
            pending += 1
            tasks.append(asyncio.create_task(guarded(unit, trial)))
    logger.info(
        "Running %d work unit(s) (%d already complete) -> %s",
        pending,
        len(completed),
        out_dir,
    )
    if tasks:
        await asyncio.gather(*tasks)
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    # Must happen before any app import: app.config enables tracing at import
    # time when LANGSMITH_API_KEY is set, and batch runs would eat the
    # free-plan trace quota.
    if not args.langsmith:
        os.environ["LANGSMITH_TRACING"] = "false"
    out_dir = asyncio.run(run_all(args))
    print(f"Bundles written to {out_dir}/bundles.jsonl")
    print(f"Next: uv run -m evals.grade --run {out_dir}")


if __name__ == "__main__":
    main()
