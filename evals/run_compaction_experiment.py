"""Run session-memory arms in randomized lockstep.

Each session/trial owns one independent thread and DeepSeek ``user_id`` per arm.
At turn N, every arm runs turn N in a deterministic shuffled order before any
arm advances to N+1. This controls provider-load timing without donating cache
entries across arms.

Example (Stage 1):

  uv run -m evals.run_compaction_experiment \
      --battery data/eval/battery_sessions_v2.jsonl \
      --tags tier1_contradiction tier2_longhorizon \
      --trials 3 --out runs/deepseek_compaction_stage1

This command makes paid model calls. It is intentionally separate from the
ordinary single-arm runner so an accidental default invocation cannot launch
the four-arm battery.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
import traceback
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .common import (
    detect_battery_type,
    ensure_battery_not_deprecated,
    load_jsonl,
    write_jsonl,
)
from .run_battery import (
    BundleSink,
    build_request,
    experiment_cache_user_id,
    make_bundle,
    record_tags,
    run_turn,
    validate_experiment_result,
    write_or_validate_run_config,
)

logger = logging.getLogger("evals.run_compaction_experiment")

STAGE1_PRESETS = (
    "exp_fh_raw",
    "exp_fh_cap10k",
    "exp_c200_raw",
    "exp_c200_cap10k",
)
TURN_MAX_ATTEMPTS = 3
TURN_RETRY_BASE_DELAY_SECONDS = 1.0
RUN_STATUS_FILENAME = "run_status.json"
MIGRATION_FILENAME = "compatibility_migration.json"
TRIGGER_VALIDATION_MIGRATION_ID = "provider_reported_trigger_validation_v1"
TRIGGER_VALIDATION_BASELINE_SOURCE_SHA256 = (
    "f7f2f675d9a961e2bd9b870e0eabdca1494048c0476e16ff668fa010ee2efe27"
)
MIGRATION_ALLOWED_MANIFEST_DRIFT = frozenset({"git_status", "source_tree_sha256"})
_RETRYABLE_TURN_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "InternalServerError",
    "PoolTimeout",
    "RateLimitError",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "TimeoutError",
    "WriteError",
    "WriteTimeout",
}


def _is_retryable_turn_error(error: str | None) -> bool:
    if not error:
        return False
    error_name = error.partition(":")[0].strip()
    return error_name in _RETRYABLE_TURN_ERROR_NAMES


def _bundle_progress(
    sinks: dict[str, BundleSink], expected_turns: dict[str, int]
) -> dict[str, Any]:
    """Read the durable bundles once for a final or failure status snapshot."""
    completed_by_arm: list[set[tuple[str, int]]] = []
    arms: dict[str, dict[str, int]] = {}
    for preset, sink in sinks.items():
        rows = load_jsonl(sink.path) if sink.path.exists() else []
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault((row["unit_id"], row["trial"]), []).append(row)
        complete = {
            key
            for key, unit_rows in grouped.items()
            if len(unit_rows) == expected_turns.get(key[0], 0)
            and not any(row.get("error") for row in unit_rows)
        }
        completed_by_arm.append(complete)
        arms[preset] = {
            "rows": len(rows),
            "error_rows": sum(bool(row.get("error")) for row in rows),
            "completed_pairs": len(complete),
            "turn_retries": sum(
                int(row.get("turn_retry_attempts") or 0) for row in rows
            ),
            "failed_attempt_usage_gaps": sum(
                bool(row.get("failed_attempt_usage_unavailable")) for row in rows
            ),
        }
    common = set.intersection(*completed_by_arm) if completed_by_arm else set()
    return {"arms": arms, "completed_pairs_all_arms": len(common)}


def _write_run_status(
    root: Path,
    *,
    state: str,
    args: argparse.Namespace,
    progress: dict[str, Any] | None = None,
    error: BaseException | None = None,
    formatted_traceback: str = "",
) -> Path:
    """Atomically publish the small file consumed by the Codex monitor."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "state": state,
        "updated_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "run_root": str(root.resolve()),
        "configuration": {
            "battery": str(Path(args.battery).resolve()),
            "presets": list(args.presets),
            "trials": args.trials,
            "arm_concurrency": args.arm_concurrency,
            "pair_concurrency": args.pair_concurrency,
            "import_completed_from": args.import_completed_from,
        },
        "progress": progress or {},
    }
    if error is not None:
        payload["fatal_error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": formatted_traceback,
        }
    path = root / RUN_STATUS_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


async def _run_turn_with_retries(request: Any) -> dict[str, Any]:
    """Retry transient aborted streams without multiplying semantic failures."""
    started = time.monotonic()
    first_started_at = ""
    failures: list[dict[str, Any]] = []
    for attempt in range(1, TURN_MAX_ATTEMPTS + 1):
        result = await run_turn(request)
        first_started_at = first_started_at or str(result.get("started_at") or "")
        error = str(result.get("error") or "")
        if error:
            retryable = _is_retryable_turn_error(error)
            failures.append(
                {
                    "attempt": attempt,
                    "error": error,
                    "duration_ms": int(result.get("duration_ms") or 0),
                    "retryable": retryable,
                    "context_stats_present": bool(result.get("context_stats")),
                }
            )
            if retryable and attempt < TURN_MAX_ATTEMPTS:
                delay = TURN_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Retrying transient turn failure after attempt %d/%d in %.1fs. "
                    "error=%s",
                    attempt,
                    TURN_MAX_ATTEMPTS,
                    delay,
                    error,
                )
                await asyncio.sleep(delay)
                continue

        result["started_at"] = first_started_at
        result["duration_ms"] = int((time.monotonic() - started) * 1000)
        result["turn_retry_attempts"] = attempt - 1
        result["turn_attempt_failures"] = failures
        # Aborted DeepSeek streams do not provide the final usage chunk. Keep
        # this explicit rather than pretending the successful retry's bill is
        # the complete operational cost of the failed+successful attempts.
        result["failed_attempt_usage_unavailable"] = bool(failures)
        stats = result.get("context_stats")
        if isinstance(stats, dict):
            stats["turn_retry_attempts"] = attempt - 1
            stats["failed_attempt_usage_unavailable"] = bool(failures)
        return result
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery", required=True)
    parser.add_argument(
        "--allow-deprecated-battery",
        action="store_true",
        help="Run a battery deprecated after a validity audit (reproduction only).",
    )
    parser.add_argument("--out", required=True, help="Root directory; one subdir/arm.")
    parser.add_argument("--presets", nargs="+", default=list(STAGE1_PRESETS))
    parser.add_argument("--model", default="")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--arm-concurrency",
        type=int,
        default=1,
        help="Arms allowed to run concurrently within one paired turn.",
    )
    parser.add_argument(
        "--pair-concurrency",
        type=int,
        default=1,
        help="Independent session/trial pairs allowed to run concurrently.",
    )
    parser.add_argument(
        "--max-pairs-this-invocation",
        type=int,
        default=0,
        help="Operational staging limit; 0 runs every pending pair.",
    )
    parser.add_argument(
        "--first-pair-id",
        default="",
        help="Operationally prioritize this session without changing the manifest.",
    )
    parser.add_argument(
        "--import-completed-from",
        default="",
        help=(
            "Import only common-complete pairs from the exact pre-fix Stage 1 "
            "source run after strict manifest compatibility checks."
        ),
    )
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--scope-sources", action="store_true")
    parser.add_argument("--enable-tools", nargs="*", default=[])
    parser.add_argument("--disable-kb", action="store_true")
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--retrieval-budget", type=int, default=0)
    parser.add_argument(
        "--retriever", default="", choices=["", "classical", "graphrag"]
    )
    parser.add_argument("--langsmith", action="store_true")
    return parser.parse_args()


def _arm_args(args: argparse.Namespace, preset: str, out: Path) -> argparse.Namespace:
    values = dict(vars(args))
    values.update(
        {
            "preset": preset,
            "out": str(out),
            "limit": 0,
            "concurrency": 1,
        }
    )
    return argparse.Namespace(**values)


def _completed_units(
    path: Path, expected_turns: dict[str, int]
) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        grouped.setdefault((row["unit_id"], row["trial"]), []).append(row)
    return {
        key
        for key, rows in grouped.items()
        if len(rows) == expected_turns.get(key[0], 0)
        and not any(row.get("error") for row in rows)
    }


def _prune_to_common_completed(
    paths: list[Path], expected_turns: dict[str, int]
) -> set[tuple[str, int]]:
    """Resume only pairs completed in every arm; prune all partial/asymmetric work."""
    if not paths:
        return set()
    completed_by_arm = [_completed_units(path, expected_turns) for path in paths]
    common = set.intersection(*completed_by_arm) if completed_by_arm else set()
    for path in paths:
        if not path.exists():
            continue
        rows = load_jsonl(path)
        kept = [row for row in rows if (row["unit_id"], row["trial"]) in common]
        if len(kept) != len(rows):
            write_jsonl(path, kept)
    return common


def _manifest_without_allowed_migration_drift(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in MIGRATION_ALLOWED_MANIFEST_DRIFT
    }


def _load_run_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing migration run config: {path}")
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise SystemExit(f"Cannot read migration run config {path}: {error}") from error
    if not payload.get("_fingerprint") or not payload.get("_manifest"):
        raise SystemExit(f"Incomplete migration run config: {path}")
    return payload


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".migration.tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def _import_compatible_completed_pairs(
    *,
    source_root: Path,
    target_root: Path,
    presets: list[str],
    target_fingerprints: dict[str, str],
    expected_turns: dict[str, int],
    eligible_pairs: set[tuple[str, int]],
) -> set[tuple[str, int]]:
    """Import exact-baseline pairs after proving all scientific inputs match.

    The only permitted manifest drift is the source-tree hash and redundant git
    status produced by this telemetry/validation fix. Imported rows are
    re-namespaced to the target fingerprint while retaining their original
    fingerprint and source-tree hash in per-row migration metadata.
    """
    source_root = source_root.resolve()
    target_root = target_root.resolve()
    if source_root == target_root:
        raise SystemExit("Migration source and target run roots must differ.")

    source_configs: dict[str, dict[str, Any]] = {}
    target_configs: dict[str, dict[str, Any]] = {}
    source_paths: dict[str, Path] = {}
    target_paths: dict[str, Path] = {}
    for preset in presets:
        source_config = _load_run_config(source_root / preset / "run_config.json")
        target_config = _load_run_config(target_root / preset / "run_config.json")
        source_manifest = source_config["_manifest"]
        target_manifest = target_config["_manifest"]
        source_hash = source_manifest.get("source_tree_sha256")
        if source_hash != TRIGGER_VALIDATION_BASELINE_SOURCE_SHA256:
            raise SystemExit(
                f"Refusing migration for {preset}: source-tree hash {source_hash!r} "
                "is not the audited pre-fix Stage 1 baseline."
            )
        source_semantic = _manifest_without_allowed_migration_drift(source_manifest)
        target_semantic = _manifest_without_allowed_migration_drift(target_manifest)
        if source_semantic != target_semantic:
            differing = sorted(
                key
                for key in set(source_semantic) | set(target_semantic)
                if source_semantic.get(key) != target_semantic.get(key)
            )
            raise SystemExit(
                f"Refusing migration for {preset}: scientific manifest drift in "
                f"{differing}."
            )
        if target_config["_fingerprint"] != target_fingerprints[preset]:
            raise SystemExit(
                f"Refusing migration for {preset}: target fingerprint mismatch."
            )
        source_configs[preset] = source_config
        target_configs[preset] = target_config
        source_paths[preset] = source_root / preset / "bundles.jsonl"
        target_paths[preset] = target_root / preset / "bundles.jsonl"

    completed_by_arm = [
        _completed_units(source_paths[preset], expected_turns) for preset in presets
    ]
    common = (
        set.intersection(*completed_by_arm) if completed_by_arm else set()
    ) & eligible_pairs
    migration_path = target_root / MIGRATION_FILENAME

    if migration_path.exists():
        migration = json.loads(migration_path.read_text())
        if (
            migration.get("migration_id") != TRIGGER_VALIDATION_MIGRATION_ID
            or Path(migration.get("source_run_root", "")).resolve() != source_root
            or migration.get("required_source_tree_sha256")
            != TRIGGER_VALIDATION_BASELINE_SOURCE_SHA256
        ):
            raise SystemExit(
                "Existing compatibility migration does not match this source run."
            )
        imported = {
            (str(pair["unit_id"]), int(pair["trial"]))
            for pair in migration.get("imported_pairs", [])
        }
        if not imported <= common:
            raise SystemExit(
                "Existing compatibility migration references source pairs that are "
                "no longer common-complete."
            )
        for preset in presets:
            rows = load_jsonl(target_paths[preset])
            imported_rows = [
                row for row in rows if (row["unit_id"], int(row["trial"])) in imported
            ]
            if len(imported_rows) != sum(
                expected_turns[unit_id] for unit_id, _trial in imported
            ):
                raise SystemExit(
                    f"Existing migration rows are incomplete for {preset}."
                )
            for row in imported_rows:
                provenance = row.get("migration") or {}
                if (
                    row.get("run_fingerprint") != target_fingerprints[preset]
                    or provenance.get("id") != TRIGGER_VALIDATION_MIGRATION_ID
                    or provenance.get("source_run_fingerprint")
                    != source_configs[preset]["_fingerprint"]
                ):
                    raise SystemExit(
                        f"Existing migration provenance is invalid for {preset}."
                    )
        return imported

    nonempty_targets = [
        str(path)
        for path in target_paths.values()
        if path.exists() and path.stat().st_size
    ]
    if nonempty_targets:
        raise SystemExit(
            "Refusing to import into nonempty target bundles without an existing "
            f"migration record: {nonempty_targets}"
        )

    migrated_by_arm: dict[str, list[dict[str, Any]]] = {}
    arm_record: dict[str, dict[str, Any]] = {}
    for preset in presets:
        source_fingerprint = source_configs[preset]["_fingerprint"]
        migrated_rows: list[dict[str, Any]] = []
        for original in load_jsonl(source_paths[preset]):
            pair = (str(original["unit_id"]), int(original["trial"]))
            if pair not in common:
                continue
            if original.get("run_fingerprint") != source_fingerprint:
                raise SystemExit(
                    f"Refusing migration for {preset}: row fingerprint drift in {pair}."
                )
            row = dict(original)
            row["run_fingerprint"] = target_fingerprints[preset]
            row["migration"] = {
                "id": TRIGGER_VALIDATION_MIGRATION_ID,
                "reason": (
                    "Telemetry and validation now recognize LangChain's "
                    "provider-reported token trigger; model inputs and treatments "
                    "are unchanged."
                ),
                "source_run_root": str(source_root),
                "source_run_fingerprint": source_fingerprint,
                "source_source_tree_sha256": (
                    TRIGGER_VALIDATION_BASELINE_SOURCE_SHA256
                ),
            }
            migrated_rows.append(row)
        expected_rows = sum(expected_turns[unit_id] for unit_id, _trial in common)
        if len(migrated_rows) != expected_rows:
            raise SystemExit(
                f"Refusing migration for {preset}: expected {expected_rows} common "
                f"rows, found {len(migrated_rows)}."
            )
        migrated_by_arm[preset] = migrated_rows
        arm_record[preset] = {
            "source_run_fingerprint": source_fingerprint,
            "target_run_fingerprint": target_fingerprints[preset],
            "rows_imported": len(migrated_rows),
        }

    for preset, rows in migrated_by_arm.items():
        _atomic_write_jsonl(target_paths[preset], rows)
    migration = {
        "schema_version": 1,
        "migration_id": TRIGGER_VALIDATION_MIGRATION_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "source_run_root": str(source_root),
        "target_run_root": str(target_root),
        "required_source_tree_sha256": TRIGGER_VALIDATION_BASELINE_SOURCE_SHA256,
        "allowed_manifest_drift_keys": sorted(MIGRATION_ALLOWED_MANIFEST_DRIFT),
        "imported_pairs": [
            {"unit_id": unit_id, "trial": trial} for unit_id, trial in sorted(common)
        ],
        "arms": arm_record,
    }
    temporary = migration_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(migration, indent=2, sort_keys=True) + "\n")
    temporary.replace(migration_path)
    return common


def _schedule_pending(
    pending: list[tuple[dict[str, Any], int]],
    *,
    seed: int,
    first_pair_id: str = "",
    max_pairs: int = 0,
) -> list[tuple[dict[str, Any], int]]:
    """Deterministically schedule this invocation's subset of pending pairs."""
    scheduled = list(pending)
    random.Random(seed).shuffle(scheduled)
    if first_pair_id:
        candidates = [
            (trial, index)
            for index, (session, trial) in enumerate(scheduled)
            if session["session_id"] == first_pair_id
        ]
        if not candidates:
            raise SystemExit(
                f"--first-pair-id {first_pair_id!r} is not pending in this run."
            )
        _, index = min(candidates)
        scheduled.insert(0, scheduled.pop(index))
    return scheduled[:max_pairs] if max_pairs else scheduled


async def _run_session_trial(
    *,
    session: dict[str, Any],
    trial: int,
    arm_args: dict[str, argparse.Namespace],
    sinks: dict[str, BundleSink],
    seed: int,
    arm_concurrency: int,
    progress_callback: Callable[[str, int, int], Awaitable[None]] | None = None,
) -> None:
    from app.chat_types import ChatTurn

    state = {preset: {"history": [], "thread_id": ""} for preset in arm_args}
    session_id = session["session_id"]
    arm_semaphore = asyncio.Semaphore(arm_concurrency)
    for turn_index, query in enumerate(session["turns"]):
        order = list(arm_args)
        random.Random(f"{seed}|{session_id}|{trial}|{turn_index}").shuffle(order)

        async def run_arm(position: int, preset: str) -> None:
            async with arm_semaphore:
                args = arm_args[preset]
                arm_state = state[preset]
                request = build_request(
                    args,
                    query=query,
                    source_key=session.get("source_key"),
                    history=tuple(arm_state["history"]),
                    thread_id=arm_state["thread_id"],
                    cache_user_id=experiment_cache_user_id(
                        preset,
                        session_id,
                        trial,
                        namespace=getattr(args, "run_fingerprint", ""),
                    ),
                )
                result = await _run_turn_with_retries(request)
                validate_experiment_result(args, result)
                arm_state["thread_id"] = result["thread_id"] or arm_state["thread_id"]
                bundle = make_bundle(
                    args=args,
                    battery_type="sessions",
                    unit_id=session_id,
                    trial=trial,
                    turn_index=turn_index,
                    query=query,
                    result=result,
                )
                bundle.update(
                    {
                        "interleave_order": order,
                        "interleave_position": position,
                        "interleave_seed": seed,
                        "arm_concurrency": arm_concurrency,
                        "pair_concurrency": getattr(args, "pair_concurrency", 1),
                    }
                )
                await sinks[preset].write([bundle])
                if result.get("error"):
                    raise RuntimeError(
                        f"{session_id} trial {trial} {preset} turn {turn_index}: "
                        f"{result['error']}"
                    )
                arm_state["history"].append(ChatTurn("user", query.strip()))
                arm_state["history"].append(ChatTurn("assistant", result["answer"]))

        # Every arm finishes turn N before any arm may start turn N+1. The
        # seeded order remains the deterministic launch/wave order when the
        # concurrency bound is smaller than the number of arms.
        await asyncio.gather(
            *(run_arm(position, preset) for position, preset in enumerate(order))
        )
        if progress_callback is not None:
            await progress_callback(session_id, trial, turn_index)


async def run_all(args: argparse.Namespace) -> Path:
    from app.memory_presets import resolve_memory_preset

    ensure_battery_not_deprecated(args.battery, override=args.allow_deprecated_battery)
    records = load_jsonl(args.battery)
    if detect_battery_type(records) != "sessions":
        raise SystemExit("Lockstep compaction experiments require a session battery.")
    if args.ids:
        wanted = set(args.ids)
        records = [row for row in records if row["session_id"] in wanted]
    if args.tags:
        wanted_tags = set(args.tags)
        records = [row for row in records if record_tags(row) & wanted_tags]
    if not records:
        raise SystemExit("No session records selected.")
    if args.arm_concurrency < 1 or args.pair_concurrency < 1:
        raise SystemExit("--arm-concurrency and --pair-concurrency must be >= 1.")
    if args.max_pairs_this_invocation < 0:
        raise SystemExit("--max-pairs-this-invocation must be >= 0.")
    if len(set(args.presets)) != len(args.presets):
        raise SystemExit("--presets contains duplicates.")
    for preset in args.presets:
        config = resolve_memory_preset(preset)
        if not config.experiment_mode:
            raise SystemExit(f"{preset!r} is not an experiment-mode preset.")

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    arm_args: dict[str, argparse.Namespace] = {}
    sinks: dict[str, BundleSink] = {}
    for preset in args.presets:
        out = root / preset
        out.mkdir(parents=True, exist_ok=True)
        resolved = _arm_args(args, preset, out)
        resolved.run_fingerprint = write_or_validate_run_config(out, resolved)
        arm_args[preset] = resolved
        sinks[preset] = BundleSink(out / "bundles.jsonl")

    expected_turns = {row["session_id"]: len(row["turns"]) for row in records}
    if args.import_completed_from:
        eligible_pairs = {
            (session["session_id"], trial)
            for session in records
            for trial in range(1, args.trials + 1)
        }
        imported = _import_compatible_completed_pairs(
            source_root=Path(args.import_completed_from),
            target_root=root,
            presets=list(args.presets),
            target_fingerprints={
                preset: resolved.run_fingerprint
                for preset, resolved in arm_args.items()
            },
            expected_turns=expected_turns,
            eligible_pairs=eligible_pairs,
        )
        logger.info(
            "Imported %d common-complete paired session-trials from %s.",
            len(imported),
            Path(args.import_completed_from).resolve(),
        )
    completed = _prune_to_common_completed(
        [sink.path for sink in sinks.values()], expected_turns
    )
    pending = [
        (session, trial)
        for session in records
        for trial in range(1, args.trials + 1)
        if (session["session_id"], trial) not in completed
    ]
    total_pending = len(pending)
    pending = _schedule_pending(
        pending,
        seed=args.seed,
        first_pair_id=args.first_pair_id,
        max_pairs=args.max_pairs_this_invocation,
    )
    logger.info(
        "Running %d/%d pending paired session-trials "
        "(%d already complete) across %d arms.",
        len(pending),
        total_pending,
        len(completed),
        len(args.presets),
    )
    progress: dict[str, Any] = {
        "total_pairs": len(records) * args.trials,
        "completed_pairs_before_invocation": len(completed),
        "completed_pairs_this_invocation": 0,
        "completed_pairs_all_arms": len(completed),
        "pending_pairs_total": total_pending,
        "scheduled_pairs_this_invocation": len(pending),
        "turn_barriers_completed_this_invocation": 0,
        "last_completed_turn": None,
    }
    _write_run_status(root, state="running", args=args, progress=progress)
    status_lock = asyncio.Lock()
    pair_semaphore = asyncio.Semaphore(args.pair_concurrency)

    async def report_turn(session_id: str, trial: int, turn_index: int) -> None:
        async with status_lock:
            progress["turn_barriers_completed_this_invocation"] += 1
            progress["last_completed_turn"] = {
                "session_id": session_id,
                "trial": trial,
                "turn_index": turn_index,
            }
            _write_run_status(root, state="running", args=args, progress=progress)

    async def run_pair(session: dict[str, Any], trial: int) -> None:
        async with pair_semaphore:
            await _run_session_trial(
                session=session,
                trial=trial,
                arm_args=arm_args,
                sinks=sinks,
                seed=args.seed,
                arm_concurrency=args.arm_concurrency,
                progress_callback=report_turn,
            )
            async with status_lock:
                progress["completed_pairs_this_invocation"] += 1
                progress["completed_pairs_all_arms"] += 1
                _write_run_status(root, state="running", args=args, progress=progress)

    try:
        await asyncio.gather(*(run_pair(session, trial) for session, trial in pending))
    except BaseException as error:
        durable_progress = {**progress, **_bundle_progress(sinks, expected_turns)}
        _write_run_status(
            root,
            state="failed",
            args=args,
            progress=durable_progress,
            error=error,
            formatted_traceback=traceback.format_exc(),
        )
        raise
    durable_progress = {**progress, **_bundle_progress(sinks, expected_turns)}
    durable_progress["all_selected_pairs_complete"] = (
        durable_progress["completed_pairs_all_arms"] == durable_progress["total_pairs"]
    )
    _write_run_status(root, state="completed", args=args, progress=durable_progress)
    return root


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if not args.langsmith:
        os.environ["LANGSMITH_TRACING"] = "false"
    try:
        root = asyncio.run(run_all(args))
    except BaseException as error:
        # Validation can fail before run_all has enough state to publish its
        # richer status. Preserve that error for the same monitor when possible.
        root = Path(args.out)
        root.mkdir(parents=True, exist_ok=True)
        status_path = root / RUN_STATUS_FILENAME
        existing_error_matches = False
        if status_path.exists():
            try:
                existing = json.loads(status_path.read_text())
                existing_error_matches = existing.get(
                    "state"
                ) == "failed" and existing.get("fatal_error", {}).get("message") == str(
                    error
                )
            except (json.JSONDecodeError, OSError):
                pass
        if not existing_error_matches:
            _write_run_status(
                root,
                state="failed",
                args=args,
                error=error,
                formatted_traceback=traceback.format_exc(),
            )
        raise
    print(f"Paired arm bundles written below {root}")
    print(
        "Next: grade each arm, run evals.check_triggers, then compare with "
        "evals.report."
    )


if __name__ == "__main__":
    main()
