from __future__ import annotations

import argparse
import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from evals.check_triggers import check_run
from evals.report import write_cost_curves
from evals.run_compaction_experiment import (
    MIGRATION_FILENAME,
    TRIGGER_VALIDATION_BASELINE_SOURCE_SHA256,
    _bundle_progress,
    _import_compatible_completed_pairs,
    _is_retryable_turn_error,
    _prune_to_common_completed,
    _run_turn_with_retries,
    _run_session_trial,
    _schedule_pending,
    _write_run_status,
)
from evals.run_battery import (
    experiment_cache_user_id,
    validate_experiment_result,
    write_or_validate_run_config,
)


def experiment_args(battery: Path, out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        battery=str(battery),
        preset="exp_fh_raw",
        model="deepseek:deepseek-v4-flash",
        trials=1,
        arm_concurrency=1,
        pair_concurrency=1,
        max_pairs_this_invocation=0,
        first_pair_id="",
        import_completed_from="",
        out=str(out),
        limit=0,
        ids=[],
        tags=[],
        concurrency=1,
        scope_sources=False,
        enable_tools=[],
        disable_kb=False,
        no_tools=False,
        retrieval_budget=0,
        retriever="",
        langsmith=False,
    )


class ImmutableManifestTests(unittest.TestCase):
    def test_status_file_is_atomic_machine_readable_and_includes_fatal_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = experiment_args(root / "battery.jsonl", root)
            args.presets = ["exp_fh_raw"]
            error = RuntimeError("guard exceeded")
            path = _write_run_status(
                root,
                state="failed",
                args=args,
                progress={"completed_pairs_all_arms": 4},
                error=error,
                formatted_traceback="traceback text",
            )
            status = json.loads(path.read_text())
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["fatal_error"]["type"], "RuntimeError")
            self.assertEqual(status["fatal_error"]["message"], "guard exceeded")
            self.assertEqual(status["progress"]["completed_pairs_all_arms"], 4)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_bundle_progress_reports_only_pairs_complete_in_every_arm(self) -> None:
        class Sink:
            def __init__(self, path: Path) -> None:
                self.path = path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / f"{name}.jsonl" for name in ("a", "b")}
            complete = [
                {"unit_id": "s1", "trial": 1, "error": None},
                {"unit_id": "s1", "trial": 1, "error": None},
            ]
            paths["a"].write_text("".join(json.dumps(row) + "\n" for row in complete))
            paths["b"].write_text(
                "".join(json.dumps(row) + "\n" for row in complete[:1])
            )
            progress = _bundle_progress(
                {name: Sink(path) for name, path in paths.items()}, {"s1": 2}
            )
            self.assertEqual(progress["arms"]["a"]["completed_pairs"], 1)
            self.assertEqual(progress["arms"]["b"]["completed_pairs"], 0)
            self.assertEqual(progress["completed_pairs_all_arms"], 0)

    def test_resume_accepts_exact_match_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            battery = root / "battery.jsonl"
            battery.write_text('{"session_id":"s","turns":["q"]}\n')
            out = root / "run"
            out.mkdir()
            args = experiment_args(battery, out)
            first = write_or_validate_run_config(out, args)
            (out / "bundles.jsonl").write_text("{}\n")
            self.assertEqual(write_or_validate_run_config(out, args), first)

            args.retrieval_budget = 30_000
            with self.assertRaises(SystemExit):
                write_or_validate_run_config(out, args)

            config = json.loads((out / "run_config.json").read_text())
            self.assertEqual(config["_fingerprint"], first)
            self.assertIn("source_tree_sha256", config["_manifest"])
            self.assertIn("pricing_snapshot_usd_per_million", config["_manifest"])

    def test_operational_staging_controls_do_not_change_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            battery = root / "battery.jsonl"
            battery.write_text('{"session_id":"s","turns":["q"]}\n')
            out = root / "run"
            out.mkdir()
            args = experiment_args(battery, out)
            first = write_or_validate_run_config(out, args)
            (out / "bundles.jsonl").write_text("{}\n")
            args.max_pairs_this_invocation = 1
            args.first_pair_id = "s"
            args.import_completed_from = str(root / "old-run")
            self.assertEqual(write_or_validate_run_config(out, args), first)

    def test_cache_ids_are_stable_and_arm_isolated(self) -> None:
        first = experiment_cache_user_id("exp_fh_raw", "session", 1)
        self.assertEqual(first, experiment_cache_user_id("exp_fh_raw", "session", 1))
        self.assertNotEqual(
            first, experiment_cache_user_id("exp_c200_raw", "session", 1)
        )
        self.assertNotEqual(
            first,
            experiment_cache_user_id("exp_fh_raw", "session", 1, namespace="new-run"),
        )
        self.assertRegex(first, r"^[a-zA-Z0-9_-]+$")

    def test_lockstep_resume_keeps_only_units_complete_in_every_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm_a = root / "a.jsonl"
            arm_b = root / "b.jsonl"
            complete = [
                {"unit_id": "s1", "trial": 1, "turn_index": 0, "error": None},
                {"unit_id": "s1", "trial": 1, "turn_index": 1, "error": None},
            ]
            partial = [{"unit_id": "s2", "trial": 1, "turn_index": 0, "error": None}]
            arm_a.write_text(
                "".join(json.dumps(row) + "\n" for row in [*complete, *partial])
            )
            arm_b.write_text("".join(json.dumps(row) + "\n" for row in complete))
            common = _prune_to_common_completed([arm_a, arm_b], {"s1": 2, "s2": 2})
            self.assertEqual(common, {("s1", 1)})
            self.assertEqual(len(arm_a.read_text().splitlines()), 2)

    def _write_migration_fixture(
        self, root: Path, *, drift: bool = False
    ) -> tuple[Path, Path, list[str], dict[str, str]]:
        source = root / "source"
        target = root / "target"
        presets = ["exp_fh_raw", "exp_c200_raw"]
        target_fingerprints: dict[str, str] = {}
        for preset in presets:
            source_arm = source / preset
            target_arm = target / preset
            source_arm.mkdir(parents=True)
            target_arm.mkdir(parents=True)
            source_fingerprint = f"source-{preset}"
            target_fingerprint = f"target-{preset}"
            target_fingerprints[preset] = target_fingerprint
            scientific_value = "changed" if drift and preset == presets[-1] else "same"
            source_manifest = {
                "source_tree_sha256": TRIGGER_VALIDATION_BASELINE_SOURCE_SHA256,
                "git_status": "old dirty state",
                "scientific_configuration": "same",
            }
            target_manifest = {
                "source_tree_sha256": "new-source-tree",
                "git_status": "new dirty state",
                "scientific_configuration": scientific_value,
            }
            (source_arm / "run_config.json").write_text(
                json.dumps(
                    {
                        "_fingerprint": source_fingerprint,
                        "_manifest": source_manifest,
                    }
                )
            )
            (target_arm / "run_config.json").write_text(
                json.dumps(
                    {
                        "_fingerprint": target_fingerprint,
                        "_manifest": target_manifest,
                    }
                )
            )
            complete = [
                {
                    "unit_id": "s1",
                    "trial": 1,
                    "turn_index": turn,
                    "error": None,
                    "run_fingerprint": source_fingerprint,
                }
                for turn in range(2)
            ]
            partial = [
                {
                    "unit_id": "s2",
                    "trial": 1,
                    "turn_index": 0,
                    "error": None,
                    "run_fingerprint": source_fingerprint,
                }
            ]
            rows = [*complete, *partial] if preset == presets[0] else complete
            (source_arm / "bundles.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
        return source, target, presets, target_fingerprints

    def test_migration_imports_only_common_complete_pairs_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target, presets, target_fingerprints = (
                self._write_migration_fixture(root)
            )
            imported = _import_compatible_completed_pairs(
                source_root=source,
                target_root=target,
                presets=presets,
                target_fingerprints=target_fingerprints,
                expected_turns={"s1": 2, "s2": 2},
                eligible_pairs={("s1", 1), ("s2", 1)},
            )
            self.assertEqual(imported, {("s1", 1)})
            for preset in presets:
                rows = [
                    json.loads(line)
                    for line in (target / preset / "bundles.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(rows), 2)
                self.assertTrue(
                    all(
                        row["run_fingerprint"] == target_fingerprints[preset]
                        for row in rows
                    )
                )
                self.assertTrue(
                    all(
                        row["migration"]["source_run_fingerprint"] == f"source-{preset}"
                        for row in rows
                    )
                )
            record = json.loads((target / MIGRATION_FILENAME).read_text())
            self.assertEqual(record["imported_pairs"], [{"trial": 1, "unit_id": "s1"}])
            self.assertEqual(
                _import_compatible_completed_pairs(
                    source_root=source,
                    target_root=target,
                    presets=presets,
                    target_fingerprints=target_fingerprints,
                    expected_turns={"s1": 2, "s2": 2},
                    eligible_pairs={("s1", 1), ("s2", 1)},
                ),
                {("s1", 1)},
            )

    def test_migration_rejects_scientific_manifest_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, target, presets, fingerprints = self._write_migration_fixture(
                Path(directory), drift=True
            )
            with self.assertRaisesRegex(SystemExit, "scientific manifest drift"):
                _import_compatible_completed_pairs(
                    source_root=source,
                    target_root=target,
                    presets=presets,
                    target_fingerprints=fingerprints,
                    expected_turns={"s1": 2, "s2": 2},
                    eligible_pairs={("s1", 1), ("s2", 1)},
                )


class PairedConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_turn_failure_retries_and_records_missing_usage(
        self,
    ) -> None:
        failed = {
            "started_at": "first",
            "duration_ms": 10,
            "error": "ReadError: stream reset",
            "context_stats": None,
        }
        succeeded = {
            "started_at": "second",
            "duration_ms": 20,
            "error": None,
            "context_stats": {},
        }
        with (
            mock.patch(
                "evals.run_compaction_experiment.run_turn",
                side_effect=[failed, succeeded],
            ) as run,
            mock.patch("evals.run_compaction_experiment.asyncio.sleep") as sleep,
        ):
            result = await _run_turn_with_retries(SimpleNamespace())
        self.assertEqual(run.await_count, 2)
        sleep.assert_awaited_once_with(1.0)
        self.assertEqual(result["started_at"], "first")
        self.assertEqual(result["turn_retry_attempts"], 1)
        self.assertTrue(result["failed_attempt_usage_unavailable"])
        self.assertEqual(result["context_stats"]["turn_retry_attempts"], 1)

    async def test_permanent_turn_failure_is_not_retried(self) -> None:
        failed = {
            "started_at": "first",
            "duration_ms": 10,
            "error": "RuntimeError: invalid state",
            "context_stats": None,
        }
        with mock.patch(
            "evals.run_compaction_experiment.run_turn", return_value=failed
        ) as run:
            result = await _run_turn_with_retries(SimpleNamespace())
        self.assertEqual(run.await_count, 1)
        self.assertEqual(result["turn_retry_attempts"], 0)
        self.assertFalse(_is_retryable_turn_error(result["error"]))

    def test_staging_prioritizes_trial_one_and_limits_invocation(self) -> None:
        sessions = [
            {"session_id": "short"},
            {"session_id": "representative"},
        ]
        pending = [(session, trial) for session in sessions for trial in (1, 2, 3)]
        scheduled = _schedule_pending(
            pending,
            seed=7,
            first_pair_id="representative",
            max_pairs=1,
        )
        self.assertEqual(scheduled[0][0]["session_id"], "representative")
        self.assertEqual(scheduled[0][1], 1)

    async def test_arms_overlap_but_turn_barrier_is_preserved(self) -> None:
        class Sink:
            def __init__(self) -> None:
                self.rows: list[dict] = []

            async def write(self, rows: list[dict]) -> None:
                self.rows.extend(rows)

        active = 0
        max_active = 0
        completed_turn_zero = 0

        def build_request(args, *, query, **_kwargs):
            return SimpleNamespace(preset=args.preset, query=query)

        async def run_turn(request):
            nonlocal active, max_active, completed_turn_zero
            if request.query == "q1":
                self.assertEqual(completed_turn_zero, 2)
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            if request.query == "q0":
                completed_turn_zero += 1
            return {
                "thread_id": f"thread-{request.preset}",
                "answer": f"answer-{request.preset}",
                "error": None,
            }

        arm_args = {
            preset: SimpleNamespace(preset=preset)
            for preset in ("exp_fh_raw", "exp_c200_raw")
        }
        sinks = {preset: Sink() for preset in arm_args}
        with (
            mock.patch(
                "evals.run_compaction_experiment.build_request",
                side_effect=build_request,
            ),
            mock.patch(
                "evals.run_compaction_experiment.run_turn", side_effect=run_turn
            ),
            mock.patch("evals.run_compaction_experiment.validate_experiment_result"),
            mock.patch(
                "evals.run_compaction_experiment.make_bundle",
                side_effect=lambda **kwargs: {"turn_index": kwargs["turn_index"]},
            ),
        ):
            await _run_session_trial(
                session={"session_id": "s", "turns": ["q0", "q1"]},
                trial=1,
                arm_args=arm_args,
                sinks=sinks,
                seed=7,
                arm_concurrency=2,
            )

        self.assertEqual(max_active, 2)
        self.assertEqual(completed_turn_zero, 2)
        self.assertTrue(all(len(sink.rows) == 2 for sink in sinks.values()))


class ExperimentValidationTests(unittest.TestCase):
    def _result(self) -> dict:
        return {
            "error": None,
            "context_stats": {
                "llm_calls": 1,
                "cost_breakdown": {"total_usd": 0.1},
                "model_calls": [
                    {
                        "sequence": 1,
                        "model": "deepseek-v4-flash",
                        "usage_reported": True,
                        "cache_details_reported": True,
                    }
                ],
                "compaction_events": [],
            },
        }

    def test_valid_deepseek_telemetry_passes(self) -> None:
        args = argparse.Namespace(
            preset="exp_fh_raw", model="deepseek:deepseek-v4-flash"
        )
        result = self._result()
        validate_experiment_result(args, result)
        self.assertIsNone(result["error"])

    def test_provider_drift_and_old_4k_summary_fail_loudly(self) -> None:
        args = argparse.Namespace(
            preset="exp_c200_raw", model="deepseek:deepseek-v4-flash"
        )
        result = self._result()
        result["context_stats"]["model_calls"][0]["model"] = "gemini-3.5-flash"
        result["context_stats"]["compaction_events"] = [
            {
                "summary_input_untrimmed": True,
                "summary_input_tokens_approx": 4_000,
            }
        ]
        validate_experiment_result(args, result)
        self.assertIn("expected only deepseek-v4-flash", result["error"])
        self.assertIn("historical 4k cap", result["error"])

    def test_provider_reported_trigger_evidence_passes_validation(self) -> None:
        args = argparse.Namespace(
            preset="exp_c200_raw", model="deepseek:deepseek-v4-flash"
        )
        result = self._result()
        result["context_stats"]["compaction_events"] = [
            {
                "configured_trigger_tokens": 200_000,
                "pre_compaction_tokens_approx": 199_567,
                "trigger_reported_tokens": 207_336,
                "trigger_source": "provider_reported",
                "summary_input_untrimmed": True,
                "summary_input_tokens_approx": 158_014,
            }
        ]
        validate_experiment_result(args, result)
        self.assertIsNone(result["error"])


class TriggerGateTests(unittest.TestCase):
    def test_expected_trigger_checks_configuration_without_rejecting_overshoot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            battery = root / "battery.jsonl"
            battery.write_text('{"session_id":"s","turns":["q"]}\n')
            run = root / "run"
            run.mkdir()
            bundle = {
                "unit_id": "s",
                "trial": 1,
                "turn_index": 0,
                "preset": "exp_c200_raw",
                "battery_path": str(battery),
                "error": None,
                "context_stats": {
                    "input_tokens": 1,
                    "est_cost_usd": 0,
                    "summary_messages": 1,
                    "compactions_this_turn": 1,
                    "compaction_events": [
                        {
                            "configured_trigger_tokens": 200_000,
                            "pre_compaction_tokens_approx": 199_567,
                            "trigger_reported_tokens": 207_336,
                            "summary_input_tokens_approx": 205_000,
                        }
                    ],
                },
            }
            (run / "bundles.jsonl").write_text(json.dumps(bundle) + "\n")
            self.assertTrue(
                check_run(
                    run,
                    False,
                    min_compactions=1,
                    min_summary_input=4_001,
                    expected_trigger_tokens=200_000,
                    first_pre_tokens_min=200_000,
                )
            )
            self.assertFalse(check_run(run, False, expected_trigger_tokens=300_000))


class CostCurveTests(unittest.TestCase):
    def test_csv_contains_cached_uncached_output_and_total_cost(self) -> None:
        bundles = []
        for turn, total in enumerate((0.1, 0.2)):
            bundles.append(
                {
                    "unit_id": "session",
                    "trial": 1,
                    "turn_index": turn,
                    "context_stats": {
                        "cost_breakdown": {
                            "cache_read_input_usd": total / 10,
                            "cache_miss_input_usd": total / 2,
                            "cache_creation_input_usd": 0,
                            "output_usd": total * 0.4,
                            "total_usd": total,
                        },
                        "summarization_cost_usd": 0,
                    },
                }
            )
        run = {
            "battery_type": "sessions",
            "label": "exp_fh_raw",
            "bundles": bundles,
        }
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with mock.patch("evals.report._plot_cost_curves", side_effect=ImportError):
                write_cost_curves([run], out)
            with (out / "trajectory_cost_by_turn.csv").open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[-1]["cumulative_total_usd"]), 0.3)
        self.assertIn("turn_cache_read_input_usd", rows[-1])
        self.assertIn("turn_cache_miss_input_usd", rows[-1])
        self.assertIn("turn_output_usd", rows[-1])


if __name__ == "__main__":
    unittest.main()


class DeprecatedBatteryGuardTest(unittest.TestCase):
    def test_deprecated_battery_refuses_new_runs(self) -> None:
        from evals.common import ensure_battery_not_deprecated

        with self.assertRaises(SystemExit) as ctx:
            ensure_battery_not_deprecated(
                "data/eval/battery_sessions_v2.jsonl", override=False
            )
        self.assertIn("battery_sessions_v2_1.jsonl", str(ctx.exception))

    def test_override_and_healthy_batteries_pass(self) -> None:
        from evals.common import ensure_battery_not_deprecated

        ensure_battery_not_deprecated(
            "data/eval/battery_sessions_v2.jsonl", override=True
        )
        ensure_battery_not_deprecated(
            "data/eval/battery_sessions_v1.jsonl", override=False
        )
