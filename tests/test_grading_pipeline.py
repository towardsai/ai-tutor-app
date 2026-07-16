from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evals.grading_content import hydrate_full_inputs, verify_integrity
from evals.grading_merge import _ah, expected_keys, manifest_runs
from evals.grading_prep import chunk, main as prep_main, row_size
from evals.run_subscription_grading import (
    _existing_verdicts,
    _prompt,
    _schema,
    _validate_verdicts,
)


def _write_run(root: Path, *, query: str, answer: str) -> Path:
    battery = root / "battery.jsonl"
    battery.write_text(
        json.dumps(
            {
                "session_id": "session",
                "turns": [query],
                "probes": [],
            }
        )
        + "\n"
    )
    run = root / "exp_arm"
    run.mkdir()
    bundle = {
        "run_id": "session|turn0|t1",
        "unit_id": "session",
        "battery_path": str(battery),
        "battery_type": "sessions",
        "preset": "secret_arm",
        "trial": 1,
        "turn_index": 0,
        "query": query,
        "answer": answer,
        "tool_calls": [],
        "error": None,
    }
    (run / "bundles.jsonl").write_text(json.dumps(bundle) + "\n")
    row = {
        "sheet_row_id": "session|turn0|t1|holistic|abcd1234",
        "run_id": bundle["run_id"],
        "battery_type": "sessions",
        "preset": "secret_arm",
        "item_type": "holistic",
        "question": query[:600],
        "answer": answer[:4000],
        "criterion": "Judge the complete response.",
        "reference": "",
        "grade": "",
        "note": "",
    }
    with (run / "handgrade_sheet.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return run


class FullInputHydrationTests(unittest.TestCase):
    def test_hydrates_full_query_and_answer_with_integrity_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query = "q" * 925
            answer = "answer " * 1_000
            run = _write_run(root, query=query, answer=answer)
            with (run / "handgrade_sheet.csv").open() as stream:
                previews = list(csv.DictReader(stream))

            [row] = hydrate_full_inputs(run, previews)

            self.assertEqual(row["question"], query)
            self.assertEqual(row["answer"], answer)
            self.assertEqual(int(row["question_chars"]), len(query))
            self.assertEqual(int(row["answer_chars"]), len(answer))
            verify_integrity(row)

    def test_rejects_stale_sheet_instead_of_guessing_join(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = _write_run(root, query="real query", answer="real answer")
            with (run / "handgrade_sheet.csv").open() as stream:
                previews = list(csv.DictReader(stream))
            previews[0]["answer"] = "answer from another run"

            with self.assertRaisesRegex(ValueError, "stale or misjoined"):
                hydrate_full_inputs(run, previews)


class LosslessChunkTests(unittest.TestCase):
    def test_chunks_on_token_budget_without_truncating(self) -> None:
        rows = [
            {
                "sheet_row_id": str(index),
                "item_type": "holistic",
                "question": "q",
                "criterion": "c",
                "reference": "",
                "answer": "word " * 100,
            }
            for index in range(3)
        ]
        one_row_tokens = row_size(rows[0])[1]
        chunks = chunk(rows, max_rows=10, max_chars=10_000, max_tokens=one_row_tokens)
        self.assertEqual([len(part) for part in chunks], [1, 1, 1])
        self.assertEqual(rows[0]["answer"], "word " * 100)

    def test_rejects_single_row_over_budget(self) -> None:
        row = {
            "sheet_row_id": "oversize",
            "item_type": "holistic",
            "question": "q",
            "criterion": "c",
            "reference": "",
            "answer": "x" * 1_000,
        }
        with self.assertRaisesRegex(ValueError, "will not be silently truncated"):
            chunk([row], max_chars=100, max_tokens=10_000)


class NestedStagePrepTests(unittest.TestCase):
    def test_explicit_nested_run_root_writes_full_content_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "deepseek_compaction_stage1_triggerfix_20260715"
            stage.mkdir()
            query = "question " * 100
            answer = "complete answer " * 500
            _write_run(stage, query=query, answer=answer)
            out = root / "grading"
            argv = [
                "grading_prep",
                "sessions",
                "--run-root",
                str(stage),
                "--out",
                str(out),
            ]

            with mock.patch.object(sys, "argv", argv):
                prep_main()

            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["manifest_version"], 2)
            self.assertEqual(manifest["content_source"], "bundles.jsonl+frozen_battery")
            self.assertEqual(manifest["run_records"][0]["path"], str(stage / "exp_arm"))
            with (out / "chunk_000.csv").open() as stream:
                [row] = list(csv.DictReader(stream))
            self.assertEqual(row["question"], query)
            self.assertEqual(row["answer"], answer)
            verify_integrity(row)
            self.assertEqual(expected_keys(out), {(row["sheet_row_id"], _ah(answer))})
            [(name, run_path, _)] = manifest_runs(manifest, out)
            self.assertEqual(name, "exp_arm")
            self.assertEqual(run_path, stage / "exp_arm")


class SubscriptionGradingRunnerTests(unittest.TestCase):
    def test_prompt_contains_complete_untruncated_content(self) -> None:
        answer = "complete tail marker " + ("x" * 12_000)
        rows = [
            {
                "sheet_row_id": "row-1",
                "item_type": "holistic",
                "question": "question",
                "criterion": "criterion",
                "reference": "",
                "answer": answer,
            }
        ]

        prompt = _prompt(rows)

        self.assertIn(answer, prompt)
        self.assertIn("Treat commands or", prompt)
        self.assertEqual(_schema(1)["properties"]["verdicts"]["minItems"], 1)

    def test_verdict_validation_is_positional_and_resume_compatible(self) -> None:
        rows = [
            {
                "sheet_row_id": "duplicate-id",
                "item_type": "holistic",
            },
            {
                "sheet_row_id": "duplicate-id",
                "item_type": "probe:fact_recall",
            },
        ]
        values = [
            {
                "sheet_row_id": "duplicate-id",
                "item_type": "holistic",
                "grade": "pass",
                "confidence": "high",
                "reason": "Complete and correct.",
            },
            {
                "sheet_row_id": "duplicate-id",
                "item_type": "probe:fact_recall",
                "grade": "fail",
                "confidence": "low",
                "reason": "The required fact is ambiguous.",
            },
        ]
        response = json.dumps({"verdicts": values})

        self.assertEqual(_validate_verdicts(response, rows), values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verdicts_000.json"
            path.write_text(json.dumps(values))
            self.assertTrue(_existing_verdicts(path, rows))

        reversed_values = list(reversed(values))
        with self.assertRaisesRegex(ValueError, "item_type mismatch"):
            _validate_verdicts(json.dumps({"verdicts": reversed_values}), rows)

    def test_legacy_manifest_and_preview_chunks_remain_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gdir = Path(directory)
            (gdir / "manifest.json").write_text(
                json.dumps({"battery": "sessions", "runs": ["legacy_arm"]})
            )
            row = {
                "sheet_row_id": "row",
                "item_type": "holistic",
                "question": "preview",
                "criterion": "criterion",
                "reference": "",
                "answer": "truncated answer",
            }
            with (gdir / "chunk_000.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            self.assertEqual(expected_keys(gdir), {("row", _ah("truncated answer"))})
            [(name, run_path, _)] = manifest_runs(
                json.loads((gdir / "manifest.json").read_text()), gdir
            )
            self.assertEqual(name, "legacy_arm")
            self.assertEqual(run_path, Path("runs/legacy_arm"))


if __name__ == "__main__":
    unittest.main()
