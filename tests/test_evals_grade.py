from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evals import check_triggers
from evals.common import detect_battery_type, normalize_url, percentile
from evals.grade import (
    FAITHFULNESS_EVIDENCE_MAX,
    behavior_heuristic,
    evidence_fits,
    evidence_is_complete,
    faithfulness_evidence,
    grade_persona_question,
    merge_handgrades,
    quality_sheet_rows,
    retrieval_metrics,
)
from evals.judge import RUBRICS, build_prompt, rubric_for


class CommonTests(unittest.TestCase):
    def test_detect_battery_type(self) -> None:
        self.assertEqual(detect_battery_type([{"case_id": "x"}]), "singleturn")
        self.assertEqual(detect_battery_type([{"session_id": "x"}]), "sessions")
        self.assertEqual(detect_battery_type([{"persona_id": "x"}]), "personas")
        self.assertEqual(detect_battery_type([{"replay_id": "x"}]), "replay")
        with self.assertRaises(ValueError):
            detect_battery_type([{"foo": 1}])

    def test_normalize_url(self) -> None:
        self.assertEqual(
            normalize_url("https://X.com/Lessons/abc/?q=1#frag"),
            "https://x.com/lessons/abc",
        )
        self.assertEqual(normalize_url(None), "")

    def test_normalize_url_strips_discussion_suffix(self) -> None:
        # Battery lesson_urls point at the discussion; matches carry the bare
        # lesson URL. Both must normalize to the same key.
        discussion = (
            "https://academy.towardsai.net/courses/take/agent-engineering/"
            "multimedia/67469688-lesson-1/discussions/12758677"
        )
        lesson = (
            "https://academy.towardsai.net/courses/take/agent-engineering/"
            "multimedia/67469688-lesson-1"
        )
        self.assertEqual(normalize_url(discussion), normalize_url(lesson))

    def test_percentile(self) -> None:
        # Nearest-rank: even-length p50 rounds to the upper of the two middles.
        self.assertEqual(percentile([1, 2, 3, 4], 50), 3)
        self.assertEqual(percentile([1, 2, 3], 50), 2)
        self.assertEqual(percentile([5], 95), 5)
        self.assertIsNone(percentile([], 50))


def bundle_with_matches(matches, tool_name="retrieve_tutor_context"):
    return {"tool_calls": [{"tool_name": tool_name, "matches": matches}]}


class RetrievalMetricsTests(unittest.TestCase):
    LESSON = "https://academy.towardsai.net/courses/take/x/lessons/123-foo"

    def test_hit_source_and_lesson_with_mrr(self) -> None:
        bundle = bundle_with_matches(
            [
                {"source_key": "other", "url": "https://elsewhere"},
                {"source_key": "full_stack_ai_engineering", "url": self.LESSON + "/"},
            ]
        )
        metrics = retrieval_metrics(bundle, "full_stack_ai_engineering", self.LESSON)
        self.assertTrue(metrics["called_retrieval"])
        self.assertTrue(metrics["recall_source"])
        self.assertTrue(metrics["recall_lesson"])
        self.assertEqual(metrics["mrr_lesson"], 0.5)

    def test_miss_lesson(self) -> None:
        bundle = bundle_with_matches(
            [{"source_key": "full_stack_ai_engineering", "url": "https://other"}]
        )
        metrics = retrieval_metrics(bundle, "full_stack_ai_engineering", self.LESSON)
        self.assertFalse(metrics["recall_lesson"])
        self.assertEqual(metrics["mrr_lesson"], 0.0)

    def test_kb_command_counts_as_retrieval_but_adds_no_matches(self) -> None:
        bundle = bundle_with_matches([], tool_name="run_kb_command")
        metrics = retrieval_metrics(bundle, "x", self.LESSON)
        self.assertTrue(metrics["called_retrieval"])
        self.assertEqual(metrics["retrieved_matches"], 0)

    def test_no_ground_truth_yields_none(self) -> None:
        bundle = bundle_with_matches([])
        metrics = retrieval_metrics(bundle, None, None)
        self.assertIsNone(metrics["recall_lesson"])
        self.assertIsNone(metrics["mrr_lesson"])


class BehaviorHeuristicTests(unittest.TestCase):
    def test_corpus_requires_tool_use(self) -> None:
        used = {"answer": "...", "tool_calls": [{"tool_name": "run_kb_command"}]}
        bare = {"answer": "...", "tool_calls": []}
        self.assertTrue(behavior_heuristic("answer_from_corpus", used))
        self.assertFalse(behavior_heuristic("answer_from_corpus", bare))

    def test_redirect_and_feedback_regexes(self) -> None:
        self.assertTrue(
            behavior_heuristic(
                "redirect_to_support",
                {"answer": "Please reach out to the academy team.", "tool_calls": []},
            )
        )
        self.assertTrue(
            behavior_heuristic(
                "acknowledge_feedback",
                {"answer": "Thank you for the suggestion!", "tool_calls": []},
            )
        )
        self.assertIsNone(
            behavior_heuristic("answer_general", {"answer": "x", "tool_calls": []})
        )


class PersonaGradingTests(unittest.TestCase):
    QUESTION = {
        "checks": [{"type": "regex_any", "patterns": ["conda"]}],
        "anti_patterns": ["uv sync", "python -m venv"],
    }

    def test_pass(self) -> None:
        result = grade_persona_question(self.QUESTION, "Use conda env create.")
        self.assertTrue(result["auto_pass"])

    def test_anti_pattern_fails_even_when_check_passes(self) -> None:
        result = grade_persona_question(
            self.QUESTION, "conda works, or run `uv sync` instead."
        )
        self.assertFalse(result["auto_pass"])
        self.assertEqual(result["anti_pattern_hits"], ["uv sync"])

    def test_case_insensitive(self) -> None:
        result = grade_persona_question(self.QUESTION, "CONDA is fine")
        self.assertTrue(result["auto_pass"])

    def test_llm_check_defers(self) -> None:
        question = {
            "checks": [
                {"type": "regex_any", "patterns": ["conda"]},
                {"type": "llm", "instruction": "is it beginner-level?"},
            ],
            "anti_patterns": [],
        }
        result = grade_persona_question(question, "conda activate course")
        self.assertIsNone(result["auto_pass"])
        self.assertTrue(result["needs_judgment"])


class FaithfulnessEvidenceTests(unittest.TestCase):
    def test_collects_retrieval_and_kb_evidence(self) -> None:
        bundle = {
            "tool_calls": [
                {
                    "tool_name": "retrieve_tutor_context",
                    "args_text": "context engineering",
                    "output_text": "Lesson 3 covers context engineering.",
                    "matches": [{"title": "Lesson 3", "url": "https://x/l3"}],
                },
                {
                    "tool_name": "run_kb_command",
                    "args_text": "rg foo",
                    "output_text": "raw/courses/agent/lesson.md: foo bar",
                },
            ]
        }
        evidence = faithfulness_evidence(bundle)
        self.assertIn("context engineering", evidence)
        self.assertIn("SOURCES: Lesson 3 <https://x/l3>", evidence)
        self.assertIn("raw/courses/agent/lesson.md", evidence)

    def test_empty_when_no_retrieval_tool(self) -> None:
        bundle = {"tool_calls": [{"tool_name": "some_other_tool", "output_text": "x"}]}
        self.assertEqual(faithfulness_evidence(bundle), "")
        self.assertEqual(faithfulness_evidence({}), "")

    def test_returns_full_evidence_untruncated(self) -> None:
        # One full-size KB capture (run_battery caps each output at 40k). The
        # old 12k slice here silently cut the judge's evidence to <1/3 of the
        # grounding; evidence must now come back whole -- fitting the judge
        # prompt is a separate gate (evidence_fits), not a slice.
        bundle = {
            "tool_calls": [{"tool_name": "run_kb_command", "output_text": "z" * 40_000}]
        }
        evidence = faithfulness_evidence(bundle)
        self.assertGreaterEqual(len(evidence), 40_000)


class EvidenceCompletenessTests(unittest.TestCase):
    def test_complete_when_no_truncation(self) -> None:
        bundle = {
            "tool_calls": [
                {"tool_name": "run_kb_command", "output_text": "abc", "output_chars": 3}
            ]
        }
        self.assertTrue(evidence_is_complete(bundle))

    def test_incomplete_when_kb_output_truncated(self) -> None:
        # output_chars (true length) exceeds captured output_text => truncated.
        bundle = {
            "tool_calls": [
                {
                    "tool_name": "run_kb_command",
                    "output_text": "x" * 6000,
                    "output_chars": 40000,
                }
            ]
        }
        self.assertFalse(evidence_is_complete(bundle))

    def test_non_retrieval_truncation_is_ignored(self) -> None:
        bundle = {
            "tool_calls": [
                {"tool_name": "some_tool", "output_text": "x", "output_chars": 99999}
            ]
        }
        self.assertTrue(evidence_is_complete(bundle))

    def test_faithfulness_row_gated_on_complete_evidence(self) -> None:
        truncated = {
            "run_id": "r",
            "battery_type": "singleturn",
            "preset": "prod",
            "query": "q",
            "answer": "a",
            "tool_calls": [
                {
                    "tool_name": "run_kb_command",
                    "args_text": "rg foo",
                    "output_text": "y" * 6000,
                    "output_chars": 40000,
                }
            ],
        }
        types = {r["item_type"] for r in quality_sheet_rows(truncated)}
        self.assertIn("holistic", types)
        self.assertNotIn("faithfulness", types)  # truncated -> no faithfulness

        full = dict(truncated)
        full["tool_calls"] = [
            {
                "tool_name": "run_kb_command",
                "args_text": "rg foo",
                "output_text": "y" * 100,
                "output_chars": 100,
            }
        ]
        types_full = {r["item_type"] for r in quality_sheet_rows(full)}
        self.assertIn("faithfulness", types_full)  # full evidence -> emitted


class EvidenceFitsTests(unittest.TestCase):
    """The judge-prompt side of the completeness guarantee: a faithfulness row
    carries the FULL evidence or is not emitted at all -- never a silent slice
    (the F23/F24 blind-judge class the capture gate exists to prevent)."""

    @staticmethod
    def _bundle(output_text: str) -> dict:
        return {
            "run_id": "r",
            "battery_type": "singleturn",
            "preset": "prod",
            "query": "q",
            "answer": "a",
            "tool_calls": [
                {
                    "tool_name": "run_kb_command",
                    "args_text": "cat raw/docs/x/big.md",
                    # Fully captured: output_chars == len(output_text), so the
                    # capture-completeness gate passes.
                    "output_text": output_text,
                    "output_chars": len(output_text),
                }
            ],
        }

    def test_evidence_fits_predicate(self) -> None:
        self.assertTrue(evidence_fits("x" * FAITHFULNESS_EVIDENCE_MAX))
        self.assertFalse(evidence_fits("x" * (FAITHFULNESS_EVIDENCE_MAX + 1)))
        self.assertTrue(evidence_fits("abc", max_chars=3))
        self.assertFalse(evidence_fits("abcd", max_chars=3))

    def test_cap_covers_a_realistic_multi_call_kb_turn(self) -> None:
        # The gate must not exclude what capture guarantees: a few full 40k
        # KB outputs (run_battery.TOOL_OUTPUT_MAX_CHARS) per turn must fit.
        self.assertGreaterEqual(FAITHFULNESS_EVIDENCE_MAX, 4 * 40_000)

    def test_faithfulness_row_carries_full_evidence(self) -> None:
        # One complete 40k KB output: under the old 12k slice the judge saw
        # ~30% of the grounding. The row must now carry all of it.
        text = ("y" * 39_990) + "END-MARKER"
        rows = [
            r
            for r in quality_sheet_rows(self._bundle(text))
            if r["item_type"] == "faithfulness"
        ]
        self.assertEqual(len(rows), 1)
        self.assertGreater(len(rows[0]["reference"]), 12_000)  # old cap is gone
        self.assertTrue(rows[0]["reference"].endswith("END-MARKER"))

    def test_faithfulness_row_excluded_when_evidence_exceeds_cap(self) -> None:
        # Fully captured but too big for the judge prompt: excluded, not
        # silently truncated. Holistic (which never uses evidence) still emits.
        bundle = self._bundle("z" * (FAITHFULNESS_EVIDENCE_MAX + 1))
        types = {r["item_type"] for r in quality_sheet_rows(bundle)}
        self.assertIn("holistic", types)
        self.assertNotIn("faithfulness", types)


class CheckTriggersMainTests(unittest.TestCase):
    """check_triggers must evaluate EVERY run: all() over a bare generator
    short-circuited on the first failing run, hiding later runs' diagnostics."""

    def _run_main(self, runs: list[str], results: list[bool]) -> tuple[list[Path], int]:
        calls: list[Path] = []

        def fake_check_run(run_dir: Path, expect_none: bool) -> bool:
            calls.append(run_dir)
            return results[len(calls) - 1]

        argv = ["check_triggers", "--runs", *runs]
        with (
            mock.patch.object(check_triggers, "check_run", fake_check_run),
            mock.patch.object(sys, "argv", argv),
            self.assertRaises(SystemExit) as ctx,
        ):
            check_triggers.main()
        return calls, ctx.exception.code

    def test_later_runs_still_checked_after_a_failure(self) -> None:
        calls, code = self._run_main(
            ["runs/a", "runs/b", "runs/c"], [False, True, True]
        )
        self.assertEqual(calls, [Path("runs/a"), Path("runs/b"), Path("runs/c")])
        self.assertEqual(code, 1)  # exit semantics preserved

    def test_exit_zero_when_all_runs_pass(self) -> None:
        calls, code = self._run_main(["runs/a", "runs/b"], [True, True])
        self.assertEqual(len(calls), 2)
        self.assertEqual(code, 0)


class JudgeRubricTests(unittest.TestCase):
    def test_holistic_and_faithfulness_rubrics_exist(self) -> None:
        self.assertIs(rubric_for("holistic"), RUBRICS["holistic"])
        self.assertIs(rubric_for("faithfulness"), RUBRICS["faithfulness"])

    def test_unknown_item_type_falls_back_to_key_point(self) -> None:
        self.assertIs(rubric_for("mystery"), RUBRICS["key_point"])

    def test_faithfulness_reference_labeled_as_evidence(self) -> None:
        row = {
            "item_type": "faithfulness",
            "question": "q",
            "criterion": "grounded?",
            "reference": "RETRIEVED_TEXT_X",
            "answer": "a",
        }
        prompt = build_prompt(row)
        self.assertIn("RETRIEVED EVIDENCE", prompt)
        self.assertNotIn("STAFF REFERENCE", prompt)

    def test_other_item_types_keep_staff_reference_label(self) -> None:
        row = {
            "item_type": "key_point",
            "question": "q",
            "criterion": "claim",
            "reference": "STAFF_TEXT",
            "answer": "a",
        }
        self.assertIn("STAFF REFERENCE", build_prompt(row))


class MergeHandgradesTests(unittest.TestCase):
    FIELDS = [
        "sheet_row_id",
        "run_id",
        "item_type",
        "criterion",
        "grade",
        "note",
    ]

    def _merge(self, grades, filled_rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "filled.csv"
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()
                writer.writerows(filled_rows)
            return merge_handgrades(grades, path)

    def test_new_item_types_and_per_metric_source(self) -> None:
        grades = [{"run_id": "r1"}]
        filled = [
            # human-graded key points
            {
                "sheet_row_id": "r1|key_point|a",
                "run_id": "r1",
                "item_type": "key_point",
                "criterion": "k1",
                "grade": "pass",
                "note": "",
            },
            {
                "sheet_row_id": "r1|key_point|b",
                "run_id": "r1",
                "item_type": "key_point",
                "criterion": "k2",
                "grade": "fail",
                "note": "",
            },
            # judge-graded new dimensions
            {
                "sheet_row_id": "r1|holistic|c",
                "run_id": "r1",
                "item_type": "holistic",
                "criterion": "h",
                "grade": "pass",
                "note": "[judge:high] good answer",
            },
            {
                "sheet_row_id": "r1|faithfulness|d",
                "run_id": "r1",
                "item_type": "faithfulness",
                "criterion": "f",
                "grade": "fail",
                "note": "[judge:low] fabricated param",
            },
        ]
        merged = self._merge(grades, filled)[0]
        self.assertEqual(merged["key_points_passed"], 1)
        self.assertEqual(merged["key_points_total"], 2)
        self.assertEqual(merged["key_points_source"], "human")
        self.assertTrue(merged["holistic_pass"])
        self.assertEqual(merged["holistic_source"], "judge")
        self.assertFalse(merged["faithfulness_pass"])
        self.assertEqual(merged["faithfulness_source"], "judge")
        # human key-points + judge new dims => overall mixed
        self.assertEqual(merged["grade_source"], "mixed")


if __name__ == "__main__":
    unittest.main()
