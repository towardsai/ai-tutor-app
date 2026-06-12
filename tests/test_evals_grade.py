from __future__ import annotations

import unittest

from evals.common import detect_battery_type, normalize_url, percentile
from evals.grade import (
    behavior_heuristic,
    grade_persona_question,
    retrieval_metrics,
)


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


if __name__ == "__main__":
    unittest.main()
