from __future__ import annotations

import unittest
from unittest.mock import patch

from app.api import SlidingWindowRateLimiter


class SlidingWindowRateLimiterTestCase(unittest.TestCase):
    def test_allows_up_to_budget_then_reports_retry_after(self) -> None:
        limiter = SlidingWindowRateLimiter()

        with patch("app.api.time.monotonic", return_value=1_000.0):
            self.assertIsNone(limiter.retry_after("client-a", 2, 60))
            self.assertIsNone(limiter.retry_after("client-a", 2, 60))
            retry_after = limiter.retry_after("client-a", 2, 60)

        self.assertIsNotNone(retry_after)
        self.assertAlmostEqual(retry_after, 60.0)
        # A different address has its own budget.
        with patch("app.api.time.monotonic", return_value=1_000.0):
            self.assertIsNone(limiter.retry_after("client-b", 2, 60))

    def test_expired_hits_no_longer_count_against_the_budget(self) -> None:
        limiter = SlidingWindowRateLimiter()

        with patch("app.api.time.monotonic", return_value=1_000.0):
            self.assertIsNone(limiter.retry_after("client-a", 1, 60))

        # One second past the window, the old hit has aged out.
        with patch("app.api.time.monotonic", return_value=1_061.0):
            self.assertIsNone(limiter.retry_after("client-a", 1, 60))

    def test_tracked_client_table_is_hard_bounded_under_address_churn(self) -> None:
        limiter = SlidingWindowRateLimiter(max_tracked_clients=8)

        with patch("app.api.time.monotonic", return_value=1_000.0):
            for index in range(100):
                self.assertIsNone(limiter.retry_after(f"198.51.100.{index}", 5, 300))

        self.assertLessEqual(len(limiter._hits), 8)

    def test_capacity_eviction_removes_the_least_recently_seen_client(self) -> None:
        limiter = SlidingWindowRateLimiter(max_tracked_clients=2)

        with patch("app.api.time.monotonic", return_value=1_000.0):
            self.assertIsNone(limiter.retry_after("client-a", 2, 60))
            self.assertIsNone(limiter.retry_after("client-b", 2, 60))
            # Refresh client-a so client-b becomes the least recently seen.
            self.assertIsNone(limiter.retry_after("client-a", 2, 60))
            self.assertIsNone(limiter.retry_after("client-c", 2, 60))

            self.assertIn("client-a", limiter._hits)
            self.assertNotIn("client-b", limiter._hits)
            # The surviving client keeps its budget: recency, not luck.
            self.assertIsNotNone(limiter.retry_after("client-a", 2, 60))


if __name__ == "__main__":
    unittest.main()
