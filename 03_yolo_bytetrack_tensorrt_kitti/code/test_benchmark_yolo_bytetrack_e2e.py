"""Tests for the deterministic statistics used by the E2E benchmark."""

import unittest

from tools.benchmark_yolo_bytetrack_e2e import reset_trackers, summarize_latency_ms


class SummarizeLatencyMsTests(unittest.TestCase):
    def test_returns_hand_checked_percentiles_and_fps(self) -> None:
        """Catches a percentile method or milliseconds-to-FPS conversion regression."""
        summary = summarize_latency_ms([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean_ms"], 2.5)
        self.assertAlmostEqual(summary["p50_ms"], 2.5)
        self.assertAlmostEqual(summary["p95_ms"], 3.85)
        self.assertAlmostEqual(summary["fps_from_mean"], 400.0)

    def test_rejects_an_empty_latency_sequence(self) -> None:
        """Catches silently emitting a misleading all-zero benchmark summary."""
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize_latency_ms([])


class ResetTrackersTests(unittest.TestCase):
    def test_resets_existing_trackers_without_reloading_the_model(self) -> None:
        """Catches a cold engine reload being used merely to start fresh track IDs."""

        class FakeTracker:
            def __init__(self) -> None:
                self.reset_calls = 0

            def reset(self) -> None:
                self.reset_calls += 1

        class FakePredictor:
            def __init__(self, trackers: list[FakeTracker]) -> None:
                self.trackers = trackers

        class FakeModel:
            def __init__(self, trackers: list[FakeTracker]) -> None:
                self.predictor = FakePredictor(trackers)

        first, second = FakeTracker(), FakeTracker()
        reset_trackers(FakeModel([first, second]))

        self.assertEqual(first.reset_calls, 1)
        self.assertEqual(second.reset_calls, 1)
