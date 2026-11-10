"""Synthetic tests for scripts/summarize_balanced_optimization.py."""

import importlib.util
import json
import tempfile
import textwrap
import unittest
from unittest import mock
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "summarize_balanced_optimization.py"
spec = importlib.util.spec_from_file_location("summarize_balanced", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)

STATS = (
    "Tracker periodic stats: frames=100 total_frames=100 queue_wait_ms=11/12 "
    "decode_ms=1/2 track_ms=3/4 pyramid_ms=5/6 flow_ms=7/8 detect_ms=9/10 "
    "input_image_ms=13/14 features_avg=100/20 overflow_events=0 overflow_dropped_pairs=0"
)


def write_run(label: Path, dataset: str, repeat: int, stats: str = STATS) -> None:
    run = label / f"{dataset}_r{repeat}"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"dataset": dataset, "cpu_seconds": 1.0}))
    (run / "estimator.log").write_text(stats + "\n")
    (run / "trajectory.csv").write_text("t,x,y,z\n1.0,1,2,3\n2.0,4,5,6\n")


class SummaryTest(unittest.TestCase):
    def test_repeated_runs_are_not_lost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in (root / "baseline", root / "optimized"):
                write_run(label, "DS", 1)
                write_run(label, "DS", 2)
            output = root / "out.json"
            import sys
            old = sys.argv
            try:
                sys.argv = ["summarize", "--baseline", str(root / "baseline"),
                            "--candidate", str(root / "optimized"), "--output", str(output)]
                self.assertEqual(summary.main(), 0)
            finally:
                sys.argv = old
            report = json.loads(output.read_text())
            self.assertEqual(sorted(report["runs"]), ["DS_r1", "DS_r2"])
            self.assertFalse(report["single_run_screening"])

    def test_final_frames_zero_and_overflow(self) -> None:
        log = textwrap.dedent(
            STATS + "\n"
            "Tracker final stats: frames=0 total_frames=100 overflow_events=3 "
            "overflow_dropped_pairs=7\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "estimator.log"
            path.write_text(log)
            result = summary.tracker_summary(path)
        self.assertEqual((result["sum_frames"], result["total_frames"]), (100, 100))
        self.assertTrue(result["frames_consistent"])
        self.assertEqual((result["overflow_events_cumulative"],
                          result["overflow_dropped_pairs_cumulative"]), (3, 7))
        self.assertEqual(result["features"], {"tracked_mean": 100.0, "detected_mean": 20.0})

    def test_frames_mismatch_is_false(self) -> None:
        log = textwrap.dedent(
            "Tracker periodic stats: frames=90 total_frames=100 track_ms=1/2\n"
            "Tracker final stats: frames=0 total_frames=100 overflow_events=0 "
            "overflow_dropped_pairs=0\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "estimator.log"
            path.write_text(log)
            result = summary.tracker_summary(path)
        self.assertFalse(result["frames_consistent"])

    def test_restart_parser_is_called(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "estimator.log"
            path.write_text(STATS + "\n")
            with mock.patch.object(
                summary.benchmark_euroc, "parse_restart_diagnostics",
                return_value={"restart_requests": 3}, create=True
            ) as parser:
                result = summary.tracker_summary(path)
        parser.assert_called_once()
        self.assertEqual(result["restart_diagnostics"]["restart_requests"], 3)

    def test_identical_trajectory_delta_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root / "baseline", "DS", 1)
            write_run(root / "optimized", "DS", 1)
            base = {"dir": root / "baseline" / "DS_r1", "metrics": {}}
            cand = {"dir": root / "optimized" / "DS_r1", "metrics": {}}
            result = summary.compare_dataset(base, cand, root / "missing.csv")
        self.assertEqual(result["common_timestamps"], 2)
        self.assertEqual(result["max_abs_position_delta_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
