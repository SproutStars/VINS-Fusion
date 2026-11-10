"""Synthetic tests for scripts/compare_frontend_modes.py."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "compare_frontend_modes.py"
spec = importlib.util.spec_from_file_location("compare_frontend_modes", SCRIPT)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)

STATS = (
    "Tracker final stats: frames=100 total_frames=100 overflow_events=0 "
    "overflow_dropped_pairs=0 track_ms=1/2 detect_ms=3/4 features_avg=80/20"
)
RESTART = (
    "[ERROR] scheduling a controlled estimator reset\n"
    "[WARN] Resetting estimator state after solver failure\n"
)
def truth_at(stamp: float):
    return (stamp, 0.2 * stamp, 0.1 * stamp * stamp, 0.05 * stamp)


def write_truth(root: Path, dataset: str, duration: float = 5.0, step: float = 0.05) -> None:
    path = root / "groundtruth" / f"{dataset}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [truth_at(i * step) for i in range(int(round(duration / step)) + 1)]
    lines = ["t,x,y,z"] + [",".join(str(value) for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def write_run(root: Path, label: str, dataset: str, repeat: int, *, times=(0.0, 1.0, 2.0, 3.0, 4.0),
              offset=0.01, scale=1.0, metric_ate=None, cpu=2.0, received=10, matched=5,
              output=5, duration=4.0, gap=0.1, restart=False, empty_trajectory=False) -> Path:
    run = root / label / f"{dataset}_r{repeat}"
    run.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": dataset, "label": label, "rate": 1.0, "wall_s": 20.0,
        "cpu_seconds": cpu, "average_cpu_percent_one_core": 10.0,
        "peak_rss_mib": 100.0, "peak_threads": 20, "output_poses": output,
        "received_images": received, "latency_misses": 0,
        "latency_p50_ms": 5.0, "latency_p95_ms": 8.0, "latency_p99_ms": 9.0,
        "failure_reboots": 0, "matched_poses": matched,
        "ate_rmse_m": metric_ate if metric_ate is not None else offset,
        "ate_median_m": metric_ate if metric_ate is not None else offset,
        "ate_p95_m": 1.5 * (metric_ate if metric_ate is not None else offset),
        "trajectory_duration_s": duration, "max_output_gap_s": gap,
        "restart_requests": 1 if restart else 0,
        "reset_entry_events": 1 if restart else 0,
    }
    (run / "metrics.json").write_text(json.dumps(metrics) + "\n")
    (run / "run_manifest.json").write_text(json.dumps({
        "label": label, "dataset": dataset, "repeat": repeat,
        "detector_label": label,
    }) + "\n")
    if empty_trajectory:
        (run / "trajectory.csv").write_text("t,x,y,z\n")
    else:
        rows = []
        for stamp in times:
            _, x, y, z = truth_at(stamp)
            rows.append([stamp, scale * x, scale * y + offset, scale * z])
        (run / "trajectory.csv").write_text(
            "t,x,y,z\n" + "\n".join(",".join(str(value) for value in row) for row in rows) + "\n")
    (run / "estimator.log").write_text(STATS + "\n" + (RESTART if restart else ""))
    return run


def run_tool(root: Path, labels, *extra) -> dict:
    output = root / "report.json"
    compare.main(["--root", str(root), "--labels", *labels, "--output", str(output), *extra])
    return json.loads(output.read_text())


class CompareFrontendModesTest(unittest.TestCase):
    def test_repeats_are_kept_and_raw_metrics_are_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_truth(root, "DS")
            metrics_paths = []
            for label in ("gftt", "fast_grid"):
                for repeat in (1, 2):
                    run = write_run(root, label, "DS", repeat)
                    metrics_paths.append(run / "metrics.json")
            before = {path: path.read_text() for path in metrics_paths}
            report = run_tool(root, ["gftt", "fast_grid"])
            self.assertEqual(sorted(report["runs"]), ["DS_r1", "DS_r2"])
            self.assertEqual(report["datasets"]["DS"], ["DS_r1", "DS_r2"])
            self.assertEqual(report["recommendation"]["labels"]["fast_grid"]["runs_checked"], 2)
            self.assertEqual(before, {path: path.read_text() for path in metrics_paths})

    def test_common_ate_uses_the_timestamp_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_truth(root, "DS")
            write_run(root, "gftt", "DS", 1, times=(0.0, 1.0, 2.0, 3.0, 4.0), offset=0.02)
            write_run(root, "fast_grid", "DS", 1, times=(1.0, 2.0, 3.0, 4.0, 5.0), offset=0.03)
            report = run_tool(root, ["gftt", "fast_grid"])
            common = report["runs"]["DS_r1"]["common_timestamp_ate"]
            self.assertEqual(common["common_timestamps"], 4)
            self.assertEqual(common["intersection_labels"], ["fast_grid", "gftt"])
            self.assertEqual(common["intersection_mode"], "2-label")
            truth = compare.dataset_truth(root, "DS", {}, [])
            poses = compare.summary.load_trajectory(root / "fast_grid" / "DS_r1" / "trajectory.csv")
            expected = compare.benchmark.trajectory_metrics(
                compare.np.array([[stamp, *poses[stamp]] for stamp in (1.0, 2.0, 3.0, 4.0)]), truth)
            self.assertAlmostEqual(
                common["labels"]["fast_grid"]["ate_rmse_m"], expected["ate_rmse_m"], places=9)

    def test_three_label_intersection_is_not_reused_as_two_label_common(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_truth(root, "DS")
            write_run(root, "gftt", "DS", 1, times=(0.0, 1.0, 2.0, 3.0, 4.0))
            write_run(root, "fast_grid", "DS", 1, times=(1.0, 2.0, 3.0, 4.0, 5.0))
            write_run(root, "gftt_pyramid", "DS", 1, times=(2.0, 3.0, 4.0))
            three = run_tool(root, ["gftt", "fast_grid", "gftt_pyramid"])
            common3 = three["runs"]["DS_r1"]["common_timestamp_ate"]
            self.assertEqual(common3["intersection_mode"], "3-label")
            self.assertEqual(common3["common_timestamps"], 3)
            self.assertEqual(common3["intersection_labels"], ["fast_grid", "gftt", "gftt_pyramid"])
            two = run_tool(root, ["gftt", "fast_grid"])
            common2 = two["runs"]["DS_r1"]["common_timestamp_ate"]
            self.assertEqual(common2["intersection_mode"], "2-label")
            self.assertEqual(common2["common_timestamps"], 4)
            self.assertEqual(common2["intersection_labels"], ["fast_grid", "gftt"])

    def test_empty_trajectory_is_reported_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_truth(root, "DS")
            write_run(root, "gftt", "DS", 1)
            write_run(root, "fast_grid", "DS", 1, empty_trajectory=True)
            report = run_tool(root, ["gftt", "fast_grid"])
            common = report["runs"]["DS_r1"]["common_timestamp_ate"]
            self.assertEqual(common["common_timestamps"], 0)
            self.assertFalse(common["comparable"])
            self.assertIsNone(common["labels"]["gftt"]["ate_rmse_m"])
            self.assertIsNone(common["labels"]["fast_grid"]["ate_rmse_m"])
            self.assertTrue(any("empty" in issue for issue in report["issues"]))

    def test_zero_restarts_do_not_make_bad_accuracy_reliable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_truth(root, "DS")
            write_run(root, "gftt", "DS", 1, offset=0.0, scale=1.0, metric_ate=0.01, restart=False)
            write_run(root, "fast_grid", "DS", 1, offset=0.0, scale=1.5, metric_ate=0.30, restart=False)
            report = run_tool(root, ["gftt", "fast_grid"])
            check = report["recommendation"]["labels"]["fast_grid"]
            self.assertTrue(check["all_restart_clean"])
            self.assertFalse(check["all_accuracy_within_manual_tolerance"])
            self.assertFalse(check["passes_manual_research_screen"])
            self.assertNotIn("fast_grid", report["recommendation"]["manual_research_shortlist"])
            self.assertIn("zero restarts do not prove ATE reliability", check["caveat"])
            self.assertIn("not user acceptance", report["recommendation"]["note"])

    def test_pose_per_image_and_temporal_coverage_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_truth(root, "DS")
            write_run(root, "gftt", "DS", 1, received=10, matched=5, duration=4.0, gap=0.25)
            write_run(root, "fast_grid", "DS", 1, received=10, matched=5, duration=4.0, gap=0.25)
            report = run_tool(root, ["gftt", "fast_grid"])
            absolute = report["runs"]["DS_r1"]["labels"]["gftt"]["absolute"]
            self.assertEqual(absolute["pose_per_image"], 0.5)
            self.assertEqual(absolute["temporal_coverage"], 0.8)
            self.assertEqual(absolute["bag_effective_duration_s"], 5.0)
            self.assertEqual(absolute["max_output_gap_s"], 0.25)
            self.assertNotIn("pose_coverage", absolute)
            self.assertIn("ate_rmse_m", absolute)
            self.assertIn("temporal_coverage", absolute)


if __name__ == "__main__":
    unittest.main()
