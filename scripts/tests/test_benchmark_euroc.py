import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("benchmark_euroc", Path(__file__).parents[1] / "benchmark_euroc.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class ArrivalTimesTests(unittest.TestCase):
    def test_crosses_microsecond_bucket_boundary(self):
        arrivals = benchmark.ArrivalTimes()
        camera = 1413393212255760500
        estimate = camera - 100
        self.assertNotEqual((camera + 500) // 1000, (estimate + 500) // 1000)
        arrivals.record(camera, 1000000)
        self.assertEqual(arrivals.latency_ms(estimate, 21000000), 20.0)

    def test_nearest_match_works_for_nonmonotonic_arrival(self):
        arrivals = benchmark.ArrivalTimes()
        arrivals.record(50000000, 2000000)
        arrivals.record(0, 1000000)
        self.assertEqual(arrivals.latency_ms(50000200, 22000000), 20.0)
        self.assertIsNone(arrivals.latency_ms(50002000, 22000000))
        self.assertIsNone(arrivals.latency_ms(0, 999999))

    def test_bounded_history(self):
        arrivals = benchmark.ArrivalTimes(2)
        for index in range(3):
            arrivals.record(index * 50000000, index)
        self.assertIsNone(arrivals.latency_ms(0, 10000000))
        self.assertEqual(len(arrivals.arrivals), 2)
        self.assertIsNone(benchmark.ArrivalTimes().latency_ms(0, 0))
        with self.assertRaises(ValueError):
            benchmark.ArrivalTimes(0)


class TrajectoryMetricsTests(unittest.TestCase):
    def test_rigid_alignment_does_not_fit_scale(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 1, 1]], dtype=float)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        estimated = positions @ rotation + np.array([4, 5, 6])
        np.testing.assert_allclose(benchmark.rigid_align(estimated, positions), positions, atol=1e-12)
        scaled = benchmark.rigid_align(positions * 2, positions)
        self.assertGreater(np.linalg.norm(scaled - positions), 0.5)

    def test_truth_outages_are_not_interpolated(self):
        truth = [[0, 0, 0, 0], [0.05, 1, 0, 0], [0.1, 2, 0, 0],
                 [1, 3, 0, 0], [1.05, 4, 0, 0], [1.1, 5, 0, 0]]
        estimate = [[0.025, 0.5, 0, 0], [0.075, 1.5, 0, 0],
                    [0.5, 100, 100, 100], [1.025, 3.5, 0, 0]]
        metrics = benchmark.trajectory_metrics(estimate, truth)
        self.assertEqual(metrics["matched_poses"], 3)
        self.assertLess(metrics["ate_rmse_m"], 1e-12)


if __name__ == "__main__":
    unittest.main()
