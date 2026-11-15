import csv
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "benchmark_euroc", Path(__file__).parents[1] / "benchmark_euroc.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def cloud_message(stamp_s, coordinates, frame_id="world"):
    sec = int(stamp_s)
    nanosec = int(round((stamp_s - sec) * 1e9))
    if nanosec >= 10**9:
        sec += 1
        nanosec -= 10**9
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=sec, nanosec=nanosec),
            frame_id=frame_id,
        ),
        points=[SimpleNamespace(x=x, y=y, z=z) for x, y, z in coordinates],
    )


class CloudMetricsCollectorTests(unittest.TestCase):
    def test_empty_data_has_zero_counts_and_no_statistics(self):
        collector = benchmark.CloudMetricsCollector("/cam0/image_raw")
        metrics = collector.metrics()
        self.assertTrue(metrics["enabled"])
        self.assertTrue(metrics["cumulative_points_are_not_unique_map_points"])
        self.assertIn("not a unique map-landmark count",
                      metrics["semantics"]["total_points"])
        json.dumps(metrics)

        for name in ("point_cloud", "margin_cloud"):
            full = metrics["topics"][name]["segments"]["full_sequence"]
            self.assertTrue(full["available"])
            self.assertEqual(full["messages"], 0)
            self.assertEqual(full["nonempty_messages"], 0)
            self.assertEqual(full["total_points"], 0)
            self.assertEqual(full["total_finite_points"], 0)
            self.assertEqual(full["total_invalid_points"], 0)
            self.assertIsNone(full["points_per_frame_mean"])
            self.assertIsNone(full["points_per_frame_p50"])
            self.assertIsNone(full["points_per_frame_p95"])
            self.assertIsNone(full["points_per_sensor_second"])
            self.assertIsNone(full["first_header_stamp_s"])
            self.assertIsNone(full["last_header_stamp_s"])

            post20 = metrics["topics"][name]["segments"][
                "after_first_left_image_plus_20s"]
            self.assertFalse(post20["available"])
            self.assertIn("no left image", post20["reason"])
            self.assertEqual(post20["messages"], 0)
            self.assertEqual(metrics["topics"][name]["messages"], [])

    def test_invalid_and_empty_points_are_counted(self):
        collector = benchmark.CloudMetricsCollector("/cam0/image_raw")
        collector.record_primary_image("/cam0/image_raw", 10_000_000_000)
        collector.record_cloud("point_cloud", cloud_message(10.1, [
            (0.0, 0.0, 0.0),
            (math.nan, 1.0, 2.0),
            (3.0, math.inf, 4.0),
            (5.0, 6.0, 7.0),
        ]))
        collector.record_cloud("margin_cloud", cloud_message(10.2, []))

        metrics = collector.metrics()
        point_cloud = metrics["topics"]["point_cloud"]
        full = point_cloud["segments"]["full_sequence"]
        self.assertEqual(full["messages"], 1)
        self.assertEqual(full["nonempty_messages"], 1)
        self.assertEqual(full["empty_messages"], 0)
        self.assertEqual(full["total_points"], 4)
        self.assertEqual(full["total_finite_points"], 2)
        self.assertEqual(full["total_invalid_points"], 2)
        self.assertEqual(full["messages_with_invalid_points"], 1)
        self.assertEqual(full["invalid_point_ratio"], 0.5)
        self.assertEqual(full["frame_ids"], ["world"])
        self.assertEqual(full["points_per_frame_mean"], 4.0)
        self.assertEqual(full["points_per_frame_p50"], 4.0)
        self.assertEqual(full["points_per_frame_p95"], 4.0)
        self.assertAlmostEqual(full["first_header_stamp_s"], 10.1)
        self.assertAlmostEqual(full["last_header_stamp_s"], 10.1)

        records = point_cloud["messages"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["point_count"], 4)
        self.assertEqual(records[0]["finite_count"], 2)
        self.assertEqual(records[0]["invalid_count"], 2)
        self.assertEqual(records[0]["frame_id"], "world")
        self.assertFalse(records[0]["after_post20s"])

        margin = metrics["topics"]["margin_cloud"]["segments"]["full_sequence"]
        self.assertEqual(margin["messages"], 1)
        self.assertEqual(margin["nonempty_messages"], 0)
        self.assertEqual(margin["empty_messages"], 1)
        self.assertEqual(margin["total_points"], 0)
        self.assertEqual(margin["points_per_frame_mean"], 0.0)

    def test_post20s_segment_includes_threshold_stamp(self):
        collector = benchmark.CloudMetricsCollector("/cam0/image_raw")
        collector.record_primary_image("/cam0/image_raw", 10_000_000_000)
        for stamp, count in ((10.0, 1), (29.999, 2), (30.0, 3), (31.0, 4)):
            collector.record_cloud(
                "point_cloud",
                cloud_message(stamp, [(float(i), 0.0, 0.0) for i in range(count)]))

        metrics = collector.metrics()
        point_cloud = metrics["topics"]["point_cloud"]
        full = point_cloud["segments"]["full_sequence"]
        post20 = point_cloud["segments"]["after_first_left_image_plus_20s"]

        self.assertEqual(full["messages"], 4)
        self.assertEqual(full["total_points"], 10)
        self.assertAlmostEqual(full["points_per_frame_mean"], 2.5)
        self.assertAlmostEqual(full["points_per_frame_p50"], 2.5)
        self.assertAlmostEqual(full["points_per_frame_p95"], 3.85)

        self.assertTrue(post20["available"])
        self.assertAlmostEqual(post20["window_start_s"], 30.0)
        self.assertEqual(post20["messages"], 2)
        self.assertEqual(post20["nonempty_messages"], 2)
        self.assertEqual(post20["total_points"], 7)
        self.assertAlmostEqual(post20["points_per_frame_mean"], 3.5)
        self.assertAlmostEqual(post20["points_per_frame_p50"], 3.5)
        self.assertAlmostEqual(post20["points_per_frame_p95"], 3.95)
        self.assertAlmostEqual(post20["first_header_stamp_s"], 30.0)
        self.assertAlmostEqual(post20["last_header_stamp_s"], 31.0)
        self.assertAlmostEqual(post20["sensor_span_s"], 1.0)
        self.assertEqual(post20["points_per_sensor_second"], 7.0)
        self.assertEqual(
            [record["after_post20s"] for record in point_cloud["messages"]],
            [False, False, True, True])

    def test_population_statistics_and_optional_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = benchmark.CloudMetricsCollector(
                "/cam0/image_raw", csv_directory=directory)
            collector.record_primary_image("/cam0/image_raw", 0)
            for stamp, count in enumerate((0, 2, 4, 6, 8)):
                collector.record_cloud(
                    "point_cloud",
                    cloud_message(float(stamp),
                                  [(float(i), 1.0, 2.0) for i in range(count)]))
            collector.close()

            metrics = collector.metrics()
            full = metrics["topics"]["point_cloud"]["segments"]["full_sequence"]
            self.assertEqual(full["messages"], 5)
            self.assertEqual(full["nonempty_messages"], 4)
            self.assertEqual(full["total_points"], 20)
            self.assertAlmostEqual(full["points_per_frame_mean"], 4.0)
            self.assertAlmostEqual(full["points_per_frame_p50"], 4.0)
            self.assertAlmostEqual(full["points_per_frame_p95"], 7.6)
            self.assertAlmostEqual(full["sensor_span_s"], 4.0)
            self.assertAlmostEqual(full["points_per_sensor_second"], 5.0)

            post20 = metrics["topics"]["point_cloud"]["segments"][
                "after_first_left_image_plus_20s"]
            self.assertTrue(post20["available"])
            self.assertEqual(post20["messages"], 0)
            self.assertIsNone(post20["points_per_frame_mean"])

            with open(Path(directory) / "point_cloud_frames.csv", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["topic"], "point_cloud")
            self.assertEqual(rows[0]["ros_topic"], "/vins_estimator/point_cloud")
            self.assertEqual(rows[0]["header_stamp_ns"], "0")
            self.assertEqual(rows[0]["point_count"], "0")
            self.assertEqual(rows[2]["point_count"], "4")
            self.assertEqual(rows[2]["frame_id"], "world")

            with open(Path(directory) / "margin_cloud_frames.csv", newline="") as handle:
                self.assertEqual(list(csv.reader(handle)), [list(
                    benchmark._CLOUD_CSV_COLUMNS)])

    def test_unexpected_cloud_topic_is_rejected(self):
        collector = benchmark.CloudMetricsCollector("/cam0/image_raw")
        with self.assertRaisesRegex(ValueError, "unexpected topic"):
            collector.record_cloud("keyframe_point", cloud_message(1.0, []))


if __name__ == "__main__":
    unittest.main()
