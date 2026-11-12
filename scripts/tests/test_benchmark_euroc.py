import importlib.util
import os
from pathlib import Path
import tempfile
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


class ImageMessageCounterTests(unittest.TestCase):
    def test_stereo_counts_pairs_and_all_messages_separately(self):
        counter = benchmark.ImageMessageCounter(["/left", "/right"])
        for topic in ("/left", "/right", "/left", "/right"):
            counter.record(topic)
        self.assertEqual(counter.metrics(), {
            "received_images": 2,
            "received_image_messages": 4,
            "received_images_by_topic": {"/left": 2, "/right": 2},
        })

    def test_mono_keeps_message_count_equal_to_frame_count(self):
        counter = benchmark.ImageMessageCounter(["/left"])
        counter.record("/left")
        counter.record("/left")
        self.assertEqual(counter.metrics(), {
            "received_images": 2,
            "received_image_messages": 2,
            "received_images_by_topic": {"/left": 2},
        })

    def test_unexpected_topic_is_rejected(self):
        counter = benchmark.ImageMessageCounter(["/left", "/right"])
        with self.assertRaisesRegex(ValueError, "unexpected topic"):
            counter.record("/other")


class CpuAffinityTests(unittest.TestCase):
    def test_parse_cpu_affinity_accepts_available_ids(self):
        self.assertEqual(benchmark.parse_cpu_affinity("6, 7", {0, 6, 7}), [6, 7])
        self.assertIsNone(benchmark.parse_cpu_affinity(None, {6, 7}))

    def test_parse_cpu_affinity_rejects_unavailable_or_malformed_ids(self):
        invalid_cases = [
            ("6,8", {6, 7}, "unavailable"),
            ("6,,7", {6, 7}, "comma-separated"),
            ("6-7", {6, 7}, "only integer"),
            ("6,6", {6, 7}, "duplicate"),
            ("-1,6", {6, 7}, "nonnegative"),
        ]
        for value, available, message in invalid_cases:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                benchmark.parse_cpu_affinity(value, available)

    def test_estimator_command_uses_taskset_prefix_only_with_affinity(self):
        binary = Path("/opt/vins/bin/vins_node")
        config = Path("/tmp/config.yaml")
        output = Path("/tmp/output/run")
        base = [
            str(binary), "--ros-args", "-r", "__node:=vins_estimator",
            "-p", f"config_file:={config}", "-p", f"output_path:={output}",
            "-p", "use_sim_time:=true", "-p", "opencv_threads:=1",
        ]
        self.assertEqual(
            benchmark.build_estimator_command(
                binary, config, output, ["opencv_threads:=1"], None, "/usr/bin/taskset"),
            base)
        self.assertEqual(
            benchmark.build_estimator_command(
                binary, config, output, ["opencv_threads:=1"], [6, 7], "/usr/bin/taskset"),
            ["/usr/bin/taskset", "-c", "6,7", *base])


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


class ReplayConfigTests(unittest.TestCase):
    def load_config(self, parameters, overrides=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(benchmark.yaml.safe_dump({
                "/**": {"ros__parameters": parameters},
            }))
            return benchmark.load_replay_config(path, overrides)

    def test_mono_imu_plays_only_the_primary_camera(self):
        replay = self.load_config({
            "imu": 1,
            "num_of_cam": 1,
            "imu_topic": "/sensors/imu",
            "image0_topic": "/sensors/left",
        })
        self.assertEqual(replay["mode"], "mono_imu")
        self.assertTrue(replay["use_imu"])
        self.assertEqual(replay["playback_topics"], ["/sensors/imu", "/sensors/left"])
        self.assertEqual(replay["topics"], {
            "imu": "/sensors/imu",
            "image0": "/sensors/left",
        })

    def test_stereo_imu_uses_both_image_topics(self):
        replay = self.load_config({
            "imu": 1,
            "num_of_cam": 2,
            "imu_topic": "/imu",
            "image0_topic": "/left",
            "image1_topic": "/right",
        })
        self.assertEqual(replay["mode"], "stereo_imu")
        self.assertEqual(replay["image_topics"], ["/left", "/right"])
        self.assertEqual(replay["playback_topics"], ["/imu", "/left", "/right"])

    def test_topic_parameter_overrides_are_applied_to_stereo_replay(self):
        replay = self.load_config(
            {
                "imu": 1,
                "num_of_cam": 2,
                "imu_topic": "/imu",
                "image0_topic": "/left",
                "image1_topic": "/right",
            },
            ["imu_topic:=/bag/imu", "image0_topic:=/bag/left",
             "image1_topic:=/bag/right"],
        )
        self.assertEqual(replay["playback_topics"],
                         ["/bag/imu", "/bag/left", "/bag/right"])
        self.assertEqual(replay["effective_parameters"]["image1_topic"], "/bag/right")

    def test_stereo_without_imu_excludes_imu_from_playback(self):
        replay = self.load_config({
            "imu": 0,
            "num_of_cam": 2,
            "image0_topic": "/left",
            "image1_topic": "/right",
        })
        self.assertEqual(replay["mode"], "stereo")
        self.assertFalse(replay["use_imu"])
        self.assertEqual(replay["playback_topics"], ["/left", "/right"])

    def test_repository_mono_and_stereo_configs_resolve(self):
        config_root = Path(__file__).parents[2] / "src/vins_estimator/config/euroc"
        mono = benchmark.load_replay_config(config_root / "euroc_mono_imu_config.yaml")
        stereo = benchmark.load_replay_config(config_root / "euroc_stereo_imu_config.yaml")
        self.assertEqual(mono["mode"], "mono_imu")
        self.assertEqual(mono["playback_topics"], ["/imu0", "/cam0/image_raw"])
        self.assertEqual(stereo["mode"], "stereo_imu")
        self.assertEqual(stereo["playback_topics"],
                         ["/imu0", "/cam0/image_raw", "/cam1/image_raw"])

    def test_invalid_replay_parameters_are_rejected(self):
        valid = {
            "imu": 1,
            "num_of_cam": 2,
            "imu_topic": "/imu",
            "image0_topic": "/left",
            "image1_topic": "/right",
        }
        invalid_cases = [
            ({**valid, "imu": 2}, "imu must be 0 or 1"),
            ({**valid, "num_of_cam": 3}, "num_of_cam must be 1 or 2"),
            ({**valid, "image1_topic": ""}, "image1_topic must be a non-empty string"),
            ({**valid, "image1_topic": "right"}, "absolute ROS topic"),
            ({**valid, "image1_topic": "/left"}, "must be distinct"),
            ({key: value for key, value in valid.items() if key != "image1_topic"},
             "image1_topic is required"),
        ]
        for parameters, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.load_config(parameters)

    def test_malformed_parameter_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected NAME:=VALUE"):
            benchmark.parse_parameter_overrides(["opencv_threads=2"])
        with self.assertRaisesRegex(ValueError, "absolute ROS topic"):
            self.load_config({
                "imu": 1,
                "num_of_cam": 1,
                "imu_topic": "/imu",
                "image0_topic": "/left",
            }, ["image0_topic:=left"])

    def test_mode_parameter_overrides_are_rejected(self):
        parameters = {
            "imu": 1,
            "num_of_cam": 2,
            "imu_topic": "/imu",
            "image0_topic": "/left",
            "image1_topic": "/right",
        }
        for override in ("imu:=0", "num_of_cam:=1"):
            with self.subTest(override=override), self.assertRaisesRegex(
                    ValueError, "is not supported by vins_node; set .* in the YAML config"):
                self.load_config(parameters, [override])

    def test_config_requires_one_ros_parameters_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("imu: 1\nnum_of_cam: 1\n")
            with self.assertRaisesRegex(ValueError, "no ros__parameters"):
                benchmark.load_ros2_parameters(path)
            path.write_text(benchmark.yaml.safe_dump({
                "/first": {"ros__parameters": {"imu": 1}},
                "/second": {"ros__parameters": {"imu": 0}},
            }))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                benchmark.load_ros2_parameters(path)


class RestartDiagnosticsTests(unittest.TestCase):
    def test_legacy_markers_are_counted_separately(self):
        diag = benchmark.parse_restart_diagnostics(
            "failure detection!\nsystem reboot!\nfailure detection!\n")
        self.assertEqual(diag["legacy_failure_reboots"], 3)
        self.assertEqual(diag["by_reason"]["legacy"], 3)
        self.assertEqual(diag["restart_requests"], 0)
        self.assertEqual(diag["reset_entry_events"], 0)
        self.assertIsNone(diag["actual_resets_completed"])

    def test_schur_request_and_reset_entry_are_separate(self):
        log = (
            "[ERROR] [1794300470.792818791] [vins_fusion]: Schur EKF visually starved "
            "for 3.100 s (> 3.000 s) at frame 74: streak=31 skipped_total=44 "
            "max_no_update_gap=3.100 s; scheduling a controlled estimator restart\n"
            "[WARN] [1794300470.794710487] [vins_fusion]: Resetting estimator state "
            "after solver failure; current-frame observations and IMU preintegration "
            "inputs were already consumed\n"
        )
        diag = benchmark.parse_restart_diagnostics(log)
        self.assertEqual(diag["restart_requests"], 1)
        self.assertEqual(diag["reset_entry_events"], 1)
        self.assertNotIn("restart_executions", diag)
        self.assertIsNone(diag["actual_resets_completed"])
        self.assertEqual(diag["starvation_restarts"], 1)
        self.assertEqual(diag["by_reason"]["starvation"], 1)
        self.assertEqual([event["kind"] for event in diag["events"]],
                         ["request", "reset_entry"])
        self.assertEqual(diag["events"][1]["reason"], "unknown")
        self.assertAlmostEqual(diag["events"][0]["timestamp"], 1794300470.792818791)

    def test_repeated_requests_do_not_inflate_reset_entries(self):
        log = (
            "[ERROR] [1.0] [vins_fusion]: Schur EKF visually starved for 3.100 s "
            "(> 3.000 s) at frame 74; scheduling a controlled estimator restart\n"
            "[ERROR] [2.0] [vins_fusion]: Schur EKF produced 60 consecutive frames "
            "without a visual update (99 skipped in total); scheduling a controlled "
            "estimator restart\n"
            "[WARN] [3.0] [vins_fusion]: Resetting estimator state after solver failure; ...\n"
        )
        diag = benchmark.parse_restart_diagnostics(log)
        self.assertEqual(diag["restart_requests"], 2)
        self.assertEqual(diag["reset_entry_events"], 1)
        self.assertEqual(diag["starvation_restarts"], 2)
        self.assertEqual(diag["by_reason"]["starvation"], 1)
        self.assertEqual(diag["by_reason"]["backstop"], 1)

    def test_soft_escalation_and_failure_detection_requests(self):
        log = (
            "[ERROR] [3.0] [vins_fusion]: Schur EKF soft restart at t=2.0 did not "
            "apply a visual update for 10.0 s (> 10.0 s) at frame 99; escalating "
            "to a full controlled estimator restart\n"
            "[WARN] [3.1] [vins_fusion]: Failure detection triggered; scheduling "
            "a controlled estimator reset\n"
        )
        diag = benchmark.parse_restart_diagnostics(log)
        self.assertEqual(diag["soft_restart_escalations"], 1)
        self.assertEqual(diag["failure_detection_triggers"], 1)
        self.assertEqual(diag["restart_requests"], 2)
        self.assertEqual(diag["by_reason"]["soft_escalation"], 1)
        self.assertEqual(diag["by_reason"]["failure_detection"], 1)

    def test_initialization_and_solver_failure_requests(self):
        log = (
            "[WARN] [4.0] [vins_fusion]: Solver failed during monocular IMU "
            "initialization; ... a controlled estimator reset is scheduled "
            "before the next frame\n"
            "[WARN] [4.1] [vins_fusion]: Initialization image history still has "
            "20 frames after trimming (limit 11); scheduling a controlled "
            "estimator reset\n"
        )
        diag = benchmark.parse_restart_diagnostics(log)
        self.assertEqual(diag["restart_requests"], 2)
        self.assertEqual(diag["by_reason"]["solver_failure"], 1)
        self.assertEqual(diag["by_reason"]["initialization"], 1)

    def test_config_and_cooldown_summary_do_not_count_as_requests(self):
        log = (
            "[INFO] [5.0] [vins_fusion]: Backend type: schur_ekf "
            "(schur window=5 warmup=30 ... restart_cooldown_seconds=8.000 "
            "soft_restart=0 soft_restart_max_seconds=10.000 ...)\n"
            "[INFO] [5.1] [vins_fusion]: schur_ekf starvation summary (100 frames): "
            "time_gate_triggers=3 deep_escalations=5 cooldown_suppressions=0 ... "
            "thresholds(max_no_update=3.00 s cooldown=8.00 s) ...\n"
        )
        diag = benchmark.parse_restart_diagnostics(log)
        self.assertEqual(diag["restart_requests"], 0)
        self.assertEqual(diag["reset_entry_events"], 0)
        self.assertEqual(diag["events"], [])

    def test_real_difficult_baseline_markers(self):
        log = (
            "[INFO] [1794300452.659986045] [vins_fusion]: Backend type: schur_ekf "
            "(schur window=5 warmup=30 ... restart_cooldown_seconds=8.000 "
            "soft_restart=0 soft_restart_max_seconds=10.000 ...)\n"
            "[ERROR] [1794300470.792818791] [vins_fusion]: Schur EKF visually starved "
            "for 3.100 s (> 3.000 s) at frame 74: streak=31 skipped_total=44 "
            "max_no_update_gap=3.100 s; scheduling a controlled estimator restart\n"
            "[WARN] [1794300470.794710487] [vins_fusion]: Resetting estimator state "
            "after solver failure; ...\n"
            "[ERROR] [1794300482.295052310] [vins_fusion]: Schur EKF visually starved "
            "for 3.100 s (> 3.000 s) at frame 100: streak=31 skipped_total=54 "
            "max_no_update_gap=3.100 s; scheduling a controlled estimator restart\n"
            "[WARN] [1794300482.297338462] [vins_fusion]: Resetting estimator state "
            "after solver failure; ...\n"
            "[ERROR] [1794300508.793828317] [vins_fusion]: Schur EKF visually starved "
            "for 3.100 s (> 3.000 s) at frame 184: streak=31 skipped_total=121 "
            "max_no_update_gap=3.100 s; scheduling a controlled estimator restart\n"
            "[WARN] [1794300508.795864097] [vins_fusion]: Resetting estimator state "
            "after solver failure; ...\n"
        )
        diag = benchmark.parse_restart_diagnostics(log)
        self.assertEqual(diag["restart_requests"], 3)
        self.assertEqual(diag["reset_entry_events"], 3)
        self.assertEqual(diag["starvation_restarts"], 3)
        self.assertEqual(diag["by_reason"]["starvation"], 3)
        self.assertEqual(diag["legacy_failure_reboots"], 0)

    def test_file_wrapper(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write("failure detection!\n")
            path = handle.name
        try:
            diag = benchmark.parse_restart_diagnostics_file(path)
        finally:
            os.unlink(path)
        self.assertEqual(diag["legacy_failure_reboots"], 1)


if __name__ == "__main__":
    unittest.main()
