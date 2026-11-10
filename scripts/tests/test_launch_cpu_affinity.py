"""Exercise the real launch prefix without starting VINS or ROS processes."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from launch import LaunchContext
from launch.substitutions import LaunchConfiguration
from launch.utilities import perform_substitutions

LAUNCH_PATH = Path(__file__).resolve().parents[2] / "src/vins_estimator/launch/vins.launch.py"
spec = importlib.util.spec_from_file_location("vins_affinity_launch", LAUNCH_PATH)
launch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch_module)


class CpuAffinityLaunchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / "config with spaces.yaml"
        self.context = LaunchContext()
        self.context.launch_configurations["config_file"] = str(self.config)

    def prefix(self, text):
        self.config.write_text(text)
        return launch_module.CpuAffinityPrefix(
            LaunchConfiguration("config_file")).perform(self.context)

    def test_missing_or_disabled_option_needs_no_taskset(self):
        with mock.patch.object(launch_module.shutil, "which", return_value=None):
            self.assertEqual(self.prefix("/**:\n  ros__parameters:\n    imu: 1\n"), "")
            self.assertEqual(self.prefix(
                "/**:\n  ros__parameters:\n    cpu_affinity_enabled: false\n"
                "    cpu_affinity_cores: [999999]\n"), "")

    def test_enabled_launch_command_binds_a_real_child_process(self):
        cpu = min(os.sched_getaffinity(0))
        prefix = self.prefix(
            "/**:\n  ros__parameters:\n    cpu_affinity_enabled: true\n"
            f"    cpu_affinity_cores: [{cpu}, {cpu}]\n")
        command = shlex.split(prefix)
        self.assertEqual(command[1:], ["--cpu-list", str(cpu)])
        proc = subprocess.run(command + [sys.executable, "-c",
            "import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))"],
            check=True, capture_output=True, text=True, timeout=10)
        self.assertEqual(json.loads(proc.stdout), [cpu])
        self.assertIn(cpu, os.sched_getaffinity(0))

    def test_disabled_legacy_config_ignores_core_list(self):
        with mock.patch.object(launch_module.shutil, "which", return_value=None):
            self.assertEqual(self.prefix(
                '%YAML:1.0\n---\ncpu_affinity_enabled: 0\ncpu_affinity_cores: "unused"\n'), "")

    def test_legacy_opencv_config(self):
        with mock.patch.object(launch_module.os, "sched_getaffinity", return_value={6, 7}):
            command = shlex.split(self.prefix(
                "%YAML:1.0\n---\ncpu_affinity_enabled: 1\ncpu_affinity_cores: [7, 6, 7]\n"))
        self.assertEqual(command[1:], ["--cpu-list", "6,7"])

    def test_invalid_core_types_and_disabled_taskset(self):
        for cores in ("[]", "[-1]", "[true]", "[1.5]", "'6,7'", "[null]"):
            with self.subTest(cores=cores), self.assertRaises((ValueError, RuntimeError)):
                self.prefix("/**:\n  ros__parameters:\n    cpu_affinity_enabled: true\n"
                            f"    cpu_affinity_cores: {cores}\n")
        with mock.patch.object(launch_module.os, "sched_getaffinity", return_value={6}), \
                mock.patch.object(launch_module.shutil, "which", return_value=None), \
                self.assertRaisesRegex(RuntimeError, "taskset"):
            self.prefix("/**:\n  ros__parameters:\n    cpu_affinity_enabled: true\n"
                        "    cpu_affinity_cores: [6]\n")

    def test_unavailable_cpu_and_invalid_switch_fail_clearly(self):
        with mock.patch.object(launch_module.os, "sched_getaffinity", return_value={0, 1}), \
                self.assertRaises((ValueError, RuntimeError)):
            self.prefix("/**:\n  ros__parameters:\n    cpu_affinity_enabled: true\n"
                        "    cpu_affinity_cores: [6, 7]\n")
        for enabled in ("'false'", "2", "[]"):
            with self.subTest(enabled=enabled), self.assertRaises((ValueError, RuntimeError)):
                self.prefix("/**:\n  ros__parameters:\n"
                            f"    cpu_affinity_enabled: {enabled}\n")

    def test_invalid_or_missing_file_reports_path(self):
        with self.assertRaisesRegex(RuntimeError, "config with spaces.yaml"):
            launch_module.CpuAffinityPrefix(LaunchConfiguration("config_file")).perform(self.context)
        with self.assertRaisesRegex(RuntimeError, "config with spaces.yaml"):
            self.prefix("/**: [unterminated\n")

    def test_only_estimator_node_uses_affinity(self):
        self.config.write_text("/**:\n  ros__parameters:\n    cpu_affinity_enabled: false\n")
        description = launch_module.generate_launch_description()
        from launch_ros.actions import Node
        for node in description.entities:
            if not isinstance(node, Node):
                continue
            prefix = node.process_description.prefix
            if node.node_executable == "vins_node":
                self.assertTrue(any(isinstance(item, launch_module.CpuAffinityPrefix) for item in prefix))
                self.assertEqual(perform_substitutions(self.context, prefix), "")
            else:
                self.assertFalse(any(isinstance(item, launch_module.CpuAffinityPrefix) for item in prefix))

    def test_standard_configs_disable_affinity_and_arm_profile_binds_a76(self):
        for path in (LAUNCH_PATH.parents[1] / "config").rglob("*config*.yaml"):
            with self.subTest(config=path):
                self.context.launch_configurations["config_file"] = str(path)
                prefix = launch_module.CpuAffinityPrefix(LaunchConfiguration("config_file"))
                if path.name == "euroc_mono_imu_arm_balanced_config.yaml":
                    with mock.patch.object(launch_module.os, "sched_getaffinity", return_value={6, 7}), \
                            mock.patch.object(launch_module.shutil, "which", return_value="/usr/bin/taskset"):
                        self.assertEqual(shlex.split(prefix.perform(self.context)),
                                         ["/usr/bin/taskset", "--cpu-list", "6,7"])
                else:
                    self.assertEqual(prefix.perform(self.context), "")


if __name__ == "__main__":
    unittest.main()
