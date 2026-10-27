"""Lightweight local-mock tests for scripts/benchmark_remote_arm.sh.

The wrapper is exercised with a fake `ssh` that runs the generated remote
script locally against a temporary repository.  No real host, bag, ROS graph,
or ARM binary is touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

WRAPPER = Path(__file__).resolve().parents[1] / "benchmark_remote_arm.sh"


class RemoteArmWrapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vins-wrapper-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

        self.home = self.tmp / "home"
        self.home.mkdir()
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.logs = self.tmp / "logs"
        self.logs.mkdir()

        self.repo = self.home / "Code" / "Piny_drone_nav"
        self.vins = self.repo / "src" / "VINS-Fusion"
        self.output = self.home / "vins_remote_eval_20261019"
        self.output.mkdir(parents=True)

        self._make_repo()
        self._make_fake_tools()
        self._make_bags()

        self.lock = Path(tempfile.gettempdir()) / (
            "vins_remote_arm_"
            + hashlib.sha256(str(self.repo).encode()).hexdigest()[:16]
            + ".lock"
        )
        self.addCleanup(lambda: self.lock.unlink(missing_ok=True))

        self.env = os.environ.copy()
        self.env.update(
            {
                "HOME": str(self.home),
                "PATH": str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
                "FAKE_COLCON_LOG": str(self.logs / "colcon.log"),
                "FAKE_INSTALL_BIN": str(
                    self.repo / "install" / "vins" / "lib" / "vins" / "vins_node"
                ),
                "FAKE_BENCH_LOG": str(self.logs / "bench.log"),
                "SSH_HOST": "fake-host",
                "REMOTE_REPO": "Code/Piny_drone_nav",
                "REMOTE_OUTPUT": "vins_remote_eval_20261019",
                "REMOTE_BAGS": "bags",
            }
        )

    def _make_repo(self) -> None:
        scripts = self.vins / "scripts"
        scripts.mkdir(parents=True)
        euroc = self.vins / "src" / "vins_estimator" / "config" / "euroc"
        euroc.mkdir(parents=True)

        (scripts / "benchmark_euroc.py").write_text(
            "import json, os, sys\n"
            "with open(os.environ['FAKE_BENCH_LOG'], 'a') as f:\n"
            "    f.write(json.dumps(sys.argv) + '\\n')\n"
        )
        (euroc / "euroc_mono_imu_config.yaml").write_text(
            "/**:\n  ros__parameters:\n    max_cnt: 200\n"
        )

        subprocess.run(["git", "init", "-q", str(self.vins)], check=True)
        subprocess.run(["git", "-C", str(self.vins), "add", "-A"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.vins),
                "-c",
                "user.name=wrapper-test",
                "-c",
                "user.email=wrapper-test@example.invalid",
                "commit",
                "-q",
                "-m",
                "fixture",
            ],
            check=True,
        )
        # Submodules use a .git *file*, not a directory.  Exercise that layout.
        gitdir = self.tmp / "vins.git"
        (self.vins / ".git").rename(gitdir)
        (self.vins / ".git").write_text(f"gitdir: {gitdir}\n")

    def _make_fake_tools(self) -> None:
        ssh = textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -e
            while (( $# > 0 )) && [[ $1 == -* ]]; do
              case "$1" in
                -o) shift 2 ;;
                *) shift ;;
              esac
            done
            (( $# >= 2 )) || { echo "fake ssh: missing host/command" >&2; exit 99; }
            shift
            exec bash -c "$1"
            """
        )
        colcon = textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -e
            printf '%s\\n' "$*" >> "${FAKE_COLCON_LOG:?}"
            if [[ -n "${FAKE_COLCON_SLEEP:-}" ]]; then
              : > "${FAKE_COLCON_STARTED:?}"
              sleep "${FAKE_COLCON_SLEEP}"
            fi
            mkdir -p "$(dirname "${FAKE_INSTALL_BIN:?}")"
            printf '#!/bin/sh\\nexit 0\\n' > "${FAKE_INSTALL_BIN:?}"
            chmod +x "${FAKE_INSTALL_BIN:?}"
            """
        )
        file_cmd = (
            "#!/usr/bin/env bash\n"
            "printf 'ELF 64-bit LSB pie executable, ARM aarch64, "
            "dynamically linked\\n'\n"
        )
        for name, body in (("ssh", ssh), ("colcon", colcon), ("file", file_cmd)):
            path = self.bin / name
            path.write_text(body)
            path.chmod(0o755)

    def _make_bags(self) -> None:
        (self.home / "bags" / "V2_01_easy").mkdir(parents=True)

    def _snapshot(self) -> Path:
        snap = self.output / "binaries" / "current_head_vins_node"
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text("fake aarch64 binary\n")
        return snap

    def _run(self, *args: str, env: dict | None = None, timeout: float = 60):
        run_env = self.env.copy()
        if env:
            run_env.update(env)
        return subprocess.run(
            ["bash", str(WRAPPER), *args],
            env=run_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    def test_help_documents_config_and_lock(self) -> None:
        proc = self._run("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--config PATH", proc.stdout)
        self.assertIn("--parameter NAME:=VALUE", proc.stdout)
        self.assertIn("flock", proc.stdout)

    def test_build_accepts_submodule_gitdir_file(self) -> None:
        proc = self._run("build", "--reuse-workspace-build")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

        prov = (self.output / "evidence" / "build_provenance.txt").read_text()
        self.assertRegex(prov, r"(?m)^head=[0-9a-f]{40}$")
        self.assertIn("source_dirty=no", prov)
        self.assertTrue(
            (self.output / "binaries" / "current_head_vins_node").is_file()
        )

        command = (self.output / "evidence" / "build_command.txt").read_text()
        self.assertIn("--packages-up-to vins", command)
        self.assertIn("--packages-up-to vins", (self.logs / "colcon.log").read_text())
        self.assertNotIn("--packages-select", command)

    def test_existing_snapshot_is_rejected_before_compiling(self) -> None:
        self._snapshot()
        proc = self._run("build", "--reuse-workspace-build")
        self.assertEqual(proc.returncode, 6, proc.stdout + proc.stderr)
        self.assertIn("snapshot already exists", proc.stdout + proc.stderr)
        self.assertFalse((self.logs / "colcon.log").exists())

    def test_snapshot_argument_does_not_self_match_process_guard(self) -> None:
        self._snapshot()
        proc = self._run(
            "benchmark",
            "--reuse-workspace-build",
            "--datasets",
            "V2_01_easy",
            "--label",
            "self-match",
            "--snapshot-name",
            "current_head_vins_node",
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        combined = proc.stdout + proc.stderr
        self.assertNotIn("another VINS or bag-play process is running", combined)
        self.assertNotIn("conflicting VINS/build process detected", combined)
        argv = json.loads((self.logs / "bench.log").read_text().splitlines()[0])
        self.assertNotIn("--parameter", argv)

    def test_real_snapshot_process_is_detected(self) -> None:
        snap = self._snapshot()
        # argv[0] is the snapshot path, as it would be when the benchmark
        # binary is running; the executable is irrelevant to the detector.
        proc = subprocess.Popen([str(snap), "5"], executable="/bin/sleep")
        try:
            time.sleep(0.2)
            run = self._run(
                "benchmark",
                "--reuse-workspace-build",
                "--datasets",
                "V2_01_easy",
                "--label",
                "detected",
            )
            self.assertEqual(run.returncode, 3, run.stdout + run.stderr)
            self.assertIn(
                "conflicting VINS/build process detected", run.stdout + run.stderr
            )
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_concurrent_step_is_rejected_by_shared_lock(self) -> None:
        started = self.tmp / "colcon.started"
        env = {
            "FAKE_COLCON_SLEEP": "2",
            "FAKE_COLCON_STARTED": str(started),
        }
        run_env = self.env.copy()
        run_env.update(env)
        first = subprocess.Popen(
            ["bash", str(WRAPPER), "build", "--reuse-workspace-build"],
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while not started.exists() and first.poll() is None:
                if time.monotonic() > deadline:
                    break
                time.sleep(0.05)
            self.assertTrue(started.exists(), "first build never reached colcon")
            second = self._run(
                "benchmark",
                "--reuse-workspace-build",
                "--datasets",
                "V2_01_easy",
                "--label",
                "locked-out",
            )
            self.assertEqual(second.returncode, 3, second.stdout + second.stderr)
            self.assertIn(
                "another VINS remote-arm step is using",
                second.stdout + second.stderr,
            )
        finally:
            out, err = first.communicate(timeout=60)
        self.assertEqual(first.returncode, 0, out + err)

    def test_benchmark_config_is_resolved_and_recorded(self) -> None:
        self._snapshot()
        config = self.output / "configs" / "max120.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("/**:\n  ros__parameters:\n    max_cnt: 120\n")

        proc = self._run(
            "benchmark",
            "--reuse-workspace-build",
            "--datasets",
            "V2_01_easy",
            "--label",
            "max120",
            "--config",
            str(config),
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

        prov = (self.output / "evidence" / "benchmark_provenance.txt").read_text()
        self.assertIn(f"config={config}", prov)
        self.assertIn("config_sha256=", prov)
        argv = json.loads((self.logs / "bench.log").read_text().splitlines()[0])
        self.assertEqual(argv[argv.index("--config") + 1], str(config))

    def test_cv2_uses_same_snapshot_and_default_config_without_build(self) -> None:
        snapshot = self._snapshot()
        config = (self.vins / "src/vins_estimator/config/euroc"
                  / "euroc_mono_imu_config.yaml")
        original_config = config.read_bytes()
        original_snapshot = snapshot.read_bytes()
        proc = self._run(
            "benchmark", "--reuse-workspace-build",
            "--datasets", "V2_01_easy", "--label", "arm_cv2_easy",
            "--domain-id", "173", "--parameter", "opencv_threads:=2",
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        argv = json.loads((self.logs / "bench.log").read_text().splitlines()[0])
        self.assertEqual(argv[argv.index("--parameter") + 1], "opencv_threads:=2")
        self.assertEqual(argv.count("--parameter"), 1)
        self.assertEqual(argv[argv.index("--binary") + 1], str(snapshot))
        self.assertEqual(argv[argv.index("--config") + 1], str(config))
        self.assertEqual(argv[argv.index("--label") + 1], "arm_cv2_easy")
        self.assertEqual(argv[argv.index("--domain-id") + 1], "173")
        self.assertEqual(argv[argv.index("--datasets") + 1], "V2_01_easy")
        self.assertEqual(argv[argv.index("--rate") + 1], "1.0")
        self.assertEqual(argv[argv.index("--repeats") + 1], "1")
        self.assertNotIn("--duration", argv)
        self.assertEqual(config.read_bytes(), original_config)
        self.assertEqual(snapshot.read_bytes(), original_snapshot)
        self.assertFalse((self.logs / "colcon.log").exists())
        provenance = (self.output / "evidence" / "benchmark_provenance.txt").read_text()
        self.assertIn("--parameter opencv_threads:=2", provenance)

    def test_repeated_parameters_preserve_argv_without_shell_expansion(self) -> None:
        self._snapshot()
        marker = self.tmp / "must-not-exist"
        parameters = ["opencv_threads:=2", f"test_string:=a b;$(touch {marker})"]
        args = ["benchmark", "--reuse-workspace-build", "--datasets", "V2_01_easy",
                "--label", "quoted-parameters"]
        for parameter in parameters:
            args += ["--parameter", parameter]
        proc = self._run(*args)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        argv = json.loads((self.logs / "bench.log").read_text().splitlines()[0])
        actual = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--parameter"]
        self.assertEqual(actual, parameters)
        self.assertFalse(marker.exists())

    def test_invalid_parameter_is_rejected_before_ssh(self) -> None:
        for suffix in [[], ["opencv_threads=2"], [":=2"]]:
            with self.subTest(suffix=suffix):
                proc = self._run("benchmark", "--parameter", *suffix)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn("--parameter requires NAME:=VALUE", proc.stderr)
                self.assertFalse((self.output / "evidence").exists())

    def test_missing_config_fails_before_benchmark(self) -> None:
        self._snapshot()
        proc = self._run(
            "benchmark",
            "--reuse-workspace-build",
            "--datasets",
            "V2_01_easy",
            "--label",
            "missing-config",
            "--config",
            "configs/does-not-exist.yaml",
        )
        self.assertEqual(proc.returncode, 6, proc.stdout + proc.stderr)
        self.assertIn("config missing", proc.stdout + proc.stderr)
        self.assertFalse((self.logs / "bench.log").exists())


if __name__ == "__main__":
    unittest.main()
