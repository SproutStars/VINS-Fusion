"""Focused tests for scripts/probe_arm_acceleration.sh."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "probe_arm_acceleration.sh"


class ProbeArmAccelerationTest(unittest.TestCase):
    def run_probe(self, *args: str, env: dict | None = None):
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            text=True,
            capture_output=True,
            env=run_env,
            timeout=30,
        )

    def test_shell_syntax(self) -> None:
        proc = subprocess.run(["bash", "-n", str(SCRIPT)], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_help(self) -> None:
        proc = self.run_probe("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--host HOST", proc.stdout)

    def test_unknown_argument_fails(self) -> None:
        proc = self.run_probe("--bogus")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("unknown argument", proc.stderr)

    def test_host_requires_exactly_one_value(self) -> None:
        proc = self.run_probe("--host")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--host requires exactly one value", proc.stderr)

    def test_ssh_is_key_only_and_has_no_password(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="probe-arm-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        log = tmp / "ssh.log"
        fake_ssh = tmp / "ssh"
        fake_ssh.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' \"$*\" >> \"${FAKE_SSH_LOG:?}\"\n"
            "cat >/dev/null\n"
        )
        fake_ssh.chmod(0o755)
        env = {
            "PATH": str(tmp) + os.pathsep + os.environ.get("PATH", ""),
            "FAKE_SSH_LOG": str(log),
        }
        proc = self.run_probe("--host", "pinydrone@192.168.5.62", env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        args = log.read_text()
        self.assertIn("BatchMode=yes", args)
        self.assertIn("pinydrone@192.168.5.62", args)
        self.assertNotIn("sshpass", SCRIPT.read_text())
        self.assertNotIn("PasswordAuthentication", SCRIPT.read_text())


if __name__ == "__main__":
    unittest.main()
