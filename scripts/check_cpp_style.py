#!/usr/bin/env python3
"""Check project C++ sources using the repository's Clang 18 configuration."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".c", ".cc", ".cpp", ".h", ".hpp", ".hxx"}


def project_source(path):
    path = path.resolve()
    return (
        path.is_relative_to(ROOT)
        and "ThirdParty" not in path.relative_to(ROOT).parts
        and path.suffix in EXTENSIONS
    )


def find_tool(name):
    override = os.environ.get(name.upper().replace("-", "_"))
    executable = override or shutil.which(name + "-18") or shutil.which(name)
    if not executable:
        raise RuntimeError(f"Install {name}-18 or set {name.upper().replace('-', '_')}")
    return executable


def compilation_units(build_dir):
    databases = sorted(build_dir.glob("*/compile_commands.json"))
    if not databases:
        databases = [build_dir / "compile_commands.json"]
    units = {}
    for database in databases:
        if not database.exists():
            continue
        for entry in json.loads(database.read_text()):
            path = Path(entry["file"])
            if not path.is_absolute():
                path = Path(entry["directory"]) / path
            if project_source(path):
                units[path.resolve()] = database.parent
    if not units:
        raise RuntimeError(
            "No project compilation database found. Run colcon build with "
            "--cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTING=ON"
        )
    return sorted(units.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("format", "tidy"))
    parser.add_argument("--fix", action="store_true", help="Apply formatting (format mode only)")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.fix and args.mode != "format":
        parser.error("--fix is only supported for format; review tidy fixes across all translation units")

    try:
        if args.mode == "format":
            tool = find_tool("clang-format")
            files = sorted(
                p for folder in (ROOT / "src", ROOT / "scripts")
                for p in folder.rglob("*") if project_source(p)
            )
            commands = [
                [tool, "--style=file", *(["-i"] if args.fix else ["--dry-run", "--Werror"]), str(p)]
                for p in files
            ]
        else:
            tool = find_tool("clang-tidy")
            commands = [
                [tool, "-p", str(database), "--warnings-as-errors=*", str(path)]
                for path, database in compilation_units(args.build_dir.resolve())
            ]

        def run(command):
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            return command[-1], result

        failures = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for filename, result in pool.map(run, commands):
                if result.returncode:
                    failures += 1
                    print(f"FAIL {Path(filename).relative_to(ROOT)}", file=sys.stderr)
                    print(result.stdout + result.stderr, file=sys.stderr, end="")
        print(f"{args.mode}: {len(commands)} files checked, {failures} failed")
        return bool(failures)
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
