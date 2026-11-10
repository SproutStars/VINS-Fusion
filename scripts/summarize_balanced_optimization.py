#!/usr/bin/env python3
"""Compare two VINS benchmark label dirs without replaying ROS.

Per run: original metrics, cumulative overflow from the final tracker block,
frame-weighted *_ms means and maxima, frame-weighted tracked/detected feature
means, exact-timestamp trajectory delta, and common-timestamp ATE via
benchmark_euroc.trajectory_metrics. Runs are matched by (dataset, repeat);
repeats are reported separately and never aggregated. Omit --output to print.
"""
import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("benchmark_euroc", HERE / "benchmark_euroc.py")
benchmark_euroc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_euroc)
STATS_RE = re.compile(r"Tracker (?:periodic|final) stats: (.*)")
PAIR_RE = re.compile(r"(\w+)=([0-9.]+)/([0-9.]+)")
INT_RE = re.compile(r"(frames|total_frames|overflow_events|overflow_dropped_pairs)=(\d+)")
RUN_RE = re.compile(r"^(?P<dataset>.+)_r(?P<repeat>\d+)$")


def load_runs(label_dir: Path) -> dict:
    runs = {}
    for path in sorted(label_dir.glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        match = RUN_RE.match(path.parent.name)
        dataset = match.group("dataset") if match else metrics.get("dataset") or path.parent.name
        repeat = int(match.group("repeat")) if match else int(metrics.get("repeat", 1))
        key = (dataset, repeat)
        if key in runs:
            raise ValueError(f"duplicate run key {key} in {label_dir}")
        runs[key] = {"dir": path.parent, "metrics": metrics, "dataset": dataset, "repeat": repeat}
    return runs



def restart_diagnostics(log_path: Path) -> dict:
    parser = getattr(benchmark_euroc, "parse_restart_diagnostics", None)
    if parser is None:
        return {"available": False, "error": "benchmark_euroc.parse_restart_diagnostics missing"}
    try:
        return {"available": True, **parser(log_path.read_text(errors="replace"))}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

def tracker_summary(log_path: Path) -> dict:
    blocks, overflow = [], {}
    for line in log_path.read_text(errors="replace").splitlines():
        match = STATS_RE.search(line)
        if not match:
            continue
        body = match.group(1)
        ints = {k: int(v) for k, v in INT_RE.findall(body)}
        if "frames" not in ints or "total_frames" not in ints:
            continue
        pairs = {k: (float(a), float(b)) for k, a, b in PAIR_RE.findall(body)}
        blocks.append((ints["frames"], ints["total_frames"], pairs))
        overflow.update({k: ints[k] for k in ("overflow_events", "overflow_dropped_pairs") if k in ints})
    weighted_ms, max_ms = {}, {}
    features = {"tracked_mean": None, "detected_mean": None}
    for key in sorted({k for _, _, pairs in blocks for k in pairs}):
        total = sum(frames for frames, _, pairs in blocks if key in pairs)
        first = sum(frames * pairs[key][0] for frames, _, pairs in blocks if key in pairs) / total if total else None
        if key.endswith("_ms"):
            weighted_ms[key] = first
            max_ms[key] = max(pairs[key][1] for _, _, pairs in blocks if key in pairs)
        elif key == "features_avg":
            second = sum(frames * pairs[key][1] for frames, _, pairs in blocks if key in pairs) / total if total else None
            features = {"tracked_mean": first, "detected_mean": second}
    sum_frames = sum(frames for frames, _, _ in blocks)
    total_frames = max((total for _, total, _ in blocks), default=0)
    return {
        "stats_blocks": len(blocks), "sum_frames": sum_frames, "total_frames": total_frames,
        "frames_consistent": bool(blocks) and sum_frames == total_frames,
        "overflow_events_cumulative": overflow.get("overflow_events"),
        "overflow_dropped_pairs_cumulative": overflow.get("overflow_dropped_pairs"),
        "overflow_source": "last Tracker periodic/final stats block",
        "weighted_mean_ms": weighted_ms, "max_ms": max_ms, "features": features,
        "restart_diagnostics": restart_diagnostics(log_path),
    }


def load_trajectory(path: Path) -> dict:
    with path.open(newline="") as handle:
        return {float(r["t"]): (float(r["x"]), float(r["y"]), float(r["z"])) for r in csv.DictReader(handle)}


def change(base, candidate):
    if base is None or candidate is None:
        return None
    delta = candidate - base
    return {"baseline": base, "candidate": candidate, "delta": delta, "pct": 100.0 * delta / base if base else None}


def compare_dataset(base: dict, candidate: dict, truth_path: Path) -> dict:
    base_traj = load_trajectory(base["dir"] / "trajectory.csv")
    cand_traj = load_trajectory(candidate["dir"] / "trajectory.csv")
    common = sorted(set(base_traj) & set(cand_traj))
    delta = np.array([np.subtract(base_traj[t], cand_traj[t]) for t in common]) if common else None
    result = {
        "common_timestamps": len(common),
        "max_abs_position_delta_m": float(np.max(np.linalg.norm(delta, axis=1))) if common else None,
        "max_abs_axis_delta_m": float(np.max(np.abs(delta))) if common else None,
        "baseline_ate_common": None, "candidate_ate_common": None,
    }
    if common and truth_path.is_file():
        truth = np.loadtxt(truth_path, delimiter=",", skiprows=1)
        result["baseline_ate_common"] = benchmark_euroc.trajectory_metrics(
            np.array([[t, *base_traj[t]] for t in common]), truth)
        result["candidate_ate_common"] = benchmark_euroc.trajectory_metrics(
            np.array([[t, *cand_traj[t]] for t in common]), truth)
    keys = ("cpu_seconds", "average_cpu_percent_one_core", "peak_rss_mib", "latency_p50_ms",
            "latency_p95_ms", "latency_p99_ms", "wall_s", "output_poses", "matched_poses",
            "ate_rmse_m", "max_output_gap_s")
    result["changes"] = {k: change(base["metrics"].get(k), candidate["metrics"].get(k)) for k in keys}
    return result


def run_name(key: tuple[str, int]) -> str:
    return f"{key[0]}_r{key[1]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="write JSON here; omit to print stdout")
    args = parser.parse_args()
    if not args.baseline.is_dir() or not args.candidate.is_dir():
        parser.error("--baseline and --candidate must be existing label directories")
    base_runs, cand_runs = load_runs(args.baseline), load_runs(args.candidate)
    if not base_runs or not cand_runs:
        parser.error("no */metrics.json runs found in one or both label directories")
    truth_dir = args.baseline.parent / "groundtruth"
    runs = {}
    for key in sorted(set(base_runs) & set(cand_runs)):
        base, cand = base_runs[key], cand_runs[key]
        base["tracker"] = tracker_summary(base["dir"] / "estimator.log")
        cand["tracker"] = tracker_summary(cand["dir"] / "estimator.log")
        runs[run_name(key)] = {
            "dataset": key[0], "repeat": key[1],
            "baseline": {**base, "dir": str(base["dir"])},
            "candidate": {**cand, "dir": str(cand["dir"])},
            "comparison": compare_dataset(base, cand, truth_dir / f"{key[0]}.csv"),
        }
    datasets = sorted({key[0] for key in set(base_runs) & set(cand_runs)})
    repeat_counts = {
        dataset: {
            "baseline": sorted(r for d, r in base_runs if d == dataset),
            "candidate": sorted(r for d, r in cand_runs if d == dataset),
        } for dataset in datasets
    }
    single = all(len(v["baseline"]) == 1 and len(v["candidate"]) == 1 for v in repeat_counts.values())
    report = {
        "baseline": str(args.baseline), "candidate": str(args.candidate),
        "single_run_screening": single, "aggregation": "none; each repeat is compared separately",
        "note": "one run per dataset; no repeat statistics" if single else "multiple repeats reported per run; no aggregation",
        "repeat_counts": repeat_counts, "runs": runs,
        "unmatched_baseline": [run_name(k) for k in sorted(set(base_runs) - set(cand_runs))],
        "unmatched_candidate": [run_name(k) for k in sorted(set(cand_runs) - set(base_runs))],
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
