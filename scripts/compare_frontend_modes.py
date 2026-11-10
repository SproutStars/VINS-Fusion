#!/usr/bin/env python3
"""Read-only comparison of gftt / fast_grid / gftt_pyramid ARM runs.

Reads <root>/<label>/<dataset>_r<repeat>/ with metrics.json, trajectory.csv,
estimator.log and run_manifest.json, plus <root>/groundtruth/<dataset>.csv.
Never replays ROS or rewrites raw run files. Every repeat stays separate.

Common ATE uses only the timestamp intersection of the labels passed in this
invocation, the same truth, and benchmark_euroc.trajectory_metrics (rigid
alignment, no scale fitting). Do not reuse a 3-label intersection as a 2-label
comparison.
"""
import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load_module("frontend_benchmark_euroc", "benchmark_euroc.py")
summary = load_module(
    "frontend_summarize_balanced", "summarize_balanced_optimization.py"
)


def num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def rat(value, base):
    value, base = num(value), num(base)
    return None if value is None or base in (None, 0) else value / base


def dif(value, base):
    value, base = num(value), num(base)
    return None if value is None or base is None else value - base


def label_path(root, label):
    for candidate in (root / label, root / "benchmark" / label):
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"missing label directory: {root / label}")


def dataset_truth(root, dataset, cache, issues):
    if dataset not in cache:
        path = root / "groundtruth" / f"{dataset}.csv"
        try:
            cache[dataset] = (
                np.loadtxt(path, delimiter=",", skiprows=1) if path.is_file() else None
            )
        except Exception as exc:
            cache[dataset] = None
            issues.append(f"groundtruth {dataset}.csv: {type(exc).__name__}: {exc}")
    if cache[dataset] is None:
        issues.append(f"groundtruth {dataset}.csv: missing")
    return cache[dataset]


def run_metrics(run, issues):
    metrics = run["metrics"]
    try:
        tracking = summary.tracker_summary(run["dir"] / "estimator.log")
        source = "estimator.log"
    except Exception as exc:
        tracking, source = {}, "metrics.json fallback"
        issues.append(f"estimator.log: {type(exc).__name__}: {exc}")
    restarts = tracking.get("restart_diagnostics")
    restarts = restarts if isinstance(restarts, dict) else {}

    def restart_count(key):
        value = restarts.get(key)
        return value if num(value) is not None else num(metrics.get(key))

    received = num(metrics.get("received_images"))
    cpu = num(metrics.get("cpu_seconds"))
    matched = num(metrics.get("matched_poses"))
    absolute = {
        key: num(metrics.get(key))
        for key in (
            "ate_rmse_m", "matched_poses", "output_poses", "received_images",
            "max_output_gap_s", "trajectory_duration_s", "cpu_seconds",
            "peak_rss_mib", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        )
    }
    absolute["pose_per_image"] = rat(matched, received)
    absolute["cpu_ms_per_received_image"] = (
        1000 * cpu / received if cpu is not None and received else None
    )
    return {"absolute": absolute, "tracking": {
        "sum_frames": num(tracking.get("sum_frames")),
        "total_frames": num(tracking.get("total_frames")),
        "frames_consistent": tracking.get("frames_consistent"),
        "overflow_events": num(tracking.get("overflow_events_cumulative")),
        "overflow_dropped_pairs": num(tracking.get("overflow_dropped_pairs_cumulative")),
        "restart_requests": restart_count("restart_requests"),
        "reset_entry_events": restart_count("reset_entry_events"),
        "source": source,
    }}


def trajectory_map(run, issues):
    path = run["dir"] / "trajectory.csv"
    if not path.is_file():
        issues.append("trajectory.csv: missing")
        return {}
    try:
        poses = summary.load_trajectory(path)
    except Exception as exc:
        issues.append(f"trajectory.csv: {type(exc).__name__}: {exc}")
        return {}
    if not poses:
        issues.append("trajectory.csv: empty (0 poses)")
    return poses


def common_ate(maps, truth, dataset):
    stamps = (
        sorted(set.intersection(*(set(poses) for poses in maps.values())))
        if maps and all(maps.values())
        else []
    )
    result = {
        "intersection_labels": sorted(maps), "intersection_mode": f"{len(maps)}-label",
        "common_timestamps": len(stamps), "labels": {}, "comparable": False, "issues": [],
    }
    if truth is None:
        result["issues"].append(f"common ATE: groundtruth missing for {dataset}")
    if maps and not all(maps.values()):
        result["issues"].append("common ATE: at least one trajectory is empty")
    for label, poses in maps.items():
        if not stamps or truth is None:
            result["labels"][label] = {"ate_rmse_m": None, "matched_poses": None}
            continue
        measured = benchmark.trajectory_metrics(
            np.array([[stamp, *poses[stamp]] for stamp in stamps]), truth
        )
        result["labels"][label] = {
            "ate_rmse_m": measured.get("ate_rmse_m"),
            "matched_poses": measured.get("matched_poses"),
        }
    result["comparable"] = bool(
        stamps and truth is not None
        and all(entry["ate_rmse_m"] is not None for entry in result["labels"].values())
    )
    return result


def normalized(run, labels, baseline):
    base = run["labels"].get(baseline)
    common = run["common_timestamp_ate"]["labels"]
    result = {}
    for label in labels:
        entry = run["labels"].get(label)
        if entry is None:
            continue
        if base is None:
            result[label] = None
            continue
        a, b = entry["absolute"], base["absolute"]
        result[label] = {
            "common_ate_ratio": rat(
                common.get(label, {}).get("ate_rmse_m"),
                common.get(baseline, {}).get("ate_rmse_m"),
            ),
            "cpu_per_image_ratio": rat(
                a["cpu_ms_per_received_image"], b["cpu_ms_per_received_image"]
            ),
            "temporal_coverage_delta": dif(a["temporal_coverage"], b["temporal_coverage"]),
            "max_output_gap_delta_s": dif(a["max_output_gap_s"], b["max_output_gap_s"]),
            "pose_per_image_delta": dif(a["pose_per_image"], b["pose_per_image"]),
        }
    return result


def recommendation(labels, runs, baseline, accuracy_tolerance, speed_tolerance,
                    max_coverage_drop):
    """Manual research screen, not an automatic best-mode selection."""
    result = {}
    for label in labels:
        rows = []
        for key, run in runs.items():
            entry = run["labels"].get(label)
            base = run["labels"].get(baseline)
            if entry is None or base is None:
                rows.append({"run": key, "status": "missing run"})
                continue
            common = run["common_timestamp_ate"]["labels"]
            ate = common.get(label, {}).get("ate_rmse_m")
            base_ate = common.get(baseline, {}).get("ate_rmse_m")
            a, b = entry["absolute"], base["absolute"]
            row = {
                "run": key, "common_ate_ratio": rat(ate, base_ate),
                "cpu_per_image_ratio": rat(
                    a["cpu_ms_per_received_image"], b["cpu_ms_per_received_image"]
                ),
                "temporal_coverage_delta": dif(a["temporal_coverage"], b["temporal_coverage"]),
                "max_output_gap_delta_s": dif(a["max_output_gap_s"], b["max_output_gap_s"]),
            }
            row["accuracy_ok"] = (
                row["common_ate_ratio"] is not None
                and row["common_ate_ratio"] <= accuracy_tolerance
                and (common.get(label, {}).get("matched_poses") or 0) >= 3
            )
            row["coverage_ok"] = (
                row["temporal_coverage_delta"] is not None
                and row["temporal_coverage_delta"] >= -max_coverage_drop
            )
            tracking = entry["tracking"]
            row["restart_clean"] = (
                tracking.get("restart_requests") == 0
                and tracking.get("reset_entry_events") == 0
            )
            rows.append(row)

        comparable = [row for row in rows if row.get("common_ate_ratio") is not None]

        def median(key):
            values = [row[key] for row in comparable if row.get(key) is not None]
            return statistics.median(values) if values else None

        check = {
            "is_baseline": label == baseline, "runs_checked": len(rows),
            "comparable_runs": len(comparable),
            "all_accuracy_within_manual_tolerance": bool(comparable)
            and all(row["accuracy_ok"] for row in comparable),
            "all_temporal_coverage_within_manual_tolerance": bool(comparable)
            and all(row["coverage_ok"] for row in comparable),
            "all_restart_clean": bool(comparable)
            and all(row["restart_clean"] for row in comparable),
            "median_common_ate_ratio": median("common_ate_ratio"),
            "median_cpu_per_image_ratio": median("cpu_per_image_ratio"),
            "runs": rows,
            "caveat": "restart_clean is continuity evidence only; zero restarts do not prove ATE reliability.",
        }
        check["passes_manual_research_screen"] = bool(
            label != baseline and comparable
            and check["all_accuracy_within_manual_tolerance"]
            and check["all_temporal_coverage_within_manual_tolerance"]
            and check["all_restart_clean"]
            and check["median_cpu_per_image_ratio"] is not None
            and check["median_cpu_per_image_ratio"] <= speed_tolerance
        )
        result[label] = check

    return {
        "manual_research_filters": {
            "baseline_label": baseline, "accuracy_tolerance_ratio": accuracy_tolerance,
            "speed_tolerance_ratio": speed_tolerance,
            "max_temporal_coverage_drop": max_coverage_drop,
        },
        "labels": result,
        "manual_research_shortlist": [
            label for label in labels if result[label]["passes_manual_research_screen"]
        ],
        "note": (
            "Manual research screening only, not user acceptance and not an automatic "
            "best. Baseline-relative ATE can pass when both modes diverge; absolute "
            "common ATE, temporal coverage and gap remain required."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--accuracy-tolerance", type=float, default=1.10)
    parser.add_argument("--speed-tolerance", type=float, default=0.95)
    parser.add_argument("--max-coverage-drop", type=float, default=0.02)
    args = parser.parse_args(argv)

    labels = list(dict.fromkeys(args.labels))
    if len(labels) < 2:
        parser.error("--labels needs at least two distinct labels")
    baseline = args.baseline or labels[0]
    if (baseline not in labels or args.accuracy_tolerance < 1
            or args.speed_tolerance <= 0 or args.max_coverage_drop < 0):
        parser.error("invalid baseline or tolerance values")

    root = args.root.resolve()
    run_sets = {label: summary.load_runs(label_path(root, label)) for label in labels}
    if any(not found for found in run_sets.values()):
        parser.error("a label has no */metrics.json runs")
    keys = set().union(*(set(found) for found in run_sets.values()))
    if args.datasets:
        keys = {key for key in keys if key[0] in set(args.datasets)}
    if not keys:
        parser.error("no runs matched the requested datasets")

    truth_cache, runs, all_issues = {}, {}, []
    for dataset, repeat in sorted(keys):
        run = {"dataset": dataset, "repeat": repeat, "labels": {}, "missing_labels": [],
               "normalized_vs_baseline": {}, "issues": []}
        maps = {}
        for label in labels:
            source = run_sets[label].get((dataset, repeat))
            if source is None:
                run["missing_labels"].append(label)
                continue
            run["labels"][label] = run_metrics(source, run["issues"])
            maps[label] = trajectory_map(source, run["issues"])
            if not (source["dir"] / "run_manifest.json").is_file():
                run["issues"].append("run_manifest.json: missing")

        truth = dataset_truth(root, dataset, truth_cache, run["issues"])
        duration = (
            float(truth[-1, 0] - truth[0, 0])
            if truth is not None and len(truth) > 1 else None
        )
        for entry in run["labels"].values():
            entry["absolute"]["bag_effective_duration_s"] = duration
            entry["absolute"]["temporal_coverage"] = rat(
                entry["absolute"]["trajectory_duration_s"], duration
            )
        common = run["common_timestamp_ate"] = common_ate(maps, truth, dataset)
        common["comparable"] &= not run["missing_labels"]
        run["issues"].extend(common.pop("issues"))
        run["normalized_vs_baseline"] = normalized(run, labels, baseline)
        all_issues.extend(f"{dataset}_r{repeat}: {issue}" for issue in run["issues"])
        runs[f"{dataset}_r{repeat}"] = run

    datasets = {}
    for dataset, repeat in sorted(keys):
        datasets.setdefault(dataset, []).append(f"{dataset}_r{repeat}")

    report = {
        "schema": "vins-arm-frontend-comparison/v1", "root": str(root),
        "labels": labels, "baseline_label": baseline,
        "repeats": "kept separate; no dataset-level swallowing",
        "datasets": datasets, "runs": runs,
        "recommendation": recommendation(
            labels, runs, baseline, args.accuracy_tolerance,
            args.speed_tolerance, args.max_coverage_drop,
        ),
        "issues": all_issues,
        "notes": [
            "Absolute metrics come from each run's metrics.json.",
            "Common ATE uses only the timestamp intersection of the labels in this "
            "invocation; do not reuse a 3-label intersection for a 2-label claim.",
            "temporal_coverage = trajectory_duration_s / bag_effective_duration_s; "
            "pose_per_image is not coverage (20 Hz images vs ~10 Hz backend).",
            "absolute ATE must be read with temporal_coverage and max_output_gap_s.",
            "fast_grid must be the current hybrid GFTT-fallback source; old FAST-only "
            "data is not comparable.",
            "Zero restart requests are not evidence of accuracy; inspect per-run ATE, "
            "temporal coverage, and gap.",
            "Benchmark runs are serial only; this tool is read-only offline aggregation.",
        ],
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
