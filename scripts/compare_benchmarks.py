#!/usr/bin/env python3
"""Compare repeated runs from benchmark_euroc.py; report, don't assume a speedup."""
import argparse
import json
from pathlib import Path
from statistics import mean

METRICS = ("cpu_seconds", "latency_p95_ms", "latency_p99_ms", "peak_rss_mib", "ate_rmse_m")


def aggregate(path):
    rows = json.loads(path.read_text())
    if not rows:
        raise ValueError(f"No measurements in {path}")
    groups = {}
    for row in rows:
        groups.setdefault(row["dataset"], []).append(row)
    result = {}
    for dataset, runs in groups.items():
        metrics = {}
        for name in METRICS:
            values = [run[name] for run in runs]
            if any(value is None for value in values):
                raise ValueError(f"Missing {name} for {dataset} in {path}")
            metrics[name] = mean(values)
        result[dataset] = {
            "runs": len(runs), "mean": metrics,
            "range": {name: [min(run[name] for run in runs), max(run[name] for run in runs)] for name in METRICS},
            "output_poses": [run["output_poses"] for run in runs],
            "received_images": [run["received_images"] for run in runs],
            "matched_poses": [run["matched_poses"] for run in runs],
            "max_output_gap_s": max(run["max_output_gap_s"] for run in runs),
            "failure_reboots": sum(run["failure_reboots"] for run in runs),
            "latency_misses": sum(run["latency_misses"] for run in runs),
            "rate": sorted(set(run["rate"] for run in runs)),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline summary.json")
    parser.add_argument("optimized", type=Path, help="Optimized summary.json")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    baseline, optimized = aggregate(args.baseline), aggregate(args.optimized)
    if baseline.keys() != optimized.keys():
        parser.error("Dataset sets differ; compare the same sequences")
    comparisons = {}
    print("| Dataset | CPU seconds (before → after) | CPU change | P95 ms | Peak RSS MiB | ATE RMSE m |")
    print("|---|---:|---:|---:|---:|---:|")
    for dataset in baseline:
        b, o = baseline[dataset], optimized[dataset]
        if b["runs"] != o["runs"] or b["rate"] != o["rate"]:
            parser.error(f"Repeat count or playback rate differs for {dataset}")
        change = {key: 100 * (o["mean"][key] / b["mean"][key] - 1) if b["mean"][key] else None
                  for key in METRICS}
        comparisons[dataset] = {"baseline": b, "optimized": o, "change_percent": change}
        cells = [f'{b["mean"][key]:.3f} → {o["mean"][key]:.3f}' for key in METRICS]
        print(f"| {dataset} | {cells[0]} | {change['cpu_seconds']:+.1f}% | {cells[1]} | {cells[3]} | {cells[4]} |")
        if sorted(b["output_poses"]) != sorted(o["output_poses"]) or o["failure_reboots"] or o["latency_misses"]:
            print(f"\nWARNING {dataset}: check output coverage, failures and unmatched latency samples.\n")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(comparisons, indent=2) + "\n")


if __name__ == "__main__":
    main()
