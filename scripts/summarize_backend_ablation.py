#!/usr/bin/env python3
"""Minimal offline summary for output/qr_imu_budget_20261115.

Reads runs.jsonl, each run's audit.json, replay manifest, trajectory.csv and
groundtruth CSV. Recomputes sensor-origin ATE by calling the frozen
benchmark_euroc.trajectory_metrics(estimated, truth, T_BS). Writes a small JSON
and Markdown report. Never runs replay/build and never edits runner/product.
"""
import argparse, hashlib, importlib.util, json, re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "output/qr_imu_budget_20261115"
LEDGER = TASK / "evidence/results/runs.jsonl"
MICRO = TASK / "evidence/microbench_local.json"
BENCH = ROOT / "scripts/benchmark_euroc.py"
JSON_OUT = ROOT / "docs/BACKEND_ABLATION_RESULTS_20261115.json"
MD_OUT = ROOT / "docs/BACKEND_ABLATION_20261115.md"
ATE_TOL = 0.05


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_euroc", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_outcome(run, audit, run_dir):
    if run["status"] == "success":
        run.update({"outcome_class": "success", "partial_excluded": False,
                    "excluded_from_comparisons": False})
        return
    log_path = run_dir / "benchmark.log"
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    exit_code = audit.get("exit_code")
    interrupted = exit_code in (-2, 130, 143) or "KeyboardInterrupt" in text or "SIGINT" in text
    run.update({
        "outcome_class": "not_algorithm_failure" if interrupted else "failure_unclassified",
        "outcome_reason": (f"interrupted; exit_code={exit_code}; KeyboardInterrupt/SIGINT evidence; "
                           "partial run excluded from metrics and comparisons") if interrupted else
                          "failed; partial run excluded from metrics and comparisons",
        "partial_excluded": True,
        "excluded_from_comparisons": True,
    })


def read_rows(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def actual_output_gap(estimated):
    t = np.unique(np.asarray(estimated, dtype=float)[:, 0])
    return float(np.max(np.diff(t))) if len(t) >= 2 else None


def truth_matching(metrics, truth_cache):
    matched = metrics.get("matched_poses")
    output = metrics.get("output_poses")
    unmatched = output - matched if isinstance(matched, int) and isinstance(output, int) else None
    fraction = matched / output if isinstance(matched, int) and isinstance(output, int) and output else None
    cache = truth_cache or {}
    return {
        "matched_poses": matched,
        "output_poses": output,
        "unmatched_output_poses": unmatched,
        "matched_fraction_of_output": fraction,
        "truth_bag_coverage_fraction": cache.get("bag_coverage_fraction"),
        "truth_rows": cache.get("rows_independently_counted") or cache.get("rows"),
        "truth_time_range_s": cache.get("time_range_independently_read_s") or cache.get("time_range_s"),
        "note": "matched/output is a matching count, not a dropped-output count; unmatched poses may fall outside the truth-valid window or across truth outages",
    }


def parse_log(path):
    text = path.read_text(errors="replace") if path.is_file() else ""
    used, budget_def, budget_exp = [], 0, 0
    step_lines, budget_lines, skip_lines = 0, 0, 0
    for line in text.splitlines():
        if "[schur-step]" in line:
            step_lines += 1
            m = re.search(r"\bobs=(\d+)", line)
            if m:
                used.append(int(m.group(1)))
            m = re.search(r"\bbudget=(\d+)/(\d+)/(\d+)", line)
            if m:
                budget_lines += 1
                budget_def += int(m.group(2))
                budget_exp += int(m.group(3))
            continue
        m = re.search(r"Schur EKF skipped frame.*?\bused=(\d+).*?\brejected=(\d+)", line)
        if m:
            skip_lines += 1
            b = re.search(r"\bbudget_def=(\d+) budget_exp=(\d+)", line)
            if b:
                budget_def += int(b.group(1))
                budget_exp += int(b.group(2))
    return {
        "usedobs_available": bool(used),
        "usedobs_source": "[schur-step] obs= lines; debug only",
        "usedobs_step_lines": step_lines,
        "usedobs_budget_lines": budget_lines,
        "usedobs_mean": float(np.mean(used)) if used else None,
        "usedobs_max": max(used) if used else None,
        "budgetdefexp_available": bool(budget_lines or budget_def or budget_exp),
        "budgetdefexp_source": "estimator.log [schur-step]/skip lines; partial if throttled",
        "budget_deferred_tracks_logged": budget_def if (step_lines or budget_def or budget_exp) else None,
        "budget_expired_tracks_logged": budget_exp if (step_lines or budget_def or budget_exp) else None,
        "skip_warning_lines": skip_lines,
    }


def collect():
    records = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    runs = []
    for record in records:
        audit_path = Path(record.get("audit") or "")
        run_dir = audit_path.parent
        audit = load_json(audit_path, {}) or {}
        dataset = record.get("dataset") or audit.get("dataset")
        metrics = audit.get("metrics") or record.get("metrics") or {}
        machine = str((audit.get("environment") or {}).get("uname", {}).get("machine", "?"))
        platform = "arm" if ("aarch64" in machine or "arm" in machine) else "local"
        replay = run_dir / "benchmark/replay"
        manifest = load_json(replay / "manifest.json", {}) or {}
        truth_meta = ((manifest.get("truth") or {}).get(dataset) or {})
        extr = truth_meta.get("extrinsic") or {}
        gt = {
            "source": extr.get("source") or (audit.get("input", {}).get("truth", {}).get("sensor_config")),
            "sha256": extr.get("sha256") or (audit.get("input", {}).get("truth", {}).get("sensor_config_sha256")),
            "sensor_type": truth_meta.get("sensor_type"),
            "ate_reference": truth_meta.get("ate_reference"),
            "t_bs_source": "manifest" if extr.get("matrix") else "audit_fallback",
        }
        tbs = extr.get("matrix") or (audit.get("input", {}).get("truth", {}).get("t_bs"))
        run = {
            "run_id": record.get("run_id"), "stage": record.get("stage"),
            "variant": record.get("variant"), "dataset": dataset,
            "session": str(record.get("run_id", "")).split("_", 1)[0],
            "status": record.get("status"), "platform": platform,
            "machine": machine, "rate": audit.get("rate") or metrics.get("rate"),
            "group": f"{platform}|rate={audit.get('rate') or metrics.get('rate')}|gt={gt['source']}|sha={str(gt['sha256'])[:12]}",
            "gt": gt, "binary_sha256": audit.get("binary", {}).get("sha256") or record.get("binary_sha256"),
            "parameters": audit.get("parameters", []), "warnings": record.get("warnings", []),
            "metrics": {k: metrics.get(k) for k in (
                "ate_rmse_m", "ate_median_m", "ate_p95_m", "matched_poses", "output_poses",
                "cpu_seconds", "average_cpu_percent_one_core", "peak_rss_mib",
                "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "max_output_gap_s",
                "received_images", "latency_misses", "restart_requests", "reset_entry_events",
                "failure_reboots", "actual_resets_completed")},
        }
        run["gt_matched_gap_s"] = metrics.get("max_output_gap_s")
        run["truth_matching"] = truth_matching(metrics, audit.get("truth_cache") or {})
        run["actual_output_gap_s"] = None
        classify_outcome(run, audit, run_dir)
        run.update(parse_log(replay / f"{dataset}_r1/estimator.log"))
        bench = load_benchmark()
        traj_path = replay / f"{dataset}_r1/trajectory.csv"
        truth_path = Path((truth_meta.get("cache") or {}).get("path") or "")
        if not truth_path.is_file():
            candidates = sorted((run_dir / "benchmark/groundtruth").glob("*.csv"))
            truth_path = candidates[0] if candidates else Path()
        if run["status"] == "success" and traj_path.is_file() and truth_path.is_file() and tbs:
            est, truth = read_rows(traj_path), read_rows(truth_path)
            recomputed = bench.trajectory_metrics(est, truth, np.asarray(tbs, dtype=float))
            run["ate"] = {"recomputed": recomputed, "original_rmse_m": metrics.get("ate_rmse_m"),
                          "matches_original": metrics.get("ate_rmse_m") is not None and
                          abs(recomputed.get("ate_rmse_m", -1) - metrics["ate_rmse_m"]) < 1e-9}
            run["_traj"], run["_truth"], run["_tbs"] = est, truth, np.asarray(tbs, dtype=float)
            run["actual_output_gap_s"] = actual_output_gap(est)
        else:
            run["ate"] = {"recomputed": None, "original_rmse_m": metrics.get("ate_rmse_m"),
                          "matches_original": None}
        runs.append(run)
    return runs


def common_ate(run_a, run_b, bench):
    if "_traj" not in run_a or "_traj" not in run_b:
        return None
    ka = np.round(run_a["_traj"][:, 0] * 1e9).astype(np.int64)
    kb = np.round(run_b["_traj"][:, 0] * 1e9).astype(np.int64)
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    if len(common) < 3:
        return {"common_timestamps": int(len(common)), "ate_a": None, "ate_b": None}
    ma = bench.trajectory_metrics(run_a["_traj"][ia], run_a["_truth"], run_a["_tbs"])
    mb = bench.trajectory_metrics(run_b["_traj"][ib], run_b["_truth"], run_b["_tbs"])
    return {"common_timestamps": int(len(common)), "ate_a": ma.get("ate_rmse_m"),
            "ate_b": mb.get("ate_rmse_m"), "matched_a": ma.get("matched_poses"),
            "matched_b": mb.get("matched_poses")}


def rel(a, b):
    return None if a is None or b in (None, 0) else 100.0 * (a - b) / b


def compare(runs, bench):
    by_key = defaultdict(list)
    for r in runs:
        if r["status"] == "success" and r["ate"]["recomputed"] and not r.get("excluded_from_comparisons"):
            by_key[(r["dataset"], r["group"])].append(r)
    out = []
    for (dataset, group_key), group in by_key.items():
        baselines = [r for r in group if r["variant"] == "baseline"]
        controls = [r for r in group if r["variant"] == "candidate_control"]
        if baselines and controls:
            b, c = baselines[-1], controls[-1]
            cm = common_ate(b, c, bench)
            cm_ate_b = cm["ate_a"] if cm else None
            cm_ate_c = cm["ate_b"] if cm else None
            out.append({"kind": "baseline_equivalence", "dataset": dataset, "group": group_key,
                        "baseline_run": b["run_id"], "candidate_control_run": c["run_id"],
                        "common_timestamps": (cm or {}).get("common_timestamps"),
                        "baseline_ate_common": cm_ate_b, "candidate_control_ate_common": cm_ate_c,
                        "rel_delta_pct": rel(cm_ate_c, cm_ate_b),
                        "pass": cm_ate_b is not None and cm_ate_c is not None and abs(rel(cm_ate_c, cm_ate_b)) <= 0.1})
        for r in group:
            if r["variant"] in ("baseline", "candidate_control"):
                continue
            same = [c for c in controls if c["session"] == r["session"]]
            default = same[-1] if same else (controls[-1] if controls else None)
            if default is None:
                out.append({"kind": "candidate", "dataset": dataset, "group": group_key,
                            "variant": r["variant"], "run": r["run_id"],
                            "status": "no_candidate_default"})
                continue
            cm = common_ate(r, default, bench)
            a_r = cm["ate_a"] if cm else r["ate"]["recomputed"].get("ate_rmse_m")
            a_d = cm["ate_b"] if cm else default["ate"]["recomputed"].get("ate_rmse_m")
            delta = None if a_r is None or a_d is None else a_r - a_d
            pct = rel(a_r, a_d)
            within = pct is not None and pct <= 100.0 * ATE_TOL
            screening = ("screening_within_5pct" if within else "screening_regression") if pct is not None else "screening_not_applicable"
            out.append({
                "kind": "candidate", "dataset": dataset, "group": group_key,
                "variant": r["variant"], "run": r["run_id"], "stage": r["stage"],
                "default_run": default["run_id"],
                "default_match": "same_session" if same else "fallback_latest_candidate_control",
                "common_timestamps": (cm or {}).get("common_timestamps"),
                "candidate_ate_common": a_r, "default_ate_common": a_d,
                "ate_delta_m": delta, "ate_rel_pct": pct,
                "screening_label": screening,
                "screening_within_5pct": within if pct is not None else None,
                "screening_threshold_pct": 100.0 * ATE_TOL,
                "screening_threshold_provenance": "engineering triage line; not pre-agreed; not user acceptance",
                "user_acceptance": False,
                "cpu_seconds": r["metrics"].get("cpu_seconds"), "default_cpu_seconds": default["metrics"].get("cpu_seconds"),
                "cpu_delta_pct": rel(r["metrics"].get("cpu_seconds"), default["metrics"].get("cpu_seconds")),
                "cpu_comparable_group": r["group"] == default["group"],
                "latency_p95_ms": r["metrics"].get("latency_p95_ms"),
                "rss_mib": r["metrics"].get("peak_rss_mib"),
                "actual_output_gap_s": r.get("actual_output_gap_s"),
                "default_actual_output_gap_s": default.get("actual_output_gap_s"),
                "gt_matched_gap_s": r.get("gt_matched_gap_s"),
                "default_gt_matched_gap_s": default.get("gt_matched_gap_s"),
                "restarts": [r["metrics"].get("restart_requests"), r["metrics"].get("reset_entry_events")],
                "latency_misses": r["metrics"].get("latency_misses"),
                "usedobs_mean": r.get("usedobs_mean"),
                "budget_def": r.get("budget_deferred_tracks_logged"),
                "budget_exp": r.get("budget_expired_tracks_logged"),
            })
    return out


def fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def markdown(runs, comparisons, micro, arm_root=None):
    lines = ["# Backend ablation - results (2026-11-15)", "",
             f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
             "Offline summary only. Sensor-origin ATE is recomputed from `trajectory.csv` + truth CSV + "
             "`T_BS` read from each run's manifest via `benchmark_euroc.trajectory_metrics`. "
             "Common-timestamp ATE uses the intersection of output timestamps for each candidate/default pair. "
             "Groups are separated by platform, rate and GT extrinsic source; CPU is never compared across rates. "
             "ARM inputs are discovered dynamically from `metrics.json` + nearest `manifest.json`. "
             "`actual gap` is the maximum consecutive timestamp gap over all rows of the original `trajectory.csv`; "
             "`GT-matched gap` is the original benchmark `max_output_gap_s` after truth-valid filtering. "
             "`matched/output` is a matching count, not a dropped-output count (truth outages/window can exclude poses).", "",
             "The 5% ATE line was **not pre-agreed** with the user. It is an engineering screening line for "
             "triage only, not user acceptance. `screening_regression` labels one run outside that line and "
             "does **not** assert a global algorithm failure.", "",
             "## Run summary", "",
             "| stage | variant | dataset | status | ATE m | ATE common m | CPU s | CPU % | lat p95 ms | RSS MiB | actual gap s | GT-matched gap s | matched/output | restarts req/reset | drops | usedobs | budget def/exp |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    comp_by_run = {c.get("run"): c for c in comparisons if c.get("kind") == "candidate"}
    for r in runs:
        m = r["metrics"]
        c = comp_by_run.get(r["run_id"], {})
        status = r["status"]
        if r.get("outcome_class") == "not_algorithm_failure":
            status = "failed (not_algorithm_failure; partial excluded)"
        tm = r.get("truth_matching") or {}
        match_text = (f"{tm.get('matched_poses')}/{tm.get('output_poses')}"
                      if tm.get("output_poses") is not None else "-")
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} | {}/{} |".format(
            r["stage"], r["variant"], r["dataset"], status,
            fmt(r["ate"]["recomputed"].get("ate_rmse_m") if r["ate"]["recomputed"] else m.get("ate_rmse_m"), 5),
            fmt(c.get("candidate_ate_common"), 5), fmt(m.get("cpu_seconds"), 2),
            fmt(m.get("average_cpu_percent_one_core"), 1), fmt(m.get("latency_p95_ms"), 2),
            fmt(m.get("peak_rss_mib"), 1), fmt(r.get("actual_output_gap_s"), 3),
            fmt(r.get("gt_matched_gap_s"), 3), match_text,
            fmt(m.get("restart_requests")), fmt(m.get("reset_entry_events")), fmt(m.get("latency_misses")),
            fmt(r.get("usedobs_mean"), 1), fmt(r.get("budget_deferred_tracks_logged")),
            fmt(r.get("budget_expired_tracks_logged"))))
    lines += ["", "## Truth matching / coverage", "",
              "`matched/output` is preserved in full. An unmatched pose is not evidence of missing output: "
              "it may be outside the truth-valid window or inside a truth outage.", "",
              "| dataset | variant | run | matched/output | unmatched | matched fraction | truth rows | truth time range s | bag truth coverage |",
              "|---|---|---|---:|---:|---:|---:|---|---:|"]
    for r in runs:
        tm = r.get("truth_matching") or {}
        if tm.get("output_poses") is None:
            continue
        lines.append(f"| {r['dataset']} | {r['variant']} | {r['run_id']} | "
                     f"{tm.get('matched_poses')}/{tm.get('output_poses')} | {fmt(tm.get('unmatched_output_poses'))} | "
                     f"{fmt(tm.get('matched_fraction_of_output'), 4)} | {fmt(tm.get('truth_rows'))} | "
                     f"{tm.get('truth_time_range_s') or '-'} | {fmt(tm.get('truth_bag_coverage_fraction'), 4)} |")
    lines += ["", "## Candidate vs candidate_default", "",
              "Each row is one run; screening sessions and repeats are never merged. "
              "The label is `screening_within_5pct` or `screening_regression` under the engineering line above.", "",
              "| dataset | variant | run | default | common ts | ATE cand | ATE default | delta % | engineering screen | CPU delta % | same group |",
              "|---|---|---|---|---:|---:|---:|---:|---|---:|---|"]
    for c in comparisons:
        if c.get("kind") != "candidate":
            continue
        lines.append(f"| {c['dataset']} | {c['variant']} | {c['run']} | {c.get('default_run','-')} | "
                     f"{fmt(c.get('common_timestamps'))} | {fmt(c.get('candidate_ate_common'),5)} | "
                     f"{fmt(c.get('default_ate_common'),5)} | {fmt(c.get('ate_rel_pct'),3)} | "
                     f"{c.get('screening_label','screening_not_applicable')} | "
                     f"{fmt(c.get('cpu_delta_pct'),1)} | {c.get('cpu_comparable_group')} |")
    lines += ["", "## Baseline equivalence (baseline vs candidate_control)", "",
              "Candidate comparisons are only attributable after this check passes.", "",
              "| dataset | baseline | candidate_control | common ts | baseline ATE | control ATE | delta % | pass |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for c in comparisons:
        if c.get("kind") == "baseline_equivalence":
            lines.append(f"| {c['dataset']} | {c['baseline_run']} | {c['candidate_control_run']} | "
                         f"{fmt(c.get('common_timestamps'))} | {fmt(c.get('baseline_ate_common'),5)} | "
                         f"{fmt(c.get('candidate_control_ate_common'),5)} | {fmt(c.get('rel_delta_pct'),4)} | "
                         f"{'equivalent' if c.get('pass') else 'not_equivalent'} |")
    excluded = [r for r in runs if r.get("partial_excluded")]
    lines += ["", "## Excluded partial runs", ""]
    if excluded:
        lines += ["| run | dataset | variant | outcome | reason | excluded from comparisons |",
                  "|---|---|---|---|---|---|"]
        for r in excluded:
            lines.append(f"| {r['run_id']} | {r['dataset']} | {r['variant']} | {r.get('outcome_class')} | "
                         f"{r.get('outcome_reason')} | {r.get('excluded_from_comparisons')} |")
    else:
        lines.append("- none")
    lines += ["", "## Microbench (evidence/microbench_local.json)", ""]
    if micro:
        red = micro.get("qr_svd", {}).get("reduce", [])
        val = micro.get("qr_svd", {}).get("reduce_validation", {})
        for item in red:
            lines.append(f"- reduce {item.get('backend')}: {fmt(item.get('us_per_call'),3)} us/call, "
                         f"checksum {item.get('checksum')}")
        if val:
            lines.append(f"- validation: valid {val.get('svd_valid')}/{val.get('qr_valid')}, "
                         f"invalid_mismatch={val.get('invalid_mismatch')}, gate_dof_mismatch={val.get('gate_dof_mismatch')}, "
                         f"finite_violations={val.get('finite_violations')}")
        lines.append("- Scope: production calls only; excludes scene build, state copy, track copy, warmup, validation.")
    else:
        lines.append("- not available")
    lines += ["", "## Research: math / implementation risks", "",
              "1. **QR nullspace projection.** `qr` replaces the historical full-U SVD with a Givens QR of the "
              "point Jacobian. The microbenchmark shows identical checksums and zero validation mismatches, but "
              "that is a local algebra check, not end-to-end trajectory equivalence; rank tolerance, pivot order "
              "and chi2-gate degrees of freedom remain the places where a silent divergence could appear.",
              "2. **Observation budget.** `obs240`/`obs480` select complete tracks before triangulation, deferring "
              "or expiring tracks atomically. The risk is a changed effective information set and selection bias: "
              "deferred tracks are retried, margin/oversized tracks are expired, and budget-selected observations "
              "remain an upper bound on observations actually used after triangulation/gating.",
              "3. **Batch IMU covariance.** `batch` switches image-time IMU covariance accumulation. The risk is "
              "changed cross-correlation/conditioning and a default-path regression; it must first pass the "
              "baseline-equivalence gate before any speed result is attributed to batching.", "",
              "Parameters stay default-off: `schur_nullspace_method=svd`, `schur_batch_imu_covariance=false`, "
              "`schur_max_update_observations=0` (unlimited).", "",
              "## Phase-gate audit", "",
              "- baseline2: complete (V1_02 0.15115 m, MH_05 0.75244 m). baseline8: in progress.",
              "- candidate build and candidate_control: complete; baseline-equivalence table above is the gate.",
              "- screen2: qr/batch/obs240/obs480 runs are reported individually, including the interrupted QR "
              "MH_05 attempt (exit -2, KeyboardInterrupt), labeled `not_algorithm_failure` and excluded as partial; "
              "the successful rerun is the screened value. No repeats or sessions are merged.",
              "- stable6 and `qr_batch` combo remain gated; ARM results are a separate platform group and must not "
              "share CPU claims with local x86 results."]
    if arm_root is not None:
        lines += ["", f"ARM discovery root: `{arm_root}` (missing or empty is reported, never substituted)."]
    return "\n".join(lines) + "\n"


def find_manifest(run_dir, root):
    for parent in [run_dir.parent, *run_dir.parents]:
        candidate = parent / "manifest.json"
        if candidate.is_file():
            return candidate
        if parent == root:
            break
    return Path()


def infer_variant(label):
    if re.search(r"(^|[_-])(?:cand_)?qrbatch($|[_-])", label):
        return "qr_batch"
    if re.search(r"(^|[_-])cand_default($|[_-])", label):
        return "candidate_control"
    for variant in ("candidate_control", "obs240", "obs480", "qr_batch", "baseline", "batch", "qr"):
        if re.search(rf"(^|[_-]){re.escape(variant)}($|[_-])", label):
            return variant
    return label


def choose_truth(manifest_path, run_dir, root, dataset, truth_meta):
    cache = truth_meta.get("cache") or {}
    expected = cache.get("sha256")
    candidates = []
    cached = Path(cache.get("path") or "")
    if cached.is_file():
        candidates.append((cached, "manifest_cache"))
    for base in (manifest_path.parent if manifest_path else None, run_dir.parent, root):
        if base and Path(base).is_dir():
            candidates += [(p, "arm_groundtruth_glob") for p in sorted(Path(base).rglob(f"groundtruth/*{dataset}*.csv"))]
    candidates += [(p, "local_groundtruth_fallback") for p in sorted(
        (TASK / "evidence/results/runs").glob(f"*/benchmark/groundtruth/*{dataset}*.csv"))]
    for path, source in candidates:
        if path.is_file() and (not expected or sha256_file(path) == expected):
            return path, source, True
    return Path(), "unavailable_or_sha_mismatch", False


def collect_arm(root):
    root = Path(root)
    if not root.is_dir():
        return []
    bench = load_benchmark()
    runs = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        metrics = load_json(metrics_path, {}) or {}
        match = re.match(r"^(?P<dataset>.+)_r(?P<repeat>\d+)$", run_dir.name)
        dataset = metrics.get("dataset") or (match.group("dataset") if match else run_dir.parent.name)
        repeat = int(match.group("repeat")) if match else int(metrics.get("repeat", 1))
        manifest_path = find_manifest(run_dir, root)
        manifest = load_json(manifest_path, {}) or {}
        label = manifest_path.parent.name if manifest_path else run_dir.parent.name
        variant = infer_variant(label)
        machine = str((manifest.get("environment") or {}).get("uname", {}).get("machine", ""))
        platform = "arm" if ("arm" in str(root).lower() or "aarch64" in machine or "arm" in machine) else "local"
        truth_meta = ((manifest.get("truth") or {}).get(dataset) or {})
        extrinsic = truth_meta.get("extrinsic") or {}
        gt = {"source": extrinsic.get("source"), "sha256": extrinsic.get("sha256"),
              "sensor_type": truth_meta.get("sensor_type"), "ate_reference": truth_meta.get("ate_reference"),
              "t_bs_source": "manifest" if extrinsic.get("matrix") else "missing"}
        truth_path, truth_source, truth_sha_ok = choose_truth(manifest_path, run_dir, root, dataset, truth_meta)
        traj_path = run_dir / "trajectory.csv"
        status = "success" if metrics.get("ate_rmse_m") is not None and traj_path.is_file() else "failed"
        rate = metrics.get("rate") or manifest.get("rate")
        run = {
            "run_id": f"arm:{label}:{dataset}_r{repeat}", "stage": "arm", "variant": variant,
            "dataset": dataset, "session": label, "status": status, "platform": platform,
            "machine": machine, "rate": rate,
            "group": f"{platform}|rate={rate}|gt={gt['source']}|sha={str(gt['sha256'])[:12]}",
            "gt": gt, "binary_sha256": manifest.get("binary_sha256") or metrics.get("binary_sha256"),
            "parameters": ((manifest.get("argv") or {}).get("parameter") or manifest.get("parameters") or []), "warnings": [],
            "source_root": str(root), "truth_path_source": truth_source, "truth_csv_sha_match": truth_sha_ok,
            "metrics": {k: metrics.get(k) for k in (
                "ate_rmse_m", "ate_median_m", "ate_p95_m", "matched_poses", "output_poses",
                "cpu_seconds", "average_cpu_percent_one_core", "peak_rss_mib",
                "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "max_output_gap_s",
                "received_images", "latency_misses", "restart_requests", "reset_entry_events",
                "failure_reboots", "actual_resets_completed")},
        }
        cache = truth_meta.get("cache") or {}
        run["gt_matched_gap_s"] = metrics.get("max_output_gap_s")
        run["truth_matching"] = truth_matching(metrics, {
            "rows_independently_counted": cache.get("rows"),
            "time_range_independently_read_s": cache.get("time_range_s"),
        })
        run["actual_output_gap_s"] = None
        if status != "success":
            run.update({"outcome_class": "failure_unclassified", "partial_excluded": True,
                        "excluded_from_comparisons": True,
                        "outcome_reason": "ARM metrics/trajectory incomplete; partial excluded"})
        else:
            run.update({"outcome_class": "success", "partial_excluded": False,
                        "excluded_from_comparisons": False})
        run.update(parse_log(run_dir / "estimator.log"))
        tbs = extrinsic.get("matrix")
        if status == "success" and truth_path.is_file() and tbs:
            est, truth = read_rows(traj_path), read_rows(truth_path)
            recomputed = bench.trajectory_metrics(est, truth, np.asarray(tbs, dtype=float))
            run["ate"] = {"recomputed": recomputed, "original_rmse_m": metrics.get("ate_rmse_m"),
                          "matches_original": abs(recomputed.get("ate_rmse_m", -1) - metrics["ate_rmse_m"]) < 1e-9}
            run["_traj"], run["_truth"], run["_tbs"] = est, truth, np.asarray(tbs, dtype=float)
            run["actual_output_gap_s"] = actual_output_gap(est)
        else:
            run["ate"] = {"recomputed": None, "original_rmse_m": metrics.get("ate_rmse_m"),
                          "matches_original": None}
        if label.startswith("diag_") or (manifest.get("argv") or {}).get("duration") is not None:
            run["excluded_from_comparisons"] = True
            run["diagnostic_only"] = True
            run["outcome_class"] = "diagnostic_only"
            run["outcome_reason"] = "Short debug replay; excluded from full-sequence accuracy/performance comparisons"
        runs.append(run)
    return runs


def main():
    global LEDGER
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--arm-results", type=Path, default=TASK / "arm_results")
    ap.add_argument("--json", type=Path, default=JSON_OUT)
    ap.add_argument("--md", type=Path, default=MD_OUT)
    args = ap.parse_args()
    LEDGER = args.ledger
    bench = load_benchmark()
    runs = collect()
    arm_runs = collect_arm(args.arm_results)
    known = {r["run_id"] for r in runs}
    runs += [r for r in arm_runs if r["run_id"] not in known]
    comparisons = compare(runs, bench)
    micro = load_json(MICRO)
    result = {
        "schema": "backend_ablation_minimal_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_tool": True,
        "tool_sha256": sha256_file(Path(__file__)),
        "ledger": str(args.ledger),
        "ledger_sha256": sha256_file(args.ledger) if args.ledger.is_file() else None,
        "arm_results_root": str(args.arm_results), "arm_runs_discovered": len(arm_runs),
        "runs_total": len(runs), "runs_success": sum(r["status"] == "success" for r in runs),
        "rules": {
            "ate_reference": "sensor_origin",
            "ate_call": "benchmark_euroc.trajectory_metrics(est, truth, T_BS)",
            "t_bs_source": "manifest", "common_timestamp_ate": True,
            "groups": ["platform", "rate", "gt_extrinsic_source"],
            "cpu_comparison": "only within identical platform+rate+GT group",
            "repeats_and_sessions": "never merged",
            "gap_semantics": {
                "actual_output_gap_s": "max consecutive diff over all unique timestamps in original trajectory.csv",
                "gt_matched_gap_s": "benchmark metrics.max_output_gap_s after truth-valid filtering",
                "report": "gap column uses actual_output_gap_s; GT-matched gap is kept separately",
            },
            "truth_matching": "matched/output preserved; unmatched is not dropped output",
            "screening_rule": f"engineering triage only: candidate common ATE <= candidate_default * {1 + ATE_TOL:.2f}",
            "screening_labels": ["screening_within_5pct", "screening_regression", "screening_not_applicable"],
            "screening_threshold_provenance": "not pre-agreed with user; not user acceptance; regression label is not a global failure claim",
            "partial_runs": "excluded from metrics and comparisons; interrupted runs labeled not_algorithm_failure with exit -2/130/143 or KeyboardInterrupt/SIGINT evidence",
            "eligible_parameter_defaults": {"schur_nullspace_method": "svd",
                                            "schur_batch_imu_covariance": False,
                                            "schur_max_update_observations": 0},
        },
        "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs],
        "comparisons": comparisons, "microbench": micro,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    args.md.write_text(markdown(result["runs"], comparisons, micro, args.arm_results))
    print(f"wrote {args.json} ({len(runs)} runs, {len(comparisons)} comparisons, {len(arm_runs)} ARM discoveries)")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
