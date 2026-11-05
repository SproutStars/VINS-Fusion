#!/usr/bin/env python3
"""Compare end-of-sequence drift using a rigid alignment fitted before the tail.

No scale or timestamp is fitted. Tail displacement error is relative to the
first matched pose of that tail; it does not hide drift with a tail-only fit.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def landing_metrics(estimated, truth, tail_seconds=10.0):
    estimated, truth = np.asarray(estimated), np.asarray(truth)
    truth = truth[np.argsort(truth[:, 0])]
    truth = truth[np.r_[True, np.diff(truth[:, 0]) > 0]]
    valid = (estimated[:, 0] >= truth[0, 0]) & (estimated[:, 0] <= truth[-1, 0])
    samples = estimated[valid]
    right = np.searchsorted(truth[:, 0], samples[:, 0], side='right').clip(1, len(truth)-1)
    samples = samples[truth[right, 0] - truth[right-1, 0] <= 0.1]
    if len(samples) < 3:
        raise ValueError('Too few matched poses')
    times = samples[:, 0]
    reference = np.column_stack([np.interp(times, truth[:, 0], truth[:, i]) for i in range(1, 4)])
    prefix = times < truth[-1, 0] - tail_seconds
    if prefix.sum() < 3:
        raise ValueError('Too few poses before tail for rigid alignment')
    source_mean = samples[prefix, 1:4].mean(axis=0)
    target_mean = reference[prefix].mean(axis=0)
    u, _, vt = np.linalg.svd((samples[prefix, 1:4] - source_mean).T @ (reference[prefix] - target_mean))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    aligned = (samples[:, 1:4] - source_mean) @ rotation + target_mean
    error = aligned - reference
    result = {'matched_poses': len(times), 'tail_alignment_exclusion_s': tail_seconds,
              'truth_end_minus_output_end_s': float(truth[-1, 0] - times[-1]),
              'max_output_gap_s': float(np.diff(times).max()),
              'prefix_ate_rmse_m': float(np.sqrt(np.mean(np.sum(error[prefix]**2, axis=1))))}
    for seconds in (10, 5, 3, 1, 0.5):
        tail = times >= truth[-1, 0] - seconds
        if tail.sum() < 2:
            result[f'tail_{seconds}s'] = {'matched_poses': int(tail.sum())}
            continue
        e = error[tail]
        relative = np.linalg.norm(e-e[0], axis=1)
        result[f'tail_{seconds}s'] = {
            'matched_poses': int(tail.sum()),
            'duration_s': float(np.ptp(times[tail])),
            'ate_rmse_m': float(np.sqrt(np.mean(np.sum(e**2, axis=1)))),
            'relative_end_error_m': float(relative[-1]),
            'relative_max_error_m': float(relative.max()),
            'estimated_path_length_m': float(np.linalg.norm(np.diff(aligned[tail], axis=0), axis=1).sum()),
            'truth_path_length_m': float(np.linalg.norm(np.diff(reference[tail], axis=0), axis=1).sum()),
        }
    return result, (times-truth[-1, 0], aligned, reference, error)


def landing_events(log_path, truth):
    """Record every detector entry against independent Vicon motion."""
    text = log_path.read_text()
    events = []
    for match in re.finditer(r'Landing ZUPT armed at frame (\d+) \(t=([\d.]+)\)', text):
        stamp = float(match.group(2))
        nearby = truth[(truth[:, 0] >= stamp - .25) & (truth[:, 0] <= stamp + .25)]
        events.append({
            'frame': int(match.group(1)), 'stamp': stamp,
            'seconds_before_truth_end': float(truth[-1, 0] - stamp),
            'vicon_samples_near_entry': len(nearby),
            'vicon_position_span_near_entry_m': (
                np.ptp(nearby[:, 1:4], axis=0).tolist() if len(nearby) > 1 else None),
        })
    return {'entries': events, 'exit_count': text.count('Landing ZUPT exited'),
            'rejected_update_logs': text.count('Landing ZUPT update rejected')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    results, curves = {}, {}
    for label in args.labels:
        results[label] = {}
        for trajectory in sorted((args.root/label).glob('*/trajectory.csv')):
            dataset = trajectory.parent.name.rsplit('_r', 1)[0]
            truth = np.loadtxt(args.root/'groundtruth'/f'{dataset}.csv', delimiter=',', skiprows=1)
            estimated = np.loadtxt(trajectory, delimiter=',', skiprows=1)
            metrics, curve = landing_metrics(estimated, truth)
            metrics['landing_detector'] = landing_events(trajectory.parent/'estimator.log', truth)
            velocity_path = trajectory.parent/'vio.csv'
            if velocity_path.exists():
                # Estimator's native CSV: timestamp(ns), p, quaternion(wxyz), v.
                velocity_rows = np.loadtxt(velocity_path, delimiter=',', usecols=range(11))
                metrics['final_speed_m_s'] = float(np.linalg.norm(velocity_rows[-1, 8:11]))
                velocity_t = velocity_rows[:, 0]*1e-9
                for seconds in (1, 0.5):
                    mask = (velocity_t >= truth[-1, 0]-seconds) & (velocity_t <= truth[-1, 0])
                    if mask.any():
                        metrics[f'tail_{seconds}s']['speed_rms_m_s'] = float(
                            np.sqrt(np.mean(np.sum(velocity_rows[mask, 8:11]**2, axis=1))))
            results[label][trajectory.parent.name] = metrics
            curves.setdefault(dataset, []).append((label, curve))
    (args.root/'landing_metrics.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(curves), 2, figsize=(12, 3.5*len(curves)), squeeze=False)
        for row, (dataset, entries) in enumerate(sorted(curves.items())):
            for label, (t, aligned, reference, error) in entries:
                tail = t >= -10
                e = error[tail]
                axes[row, 0].plot(t[tail], np.linalg.norm(e-e[0], axis=1), label=label)
                axes[row, 1].plot(t[tail], aligned[tail, 2]-aligned[tail, 2][0], label=label)
            axes[row, 1].plot(t[tail], reference[tail, 2]-reference[tail, 2][0], 'k--', label='Vicon')
            axes[row, 0].set_title(dataset+' — relative displacement error')
            axes[row, 1].set_title(dataset+' — vertical displacement')
            for ax in axes[row]:
                ax.set_xlabel('Seconds before Vicon end'); ax.set_ylabel('m'); ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(args.root/'landing_comparison.png', dpi=160)


if __name__ == '__main__':
    main()
