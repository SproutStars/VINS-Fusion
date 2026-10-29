#!/usr/bin/env python3
"""Inspect saved margin-cloud geometry; distance tails are diagnostics, not map accuracy.

Input files come from validate_schur_loop.py --save-margin. Uses identical axes
for before/after plots, with no display clipping or point removal. Vicon provides
trajectory ground truth, not surface truth: these statistics cannot establish
absolute reconstruction accuracy or parity with a Ceres map.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    data = np.load(path)
    rows, trajectory = data['points'], data['odometry']
    finite = np.isfinite(rows).all(axis=1)
    rows = rows[finite]
    # Discard startup/Ceres warmup consistently, relative to first VIO pose.
    rows = rows[rows[:, 0] / 1e9 >= trajectory[0, 0] + 20]
    xyz = rows[:, 1:]
    camera = np.column_stack([np.interp(rows[:, 0] / 1e9, trajectory[:, 0], trajectory[:, i])
                              for i in range(1, 4)])
    distance = np.linalg.norm(xyz - camera, axis=1)
    stats = {'points': len(xyz), 'nonfinite': int((~finite).sum()),
             'distance_to_body_p50_m': float(np.percentile(distance, 50)) if len(xyz) else None,
             'distance_to_body_p95_m': float(np.percentile(distance, 95)) if len(xyz) else None,
             'distance_to_body_p99_m': float(np.percentile(distance, 99)) if len(xyz) else None,
             'distance_over_10m_count': int((distance > 10).sum()),
             'distance_over_10m_fraction': float(np.mean(distance > 10)) if len(xyz) else None,
             'distance_over_20m_count': int((distance > 20).sum())}
    return xyz, trajectory[:, 1:], stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) != len(args.labels):
        parser.error('one label is needed for each input')
    values = [load(p) for p in args.inputs]
    all_xyz = np.concatenate([v[0] for v in values] + [v[1] for v in values])
    lo, hi = all_xyz.min(axis=0), all_xyz.max(axis=0)
    pad = np.maximum((hi - lo) * .03, .2)
    fig, axes = plt.subplots(2, len(values), figsize=(6 * len(values), 10), squeeze=False)
    for j, ((xyz, trajectory, stats), label) in enumerate(zip(values, args.labels)):
        for i, (a, b) in enumerate(((0, 1), (0, 2))):
            ax = axes[i, j]
            ax.scatter(xyz[:, a], xyz[:, b], s=.4, alpha=.25, rasterized=True)
            ax.plot(trajectory[:, a], trajectory[:, b], color='orange', lw=.8)
            ax.set_xlim(lo[a]-pad[a], hi[a]+pad[a]); ax.set_ylim(lo[b]-pad[b], hi[b]+pad[b])
            ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.15)
            ax.set_xlabel('XYZ'[a] + ' (m)'); ax.set_ylabel('XYZ'[b] + ' (m)')
            ax.set_title(f'{label}: {stats["points"]:,} history points')
    fig.suptitle('Accumulated margin cloud / identical axes / no clipping', fontsize=14)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=160)
    args.output.with_suffix('.json').write_text(json.dumps(dict(zip(args.labels, [v[2] for v in values])), indent=2)+'\n')
    print(args.output.with_suffix('.json').read_text())


if __name__ == '__main__':
    main()
