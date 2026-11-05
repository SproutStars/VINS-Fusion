import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'analyze_landing_drift', Path(__file__).parents[1] / 'analyze_landing_drift.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rigid_frame_change_does_not_create_tail_drift():
    t = np.arange(0, 31, .05)
    positions = np.column_stack((np.sin(t), np.cos(t), .05*t))
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    truth = np.column_stack((t, positions))
    estimated = np.column_stack((t, positions @ rotation + [4, -2, 7]))
    metrics, _ = module.landing_metrics(estimated, truth)
    assert metrics['tail_5s']['relative_end_error_m'] < 1e-12
    assert metrics['prefix_ate_rmse_m'] < 1e-12


def test_tail_fit_cannot_absorb_drift_and_missing_tail_is_reported():
    t = np.arange(0, 31, .05)
    positions = np.column_stack((np.sin(t), np.cos(t), .05*t))
    truth = np.column_stack((t, positions))
    estimated = truth.copy()
    estimated[:, 1] += np.maximum(t - (t[-1]-10), 0)*.2
    metrics, _ = module.landing_metrics(estimated, truth)
    assert metrics['tail_5s']['relative_end_error_m'] == pytest.approx(.2 * metrics['tail_5s']['duration_s'])
    assert metrics['prefix_ate_rmse_m'] < 1e-12
    shortened, _ = module.landing_metrics(estimated[t < 25], truth)
    assert shortened['truth_end_minus_output_end_s'] > 5
    assert shortened['tail_1s']['matched_poses'] == 0
