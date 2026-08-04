"""Tests for event-aligned spike tensor construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pynapple as nap

from src.Q4_v2 import (
    construct_event_aligned_spike_tensors,
    plot_checkpoint_correlation_matrices,
    plot_time_resolved_correlations,
    residualize_spike_counts_by_condition,
    save_event_aligned_spike_tensors,
    save_residual_spike_tensor,
    save_time_resolved_correlations,
    time_resolved_residual_correlations,
)
from src.data_loader import CON2PHYSSession


def test_event_aligned_spike_tensor_shape_and_counts(tmp_path) -> None:
    spikes = nap.TsGroup(
        {
            10: nap.Ts(t=[0.95, 1.005, 1.015, 2.005]),
            20: nap.Ts(t=[1.025, 2.015]),
        },
        metadata=pd.DataFrame({"brain_area": [1, 2]}, index=[10, 20]),
    )
    trial_table = pd.DataFrame(
        {
            "variable_A": [0, 0],
            "variable_B": [1, 1],
            "variable_C": [2, 2],
            "trial_start": [1.0, 2.0],
            "stim_start": [1.1, 2.1],
            "outcome": [1.2, 2.2],
            "trial_end": [1.3, 2.3],
        },
        index=[5, 6],
    )
    session = CON2PHYSSession(
        subject=1,
        lfp={},
        spikes=spikes,
        trials=nap.IntervalSet(start=[1, 2], end=[1.3, 2.3]),
        events={},
        trial_table=trial_table,
        waveforms=None,
        sampling_rate=500,
    )

    tensors = construct_event_aligned_spike_tensors(
        session, window_s=(-0.1, 0.1), bin_size_s=0.01
    )

    assert set(tensors) == {
        "trial_onset",
        "stimulus_presentation",
        "outcome",
        "trial_end",
    }
    assert tensors["trial_onset"].shape == (20, 2, 2)
    assert np.asarray(tensors["trial_onset"]).sum() == 6
    np.testing.assert_allclose(
        tensors["trial_onset"].t[[0, -1]], [-0.095, 0.095]
    )

    output = tmp_path / "tensor.npz"
    unit_metadata = save_event_aligned_spike_tensors(tensors, session, output)
    saved = np.load(output)
    assert saved["spike_counts"].shape == (4, 2, 20, 2)
    assert saved["trial_ids"].tolist() == [5, 6]
    assert saved["unit_ids"].tolist() == [10, 20]
    assert saved["brain_areas"].tolist() == [1, 2]
    assert saved["unit_axis"].tolist() == [0, 1]
    assert unit_metadata[["unit_index", "unit_id", "brain_area"]].to_dict(
        "records"
    ) == [
        {"unit_index": 0, "unit_id": 10, "brain_area": 1},
        {"unit_index": 1, "unit_id": 20, "brain_area": 2},
    ]
    assert (tmp_path / "tensor_unit_metadata.csv").is_file()

    residuals, condition_codes, condition_table = (
        residualize_spike_counts_by_condition(tensors, session)
    )
    assert residuals.shape == saved["spike_counts"].shape
    np.testing.assert_allclose(residuals.mean(axis=1), 0.0, atol=1e-7)
    assert condition_codes.tolist() == [0, 0]
    assert condition_table["n_trials"].tolist() == [2]

    residual_output = tmp_path / "residuals.npz"
    save_residual_spike_tensor(
        residuals,
        condition_codes,
        condition_table,
        tensors,
        session,
        residual_output,
    )
    residual_saved = np.load(residual_output)
    assert residual_saved["residual_spike_counts"].shape == (4, 2, 20, 2)
    assert residual_saved["brain_areas"].tolist() == [1, 2]

    correlations, starts, ends = time_resolved_residual_correlations(
        residuals,
        tensors["trial_onset"].t,
        window_s=0.05,
        step_s=0.02,
    )
    assert correlations.shape == (4, 8, 2, 2)
    np.testing.assert_allclose(
        correlations, correlations.transpose(0, 1, 3, 2), equal_nan=True
    )
    assert np.nanmax(np.abs(correlations)) <= 1.0
    np.testing.assert_allclose(ends - starts, 0.05)

    correlation_output = tmp_path / "correlations.npz"
    save_time_resolved_correlations(
        correlations, starts, ends, session, correlation_output
    )
    correlation_saved = np.load(correlation_output)
    assert correlation_saved["correlations"].shape == (4, 8, 2, 2)
    assert correlation_saved["unit_ids"].tolist() == [10, 20]

    time_plot = tmp_path / "correlation_time_courses.png"
    matrix_plot = tmp_path / "correlation_matrices.png"
    plot_time_resolved_correlations(
        correlations, starts, ends, np.array([1, 2]), time_plot
    )
    plot_checkpoint_correlation_matrices(
        correlations, starts, ends, np.array([1, 2]), matrix_plot
    )
    assert time_plot.is_file() and time_plot.stat().st_size > 0
    assert matrix_plot.is_file() and matrix_plot.stat().st_size > 0
