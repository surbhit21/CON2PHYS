"""Tests for shift-predictor-corrected Pynapple cross-correlograms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pynapple as nap

from src.Q4_v3 import (
    corrected_crosscorrelograms,
    plot_corrected_crosscorrelograms_by_area,
    plot_pairwise_corrected_crosscorrelogram_heatmaps,
)
from src.data_loader import CON2PHYSSession


def test_corrected_crosscorrelogram_shapes(tmp_path) -> None:
    onsets = np.arange(2.0, 14.0, 2.0)
    unit_1 = np.concatenate([onsets + 0.01, onsets + 0.20])
    unit_2 = np.concatenate([onsets + 0.02, onsets + 0.25])
    spikes = nap.TsGroup(
        {10: nap.Ts(t=np.sort(unit_1)), 20: nap.Ts(t=np.sort(unit_2))},
        metadata=pd.DataFrame({"brain_area": [1, 2]}, index=[10, 20]),
    )
    trials = pd.DataFrame(
        {
            "variable_A": [0, 0, 0, 1, 1, 1],
            "variable_B": [0, 0, 0, 1, 1, 1],
            "variable_C": [0, 0, 0, 2, 2, 2],
            "trial_start": onsets,
            "stim_start": onsets + 0.1,
            "outcome": onsets + 0.3,
            "trial_end": onsets + 0.5,
        }
    )
    session = CON2PHYSSession(
        subject=1,
        lfp={},
        spikes=spikes,
        trials=nap.IntervalSet(start=onsets, end=onsets + 0.5),
        events={},
        trial_table=trials,
        waveforms=None,
        sampling_rate=500,
    )

    corrected, observed, predictor, pairs, peaks = corrected_crosscorrelograms(
        session, bin_size_ms=10, max_lag_ms=20, n_shift_predictors=2
    )

    assert corrected.shape == observed.shape == predictor.shape == (4, 5, 1)
    assert pairs == [(10, 20)]
    np.testing.assert_allclose(corrected, observed - predictor)
    assert set(peaks.columns).issuperset(
        {"event_type", "unit_a", "unit_b", "lag_ms", "peak_type"}
    )
    lags = np.linspace(-20, 20, corrected.shape[1])
    area_plot = tmp_path / "areas.png"
    heatmap = tmp_path / "heatmap.png"
    plot_corrected_crosscorrelograms_by_area(
        corrected, lags, np.array([1]), np.array([2]), area_plot
    )
    plot_pairwise_corrected_crosscorrelogram_heatmaps(
        corrected, lags, heatmap
    )
    assert area_plot.is_file() and area_plot.stat().st_size > 0
    assert heatmap.is_file() and heatmap.stat().st_size > 0
