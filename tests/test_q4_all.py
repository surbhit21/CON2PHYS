"""Tests for across-participant near-zero CCG peak analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from CON2PHYS.src.Q4_final import participant_peak_metrics, summarize_across_participants


def test_peak_metric_and_paired_area_summary() -> None:
    pairs = [(10, 11), (20, 21), (30, 31)]
    areas = pd.Series({10: 1, 11: 1, 20: 2, 21: 2, 30: 3, 31: 3})
    lags = np.arange(-100, 101, 10)
    corrected = np.zeros((4, len(lags), 3))
    corrected[:, lags == 0, :] = [1.0, 2.0, 3.0]
    one = participant_peak_metrics(1, corrected, pairs, areas, lags)
    assert one.groupby("brain_area")["narrow_peak_above_baseline_hz"].mean().to_dict() == {
        1: 1.0,
        2: 2.0,
        3: 3.0,
    }

    all_metrics = pd.concat(
        [one.assign(participant=participant) for participant in range(1, 7)],
        ignore_index=True,
    )
    participant_metrics, comparisons, friedman = summarize_across_participants(
        all_metrics
    )
    assert len(participant_metrics) == 18
    assert len(comparisons) == 3
    assert friedman.pvalue < 0.05
