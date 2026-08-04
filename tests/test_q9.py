from __future__ import annotations

import numpy as np
import pandas as pd
import pynapple as nap

from src.Q9 import (
    classify_and_calculate_unit_rates,
    compare_brain_area_ratios,
    excitation_inhibition_ratio,
    summarize_ratios_across_participants,
)
from src.data_loader import CON2PHYSSession


def _session() -> CON2PHYSSession:
    # At 10 kHz: unit 10 is narrow (0.2 ms), units 11/12 are broad (0.5 ms).
    waveforms = pd.DataFrame(
        [
            [0, 0, -3, -1, 2, 0, 0, 0],
            [0, 0, -3, -2, -1, 0, 1, 2],
            [0, 0, -3, -2, -1, 0, 1, 2],
        ],
        index=[10, 11, 12],
    )
    spikes = nap.TsGroup(
        {
            10: nap.Ts(t=[0.015, 0.055]),
            11: nap.Ts(t=[0.015, 0.025, 0.055, 0.065]),
            12: nap.Ts(t=[0.035, 0.075]),
        },
        metadata=pd.DataFrame({"brain_area": [1, 1, 1]}, index=[10, 11, 12]),
    )
    trials = pd.DataFrame(
        {"trial_start": [0.0, 0.05], "trial_end": [0.04, 0.09]}
    )
    return CON2PHYSSession(
        subject=1,
        lfp={},
        spikes=spikes,
        trials=nap.IntervalSet(start=trials.trial_start, end=trials.trial_end),
        events={},
        trial_table=trials,
        waveforms=waveforms,
        sampling_rate=500,
    )


def test_q9_classification_rates_and_ratio() -> None:
    rates = classify_and_calculate_unit_rates(
        _session(),
        bin_size_s=0.01,
        waveform_sampling_rate=10_000,
        baseline_samples=2,
    )
    assert rates.loc[10, "cell_class"] == "putative_inhibitory"
    assert rates.loc[11, "cell_class"] == "putative_excitatory"
    assert rates.loc[12, "cell_class"] == "putative_excitatory"
    np.testing.assert_allclose(rates["mean_firing_rate_hz"], [25, 50, 25])

    summary = excitation_inhibition_ratio(rates)
    overall = summary.loc[summary["brain_area"] == "all"].iloc[0]
    assert overall["mean_excitatory_firing_rate_hz"] == 37.5
    assert overall["mean_inhibitory_firing_rate_hz"] == 25
    assert overall["excitation_to_inhibition_ratio"] == 1.5


def test_group_ratio_summary_and_paired_area_tests() -> None:
    ratios = pd.DataFrame(
        {
            "participant": np.repeat(np.arange(1, 9), 3),
            "brain_area": np.tile([1, 2, 3], 8),
            "excitation_to_inhibition_ratio": np.array(
                [
                    [1.0, 3.0, 1.5], [1.2, 3.2, 1.4],
                    [0.9, 2.8, 1.3], [1.1, 3.1, 1.6],
                    [1.0, 3.3, 1.2], [1.3, 3.4, 1.7],
                    [0.8, 2.9, 1.4], [1.1, 3.2, 1.5],
                ]
            ).ravel(),
        }
    )
    summary = summarize_ratios_across_participants(ratios)
    highest, pairwise, friedman = compare_brain_area_ratios(ratios)
    assert summary.loc[2, "n_participants"] == 8
    assert summary.loc[2, "ci_low"] < summary.loc[2, "mean_ratio"] < summary.loc[2, "ci_high"]
    assert highest == 2
    assert friedman.pvalue < 0.05
    assert pairwise["significant_0.05"].all()
