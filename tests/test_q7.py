"""Tests for waveform-based fast-spiking analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pynapple as nap
import pytest

from src.Q7 import (
    compare_area_densities,
    fast_spiking_density,
    plot_waveform_examples,
    summarize_participant_densities,
    waveform_features,
)
from src.data_loader import CON2PHYSSession


def _waveform_session() -> CON2PHYSSession:
    # At 10 kHz these trough-to-peak widths are 0.2, 0.5, and 0.3 ms.
    waveforms = pd.DataFrame(
        [
            [0, 0, -3, -1, 2, 0, 0],
            [0, 0, -3, -2, -1, 0, 2],
            [0, 0, -3, -2, 0, 2, 0],
        ],
        index=[10, 11, 12],
    )
    spikes = nap.TsGroup(
        {
            10: nap.Ts(t=[0.1, 0.8]),
            11: nap.Ts(t=[0.2, 0.7]),
            12: nap.Ts(t=[0.3, 0.6]),
        },
        metadata=pd.DataFrame(
            {"brain_area": [1, 1, 2]},
            index=[10, 11, 12],
        ),
    )
    return CON2PHYSSession(
        subject=1,
        lfp={},
        spikes=spikes,
        trials=nap.IntervalSet(start=[0], end=[1]),
        events={},
        trial_table=pd.DataFrame(),
        waveforms=waveforms,
        sampling_rate=500,
    )


def test_waveform_width_classifies_fast_spiking_units() -> None:
    features = waveform_features(
        _waveform_session(),
        sampling_rate=10_000,
        threshold_ms=0.4,
        baseline_samples=2,
    )

    np.testing.assert_allclose(features["trough_to_peak_ms"], [0.2, 0.4, 0.3])
    assert features["putative_fast_spiking"].tolist() == [True, True, True]
    assert features["brain_area"].tolist() == [1, 1, 2]


def test_fast_spiking_density_is_calculated_per_area() -> None:
    features = pd.DataFrame(
        {
            "brain_area": [1, 1, 1, 2, 2],
            "putative_fast_spiking": [True, False, False, True, True],
        }
    )

    density = fast_spiking_density(features)

    assert density.loc[1, "n_units"] == 3
    assert density.loc[1, "n_putative_fast_spiking"] == 1
    assert density.loc[1, "density"] == pytest.approx(1 / 3)
    assert density.loc[2, "density_percent"] == 100
    assert 0 <= density.loc[1, "ci95_low"] < density.loc[1, "ci95_high"] <= 1


def test_example_waveform_plot_is_created(tmp_path) -> None:
    session = _waveform_session()
    features = waveform_features(
        session,
        sampling_rate=10_000,
        threshold_ms=0.4,
        baseline_samples=2,
    )
    output = tmp_path / "examples.png"

    plot_waveform_examples(
        session,
        features,
        output,
        sampling_rate=10_000,
        examples_per_area=2,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_group_summary_and_highest_density_tests() -> None:
    participant_density = pd.DataFrame(
        {
            "participant": np.repeat(np.arange(1, 7), 3),
            "brain_area": np.tile([1, 2, 3], 6),
            "density": np.array(
                [
                    [0.1, 0.5, 0.2],
                    [0.2, 0.6, 0.3],
                    [0.1, 0.7, 0.2],
                    [0.2, 0.8, 0.4],
                    [0.3, 0.9, 0.5],
                    [0.2, 0.7, 0.3],
                ]
            ).ravel(),
        }
    )

    summary = summarize_participant_densities(participant_density)
    highest, pairwise, friedman = compare_area_densities(participant_density)

    assert summary.loc[2, "n_participants"] == 6
    assert summary.loc[2, "ci95_low"] < summary.loc[2, "mean_density"]
    assert summary.loc[2, "ci95_high"] > summary.loc[2, "mean_density"]
    assert highest == 2
    assert set(pairwise["comparison_area"]) == {1, 3}
    assert friedman.pvalue < 0.05
