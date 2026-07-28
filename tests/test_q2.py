"""Tests for the subject 1 LFP loading script."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pynapple as nap
import pytest

from src.Q2 import (
    broadband_power_by_area,
    compare_brain_areas,
    load_subject_1_lfp,
    summarize_subject_power,
)
from src.data_loader import CON2PHYSSession


def test_load_subject_1_lfp_uses_memory_conscious_options() -> None:
    sentinel = object()
    with patch("src.Q2.load_session", return_value=sentinel) as loader:
        result = load_subject_1_lfp((1, 2))

    assert result is sentinel
    loader.assert_called_once_with(
        subject=1,
        lfp_banks=(1, 2),
        load_spikes=False,
        lfp_dtype="float32",
    )


def test_broadband_power_is_summarized_across_channels() -> None:
    fs = 500.0
    times = np.arange(0, 8, 1 / fs)
    wave = np.sin(2 * np.pi * 10 * times)
    lfp = {
        1: nap.TsdFrame(
            t=times,
            d=np.column_stack([wave, 2 * wave, 3 * wave]),
            columns=["a", "b", "c"],
        )
    }
    session = CON2PHYSSession(
        subject=1,
        lfp=lfp,
        spikes=None,
        trials=nap.IntervalSet(start=[0], end=[8]),
        events={},
        trial_table=None,
        waveforms=None,
        sampling_rate=fs,
    )

    channels, summary = broadband_power_by_area(session)

    assert channels["broadband_power"].to_numpy() == pytest.approx([0.5, 2.0, 4.5])
    assert summary.loc[1, "n_channels"] == 3
    assert summary.loc[1, "mean_power"] == pytest.approx(7 / 3)
    assert summary.columns.tolist() == ["n_channels", "mean_power"]


def test_group_summary_and_highest_area_tests() -> None:
    subject_power = pd.DataFrame(
        {
            "subject": np.repeat(np.arange(1, 7), 3),
            "brain_area": np.tile([1, 2, 3], 6),
            "mean_power": np.array(
                [
                    [1, 3, 2],
                    [2, 5, 3],
                    [1, 4, 2],
                    [2, 6, 4],
                    [3, 7, 5],
                    [2, 5, 3],
                ]
            ).ravel(),
        }
    )

    summary = summarize_subject_power(subject_power)
    highest, pairwise, friedman = compare_brain_areas(subject_power)

    assert summary.loc[2, "n_subjects"] == 6
    assert summary.loc[2, "ci95_low"] < summary.loc[2, "mean_power"]
    assert summary.loc[2, "ci95_high"] > summary.loc[2, "mean_power"]
    assert highest == 2
    assert set(pairwise["comparison_area"]) == {1, 3}
    assert friedman.pvalue < 0.05
