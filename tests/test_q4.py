"""Tests for within-region spike cross-correlograms."""

from __future__ import annotations

import pandas as pd
import pynapple as nap

from src.Q4 import (
    crosscorrelograms_by_region,
    isi_distributions_by_region,
    plot_inter_spike_interval_distribution,
    plot_trialwise_isi_distributions,
    plot_trialwise_crosscorrelograms,
    plot_trial_normalized_firing_rates,
    summarize_crosscorrelograms,
    trial_normalized_firing_rates,
    trialwise_crosscorrelogram_summary,
    trialwise_isi_distributions,
)
from src.data_loader import CON2PHYSSession


def test_crosscorrelograms_include_each_within_region_pair() -> None:
    spikes = nap.TsGroup(
        {
            10: nap.Ts(t=[0.1, 0.2, 0.3]),
            11: nap.Ts(t=[0.101, 0.201, 0.301]),
            12: nap.Ts(t=[0.15, 0.25]),
            20: nap.Ts(t=[0.1, 0.3]),
        },
        metadata=pd.DataFrame(
            {"brain_area": [1, 1, 1, 2]}, index=[10, 11, 12, 20]
        ),
    )
    session = CON2PHYSSession(
        subject=1,
        lfp={},
        spikes=spikes,
        trials=nap.IntervalSet(start=[0], end=[0.4]),
        events={},
        trial_table=pd.DataFrame(),
        waveforms=None,
        sampling_rate=500,
    )

    result = crosscorrelograms_by_region(
        session, bin_size_ms=1, window_size_ms=2
    )
    pairs = result[
        ["brain_area", "unit_reference", "unit_target"]
    ].drop_duplicates()

    assert set(map(tuple, pairs.to_numpy())) == {
        (1, 10, 11),
        (1, 10, 12),
        (1, 11, 12),
    }
    assert set(result["lag_ms"]) == {-2, -1, 0, 1, 2}
    assert (result["rate_hz"] >= 0).all()

    summary = summarize_crosscorrelograms(result)
    assert (summary["n_pairs"] == 3).all()


def test_inter_spike_intervals_and_plot(tmp_path) -> None:
    spikes = nap.TsGroup(
        {
            10: nap.Ts(t=[0.1, 0.11, 0.14]),
            20: nap.Ts(t=[0.2, 0.25]),
        },
        metadata=pd.DataFrame({"brain_area": [1, 2]}, index=[10, 20]),
    )
    session = CON2PHYSSession(
        subject=1,
        lfp={},
        spikes=spikes,
        trials=nap.IntervalSet(start=[0], end=[0.4]),
        events={},
        trial_table=pd.DataFrame(),
        waveforms=None,
        sampling_rate=500,
    )

    distributions = isi_distributions_by_region(session, n_bins=10)
    output = tmp_path / "isi.png"
    plot_inter_spike_interval_distribution(distributions, output)

    assert set(distributions["brain_area"]) == {1, 2}
    assert distributions.groupby("brain_area")["count"].sum().to_dict() == {
        1: 2,
        2: 1,
    }
    probabilities = distributions.groupby("brain_area")["probability"].sum()
    assert probabilities.round(12).to_dict() == {1: 1.0, 2: 1.0}
    assert output.is_file()
    assert output.stat().st_size > 0


def test_trialwise_isi_uses_stimulus_to_outcome_epochs(tmp_path) -> None:
    spikes = nap.TsGroup(
        {
            10: nap.Ts(t=[0.05, 0.11, 0.13, 0.19, 0.31, 0.34, 0.39]),
            20: nap.Ts(t=[0.12, 0.16, 0.32, 0.38]),
        },
        metadata=pd.DataFrame({"brain_area": [1, 2]}, index=[10, 20]),
    )
    session = CON2PHYSSession(
        subject=1,
        lfp={},
        spikes=spikes,
        trials=nap.IntervalSet(start=[0, 0.25], end=[0.2, 0.4]),
        events={},
        trial_table=pd.DataFrame(
            {
                "trial_start": [0.0, 0.25],
                "stim_start": [0.1, 0.3],
                "outcome": [0.2, 0.4],
                "trial_end": [0.21, 0.41],
            },
            index=[1, 2],
        ),
        waveforms=None,
        sampling_rate=500,
    )

    distributions = trialwise_isi_distributions(
        session, n_bins=10, minimum_isi_ms=1
    )
    output = tmp_path / "trialwise_isi.png"
    plot_trialwise_isi_distributions(distributions, output)

    totals = (
        distributions.groupby(["trial", "brain_area"])["n_intervals"].first()
    )
    assert totals.to_dict() == {(1, 1): 2, (1, 2): 1, (2, 1): 2, (2, 2): 1}
    assert output.is_file()
    assert output.stat().st_size > 0

    ccg_spikes = nap.TsGroup(
        {
            10: nap.Ts(t=[0.11, 0.13, 0.31, 0.34]),
            11: nap.Ts(t=[0.111, 0.131, 0.311, 0.341]),
        },
        metadata=pd.DataFrame({"brain_area": [1, 1]}, index=[10, 11]),
    )
    ccg_session = CON2PHYSSession(
        subject=1,
        lfp={},
        spikes=ccg_spikes,
        trials=session.trials,
        events={},
        trial_table=session.trial_table,
        waveforms=None,
        sampling_rate=500,
    )
    ccg = trialwise_crosscorrelogram_summary(
        ccg_session, bin_size_ms=1, window_size_ms=2
    )
    ccg_output = tmp_path / "trialwise_ccg.png"
    plot_trialwise_crosscorrelograms(ccg, ccg_output)

    assert set(ccg["trial"]) == {1, 2}
    assert set(ccg["brain_area"]) == {1}
    assert set(ccg["lag_ms"]) == {-2, -1, 0, 1, 2}
    assert ccg_output.is_file()
    assert ccg_output.stat().st_size > 0

    trial_rates, rate_summary, timeline = trial_normalized_firing_rates(
        session, n_bins=10
    )
    rate_output = tmp_path / "trial_firing_rate.png"
    plot_trial_normalized_firing_rates(rate_summary, timeline, rate_output)

    assert trial_rates["trial"].nunique() == 2
    assert set(rate_summary["brain_area"]) == {1, 2}
    assert (rate_summary["n_trials"] == 2).all()
    assert timeline["n_trials"] == 2
    assert timeline["stimulus_percent"] < timeline["outcome_percent"]
    assert rate_output.is_file()
    assert rate_output.stat().st_size > 0
