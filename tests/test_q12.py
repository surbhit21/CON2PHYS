"""Tests for complete-epoch decoding of variable C."""

import numpy as np
import pandas as pd
import pynapple as nap

from src.Q12 import (
    epoch_firing_rates,
    epoch_statistical_tests,
    plot_mean_decoding_metrics,
    stratified_train_val_test_split,
    summarize_results,
)


def test_split_is_stratified_60_20_20_and_disjoint() -> None:
    labels = np.repeat([0, 1], 50)
    train, validation, test = stratified_train_val_test_split(
        labels, np.random.default_rng(4)
    )
    assert (len(train), len(validation), len(test)) == (60, 20, 20)
    assert [np.bincount(labels[x]).tolist() for x in (train, validation, test)] == [
        [30, 30], [10, 10], [10, 10]
    ]
    assert len(np.unique(np.concatenate([train, validation, test]))) == 100


def test_epoch_rates_use_each_trials_own_duration() -> None:
    spikes = nap.TsGroup({7: nap.Ts(t=[0.2, 0.8, 2.2, 2.4, 2.6, 2.8])})
    trials = pd.DataFrame({"start": [0.0, 2.0], "end": [1.0, 3.0]})
    rates, valid = epoch_firing_rates(spikes, trials, "start", "end")
    assert valid.tolist() == [True, True]
    assert rates[:, 0].tolist() == [2.0, 4.0]


def test_summary_ranks_best_epoch() -> None:
    rows = []
    for participant in (1, 2):
        for epoch, score in zip(
            ("trial_start_to_stim", "stim_to_outcome", "outcome_to_trial_end"),
            (0.55, 0.8, 0.6),
        ):
            rows.append({"participant": participant, "repeat": 0, "epoch": epoch,
                         "balanced_accuracy": score})
    summary, comparisons = summarize_results(pd.DataFrame(rows))
    assert summary.iloc[0]["epoch"] == "stim_to_outcome"
    assert {"ci95_low", "ci95_high"}.issubset(summary.columns)
    assert (summary["ci95_low"] <= summary["mean"]).all()
    assert (summary["mean"] <= summary["ci95_high"]).all()
    assert len(comparisons) == 3
    assert {"difference_ci95_low", "difference_ci95_high"}.issubset(comparisons.columns)


def test_mean_metric_plot_contains_all_held_out_scores(tmp_path) -> None:
    rows = []
    for participant in (1, 2):
        for epoch in ("trial_start_to_stim", "stim_to_outcome", "outcome_to_trial_end"):
            rows.append(
                {
                    "participant": participant,
                    "repeat": 0,
                    "epoch": epoch,
                    "accuracy": 0.7,
                    "balanced_accuracy": 0.68,
                    "roc_auc": 0.75,
                }
            )
    path = plot_mean_decoding_metrics(pd.DataFrame(rows), tmp_path / "metrics.png")
    assert path.is_file()


def test_epoch_statistics_include_omnibus_and_corrected_pairwise_tests() -> None:
    rows = []
    for participant in range(1, 7):
        for epoch, score in zip(
            ("trial_start_to_stim", "stim_to_outcome", "outcome_to_trial_end"),
            (0.55, 0.80, 0.65),
        ):
            rows.append({"participant": participant, "epoch": epoch,
                         "balanced_accuracy": score + participant * 0.001})
    omnibus, pairwise = epoch_statistical_tests(pd.DataFrame(rows))
    assert omnibus.iloc[0]["test"] == "Friedman"
    assert omnibus.iloc[0]["n_participants"] == 6
    assert len(pairwise) == 3
    assert pairwise["p_value_holm"].between(0, 1).all()
