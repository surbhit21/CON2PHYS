"""Tests for time-resolved regional decoding."""

import numpy as np
import pynapple as nap
import pytest

from src.Q11 import (
    WINDOW_SIZE_S,
    decode_all_participants,
    event_spike_counts,
    l2_logistic_accuracy,
    l2_logistic_scores,
    plot_decoding_accuracy,
    stratified_folds,
    window_starts,
)


def test_windows_have_requested_width_step_and_extent() -> None:
    starts = window_starts()
    assert len(starts) == 56
    assert np.isclose(starts[0], -0.2)
    assert np.isclose(starts[-1], 0.9)
    assert np.isclose(starts[-1] + WINDOW_SIZE_S, 1.0)
    assert np.allclose(np.diff(starts), 0.02)
    assert WINDOW_SIZE_S == 0.1


def test_zero_parallel_workers_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        decode_all_participants(n_jobs=0)


def test_event_spike_counts_keeps_neurons_separate() -> None:
    spikes = nap.TsGroup(
        {
            4: nap.Ts(t=[0.95, 1.01, 1.08, 2.03]),
            9: nap.Ts(t=[1.04, 2.01, 2.09]),
        }
    )
    counts, unit_ids = event_spike_counts(
        spikes, np.array([1.0, 2.0]), starts=np.array([0.0])
    )
    assert unit_ids == [4, 9]
    assert counts[:, 0, :].tolist() == [[2, 1], [1, 2]]


def test_stratified_five_fold_is_an_80_20_partition() -> None:
    labels = np.repeat([0, 1], 50)
    folds = stratified_folds(labels, 5, np.random.default_rng(2))
    assert len(folds) == 5
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(100))
    for train, test in folds:
        assert len(train) == 80
        assert len(test) == 20
        assert np.bincount(labels[test]).tolist() == [10, 10]


def test_l2_logistic_regression_decodes_separable_data() -> None:
    rng = np.random.default_rng(4)
    labels = np.repeat([0, 1], 50)
    values = labels[:, None] * 3.0 + rng.normal(0, 0.2, size=(100, 2))
    train, test = stratified_folds(labels, 5, rng)[0]
    accuracy = l2_logistic_accuracy(
        values[train], labels[train], values[test], labels[test]
    )
    assert accuracy == 1.0


def test_decoder_returns_balanced_accuracy_and_auc() -> None:
    rng = np.random.default_rng(8)
    labels = np.repeat([0, 1], 50)
    values = labels[:, None] * 4.0 + rng.normal(0, 0.1, size=(100, 2))
    train, test = stratified_folds(labels, 5, rng)[0]
    scores = l2_logistic_scores(
        values[train], labels[train], values[test], labels[test]
    )
    assert scores == {"accuracy": 1.0, "balanced_accuracy": 1.0, "roc_auc": 1.0}


def test_decoder_does_not_modify_held_out_data() -> None:
    x_train = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1, 1])
    x_test = np.array([[-1.5], [1.5]])
    y_test = np.array([0, 1])
    original_test = x_test.copy()
    l2_logistic_accuracy(x_train, y_train, x_test, y_test)
    assert np.array_equal(x_test, original_test)


def test_accuracy_plot_overlays_regions(tmp_path) -> None:
    import pandas as pd

    rows = []
    for participant in (1, 2):
        for event in ("stim_start", "outcome"):
            for region in (1, 2, 3):
                for center in (-0.15, 0.05):
                    for is_null, accuracy in ((False, 0.6), (True, 0.5)):
                        rows.append(
                            {"participant": participant, "event": event,
                             "region": region, "window_center_s": center,
                             "is_null": is_null, "accuracy": accuracy}
                        )
    path = plot_decoding_accuracy(pd.DataFrame(rows), tmp_path / "accuracy.png")
    assert path.is_file()
