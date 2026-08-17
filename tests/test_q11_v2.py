"""Tests for the multi-region reduced-rank decoder."""

import numpy as np

from src.Q11_v2 import (
    MultiRegionRRR,
    Recording,
    SplitRecording,
    _mean_ci95,
    _safe_wilcoxon,
    available_devices,
    make_fold_splits,
    run_rrr_decoding,
)
import pytest


def test_fold_scaling_uses_training_trials_only() -> None:
    labels = np.repeat([0, 1], 10)
    counts = np.zeros((20, 2, 2), dtype=np.float32)
    counts[-1] = 1_000  # potential held-out outlier
    recordings = [Recording(1, region, counts.copy(), labels) for region in (1, 2, 3)]
    splits = make_fold_splits(recordings, fold=0, seed=3)
    for split in splits:
        assert len(split.y_train) == 16
        assert len(split.y_test) == 4
        assert np.isfinite(split.x_train).all()
        assert np.intersect1d(split.x_train.reshape(-1), [1_000]).size == 0


def test_zero_parallel_workers_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        run_rrr_decoding(n_jobs=0)


def test_rrr_factor_shapes_and_probabilities() -> None:
    model = MultiRegionRRR({(1, 1): 5, (2, 2): 3}, 4, rank=2, global_rank=2)
    assert tuple(model.U[model._key(1, 1)].shape) == (5, 2)
    assert model.A.shape == (3, 2, 2)
    assert model.B.shape == (2, 4, 2)
    probability = model.predict_proba(np.zeros((7, 4, 5)), 1, 1)
    assert probability.shape == (7, 2)
    assert np.allclose(probability.sum(axis=1), 1)


def test_cpu_device_can_be_selected_explicitly() -> None:
    assert available_devices("cpu") == ["cpu"]


def test_confidence_interval_edge_cases_do_not_error() -> None:
    empty = _mean_ci95([np.nan])
    singleton = _mean_ci95([0.6])
    regular = _mean_ci95([0.4, 0.6, 0.8])
    assert empty["n"] == 0 and np.isnan(empty["ci95_low"])
    assert singleton["n"] == 1 and np.isnan(singleton["ci95_high"])
    assert 0 <= regular["ci95_low"] <= regular["ci95_high"] <= 1
    statistic, pvalue, n = _safe_wilcoxon([0.0, 0.0, np.nan])
    assert (statistic, pvalue, n) == (0.0, 1.0, 2)


def test_rrr_learns_shared_binary_signal() -> None:
    rng = np.random.default_rng(12)
    recordings = []
    counts = {}
    for participant in (1, 2):
        for region in (1, 2, 3):
            n_neurons = region + 2
            y_train = np.repeat([0, 1], 30)
            y_test = np.repeat([0, 1], 10)
            x_train = rng.normal(0, 0.3, (60, 3, n_neurons)).astype(np.float32)
            x_test = rng.normal(0, 0.3, (20, 3, n_neurons)).astype(np.float32)
            x_train += y_train[:, None, None] * 1.5
            x_test += y_test[:, None, None] * 1.5
            recordings.append(
                SplitRecording(
                    participant, region, x_train, y_train, x_test, y_test
                )
            )
            counts[(participant, region)] = n_neurons
    model = MultiRegionRRR(counts, 3, rank=2, global_rank=2, seed=1)
    model.fit(recordings, epochs=80, learning_rate=0.03, batch_size=60, seed=2)
    accuracies = []
    for item in recordings:
        prediction = model.predict_proba(
            item.x_test, item.participant, item.region
        ).argmax(axis=1)
        accuracies.append(np.mean(prediction == item.y_test))
    assert np.mean(accuracies) > 0.95
