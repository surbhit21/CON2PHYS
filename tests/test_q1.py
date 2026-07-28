"""Tests for the across-participant firing-rate statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.Q1 import compare_brain_areas, summarize_firing_rates


def _example_rates() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [[1, 2, 3, 4], [1, 2, 3]], names=["subject", "brain_area"]
    )
    # Area 2 is consistently lower for every participant.
    values = [5, 1, 8, 6, 2, 9, 7, 3, 10, 8, 4, 11]
    return pd.DataFrame(
        {"mean_unit_firing_rate_hz": values},
        index=index,
    )


def test_summary_contains_mean_and_t_confidence_interval() -> None:
    summary = summarize_firing_rates(_example_rates())

    assert summary.loc[2, "n_subjects"] == 4
    assert summary.loc[2, "mean_firing_rate_hz"] == 2.5
    assert summary.loc[2, "ci95_low_hz"] < 2.5
    assert summary.loc[2, "ci95_high_hz"] > 2.5


def test_comparison_identifies_lowest_area() -> None:
    lowest, pairwise, friedman = compare_brain_areas(_example_rates())

    assert lowest.name == 2
    assert pairwise["lowest_area"].tolist() == [2, 2]
    assert pairwise["comparison_area"].tolist() == [1, 3]
    assert np.isfinite(pairwise["p_value_holm"]).all()
    assert friedman.statistic > 0
