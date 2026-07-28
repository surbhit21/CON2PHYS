"""Compare mean firing rates between brain areas across all participants.

Run from the repository root with::

    python src/Q1.py

The participant, rather than each recorded neuron, is the statistical unit.
This avoids pseudoreplication and allows paired comparisons between areas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

if __package__:
    from .data_loader import available_subjects, load_session
    from .firing_rate import firing_rate_by_brain_area
    from .stat_test import holm_adjust
else:
    from data_loader import available_subjects, load_session
    from firing_rate import firing_rate_by_brain_area
    from stat_test import holm_adjust


RATE_COLUMN = "mean_unit_firing_rate_hz"


def load_all_subject_rates() -> pd.DataFrame:
    """Load spike-only data and return one firing-rate row per subject/area."""
    rates: dict[int, pd.DataFrame] = {}
    for subject in available_subjects():
        session = load_session(subject=subject, lfp_banks=())
        if session.spikes is None:
            continue
        rates[subject] = firing_rate_by_brain_area(session.spikes)
    if not rates:
        raise ValueError("no subject spike data were found")
    return pd.concat(rates, names=["subject", "brain_area"])


def summarize_firing_rates(rates: pd.DataFrame) -> pd.DataFrame:
    """Calculate across-subject means and Student-t 95% confidence intervals."""
    if RATE_COLUMN not in rates:
        raise ValueError(f"rates must contain {RATE_COLUMN!r}")
    grouped = rates[RATE_COLUMN].groupby(level="brain_area")
    summary = grouped.agg(n_subjects="count", mean_firing_rate_hz="mean", sd_hz="std")
    summary["sem_hz"] = summary["sd_hz"] / np.sqrt(summary["n_subjects"])
    critical_t = summary["n_subjects"].map(
        lambda count: stats.t.ppf(0.975, count - 1) if count > 1 else np.nan
    )
    margin = critical_t * summary["sem_hz"]
    summary["ci95_low_hz"] = summary["mean_firing_rate_hz"] - margin
    summary["ci95_high_hz"] = summary["mean_firing_rate_hz"] + margin
    return summary


def compare_brain_areas(
    rates: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, object]:
    """Test whether the lowest-mean area fires less than the other areas.

    Only participants with measurements in every area enter the paired tests.
    Returns the lowest area's summary row, the pairwise Wilcoxon results, and
    the SciPy Friedman test result.
    """
    summary = summarize_firing_rates(rates)
    lowest_area = summary["mean_firing_rate_hz"].idxmin()
    complete = rates[RATE_COLUMN].unstack("brain_area").dropna()
    if complete.shape[1] < 3:
        raise ValueError("the Friedman test requires at least three brain areas")
    if complete.shape[0] < 2:
        raise ValueError("at least two complete participants are required")

    friedman = stats.friedmanchisquare(
        *(complete[area].to_numpy() for area in complete.columns)
    )
    comparisons: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for other_area in complete.columns:
        if other_area == lowest_area:
            continue
        result = stats.wilcoxon(
            complete[lowest_area],
            complete[other_area],
            alternative="less",
        )
        raw_p_values.append(float(result.pvalue))
        comparisons.append(
            {
                "lowest_area": lowest_area,
                "comparison_area": other_area,
                "n_subjects": len(complete),
                "wilcoxon_statistic": float(result.statistic),
                "p_value_one_sided": float(result.pvalue),
            }
        )

    adjusted = holm_adjust(raw_p_values)
    for comparison, p_adjusted in zip(comparisons, adjusted):
        comparison["p_value_holm"] = p_adjusted
        comparison["significant_0.05"] = p_adjusted < 0.05
    return summary.loc[lowest_area], pd.DataFrame(comparisons), friedman


def main() -> None:
    rates = load_all_subject_rates()
    summary = summarize_firing_rates(rates)
    lowest, pairwise, friedman = compare_brain_areas(rates)

    print("\nMean firing rate by brain area (participant-level mean and 95% CI)")
    print(
        summary[
            ["n_subjects", "mean_firing_rate_hz", "ci95_low_hz", "ci95_high_hz"]
        ].to_string(float_format=lambda value: f"{value:.3f}")
    )
    print(
        f"\nLowest observed mean: brain area {lowest.name} "
        f"({lowest['mean_firing_rate_hz']:.3f} Hz)"
    )
    print(
        "Friedman omnibus test: "
        f"chi-square={friedman.statistic:.3f}, p={friedman.pvalue:.6g}"
    )
    print("\nOne-sided paired Wilcoxon tests (lowest area < comparison area)")
    print(pairwise.to_string(index=False, float_format=lambda value: f"{value:.6g}"))


if __name__ == "__main__":
    main()
