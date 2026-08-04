"""Compare undirected cross-region corrected CCG peaks across participants."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if __package__:
    from .Q4_final import DEFAULT_OUTPUT_PATH as INPUT_PATH
    from .stat_test import holm_adjust
else:
    from CON2PHYS.src.Q4_final import DEFAULT_OUTPUT_PATH as INPUT_PATH
    from stat_test import holm_adjust


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "analysis" / "Q4_cross_region"
CROSS_REGION_PAIRS = ((1, 2), (1, 3), (2, 3))


def cross_region_peak_metrics(
    input_path: str | Path = INPUT_PATH,
) -> pd.DataFrame:
    """Extract participant/checkpoint undirected cross-region peak strengths."""
    rows = []
    for participant in range(1, 19):
        data = np.load(
            Path(input_path)
            / f"participant_{participant}"
            / "corrected_crosscorrelograms.npz"
        )
        corrected = data["corrected_rate_hz"]
        lags = data["lags_ms"]
        area_a = data["pair_brain_area_a"]
        area_b = data["pair_brain_area_b"]
        near = np.abs(lags) <= 10
        shoulder = np.abs(lags) >= 50
        for event_index, event_name in enumerate(data["event_types"]):
            for left_area, right_area in CROSS_REGION_PAIRS:
                pair_mask = (
                    ((area_a == left_area) & (area_b == right_area))
                    | ((area_a == right_area) & (area_b == left_area))
                )
                mean_ccg = np.nanmean(
                    corrected[event_index][:, pair_mask], axis=1
                )
                near_indices = np.flatnonzero(near)
                peak_index = near_indices[int(np.nanargmax(mean_ccg[near]))]
                baseline = float(np.nanmean(mean_ccg[shoulder]))
                rows.append(
                    {
                        "participant": participant,
                        "event_type": str(event_name),
                        "area_a": left_area,
                        "area_b": right_area,
                        "connection": f"Area {left_area}–{right_area}",
                        "n_pairs": int(pair_mask.sum()),
                        "peak_lag_ms": float(lags[peak_index]),
                        "near_peak_corrected_hz": float(mean_ccg[peak_index]),
                        "shoulder_baseline_hz": baseline,
                        "undirected_near_zero_mean_hz": float(
                            np.nanmean(mean_ccg[near]) - baseline
                        ),
                    }
                )
    return pd.DataFrame(rows)


def cross_region_statistics(
    participant_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Test each cross-region connection against zero and compare classes."""
    against_zero = []
    raw_p = []
    for connection, group in participant_metrics.groupby("connection", sort=True):
        values = group["mean_undirected_coupling_hz"].to_numpy(dtype=float)
        test = stats.wilcoxon(values, alternative="greater")
        sem = stats.sem(values)
        ci_low, ci_high = stats.t.interval(
            0.95, len(values) - 1, loc=values.mean(), scale=sem
        )
        raw_p.append(float(test.pvalue))
        against_zero.append(
            {
                "connection": connection,
                "n_participants": len(values),
                "n_positive": int((values > 0).sum()),
                "mean_coupling_hz": values.mean(),
                "median_coupling_hz": np.median(values),
                "ci95_low_hz": ci_low,
                "ci95_high_hz": ci_high,
                "wilcoxon_statistic": float(test.statistic),
                "p_value_one_sided": float(test.pvalue),
            }
        )
    for row, adjusted in zip(against_zero, holm_adjust(raw_p)):
        row["p_value_holm"] = adjusted
        row["significant_0.05"] = adjusted < 0.05

    pivot = participant_metrics.pivot(
        index="participant", columns="connection", values="mean_undirected_coupling_hz"
    ).dropna()
    friedman = stats.friedmanchisquare(*(pivot[column] for column in pivot.columns))
    between = []
    between_raw_p = []
    columns = list(pivot.columns)
    for index, first in enumerate(columns):
        for second in columns[index + 1 :]:
            difference = pivot[first] - pivot[second]
            test = stats.wilcoxon(pivot[first], pivot[second])
            between_raw_p.append(float(test.pvalue))
            between.append(
                {
                    "connection_a": first,
                    "connection_b": second,
                    "median_difference_hz": np.median(difference),
                    "wilcoxon_statistic": float(test.statistic),
                    "p_value_two_sided": float(test.pvalue),
                }
            )
    for row, adjusted in zip(between, holm_adjust(between_raw_p)):
        row["p_value_holm"] = adjusted
        row["significant_0.05"] = adjusted < 0.05
    return pd.DataFrame(against_zero), pd.DataFrame(between), friedman


def plot_cross_region_peaks(
    participant_metrics: pd.DataFrame,
    path: str | Path = OUTPUT_PATH / "cross_region_narrow_peaks.png",
) -> None:
    """Plot paired participant cross-region coupling strengths."""
    order = [f"Area {a}–{b}" for a, b in CROSS_REGION_PAIRS]
    pivot = participant_metrics.pivot(
        index="participant", columns="connection", values="mean_undirected_coupling_hz"
    )[order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, values in pivot.iterrows():
        ax.plot(x, values, color="0.75", linewidth=0.8, alpha=0.7)
        ax.scatter(x, values, color="0.35", s=16)
    means = pivot.mean().to_numpy()
    sem = pivot.sem().to_numpy()
    critical = stats.t.ppf(0.975, len(pivot) - 1)
    ax.errorbar(
        x,
        means,
        yerr=critical * sem,
        fmt="o",
        color="tab:red",
        capsize=5,
        linewidth=2,
        label="Mean ± 95% CI",
    )
    ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
    ax.set(
        xticks=x,
        xticklabels=order,
        ylabel="Cross-region near-zero coupling (Hz)",
        title=(
            "Undirected cross-region shift-corrected coupling\n"
            "mean within ±10 ms minus |lag| ≥ 50 ms baseline"
        ),
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    checkpoint_metrics = cross_region_peak_metrics()
    participant_metrics = (
        checkpoint_metrics.groupby(
            ["participant", "area_a", "area_b", "connection"], as_index=False
        )
        .agg(
            n_checkpoints=("event_type", "nunique"),
            mean_undirected_coupling_hz=("undirected_near_zero_mean_hz", "mean"),
            mean_peak_lag_ms=("peak_lag_ms", "mean"),
        )
    )
    against_zero, between, friedman = cross_region_statistics(participant_metrics)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    checkpoint_metrics.to_csv(
        OUTPUT_PATH / "checkpoint_cross_region_metrics.csv", index=False
    )
    participant_metrics.to_csv(
        OUTPUT_PATH / "participant_cross_region_metrics.csv", index=False
    )
    against_zero.to_csv(OUTPUT_PATH / "cross_region_tests_against_zero.csv", index=False)
    between.to_csv(OUTPUT_PATH / "cross_region_pairwise_comparisons.csv", index=False)
    pd.DataFrame(
        [{
            "friedman_statistic": friedman.statistic,
            "p_value": friedman.pvalue,
            "n_participants": participant_metrics["participant"].nunique(),
        }]
    ).to_csv(OUTPUT_PATH / "cross_region_friedman_test.csv", index=False)
    plot_cross_region_peaks(participant_metrics)
    print("\nCross-region tests against zero")
    print(against_zero.to_string(index=False))
    print(
        f"\nCross-region class comparison: Friedman "
        f"chi-square={friedman.statistic:.6g}, p={friedman.pvalue:.6g}"
    )
    print(between.to_string(index=False))


if __name__ == "__main__":
    main()
