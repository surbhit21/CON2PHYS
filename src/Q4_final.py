"""Run shift-corrected CCG analysis across all CON2PHYS participants."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if __package__:
    from .Q4_v3 import (
        CCG_BIN_SIZE_MS,
        CCG_MAX_LAG_MS,
        EVENT_COLUMNS,
        N_SHIFT_PREDICTORS,
        corrected_crosscorrelograms,
    )
    from .data_loader import available_subjects, load_session
    from .stat_test import holm_adjust
else:
    from Q4_v3 import (
        CCG_BIN_SIZE_MS,
        CCG_MAX_LAG_MS,
        EVENT_COLUMNS,
        N_SHIFT_PREDICTORS,
        corrected_crosscorrelograms,
    )
    from data_loader import available_subjects, load_session
    from stat_test import holm_adjust


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "analysis" / "Q4_all"


def participant_peak_metrics(
    participant: int,
    corrected: np.ndarray,
    pairs: list[tuple[int, int]],
    brain_areas: pd.Series,
    lags_ms: np.ndarray,
) -> pd.DataFrame:
    """Quantify within-area narrow peaks for one participant/checkpoint.

    Narrow peak height is the maximum mean corrected CCG within ±10 ms minus
    its shoulder baseline, defined as the mean at |lag| >= 50 ms.
    """
    pair_ids = np.asarray(pairs, dtype=int)
    pair_area_a = brain_areas.loc[pair_ids[:, 0]].to_numpy()
    pair_area_b = brain_areas.loc[pair_ids[:, 1]].to_numpy()
    near = np.abs(lags_ms) <= 10
    baseline = np.abs(lags_ms) >= 50
    rows: list[dict[str, object]] = []
    for event_index, event_name in enumerate(EVENT_COLUMNS):
        for area in sorted(brain_areas.unique()):
            mask = (pair_area_a == area) & (pair_area_b == area)
            if not mask.any():
                continue
            mean_ccg = np.nanmean(corrected[event_index][:, mask], axis=1)
            near_indices = np.flatnonzero(near)
            peak_local = int(np.nanargmax(mean_ccg[near]))
            peak_index = near_indices[peak_local]
            baseline_hz = float(np.nanmean(mean_ccg[baseline]))
            rows.append(
                {
                    "participant": participant,
                    "event_type": event_name,
                    "brain_area": int(area),
                    "n_units": int((brain_areas == area).sum()),
                    "n_pairs": int(mask.sum()),
                    "near_peak_lag_ms": float(lags_ms[peak_index]),
                    "near_peak_corrected_hz": float(mean_ccg[peak_index]),
                    "shoulder_baseline_hz": baseline_hz,
                    "narrow_peak_above_baseline_hz": float(
                        mean_ccg[peak_index] - baseline_hz
                    ),
                }
            )
    return pd.DataFrame(rows)


def analyze_participant(
    participant: int,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Run or load one participant's resumable corrected-CCG analysis."""
    participant_path = Path(output_path) / f"participant_{participant}"
    metrics_path = participant_path / "narrow_near_zero_peak_metrics.csv"
    if metrics_path.is_file() and not force:
        return pd.read_csv(metrics_path)

    session = load_session(participant, lfp_banks=())
    corrected, observed, predictor, pairs, peaks = corrected_crosscorrelograms(
        session
    )
    lags_ms = np.linspace(-CCG_MAX_LAG_MS, CCG_MAX_LAG_MS, corrected.shape[1])
    pair_ids = np.asarray(pairs, dtype=int)
    areas = session.spikes.metadata["brain_area"]
    pair_area_a = areas.loc[pair_ids[:, 0]].to_numpy()
    pair_area_b = areas.loc[pair_ids[:, 1]].to_numpy()
    participant_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        participant_path / "corrected_crosscorrelograms.npz",
        corrected_rate_hz=corrected,
        observed_rate_hz=observed,
        shift_predictor_rate_hz=predictor,
        event_types=np.asarray(list(EVENT_COLUMNS)),
        lags_ms=lags_ms,
        pair_unit_a=pair_ids[:, 0],
        pair_unit_b=pair_ids[:, 1],
        pair_brain_area_a=pair_area_a,
        pair_brain_area_b=pair_area_b,
        bin_size_ms=CCG_BIN_SIZE_MS,
        max_lag_ms=CCG_MAX_LAG_MS,
        n_shift_predictors=N_SHIFT_PREDICTORS,
    )
    peaks.insert(0, "participant", participant)
    peaks.insert(4, "brain_area_a", peaks["unit_a"].map(areas))
    peaks.insert(5, "brain_area_b", peaks["unit_b"].map(areas))
    peaks.to_csv(participant_path / "significant_lagged_peaks.csv", index=False)
    metrics = participant_peak_metrics(
        participant, corrected, pairs, areas, lags_ms
    )
    metrics.to_csv(metrics_path, index=False)
    return metrics


def summarize_across_participants(
    checkpoint_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Average checkpoints within participant and compare areas paired-wise."""
    participant_metrics = (
        checkpoint_metrics.groupby(["participant", "brain_area"], as_index=False)
        .agg(
            n_checkpoints=("event_type", "nunique"),
            mean_narrow_peak_hz=("narrow_peak_above_baseline_hz", "mean"),
            mean_zero_region_peak_hz=("near_peak_corrected_hz", "mean"),
        )
    )
    pivot = participant_metrics.pivot(
        index="participant", columns="brain_area", values="mean_narrow_peak_hz"
    ).dropna()
    if pivot.shape[1] < 3 or pivot.shape[0] < 2:
        raise ValueError("three areas and at least two complete participants are required")
    friedman = stats.friedmanchisquare(
        *(pivot[area].to_numpy() for area in pivot.columns)
    )
    comparisons = []
    raw_p = []
    for index, area_a in enumerate(pivot.columns):
        for area_b in pivot.columns[index + 1 :]:
            differences = pivot[area_a] - pivot[area_b]
            if np.allclose(differences, 0):
                statistic, p_value = 0.0, 1.0
            else:
                test = stats.wilcoxon(pivot[area_a], pivot[area_b])
                statistic, p_value = float(test.statistic), float(test.pvalue)
            raw_p.append(p_value)
            comparisons.append(
                {
                    "area_a": int(area_a),
                    "area_b": int(area_b),
                    "n_participants": len(pivot),
                    "median_difference_hz": float(np.median(differences)),
                    "wilcoxon_statistic": statistic,
                    "p_value_two_sided": p_value,
                }
            )
    adjusted = holm_adjust(raw_p)
    for row, value in zip(comparisons, adjusted):
        row["p_value_holm"] = value
        row["significant_0.05"] = value < 0.05
    return participant_metrics, pd.DataFrame(comparisons), friedman


def area_group_summary(participant_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return participant-level descriptive statistics and t confidence intervals."""
    rows = []
    for area, group in participant_metrics.groupby("brain_area", sort=True):
        values = group["mean_narrow_peak_hz"].to_numpy(dtype=float)
        sem = stats.sem(values)
        ci_low, ci_high = stats.t.interval(
            0.95, len(values) - 1, loc=values.mean(), scale=sem
        )
        rows.append(
            {
                "brain_area": int(area),
                "n_participants": len(values),
                "mean_narrow_peak_hz": values.mean(),
                "sd_hz": values.std(ddof=1),
                "median_narrow_peak_hz": np.median(values),
                "ci95_low_hz": ci_low,
                "ci95_high_hz": ci_high,
            }
        )
    return pd.DataFrame(rows)


def plot_participant_area_peaks(
    participant_metrics: pd.DataFrame,
    path: str | Path = DEFAULT_OUTPUT_PATH / "narrow_peak_by_brain_area.png",
) -> None:
    """Plot paired participant peak values and group mean 95% CIs."""
    pivot = participant_metrics.pivot(
        index="participant", columns="brain_area", values="mean_narrow_peak_hz"
    ).dropna()
    areas = list(pivot.columns)
    fig, ax = plt.subplots(figsize=(7, 5))
    for participant, values in pivot.iterrows():
        ax.plot(areas, values, color="0.75", linewidth=0.8, alpha=0.7)
        ax.scatter(areas, values, color="0.35", s=16, alpha=0.8)
    summary = area_group_summary(participant_metrics).set_index("brain_area")
    means = summary.loc[areas, "mean_narrow_peak_hz"].to_numpy()
    errors = np.vstack(
        [
            means - summary.loc[areas, "ci95_low_hz"].to_numpy(),
            summary.loc[areas, "ci95_high_hz"].to_numpy() - means,
        ]
    )
    ax.errorbar(
        areas,
        means,
        yerr=errors,
        fmt="o",
        color="tab:red",
        capsize=5,
        linewidth=2,
        markersize=7,
        label="Mean ± 95% CI",
    )
    ax.set(
        xticks=areas,
        xlabel="Brain area",
        ylabel="Narrow near-zero peak above shoulder baseline (Hz)",
        title=(
            "Within-area shift-corrected CCG peak across participants\n"
            "maximum within ±10 ms minus mean at |lag| ≥ 50 ms"
        ),
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute completed participants")
    parser.add_argument("--participants", nargs="+", type=int)
    args = parser.parse_args()
    participants = args.participants or available_subjects()
    results = []
    for position, participant in enumerate(participants, start=1):
        print(
            f"[{position}/{len(participants)}] participant {participant}",
            flush=True,
        )
        metrics = analyze_participant(participant, force=args.force)
        results.append(metrics)
        print(f"  completed ({len(metrics)} checkpoint-area rows)", flush=True)

    checkpoint_metrics = pd.concat(results, ignore_index=True)
    participant_metrics, comparisons, friedman = summarize_across_participants(
        checkpoint_metrics
    )
    DEFAULT_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    checkpoint_metrics.to_csv(
        DEFAULT_OUTPUT_PATH / "checkpoint_area_peak_metrics.csv", index=False
    )
    participant_metrics.to_csv(
        DEFAULT_OUTPUT_PATH / "participant_area_peak_metrics.csv", index=False
    )
    comparisons.to_csv(
        DEFAULT_OUTPUT_PATH / "area_pairwise_wilcoxon_tests.csv", index=False
    )
    pd.DataFrame(
        [{
            "n_participants": participant_metrics["participant"].nunique(),
            "friedman_statistic": friedman.statistic,
            "p_value": friedman.pvalue,
        }]
    ).to_csv(DEFAULT_OUTPUT_PATH / "area_friedman_test.csv", index=False)
    area_group_summary(participant_metrics).to_csv(
        DEFAULT_OUTPUT_PATH / "area_group_summary.csv", index=False
    )
    plot_participant_area_peaks(participant_metrics)
    print(
        f"Friedman chi-square={friedman.statistic:.6g}, p={friedman.pvalue:.6g}"
    )
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
