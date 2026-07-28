"""Estimate putative fast-spiking interneuron density from spike waveforms.

This analysis classifies extracellular units by trough-to-peak duration.
Waveform width is an operational marker, not definitive cell-type evidence,
so classified units are described as *putative* fast-spiking interneurons.

Run participant 1 from the repository root with::

    python src/Q7.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if __package__:
    from .data_loader import (
        CON2PHYSSession,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )
    from .stat_test import holm_adjust
else:
    from data_loader import (
        CON2PHYSSession,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )
    from stat_test import holm_adjust


PARTICIPANT = 1
WAVEFORM_SAMPLING_RATE = 30_000.0
FAST_SPIKING_THRESHOLD_MS = 0.4


def load_participant_waveforms(participant: int = PARTICIPANT) -> CON2PHYSSession:
    """Load waveforms and unit metadata without loading the large LFP arrays."""
    return load_session(subject=participant, lfp_banks=())


def waveform_features(
    session: CON2PHYSSession,
    *,
    sampling_rate: float = WAVEFORM_SAMPLING_RATE,
    threshold_ms: float = FAST_SPIKING_THRESHOLD_MS,
    baseline_samples: int = 10,
) -> pd.DataFrame:
    """Extract waveform width and classify putative fast-spiking units.

    Each waveform is baseline-corrected. Positive-dominant waveforms are
    inverted so that width is consistently measured from the negative trough
    to the following positive peak.
    """
    if session.waveforms is None or session.spikes is None:
        raise ValueError("session must contain waveforms and spike metadata")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if not np.isfinite(threshold_ms) or threshold_ms <= 0:
        raise ValueError("threshold_ms must be positive")

    values = session.waveforms.to_numpy(dtype=float, copy=True)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("waveforms must contain at least three samples per unit")
    if not 1 <= baseline_samples < values.shape[1]:
        raise ValueError("baseline_samples must be within the waveform")

    values -= values[:, :baseline_samples].mean(axis=1, keepdims=True)
    positive_dominant = np.abs(values.max(axis=1)) > np.abs(values.min(axis=1))
    values[positive_dominant] *= -1
    trough_sample = np.argmin(values, axis=1)

    peak_sample = np.full(len(values), -1, dtype=int)
    for row, trough in enumerate(trough_sample):
        if trough + 1 < values.shape[1]:
            peak_sample[row] = trough + 1 + np.argmax(values[row, trough + 1 :])

    width_samples = peak_sample - trough_sample
    valid = width_samples > 0
    width_ms = np.full(len(values), np.nan)
    width_ms[valid] = width_samples[valid] * 1_000.0 / sampling_rate

    unit_ids = session.waveforms.index
    metadata = session.spikes.metadata.reindex(unit_ids)
    if "brain_area" not in metadata or metadata["brain_area"].isna().any():
        raise ValueError("every waveform must have brain_area metadata")

    features = pd.DataFrame(
        {
            "brain_area": metadata["brain_area"].to_numpy(),
            "trough_sample": trough_sample,
            "peak_sample": peak_sample,
            "trough_to_peak_ms": width_ms,
            "putative_fast_spiking": valid & (width_ms <= threshold_ms),
        },
        index=unit_ids,
    )
    features.index.name = "cluster_id"
    return features


def fast_spiking_density(
    features: pd.DataFrame,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Calculate the proportion of putative fast-spiking units in each area."""
    required = {"brain_area", "putative_fast_spiking"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"features are missing columns: {sorted(missing)}")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")

    summary = features.groupby("brain_area", sort=True).agg(
        n_units=("putative_fast_spiking", "size"),
        n_putative_fast_spiking=("putative_fast_spiking", "sum"),
    )
    summary["density"] = summary["n_putative_fast_spiking"] / summary["n_units"]
    summary["density_percent"] = 100 * summary["density"]

    # Wilson intervals behave better than normal intervals for small groups,
    # including participant 1's seven units in brain area 3.
    z = stats.norm.ppf(0.5 + confidence / 2)
    n = summary["n_units"].astype(float)
    proportion = summary["density"]
    denominator = 1 + z**2 / n
    centre = (proportion + z**2 / (2 * n)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2))
        / denominator
    )
    summary["ci95_low"] = centre - margin
    summary["ci95_high"] = centre + margin
    return summary


def load_all_participant_densities(
    *,
    sampling_rate: float = WAVEFORM_SAMPLING_RATE,
    threshold_ms: float = FAST_SPIKING_THRESHOLD_MS,
) -> pd.DataFrame:
    """Calculate one putative fast-spiking density per participant and area."""
    results: list[pd.DataFrame] = []
    for participant in available_subjects():
        session = load_participant_waveforms(participant)
        features = waveform_features(
            session,
            sampling_rate=sampling_rate,
            threshold_ms=threshold_ms,
        )
        density = fast_spiking_density(features).reset_index()
        density.insert(0, "participant", participant)
        results.append(density)
    if not results:
        raise ValueError("no participant waveform data were found")
    return pd.concat(results, ignore_index=True)


def summarize_participant_densities(
    participant_density: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate across-participant mean density and Student-t 95% CIs."""
    required = {"participant", "brain_area", "density"}
    missing = required.difference(participant_density.columns)
    if missing:
        raise ValueError(f"participant_density is missing columns: {sorted(missing)}")
    grouped = participant_density.groupby("brain_area")["density"]
    summary = grouped.agg(n_participants="count", mean_density="mean", sd="std")
    summary["sem"] = summary["sd"] / np.sqrt(summary["n_participants"])
    critical_t = summary["n_participants"].map(
        lambda count: stats.t.ppf(0.975, count - 1) if count > 1 else np.nan
    )
    margin = critical_t * summary["sem"]
    summary["ci95_low"] = summary["mean_density"] - margin
    summary["ci95_high"] = summary["mean_density"] + margin
    summary["mean_density_percent"] = 100 * summary["mean_density"]
    summary["ci95_low_percent"] = 100 * summary["ci95_low"]
    summary["ci95_high_percent"] = 100 * summary["ci95_high"]
    return summary


def compare_area_densities(
    participant_density: pd.DataFrame,
) -> tuple[object, pd.DataFrame, object]:
    """Compare the highest-mean area with all other areas using paired tests."""
    summary = summarize_participant_densities(participant_density)
    highest_area = summary["mean_density"].idxmax()
    complete = participant_density.pivot(
        index="participant", columns="brain_area", values="density"
    ).dropna()
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
        if other_area == highest_area:
            continue
        result = stats.wilcoxon(
            complete[highest_area],
            complete[other_area],
            alternative="greater",
        )
        raw_p_values.append(float(result.pvalue))
        comparisons.append(
            {
                "highest_area": highest_area,
                "comparison_area": other_area,
                "n_participants": len(complete),
                "wilcoxon_statistic": float(result.statistic),
                "p_value_one_sided": float(result.pvalue),
            }
        )
    for comparison, adjusted in zip(comparisons, holm_adjust(raw_p_values)):
        comparison["p_value_holm"] = adjusted
        comparison["significant_0.05"] = adjusted < 0.05
    return highest_area, pd.DataFrame(comparisons), friedman


def plot_group_density(
    summary: pd.DataFrame,
    path: str | Path = DEFAULT_PLOT_PATH / "fast_spiking_density_all_participants.png",
) -> None:
    """Plot across-participant means and 95% confidence intervals."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    means = summary["mean_density_percent"].to_numpy()
    errors = np.vstack(
        [
            means - summary["ci95_low_percent"].to_numpy(),
            summary["ci95_high_percent"].to_numpy() - means,
        ]
    )
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.errorbar(
        summary.index.astype(str),
        means,
        yerr=errors,
        fmt="o",
        capsize=5,
        linewidth=1.5,
        color="tab:red",
    )
    axis.set(
        xlabel="Brain area",
        ylabel="Putative fast-spiking units (%)",
        title="Mean fast-spiking density across participants",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_waveform_analysis(
    session: CON2PHYSSession,
    features: pd.DataFrame,
    density: pd.DataFrame,
    path: str | Path = DEFAULT_PLOT_PATH / "participant_1_fast_spiking_waveforms.png",
    *,
    sampling_rate: float = WAVEFORM_SAMPLING_RATE,
) -> None:
    """Plot normalized unit waveforms and fast-spiking density by brain area."""
    if session.waveforms is None:
        raise ValueError("session must contain waveforms")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    areas = density.index.tolist()
    fig, axes = plt.subplots(1, len(areas) + 1, figsize=(4 * (len(areas) + 1), 4))
    time_ms = np.arange(session.waveforms.shape[1]) * 1_000 / sampling_rate

    for axis, area in zip(axes[:-1], areas):
        area_features = features[features["brain_area"] == area]
        for unit_id, row in area_features.iterrows():
            waveform = session.waveforms.loc[unit_id].to_numpy(dtype=float)
            waveform -= waveform[:10].mean()
            if abs(waveform.max()) > abs(waveform.min()):
                waveform *= -1
            scale = np.max(np.abs(waveform))
            if scale:
                waveform /= scale
            color = "tab:red" if row["putative_fast_spiking"] else "0.65"
            axis.plot(time_ms, waveform, color=color, alpha=0.55, linewidth=0.8)
        axis.set(
            title=f"Brain area {area}",
            xlabel="Waveform time (ms)",
            ylabel="Normalized amplitude",
        )

    bar_axis = axes[-1]
    proportions = density["density"].to_numpy()
    errors = np.vstack(
        [
            proportions - density["ci95_low"].to_numpy(),
            density["ci95_high"].to_numpy() - proportions,
        ]
    )
    bar_axis.bar(density.index.astype(str), proportions, color="tab:red", alpha=0.8)
    bar_axis.errorbar(
        density.index.astype(str),
        proportions,
        yerr=errors,
        fmt="none",
        color="black",
        capsize=4,
    )
    bar_axis.set(
        xlabel="Brain area",
        ylabel="Putative FS proportion",
        title="Fast-spiking density (95% CI)",
        ylim=(0, 1),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_waveform_examples(
    session: CON2PHYSSession,
    features: pd.DataFrame,
    path: str | Path = DEFAULT_PLOT_PATH / "participant_1_waveform_examples.png",
    *,
    sampling_rate: float = WAVEFORM_SAMPLING_RATE,
    examples_per_area: int = 4,
) -> None:
    """Plot representative individual waveforms from every brain area.

    Examples are selected at evenly spaced ranks of trough-to-peak width. This
    shows the narrow-to-broad range in each area without random selection.
    """
    if session.waveforms is None:
        raise ValueError("session must contain waveforms")
    if examples_per_area < 1:
        raise ValueError("examples_per_area must be at least one")

    areas = sorted(features["brain_area"].unique())
    selected: dict[object, pd.DataFrame] = {}
    for area in areas:
        candidates = (
            features[features["brain_area"] == area]
            .dropna(subset=["trough_to_peak_ms"])
            .sort_values(["trough_to_peak_ms", "trough_sample"])
        )
        count = min(examples_per_area, len(candidates))
        ranks = np.linspace(0, len(candidates) - 1, count).round().astype(int)
        selected[area] = candidates.iloc[np.unique(ranks)]

    max_examples = max(len(examples) for examples in selected.values())
    fig, axes = plt.subplots(
        len(areas),
        max_examples,
        figsize=(3.2 * max_examples, 2.7 * len(areas)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    time_ms = np.arange(session.waveforms.shape[1]) * 1_000 / sampling_rate
    for row_index, area in enumerate(areas):
        examples = selected[area]
        for column_index in range(max_examples):
            axis = axes[row_index, column_index]
            if column_index >= len(examples):
                axis.set_visible(False)
                continue
            unit_id = examples.index[column_index]
            feature = examples.iloc[column_index]
            waveform = session.waveforms.loc[unit_id].to_numpy(dtype=float)
            baseline_count = min(10, len(waveform) - 1)
            waveform -= waveform[:baseline_count].mean()
            if abs(waveform.max()) > abs(waveform.min()):
                waveform *= -1
            scale = np.max(np.abs(waveform))
            if scale:
                waveform /= scale
            is_fast = bool(feature["putative_fast_spiking"])
            axis.plot(
                time_ms,
                waveform,
                color="tab:red" if is_fast else "tab:blue",
                linewidth=1.5,
            )
            axis.axvline(
                feature["trough_sample"] * 1_000 / sampling_rate,
                color="0.5",
                linestyle=":",
                linewidth=0.8,
            )
            axis.axvline(
                feature["peak_sample"] * 1_000 / sampling_rate,
                color="0.5",
                linestyle=":",
                linewidth=0.8,
            )
            label = "putative FS" if is_fast else "broad"
            axis.set_title(
                f"Unit {unit_id}: {feature['trough_to_peak_ms']:.3f} ms\n{label}",
                fontsize=9,
            )
            if column_index == 0:
                axis.set_ylabel(f"Area {area}\nnormalized amplitude")
            if row_index == len(areas) - 1:
                axis.set_xlabel("Waveform time (ms)")

    fig.suptitle(
        "Representative waveforms spanning trough-to-peak widths",
        y=1.01,
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    participant_density = load_all_participant_densities()
    summary = summarize_participant_densities(participant_density)
    highest_area, pairwise, friedman = compare_area_densities(participant_density)

    DEFAULT_PLOT_PATH.mkdir(parents=True, exist_ok=True)
    participant_density.to_csv(
        DEFAULT_PLOT_PATH / "fast_spiking_density_by_participant.csv", index=False
    )
    summary.to_csv(DEFAULT_PLOT_PATH / "fast_spiking_density_group_summary.csv")
    pairwise.to_csv(
        DEFAULT_PLOT_PATH / "fast_spiking_density_pairwise_tests.csv", index=False
    )
    plot_group_density(summary)

    print(
        "\nPutative fast-spiking interneuron density across participants "
        f"(trough-to-peak <= {FAST_SPIKING_THRESHOLD_MS:g} ms)"
    )
    print(
        summary[
            [
                "n_participants",
                "mean_density_percent",
                "ci95_low_percent",
                "ci95_high_percent",
            ]
        ].to_string(float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nHighest observed mean density: brain area {highest_area}")
    print(
        "Friedman omnibus test: "
        f"chi-square={friedman.statistic:.6g}, p={friedman.pvalue:.6g}"
    )
    print("\nOne-sided paired Wilcoxon tests (highest area > comparison area)")
    print(pairwise.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(
        "\nClassification assumes waveforms were sampled at "
        f"{WAVEFORM_SAMPLING_RATE / 1_000:g} kHz."
    )


if __name__ == "__main__":
    main()
