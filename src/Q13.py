"""Estimate population dimensionality from participant 1 spike trains.

The analysis bins spikes from stimulus onset to outcome into firing-rate
vectors (time bins × neurons), then performs PCA independently for each of the
three brain regions.
Run from the repository root with::

    python src/Q13.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .data_loader import CON2PHYSSession, DEFAULT_PLOT_PATH, load_session
else:
    from data_loader import CON2PHYSSession, DEFAULT_PLOT_PATH, load_session


PARTICIPANT = 1
BRAIN_REGIONS = (1, 2, 3)
BIN_SIZE_S = 1.0
VARIANCE_THRESHOLD = 0.90
DEFAULT_VARIANCE_PLOT = DEFAULT_PLOT_PATH / "Q13_pca_explained_variance.png"


def load_participant_1_spikes() -> CON2PHYSSession:
    """Load participant 1 without loading the large LFP arrays."""
    return load_session(PARTICIPANT, lfp_banks=(), load_spikes=True)


def binned_firing_rates(
    spikes: nap.TsGroup,
    epochs: nap.IntervalSet,
    unit_ids: list[int],
    *,
    bin_size_s: float = BIN_SIZE_S,
) -> np.ndarray:
    """Return epoch-only firing rates with shape ``(time bins, neurons)``.

    Bins never cross an epoch boundary. A final partial bin is omitted from
    each epoch so every spike count is divided by the same duration.
    """
    if not np.isfinite(bin_size_s) or bin_size_s <= 0:
        raise ValueError("bin_size_s must be positive")
    if not unit_ids:
        raise ValueError("at least one neuron is required")

    starts = np.asarray(epochs.start)
    ends = np.asarray(epochs.end)
    rows: list[np.ndarray] = []
    spike_times = {
        unit_id: np.asarray(spikes[unit_id].index, dtype=float)
        for unit_id in unit_ids
    }

    for start, end in zip(starts, ends):
        n_bins = int(np.floor((float(end) - float(start)) / bin_size_s))
        if n_bins < 1:
            continue
        edges = float(start) + np.arange(n_bins + 1) * bin_size_s
        trial_rates = np.column_stack(
            [np.histogram(spike_times[unit_id], bins=edges)[0] for unit_id in unit_ids]
        )
        rows.append(trial_rates / bin_size_s)

    if not rows:
        raise ValueError("the epochs do not contain any complete time bins")
    return np.vstack(rows)


def stimulus_to_outcome_epochs(trial_table: pd.DataFrame) -> nap.IntervalSet:
    """Construct one analysis epoch from stimulus onset to each outcome."""
    required = {"stim_start", "outcome"}
    missing = required.difference(trial_table.columns)
    if missing:
        raise ValueError(f"trial data is missing columns: {sorted(missing)}")

    starts = trial_table["stim_start"].to_numpy(dtype=float)
    ends = trial_table["outcome"].to_numpy(dtype=float)
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("stim_start and outcome must be finite")
    if np.any(starts >= ends):
        raise ValueError("every stim_start must occur before outcome")
    return nap.IntervalSet(start=starts, end=ends, time_units="s")


def dimensions_for_variance(
    firing_rates: np.ndarray,
    *,
    threshold: float = VARIANCE_THRESHOLD,
) -> tuple[int, np.ndarray]:
    """Return the PCA dimension count and cumulative explained variance."""
    values = np.asarray(firing_rates, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("firing_rates must contain at least two bins and neurons")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")

    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    explained_variance = singular_values**2
    total_variance = explained_variance.sum()
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise ValueError("firing rates have no variance")

    cumulative = np.cumsum(explained_variance / total_variance)
    dimensions = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    return dimensions, cumulative


def pca_variance_spectrum(firing_rates: np.ndarray) -> np.ndarray:
    """Return explained-variance ratios ordered from largest to smallest PC."""
    values = np.asarray(firing_rates, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("firing_rates must contain at least two bins and neurons")
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = singular_values**2
    total = eigenvalues.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("firing rates have no variance")
    return eigenvalues / total


def participation_ratio(explained_variance_ratio: np.ndarray) -> float:
    """Return effective dimensionality from a normalized PCA spectrum."""
    spectrum = np.asarray(explained_variance_ratio, dtype=float)
    if spectrum.ndim != 1 or spectrum.size == 0:
        raise ValueError("explained_variance_ratio must be a non-empty vector")
    if np.any(spectrum < 0) or not np.all(np.isfinite(spectrum)):
        raise ValueError("explained_variance_ratio must be finite and non-negative")
    total = spectrum.sum()
    if total <= 0:
        raise ValueError("explained_variance_ratio must have positive mass")
    return float(total**2 / np.square(spectrum).sum())


def plot_variance_spectra(
    spectra: dict[int, np.ndarray],
    path=DEFAULT_VARIANCE_PLOT,
) -> None:
    """Plot sorted per-PC and cumulative explained variance by brain region."""
    figure, axes = plt.subplots(1, len(spectra), figsize=(15, 4.5), sharey=True)
    if len(spectra) == 1:
        axes = [axes]

    for axis, (region, spectrum) in zip(axes, sorted(spectra.items())):
        components = np.arange(1, len(spectrum) + 1)
        cumulative = np.cumsum(spectrum)
        axis.bar(
            components,
            spectrum * 100,
            color="#4C78A8",
            alpha=0.8,
            label="Individual PC",
        )
        axis.plot(
            components,
            cumulative * 100,
            color="#E45756",
            linewidth=2,
            label="Cumulative",
        )
        axis.axhline(90, color="black", linestyle="--", linewidth=1, label="90%")
        axis.set_title(f"Brain region {region}")
        axis.set_xlabel("Principal component (sorted)")
        axis.set_ylim(0, 102)
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Explained variance (%)")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def participant_1_pca(
    *,
    bin_size_s: float = BIN_SIZE_S,
    threshold: float = VARIANCE_THRESHOLD,
) -> pd.DataFrame:
    """Run PCA per brain region and return its population dimensionality."""
    session = load_participant_1_spikes()
    if session.spikes is None:
        raise ValueError("participant 1 has no spike data")

    metadata = session.spikes.metadata
    if "brain_area" not in metadata:
        raise ValueError("spike metadata does not contain 'brain_area'")
    analysis_epochs = stimulus_to_outcome_epochs(session.trial_table)

    records: list[dict[str, float | int]] = []
    spectra: dict[int, np.ndarray] = {}
    for region in BRAIN_REGIONS:
        unit_ids = metadata.index[metadata["brain_area"] == region].astype(int).tolist()
        rates = binned_firing_rates(
            session.spikes, analysis_epochs, unit_ids, bin_size_s=bin_size_s
        )
        spectrum = pca_variance_spectrum(rates)
        cumulative = np.cumsum(spectrum)
        dimensions = int(np.searchsorted(cumulative, threshold, side="left") + 1)
        spectra[region] = spectrum
        records.append(
            {
                "brain_region": region,
                "n_neurons": len(unit_ids),
                "n_time_bins": rates.shape[0],
                "dimensions_for_90pct": dimensions,
                "variance_explained": cumulative[dimensions - 1],
                "participation_ratio": participation_ratio(spectrum),
            }
        )
    plot_variance_spectra(spectra)
    return pd.DataFrame.from_records(records)


def main() -> None:
    results = participant_1_pca()
    print(
        f"Participant {PARTICIPANT}: PCA of {BIN_SIZE_S:g}-s firing rates from "
        f"stimulus onset to outcome "
        f"(target={VARIANCE_THRESHOLD:.0%})"
    )
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nExplained-variance plot: {DEFAULT_VARIANCE_PLOT}")


if __name__ == "__main__":
    main()
