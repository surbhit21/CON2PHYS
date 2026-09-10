"""Estimate population dimensionality from participant 1 spike trains.

The analysis counts spikes from stimulus onset to outcome in overlapping
20-ms windows stepped every 5 ms (windows × neurons), then performs PCA
independently for each of the three brain regions.
Run from the repository root with::

    python src/Q13.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib
from scipy.stats import friedmanchisquare, linregress, ttest_rel
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .data_loader import (
        CON2PHYSSession,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )
else:
    from data_loader import (
        CON2PHYSSession,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )


PARTICIPANT = 1
BRAIN_REGIONS = (1, 2, 3)
BIN_SIZE_S = 0.020
STEP_SIZE_S = 0.005
VARIANCE_THRESHOLD = 0.95
TRAIN_FRACTION = 0.50
RANDOM_SEED = 13
N_SHUFFLES = 20
MANIFOLD_MAX_SAMPLES = 3000
MANIFOLD_N_NEIGHBORS = 15
DEFAULT_VARIANCE_PLOT = DEFAULT_PLOT_PATH / "Q13_pca_explained_variance.png"
DEFAULT_PARTICIPANT_RESULTS = DEFAULT_PLOT_PATH / "Q13_pca_by_participant.csv"
DEFAULT_GROUP_SUMMARY = DEFAULT_PLOT_PATH / "Q13_pca_group_summary.csv"
DEFAULT_GROUP_PLOT = DEFAULT_PLOT_PATH / "Q13_pca_group_summary.png"
DEFAULT_NEURON_COUNT_RESULTS = DEFAULT_PLOT_PATH / "Q13_pca_neuron_count_control.csv"
DEFAULT_MATCHED_RESULTS = DEFAULT_PLOT_PATH / "Q13_pca_matched_by_participant.csv"
DEFAULT_MATCHED_SUMMARY = DEFAULT_PLOT_PATH / "Q13_pca_matched_group_summary.csv"
DEFAULT_MATCHED_PLOT = DEFAULT_PLOT_PATH / "Q13_pca_matched_group_summary.png"
DEFAULT_PARTICIPATION_TESTS = (
    DEFAULT_PLOT_PATH / "Q13_pca_matched_participation_ratio_tests.csv"
)
DEFAULT_MANIFOLD_RESULTS = DEFAULT_PLOT_PATH / "Q13_manifold_comparison.csv"
DEFAULT_MANIFOLD_PLOT = DEFAULT_PLOT_PATH / "Q13_manifold_comparison.png"


def load_participant_1_spikes() -> CON2PHYSSession:
    """Load participant 1 without loading the large LFP arrays."""
    return load_session(PARTICIPANT, lfp_banks=(), load_spikes=True)


def load_participant_spikes(participant: int) -> CON2PHYSSession:
    """Load one participant without loading the large LFP arrays."""
    return load_session(participant, lfp_banks=(), load_spikes=True)


def sliding_spike_counts(
    spikes: nap.TsGroup,
    epochs: nap.IntervalSet,
    unit_ids: list[int],
    *,
    bin_size_s: float = BIN_SIZE_S,
    step_size_s: float = STEP_SIZE_S,
) -> np.ndarray:
    """Return spike counts with shape ``(windows, neurons)``.

    Windows start at stimulus onset, never cross an outcome boundary, and are
    half-open intervals ``[start, start + bin_size_s)``.  Consequently, a
    spike can contribute to multiple overlapping windows.
    """
    if not np.isfinite(bin_size_s) or bin_size_s <= 0:
        raise ValueError("bin_size_s must be positive")
    if not np.isfinite(step_size_s) or step_size_s <= 0:
        raise ValueError("step_size_s must be positive")
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
        duration = float(end) - float(start)
        if duration < bin_size_s:
            continue
        n_windows = int(np.floor((duration - bin_size_s) / step_size_s + 1e-10)) + 1
        window_starts = float(start) + np.arange(n_windows) * step_size_s
        window_ends = window_starts + bin_size_s
        trial_counts = np.empty((n_windows, len(unit_ids)), dtype=np.int64)
        for column, unit_id in enumerate(unit_ids):
            times = spike_times[unit_id]
            left = np.searchsorted(times, window_starts, side="left")
            right = np.searchsorted(times, window_ends, side="left")
            trial_counts[:, column] = right - left
        rows.append(trial_counts)

    if not rows:
        raise ValueError("the epochs do not contain any complete windows")
    return np.vstack(rows)


def binned_firing_rates(
    spikes: nap.TsGroup,
    epochs: nap.IntervalSet,
    unit_ids: list[int],
    *,
    bin_size_s: float = BIN_SIZE_S,
) -> np.ndarray:
    """Backward-compatible non-overlapping firing-rate helper."""
    counts = sliding_spike_counts(
        spikes, epochs, unit_ids, bin_size_s=bin_size_s, step_size_s=bin_size_s
    )
    return counts / bin_size_s


def normalized_firing_rates(
    train_counts: np.ndarray,
    test_counts: np.ndarray,
    *,
    bin_size_s: float = BIN_SIZE_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert counts to Hz and z-score neurons using training statistics.

    Neurons with no training-set variance are retained as zero-valued training
    features by assigning them unit scale. The returned mask records all
    finite input neurons retained in the matched population.
    """
    train = np.asarray(train_counts, dtype=float) / bin_size_s
    test = np.asarray(test_counts, dtype=float) / bin_size_s
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("train and test counts must be 2-D with equal neurons")
    if min(train.shape[0], test.shape[0]) < 2:
        raise ValueError("train and test must each contain at least two windows")

    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=1)
    retained = np.isfinite(mean) & np.isfinite(scale)
    if retained.sum() < 2:
        raise ValueError("fewer than two neurons have finite training data")
    safe_scale = scale[retained].copy()
    safe_scale[safe_scale == 0] = 1.0
    return (
        (train[:, retained] - mean[retained]) / safe_scale,
        (test[:, retained] - mean[retained]) / safe_scale,
        retained,
    )


def cross_validated_pca(
    train_rates: np.ndarray,
    test_rates: np.ndarray,
    *,
    threshold: float = VARIANCE_THRESHOLD,
) -> tuple[int, np.ndarray]:
    """Fit PCA on training rates and evaluate variance on held-out rates."""
    train = np.asarray(train_rates, dtype=float)
    test = np.asarray(test_rates, dtype=float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("train and test rates must be 2-D with equal neurons")
    if min(train.shape) < 2 or test.shape[0] < 2:
        raise ValueError("train and test data must contain at least two samples")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")

    train_centered = train - train.mean(axis=0, keepdims=True)
    covariance = train_centered.T @ train_centered
    _, components = np.linalg.eigh(covariance)
    components = components[:, ::-1].T

    # Center the held-out observations only for measuring their variability;
    # neither their mean nor variance is used when fitting or normalizing PCA.
    test_centered = test - test.mean(axis=0, keepdims=True)
    total_test_variance = np.square(test_centered).sum()
    if not np.isfinite(total_test_variance) or total_test_variance <= 0:
        raise ValueError("test firing rates have no variance")
    scores = test_centered @ components.T
    held_out_spectrum = np.square(scores).sum(axis=0) / total_test_variance
    cumulative = np.cumsum(held_out_spectrum)
    dimensions = int(np.searchsorted(cumulative, threshold, side="left") + 1)
    if dimensions > len(cumulative):
        raise ValueError("training PCA cannot explain the requested test variance")
    return dimensions, cumulative


def independently_shuffle_neurons(
    rates: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independently permute each neuron's windows to destroy correlations."""
    values = np.asarray(rates, dtype=float)
    shuffled = np.empty_like(values)
    for neuron in range(values.shape[1]):
        shuffled[:, neuron] = values[rng.permutation(values.shape[0]), neuron]
    return shuffled


def shuffled_pca_dimensions(
    train_rates: np.ndarray,
    test_rates: np.ndarray,
    *,
    threshold: float,
    n_shuffles: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return held-out dimensions and participation ratios for shuffle nulls."""
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be at least one")
    rng = np.random.default_rng(random_seed)
    dimensions = np.empty(n_shuffles, dtype=int)
    ratios = np.empty(n_shuffles, dtype=float)
    for repeat in range(n_shuffles):
        shuffled_train = independently_shuffle_neurons(train_rates, rng)
        shuffled_test = independently_shuffle_neurons(test_rates, rng)
        dimensions[repeat], cumulative = cross_validated_pca(
            shuffled_train, shuffled_test, threshold=threshold
        )
        spectrum = np.diff(np.concatenate(([0.0], cumulative)))
        ratios[repeat] = participation_ratio(spectrum)
    return dimensions, ratios


def split_epochs_half(
    epochs: nap.IntervalSet,
    *,
    random_seed: int,
) -> tuple[nap.IntervalSet, nap.IntervalSet]:
    """Randomly split whole trials into equal-sized train and test sets."""
    starts = np.asarray(epochs.start, dtype=float)
    ends = np.asarray(epochs.end, dtype=float)
    if len(starts) < 4:
        raise ValueError("at least four valid trials are required")
    order = np.random.default_rng(random_seed).permutation(len(starts))
    split = len(order) // 2
    train_indices = np.sort(order[:split])
    test_indices = np.sort(order[split:])
    return (
        nap.IntervalSet(
            start=starts[train_indices], end=ends[train_indices], time_units="s"
        ),
        nap.IntervalSet(
            start=starts[test_indices], end=ends[test_indices], time_units="s"
        ),
    )


def subsample_epochs(
    epochs: nap.IntervalSet,
    n_trials: int,
    *,
    random_seed: int,
) -> nap.IntervalSet:
    """Select a reproducible subset of complete trial epochs."""
    starts = np.asarray(epochs.start, dtype=float)
    ends = np.asarray(epochs.end, dtype=float)
    if not 4 <= n_trials <= len(starts):
        raise ValueError("n_trials must be between 4 and the number of valid trials")
    selected = np.sort(
        np.random.default_rng(random_seed).choice(
            len(starts), size=n_trials, replace=False
        )
    )
    return nap.IntervalSet(
        start=starts[selected], end=ends[selected], time_units="s"
    )


def stimulus_to_outcome_epochs(trial_table: pd.DataFrame) -> nap.IntervalSet:
    """Construct analysis epochs, omitting trials without a valid interval."""
    required = {"stim_start", "outcome"}
    missing = required.difference(trial_table.columns)
    if missing:
        raise ValueError(f"trial data is missing columns: {sorted(missing)}")

    starts = trial_table["stim_start"].to_numpy(dtype=float)
    ends = trial_table["outcome"].to_numpy(dtype=float)
    valid = np.isfinite(starts) & np.isfinite(ends) & (starts < ends)
    n_omitted = int((~valid).sum())
    if n_omitted:
        warnings.warn(
            f"omitting {n_omitted} trial(s) with missing or invalid "
            "stimulus-to-outcome times",
            RuntimeWarning,
            stacklevel=2,
        )
    if not np.any(valid):
        raise ValueError("no valid stimulus-to-outcome intervals were found")
    return nap.IntervalSet(start=starts[valid], end=ends[valid], time_units="s")


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
    *,
    threshold: float = VARIANCE_THRESHOLD,
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
        target_percent = threshold * 100
        axis.axhline(
            target_percent,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"{target_percent:g}%",
        )
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


def nonlinear_embedding_comparison(
    firing_rates: np.ndarray,
    *,
    max_samples: int = MANIFOLD_MAX_SAMPLES,
    random_seed: int = RANDOM_SEED,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compare 2-D PCA, UMAP, and t-SNE on identical sampled observations.

    Trustworthiness quantifies local-neighborhood preservation from the full
    neural space. UMAP and t-SNE have no explained-variance spectrum, so this
    complements rather than replaces PCA's 95%-variance dimensionality.
    """
    values = np.asarray(firing_rates, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("firing_rates must contain at least two bins and neurons")
    if not np.all(np.isfinite(values)):
        raise ValueError("firing_rates must be finite")
    if max_samples < 3:
        raise ValueError("max_samples must be at least three")

    rng = np.random.default_rng(random_seed)
    if len(values) > max_samples:
        sample = np.sort(rng.choice(len(values), size=max_samples, replace=False))
        values = values[sample]

    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "UMAP comparison requires 'umap-learn'; install project requirements"
        ) from error

    # sklearn's trustworthiness requires n_neighbors < n_samples / 2.
    n_neighbors = min(MANIFOLD_N_NEIGHBORS, max(1, (len(values) - 1) // 2))
    perplexity = min(30.0, max(1.0, (len(values) - 1) / 3.0))
    embeddings = {
        "PCA": PCA(n_components=2).fit_transform(values),
        "UMAP": umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=random_seed,
        ).fit_transform(values),
        "t-SNE": TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_seed,
        ).fit_transform(values),
    }
    scores = {
        method: float(trustworthiness(values, embedding, n_neighbors=n_neighbors))
        for method, embedding in embeddings.items()
    }
    return embeddings, scores


def participant_manifold_comparison(
    participant: int = PARTICIPANT,
    *,
    max_samples: int = MANIFOLD_MAX_SAMPLES,
    path=DEFAULT_MANIFOLD_PLOT,
) -> pd.DataFrame:
    """Plot PCA/UMAP/t-SNE embeddings and score them for each brain region."""
    session = load_participant_spikes(participant)
    if session.spikes is None:
        raise ValueError(f"participant {participant} has no spike data")
    metadata = session.spikes.metadata
    epochs = stimulus_to_outcome_epochs(session.trial_table)
    train_epochs, test_epochs = split_epochs_half(
        epochs, random_seed=RANDOM_SEED + participant
    )
    figure, axes = plt.subplots(
        len(BRAIN_REGIONS), 3, figsize=(12, 11), constrained_layout=True
    )
    records: list[dict[str, float | int | str]] = []

    for row, region in enumerate(BRAIN_REGIONS):
        unit_ids = metadata.index[metadata["brain_area"] == region].astype(int).tolist()
        train_counts = sliding_spike_counts(session.spikes, train_epochs, unit_ids)
        test_counts = sliding_spike_counts(session.spikes, test_epochs, unit_ids)
        train_rates, test_rates, retained = normalized_firing_rates(
            train_counts, test_counts
        )
        rates = np.vstack((train_rates, test_rates))
        embeddings, scores = nonlinear_embedding_comparison(
            rates,
            max_samples=max_samples,
            random_seed=RANDOM_SEED + participant * 100 + region,
        )
        for column, method in enumerate(("PCA", "UMAP", "t-SNE")):
            embedding = embeddings[method]
            axes[row, column].scatter(
                embedding[:, 0], embedding[:, 1], s=3, alpha=0.35,
                c=np.arange(len(embedding)), cmap="viridis", rasterized=True,
            )
            axes[row, column].set_title(
                f"Region {region}: {method}\ntrustworthiness={scores[method]:.3f}"
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            records.append(
                {
                    "participant": participant,
                    "brain_region": region,
                    "method": method,
                    "n_neurons_used": int(retained.sum()),
                    "n_samples": len(embedding),
                    "trustworthiness": scores[method],
                }
            )

    figure.suptitle(
        "Linear and nonlinear embeddings (colour follows sampled-window order)"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame.from_records(records)


def participant_pca(
    participant: int,
    *,
    bin_size_s: float = BIN_SIZE_S,
    step_size_s: float = STEP_SIZE_S,
    threshold: float = VARIANCE_THRESHOLD,
    n_shuffles: int = N_SHUFFLES,
    n_neurons: int | None = None,
    n_trials: int | None = None,
    match_neurons_within_participant: bool = False,
    make_plot: bool = False,
) -> pd.DataFrame:
    """Run PCA for one participant, separately for each brain region."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    session = load_participant_spikes(participant)
    if session.spikes is None:
        raise ValueError(f"participant {participant} has no spike data")

    metadata = session.spikes.metadata
    if "brain_area" not in metadata:
        raise ValueError("spike metadata does not contain 'brain_area'")
    analysis_epochs = stimulus_to_outcome_epochs(session.trial_table)
    if n_trials is not None:
        analysis_epochs = subsample_epochs(
            analysis_epochs,
            n_trials,
            random_seed=RANDOM_SEED + participant * 10,
        )
    train_epochs, test_epochs = split_epochs_half(
        analysis_epochs, random_seed=RANDOM_SEED + participant
    )

    records: list[dict[str, float | int]] = []
    spectra: dict[int, np.ndarray] = {}
    if match_neurons_within_participant:
        n_neurons = min(
            int((metadata["brain_area"] == region).sum())
            for region in BRAIN_REGIONS
        )
    for region in BRAIN_REGIONS:
        unit_ids = metadata.index[metadata["brain_area"] == region].astype(int).tolist()
        if n_neurons is not None:
            if len(unit_ids) < n_neurons:
                raise ValueError(
                    f"participant {participant}, region {region} has fewer than "
                    f"{n_neurons} neurons"
                )
            unit_ids = np.random.default_rng(
                RANDOM_SEED + participant * 100 + region
            ).choice(unit_ids, size=n_neurons, replace=False).tolist()
        train_counts = sliding_spike_counts(
            session.spikes,
            train_epochs,
            unit_ids,
            bin_size_s=bin_size_s,
            step_size_s=step_size_s,
        )
        test_counts = sliding_spike_counts(
            session.spikes,
            test_epochs,
            unit_ids,
            bin_size_s=bin_size_s,
            step_size_s=step_size_s,
        )
        train_rates, test_rates, retained = normalized_firing_rates(
            train_counts, test_counts, bin_size_s=bin_size_s
        )
        dimensions, cumulative = cross_validated_pca(
            train_rates, test_rates, threshold=threshold
        )
        null_dimensions, null_ratios = shuffled_pca_dimensions(
            train_rates,
            test_rates,
            threshold=threshold,
            n_shuffles=n_shuffles,
            random_seed=RANDOM_SEED + participant * 100 + region,
        )
        spectrum = np.diff(np.concatenate(([0.0], cumulative)))
        observed_ratio = participation_ratio(spectrum)
        spectra[region] = spectrum
        records.append(
            {
                "participant": participant,
                "brain_region": region,
                "n_neurons": len(unit_ids),
                "n_neurons_used": int(retained.sum()),
                "n_train_trials": len(train_epochs),
                "n_test_trials": len(test_epochs),
                "n_train_time_bins": train_rates.shape[0],
                "n_test_time_bins": test_rates.shape[0],
                "dimensions_for_95pct": dimensions,
                "dimension_fraction": dimensions / int(retained.sum()),
                "variance_explained": cumulative[dimensions - 1],
                "participation_ratio": observed_ratio,
                "null_mean_components": float(null_dimensions.mean()),
                "null_mean_participation_ratio": float(null_ratios.mean()),
                "null_ci95_low": float(np.quantile(null_dimensions, 0.025)),
                "null_ci95_high": float(np.quantile(null_dimensions, 0.975)),
                "observed_minus_null": dimensions - float(null_dimensions.mean()),
            }
        )
    if make_plot:
        plot_variance_spectra(spectra, threshold=threshold)
    return pd.DataFrame.from_records(records)


def participant_1_pca(
    *,
    bin_size_s: float = BIN_SIZE_S,
    step_size_s: float = STEP_SIZE_S,
    threshold: float = VARIANCE_THRESHOLD,
    n_shuffles: int = N_SHUFFLES,
) -> pd.DataFrame:
    """Backward-compatible participant 1 analysis, including its PCA plot."""
    return participant_pca(
        PARTICIPANT,
        bin_size_s=bin_size_s,
        step_size_s=step_size_s,
        threshold=threshold,
        n_shuffles=n_shuffles,
        make_plot=True,
    )


def summarize_component_counts(results: pd.DataFrame) -> pd.DataFrame:
    """Return the mean and two-sided Student-t 95% CI across participants."""
    required = {
        "participant",
        "brain_region",
        "n_neurons_used",
        "dimensions_for_95pct",
        "dimension_fraction",
        "participation_ratio",
        "null_mean_components",
        "null_mean_participation_ratio",
        "observed_minus_null",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results are missing columns: {sorted(missing)}")

    summary = results.groupby("brain_region", as_index=False).agg(
        n_participants=("participant", "nunique"),
        mean_neurons=("n_neurons_used", "mean"),
        mean_components=("dimensions_for_95pct", "mean"),
        sd_components=("dimensions_for_95pct", "std"),
        mean_dimension_fraction=("dimension_fraction", "mean"),
        mean_participation_ratio=("participation_ratio", "mean"),
        sd_participation_ratio=("participation_ratio", "std"),
        mean_null_components=("null_mean_components", "mean"),
        mean_null_participation_ratio=(
            "null_mean_participation_ratio",
            "mean",
        ),
        sd_null_components=("null_mean_components", "std"),
        mean_observed_minus_null=("observed_minus_null", "mean"),
        sd_observed_minus_null=("observed_minus_null", "std"),
    )
    summary["sem_components"] = (
        summary["sd_components"] / np.sqrt(summary["n_participants"])
    )
    critical_t = student_t.ppf(0.975, summary["n_participants"] - 1)
    margin = critical_t * summary["sem_components"]
    summary["ci95_low"] = summary["mean_components"] - margin
    summary["ci95_high"] = summary["mean_components"] + margin
    participation_sem = (
        summary["sd_participation_ratio"] / np.sqrt(summary["n_participants"])
    )
    participation_margin = critical_t * participation_sem
    summary["participation_ratio_ci95_low"] = (
        summary["mean_participation_ratio"] - participation_margin
    )
    summary["participation_ratio_ci95_high"] = (
        summary["mean_participation_ratio"] + participation_margin
    )
    summary["sem_null_components"] = (
        summary["sd_null_components"] / np.sqrt(summary["n_participants"])
    )
    null_margin = critical_t * summary["sem_null_components"]
    summary["null_ci95_low"] = summary["mean_null_components"] - null_margin
    summary["null_ci95_high"] = summary["mean_null_components"] + null_margin
    difference_sem = (
        summary["sd_observed_minus_null"] / np.sqrt(summary["n_participants"])
    )
    difference_margin = critical_t * difference_sem
    summary["difference_ci95_low"] = (
        summary["mean_observed_minus_null"] - difference_margin
    )
    summary["difference_ci95_high"] = (
        summary["mean_observed_minus_null"] + difference_margin
    )
    p_values = {
        region: ttest_1samp(group["observed_minus_null"], popmean=0.0).pvalue
        for region, group in results.groupby("brain_region")
    }
    summary["observed_vs_null_p_value"] = summary["brain_region"].map(p_values)
    return summary


def neuron_count_control(results: pd.DataFrame) -> pd.DataFrame:
    """Quantify the within-region relationship between neurons and dimension."""
    records = []
    for region, group in results.groupby("brain_region"):
        regression = linregress(
            group["n_neurons_used"].to_numpy(dtype=float),
            group["dimensions_for_95pct"].to_numpy(dtype=float),
        )
        records.append(
            {
                "brain_region": region,
                "n_participants": len(group),
                "pearson_r": regression.rvalue,
                "p_value": regression.pvalue,
                "r_squared": regression.rvalue**2,
                "components_per_additional_neuron": regression.slope,
                "intercept": regression.intercept,
            }
        )
    return pd.DataFrame.from_records(records)


def participation_ratio_tests(results: pd.DataFrame) -> pd.DataFrame:
    """Run paired omnibus and Holm-corrected regional participation-ratio tests."""
    pivot = results.pivot(
        index="participant", columns="brain_region", values="participation_ratio"
    ).dropna()
    regions = sorted(pivot.columns)
    if len(regions) != len(BRAIN_REGIONS) or len(pivot) < 2:
        raise ValueError("paired tests require all regions for at least two participants")

    omnibus = friedmanchisquare(*(pivot[region] for region in regions))
    records: list[dict[str, float | int | str]] = [
        {
            "test": "Friedman omnibus",
            "region_a": "all",
            "region_b": "all",
            "n_participants": len(pivot),
            "statistic": omnibus.statistic,
            "mean_difference": np.nan,
            "difference_ci95_low": np.nan,
            "difference_ci95_high": np.nan,
            "p_value": omnibus.pvalue,
            "p_value_holm": omnibus.pvalue,
        }
    ]
    pairwise = []
    for index, region_a in enumerate(regions):
        for region_b in regions[index + 1 :]:
            differences = (pivot[region_a] - pivot[region_b]).to_numpy(dtype=float)
            test = ttest_rel(pivot[region_a], pivot[region_b])
            sem = differences.std(ddof=1) / np.sqrt(len(differences))
            margin = student_t.ppf(0.975, len(differences) - 1) * sem
            pairwise.append(
                {
                    "test": "paired t-test",
                    "region_a": region_a,
                    "region_b": region_b,
                    "n_participants": len(differences),
                    "statistic": test.statistic,
                    "mean_difference": differences.mean(),
                    "difference_ci95_low": differences.mean() - margin,
                    "difference_ci95_high": differences.mean() + margin,
                    "p_value": test.pvalue,
                }
            )

    order = np.argsort([record["p_value"] for record in pairwise])
    adjusted = np.empty(len(pairwise), dtype=float)
    running_max = 0.0
    for rank, location in enumerate(order):
        value = min(1.0, (len(pairwise) - rank) * pairwise[location]["p_value"])
        running_max = max(running_max, value)
        adjusted[location] = running_max
    for record, corrected in zip(pairwise, adjusted):
        record["p_value_holm"] = corrected
    records.extend(pairwise)
    return pd.DataFrame.from_records(records)


def plot_group_component_counts(
    summary: pd.DataFrame,
    path=DEFAULT_GROUP_PLOT,
) -> None:
    """Plot mean PCA component counts and 95% CIs by brain region."""
    required = {
        "brain_region",
        "mean_components",
        "ci95_low",
        "ci95_high",
        "mean_null_components",
        "null_ci95_low",
        "null_ci95_high",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")

    ordered = summary.sort_values("brain_region")
    means = ordered["mean_components"].to_numpy(dtype=float)
    lower = ordered["ci95_low"].to_numpy(dtype=float)
    upper = ordered["ci95_high"].to_numpy(dtype=float)
    errors = np.vstack((means - lower, upper - means))
    null_means = ordered["mean_null_components"].to_numpy(dtype=float)
    null_lower = ordered["null_ci95_low"].to_numpy(dtype=float)
    null_upper = ordered["null_ci95_high"].to_numpy(dtype=float)
    null_errors = np.vstack((null_means - null_lower, null_upper - null_means))
    positions = np.arange(len(ordered))
    width = 0.36

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(
        positions - width / 2,
        means,
        width=width,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.7,
        yerr=errors,
        capsize=6,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
        label="Observed",
    )
    axis.bar(
        positions + width / 2,
        null_means,
        width=width,
        color="#F58518",
        edgecolor="black",
        linewidth=0.7,
        yerr=null_errors,
        capsize=6,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
        label="Independent-neuron shuffle",
    )
    axis.set_xticks(
        positions,
        [f"Brain region {region}" for region in ordered["brain_region"]],
    )
    axis.set_ylabel("Components explaining 95% of variance")
    axis.set_title("Population dimensionality across participants\nMean and 95% CI")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def all_participants_pca(
    *,
    bin_size_s: float = BIN_SIZE_S,
    step_size_s: float = STEP_SIZE_S,
    threshold: float = VARIANCE_THRESHOLD,
    n_shuffles: int = N_SHUFFLES,
    match_neurons_and_trials: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze all available participants and return raw and group results."""
    participants = available_subjects()
    if not participants:
        raise ValueError("no participants were found")

    if match_neurons_and_trials:
        print(
            "Participant-level matching: equal neuron counts across regions "
            "within each participant; all valid participant trials retained",
            flush=True,
        )

    participant_results = []
    for position, participant in enumerate(participants, start=1):
        print(f"[{position}/{len(participants)}] participant {participant}", flush=True)
        participant_results.append(
            participant_pca(
                participant,
                bin_size_s=bin_size_s,
                step_size_s=step_size_s,
                threshold=threshold,
                n_shuffles=n_shuffles,
                match_neurons_within_participant=match_neurons_and_trials,
            )
        )
    results = pd.concat(participant_results, ignore_index=True)
    summary = summarize_component_counts(results)
    DEFAULT_PLOT_PATH.mkdir(parents=True, exist_ok=True)
    if match_neurons_and_trials:
        results.to_csv(DEFAULT_MATCHED_RESULTS, index=False)
        summary.to_csv(DEFAULT_MATCHED_SUMMARY, index=False)
        participation_ratio_tests(results).to_csv(
            DEFAULT_PARTICIPATION_TESTS, index=False
        )
        plot_group_component_counts(summary, path=DEFAULT_MATCHED_PLOT)
    else:
        results.to_csv(DEFAULT_PARTICIPANT_RESULTS, index=False)
        summary.to_csv(DEFAULT_GROUP_SUMMARY, index=False)
        neuron_count_control(results).to_csv(
            DEFAULT_NEURON_COUNT_RESULTS, index=False
        )
        plot_group_component_counts(summary)
    return results, summary


def main() -> None:
    _, summary = all_participants_pca(match_neurons_and_trials=True)
    manifold_results = participant_manifold_comparison()
    manifold_results.to_csv(DEFAULT_MANIFOLD_RESULTS, index=False)
    print(
        f"\nCross-validated PCA of training-normalized firing rates in "
        f"{BIN_SIZE_S * 1000:g}-ms windows stepped by "
        f"{STEP_SIZE_S * 1000:g} ms from stimulus onset to outcome "
        f"(participant-level neuron/trial matched; 50/50 trial split; "
        f"held-out target={VARIANCE_THRESHOLD:.0%})"
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nMatched participant results: {DEFAULT_MATCHED_RESULTS}")
    print(f"Matched group summary: {DEFAULT_MATCHED_SUMMARY}")
    print(f"Matched group plot: {DEFAULT_MATCHED_PLOT}")
    print(f"Participation-ratio tests: {DEFAULT_PARTICIPATION_TESTS}")
    print(f"PCA/UMAP/t-SNE comparison: {DEFAULT_MANIFOLD_RESULTS}")
    print(f"PCA/UMAP/t-SNE plot: {DEFAULT_MANIFOLD_PLOT}")


if __name__ == "__main__":
    main()
