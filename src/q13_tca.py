"""Reproducible PCA and CP/PARAFAC analysis of CON2PHYS spike trains."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorly as tl
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from tensorly.cp_tensor import cp_normalize, cp_to_tensor
from tensorly.decomposition import non_negative_parafac, parafac

from .data_loader import DEFAULT_DATA_PATH, load_session

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisConfig:
    data_path: str = str(DEFAULT_DATA_PATH)
    output_dir: str = "analysis/Q13"
    participant: int = 1
    bin_width_ms: int = 100
    event_alignment: str = "stim_start_to_outcome"
    max_rank: int = 10
    n_cv_folds: int = 3
    n_initializations: int = 5
    validation_fraction: float = 0.10
    random_seed: int = 42
    use_nonnegative_tca: bool = True
    preprocessing: str = "sqrt"
    smoothing_sigma_bins: float = 0.0
    max_iterations: int = 100
    tolerance: float = 1e-6
    n_bootstraps: int = 3


@dataclass
class TensorData:
    counts: np.ndarray
    valid_mask: np.ndarray
    unit_ids: np.ndarray
    brain_areas: np.ndarray
    time_ms: np.ndarray
    trial_metadata: pd.DataFrame
    excluded_trials: list[dict[str, Any]]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_jsonable) + "\n")


def build_spike_count_tensor(config: AnalysisConfig) -> TensorData:
    """Bin timestamps into neurons × time × trials from stimulus to outcome.

    Variable-duration trials are right-padded with zeros and accompanied by a
    Boolean validity mask. Padding is never treated as observed activity.
    Trials whose outcome is outside their trial interval are explicitly
    excluded as invalid metadata.
    """
    session = load_session(
        config.participant, config.data_path, lfp_banks=(), load_spikes=True
    )
    if session.spikes is None:
        raise ValueError("the participant contains no spike trains")
    table = session.trial_table.copy()
    required = {"trial_start", "stim_start", "outcome", "trial_end"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"trial metadata is missing {sorted(missing)}")

    timing = table[list(required)].to_numpy(dtype=float)
    valid = np.isfinite(timing).all(axis=1)
    valid &= table["trial_start"].to_numpy() <= table["stim_start"].to_numpy()
    valid &= table["stim_start"].to_numpy() < table["outcome"].to_numpy()
    valid &= table["outcome"].to_numpy() <= table["trial_end"].to_numpy()
    excluded = [
        {"trial": int(index), "reason": "invalid event ordering"}
        for index in table.index[~valid]
    ]
    if excluded:
        LOGGER.warning("excluding %d trials with invalid event timing", len(excluded))
    table = table.loc[valid].copy()
    if table.empty:
        raise ValueError("no trials have valid stimulus-to-outcome timing")

    bin_s = config.bin_width_ms / 1000.0
    durations = (table["outcome"] - table["stim_start"]).to_numpy()
    n_bins_by_trial = np.maximum(1, np.ceil(durations / bin_s).astype(int))
    n_time = int(n_bins_by_trial.max())
    metadata = session.spikes.metadata
    unit_ids = metadata.index.to_numpy(dtype=int)
    brain_areas = metadata["brain_area"].to_numpy()
    counts = np.zeros((len(unit_ids), n_time, len(table)), dtype=np.float32)
    mask = np.zeros_like(counts, dtype=bool)

    for trial_position, ((_, trial), n_bins) in enumerate(
        zip(table.iterrows(), n_bins_by_trial)
    ):
        edges = float(trial["stim_start"]) + np.arange(n_bins + 1) * bin_s
        edges[-1] = float(trial["outcome"])
        if edges[-1] <= edges[-2]:
            edges = edges[:-1]
            n_bins -= 1
        for neuron_position, unit_id in enumerate(unit_ids):
            times = np.asarray(session.spikes[int(unit_id)].index)
            counts[neuron_position, :n_bins, trial_position] = np.histogram(
                times, bins=edges
            )[0]
        mask[:, :n_bins, trial_position] = True

    return TensorData(
        counts=counts,
        valid_mask=mask,
        unit_ids=unit_ids,
        brain_areas=brain_areas,
        time_ms=np.arange(n_time) * config.bin_width_ms,
        trial_metadata=table,
        excluded_trials=excluded,
    )


def preprocess_tensor(
    counts: np.ndarray,
    mask: np.ndarray,
    method: str,
    smoothing_sigma_bins: float,
    nonnegative: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Validate, filter empty neurons, transform, and optionally smooth."""
    if counts.shape != mask.shape or counts.ndim != 3:
        raise ValueError("counts and mask must be equal neurons × time × trials arrays")
    observed = counts[mask]
    if not np.isfinite(observed).all():
        raise ValueError("observed tensor entries contain NaN or infinity")
    if np.any(observed < 0):
        raise ValueError("spike counts cannot be negative")
    totals = np.sum(counts * mask, axis=(1, 2))
    keep = totals > 0
    if not keep.any():
        raise ValueError("all neurons are empty")
    values = counts[keep].astype(np.float64)
    output_mask = mask[keep]
    allowed = {"none", "sqrt", "log1p", "zscore_per_neuron"}
    if method not in allowed:
        raise ValueError(f"preprocessing must be one of {sorted(allowed)}")
    if method == "sqrt":
        values = np.sqrt(values)
    elif method == "log1p":
        values = np.log1p(values)
    elif method == "zscore_per_neuron":
        if nonnegative:
            raise ValueError("zscore_per_neuron is incompatible with non-negative TCA")
        for neuron in range(values.shape[0]):
            selected = output_mask[neuron]
            mean = values[neuron][selected].mean()
            std = values[neuron][selected].std()
            values[neuron][selected] = (
                values[neuron][selected] - mean
            ) / (std if std > 0 else 1.0)
    if smoothing_sigma_bins > 0:
        numerator = gaussian_filter1d(values * output_mask, smoothing_sigma_bins, axis=1)
        denominator = gaussian_filter1d(
            output_mask.astype(float), smoothing_sigma_bins, axis=1
        )
        values = np.divide(
            numerator, denominator, out=np.zeros_like(values), where=denominator > 0
        )
    values[~output_mask] = 0
    trial_totals = np.sum(counts * mask, axis=(0, 1))
    summary = {
        "input_shape": list(counts.shape),
        "final_shape": list(values.shape),
        "removed_empty_neuron_indices": np.flatnonzero(~keep).tolist(),
        "n_removed_empty_neurons": int((~keep).sum()),
        "preprocessing": method,
        "smoothing_sigma_bins": smoothing_sigma_bins,
        "fraction_zero_observed": float(np.mean(observed == 0)),
        "neuron_total_activity_outliers_iqr": _iqr_outliers(totals).tolist(),
        "trial_total_activity_outliers_iqr": _iqr_outliers(trial_totals).tolist(),
    }
    return values, keep, summary


def _iqr_outliers(values: np.ndarray) -> np.ndarray:
    q1, q3 = np.quantile(values, [0.25, 0.75])
    spread = q3 - q1
    return np.flatnonzero((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread))


def pca_baseline(tensor: np.ndarray, mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Run centered PCA on valid trial–time observations × neurons."""
    valid_observations = mask.all(axis=0)
    samples = tensor[:, valid_observations].T
    samples -= samples.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(samples, full_matrices=False, compute_uv=False)
    eigenvalues = singular**2 / max(1, samples.shape[0] - 1)
    ratios = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(ratios)
    table = pd.DataFrame(
        {
            "component": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance": cumulative,
        }
    )
    metrics = {
        f"d{int(level * 100)}": int(np.searchsorted(cumulative, level) + 1)
        for level in (0.8, 0.9, 0.95)
    }
    metrics["participation_ratio"] = float(
        eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()
    )
    metrics["n_samples"] = int(samples.shape[0])
    metrics["n_neurons"] = int(samples.shape[1])
    return table, metrics


def fit_cp(
    tensor: np.ndarray,
    mask: np.ndarray,
    rank: int,
    seed: int,
    config: AnalysisConfig,
) -> tuple[Any, list[float], bool]:
    """Fit mask-aware CP without using unobserved entries as targets."""
    solver = non_negative_parafac if config.use_nonnegative_tca else parafac
    # TensorLy uses masked positions during its initial missing-value
    # imputation even though they are excluded from the weighted objective.
    # Canonicalising them prevents held-out target values from leaking into
    # initialization; the mask continues to distinguish them from real zeros.
    fitting_tensor = np.where(mask, tensor, 0.0)
    model, errors = solver(
        fitting_tensor,
        rank=rank,
        mask=mask.astype(float),
        init="random",
        random_state=seed,
        n_iter_max=config.max_iterations,
        tol=config.tolerance,
        normalize_factors=False,
        return_errors=True,
    )
    model = cp_normalize(model)
    converged = len(errors) < config.max_iterations
    return model, [float(value) for value in errors], converged


def reconstruction_metrics(
    truth: np.ndarray, prediction: np.ndarray, evaluation_mask: np.ndarray
) -> dict[str, float]:
    residual = truth[evaluation_mask] - prediction[evaluation_mask]
    rmse = float(np.sqrt(np.mean(residual**2)))
    scale = float(np.std(truth[evaluation_mask]))
    variance = float(np.var(truth[evaluation_mask]))
    return {
        "rmse": rmse,
        "normalized_rmse": rmse / scale if scale > 0 else np.nan,
        "explained_variance": 1 - float(np.var(residual)) / variance
        if variance > 0
        else np.nan,
    }


def make_cv_masks(
    valid_mask: np.ndarray, folds: int, fraction: float, seed: int
) -> list[np.ndarray]:
    """Mask observed entries while retaining observations in every mode fiber."""
    rng = np.random.default_rng(seed)
    valid_indices = np.argwhere(valid_mask)
    outputs: list[np.ndarray] = []
    for _ in range(folds):
        holdout = np.zeros_like(valid_mask)
        order = rng.permutation(len(valid_indices))
        target = int(round(fraction * len(valid_indices)))
        mode_counts = [
            valid_mask.sum(axis=(1, 2)).astype(int),
            valid_mask.sum(axis=(0, 2)).astype(int),
            valid_mask.sum(axis=(0, 1)).astype(int),
        ]
        for position in order:
            if holdout.sum() >= target:
                break
            index = tuple(valid_indices[position])
            if all(mode_counts[mode][index[mode]] > 1 for mode in range(3)):
                holdout[index] = True
                for mode in range(3):
                    mode_counts[mode][index[mode]] -= 1
        outputs.append(holdout)
    return outputs


def factor_match_score(reference: Any, candidate: Any) -> float:
    """Match CP components with mean cosine similarity across all three modes."""
    ref = cp_normalize(reference).factors
    other = cp_normalize(candidate).factors
    rank = ref[0].shape[1]
    similarities = np.zeros((rank, rank))
    for left in range(rank):
        for right in range(rank):
            mode_scores = []
            for ref_factor, candidate_factor in zip(ref, other):
                a, b = ref_factor[:, left], candidate_factor[:, right]
                cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15)
                mode_scores.append(abs(float(cosine)))
            similarities[left, right] = np.mean(mode_scores)
    rows, columns = linear_sum_assignment(-similarities)
    return float(similarities[rows, columns].mean())


def run_tca(
    tensor: np.ndarray, valid_mask: np.ndarray, config: AnalysisConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, Any], dict[int, list[Any]]]:
    """Fit ranks, cross-validate masked entries, and quantify multi-start stability."""
    initialization_rows, cv_rows, stability_rows = [], [], []
    best_models: dict[int, Any] = {}
    all_models: dict[int, list[Any]] = {}
    cv_masks = make_cv_masks(
        valid_mask, config.n_cv_folds, config.validation_fraction, config.random_seed
    )
    for rank in range(1, config.max_rank + 1):
        LOGGER.info("fitting rank %d", rank)
        models: list[Any] = []
        losses: list[float] = []
        for initialization in range(config.n_initializations):
            seed = config.random_seed + rank * 1000 + initialization
            model, errors, converged = fit_cp(tensor, valid_mask, rank, seed, config)
            prediction = cp_to_tensor(model)
            loss = reconstruction_metrics(tensor, prediction, valid_mask)["rmse"]
            models.append(model)
            losses.append(loss)
            initialization_rows.append(
                {
                    "rank": rank,
                    "initialization": initialization,
                    "seed": seed,
                    "training_rmse": loss,
                    "iterations": len(errors),
                    "converged": converged,
                    "final_relative_error": errors[-1],
                }
            )
        best_models[rank] = models[int(np.argmin(losses))]
        all_models[rank] = models
        scores = [
            factor_match_score(left, right)
            for left, right in combinations(models, 2)
        ]
        stability_rows.append(
            {
                "rank": rank,
                "stability_type": "initializations",
                "mean_stability": float(np.mean(scores)) if scores else 1.0,
                "std_stability": float(np.std(scores)) if scores else 0.0,
                "n_comparisons": len(scores),
            }
        )
        for fold, holdout in enumerate(cv_masks):
            training_mask = valid_mask & ~holdout
            fold_metrics = []
            for initialization in range(config.n_initializations):
                seed = config.random_seed + rank * 10000 + fold * 100 + initialization
                model, errors, converged = fit_cp(
                    tensor, training_mask, rank, seed, config
                )
                metrics = reconstruction_metrics(tensor, cp_to_tensor(model), holdout)
                fold_metrics.append((metrics["rmse"], metrics, seed, len(errors), converged))
            _, metrics, seed, iterations, converged = min(fold_metrics)
            cv_rows.append(
                {
                    "rank": rank,
                    "fold": fold,
                    **{f"validation_{key}": value for key, value in metrics.items()},
                    "best_seed": seed,
                    "iterations": iterations,
                    "converged": converged,
                    "n_validation_entries": int(holdout.sum()),
                }
            )
    return (
        pd.DataFrame(cv_rows),
        pd.DataFrame(initialization_rows),
        pd.DataFrame(stability_rows),
        best_models,
        all_models,
    )


def select_rank(cv: pd.DataFrame, stability: pd.DataFrame) -> dict[str, Any]:
    grouped = cv.groupby("rank")["validation_rmse"].agg(["mean", "std", "count"])
    grouped["se"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"])
    best_rank = int(grouped["mean"].idxmin())
    cutoff = grouped.loc[best_rank, "mean"] + grouped.loc[best_rank, "se"]
    cv_rank = int(grouped.index[grouped["mean"] <= cutoff].min())
    init_stability = stability[stability["stability_type"] == "initializations"]
    stable_rank = int(init_stability.loc[init_stability["mean_stability"].idxmax(), "rank"])
    candidate_stability = float(
        init_stability.set_index("rank").loc[cv_rank, "mean_stability"]
    )
    recommended = cv_rank
    reasoning = "smallest rank within one SE of the best validation RMSE"
    if candidate_stability < 0.80:
        eligible = init_stability[init_stability["rank"].isin(grouped.index[grouped["mean"] <= cutoff])]
        recommended = int(eligible.loc[eligible["mean_stability"].idxmax(), "rank"])
        reasoning += "; adjusted within the one-SE set because the CV rank was unstable"
    return {
        "cross_validated_rank": cv_rank,
        "best_mean_cv_rank": best_rank,
        "most_stable_rank": stable_rank,
        "final_recommended_rank": recommended,
        "recommendation_reason": reasoning,
        "one_se_rmse_cutoff": float(cutoff),
    }


def bootstrap_stability(
    tensor: np.ndarray,
    mask: np.ndarray,
    rank: int,
    reference: Any,
    config: AnalysisConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed + 99999)
    rows = []
    for bootstrap in range(config.n_bootstraps):
        indices = rng.integers(0, tensor.shape[2], tensor.shape[2])
        model, errors, converged = fit_cp(
            tensor[:, :, indices],
            mask[:, :, indices],
            rank,
            config.random_seed + 100000 + bootstrap,
            config,
        )
        # Trial factors cannot be compared after resampling; compare neuron/time.
        ref_short = (reference.weights, reference.factors[:2])
        model_short = (model.weights, model.factors[:2])
        score = factor_match_score(ref_short, model_short)
        rows.append(
            {
                "rank": rank,
                "stability_type": "trial_bootstrap",
                "bootstrap": bootstrap,
                "mean_stability": score,
                "iterations": len(errors),
                "converged": converged,
            }
        )
    return pd.DataFrame(rows)


def save_pca_plots(table: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["component"], table["eigenvalue"], marker="o", ms=3)
    ax.set(xlabel="Principal component", ylabel="Eigenvalue", title="PCA scree plot")
    fig.tight_layout()
    fig.savefig(output / "pca_scree_plot.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["component"], 100 * table["cumulative_explained_variance"])
    for level in (80, 90, 95):
        ax.axhline(level, ls="--", lw=1)
    ax.set(xlabel="Principal component", ylabel="Cumulative variance (%)")
    fig.tight_layout()
    fig.savefig(output / "pca_cumulative_variance.png", dpi=200)
    plt.close(fig)


def save_qc_plots(data: TensorData, tensor: np.ndarray, mask: np.ndarray, output: Path) -> None:
    plots = {
        "qc_mean_activity_per_neuron": (
            np.divide((tensor * mask).sum((1, 2)), mask.sum((1, 2))),
            "Neuron", "Mean transformed activity",
        ),
        "qc_total_activity_per_trial": (
            (tensor * mask).sum((0, 1)), "Trial", "Total transformed activity",
        ),
        "qc_population_activity_over_time": (
            np.divide((tensor * mask).sum((0, 2)), mask.sum((0, 2)), where=mask.sum((0, 2)) > 0),
            "Time bin", "Population mean",
        ),
    }
    for name, (values, xlabel, ylabel) in plots.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(values)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=160)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    average = np.divide(
        (tensor * mask).sum(axis=2),
        mask.sum(axis=2),
        out=np.zeros(tensor.shape[:2]),
        where=mask.sum(axis=2) > 0,
    )
    image = ax.imshow(average, aspect="auto", interpolation="nearest")
    ax.set(xlabel="Time bin", ylabel="Neuron", title="Trial-averaged activity")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "qc_trial_averaged_heatmap.png", dpi=160)
    fig.savefig(output / "qc_trial_averaged_heatmap.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for neuron in range(min(5, tensor.shape[0])):
        ax.plot(average[neuron], label=f"neuron {data.unit_ids[neuron]}")
    ax.set(xlabel="Time bin", ylabel="Mean activity", title="Example neuron traces")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "qc_example_neuron_traces.png", dpi=160)
    fig.savefig(output / "qc_example_neuron_traces.pdf")
    plt.close(fig)
    save_json(
        output / "qc_sparsity.json",
        {"fraction_zero_observed": float(np.mean(tensor[mask] == 0))},
    )


def save_rank_plots(cv: pd.DataFrame, stability: pd.DataFrame, output: Path) -> None:
    summary = cv.groupby("rank")["validation_rmse"].agg(["mean", "sem"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(summary.index, summary["mean"], yerr=summary["sem"], marker="o")
    ax.set(xlabel="CP rank", ylabel="Validation RMSE", title="Masked-entry CV")
    fig.tight_layout()
    fig.savefig(output / "tca_rank_selection.png", dpi=200)
    plt.close(fig)
    init = stability[stability["stability_type"] == "initializations"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(init["rank"], init["mean_stability"], yerr=init["std_stability"], marker="o")
    ax.set(xlabel="CP rank", ylabel="Factor-match score", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(output / "tca_stability_vs_rank.png", dpi=200)
    plt.close(fig)


def save_component_plots(model: Any, data: TensorData, output: Path) -> None:
    weights, factors = cp_normalize(model)
    for component in range(len(weights)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        order = np.argsort(factors[0][:, component])[::-1]
        axes[0].bar(np.arange(len(order)), factors[0][order, component])
        axes[0].set(title="Neuron loadings (sorted)", xlabel="Neuron rank")
        axes[1].plot(data.time_ms, factors[1][:, component])
        axes[1].set(title="Temporal factor", xlabel="ms from stimulus")
        axes[2].plot(factors[2][:, component])
        axes[2].set(title="Trial factor", xlabel="Trial order")
        fig.suptitle(f"Component {component + 1}; weight={weights[component]:.3g}")
        fig.tight_layout()
        fig.savefig(output / f"component_{component + 1:02d}.png", dpi=180)
        plt.close(fig)


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    """Run the complete real-data pipeline and save all requested outputs."""
    np.random.seed(config.random_seed)
    tl.set_backend("numpy")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "analysis_config.json", asdict(config))
    data = build_spike_count_tensor(config)
    tensor, keep, preprocessing = preprocess_tensor(
        data.counts,
        data.valid_mask,
        config.preprocessing,
        config.smoothing_sigma_bins,
        config.use_nonnegative_tca,
    )
    mask = data.valid_mask[keep]
    data.unit_ids = data.unit_ids[keep]
    data.brain_areas = data.brain_areas[keep]
    dataset_summary = {
        "source_format": "nested ZIP64 archives containing NumPy arrays and CSV",
        "source_arrays": ["spikes.npy", "clusters.npy", "brain_area.npy", "trial_data.csv"],
        "source_activity_type": "individual spike timestamps",
        "tensor_shape": list(tensor.shape),
        "axis_order": ["neurons", "time", "trials"],
        "dtype": str(data.counts.dtype),
        "n_nan_observed": int(np.isnan(data.counts[data.valid_mask]).sum()),
        "n_inf_observed": int(np.isinf(data.counts[data.valid_mask]).sum()),
        "n_negative_observed": int((data.counts[data.valid_mask] < 0).sum()),
        "trial_metadata_columns": list(data.trial_metadata.columns),
        "excluded_trials": data.excluded_trials,
        "event_alignment": config.event_alignment,
        "bin_width_ms": config.bin_width_ms,
    }
    save_json(output / "dataset_summary.json", dataset_summary)
    save_json(output / "preprocessing_summary.json", preprocessing)
    data.trial_metadata.to_csv(output / "trial_metadata.csv")
    save_qc_plots(data, tensor, mask, output)
    pca_table, pca_metrics = pca_baseline(tensor, mask)
    pca_table.to_csv(output / "pca_metrics.csv", index=False)
    save_pca_plots(pca_table, output)
    cv, initializations, stability, best_models, _ = run_tca(tensor, mask, config)
    selection = select_rank(cv, stability)
    selected_rank = selection["final_recommended_rank"]
    bootstrap = bootstrap_stability(
        tensor, mask, selected_rank, best_models[selected_rank], config
    )
    stability = pd.concat([stability, bootstrap], ignore_index=True, sort=False)
    cv.to_csv(output / "tca_cv_metrics.csv", index=False)
    initializations.to_csv(output / "tca_initialization_metrics.csv", index=False)
    stability.to_csv(output / "tca_stability_metrics.csv", index=False)
    save_rank_plots(cv, stability, output)
    save_json(output / "selected_rank.json", selection)
    model = cp_normalize(best_models[selected_rank])
    prediction = cp_to_tensor(model)
    reconstruction = reconstruction_metrics(tensor, prediction, mask)
    save_json(output / "selected_reconstruction_metrics.json", reconstruction)
    np.savez(
        output / "selected_factors.npz",
        neuron_factors=model.factors[0],
        time_factors=model.factors[1],
        trial_factors=model.factors[2],
        component_weights=model.weights,
        selected_rank=selected_rank,
        axis_labels=np.array(["neurons", "time", "trials"]),
        preprocessing_parameters=json.dumps(preprocessing),
        unit_ids=data.unit_ids,
        brain_areas=data.brain_areas,
        time_ms=data.time_ms,
    )
    save_component_plots(model, data, output)
    _write_report(output, dataset_summary, preprocessing, pca_metrics, selection, stability)
    LOGGER.info("dataset summary: %s", dataset_summary)
    return {"pca": pca_metrics, "rank_selection": selection, "output_dir": str(output)}


def _write_report(
    output: Path, dataset: dict, preprocessing: dict, pca: dict,
    selection: dict, stability: pd.DataFrame
) -> None:
    bootstrap = stability[stability["stability_type"] == "trial_bootstrap"]
    bootstrap_mean = float(bootstrap["mean_stability"].mean())
    report = f"""# Q13 PCA and tensor component analysis

## Dataset

Participant 1 spike timestamps were binned at 100 ms from stimulus onset to
outcome. The validated tensor shape is `{dataset['tensor_shape']}` in
neurons × time × trials order. Trial metadata columns were preserved:
{', '.join(dataset['trial_metadata_columns'])}. Invalid timing records:
{len(dataset['excluded_trials'])}.

## Preprocessing

Method: `{preprocessing['preprocessing']}`. Empty neurons removed:
{preprocessing['n_removed_empty_neurons']}. Observed zero fraction:
{preprocessing['fraction_zero_observed']:.3f}. Variable trial durations are
represented by a validity mask; padded entries are excluded from all losses.

## Dimensionality summary

| Measure | Result |
|---|---:|
| PCA d80 | {pca['d80']} |
| PCA d90 | {pca['d90']} |
| PCA d95 | {pca['d95']} |
| PCA participation ratio | {pca['participation_ratio']:.3f} |
| Cross-validated TCA rank | {selection['cross_validated_rank']} |
| Most stable TCA rank | {selection['most_stable_rank']} |
| Final recommended TCA rank | {selection['final_recommended_rank']} |

## TCA selection and stability

Recommendation: {selection['recommendation_reason']}. Mean selected-rank
trial-bootstrap factor match (neuron and temporal modes): {bootstrap_mean:.3f}.
TCA rank is the number of reproducible separable neuron × time × trial
patterns supported by CP/PARAFAC, not intrinsic manifold dimensionality.

## Component interpretation

See `component_*.png` for neuron, temporal, and trial loadings. Large neuron
loadings identify units contributing most strongly; temporal loadings show
latency from stimulus; trial loadings show expression across trial order.
Associations with anonymized variables A–C are left descriptive because their
scientific meanings are not supplied.

## Limitations

Trials have unequal stimulus-to-outcome durations, so later time bins contain
fewer trials. One trial with outcome after trial end was excluded. CP models
assume separable multilinear components, and rank can depend on bin width,
preprocessing, solver convergence, and sparsity. Bootstrap confidence intervals
are limited by the configured number of bootstrap fits.

## Reproduce

```bash
../.venv/bin/python analysis/Q13/run_analysis.py
```
"""
    (output / "analysis_report.md").write_text(report)


def synthetic_tensor(
    rank: int = 3, shape: tuple[int, int, int] = (18, 20, 24), seed: int = 42
) -> tuple[np.ndarray, Any]:
    rng = np.random.default_rng(seed)
    factors = [rng.gamma(2, 1, (size, rank)) for size in shape]
    model = tl.cp_tensor.CPTensor((np.ones(rank), factors))
    rates = cp_to_tensor(model)
    rates *= 2 / rates.mean()
    return rng.poisson(rates).astype(float), model
