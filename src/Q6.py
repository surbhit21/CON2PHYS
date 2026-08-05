"""Estimate directed functional connectivity with spike-history Poisson GLMs.

For every target neuron in ``target_area``, the model predicts its spike count
in 10-ms bins from causal histories of itself and other target-area neurons
(150 ms), source-area neurons (300 ms), and trial covariates.  History is reset
at trial boundaries.  Positive/negative source coefficients are statistical
associations and should not be interpreted as monosynaptic excitation or
inhibition without additional controls.

Run participant 1 from the repository root with::

    python src/Q6.py --source-area 1 --target-area 2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize, sparse, stats

if __package__:
    from .data_loader import (
        CON2PHYSSession,
        DEFAULT_DATA_PATH,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )
    from .stat_test import holm_adjust
else:
    from data_loader import (
        CON2PHYSSession,
        DEFAULT_DATA_PATH,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )
    from stat_test import holm_adjust


PARTICIPANT = 1
BIN_SIZE = 0.010
TARGET_HISTORY = 0.150
SOURCE_HISTORY = 0.300
DEFAULT_COUPLING_PLOT = DEFAULT_PLOT_PATH / "Q6" / "inter_region_coupling.png"
DEFAULT_COUPLING_HISTOGRAM = (
    DEFAULT_PLOT_PATH / "Q6" / "coupling_strength_histogram.png"
)
DEFAULT_REGION_PAIRS = ((3, 2), (3, 1), (1, 2))


def region_pair_name(source_area: object, target_area: object) -> str:
    """Return the directory name used to keep directed-pair runs separate."""
    return f"source_{source_area}_to_target_{target_area}"


@dataclass(frozen=True)
class GLMDesign:
    """Sparse, trial-concatenated design shared by all target-neuron fits."""

    counts: sparse.csr_matrix
    design: sparse.csr_matrix
    feature_table: pd.DataFrame
    target_unit_ids: np.ndarray
    source_unit_ids: np.ndarray
    trial_ids: np.ndarray
    bin_times: np.ndarray


@dataclass(frozen=True)
class ConnectivityResult:
    """Fitted models, source-history coefficients, and held-out scores."""

    coefficients: pd.DataFrame
    source_connectivity: pd.DataFrame
    model_scores: pd.DataFrame


def load_participant_spikes(
    participant: int = PARTICIPANT,
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> CON2PHYSSession:
    """Load spikes, unit metadata, and trials without loading LFP arrays."""
    return load_session(subject=participant, data_path=data_path, lfp_banks=())


def _validate_seconds(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _unit_ids(session: CON2PHYSSession, area: object) -> np.ndarray:
    if session.spikes is None:
        raise ValueError("session must contain spike data")
    metadata = session.spikes.metadata
    if "brain_area" not in metadata:
        raise ValueError("spike metadata must contain brain_area")
    return metadata.index[metadata["brain_area"] == area].to_numpy(dtype=int)


def _bin_trial_spikes(
    session: CON2PHYSSession,
    unit_ids: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    bin_size: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    assert session.spikes is not None
    trial_counts: list[np.ndarray] = []
    trial_times: list[np.ndarray] = []
    spike_arrays = {
        int(unit): np.asarray(session.spikes[int(unit)].t, dtype=float)
        for unit in unit_ids
    }
    for start, end in zip(starts, ends):
        # The tolerance prevents e.g. 0.9 / 0.1 evaluating just below 9.
        n_bins = int(np.floor((end - start) / bin_size + 1e-9))
        if n_bins < 1:
            trial_counts.append(np.empty((0, len(unit_ids)), dtype=np.int16))
            trial_times.append(np.empty(0))
            continue
        edges = start + np.arange(n_bins + 1) * bin_size
        counts = np.empty((n_bins, len(unit_ids)), dtype=np.int16)
        for column, unit in enumerate(unit_ids):
            counts[:, column] = np.histogram(spike_arrays[int(unit)], edges)[0]
        trial_counts.append(counts)
        trial_times.append(edges[:-1] + bin_size / 2)
    return trial_counts, trial_times


def _lagged_block(
    trial_arrays: list[np.ndarray],
    n_lags: int,
    unit_ids: np.ndarray,
    group: str,
    bin_size: float,
) -> tuple[sparse.csr_matrix, list[dict[str, object]]]:
    blocks: list[sparse.csr_matrix] = []
    for values in trial_arrays:
        n_rows, n_units = values.shape
        lagged = []
        for lag in range(1, n_lags + 1):
            if lag >= n_rows:
                shifted = sparse.csr_matrix((n_rows, n_units), dtype=float)
            else:
                shifted = sparse.vstack(
                    [sparse.csr_matrix((lag, n_units)), sparse.csr_matrix(values[:-lag])],
                    format="csr",
                )
            lagged.append(shifted)
        blocks.append(sparse.hstack(lagged, format="csr"))
    matrix = sparse.vstack(blocks, format="csr")
    features = [
        {
            "feature_group": group,
            "unit_id": int(unit),
            "lag_bin": lag,
            "lag_ms": round(lag * bin_size * 1_000, 9),
            "covariate": None,
        }
        for lag in range(1, n_lags + 1)
        for unit in unit_ids
    ]
    return matrix, features


def _trial_covariates(
    trial_table: pd.DataFrame,
    trial_lengths: list[int],
    trial_times: list[np.ndarray],
) -> tuple[sparse.csr_matrix, list[dict[str, object]]]:
    excluded = {"trial_start", "trial_end", "stim_start", "outcome"}
    columns = [column for column in trial_table.columns if column not in excluded]
    encoded_parts: list[pd.DataFrame] = []
    names: list[str] = []
    for column in columns:
        series = trial_table[column]
        if pd.api.types.is_numeric_dtype(series):
            values = series.astype(float)
            if values.nunique(dropna=True) > 1:
                values = (values - values.mean()) / values.std(ddof=0)
            encoded_parts.append(values.fillna(0).to_frame(str(column)))
            names.append(str(column))
        else:
            dummy = pd.get_dummies(series, prefix=str(column), drop_first=True, dtype=float)
            encoded_parts.append(dummy)
            names.extend(dummy.columns.astype(str))
    trial_encoded = (
        pd.concat(encoded_parts, axis=1).to_numpy(float)
        if encoded_parts
        else np.empty((len(trial_table), 0))
    )

    # Event-related state and elapsed time let the baseline rate vary within a trial.
    rows = []
    for row_index, (n_bins, times) in enumerate(zip(trial_lengths, trial_times)):
        base = np.repeat(trial_encoded[row_index : row_index + 1], n_bins, axis=0)
        row = trial_table.iloc[row_index]
        duration = max(float(row["trial_end"] - row["trial_start"]), np.finfo(float).eps)
        time_in_trial = (times - float(row["trial_start"])) / duration
        post_stimulus = (times >= float(row["stim_start"])).astype(float)
        post_outcome = (times >= float(row["outcome"])).astype(float)
        rows.append(np.column_stack([base, time_in_trial, post_stimulus, post_outcome]))
    names.extend(["time_in_trial", "post_stimulus", "post_outcome"])
    values = np.vstack(rows) if rows else np.empty((0, len(names)))
    features = [
        {
            "feature_group": "trial",
            "unit_id": None,
            "lag_bin": None,
            "lag_ms": None,
            "covariate": name,
        }
        for name in names
    ]
    return sparse.csr_matrix(values), features


def build_glm_design(
    session: CON2PHYSSession,
    *,
    source_area: object = 1,
    target_area: object = 2,
    bin_size: float = BIN_SIZE,
    target_history: float = TARGET_HISTORY,
    source_history: float = SOURCE_HISTORY,
) -> GLMDesign:
    """Bin spikes and construct causal sparse predictors without trial leakage."""
    _validate_seconds(bin_size, "bin_size")
    _validate_seconds(target_history, "target_history")
    _validate_seconds(source_history, "source_history")
    target_lags = int(round(target_history / bin_size))
    source_lags = int(round(source_history / bin_size))
    if target_lags < 1 or source_lags < 1:
        raise ValueError("history windows must contain at least one bin")
    target_ids = _unit_ids(session, target_area)
    source_ids = _unit_ids(session, source_area)
    if not len(target_ids) or not len(source_ids):
        raise ValueError("source_area and target_area must each contain at least one unit")

    trials = session.trial_table.reset_index(drop=False)
    starts = trials["trial_start"].to_numpy(float)
    ends = trials["trial_end"].to_numpy(float)
    all_ids = np.concatenate([target_ids, source_ids])
    all_counts, trial_times = _bin_trial_spikes(
        session, all_ids, starts, ends, bin_size
    )
    target_counts = [values[:, : len(target_ids)] for values in all_counts]
    source_counts = [values[:, len(target_ids) :] for values in all_counts]
    lengths = [len(values) for values in all_counts]

    target_block, target_features = _lagged_block(
        target_counts, target_lags, target_ids, "target_history", bin_size
    )
    source_block, source_features = _lagged_block(
        source_counts, source_lags, source_ids, "source_history", bin_size
    )
    trial_block, trial_features = _trial_covariates(
        session.trial_table.reset_index(drop=True), lengths, trial_times
    )
    design = sparse.hstack([target_block, source_block, trial_block], format="csr")
    counts = sparse.csr_matrix(np.vstack(target_counts))
    trial_ids = np.concatenate(
        [np.repeat(trial, length) for trial, length in zip(session.trial_table.index, lengths)]
    )
    bin_times = np.concatenate(trial_times)
    feature_table = pd.DataFrame(target_features + source_features + trial_features)
    feature_table.index.name = "feature_index"
    return GLMDesign(
        counts=counts,
        design=design,
        feature_table=feature_table,
        target_unit_ids=target_ids,
        source_unit_ids=source_ids,
        trial_ids=trial_ids,
        bin_times=bin_times,
    )


def _poisson_fit(
    matrix: sparse.csr_matrix,
    response: np.ndarray,
    penalty: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, bool, int]:
    n_rows = len(response)
    initial_rate = max(float(response.mean()), 1e-8)
    initial = np.zeros(matrix.shape[1] + 1)
    initial[0] = np.log(initial_rate)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = np.clip(parameters[0] + matrix @ parameters[1:], -30, 30)
        mean = np.exp(linear)
        residual = mean - response
        value = (mean.sum() - response @ linear) / n_rows
        value += 0.5 * np.sum(penalty * parameters[1:] ** 2)
        gradient = np.r_[residual.mean(), np.asarray(matrix.T @ residual).ravel() / n_rows]
        gradient[1:] += penalty * parameters[1:]
        return float(value), gradient

    result = optimize.minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1e-9},
    )
    return result.x, bool(result.success), int(result.nit)


def _poisson_deviance(y: np.ndarray, mean: np.ndarray) -> float:
    mean = np.maximum(mean, 1e-12)
    terms = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mean) - (y - mean), mean)
    return float(2 * terms.sum())


def fit_connectivity_glms(
    design: GLMDesign,
    *,
    ridge: float = 1.0,
    test_fraction: float = 0.2,
    random_state: int = 0,
    max_iter: int = 200,
    target_units: Iterable[int] | None = None,
) -> ConnectivityResult:
    """Fit one ridge-regularized Poisson GLM per target unit.

    Entire trials, rather than individual bins, are assigned to the held-out
    set. The reported pseudo-R2 is ``1 - model_deviance/null_deviance``.
    """
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    unique_trials = np.unique(design.trial_ids)
    if len(unique_trials) < 2:
        raise ValueError("at least two trials are required")
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unique_trials)
    n_test = min(max(1, int(round(test_fraction * len(shuffled)))), len(shuffled) - 1)
    test_mask = np.isin(design.trial_ids, shuffled[:n_test])
    train_mask = ~test_mask

    selected = set(map(int, target_units)) if target_units is not None else None
    feature = design.feature_table
    penalty = np.full(len(feature), ridge, dtype=float)
    penalty[feature["feature_group"].eq("trial").to_numpy()] = 0.0
    coefficient_frames: list[pd.DataFrame] = []
    score_rows: list[dict[str, object]] = []
    for column, target_id in enumerate(design.target_unit_ids):
        if selected is not None and int(target_id) not in selected:
            continue
        response = design.counts[:, column].toarray().ravel().astype(float)
        parameters, converged, iterations = _poisson_fit(
            design.design[train_mask], response[train_mask], penalty, max_iter
        )
        frame = feature.copy()
        frame.insert(0, "target_unit_id", int(target_id))
        frame["coefficient"] = parameters[1:]
        frame["rate_ratio"] = np.exp(np.clip(parameters[1:], -30, 30))
        coefficient_frames.append(frame.reset_index())

        held_out = response[test_mask]
        predicted = np.exp(np.clip(parameters[0] + design.design[test_mask] @ parameters[1:], -30, 30))
        null_mean = np.full(len(held_out), max(response[train_mask].mean(), 1e-12))
        null_deviance = _poisson_deviance(held_out, null_mean)
        model_deviance = _poisson_deviance(held_out, predicted)
        score_rows.append(
            {
                "target_unit_id": int(target_id),
                "intercept": parameters[0],
                "converged": converged,
                "iterations": iterations,
                "n_train_bins": int(train_mask.sum()),
                "n_test_bins": int(test_mask.sum()),
                "test_log_likelihood_per_bin": float(np.mean(held_out * np.log(predicted) - predicted)),
                "test_deviance": model_deviance,
                "test_pseudo_r2": 1 - model_deviance / null_deviance if null_deviance > 0 else np.nan,
            }
        )
    if not coefficient_frames:
        raise ValueError("none of target_units are present in the target area")
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    source = coefficients.loc[coefficients["feature_group"] == "source_history"].copy()
    source_connectivity = (
        source.groupby(["target_unit_id", "unit_id"], as_index=False)
        .agg(
            summed_coefficient=("coefficient", "sum"),
            mean_coefficient=("coefficient", "mean"),
            peak_absolute_coefficient=("coefficient", lambda x: float(x.iloc[np.abs(x.to_numpy()).argmax()])),
        )
        .rename(columns={"unit_id": "source_unit_id"})
    )
    return ConnectivityResult(
        coefficients=coefficients,
        source_connectivity=source_connectivity,
        model_scores=pd.DataFrame(score_rows),
    )


def save_connectivity_results(result: ConnectivityResult, output_dir: str | Path) -> None:
    """Write full lag kernels, pair summaries, and model diagnostics to CSV."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    result.coefficients.to_csv(path / "glm_coefficients.csv", index=False)
    result.source_connectivity.to_csv(path / "source_target_connectivity.csv", index=False)
    result.model_scores.to_csv(path / "glm_model_scores.csv", index=False)
    summarize_coupling_strengths(result.source_connectivity).to_csv(
        path / "coupling_strength_summary.csv", index=False
    )


def summarize_coupling_strengths(
    connectivity: pd.DataFrame,
    *,
    group_column: str | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Return mean lag-summed coupling and a Student-t confidence interval.

    When ``group_column`` is supplied, one mean is first calculated within
    each group and the reported CI is across those group means. For the batch
    analysis, grouping by participant makes participants—not neuron pairs—the
    independent observations used for population inference.
    """
    if "summed_coefficient" not in connectivity:
        raise ValueError("connectivity is missing summed_coefficient")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if group_column is not None:
        if group_column not in connectivity:
            raise ValueError(f"connectivity is missing {group_column}")
        values = (
            connectivity.groupby(group_column)["summed_coefficient"]
            .mean()
            .dropna()
            .to_numpy(float)
        )
        observation = f"{group_column}_means"
    else:
        values = connectivity["summed_coefficient"].dropna().to_numpy(float)
        observation = "source_target_pairs"
    if not len(values):
        raise ValueError("no finite coupling strengths are available")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    sem = sd / np.sqrt(len(values)) if len(values) > 1 else np.nan
    critical = stats.t.ppf(0.5 + confidence / 2, len(values) - 1) if len(values) > 1 else np.nan
    margin = critical * sem
    return pd.DataFrame(
        [
            {
                "observation_unit": observation,
                "n": len(values),
                "mean_coupling_strength": mean,
                "sd": sd,
                "sem": sem,
                "confidence": confidence,
                "ci_low": mean - margin if len(values) > 1 else np.nan,
                "ci_high": mean + margin if len(values) > 1 else np.nan,
            }
        ]
    )


def plot_inter_region_coupling(
    result: ConnectivityResult,
    path: str | Path = DEFAULT_COUPLING_PLOT,
) -> Path:
    """Plot source-to-target coupling strengths and the mean temporal kernel.

    The heatmap contains each source-target pair's coefficient summed over the
    source-history lags. The second panel shows the mean lag-resolved source
    coefficient across all pairs, with a 95% normal confidence band.
    """
    required = {"target_unit_id", "source_unit_id", "summed_coefficient"}
    missing = required.difference(result.source_connectivity.columns)
    if missing:
        raise ValueError(f"source_connectivity is missing columns: {sorted(missing)}")
    source_lags = result.coefficients.loc[
        result.coefficients["feature_group"] == "source_history"
    ].copy()
    if source_lags.empty:
        raise ValueError("result does not contain source-history coefficients")

    heatmap = result.source_connectivity.pivot(
        index="target_unit_id",
        columns="source_unit_id",
        values="summed_coefficient",
    ).sort_index().sort_index(axis=1)
    lag_summary = source_lags.groupby("lag_ms")["coefficient"].agg(
        mean="mean", std="std", count="count"
    )
    lag_summary["ci95"] = 1.96 * lag_summary["std"].fillna(0) / np.sqrt(
        lag_summary["count"]
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, max(5.5, 0.22 * len(heatmap.index) + 2.0)),
        gridspec_kw={"width_ratios": [2.2, 1]},
        constrained_layout=True,
    )
    limit = float(np.nanmax(np.abs(heatmap.to_numpy())))
    if not np.isfinite(limit) or limit == 0:
        limit = 1.0
    image = axes[0].imshow(
        heatmap.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axes[0].set_title("Source-to-target GLM coupling")
    axes[0].set_xlabel("Source-region neuron ID")
    axes[0].set_ylabel("Target-region neuron ID")
    x_step = max(1, int(np.ceil(len(heatmap.columns) / 15)))
    x_positions = np.arange(0, len(heatmap.columns), x_step)
    axes[0].set_xticks(x_positions, heatmap.columns[x_positions], rotation=90)
    axes[0].set_yticks(np.arange(len(heatmap.index)), heatmap.index)
    colorbar = figure.colorbar(image, ax=axes[0], shrink=0.9)
    colorbar.set_label("Summed source-history coefficient")

    lags = lag_summary.index.to_numpy(float)
    means = lag_summary["mean"].to_numpy(float)
    ci = lag_summary["ci95"].to_numpy(float)
    axes[1].axhline(0, color="0.35", linewidth=1, linestyle="--")
    axes[1].plot(lags, means, color="#155f8a", linewidth=2)
    axes[1].fill_between(lags, means - ci, means + ci, color="#155f8a", alpha=0.22)
    axes[1].set_title("Mean source coupling kernel")
    axes[1].set_xlabel("Source-history lag (ms)")
    axes[1].set_ylabel("GLM coefficient (mean ± 95% CI)")
    axes[1].spines[["top", "right"]].set_visible(False)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_coupling_strength_histogram(
    result: ConnectivityResult,
    path: str | Path = DEFAULT_COUPLING_HISTOGRAM,
    *,
    bins: int = 20,
) -> Path:
    """Plot the distribution of lag-summed source-target coupling strengths."""
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if "summed_coefficient" not in result.source_connectivity:
        raise ValueError("source_connectivity is missing summed_coefficient")
    strengths = result.source_connectivity["summed_coefficient"].dropna().to_numpy(float)
    if not len(strengths):
        raise ValueError("no finite coupling strengths are available")

    figure, axis = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    axis.hist(
        strengths,
        bins=bins,
        color="#2474a6",
        edgecolor="white",
        linewidth=0.8,
    )
    axis.axvline(0, color="0.25", linewidth=1.2, linestyle="--", label="Zero coupling")
    axis.axvline(
        np.mean(strengths),
        color="#b33b2e",
        linewidth=1.5,
        label=f"Mean = {np.mean(strengths):.3g}",
    )
    axis.set_title("Distribution of source-to-target coupling strengths")
    axis.set_xlabel("Summed source-history GLM coefficient")
    axis.set_ylabel("Source-target neuron pairs")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_participant_analysis(
    participant: int,
    *,
    source_area: object = 1,
    target_area: object = 2,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = Path("analysis/Q6"),
    plot_dir: str | Path = DEFAULT_PLOT_PATH / "Q6",
    ridge: float = 1.0,
    test_fraction: float = 0.2,
    random_state: int = 0,
    max_iter: int = 200,
    target_units: Iterable[int] | None = None,
) -> ConnectivityResult:
    """Fit, save, and plot one participant's inter-region GLMs."""
    session = load_participant_spikes(participant, data_path=data_path)
    design = build_glm_design(
        session, source_area=source_area, target_area=target_area
    )
    result = fit_connectivity_glms(
        design,
        ridge=ridge,
        test_fraction=test_fraction,
        random_state=random_state,
        max_iter=max_iter,
        target_units=target_units,
    )
    save_connectivity_results(result, output_dir)
    plot_path = Path(plot_dir)
    plot_inter_region_coupling(result, plot_path / "inter_region_coupling.png")
    plot_coupling_strength_histogram(
        result, plot_path / "coupling_strength_histogram.png", bins=20
    )
    return result


def run_all_participants(
    *,
    source_area: object = 1,
    target_area: object = 2,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_root: str | Path = Path("analysis/Q6"),
    plot_root: str | Path = DEFAULT_PLOT_PATH / "Q6",
    ridge: float = 1.0,
    test_fraction: float = 0.2,
    random_state: int = 0,
    max_iter: int = 200,
) -> pd.DataFrame:
    """Run Q6 sequentially for every available participant.

    Participant-specific outputs are placed in ``participant_<id>`` folders.
    Combined pairwise coupling and model-score tables, a run-status table, and
    a pooled 20-bin histogram are also written at the output roots. Sequential
    execution bounds memory use to one participant's sparse design at a time.
    """
    output_root = Path(output_root)
    plot_root = Path(plot_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    connectivity_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    statuses: list[dict[str, object]] = []
    for participant in available_subjects(data_path):
        print(f"Participant {participant}: loading and fitting", flush=True)
        try:
            result = run_participant_analysis(
                participant,
                source_area=source_area,
                target_area=target_area,
                data_path=data_path,
                output_dir=output_root / f"participant_{participant}",
                plot_dir=plot_root / f"participant_{participant}",
                ridge=ridge,
                test_fraction=test_fraction,
                random_state=random_state + participant,
                max_iter=max_iter,
            )
        except ValueError as error:
            # A participant can legitimately lack one of a user-selected pair
            # of regions. Record it without discarding successful participants.
            statuses.append(
                {"participant": participant, "status": "skipped", "message": str(error)}
            )
            print(f"Participant {participant}: skipped ({error})", flush=True)
            continue
        connectivity = result.source_connectivity.copy()
        connectivity.insert(0, "participant", participant)
        connectivity_frames.append(connectivity)
        scores = result.model_scores.copy()
        scores.insert(0, "participant", participant)
        score_frames.append(scores)
        statuses.append(
            {
                "participant": participant,
                "status": "completed",
                "message": "",
                "n_target_models": len(scores),
                "n_source_target_pairs": len(connectivity),
            }
        )
        print(f"Participant {participant}: fitted {len(scores)} target neurons", flush=True)

    status_table = pd.DataFrame(statuses)
    status_table.to_csv(output_root / "all_participants_run_status.csv", index=False)
    if not connectivity_frames:
        raise ValueError("no participants could be analyzed for the selected regions")
    combined_connectivity = pd.concat(connectivity_frames, ignore_index=True)
    combined_scores = pd.concat(score_frames, ignore_index=True)
    combined_connectivity.to_csv(
        output_root / "all_participants_source_target_connectivity.csv", index=False
    )
    combined_scores.to_csv(
        output_root / "all_participants_glm_model_scores.csv", index=False
    )
    participant_summary = (
        combined_connectivity.assign(
            absolute_coupling_strength=lambda frame: frame["summed_coefficient"].abs()
        )
        .groupby("participant", as_index=False)
        .agg(
            n_source_target_pairs=("summed_coefficient", "count"),
            mean_signed_coupling=("summed_coefficient", "mean"),
            mean_absolute_coupling_strength=("absolute_coupling_strength", "mean"),
        )
    )
    participant_summary.to_csv(
        output_root / "coupling_strength_by_participant.csv", index=False
    )
    summarize_coupling_strengths(
        combined_connectivity, group_column="participant"
    ).to_csv(output_root / "coupling_strength_group_summary.csv", index=False)
    pooled = ConnectivityResult(
        coefficients=pd.DataFrame(),
        source_connectivity=combined_connectivity,
        model_scores=combined_scores,
    )
    plot_coupling_strength_histogram(
        pooled, plot_root / "all_participants_coupling_strength_histogram.png", bins=20
    )
    return status_table


def compare_coupling_across_region_pairs(
    output_root: str | Path = Path("analysis/Q6"),
    *,
    pairs: Iterable[tuple[object, object]] = DEFAULT_REGION_PAIRS,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Compare absolute coupling magnitudes across directed region pairs.

    Each observation is one participant's mean absolute lag-summed coupling
    across its source-target neuron pairs. A Friedman test compares all
    directions; the highest-mean direction is then tested against the others
    with one-sided paired Wilcoxon tests and Holm correction.
    """
    root = Path(output_root)
    participant_frames: list[pd.DataFrame] = []
    pair_list = list(pairs)
    if len(pair_list) < 3:
        raise ValueError("at least three directed region pairs are required")
    for source_area, target_area in pair_list:
        label = region_pair_name(source_area, target_area)
        path = root / label / "all_participants_source_target_connectivity.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run Q6 for source {source_area} -> target {target_area}"
            )
        connectivity = pd.read_csv(path)
        required = {"participant", "summed_coefficient"}
        missing = required.difference(connectivity.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        participant_mean = (
            connectivity.assign(
                absolute_coupling_strength=connectivity["summed_coefficient"].abs()
            )
            .groupby("participant", as_index=False)
            .agg(
                n_source_target_pairs=("summed_coefficient", "count"),
                mean_signed_coupling=("summed_coefficient", "mean"),
                mean_absolute_coupling_strength=("absolute_coupling_strength", "mean"),
            )
        )
        participant_mean.insert(1, "directed_pair", label)
        participant_mean.insert(2, "source_area", source_area)
        participant_mean.insert(3, "target_area", target_area)
        participant_frames.append(participant_mean)
    participant_table = pd.concat(participant_frames, ignore_index=True)

    grouped = participant_table.groupby("directed_pair")[
        "mean_absolute_coupling_strength"
    ]
    summary = grouped.agg(n_participants="count", mean_coupling="mean", sd="std")
    summary["sem"] = summary["sd"] / np.sqrt(summary["n_participants"])
    critical = summary["n_participants"].map(
        lambda n: stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    )
    margin = critical * summary["sem"]
    summary["ci95_low"] = summary["mean_coupling"] - margin
    summary["ci95_high"] = summary["mean_coupling"] + margin
    highest_pair = summary["mean_coupling"].idxmax()

    complete = participant_table.pivot(
        index="participant",
        columns="directed_pair",
        values="mean_absolute_coupling_strength",
    ).dropna()
    if complete.shape[0] < 2:
        raise ValueError("at least two participants with all pair fits are required")
    friedman = stats.friedmanchisquare(
        *(complete[label].to_numpy() for label in complete.columns)
    )
    comparisons: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for other_pair in complete.columns:
        if other_pair == highest_pair:
            continue
        test = stats.wilcoxon(
            complete[highest_pair], complete[other_pair], alternative="greater"
        )
        raw_p_values.append(float(test.pvalue))
        comparisons.append(
            {
                "highest_pair": highest_pair,
                "comparison_pair": other_pair,
                "n_complete_participants": len(complete),
                "wilcoxon_statistic": float(test.statistic),
                "p_value_one_sided": float(test.pvalue),
            }
        )
    for comparison, adjusted in zip(comparisons, holm_adjust(raw_p_values)):
        comparison["p_value_holm"] = adjusted
        comparison["significant_0.05"] = adjusted < 0.05
    pairwise = pd.DataFrame(comparisons)

    root.mkdir(parents=True, exist_ok=True)
    participant_table.to_csv(root / "region_pair_coupling_by_participant.csv", index=False)
    summary.to_csv(root / "region_pair_coupling_group_summary.csv")
    pairwise.to_csv(root / "region_pair_coupling_pairwise_tests.csv", index=False)
    pd.DataFrame(
        [{
            "n_complete_participants": len(complete),
            "friedman_statistic": float(friedman.statistic),
            "p_value": float(friedman.pvalue),
            "highest_mean_pair": highest_pair,
        }]
    ).to_csv(root / "region_pair_coupling_friedman_test.csv", index=False)
    return summary, pairwise, friedman


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", type=int, default=PARTICIPANT)
    parser.add_argument(
        "--all-participants",
        action="store_true",
        help="analyze every participant sequentially",
    )
    parser.add_argument(
        "--compare-pairs",
        action="store_true",
        help="compare completed 3->2, 3->1, and 1->2 all-participant runs",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--source-area", type=int, default=1)
    parser.add_argument("--target-area", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--target-unit", type=int, action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/Q6"))
    parser.add_argument("--plot", type=Path, default=DEFAULT_COUPLING_PLOT)
    parser.add_argument(
        "--histogram", type=Path, default=DEFAULT_COUPLING_HISTOGRAM
    )
    args = parser.parse_args()
    if args.compare_pairs:
        summary, pairwise, friedman = compare_coupling_across_region_pairs(
            args.output_dir
        )
        highest = summary["mean_coupling"].idxmax()
        print("Mean absolute coupling strength and 95% CI")
        print(summary.to_string(float_format=lambda value: f"{value:.6g}"))
        print(
            f"\nFriedman test: chi-square={friedman.statistic:.6g}, "
            f"p={friedman.pvalue:.6g}"
        )
        print(f"Highest mean coupling: {highest}")
        print("\nHolm-corrected paired comparisons")
        print(pairwise.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
        return
    pair_directory = region_pair_name(args.source_area, args.target_area)
    pair_output = args.output_dir / pair_directory
    pair_plots = args.plot.parent / pair_directory
    if args.all_participants:
        if args.target_unit:
            parser.error("--target-unit cannot be combined with --all-participants")
        statuses = run_all_participants(
            source_area=args.source_area,
            target_area=args.target_area,
            data_path=args.data_path,
            output_root=pair_output,
            plot_root=pair_plots,
            ridge=args.ridge,
            test_fraction=args.test_fraction,
            max_iter=args.max_iter,
        )
        completed = int(statuses["status"].eq("completed").sum())
        print(f"Completed {completed} of {len(statuses)} participants")
        print(f"Combined results: {pair_output.resolve()}")
        print(f"Plots: {pair_plots.resolve()}")
    else:
        result = run_participant_analysis(
            args.participant,
            source_area=args.source_area,
            target_area=args.target_area,
            data_path=args.data_path,
            output_dir=pair_output,
            plot_dir=pair_plots,
            ridge=args.ridge,
            test_fraction=args.test_fraction,
            max_iter=args.max_iter,
            target_units=args.target_unit,
        )
        # Respect explicit filenames supplied for the legacy single-subject CLI.
        if args.plot != DEFAULT_COUPLING_PLOT:
            plot_inter_region_coupling(result, args.plot)
        if args.histogram != DEFAULT_COUPLING_HISTOGRAM:
            plot_coupling_strength_histogram(result, args.histogram, bins=20)
        print(f"Fitted {len(result.model_scores)} target-neuron GLMs")
        print(f"Results: {pair_output.resolve()}")
        print(f"Plots: {pair_plots.resolve()}")


if __name__ == "__main__":
    main()
