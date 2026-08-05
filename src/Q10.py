"""Bin ITI spiking data and estimate brain-area neuronal timescales."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats
from scipy.optimize import curve_fit, least_squares

if __package__:
    from .data_loader import (
        CON2PHYSSession,
        DEFAULT_DATA_PATH,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )
else:
    from data_loader import (
        CON2PHYSSession,
        DEFAULT_DATA_PATH,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )


PATIENT = 1
BIN_WIDTH_S = 0.005
MAX_LAG_MS = 900.0
MIN_LONG_ITI_S = 1.0
LONG_ITI_MAX_LAG_MS = 1_000.0
DEFAULT_AUTOCORRELATION_PLOT = (
    DEFAULT_PLOT_PATH / "Q10_mean_ITI_spike_count_autocorrelation_by_area.png"
)
DEFAULT_TIMESCALE_PLOT = (
    DEFAULT_PLOT_PATH / "Q10_intrinsic_timescale_exponential_fits_by_area.png"
)
DEFAULT_POOLED_LONG_ITI_PLOT = (
    DEFAULT_PLOT_PATH / "Q10_pooled_long_ITI_autocorrelation_by_area.png"
)
DEFAULT_POOLED_LONG_ITI_LOG_PLOT = (
    DEFAULT_PLOT_PATH / "Q10_pooled_long_ITI_log_autocorrelation_by_area.png"
)
DEFAULT_MULTIEXPONENTIAL_PLOT = (
    DEFAULT_PLOT_PATH / "Q10_BIC_selected_multiexponential_fits_by_area.png"
)
DEFAULT_MULTIEXPONENTIAL_LOGLOG_PLOT = (
    DEFAULT_PLOT_PATH / "Q10_BIC_selected_multiexponential_fits_loglog_by_area.png"
)
DEFAULT_MULTIEXPONENTIAL_LOGY_PLOT = (
    DEFAULT_PLOT_PATH / "Q10_BIC_selected_multiexponential_fits_log_y_by_area.png"
)


def load_patient_1_spikes() -> CON2PHYSSession:
    """Load patient 1 spike trains, metadata, and trials without LFP data."""
    return load_session(subject=PATIENT, lfp_banks=(), load_spikes=True)


def bin_inter_trial_spikes(
    session: CON2PHYSSession,
    *,
    bin_width_s: float = BIN_WIDTH_S,
) -> pd.DataFrame:
    """Count each unit's spikes in 5-ms inter-trial-interval bins.

    An inter-trial interval (ITI) runs from the end of one trial to the start
    of the following trial. Bins are constructed separately within each ITI,
    so no bin crosses a trial boundary. Any final fragment shorter than one
    complete bin is omitted.

    Returns
    -------
    pandas.DataFrame
        One row per complete bin and one column per unit. The row index stores
        the preceding trial, following trial, within-ITI bin number, and the
        absolute bin start/end times in seconds.
    """
    if session.spikes is None:
        raise ValueError("session must contain spiking data")
    if not np.isfinite(bin_width_s) or bin_width_s <= 0:
        raise ValueError("bin_width_s must be positive")

    trials = session.trial_table
    starts = trials["trial_start"].to_numpy(dtype=float)
    ends = trials["trial_end"].to_numpy(dtype=float)
    trial_ids = trials.index.to_numpy()
    unit_ids = [int(unit_id) for unit_id in session.spikes.keys()]
    spike_times = {
        unit_id: np.asarray(session.spikes[unit_id].index, dtype=float)
        for unit_id in unit_ids
    }

    counts: list[np.ndarray] = []
    index_rows: list[tuple[object, object, int, float, float]] = []
    for position in range(len(trials) - 1):
        iti_start = ends[position]
        iti_end = starts[position + 1]
        if not np.isfinite(iti_start) or not np.isfinite(iti_end):
            continue

        n_bins = int(np.floor((iti_end - iti_start) / bin_width_s + 1e-12))
        if n_bins < 1:
            continue
        edges = iti_start + np.arange(n_bins + 1, dtype=float) * bin_width_s
        interval_counts = np.column_stack(
            [
                np.diff(np.searchsorted(spike_times[unit_id], edges, side="left"))
                for unit_id in unit_ids
            ]
        )
        counts.append(interval_counts)
        index_rows.extend(
            (
                trial_ids[position],
                trial_ids[position + 1],
                bin_number,
                edges[bin_number],
                edges[bin_number + 1],
            )
            for bin_number in range(n_bins)
        )

    index = pd.MultiIndex.from_tuples(
        index_rows,
        names=[
            "preceding_trial",
            "following_trial",
            "iti_bin",
            "bin_start_s",
            "bin_end_s",
        ],
    )
    values = (
        np.vstack(counts)
        if counts
        else np.empty((0, len(unit_ids)), dtype=np.int64)
    )
    result = pd.DataFrame(values, index=index, columns=unit_ids)
    result.columns.name = "unit_id"
    return result


def inter_trial_intervals(session: CON2PHYSSession) -> nap.IntervalSet:
    """Return valid intervals from each trial end to the next trial start."""
    trials = session.trial_table
    starts = trials["trial_end"].to_numpy(dtype=float)[:-1]
    ends = trials["trial_start"].to_numpy(dtype=float)[1:]
    valid = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    if not valid.any():
        raise ValueError("session has no valid inter-trial intervals")
    return nap.IntervalSet(start=starts[valid], end=ends[valid], time_units="s")


def inter_trial_interval_table(session: CON2PHYSSession) -> pd.DataFrame:
    """Return one row per valid ITI with its duration and trial identifiers."""
    trials = session.trial_table
    starts = trials["trial_end"].to_numpy(dtype=float)[:-1]
    ends = trials["trial_start"].to_numpy(dtype=float)[1:]
    trial_ids = trials.index.to_numpy()
    valid = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    return pd.DataFrame(
        {
            "participant": session.subject,
            "preceding_trial": trial_ids[:-1][valid],
            "following_trial": trial_ids[1:][valid],
            "iti_start_s": starts[valid],
            "iti_end_s": ends[valid],
            "iti_duration_s": ends[valid] - starts[valid],
        }
    )


def bin_all_participants_iti_spikes_by_area(
    participants: list[int] | tuple[int, ...] | None = None,
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    bin_width_s: float = BIN_WIDTH_S,
) -> tuple[dict[int, dict[object, pd.DataFrame]], pd.DataFrame, pd.DataFrame]:
    """Load and bin every participant's ITI spikes, grouped by brain area.

    Participants are processed sequentially to avoid retaining all raw spike
    and waveform data in memory. Binned counts are stored as sparse integer
    DataFrames because most 5-ms unit bins contain zero spikes.

    Returns
    -------
    counts_by_participant_and_area, unit_summary, iti_table
        ``counts[participant][brain_area]`` contains rows for complete ITI
        bins and columns for units in that area. ``unit_summary`` reports the
        number of loaded units and bins for every participant/area, while
        ``iti_table`` contains every valid ITI and its duration.
    """
    if not np.isfinite(bin_width_s) or bin_width_s <= 0:
        raise ValueError("bin_width_s must be positive")
    selected = (
        available_subjects(data_path)
        if participants is None
        else list(dict.fromkeys(int(participant) for participant in participants))
    )
    if not selected:
        raise ValueError("no participants were selected or found")

    all_counts: dict[int, dict[object, pd.DataFrame]] = {}
    unit_records: list[dict[str, object]] = []
    iti_tables: list[pd.DataFrame] = []
    sparse_integer = pd.SparseDtype(np.int64, fill_value=0)
    for participant in selected:
        session = load_session(
            subject=participant,
            data_path=data_path,
            lfp_banks=(),
            load_spikes=True,
        )
        if session.spikes is None:
            raise RuntimeError(f"participant {participant} spikes were not loaded")
        if "brain_area" not in session.spikes.metadata.columns:
            raise ValueError(
                f"participant {participant} spike metadata has no brain_area"
            )

        participant_counts = bin_inter_trial_spikes(
            session, bin_width_s=bin_width_s
        )
        iti_tables.append(inter_trial_interval_table(session))
        areas = session.spikes.metadata["brain_area"]
        area_counts: dict[object, pd.DataFrame] = {}
        for area in sorted(areas.dropna().unique()):
            unit_ids = [
                int(unit_id)
                for unit_id in participant_counts.columns
                if areas.loc[unit_id] == area
            ]
            area_counts[area] = participant_counts[unit_ids].astype(sparse_integer)
            unit_records.append(
                {
                    "participant": participant,
                    "brain_area": area,
                    "n_units": len(unit_ids),
                    "n_complete_iti_bins": len(participant_counts),
                }
            )
        all_counts[participant] = area_counts

    unit_summary = pd.DataFrame.from_records(unit_records).set_index(
        ["participant", "brain_area"]
    )
    all_itis = pd.concat(iti_tables, ignore_index=True)
    return all_counts, unit_summary, all_itis


def pooled_long_iti_autocorrelation_by_area(
    participants: list[int] | tuple[int, ...] | None = None,
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    minimum_iti_s: float = MIN_LONG_ITI_S,
    bin_width_ms: float = BIN_WIDTH_S * 1_000,
    max_lag_ms: float = LONG_ITI_MAX_LAG_MS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pool unit ACFs across participants within brain areas for long ITIs.

    Each participant's unit autocorrelograms are calculated separately using
    only that participant's ITIs of at least ``minimum_iti_s``. This prevents
    correlations across ITI or participant boundaries. Unit columns retain a
    ``(participant, unit_id)`` identity before their ACFs are averaged within
    brain area. Units from participants with no qualifying ITI are excluded.

    Units with fewer than two spikes across their qualifying ITIs are also
    excluded because a spike autocorrelogram cannot be estimated for them.
    """
    if not np.isfinite(minimum_iti_s) or minimum_iti_s <= 0:
        raise ValueError("minimum_iti_s must be positive")
    if not np.isfinite(bin_width_ms) or bin_width_ms <= 0:
        raise ValueError("bin_width_ms must be positive")
    if not np.isfinite(max_lag_ms) or max_lag_ms <= 0:
        raise ValueError("max_lag_ms must be positive")
    selected = (
        available_subjects(data_path)
        if participants is None
        else list(dict.fromkeys(int(participant) for participant in participants))
    )

    unit_tables: list[pd.DataFrame] = []
    metadata_records: list[dict[str, object]] = []
    for participant in selected:
        session = load_session(
            subject=participant,
            data_path=data_path,
            lfp_banks=(),
            load_spikes=True,
        )
        if session.spikes is None:
            continue
        iti_table = inter_trial_interval_table(session)
        qualifying = iti_table.loc[iti_table["iti_duration_s"] >= minimum_iti_s]
        if qualifying.empty:
            continue
        epochs = nap.IntervalSet(
            start=qualifying["iti_start_s"].to_numpy(float),
            end=qualifying["iti_end_s"].to_numpy(float),
            time_units="s",
        )

        eligible_units: dict[int, nap.Ts] = {}
        for unit_id in session.spikes.keys():
            restricted = session.spikes[int(unit_id)].restrict(epochs)
            if len(restricted) >= 2:
                eligible_units[int(unit_id)] = restricted
        if not eligible_units:
            continue
        eligible_group = nap.TsGroup(eligible_units, time_support=epochs)
        autocorrelation_hz = nap.compute_autocorrelogram(
            group=eligible_group,
            binsize=float(bin_width_ms),
            windowsize=float(max_lag_ms),
            ep=epochs,
            norm=False,
            time_units="ms",
        )
        autocorrelation = autocorrelation_hz * (bin_width_ms / 1_000.0)
        autocorrelation.index = autocorrelation.index.to_numpy(float) * 1_000
        autocorrelation.index.name = "lag_ms"
        autocorrelation.columns = pd.MultiIndex.from_tuples(
            [(participant, int(unit_id)) for unit_id in autocorrelation.columns],
            names=["participant", "unit_id"],
        )
        unit_tables.append(autocorrelation)

        areas = session.spikes.metadata["brain_area"]
        for unit_id in eligible_units:
            metadata_records.append(
                {
                    "participant": participant,
                    "unit_id": unit_id,
                    "brain_area": areas.loc[unit_id],
                    "n_qualifying_itis": len(qualifying),
                    "qualifying_iti_duration_s": qualifying[
                        "iti_duration_s"
                    ].sum(),
                    "n_spikes_in_qualifying_itis": len(eligible_units[unit_id]),
                }
            )

    if not unit_tables:
        raise ValueError(
            "no neuron has at least two spikes in an ITI meeting the duration rule"
        )
    unit_autocorrelation = pd.concat(unit_tables, axis=1)
    unit_metadata = pd.DataFrame.from_records(metadata_records).set_index(
        ["participant", "unit_id"]
    )
    area_means: dict[object, pd.Series] = {}
    for area in sorted(unit_metadata["brain_area"].dropna().unique()):
        columns = unit_metadata.index[unit_metadata["brain_area"] == area]
        area_means[area] = unit_autocorrelation.loc[:, columns].mean(axis=1)
    area_mean_autocorrelation = pd.DataFrame(area_means)
    area_mean_autocorrelation.columns.name = "brain_area"
    return unit_autocorrelation, area_mean_autocorrelation, unit_metadata


def plot_pooled_long_iti_autocorrelation(
    area_mean_autocorrelation: pd.DataFrame,
    path: str | Path = DEFAULT_POOLED_LONG_ITI_PLOT,
    *,
    minimum_iti_s: float = MIN_LONG_ITI_S,
) -> Path:
    """Plot pooled-neuron, brain-area mean ACFs for qualifying long ITIs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    for area in area_mean_autocorrelation.columns:
        axis.plot(
            area_mean_autocorrelation.index,
            area_mean_autocorrelation[area],
            linewidth=1.5,
            label=f"Brain area {area}",
        )
    axis.axvline(0, color="0.4", linestyle="--", linewidth=1, alpha=0.7)
    axis.set(
        xlabel="Lag (ms)",
        ylabel="Mean spike count per reference spike (5-ms bin)",
        title=rf"All eligible neurons: ITI $\geq$ {minimum_iti_s:g} s autocorrelation",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pooled_long_iti_log_autocorrelation(
    area_mean_autocorrelation: pd.DataFrame,
    path: str | Path = DEFAULT_POOLED_LONG_ITI_LOG_PLOT,
    *,
    minimum_iti_s: float = MIN_LONG_ITI_S,
) -> Path:
    """Plot log mean ACF against positive lag separately by brain area."""
    positive = area_mean_autocorrelation.loc[
        area_mean_autocorrelation.index.to_numpy(float) > 0
    ]
    if positive.empty:
        raise ValueError("no positive autocorrelation lags are available")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    areas = list(positive.columns)
    fig, axes = plt.subplots(
        1, len(areas), figsize=(5 * len(areas), 4.5), sharex=True, squeeze=False
    )
    for axis, area in zip(axes[0], areas):
        values = positive[area].to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0)
        axis.plot(
            positive.index.to_numpy(dtype=float)[valid],
            np.log(values[valid]),
            linewidth=1.25,
        )
        axis.set(
            title=f"Brain area {area}",
            xlabel="Positive lag (ms)",
        )
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("ln(mean spike-count autocorrelation)")
    fig.suptitle(
        rf"All eligible neurons: log mean autocorrelation for ITIs $\geq$ "
        rf"{minimum_iti_s:g} s"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _decode_exponential_mixture(
    parameters: np.ndarray,
    n_exponents: int,
    minimum_amplitude_share: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Decode constrained mixture parameters used by least squares."""
    intercept = float(parameters[0])
    total_amplitude = float(np.exp(parameters[1]))
    free_logits = parameters[2 : 2 + n_exponents - 1]
    logits = np.r_[free_logits, 0.0]
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    shares = minimum_amplitude_share + (
        1.0 - n_exponents * minimum_amplitude_share
    ) * probabilities
    amplitudes = total_amplitude * shares
    log_tau_start = 2 + n_exponents - 1
    timescales_ms = np.exp(
        parameters[log_tau_start : log_tau_start + n_exponents]
    )
    order = np.argsort(timescales_ms)
    return intercept, amplitudes[order], timescales_ms[order]


def _evaluate_exponential_mixture(
    lag_ms: np.ndarray,
    intercept: float,
    amplitudes: np.ndarray,
    timescales_ms: np.ndarray,
) -> np.ndarray:
    return intercept + np.sum(
        amplitudes[None, :] * np.exp(-lag_ms[:, None] / timescales_ms[None, :]),
        axis=1,
    )


def fit_exponential_mixtures_by_bic(
    area_mean_autocorrelation: pd.DataFrame,
    *,
    maximum_exponents: int = 3,
    minimum_amplitude_share: float = 0.01,
    n_initializations: int = 30,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit 1--3 exponentials per area and select the minimum-BIC model.

    Every model is ``b + sum(c_i * exp(-lag/tau_i))``. The parameterization
    guarantees ``c_i / sum(c) >= minimum_amplitude_share`` during fitting.
    BIC is ``n*log(RSS/n) + p*log(n)``, with ``p = 2*K + 1`` including the
    intercept, K amplitudes, and K timescales.
    """
    if maximum_exponents not in (1, 2, 3):
        raise ValueError("maximum_exponents must be 1, 2, or 3")
    if not 0 < minimum_amplitude_share < 1 / maximum_exponents:
        raise ValueError("minimum_amplitude_share is incompatible with model size")
    if n_initializations < 1:
        raise ValueError("n_initializations must be positive")

    positive = area_mean_autocorrelation.loc[
        area_mean_autocorrelation.index.to_numpy(float) > 0
    ]
    lag_ms = positive.index.to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    model_records: list[dict[str, object]] = []
    component_records: list[dict[str, object]] = []
    predictions: dict[object, np.ndarray] = {}

    for area in positive.columns:
        values = positive[area].to_numpy(dtype=float)
        valid = np.isfinite(lag_ms) & np.isfinite(values) & (values >= 0)
        x = lag_ms[valid]
        y = values[valid]
        if len(y) < 8:
            raise ValueError(f"brain area {area} has too few valid positive lags")
        max_tau_ms = max(float(x.max()) * 10.0, 100.0)
        area_models: list[dict[str, object]] = []

        for n_exponents in range(1, maximum_exponents + 1):
            n_logits = n_exponents - 1
            lower = np.r_[
                0.0,
                np.log(np.finfo(float).tiny),
                np.full(n_logits, -20.0),
                np.full(n_exponents, np.log(0.5)),
            ]
            upper = np.r_[
                np.inf,
                np.log(max(float(y.max()) * 100.0, 1.0)),
                np.full(n_logits, 20.0),
                np.full(n_exponents, np.log(max_tau_ms)),
            ]
            best_result = None
            best_rss = np.inf
            for initialization in range(n_initializations):
                if initialization == 0:
                    intercept_0 = max(float(y[-1]), 0.0)
                    amplitude_0 = max(float(y[0] - intercept_0), 1e-8)
                    tau_0 = np.geomspace(20.0, min(max_tau_ms, x.max()), n_exponents)
                    logits_0 = np.zeros(n_logits)
                else:
                    intercept_0 = rng.uniform(0.0, max(float(y.min()), 1e-8))
                    amplitude_0 = rng.uniform(
                        max(float(np.ptp(y)) * 0.25, 1e-8),
                        max(float(y.max()) * 2.0, 2e-8),
                    )
                    tau_0 = np.exp(
                        rng.uniform(np.log(1.0), np.log(max_tau_ms), n_exponents)
                    )
                    logits_0 = rng.normal(0.0, 1.5, n_logits)
                initial = np.r_[
                    intercept_0,
                    np.log(amplitude_0),
                    logits_0,
                    np.log(tau_0),
                ]

                def residuals(parameters: np.ndarray) -> np.ndarray:
                    intercept, amplitudes, timescales = _decode_exponential_mixture(
                        parameters, n_exponents, minimum_amplitude_share
                    )
                    return _evaluate_exponential_mixture(
                        x, intercept, amplitudes, timescales
                    ) - y

                result = least_squares(
                    residuals,
                    initial,
                    bounds=(lower, upper),
                    max_nfev=20_000,
                )
                rss = float(np.sum(result.fun**2))
                if result.success and rss < best_rss:
                    best_result = result
                    best_rss = rss
            if best_result is None:
                raise RuntimeError(
                    f"all {n_exponents}-exponential fits failed for area {area}"
                )

            intercept, amplitudes, timescales = _decode_exponential_mixture(
                best_result.x, n_exponents, minimum_amplitude_share
            )
            n_observations = len(y)
            n_parameters = 2 * n_exponents + 1
            bic = n_observations * np.log(
                max(best_rss / n_observations, np.finfo(float).tiny)
            ) + n_parameters * np.log(n_observations)
            total_ss = float(np.sum((y - y.mean()) ** 2))
            r_squared = 1.0 - best_rss / total_ss if total_ss > 0 else np.nan
            area_models.append(
                {
                    "brain_area": area,
                    "n_exponents": n_exponents,
                    "intercept": intercept,
                    "rss": best_rss,
                    "bic": bic,
                    "r_squared": r_squared,
                    "n_observations": n_observations,
                    "n_parameters": n_parameters,
                    "amplitudes": amplitudes,
                    "timescales_ms": timescales,
                }
            )

        selected = min(area_models, key=lambda model: float(model["bic"]))
        predictions[area] = _evaluate_exponential_mixture(
            lag_ms,
            float(selected["intercept"]),
            np.asarray(selected["amplitudes"]),
            np.asarray(selected["timescales_ms"]),
        )
        for model in area_models:
            model_records.append(
                {
                    key: value
                    for key, value in model.items()
                    if key not in {"amplitudes", "timescales_ms"}
                }
                | {"selected": model is selected}
            )
            amplitudes = np.asarray(model["amplitudes"])
            timescales = np.asarray(model["timescales_ms"])
            for component, (amplitude, tau_ms) in enumerate(
                zip(amplitudes, timescales), start=1
            ):
                component_records.append(
                    {
                        "brain_area": area,
                        "n_exponents": model["n_exponents"],
                        "component": component,
                        "amplitude": amplitude,
                        "amplitude_share": amplitude / amplitudes.sum(),
                        "tau_ms": tau_ms,
                        "selected": model is selected,
                    }
                )

    model_summary = pd.DataFrame.from_records(model_records).set_index(
        ["brain_area", "n_exponents"]
    )
    components = pd.DataFrame.from_records(component_records).set_index(
        ["brain_area", "n_exponents", "component"]
    )
    fitted = pd.DataFrame(predictions, index=positive.index)
    fitted.columns.name = "brain_area"
    return model_summary, components, fitted


def plot_bic_selected_exponential_mixtures(
    area_mean_autocorrelation: pd.DataFrame,
    fitted_autocorrelation: pd.DataFrame,
    model_summary: pd.DataFrame,
    components: pd.DataFrame,
    path: str | Path = DEFAULT_MULTIEXPONENTIAL_PLOT,
) -> Path:
    """Plot each area's positive-lag ACF and BIC-selected mixture fit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    areas = list(fitted_autocorrelation.columns)
    fig, axes = plt.subplots(
        1, len(areas), figsize=(5 * len(areas), 4.5), sharex=True, squeeze=False
    )
    for axis, area in zip(axes[0], areas):
        observed = area_mean_autocorrelation.loc[
            area_mean_autocorrelation.index.to_numpy(float) > 0, area
        ]
        selected_row = model_summary.loc[area].query("selected").iloc[0]
        n_exponents = int(selected_row.name)
        selected_components = components.loc[(area, n_exponents)]
        tau_text = ", ".join(
            rf"$\tau_{component}$={row['tau_ms']:.1f} ms"
            for component, row in selected_components.iterrows()
        )
        axis.plot(observed.index, observed, linewidth=1.2, label="Mean ACF")
        axis.plot(
            fitted_autocorrelation.index,
            fitted_autocorrelation[area],
            linewidth=2,
            label=f"Selected {n_exponents}-exp fit",
        )
        axis.set_title(
            f"Brain area {area}: BIC={selected_row['bic']:.1f}\n{tau_text}"
        )
        axis.set_xlabel("Positive lag (ms)")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Mean spike count per reference spike")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("BIC-selected exponential-mixture autocorrelation fits")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bic_selected_exponential_mixtures_loglog(
    area_mean_autocorrelation: pd.DataFrame,
    fitted_autocorrelation: pd.DataFrame,
    model_summary: pd.DataFrame,
    components: pd.DataFrame,
    path: str | Path = DEFAULT_MULTIEXPONENTIAL_LOGLOG_PLOT,
) -> Path:
    """Plot BIC-selected fits with logarithmic lag and ACF axes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    areas = list(fitted_autocorrelation.columns)
    fig, axes = plt.subplots(
        1, len(areas), figsize=(5 * len(areas), 4.5), sharex=True, squeeze=False
    )
    for axis, area in zip(axes[0], areas):
        observed = area_mean_autocorrelation.loc[
            area_mean_autocorrelation.index.to_numpy(float) > 0, area
        ]
        valid_observed = np.isfinite(observed.to_numpy(float)) & (
            observed.to_numpy(float) > 0
        )
        fitted = fitted_autocorrelation[area]
        valid_fitted = np.isfinite(fitted.to_numpy(float)) & (
            fitted.to_numpy(float) > 0
        )
        selected_row = model_summary.loc[area].query("selected").iloc[0]
        n_exponents = int(selected_row.name)
        selected_components = components.loc[(area, n_exponents)]
        tau_text = ", ".join(
            rf"$\tau_{component}$={row['tau_ms']:.1f} ms"
            for component, row in selected_components.iterrows()
        )
        axis.plot(
            observed.index.to_numpy(float)[valid_observed],
            observed.to_numpy(float)[valid_observed],
            linewidth=1.2,
            label="Mean ACF",
        )
        axis.plot(
            fitted.index.to_numpy(float)[valid_fitted],
            fitted.to_numpy(float)[valid_fitted],
            linewidth=2,
            label=f"Selected {n_exponents}-exp fit",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"Brain area {area}\n{tau_text}")
        axis.set_xlabel("Positive lag (ms, log scale)")
        axis.grid(which="both", alpha=0.2)
    axes[0, 0].set_ylabel("Mean spike-count autocorrelation (log scale)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("BIC-selected exponential-mixture fits on log–log axes")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bic_selected_exponential_mixtures_log_y(
    area_mean_autocorrelation: pd.DataFrame,
    fitted_autocorrelation: pd.DataFrame,
    model_summary: pd.DataFrame,
    components: pd.DataFrame,
    path: str | Path = DEFAULT_MULTIEXPONENTIAL_LOGY_PLOT,
) -> Path:
    """Plot ln(ACF) and ln(selected fit) against linear positive lag."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    areas = list(fitted_autocorrelation.columns)
    fig, axes = plt.subplots(
        1, len(areas), figsize=(5 * len(areas), 4.5), sharex=True, squeeze=False
    )
    for axis, area in zip(axes[0], areas):
        observed = area_mean_autocorrelation.loc[
            area_mean_autocorrelation.index.to_numpy(float) > 0, area
        ]
        fitted = fitted_autocorrelation[area]
        valid_observed = np.isfinite(observed.to_numpy(float)) & (
            observed.to_numpy(float) > 0
        )
        valid_fitted = np.isfinite(fitted.to_numpy(float)) & (
            fitted.to_numpy(float) > 0
        )
        selected_row = model_summary.loc[area].query("selected").iloc[0]
        n_exponents = int(selected_row.name)
        selected_components = components.loc[(area, n_exponents)]
        tau_text = ", ".join(
            rf"$\tau_{component}$={row['tau_ms']:.1f} ms"
            for component, row in selected_components.iterrows()
        )
        axis.plot(
            observed.index.to_numpy(float)[valid_observed],
            np.log(observed.to_numpy(float)[valid_observed]),
            linewidth=1.2,
            label="ln(mean ACF)",
        )
        axis.plot(
            fitted.index.to_numpy(float)[valid_fitted],
            np.log(fitted.to_numpy(float)[valid_fitted]),
            linewidth=2,
            label=f"ln(selected {n_exponents}-exp fit)",
        )
        axis.set_title(f"Brain area {area}\n{tau_text}")
        axis.set_xlabel("Positive lag (ms)")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("ln(mean spike-count autocorrelation)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Log autocorrelation with BIC-selected exponential-mixture fits")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def bootstrap_effective_timescales(
    unit_autocorrelation: pd.DataFrame,
    unit_metadata: pd.DataFrame,
    model_summary: pd.DataFrame,
    components: pd.DataFrame,
    *,
    minimum_amplitude_share: float = 0.01,
    n_bootstrap: int = 1_000,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate amplitude-weighted effective timescales and neuron-bootstrap CIs.

    Each bootstrap sample resamples neurons with replacement within a brain
    area, recalculates its mean ACF, and refits the number of exponentials
    selected by BIC on the original mean ACF. The effective timescale is
    ``sum(c_i * tau_i) / sum(c_i)``.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie between zero and one")

    positive = unit_autocorrelation.loc[
        unit_autocorrelation.index.to_numpy(float) > 0
    ]
    lag_ms = positive.index.to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    bootstrap_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for area in sorted(unit_metadata["brain_area"].dropna().unique()):
        selected_row = model_summary.loc[area].query("selected").iloc[0]
        n_exponents = int(selected_row.name)
        selected_components = components.loc[(area, n_exponents)]
        amplitudes = selected_components["amplitude"].to_numpy(float)
        timescales = selected_components["tau_ms"].to_numpy(float)
        intercept = float(selected_row["intercept"])
        point_effective_tau = float(
            np.sum(amplitudes * timescales) / np.sum(amplitudes)
        )

        shares = amplitudes / amplitudes.sum()
        if n_exponents > 1:
            probabilities = (
                shares - minimum_amplitude_share
            ) / (1.0 - n_exponents * minimum_amplitude_share)
            probabilities = np.maximum(probabilities, np.finfo(float).eps)
            logits = np.log(probabilities[:-1] / probabilities[-1])
        else:
            logits = np.empty(0)
        initial = np.r_[
            intercept,
            np.log(amplitudes.sum()),
            logits,
            np.log(timescales),
        ]
        n_logits = n_exponents - 1
        max_tau_ms = max(float(lag_ms.max()) * 10.0, 100.0)
        lower = np.r_[
            0.0,
            np.log(np.finfo(float).tiny),
            np.full(n_logits, -20.0),
            np.full(n_exponents, np.log(0.5)),
        ]
        upper = np.r_[
            np.inf,
            np.log(100.0),
            np.full(n_logits, 20.0),
            np.full(n_exponents, np.log(max_tau_ms)),
        ]

        area_columns = unit_metadata.index[unit_metadata["brain_area"] == area]
        area_unit_acfs = positive.loc[:, area_columns].to_numpy(dtype=float)
        n_units = area_unit_acfs.shape[1]
        successful_taus: list[float] = []
        for bootstrap_id in range(n_bootstrap):
            sampled = rng.integers(0, n_units, size=n_units)
            bootstrap_mean = np.nanmean(area_unit_acfs[:, sampled], axis=1)
            valid = np.isfinite(lag_ms) & np.isfinite(bootstrap_mean)
            x = lag_ms[valid]
            y = bootstrap_mean[valid]

            def residuals(parameters: np.ndarray) -> np.ndarray:
                fit_intercept, fit_amplitudes, fit_timescales = (
                    _decode_exponential_mixture(
                        parameters, n_exponents, minimum_amplitude_share
                    )
                )
                return _evaluate_exponential_mixture(
                    x, fit_intercept, fit_amplitudes, fit_timescales
                ) - y

            result = least_squares(
                residuals,
                initial,
                bounds=(lower, upper),
                max_nfev=5_000,
            )
            if not result.success:
                continue
            _, fit_amplitudes, fit_timescales = _decode_exponential_mixture(
                result.x, n_exponents, minimum_amplitude_share
            )
            effective_tau = float(
                np.sum(fit_amplitudes * fit_timescales) / fit_amplitudes.sum()
            )
            if np.isfinite(effective_tau):
                successful_taus.append(effective_tau)
                bootstrap_records.append(
                    {
                        "brain_area": area,
                        "bootstrap": bootstrap_id,
                        "effective_tau_ms": effective_tau,
                    }
                )

        if not successful_taus:
            raise RuntimeError(f"all effective-timescale bootstraps failed for area {area}")
        alpha = (1.0 - confidence_level) / 2.0
        bootstrap_values = np.asarray(successful_taus)
        summary_records.append(
            {
                "brain_area": area,
                "n_units": n_units,
                "n_exponents": n_exponents,
                "point_effective_tau_ms": point_effective_tau,
                "bootstrap_mean_tau_ms": bootstrap_values.mean(),
                "ci95_low_ms": np.quantile(bootstrap_values, alpha),
                "ci95_high_ms": np.quantile(bootstrap_values, 1.0 - alpha),
                "bootstrap_sd_ms": bootstrap_values.std(ddof=1),
                "n_bootstrap_successful": len(bootstrap_values),
            }
        )

    summary = pd.DataFrame.from_records(summary_records).set_index("brain_area")
    bootstrap_distribution = pd.DataFrame.from_records(bootstrap_records).set_index(
        ["brain_area", "bootstrap"]
    )
    return summary, bootstrap_distribution


def pairwise_effective_timescale_tests(
    effective_timescale_summary: pd.DataFrame,
    bootstrap_distribution: pd.DataFrame,
    *,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Test pairwise area differences with centered neuron bootstraps.

    Raw two-sided p-values compare the absolute observed contrast with the
    centered bootstrap null distribution. A plus-one correction prevents a
    zero Monte Carlo p-value. Holm adjustment controls family-wise error over
    all brain-area pairs.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie between zero and one")
    areas = list(effective_timescale_summary.index)
    records: list[dict[str, object]] = []
    alpha = (1.0 - confidence_level) / 2.0
    for area_a, area_b in combinations(areas, 2):
        values_a = bootstrap_distribution.loc[area_a, "effective_tau_ms"]
        values_b = bootstrap_distribution.loc[area_b, "effective_tau_ms"]
        common = values_a.index.intersection(values_b.index)
        differences = (
            values_a.loc[common].to_numpy(float)
            - values_b.loc[common].to_numpy(float)
        )
        observed = float(
            effective_timescale_summary.loc[area_a, "point_effective_tau_ms"]
            - effective_timescale_summary.loc[area_b, "point_effective_tau_ms"]
        )
        centered_null = differences - differences.mean()
        exceedances = int(np.sum(np.abs(centered_null) >= abs(observed)))
        p_value = (exceedances + 1) / (len(centered_null) + 1)
        records.append(
            {
                "brain_area_a": area_a,
                "brain_area_b": area_b,
                "difference_a_minus_b_ms": observed,
                "ci95_low_ms": np.quantile(differences, alpha),
                "ci95_high_ms": np.quantile(differences, 1.0 - alpha),
                "p_value": p_value,
                "n_bootstrap": len(differences),
            }
        )

    result = pd.DataFrame.from_records(records)
    order = np.argsort(result["p_value"].to_numpy(float))
    adjusted = np.empty(len(result), dtype=float)
    running_max = 0.0
    for rank, position in enumerate(order):
        candidate = min((len(result) - rank) * result.loc[position, "p_value"], 1.0)
        running_max = max(running_max, candidate)
        adjusted[position] = running_max
    result["p_value_holm"] = adjusted
    result["significant_0.05"] = result["p_value_holm"] < 0.05
    return result


def fit_individual_neuron_timescales_by_bic(
    unit_autocorrelation: pd.DataFrame,
    unit_metadata: pd.DataFrame,
    *,
    maximum_exponents: int = 3,
    minimum_amplitude_share: float = 0.01,
    n_initializations: int = 5,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select an exponential mixture and effective timescale for every neuron."""
    positive = unit_autocorrelation.loc[
        unit_autocorrelation.index.to_numpy(float) > 0
    ]
    fitted_records: list[dict[str, object]] = []
    excluded_records: list[dict[str, object]] = []
    for neuron_number, column in enumerate(positive.columns):
        participant, unit_id = column
        values = positive[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if (
            len(finite) < 8
            or not np.any(finite > 0)
            or float(np.ptp(finite)) <= np.finfo(float).eps
        ):
            excluded_records.append(
                {
                    "participant": participant,
                    "unit_id": unit_id,
                    "brain_area": unit_metadata.loc[column, "brain_area"],
                    "reason": "no estimable positive-lag autocorrelation variation",
                }
            )
            continue
        try:
            neuron_table = pd.DataFrame(
                {"neuron": values}, index=positive.index
            )
            models, components, _ = fit_exponential_mixtures_by_bic(
                neuron_table,
                maximum_exponents=maximum_exponents,
                minimum_amplitude_share=minimum_amplitude_share,
                n_initializations=n_initializations,
                random_seed=random_seed + neuron_number,
            )
            selected = models.loc["neuron"].query("selected").iloc[0]
            n_exponents = int(selected.name)
            selected_components = components.loc[("neuron", n_exponents)]
            amplitudes = selected_components["amplitude"].to_numpy(float)
            timescales = selected_components["tau_ms"].to_numpy(float)
            effective_tau = float(
                np.sum(amplitudes * timescales) / amplitudes.sum()
            )
            fitted_records.append(
                {
                    "participant": participant,
                    "unit_id": unit_id,
                    "brain_area": unit_metadata.loc[column, "brain_area"],
                    "n_spikes_in_qualifying_itis": unit_metadata.loc[
                        column, "n_spikes_in_qualifying_itis"
                    ],
                    "n_exponents": n_exponents,
                    "effective_tau_ms": effective_tau,
                    "minimum_tau_ms": timescales.min(),
                    "maximum_tau_ms": timescales.max(),
                    "intercept": selected["intercept"],
                    "bic": selected["bic"],
                    "r_squared": selected["r_squared"],
                    "timescale_at_upper_bound": bool(
                        timescales.max() >= positive.index.max() * 10 * 0.999
                    ),
                }
            )
        except (RuntimeError, ValueError, FloatingPointError) as error:
            excluded_records.append(
                {
                    "participant": participant,
                    "unit_id": unit_id,
                    "brain_area": unit_metadata.loc[column, "brain_area"],
                    "reason": f"fit failed: {error}",
                }
            )

    if not fitted_records:
        raise RuntimeError("no individual-neuron timescale fit succeeded")
    fitted = pd.DataFrame.from_records(fitted_records).set_index(
        ["participant", "unit_id"]
    )
    excluded = pd.DataFrame.from_records(excluded_records)
    if not excluded.empty:
        excluded = excluded.set_index(["participant", "unit_id"])
    return fitted, excluded


def summarize_individual_neuron_timescales(
    individual_timescales: pd.DataFrame,
    *,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Return area means and neuron-bootstrap CIs of individual timescales."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(random_seed)
    alpha = (1.0 - confidence_level) / 2.0
    records: list[dict[str, object]] = []
    for area, group in individual_timescales.groupby("brain_area", sort=True):
        values = group["effective_tau_ms"].to_numpy(dtype=float)
        bootstrap_means = np.mean(
            values[rng.integers(0, len(values), size=(n_bootstrap, len(values)))],
            axis=1,
        )
        records.append(
            {
                "brain_area": area,
                "n_neurons": len(values),
                "mean_effective_tau_ms": values.mean(),
                "median_effective_tau_ms": np.median(values),
                "sd_effective_tau_ms": values.std(ddof=1),
                "ci95_low_ms": np.quantile(bootstrap_means, alpha),
                "ci95_high_ms": np.quantile(bootstrap_means, 1.0 - alpha),
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame.from_records(records).set_index("brain_area")


def test_area_1_lower_individual_timescales(
    individual_timescales: pd.DataFrame,
) -> pd.DataFrame:
    """Run one-sided Welch tests that area 1 has a lower mean than other areas."""
    area_1 = individual_timescales.loc[
        individual_timescales["brain_area"] == 1, "effective_tau_ms"
    ].to_numpy(float)
    records: list[dict[str, object]] = []
    for comparison_area in sorted(
        area
        for area in individual_timescales["brain_area"].unique()
        if area != 1
    ):
        comparison = individual_timescales.loc[
            individual_timescales["brain_area"] == comparison_area,
            "effective_tau_ms",
        ].to_numpy(float)
        test = stats.ttest_ind(
            area_1,
            comparison,
            equal_var=False,
            alternative="less",
        )
        pooled_sd = np.sqrt(
            (
                (len(area_1) - 1) * area_1.var(ddof=1)
                + (len(comparison) - 1) * comparison.var(ddof=1)
            )
            / (len(area_1) + len(comparison) - 2)
        )
        correction = 1.0 - 3.0 / (
            4.0 * (len(area_1) + len(comparison)) - 9.0
        )
        records.append(
            {
                "comparison": f"area 1 < area {comparison_area}",
                "comparison_area": comparison_area,
                "mean_difference_ms": area_1.mean() - comparison.mean(),
                "welch_t": test.statistic,
                "welch_df": test.df,
                "p_value_one_sided": test.pvalue,
                "hedges_g": correction
                * (area_1.mean() - comparison.mean())
                / pooled_sd,
            }
        )
    result = pd.DataFrame.from_records(records)
    order = np.argsort(result["p_value_one_sided"].to_numpy(float))
    adjusted = np.empty(len(result), dtype=float)
    running_max = 0.0
    for rank, position in enumerate(order):
        candidate = min(
            (len(result) - rank) * result.loc[position, "p_value_one_sided"],
            1.0,
        )
        running_max = max(running_max, candidate)
        adjusted[position] = running_max
    result["p_value_holm"] = adjusted
    result["significant_0.05"] = result["p_value_holm"] < 0.05
    return result.set_index("comparison")


def mean_iti_spike_count_autocorrelation_by_area(
    session: CON2PHYSSession,
    *,
    bin_width_ms: float = BIN_WIDTH_S * 1_000,
    max_lag_ms: float = MAX_LAG_MS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate unit and brain-area mean ITI spike-count autocorrelations.

    Pynapple computes each unit's autocorrelogram only within the supplied ITI
    epochs. Its unnormalized output is a conditional firing rate in Hz; this
    is multiplied by the bin duration to express each value as the mean spike
    count per reference spike in that lag bin. The zero-lag self-spike is
    excluded by Pynapple.

    Returns
    -------
    unit_autocorrelation, area_mean_autocorrelation : tuple of DataFrames
        The first table has one column per unit. The second has one column per
        brain area and is the arithmetic mean across units in that area.
        Both indexes are lags in milliseconds.
    """
    if session.spikes is None:
        raise ValueError("session must contain spiking data")
    if "brain_area" not in session.spikes.metadata.columns:
        raise ValueError("spike metadata must contain brain_area")
    if not np.isfinite(bin_width_ms) or bin_width_ms <= 0:
        raise ValueError("bin_width_ms must be positive")
    if not np.isfinite(max_lag_ms) or max_lag_ms <= 0:
        raise ValueError("max_lag_ms must be positive")
    if max_lag_ms < bin_width_ms:
        raise ValueError("max_lag_ms must be at least bin_width_ms")

    autocorrelation_hz = nap.compute_autocorrelogram(
        group=session.spikes,
        binsize=float(bin_width_ms),
        windowsize=float(max_lag_ms),
        ep=inter_trial_intervals(session),
        norm=False,
        time_units="ms",
    )
    unit_autocorrelation = autocorrelation_hz * (bin_width_ms / 1_000.0)
    # Pynapple reports the lag index in seconds regardless of parameter units.
    unit_autocorrelation.index = unit_autocorrelation.index.to_numpy(float) * 1_000
    unit_autocorrelation.index.name = "lag_ms"
    unit_autocorrelation.columns.name = "unit_id"

    areas = session.spikes.metadata["brain_area"]
    area_means: dict[object, pd.Series] = {}
    for area in sorted(areas.dropna().unique()):
        unit_ids = [
            unit_id
            for unit_id in unit_autocorrelation.columns
            if areas.loc[unit_id] == area
        ]
        if unit_ids:
            area_means[area] = unit_autocorrelation[unit_ids].mean(axis=1)
    area_mean_autocorrelation = pd.DataFrame(area_means)
    area_mean_autocorrelation.columns.name = "brain_area"
    return unit_autocorrelation, area_mean_autocorrelation


def plot_mean_iti_autocorrelation_by_area(
    area_mean_autocorrelation: pd.DataFrame,
    path: str | Path = DEFAULT_AUTOCORRELATION_PLOT,
) -> Path:
    """Plot the mean ITI spike-count autocorrelation for each brain area."""
    if area_mean_autocorrelation.empty:
        raise ValueError("area_mean_autocorrelation must not be empty")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    for area in area_mean_autocorrelation.columns:
        axis.plot(
            area_mean_autocorrelation.index,
            area_mean_autocorrelation[area],
            linewidth=2,
            label=f"Brain area {area}",
        )
    axis.axvline(0, color="0.4", linestyle="--", linewidth=1, alpha=0.7)
    axis.set(
        xlabel="Lag (ms)",
        ylabel=f"Mean spike count per reference spike ({BIN_WIDTH_S * 1_000:g}-ms bin)",
        title="Patient 1: mean ITI spike-count autocorrelation",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _exponential_autocorrelation(
    lag_ms: np.ndarray,
    intercept: float,
    amplitude: float,
    tau_ms: float,
) -> np.ndarray:
    """Exponential autocorrelation model with a nonzero asymptote."""
    return intercept + amplitude * np.exp(-lag_ms / tau_ms)


def fit_intrinsic_timescales(
    area_mean_autocorrelation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit positive-lag area means with ``b + A * exp(-lag / tau)``.

    The intercept, amplitude, and timescale are constrained to be nonnegative.
    Standard errors are derived from the nonlinear least-squares covariance
    matrix and should be treated cautiously when a parameter reaches a bound.
    """
    positive = area_mean_autocorrelation.loc[
        area_mean_autocorrelation.index.to_numpy(float) > 0
    ]
    if len(positive) < 4:
        raise ValueError("at least four positive autocorrelation lags are required")

    lag_ms = positive.index.to_numpy(dtype=float)
    fitted = pd.DataFrame(index=positive.index)
    fitted.index.name = "lag_ms"
    records: list[dict[str, object]] = []
    for area in positive.columns:
        values = positive[area].to_numpy(dtype=float)
        valid = np.isfinite(lag_ms) & np.isfinite(values)
        x = lag_ms[valid]
        y = values[valid]
        if len(y) < 4:
            raise ValueError(f"brain area {area} has too few finite positive lags")

        initial_intercept = max(float(y[-1]), 0.0)
        initial_amplitude = max(float(y[0] - initial_intercept), 1e-8)
        parameters, covariance = curve_fit(
            _exponential_autocorrelation,
            x,
            y,
            p0=(initial_intercept, initial_amplitude, 200.0),
            bounds=([0.0, 0.0, np.finfo(float).eps], [np.inf, np.inf, np.inf]),
            maxfev=100_000,
        )
        intercept, amplitude, tau_ms = parameters
        errors = np.sqrt(np.diag(covariance))
        prediction = _exponential_autocorrelation(x, *parameters)
        residual_ss = float(np.sum((y - prediction) ** 2))
        total_ss = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else np.nan

        fitted[area] = _exponential_autocorrelation(lag_ms, *parameters)
        records.append(
            {
                "brain_area": area,
                "intercept": intercept,
                "intercept_se": errors[0],
                "amplitude": amplitude,
                "amplitude_se": errors[1],
                "tau_ms": tau_ms,
                "tau_se_ms": errors[2],
                "r_squared": r_squared,
                "n_lags": len(y),
                "intercept_at_lower_bound": bool(intercept < 1e-10),
            }
        )

    fitted.columns.name = "brain_area"
    summary = pd.DataFrame.from_records(records).set_index("brain_area")
    return summary, fitted


def plot_intrinsic_timescale_fits(
    area_mean_autocorrelation: pd.DataFrame,
    fitted_autocorrelation: pd.DataFrame,
    fit_summary: pd.DataFrame,
    path: str | Path = DEFAULT_TIMESCALE_PLOT,
) -> Path:
    """Plot positive-lag area means and their exponential timescale fits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    areas = list(fitted_autocorrelation.columns)
    fig, axes = plt.subplots(
        1, len(areas), figsize=(5 * len(areas), 4.5), sharex=True, squeeze=False
    )
    for axis, area in zip(axes[0], areas):
        observed = area_mean_autocorrelation.loc[
            area_mean_autocorrelation.index.to_numpy(float) > 0, area
        ]
        axis.plot(observed.index, observed, "o", markersize=4, label="Mean ACF")
        axis.plot(
            fitted_autocorrelation.index,
            fitted_autocorrelation[area],
            linewidth=2,
            label="Exponential fit",
        )
        fit = fit_summary.loc[area]
        axis.set_title(
            f"Brain area {area}\n"
            rf"$\tau$ = {fit['tau_ms']:.1f} $\pm$ {fit['tau_se_ms']:.1f} ms, "
            rf"$R^2$ = {fit['r_squared']:.3f}"
        )
        axis.set_xlabel("Positive lag (ms)")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Mean spike count per reference spike")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Patient 1: intrinsic neuronal timescale fits")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    patient_1 = load_patient_1_spikes()
    if patient_1.spikes is None:
        raise RuntimeError("patient 1 spiking data were not loaded")
    iti_spike_counts = bin_inter_trial_spikes(patient_1)
    unit_autocorrelation, area_mean_autocorrelation = (
        mean_iti_spike_count_autocorrelation_by_area(patient_1)
    )
    autocorrelation_plot = plot_mean_iti_autocorrelation_by_area(
        area_mean_autocorrelation
    )
    timescale_summary, fitted_autocorrelation = fit_intrinsic_timescales(
        area_mean_autocorrelation
    )
    timescale_plot = plot_intrinsic_timescale_fits(
        area_mean_autocorrelation,
        fitted_autocorrelation,
        timescale_summary,
    )
    print(f"Loaded {len(patient_1.spikes)} spike trains for patient {PATIENT}.")
    print(
        f"Binned the inter-trial intervals into {len(iti_spike_counts)} "
        f"complete {BIN_WIDTH_S * 1_000:g}-ms bins."
    )
    print("\nMean ITI spike-count autocorrelation by brain area:")
    print(area_mean_autocorrelation)
    print(f"\nSaved plot to {autocorrelation_plot}")
    print("\nPositive-lag exponential intrinsic timescale fits:")
    print(timescale_summary)
    print(f"\nSaved timescale plot to {timescale_plot}")
