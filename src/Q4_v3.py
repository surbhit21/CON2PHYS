"""Shift-predictor-corrected cross-correlograms for participant 1.

Observed and trial-shuffled cross-correlograms are calculated with Pynapple
for each behavioral checkpoint. Trials are mismatched only within their joint
behavioral condition (variable_A, variable_B, variable_C).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

if __package__:
    from .Q4_v2 import CONDITION_COLUMNS, EVENT_COLUMNS, load_participant_spikes
    from .data_loader import CON2PHYSSession
else:
    from Q4_v2 import CONDITION_COLUMNS, EVENT_COLUMNS, load_participant_spikes
    from data_loader import CON2PHYSSession


PARTICIPANT = 1
PERI_EVENT_WINDOW_S = (-1.0, 1.0)
CCG_BIN_SIZE_MS = 10.0
CCG_MAX_LAG_MS = 100.0
N_SHIFT_PREDICTORS = 20
SIGNIFICANCE_LEVEL = 0.05
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "analysis" / "Q4_v3"


def _aligned_trial_spikes(
    session: CON2PHYSSession, event_column: str
) -> tuple[list[int], dict[int, list[np.ndarray]], np.ndarray]:
    """Align every unit/trial to one checkpoint with Pynapple."""
    all_event_times = session.trial_table[event_column].to_numpy(dtype=float)
    valid_trials = np.flatnonzero(np.isfinite(all_event_times))
    if not len(valid_trials):
        raise ValueError(f"{event_column} has no finite event times")
    event_times = all_event_times[valid_trials]
    sorted_order = np.argsort(event_times, kind="stable")
    sorted_position = np.empty(len(event_times), dtype=int)
    sorted_position[sorted_order] = np.arange(len(event_times))
    events = nap.Ts(t=event_times, time_units="s")
    event_epoch = nap.IntervalSet(
        start=[event_times.min()], end=[event_times.max()], time_units="s"
    )
    aligned = nap.compute_perievent(
        session.spikes,
        events,
        window=PERI_EVENT_WINDOW_S,
        time_unit="s",
        epochs=event_epoch,
    )
    unit_ids = list(session.spikes.keys())
    trial_spikes = {
        unit_id: [
            np.asarray(aligned[unit_id][sorted_position[trial]].t)
            for trial in range(len(events))
        ]
        for unit_id in unit_ids
    }
    return unit_ids, trial_spikes, valid_trials


def _concatenate_trials(
    unit_ids: list[int],
    trial_spikes: dict[int, list[np.ndarray]],
    source_trials: np.ndarray,
    *,
    trial_spacing_s: float,
) -> nap.TsGroup:
    """Place trial spike trains into guarded synthetic recording intervals."""
    units = {}
    offset = -PERI_EVENT_WINDOW_S[0]
    destinations = np.arange(len(source_trials))
    for unit_id in unit_ids:
        pieces = [
            trial_spikes[unit_id][source] + destination * trial_spacing_s + offset
            for destination, source in zip(destinations, source_trials)
        ]
        units[unit_id] = nap.Ts(t=np.concatenate(pieces), time_units="s")
    return nap.TsGroup(units, time_units="s")


def _condition_shift_indices(
    condition_codes: np.ndarray, shift_number: int
) -> np.ndarray:
    """Circularly mismatch trials separately within every condition."""
    source = np.arange(len(condition_codes))
    for code in np.unique(condition_codes):
        trials = np.flatnonzero(condition_codes == code)
        offset = shift_number % len(trials)
        if offset == 0:
            offset = 1
        source[trials] = np.roll(trials, offset)
    return source


def _significant_peaks(
    event_name: str,
    corrected: np.ndarray,
    null_deviations: np.ndarray,
    lags_ms: np.ndarray,
    pairs: list[tuple[int, int]],
    alpha: float,
) -> pd.DataFrame:
    """Extract one extremum from each contiguous significant lag cluster."""
    upper = np.quantile(np.nanmax(null_deviations, axis=1), 1 - alpha, axis=0)
    lower = np.quantile(np.nanmin(null_deviations, axis=1), alpha, axis=0)
    rows: list[dict[str, object]] = []
    for pair_index, (unit_a, unit_b) in enumerate(pairs):
        for kind, mask in (
            ("positive_peak", corrected[:, pair_index] > upper[pair_index]),
            ("negative_trough", corrected[:, pair_index] < lower[pair_index]),
        ):
            boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
            for start, stop in boundaries.reshape(-1, 2):
                values = corrected[start:stop, pair_index]
                local = np.argmax(values) if kind == "positive_peak" else np.argmin(values)
                lag_index = start + int(local)
                rows.append(
                    {
                        "event_type": event_name,
                        "unit_a": unit_a,
                        "unit_b": unit_b,
                        "peak_type": kind,
                        "lag_ms": lags_ms[lag_index],
                        "corrected_rate_hz": corrected[lag_index, pair_index],
                        "familywise_threshold_hz": (
                            upper[pair_index]
                            if kind == "positive_peak"
                            else lower[pair_index]
                        ),
                        "cluster_start_ms": lags_ms[start],
                        "cluster_end_ms": lags_ms[stop - 1],
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "event_type",
            "unit_a",
            "unit_b",
            "peak_type",
            "lag_ms",
            "corrected_rate_hz",
            "familywise_threshold_hz",
            "cluster_start_ms",
            "cluster_end_ms",
        ],
    )


def corrected_crosscorrelograms(
    session: CON2PHYSSession,
    *,
    bin_size_ms: float = CCG_BIN_SIZE_MS,
    max_lag_ms: float = CCG_MAX_LAG_MS,
    n_shift_predictors: int = N_SHIFT_PREDICTORS,
    alpha: float = SIGNIFICANCE_LEVEL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], pd.DataFrame]:
    """Calculate Pynapple CCGs, shift predictors, corrections, and peaks."""
    if session.spikes is None:
        raise ValueError("session must contain spikes")
    if not 5 <= bin_size_ms <= 20:
        raise ValueError("bin_size_ms must be between 5 and 20 ms")
    if max_lag_ms <= 0 or n_shift_predictors < 2:
        raise ValueError("max_lag_ms and n_shift_predictors must be positive")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")

    conditions = session.trial_table.loc[:, CONDITION_COLUMNS]
    condition_codes, _ = pd.factorize(pd.MultiIndex.from_frame(conditions), sort=True)
    # A gap larger than twice the requested lag prevents cross-trial pairs.
    spacing = (PERI_EVENT_WINDOW_S[1] - PERI_EVENT_WINDOW_S[0]) + 2 * max_lag_ms / 1_000

    observed_events: list[np.ndarray] = []
    predictor_events: list[np.ndarray] = []
    peak_tables: list[pd.DataFrame] = []
    pairs: list[tuple[int, int]] | None = None
    lags_ms: np.ndarray | None = None

    for event_name, event_column in EVENT_COLUMNS.items():
        unit_ids, trial_spikes, valid_trials = _aligned_trial_spikes(
            session, event_column
        )
        event_condition_codes = condition_codes[valid_trials]
        identity = np.arange(len(valid_trials))
        observed_group = _concatenate_trials(
            unit_ids, trial_spikes, identity, trial_spacing_s=spacing
        )
        observed_frame = nap.compute_crosscorrelogram(
            observed_group,
            binsize=float(bin_size_ms),
            windowsize=float(max_lag_ms),
            norm=False,
            time_units="ms",
        )
        event_pairs = [tuple(map(int, pair)) for pair in observed_frame.columns]
        if pairs is None:
            pairs = event_pairs
            lags_ms = observed_frame.index.to_numpy(dtype=float) * 1_000
        elif event_pairs != pairs:
            raise RuntimeError("Pynapple returned inconsistent unit-pair ordering")
        observed = observed_frame.to_numpy(dtype=float)

        shuffled = np.empty(
            (n_shift_predictors, observed.shape[0], observed.shape[1]), dtype=np.float32
        )
        for shift_index in range(n_shift_predictors):
            source = _condition_shift_indices(
                event_condition_codes, shift_index + 1
            )
            shifted_group = _concatenate_trials(
                unit_ids, trial_spikes, source, trial_spacing_s=spacing
            )
            predictor = nap.compute_crosscorrelogram(
                (observed_group, shifted_group),
                binsize=float(bin_size_ms),
                windowsize=float(max_lag_ms),
                norm=False,
                time_units="ms",
            )
            shuffled[shift_index] = predictor.loc[:, pairs].to_numpy(dtype=np.float32)

        shift_mean = shuffled.mean(axis=0)
        corrected = observed - shift_mean
        null_deviations = shuffled - shift_mean[np.newaxis]
        observed_events.append(observed.astype(np.float32))
        predictor_events.append(shift_mean.astype(np.float32))
        peak_tables.append(
            _significant_peaks(
                event_name, corrected, null_deviations, lags_ms, pairs, alpha
            )
        )

    assert pairs is not None and lags_ms is not None
    observed_array = np.stack(observed_events)
    predictor_array = np.stack(predictor_events)
    corrected_array = observed_array - predictor_array
    peaks = pd.concat(peak_tables, ignore_index=True)
    return corrected_array, observed_array, predictor_array, pairs, peaks


def plot_corrected_crosscorrelograms_by_area(
    corrected: np.ndarray,
    lags_ms: np.ndarray,
    pair_brain_area_a: np.ndarray,
    pair_brain_area_b: np.ndarray,
    path: str | Path = DEFAULT_OUTPUT_PATH
    / "participant_1_corrected_crosscorrelograms_by_area.png",
) -> None:
    """Plot mean corrected CCGs for each within/between-area pair class."""
    areas = sorted(np.unique(np.r_[pair_brain_area_a, pair_brain_area_b]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for event_index, (event_name, ax) in enumerate(zip(EVENT_COLUMNS, axes.flat)):
        for left_index, left_area in enumerate(areas):
            for right_area in areas[left_index:]:
                mask = (
                    ((pair_brain_area_a == left_area) & (pair_brain_area_b == right_area))
                    | (
                        (pair_brain_area_a == right_area)
                        & (pair_brain_area_b == left_area)
                    )
                )
                if not mask.any():
                    continue
                values = corrected[event_index][:, mask]
                mean = np.nanmean(values, axis=1)
                sem = (
                    np.nanstd(values, axis=1, ddof=1) / np.sqrt(mask.sum())
                    if mask.sum() > 1
                    else np.zeros_like(mean)
                )
                line = ax.plot(
                    lags_ms,
                    mean,
                    linewidth=1.5,
                    label=f"Area {left_area}–{right_area}",
                )[0]
                ax.fill_between(
                    lags_ms,
                    mean - 1.96 * sem,
                    mean + 1.96 * sem,
                    color=line.get_color(),
                    alpha=0.12,
                )
        ax.axvline(0, color="0.25", linestyle="--", linewidth=1)
        ax.axhline(0, color="0.55", linestyle=":", linewidth=0.8)
        ax.set_title(event_name.replace("_", " ").title())
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Shift-corrected rate (Hz)")
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        f"Participant {PARTICIPANT}: shift-predictor-corrected CCGs\n"
        f"mean ± 95% CI across neuron pairs; {CCG_BIN_SIZE_MS:g} ms bins"
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_corrected_crosscorrelogram_heatmaps(
    corrected: np.ndarray,
    lags_ms: np.ndarray,
    path: str | Path = DEFAULT_OUTPUT_PATH
    / "participant_1_pairwise_corrected_crosscorrelogram_heatmaps.png",
) -> None:
    """Plot every neuron pair, sorted by maximum absolute corrected response."""
    finite = np.abs(corrected[np.isfinite(corrected)])
    color_limit = float(np.percentile(finite, 99)) if len(finite) else 1.0
    color_limit = max(color_limit, 1e-6)
    fig, axes = plt.subplots(1, len(EVENT_COLUMNS), figsize=(16, 6), sharey=True)
    image = None
    for event_index, (event_name, ax) in enumerate(zip(EVENT_COLUMNS, axes)):
        event_values = corrected[event_index].T
        strength = np.nanmax(np.abs(event_values), axis=1)
        order = np.argsort(strength)
        image = ax.imshow(
            event_values[order],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[lags_ms[0], lags_ms[-1], 0, len(order)],
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(event_name.replace("_", " ").title())
        ax.set_xlabel("Lag (ms)")
    axes[0].set_ylabel("Neuron pair (sorted by response strength)")
    if image is not None:
        fig.colorbar(image, ax=axes, label="Shift-corrected rate (Hz)")
    fig.suptitle(f"Participant {PARTICIPANT}: pairwise corrected CCGs")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    session = load_participant_spikes()
    corrected, observed, predictor, pairs, peaks = corrected_crosscorrelograms(session)
    unit_metadata = session.spikes.metadata
    pair_ids = np.asarray(pairs, dtype=int)
    output = DEFAULT_OUTPUT_PATH / "participant_1_corrected_crosscorrelograms.npz"
    DEFAULT_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    lags_ms = np.linspace(-CCG_MAX_LAG_MS, CCG_MAX_LAG_MS, corrected.shape[1])
    np.savez_compressed(
        output,
        corrected_rate_hz=corrected,
        observed_rate_hz=observed,
        shift_predictor_rate_hz=predictor,
        event_types=np.asarray(list(EVENT_COLUMNS)),
        lags_ms=lags_ms,
        pair_unit_a=pair_ids[:, 0],
        pair_unit_b=pair_ids[:, 1],
        pair_brain_area_a=unit_metadata.loc[pair_ids[:, 0], "brain_area"].to_numpy(),
        pair_brain_area_b=unit_metadata.loc[pair_ids[:, 1], "brain_area"].to_numpy(),
        bin_size_ms=CCG_BIN_SIZE_MS,
        max_lag_ms=CCG_MAX_LAG_MS,
        n_shift_predictors=N_SHIFT_PREDICTORS,
    )
    area_lookup = unit_metadata["brain_area"]
    peaks.insert(3, "brain_area_a", peaks["unit_a"].map(area_lookup))
    peaks.insert(4, "brain_area_b", peaks["unit_b"].map(area_lookup))
    peaks.to_csv(DEFAULT_OUTPUT_PATH / "participant_1_significant_lagged_peaks.csv", index=False)
    pair_area_a = unit_metadata.loc[pair_ids[:, 0], "brain_area"].to_numpy()
    pair_area_b = unit_metadata.loc[pair_ids[:, 1], "brain_area"].to_numpy()
    plot_corrected_crosscorrelograms_by_area(
        corrected, lags_ms, pair_area_a, pair_area_b
    )
    plot_pairwise_corrected_crosscorrelogram_heatmaps(corrected, lags_ms)
    print(f"Corrected CCG shape (events, lags, pairs): {corrected.shape}")
    print(f"Significant lagged peaks/troughs: {len(peaks)}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
