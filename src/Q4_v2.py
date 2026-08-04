"""Construct participant 1 event-aligned spike tensors with Pynapple.

Spikes are aligned to trial onset, stimulus presentation, outcome, and trial
end over a -1 to +1 second window, then counted in 10 ms bins. Run with::

    python src/Q4_v2.py

The saved tensor has dimensions ``event_type x trial x time_bin x unit``.
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
    from .data_loader import CON2PHYSSession, load_session
else:
    from data_loader import CON2PHYSSession, load_session


PARTICIPANT = 1
PERI_EVENT_WINDOW_S = (-1.0, 1.0)
BIN_SIZE_S = 0.010
CORRELATION_WINDOW_S = 0.250
CORRELATION_STEP_S = 0.020
EVENT_COLUMNS = {
    "trial_onset": "trial_start",
    "stimulus_presentation": "stim_start",
    "outcome": "outcome",
    "trial_end": "trial_end",
}
CONDITION_COLUMNS = ("variable_A", "variable_B", "variable_C")
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "analysis" / "Q4_v2"


def load_participant_spikes(participant: int = PARTICIPANT) -> CON2PHYSSession:
    """Load participant spikes and behavioral times without loading LFP data."""
    return load_session(subject=participant, lfp_banks=())


def construct_event_aligned_spike_tensors(
    session: CON2PHYSSession,
    *,
    window_s: tuple[float, float] = PERI_EVENT_WINDOW_S,
    bin_size_s: float = BIN_SIZE_S,
) -> dict[str, nap.TsdTensor]:
    """Return one Pynapple spike-count tensor for each behavioral event.

    Each returned ``TsdTensor`` has shape ``time_bin x trial x unit``. The time
    index contains bin centers relative to the event in seconds. Alignment is
    performed with :func:`pynapple.compute_perievent`, and the aligned spike
    trains are binned with :meth:`pynapple.TsGroup.count`.
    """
    if session.spikes is None:
        raise ValueError("session must contain spike trains")
    missing = set(EVENT_COLUMNS.values()).difference(session.trial_table.columns)
    if missing:
        raise ValueError(f"trial table is missing columns: {sorted(missing)}")
    if len(window_s) != 2 or not np.all(np.isfinite(window_s)):
        raise ValueError("window_s must contain two finite values")
    window_start, window_end = map(float, window_s)
    if window_start >= window_end:
        raise ValueError("window_s start must precede its end")
    if not np.isfinite(bin_size_s) or bin_size_s <= 0:
        raise ValueError("bin_size_s must be positive")
    n_bins_float = (window_end - window_start) / bin_size_s
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins_float, n_bins):
        raise ValueError("the peri-event window must be divisible by bin_size_s")

    unit_ids = list(session.spikes.keys())
    trial_ids = session.trial_table.index.to_numpy()
    bin_epoch = nap.IntervalSet(
        start=[window_start], end=[window_end], time_units="s"
    )
    tensors: dict[str, nap.TsdTensor] = {}

    for event_name, column in EVENT_COLUMNS.items():
        event_times = session.trial_table[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(event_times)):
            raise ValueError(f"{column} contains non-finite event times")
        events = nap.Ts(t=event_times, time_units="s")
        # Passing an explicit epoch preserves behavioral events just outside
        # the observed spike time support; their peri-event counts are zero.
        event_epoch = nap.IntervalSet(
            start=[event_times.min()], end=[event_times.max()], time_units="s"
        )
        aligned = nap.compute_perievent(
            session.spikes,
            events,
            window=(window_start, window_end),
            time_unit="s",
            epochs=event_epoch,
        )

        # Fill time x trial x unit, matching the native time-first convention
        # used by Pynapple's TsdTensor.
        values = np.empty((n_bins, len(trial_ids), len(unit_ids)), dtype=np.uint16)
        bin_centers = None
        for unit_position, unit_id in enumerate(unit_ids):
            counts = aligned[unit_id].count(bin_size=bin_size_s, ep=bin_epoch)
            unit_counts = np.asarray(counts)
            if unit_counts.shape != (n_bins, len(trial_ids)):
                raise RuntimeError(
                    "Pynapple returned an unexpected peri-event count shape"
                )
            if unit_counts.max(initial=0) > np.iinfo(np.uint16).max:
                raise OverflowError("a spike count exceeds uint16 capacity")
            values[:, :, unit_position] = unit_counts.astype(np.uint16)
            if bin_centers is None:
                bin_centers = np.asarray(counts.t, dtype=float)

        assert bin_centers is not None
        tensors[event_name] = nap.TsdTensor(
            t=bin_centers,
            d=values,
            time_units="s",
        )
    return tensors


def save_event_aligned_spike_tensors(
    tensors: dict[str, nap.TsdTensor],
    session: CON2PHYSSession,
    path: str | Path = DEFAULT_OUTPUT_PATH / "participant_1_event_spike_tensor.npz",
) -> pd.DataFrame:
    """Save tensors, dimension labels, and unit-to-brain-area metadata.

    Returns
    -------
    pandas.DataFrame
        Unit metadata in the exact order of the tensor's final (unit) axis.
    """
    event_names = list(EVENT_COLUMNS)
    missing = set(event_names).difference(tensors)
    if missing:
        raise ValueError(f"tensors are missing events: {sorted(missing)}")
    first = tensors[event_names[0]]
    values = np.stack(
        [np.asarray(tensors[name]).transpose(1, 0, 2) for name in event_names],
        axis=0,
    )
    unit_ids = np.asarray(list(session.spikes.keys()), dtype=int)
    metadata = session.spikes.metadata.reindex(unit_ids)
    if "brain_area" not in metadata or metadata["brain_area"].isna().any():
        raise ValueError("every tensor unit must have brain_area metadata")
    unit_metadata = metadata.reset_index().rename(
        columns={metadata.index.name or "index": "unit_id", "rate": "firing_rate_hz"}
    )
    unit_metadata.insert(0, "unit_index", np.arange(len(unit_metadata)))
    time_centers = np.asarray(first.t, dtype=float)
    bin_size_s = float(np.median(np.diff(time_centers)))
    window_s = np.asarray(
        [time_centers[0] - bin_size_s / 2, time_centers[-1] + bin_size_s / 2]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        spike_counts=values,
        event_types=np.asarray(event_names),
        trial_ids=session.trial_table.index.to_numpy(),
        time_bin_centers_s=time_centers,
        unit_ids=unit_ids,
        brain_areas=metadata["brain_area"].to_numpy(),
        unit_axis=np.arange(len(unit_ids)),
        unit_firing_rates_hz=metadata["rate"].to_numpy(dtype=float),
        bin_size_s=bin_size_s,
        window_s=window_s,
    )
    unit_metadata.to_csv(path.with_name(f"{path.stem}_unit_metadata.csv"), index=False)
    return unit_metadata


def residualize_spike_counts_by_condition(
    tensors: dict[str, nap.TsdTensor],
    session: CON2PHYSSession,
    *,
    condition_columns: tuple[str, ...] = CONDITION_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Subtract condition-specific trial means from the spike-count tensor.

    For each event, time bin, and neuron, the mean across trials sharing the
    same joint behavioral condition is subtracted from every trial in that
    condition. The returned tensor is ordered
    ``event_type x trial x time_bin x unit``, matching the saved input tensor.
    """
    if not condition_columns:
        raise ValueError("at least one behavioral condition column is required")
    missing = set(condition_columns).difference(session.trial_table.columns)
    if missing:
        raise ValueError(f"trial table is missing conditions: {sorted(missing)}")
    conditions = session.trial_table.loc[:, condition_columns]
    if conditions.isna().any().any():
        raise ValueError("behavioral conditions must not contain missing values")

    event_names = list(EVENT_COLUMNS)
    missing_events = set(event_names).difference(tensors)
    if missing_events:
        raise ValueError(f"tensors are missing events: {sorted(missing_events)}")
    counts = np.stack(
        [np.asarray(tensors[name]).transpose(1, 0, 2) for name in event_names],
        axis=0,
    ).astype(np.float32)
    if counts.shape[1] != len(conditions):
        raise ValueError("tensor trial axis does not match the trial table")

    condition_codes, unique_rows = pd.factorize(
        pd.MultiIndex.from_frame(conditions), sort=True
    )
    residuals = counts.copy()
    condition_table = unique_rows.to_frame(index=False)
    condition_table.columns = list(condition_columns)
    condition_table.insert(0, "condition_code", np.arange(len(condition_table)))
    condition_table["n_trials"] = [
        int(np.sum(condition_codes == code)) for code in condition_table["condition_code"]
    ]
    for code in condition_table["condition_code"]:
        trial_mask = condition_codes == code
        condition_mean = counts[:, trial_mask, :, :].mean(axis=1, keepdims=True)
        residuals[:, trial_mask, :, :] -= condition_mean
    return residuals, condition_codes, condition_table


def save_residual_spike_tensor(
    residuals: np.ndarray,
    condition_codes: np.ndarray,
    condition_table: pd.DataFrame,
    tensors: dict[str, nap.TsdTensor],
    session: CON2PHYSSession,
    path: str | Path = DEFAULT_OUTPUT_PATH
    / "participant_1_event_spike_tensor_condition_residuals.npz",
) -> None:
    """Save the condition-residual tensor with input-axis metadata."""
    event_names = list(EVENT_COLUMNS)
    first = tensors[event_names[0]]
    expected_shape = (
        len(event_names),
        len(session.trial_table),
        first.shape[0],
        first.shape[2],
    )
    if residuals.shape != expected_shape:
        raise ValueError(f"residual tensor must have shape {expected_shape}")
    unit_ids = np.asarray(list(session.spikes.keys()), dtype=int)
    metadata = session.spikes.metadata.reindex(unit_ids)
    time_centers = np.asarray(first.t, dtype=float)
    bin_size_s = float(np.median(np.diff(time_centers)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        residual_spike_counts=residuals.astype(np.float32, copy=False),
        event_types=np.asarray(event_names),
        trial_ids=session.trial_table.index.to_numpy(),
        time_bin_centers_s=time_centers,
        unit_axis=np.arange(len(unit_ids)),
        unit_ids=unit_ids,
        brain_areas=metadata["brain_area"].to_numpy(),
        condition_columns=np.asarray(CONDITION_COLUMNS),
        trial_condition_codes=condition_codes,
        condition_values=condition_table.loc[:, CONDITION_COLUMNS].to_numpy(),
        condition_trial_counts=condition_table["n_trials"].to_numpy(),
        bin_size_s=bin_size_s,
        window_s=np.asarray(
            [time_centers[0] - bin_size_s / 2, time_centers[-1] + bin_size_s / 2]
        ),
    )
    condition_table.to_csv(
        path.with_name(f"{path.stem}_conditions.csv"), index=False
    )


def time_resolved_residual_correlations(
    residuals: np.ndarray,
    time_bin_centers_s: np.ndarray,
    *,
    window_s: float = CORRELATION_WINDOW_S,
    step_s: float = CORRELATION_STEP_S,
    unit_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate pairwise residual-count correlations in sliding windows.

    Parameters
    ----------
    residuals:
        Tensor ordered ``event x trial x time x unit``.
    time_bin_centers_s:
        Peri-event time-bin centers in seconds.
    window_s, step_s:
        Sliding-window width and step. Both must be exact multiples of the
        underlying time-bin size.

    Returns
    -------
    correlations, window_starts_s, window_ends_s
        Correlations have shape ``event x window x unit x unit``. Correlations
        involving a neuron with zero across-trial variance are NaN.
    """
    residuals = np.asarray(residuals, dtype=np.float32)
    time_bin_centers_s = np.asarray(time_bin_centers_s, dtype=float)
    if residuals.ndim != 4:
        raise ValueError("residuals must have event x trial x time x unit axes")
    if len(time_bin_centers_s) != residuals.shape[2]:
        raise ValueError("time centers do not match the residual time axis")
    if len(time_bin_centers_s) < 2:
        raise ValueError("at least two time bins are required")
    bin_size_s = float(np.median(np.diff(time_bin_centers_s)))
    if not np.allclose(np.diff(time_bin_centers_s), bin_size_s):
        raise ValueError("time bins must be regularly spaced")
    if window_s <= 0 or step_s <= 0:
        raise ValueError("window_s and step_s must be positive")
    window_bins_float = window_s / bin_size_s
    step_bins_float = step_s / bin_size_s
    window_bins = int(round(window_bins_float))
    step_bins = int(round(step_bins_float))
    if not np.isclose(window_bins_float, window_bins):
        raise ValueError("window_s must be an exact multiple of the bin size")
    if not np.isclose(step_bins_float, step_bins):
        raise ValueError("step_s must be an exact multiple of the bin size")
    if window_bins > residuals.shape[2]:
        raise ValueError("window_s exceeds the peri-event interval")

    start_indices = np.arange(
        0, residuals.shape[2] - window_bins + 1, step_bins, dtype=int
    )
    bin_left_edge = time_bin_centers_s[0] - bin_size_s / 2
    window_starts_s = bin_left_edge + start_indices * bin_size_s
    window_ends_s = window_starts_s + window_bins * bin_size_s
    n_events, _, _, n_units = residuals.shape
    if unit_ids is None:
        unit_ids = np.arange(n_units)
    unit_ids = np.asarray(unit_ids)
    if unit_ids.shape != (n_units,):
        raise ValueError("unit_ids must match the residual unit axis")
    correlations = np.empty(
        (n_events, len(start_indices), n_units, n_units), dtype=np.float32
    )

    cumulative = np.concatenate(
        [
            np.zeros(
                (n_events, residuals.shape[1], 1, n_units), dtype=np.float32
            ),
            np.cumsum(residuals, axis=2, dtype=np.float32),
        ],
        axis=2,
    )
    for window_index, start in enumerate(start_indices):
        activity = cumulative[:, :, start + window_bins, :] - cumulative[
            :, :, start, :
        ]
        for event_index in range(n_events):
            values = activity[event_index]
            trial_activity = nap.TsdFrame(
                t=np.arange(values.shape[0], dtype=float),
                d=values,
                columns=unit_ids,
            )
            # Pynapple has no dedicated spike-count Pearson-correlation
            # function. Its labeled DataFrame representation supplies the
            # standard pairwise correlation across the trial samples.
            correlations[event_index, window_index] = (
                trial_activity.as_dataframe().corr().to_numpy(dtype=np.float32)
            )
            np.clip(
                correlations[event_index, window_index],
                -1.0,
                1.0,
                out=correlations[event_index, window_index],
            )
    return correlations, window_starts_s, window_ends_s


def save_time_resolved_correlations(
    correlations: np.ndarray,
    window_starts_s: np.ndarray,
    window_ends_s: np.ndarray,
    session: CON2PHYSSession,
    path: str | Path = DEFAULT_OUTPUT_PATH
    / "participant_1_time_resolved_residual_correlations.npz",
) -> None:
    """Save sliding-window correlation matrices and axis metadata."""
    unit_ids = np.asarray(list(session.spikes.keys()), dtype=int)
    metadata = session.spikes.metadata.reindex(unit_ids)
    expected_shape = (
        len(EVENT_COLUMNS),
        len(window_starts_s),
        len(unit_ids),
        len(unit_ids),
    )
    if correlations.shape != expected_shape:
        raise ValueError(f"correlations must have shape {expected_shape}")
    window_starts_s = np.asarray(window_starts_s, dtype=float)
    window_ends_s = np.asarray(window_ends_s, dtype=float)
    window_size_s = float(np.median(window_ends_s - window_starts_s))
    step_size_s = (
        float(np.median(np.diff(window_starts_s)))
        if len(window_starts_s) > 1
        else np.nan
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        correlations=correlations.astype(np.float32, copy=False),
        event_types=np.asarray(list(EVENT_COLUMNS)),
        window_starts_s=window_starts_s,
        window_ends_s=window_ends_s,
        window_centers_s=(window_starts_s + window_ends_s) / 2,
        unit_axis=np.arange(len(unit_ids)),
        unit_ids=unit_ids,
        brain_areas=metadata["brain_area"].to_numpy(),
        window_size_s=window_size_s,
        step_size_s=step_size_s,
        n_trials=len(session.trial_table),
    )


def plot_time_resolved_correlations(
    correlations: np.ndarray,
    window_starts_s: np.ndarray,
    window_ends_s: np.ndarray,
    brain_areas: np.ndarray,
    path: str | Path = DEFAULT_OUTPUT_PATH
    / "participant_1_time_resolved_residual_correlations.png",
) -> None:
    """Plot mean residual correlations for within/between-area unit pairs."""
    correlations = np.asarray(correlations, dtype=float)
    brain_areas = np.asarray(brain_areas)
    if correlations.ndim != 4 or correlations.shape[-1] != correlations.shape[-2]:
        raise ValueError("correlations must have event x window x unit x unit axes")
    if brain_areas.shape != (correlations.shape[-1],):
        raise ValueError("brain_areas must match the correlation unit axes")
    window_centers = (np.asarray(window_starts_s) + window_ends_s) / 2
    if len(window_centers) != correlations.shape[1]:
        raise ValueError("window times must match the correlation window axis")

    areas = sorted(np.unique(brain_areas))
    pair_masks: list[tuple[str, np.ndarray]] = []
    for left_index, left_area in enumerate(areas):
        for right_area in areas[left_index:]:
            if left_area == right_area:
                in_area = brain_areas == left_area
                mask = np.outer(in_area, in_area) & np.triu(
                    np.ones((len(brain_areas), len(brain_areas)), dtype=bool), k=1
                )
            else:
                mask = np.outer(
                    brain_areas == left_area, brain_areas == right_area
                )
            pair_masks.append((f"Area {left_area}–{right_area}", mask))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for event_index, (event_name, ax) in enumerate(
        zip(EVENT_COLUMNS, axes.flat)
    ):
        for label, mask in pair_masks:
            pair_values = correlations[event_index][:, mask]
            valid_counts = np.isfinite(pair_values).sum(axis=1)
            means = np.divide(
                np.nansum(pair_values, axis=1),
                valid_counts,
                out=np.full(len(window_centers), np.nan),
                where=valid_counts > 0,
            )
            ax.plot(window_centers, means, linewidth=1.4, label=label)
        ax.axvline(0, color="0.25", linestyle="--", linewidth=1)
        ax.axhline(0, color="0.6", linestyle=":", linewidth=0.8)
        ax.set_title(event_name.replace("_", " ").title())
        ax.set_xlabel("Peri-event window center (s)")
        ax.set_ylabel("Mean residual spike-count correlation")
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        f"Participant {PARTICIPANT}: time-resolved residual correlations\n"
        f"{CORRELATION_WINDOW_S * 1_000:g} ms windows, "
        f"{CORRELATION_STEP_S * 1_000:g} ms steps"
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_checkpoint_correlation_matrices(
    correlations: np.ndarray,
    window_starts_s: np.ndarray,
    window_ends_s: np.ndarray,
    brain_areas: np.ndarray,
    path: str | Path = DEFAULT_OUTPUT_PATH
    / "participant_1_checkpoint_correlation_matrices.png",
) -> None:
    """Plot unit correlation matrices for windows centered nearest time zero."""
    correlations = np.asarray(correlations, dtype=float)
    brain_areas = np.asarray(brain_areas)
    window_centers = (np.asarray(window_starts_s) + window_ends_s) / 2
    zero_window = int(np.argmin(np.abs(window_centers)))
    matrices = correlations[:, zero_window].copy()
    for matrix in matrices:
        np.fill_diagonal(matrix, np.nan)
    finite = np.abs(matrices[np.isfinite(matrices)])
    color_limit = float(np.percentile(finite, 99)) if len(finite) else 1.0
    color_limit = max(color_limit, 0.05)

    fig, axes = plt.subplots(1, len(EVENT_COLUMNS), figsize=(16, 4), sharey=True)
    image = None
    for event_index, (event_name, ax) in enumerate(zip(EVENT_COLUMNS, axes)):
        image = ax.imshow(
            matrices[event_index],
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
            interpolation="nearest",
            origin="upper",
        )
        ax.set_title(event_name.replace("_", " ").title())
        ax.set_xlabel("Unit index")
        for boundary in np.flatnonzero(np.diff(brain_areas)) + 0.5:
            ax.axvline(boundary, color="black", linewidth=0.6)
            ax.axhline(boundary, color="black", linewidth=0.6)
    axes[0].set_ylabel("Unit index")
    if image is not None:
        fig.colorbar(image, ax=axes, label="Residual spike-count correlation")
    fig.suptitle(
        "Correlation matrices for the window centered nearest checkpoint "
        f"({window_centers[zero_window]:+.3f} s)"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    session = load_participant_spikes()
    tensors = construct_event_aligned_spike_tensors(session)
    output = DEFAULT_OUTPUT_PATH / "participant_1_event_spike_tensor.npz"
    unit_metadata = save_event_aligned_spike_tensors(tensors, session, output)
    residuals, condition_codes, condition_table = (
        residualize_spike_counts_by_condition(tensors, session)
    )
    residual_output = (
        DEFAULT_OUTPUT_PATH
        / "participant_1_event_spike_tensor_condition_residuals.npz"
    )
    save_residual_spike_tensor(
        residuals,
        condition_codes,
        condition_table,
        tensors,
        session,
        residual_output,
    )
    correlations, window_starts_s, window_ends_s = (
        time_resolved_residual_correlations(
            residuals,
            np.asarray(tensors[next(iter(EVENT_COLUMNS))].t),
            unit_ids=np.asarray(list(session.spikes.keys())),
        )
    )
    correlation_output = (
        DEFAULT_OUTPUT_PATH
        / "participant_1_time_resolved_residual_correlations.npz"
    )
    save_time_resolved_correlations(
        correlations,
        window_starts_s,
        window_ends_s,
        session,
        correlation_output,
    )
    brain_areas = session.spikes.metadata.reindex(
        list(session.spikes.keys())
    )["brain_area"].to_numpy()
    plot_time_resolved_correlations(
        correlations, window_starts_s, window_ends_s, brain_areas
    )
    plot_checkpoint_correlation_matrices(
        correlations, window_starts_s, window_ends_s, brain_areas
    )

    first = tensors[next(iter(EVENT_COLUMNS))]
    print("\nParticipant 1 event-aligned spike tensors")
    print(f"Events: {', '.join(EVENT_COLUMNS)}")
    print(
        "Per-event Pynapple shape (time bins, trials, units): "
        f"{first.shape}"
    )
    print(
        "Saved array shape (events, trials, time bins, units): "
        f"({len(tensors)}, {first.shape[1]}, {first.shape[0]}, {first.shape[2]})"
    )
    print(f"Window: {PERI_EVENT_WINDOW_S[0]:g} to {PERI_EVENT_WINDOW_S[1]:g} s")
    print(f"Bin size: {BIN_SIZE_S * 1_000:g} ms")
    print(f"Saved to {output}")
    print(
        "Unit metadata saved to "
        f"{output.with_name(f'{output.stem}_unit_metadata.csv')}"
    )
    print("\nUnits by brain area:")
    print(unit_metadata.groupby("brain_area").size().rename("n_units").to_string())
    print(f"\nCondition-residual tensor saved to {residual_output}")
    print("Behavioral conditions:")
    print(condition_table.to_string(index=False))
    print(f"\nTime-resolved residual correlations saved to {correlation_output}")
    print(
        "Correlation shape (events, windows, units, units): "
        f"{correlations.shape}"
    )
    print(
        f"Sliding window: {CORRELATION_WINDOW_S * 1_000:g} ms; "
        f"step: {CORRELATION_STEP_S * 1_000:g} ms"
    )


if __name__ == "__main__":
    main()
