"""Load participant 1 spike/LFP data and theta-band filter the LFP.

The three LFP files in the CON2PHYS release are referred to as areas (or
banks) 1--3.  Spike trains retain the ``brain_area`` metadata supplied with
the dataset.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats
from scipy.signal import butter, hilbert, sosfiltfilt

if __package__:
    from .data_loader import (
        CON2PHYSSession,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )
    from .stat_test import holm_adjust
else:
    from data_loader import (  # type: ignore[no-redef]
        CON2PHYSSession,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )
    from stat_test import holm_adjust


PARTICIPANT = 1
LFP_AREAS = (1, 2, 3)
LFP_BAND_HZ = (4.0, 10.0)
DEFAULT_PHASE_PLOT = DEFAULT_PLOT_PATH / "Q8_example_neuron_phase_distributions.png"
DEFAULT_AREA_MEAN_PLOT = (
    DEFAULT_PLOT_PATH / "Q8_mean_spike_phase_distribution_by_area.png"
)
DEFAULT_FIRING_RATE_PLOT = (
    DEFAULT_PLOT_PATH / "Q8_mean_phase_binned_firing_rate_by_area.png"
)
DEFAULT_NEURON_PLV_TABLE = DEFAULT_PLOT_PATH / "Q8_neuron_phase_locking_values.csv"
DEFAULT_AREA_PLV_TABLE = DEFAULT_PLOT_PATH / "Q8_phase_locking_summary_by_area.csv"
ALL_PARTICIPANT_PPC_TABLE = DEFAULT_PLOT_PATH / "Q8_ppc_by_participant_area.csv"
ALL_PARTICIPANT_PPC_SUMMARY = DEFAULT_PLOT_PATH / "Q8_ppc_group_summary.csv"
ALL_PARTICIPANT_PPC_TESTS = DEFAULT_PLOT_PATH / "Q8_ppc_pairwise_tests.csv"
ALL_PARTICIPANT_PPC_PLOT = DEFAULT_PLOT_PATH / "Q8_ppc_all_participants.png"


def load_participant_data(participant: int) -> CON2PHYSSession:
    """Load spikes and all three LFP areas for one participant.

    LFP samples are converted to ``float32`` while loading because the three
    recordings are large.  Spike trains are available in ``session.spikes``
    and can be selected by anatomical area using their ``brain_area``
    metadata.
    """
    return load_session(
        subject=participant,
        lfp_banks=LFP_AREAS,
        load_spikes=True,
        lfp_dtype="float32",
    )


def load_participant_1_data() -> CON2PHYSSession:
    """Load spikes and all LFP areas for participant 1."""
    return load_participant_data(PARTICIPANT)


def bandpass_filter_lfp(
    lfp_by_area: Mapping[int, nap.TsdFrame],
    sampling_rate: float,
    *,
    band: tuple[float, float] = LFP_BAND_HZ,
    order: int = 4,
    channel_chunk_size: int = 16,
) -> dict[int, nap.TsdFrame]:
    """Bandpass filter every LFP channel and return one frame per area.

    A zero-phase Butterworth filter is used, so the filtered signals remain
    aligned with spike times and behavioural events.  Channels are processed
    in small groups to limit temporary memory use.
    """
    low, high = map(float, band)
    nyquist = float(sampling_rate) / 2.0
    if not 0.0 < low < high < nyquist:
        raise ValueError(
            f"band must satisfy 0 < low < high < Nyquist ({nyquist:g} Hz)"
        )
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    if (
        not isinstance(channel_chunk_size, (int, np.integer))
        or channel_chunk_size < 1
    ):
        raise ValueError("channel_chunk_size must be a positive integer")

    sos = butter(order, (low, high), btype="bandpass", fs=sampling_rate, output="sos")
    filtered: dict[int, nap.TsdFrame] = {}

    for area, lfp in lfp_by_area.items():
        values = np.asarray(lfp.values)
        if values.ndim != 2:
            raise ValueError(f"LFP area {area} must be two-dimensional")

        filtered_values = np.empty(values.shape, dtype=np.float32)
        for start in range(0, values.shape[1], channel_chunk_size):
            stop = min(start + channel_chunk_size, values.shape[1])
            filtered_values[:, start:stop] = sosfiltfilt(
                sos, values[:, start:stop], axis=0
            )

        filtered[area] = nap.TsdFrame(
            t=np.asarray(lfp.index),
            d=filtered_values,
            time_units="s",
            columns=list(lfp.columns),
        )

    return filtered


def load_and_filter_participant_1(
    *,
    band: tuple[float, float] = LFP_BAND_HZ,
) -> tuple[CON2PHYSSession, dict[int, nap.TsdFrame]]:
    """Load participant 1 and return both raw and bandpass-filtered LFP."""
    session = load_participant_1_data()
    filtered_lfp = bandpass_filter_lfp(
        session.lfp,
        session.sampling_rate,
        band=band,
    )
    return session, filtered_lfp


def instantaneous_lfp_phase(
    filtered_lfp: Mapping[int, nap.TsdFrame],
    *,
    channel_chunk_size: int = 4,
) -> dict[int, nap.TsdFrame]:
    """Calculate instantaneous phase (radians) for every LFP channel.

    Phase is the angle of the analytic signal returned by the Hilbert
    transform and lies in ``[-pi, pi]``.
    """
    if (
        not isinstance(channel_chunk_size, (int, np.integer))
        or channel_chunk_size < 1
    ):
        raise ValueError("channel_chunk_size must be a positive integer")

    phase_by_area: dict[int, nap.TsdFrame] = {}
    for area, lfp in filtered_lfp.items():
        values = np.asarray(lfp.values)
        if values.ndim != 2:
            raise ValueError(f"LFP area {area} must be two-dimensional")

        phase = np.empty(values.shape, dtype=np.float32)
        for start in range(0, values.shape[1], channel_chunk_size):
            stop = min(start + channel_chunk_size, values.shape[1])
            phase[:, start:stop] = np.angle(
                hilbert(values[:, start:stop], axis=0)
            )

        phase_by_area[area] = nap.TsdFrame(
            t=np.asarray(lfp.index),
            d=phase,
            time_units="s",
            columns=list(lfp.columns),
        )

    return phase_by_area


def spike_phases(
    spikes: nap.TsGroup,
    lfp_phase: Mapping[int, nap.TsdFrame],
    *,
    area_column: str = "brain_area",
) -> pd.DataFrame:
    """Assign an area-reference LFP phase to every spike from every neuron.

    The reference at each LFP sample is the circular mean across all channels
    in the neuron's brain area.  Its unwrapped phase is linearly interpolated
    to each spike time and then wrapped back into ``[-pi, pi]``.  Spikes
    outside the LFP recording receive ``NaN``.
    """
    if not isinstance(spikes, nap.TsGroup):
        raise TypeError("spikes must be a pynapple.TsGroup")
    if area_column not in spikes.metadata.columns:
        raise ValueError(f"spike metadata has no {area_column!r} column")

    area_references: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for area, phase_frame in lfp_phase.items():
        channel_phase = np.asarray(phase_frame.values)
        if channel_phase.ndim != 2 or channel_phase.shape[1] == 0:
            raise ValueError(f"LFP phase area {area} has no channels")
        # Accumulate unit phasors channel-by-channel to avoid constructing a
        # recording-sized complex array.
        cosine_sum = np.zeros(channel_phase.shape[0], dtype=np.float64)
        sine_sum = np.zeros(channel_phase.shape[0], dtype=np.float64)
        for channel in range(channel_phase.shape[1]):
            cosine_sum += np.cos(channel_phase[:, channel])
            sine_sum += np.sin(channel_phase[:, channel])
        reference = np.arctan2(sine_sum, cosine_sum)
        area_references[int(area)] = (
            np.asarray(phase_frame.index, dtype=float),
            np.unwrap(reference),
        )

    rows: list[pd.DataFrame] = []
    for unit_id, metadata in spikes.metadata.iterrows():
        area = int(metadata[area_column])
        if area not in area_references:
            raise ValueError(
                f"neuron {unit_id} belongs to area {area}, "
                "but that LFP area was not supplied"
            )

        times = np.asarray(spikes[int(unit_id)].index, dtype=float)
        lfp_times, unwrapped_reference = area_references[area]
        phases = np.full(times.shape, np.nan, dtype=np.float32)
        in_recording = (times >= lfp_times[0]) & (times <= lfp_times[-1])
        interpolated = np.interp(
            times[in_recording], lfp_times, unwrapped_reference
        )
        phases[in_recording] = (
            (interpolated + np.pi) % (2.0 * np.pi) - np.pi
        ).astype(np.float32)
        rows.append(
            pd.DataFrame(
                {
                    "neuron_id": int(unit_id),
                    "brain_area": area,
                    "spike_time": times,
                    "phase": phases,
                }
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=["neuron_id", "brain_area", "spike_time", "phase"]
        )
    return pd.concat(rows, ignore_index=True)


def calculate_spike_ppc(phases_at_spikes: pd.DataFrame) -> pd.DataFrame:
    """Calculate unbiased pairwise phase consistency for every neuron."""
    required = {"neuron_id", "brain_area", "phase"}
    missing = required.difference(phases_at_spikes.columns)
    if missing:
        raise ValueError(
            f"phases_at_spikes is missing columns: {sorted(missing)}"
        )

    rows: list[dict[str, float | int]] = []
    valid = phases_at_spikes.dropna(subset=["phase"])
    for (area, neuron_id), group in valid.groupby(
        ["brain_area", "neuron_id"], sort=True
    ):
        phases = group["phase"].to_numpy(dtype=float)
        n_spikes = len(phases)
        if n_spikes >= 2:
            phasor_sum = np.exp(1j * phases).sum()
            ppc = (np.abs(phasor_sum) ** 2 - n_spikes) / (
                n_spikes * (n_spikes - 1)
            )
        else:
            ppc = np.nan
        rows.append(
            {
                "neuron_id": int(neuron_id),
                "brain_area": int(area),
                "n_spikes": n_spikes,
                "spike_ppc": float(ppc),
            }
        )
    return pd.DataFrame(rows)


def run_participant_1_phase_analysis(
    *,
    band: tuple[float, float] = LFP_BAND_HZ,
) -> tuple[
    CON2PHYSSession,
    dict[int, nap.TsdFrame],
    dict[int, nap.TsdFrame],
    pd.DataFrame,
]:
    """Load/filter participant 1 and calculate LFP and per-spike phases."""
    session, filtered_lfp = load_and_filter_participant_1(band=band)
    lfp_phase = instantaneous_lfp_phase(filtered_lfp)
    if session.spikes is None:
        raise RuntimeError("participant 1 spike data were not loaded")
    phases_at_spikes = spike_phases(session.spikes, lfp_phase)
    return session, filtered_lfp, lfp_phase, phases_at_spikes


def plot_example_neuron_phase_distributions(
    phases_at_spikes: pd.DataFrame,
    *,
    neurons_per_area: int = 3,
    n_bins: int = 24,
    path: str | Path | None = DEFAULT_PHASE_PLOT,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot spike probability versus LFP phase for example neurons per area.

    Examples are selected independently within each area by the number of
    spikes having a valid phase.  Histogram counts are divided by each
    neuron's total, so every displayed distribution sums to one.
    """
    required = {"neuron_id", "brain_area", "phase"}
    missing = required.difference(phases_at_spikes.columns)
    if missing:
        raise ValueError(
            f"phases_at_spikes is missing columns: {sorted(missing)}"
        )
    if not isinstance(neurons_per_area, (int, np.integer)) or neurons_per_area < 1:
        raise ValueError("neurons_per_area must be a positive integer")
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 4:
        raise ValueError("n_bins must be an integer of at least 4")

    valid = phases_at_spikes.dropna(subset=["phase"])
    areas = sorted(valid["brain_area"].unique())
    if not areas:
        raise ValueError("no spikes with valid phase values are available")

    figure, axes = plt.subplots(
        len(areas),
        1,
        figsize=(8, 3.2 * len(areas)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    edges = np.linspace(-np.pi, np.pi, int(n_bins) + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    for axis, area in zip(axes, areas):
        area_spikes = valid.loc[valid["brain_area"] == area]
        example_ids = (
            area_spikes.groupby("neuron_id", sort=False)
            .size()
            .sort_values(ascending=False, kind="stable")
            .head(neurons_per_area)
            .index
        )

        for neuron_id in example_ids:
            phases = area_spikes.loc[
                area_spikes["neuron_id"] == neuron_id, "phase"
            ].to_numpy()
            counts, _ = np.histogram(phases, bins=edges)
            probability = counts / counts.sum()
            axis.plot(
                centers,
                probability,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=f"Neuron {neuron_id} (n={len(phases):,})",
            )

        axis.set_ylabel("Spike probability")
        axis.set_title(f"Brain area {area}")
        axis.set_xlim(-np.pi, np.pi)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

    axes[-1].set_xlabel("LFP phase (radians)")
    axes[-1].set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    figure.suptitle(
        "Example neuron spike distributions across 4–10 Hz LFP phase",
        y=1.0,
    )
    figure.tight_layout()

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    return figure, axes


def plot_mean_phase_distribution_by_area(
    phases_at_spikes: pd.DataFrame,
    *,
    n_bins: int = 24,
    path: str | Path | None = DEFAULT_AREA_MEAN_PLOT,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot the mean per-neuron spike-phase distribution for each brain area.

    Every neuron's histogram is normalized before averaging, giving neurons
    equal weight regardless of firing rate.  Shading shows the standard error
    of the mean across neurons.
    """
    required = {"neuron_id", "brain_area", "phase"}
    missing = required.difference(phases_at_spikes.columns)
    if missing:
        raise ValueError(
            f"phases_at_spikes is missing columns: {sorted(missing)}"
        )
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 4:
        raise ValueError("n_bins must be an integer of at least 4")

    valid = phases_at_spikes.dropna(subset=["phase"])
    if valid.empty:
        raise ValueError("no spikes with valid phase values are available")

    edges = np.linspace(-np.pi, np.pi, int(n_bins) + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    figure, axis = plt.subplots(figsize=(8, 4.8))
    summary_rows: list[dict[str, float | int]] = []

    for area, area_spikes in valid.groupby("brain_area", sort=True):
        neuron_distributions = []
        for _, neuron_spikes in area_spikes.groupby("neuron_id", sort=False):
            counts, _ = np.histogram(neuron_spikes["phase"].to_numpy(), bins=edges)
            if counts.sum() > 0:
                neuron_distributions.append(counts / counts.sum())

        distributions = np.asarray(neuron_distributions)
        mean_probability = distributions.mean(axis=0)
        if len(distributions) > 1:
            sem = distributions.std(axis=0, ddof=1) / np.sqrt(len(distributions))
        else:
            sem = np.zeros_like(mean_probability)

        line = axis.plot(
            centers,
            mean_probability,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=f"Area {area} (n={len(distributions)} neurons)",
        )[0]
        axis.fill_between(
            centers,
            np.maximum(0.0, mean_probability - sem),
            mean_probability + sem,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )

        for center, mean, error in zip(centers, mean_probability, sem):
            summary_rows.append(
                {
                    "brain_area": int(area),
                    "phase_bin_center": float(center),
                    "mean_probability": float(mean),
                    "sem_probability": float(error),
                    "n_neurons": len(distributions),
                }
            )

    axis.set(
        xlim=(-np.pi, np.pi),
        xlabel="LFP phase (radians)",
        ylabel="Mean spike probability",
        title="Mean neuron spike distribution across 4–10 Hz LFP phase",
    )
    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    summary = pd.DataFrame(summary_rows)
    return figure, axis, summary


def phase_binned_firing_rates(
    phases_at_spikes: pd.DataFrame,
    lfp_phase: Mapping[int, nap.TsdFrame],
    sampling_rate: float,
    *,
    n_bins: int = 24,
) -> pd.DataFrame:
    """Calculate each neuron's firing rate as a function of LFP phase.

    Rate in a phase bin is its spike count divided by the number of LFP
    samples in that bin and the sampling rate.  This corrects for unequal
    amounts of recording time spent at different phases.
    """
    required = {"neuron_id", "brain_area", "phase"}
    missing = required.difference(phases_at_spikes.columns)
    if missing:
        raise ValueError(
            f"phases_at_spikes is missing columns: {sorted(missing)}"
        )
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 4:
        raise ValueError("n_bins must be an integer of at least 4")

    edges = np.linspace(-np.pi, np.pi, int(n_bins) + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    rows: list[dict[str, float | int]] = []

    for area, phase_frame in sorted(lfp_phase.items()):
        channel_phase = np.asarray(phase_frame.values)
        if channel_phase.ndim != 2 or channel_phase.shape[1] == 0:
            raise ValueError(f"LFP phase area {area} has no channels")

        cosine_sum = np.zeros(channel_phase.shape[0], dtype=np.float64)
        sine_sum = np.zeros(channel_phase.shape[0], dtype=np.float64)
        for channel in range(channel_phase.shape[1]):
            cosine_sum += np.cos(channel_phase[:, channel])
            sine_sum += np.sin(channel_phase[:, channel])
        reference_phase = np.arctan2(sine_sum, cosine_sum)
        occupancy_samples, _ = np.histogram(reference_phase, bins=edges)
        occupancy_seconds = occupancy_samples / float(sampling_rate)

        area_spikes = phases_at_spikes.loc[
            (phases_at_spikes["brain_area"] == area)
            & phases_at_spikes["phase"].notna()
        ]
        for neuron_id, neuron_spikes in area_spikes.groupby(
            "neuron_id", sort=False
        ):
            spike_counts, _ = np.histogram(
                neuron_spikes["phase"].to_numpy(), bins=edges
            )
            firing_rate = np.divide(
                spike_counts,
                occupancy_seconds,
                out=np.full(int(n_bins), np.nan, dtype=float),
                where=occupancy_seconds > 0,
            )
            for center, rate, count, duration in zip(
                centers, firing_rate, spike_counts, occupancy_seconds
            ):
                rows.append(
                    {
                        "neuron_id": int(neuron_id),
                        "brain_area": int(area),
                        "phase_bin_center": float(center),
                        "firing_rate_hz": float(rate),
                        "spike_count": int(count),
                        "occupancy_seconds": float(duration),
                    }
                )

    return pd.DataFrame(rows)


def plot_mean_phase_binned_firing_rate(
    firing_rates: pd.DataFrame,
    *,
    path: str | Path | None = DEFAULT_FIRING_RATE_PLOT,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot mean phase-binned firing rate and SEM across neurons by area."""
    required = {
        "neuron_id",
        "brain_area",
        "phase_bin_center",
        "firing_rate_hz",
    }
    missing = required.difference(firing_rates.columns)
    if missing:
        raise ValueError(f"firing_rates is missing columns: {sorted(missing)}")
    if firing_rates.empty:
        raise ValueError("no phase-binned firing rates are available")

    summary = (
        firing_rates.groupby(
            ["brain_area", "phase_bin_center"], sort=True, as_index=False
        )
        .agg(
            mean_firing_rate_hz=("firing_rate_hz", "mean"),
            sem_firing_rate_hz=("firing_rate_hz", "sem"),
            n_neurons=("neuron_id", "nunique"),
        )
        .fillna({"sem_firing_rate_hz": 0.0})
    )

    figure, axis = plt.subplots(figsize=(8, 4.8))
    for area, area_summary in summary.groupby("brain_area", sort=True):
        x = area_summary["phase_bin_center"].to_numpy()
        mean = area_summary["mean_firing_rate_hz"].to_numpy()
        sem = area_summary["sem_firing_rate_hz"].to_numpy()
        n_neurons = int(area_summary["n_neurons"].max())
        line = axis.plot(
            x,
            mean,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=f"Area {area} (n={n_neurons} neurons)",
        )[0]
        axis.fill_between(
            x,
            np.maximum(0.0, mean - sem),
            mean + sem,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )

    axis.set(
        xlim=(-np.pi, np.pi),
        xlabel="LFP phase (radians)",
        ylabel="Mean firing rate (Hz)",
        title="Mean firing rate across 4–10 Hz LFP phase",
    )
    axis.set_xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    return figure, axis, summary


def calculate_phase_locking_values(
    phases_at_spikes: pd.DataFrame,
    firing_rates: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate spike-based and firing-rate-based PLV for every neuron.

    ``spike_plv`` is the length of the mean unit vector over individual spike
    phases. ``spike_ppc`` is the unbiased mean cosine of all distinct
    spike-phase pairs and therefore does not have PLV's upward bias at low
    spike counts. ``firing_rate_plv`` is the circular resultant across
    phase-bin centres, weighted by the neuron's firing rate in each bin.
    """
    spike_required = {"neuron_id", "brain_area", "phase"}
    rate_required = {
        "neuron_id",
        "brain_area",
        "phase_bin_center",
        "firing_rate_hz",
    }
    spike_missing = spike_required.difference(phases_at_spikes.columns)
    rate_missing = rate_required.difference(firing_rates.columns)
    if spike_missing:
        raise ValueError(
            f"phases_at_spikes is missing columns: {sorted(spike_missing)}"
        )
    if rate_missing:
        raise ValueError(
            f"firing_rates is missing columns: {sorted(rate_missing)}"
        )

    spike_rows: list[dict[str, float | int]] = []
    valid_spikes = phases_at_spikes.dropna(subset=["phase"])
    for (area, neuron_id), group in valid_spikes.groupby(
        ["brain_area", "neuron_id"], sort=True
    ):
        phases = group["phase"].to_numpy(dtype=float)
        phasor_sum = np.exp(1j * phases).sum()
        resultant = phasor_sum / len(phases)
        if len(phases) >= 2:
            ppc = (np.abs(phasor_sum) ** 2 - len(phases)) / (
                len(phases) * (len(phases) - 1)
            )
        else:
            ppc = np.nan
        spike_rows.append(
            {
                "neuron_id": int(neuron_id),
                "brain_area": int(area),
                "n_spikes": len(phases),
                "spike_plv": float(np.abs(resultant)),
                "spike_ppc": float(ppc),
                "spike_preferred_phase": float(np.angle(resultant)),
            }
        )

    rate_rows: list[dict[str, float | int]] = []
    valid_rates = firing_rates.dropna(subset=["firing_rate_hz"])
    for (area, neuron_id), group in valid_rates.groupby(
        ["brain_area", "neuron_id"], sort=True
    ):
        phase_centers = group["phase_bin_center"].to_numpy(dtype=float)
        rates = group["firing_rate_hz"].to_numpy(dtype=float)
        total_rate = rates.sum()
        if total_rate > 0:
            resultant = np.sum(rates * np.exp(1j * phase_centers)) / total_rate
            rate_plv = float(np.abs(resultant))
            preferred_phase = float(np.angle(resultant))
        else:
            rate_plv = np.nan
            preferred_phase = np.nan
        rate_rows.append(
            {
                "neuron_id": int(neuron_id),
                "brain_area": int(area),
                "mean_firing_rate_hz": float(np.mean(rates)),
                "firing_rate_plv": rate_plv,
                "firing_rate_preferred_phase": preferred_phase,
            }
        )

    spike_table = pd.DataFrame(spike_rows)
    rate_table = pd.DataFrame(rate_rows)
    return spike_table.merge(
        rate_table,
        on=["neuron_id", "brain_area"],
        how="outer",
        validate="one_to_one",
    ).sort_values(["brain_area", "neuron_id"], ignore_index=True)


def summarize_phase_locking_by_area(
    neuron_plv: pd.DataFrame,
) -> pd.DataFrame:
    """Return area-wise means and SEMs of neuron-level PLV and PPC measures."""
    required = {
        "neuron_id",
        "brain_area",
        "spike_plv",
        "spike_ppc",
        "firing_rate_plv",
    }
    missing = required.difference(neuron_plv.columns)
    if missing:
        raise ValueError(f"neuron_plv is missing columns: {sorted(missing)}")

    return (
        neuron_plv.groupby("brain_area", sort=True, as_index=False)
        .agg(
            n_neurons=("neuron_id", "nunique"),
            mean_spike_plv=("spike_plv", "mean"),
            sem_spike_plv=("spike_plv", "sem"),
            mean_spike_ppc=("spike_ppc", "mean"),
            sem_spike_ppc=("spike_ppc", "sem"),
            mean_firing_rate_plv=("firing_rate_plv", "mean"),
            sem_firing_rate_plv=("firing_rate_plv", "sem"),
        )
        .fillna(
            {
                "sem_spike_plv": 0.0,
                "sem_spike_ppc": 0.0,
                "sem_firing_rate_plv": 0.0,
            }
        )
    )


def save_phase_locking_results(
    neuron_plv: pd.DataFrame,
    area_summary: pd.DataFrame,
    *,
    neuron_path: str | Path = DEFAULT_NEURON_PLV_TABLE,
    area_path: str | Path = DEFAULT_AREA_PLV_TABLE,
) -> None:
    """Save neuron-level PLVs and their area-wise summary as CSV files."""
    neuron_output = Path(neuron_path)
    area_output = Path(area_path)
    neuron_output.parent.mkdir(parents=True, exist_ok=True)
    area_output.parent.mkdir(parents=True, exist_ok=True)
    neuron_plv.to_csv(neuron_output, index=False)
    area_summary.to_csv(area_output, index=False)


def calculate_all_participant_ppc() -> pd.DataFrame:
    """Process every participant and return one mean-neuron PPC per area.

    Participants are processed sequentially and large intermediate arrays are
    released after each recording.
    """
    participant_rows: list[dict[str, float | int]] = []
    for participant in available_subjects():
        print(f"Calculating PPC for participant {participant}...")
        session = load_participant_data(participant)
        filtered_lfp = bandpass_filter_lfp(session.lfp, session.sampling_rate)
        lfp_phase = instantaneous_lfp_phase(filtered_lfp)
        if session.spikes is None:
            raise RuntimeError(f"participant {participant} spikes were not loaded")
        phases = spike_phases(session.spikes, lfp_phase)
        neuron_ppc = calculate_spike_ppc(phases)

        for area, group in neuron_ppc.groupby("brain_area", sort=True):
            valid_ppc = group["spike_ppc"].dropna()
            participant_rows.append(
                {
                    "participant": int(participant),
                    "brain_area": int(area),
                    "n_neurons": int(valid_ppc.size),
                    "mean_ppc": float(valid_ppc.mean()),
                }
            )

        del session, filtered_lfp, lfp_phase, phases, neuron_ppc
        gc.collect()

    if not participant_rows:
        raise ValueError("no participant PPC results were calculated")
    return pd.DataFrame(participant_rows)


def summarize_all_participant_ppc(
    participant_ppc: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate mean PPC and Student-t 95% CIs across participants."""
    required = {"participant", "brain_area", "mean_ppc"}
    missing = required.difference(participant_ppc.columns)
    if missing:
        raise ValueError(f"participant_ppc is missing columns: {sorted(missing)}")

    summary = (
        participant_ppc.groupby("brain_area", sort=True)["mean_ppc"]
        .agg(n_participants="count", mean_ppc="mean", sd_ppc="std")
        .reset_index()
    )
    summary["sem_ppc"] = summary["sd_ppc"] / np.sqrt(summary["n_participants"])
    critical_t = summary["n_participants"].map(
        lambda count: stats.t.ppf(0.975, count - 1) if count > 1 else np.nan
    )
    margin = critical_t * summary["sem_ppc"]
    summary["ci95_low_ppc"] = summary["mean_ppc"] - margin
    summary["ci95_high_ppc"] = summary["mean_ppc"] + margin
    return summary


def compare_all_participant_ppc(
    participant_ppc: pd.DataFrame,
) -> tuple[int, pd.DataFrame, object]:
    """Run a Friedman test and Holm-corrected paired Wilcoxon tests."""
    summary = summarize_all_participant_ppc(participant_ppc)
    highest_area = int(
        summary.loc[summary["mean_ppc"].idxmax(), "brain_area"]
    )
    wide = participant_ppc.pivot(
        index="participant", columns="brain_area", values="mean_ppc"
    )
    complete = wide.dropna()
    if complete.shape[1] < 3:
        raise ValueError("the Friedman test requires at least three brain areas")
    if complete.shape[0] < 2:
        raise ValueError("at least two complete participants are required")
    friedman = stats.friedmanchisquare(
        *(complete[area].to_numpy() for area in complete.columns)
    )

    comparisons: list[dict[str, float | int | bool]] = []
    raw_p_values: list[float] = []
    for first_area, second_area in combinations(wide.columns, 2):
        paired = wide[[first_area, second_area]].dropna()
        differences = paired[first_area] - paired[second_area]
        if np.allclose(differences, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            result = stats.wilcoxon(
                paired[first_area],
                paired[second_area],
                alternative="two-sided",
            )
            statistic, p_value = float(result.statistic), float(result.pvalue)
        raw_p_values.append(p_value)
        comparisons.append(
            {
                "area_1": int(first_area),
                "area_2": int(second_area),
                "n_participants": len(paired),
                "mean_difference": float(
                    (paired[first_area] - paired[second_area]).mean()
                ),
                "wilcoxon_statistic": statistic,
                "p_value_two_sided": p_value,
            }
        )

    for comparison, adjusted in zip(comparisons, holm_adjust(raw_p_values)):
        comparison["p_value_holm"] = adjusted
        comparison["significant_0.05"] = adjusted < 0.05
    return highest_area, pd.DataFrame(comparisons), friedman


def plot_all_participant_ppc(
    summary: pd.DataFrame,
    *,
    path: str | Path = ALL_PARTICIPANT_PPC_PLOT,
) -> None:
    """Plot area means and 95% confidence intervals across participants."""
    means = summary["mean_ppc"].to_numpy()
    errors = np.vstack(
        [
            means - summary["ci95_low_ppc"].to_numpy(),
            summary["ci95_high_ppc"].to_numpy() - means,
        ]
    )
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.errorbar(
        summary["brain_area"].astype(str),
        means,
        yerr=errors,
        fmt="o",
        capsize=5,
        linewidth=1.5,
    )
    axis.set(
        xlabel="Brain area",
        ylabel="Mean pairwise phase consistency",
        title="PPC across participants",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_all_participants_ppc_analysis() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run, save, plot, and report the complete group PPC analysis."""
    participant_ppc = calculate_all_participant_ppc()
    summary = summarize_all_participant_ppc(participant_ppc)
    highest_area, pairwise, friedman = compare_all_participant_ppc(
        participant_ppc
    )

    DEFAULT_PLOT_PATH.mkdir(parents=True, exist_ok=True)
    participant_ppc.to_csv(ALL_PARTICIPANT_PPC_TABLE, index=False)
    summary.to_csv(ALL_PARTICIPANT_PPC_SUMMARY, index=False)
    pairwise.to_csv(ALL_PARTICIPANT_PPC_TESTS, index=False)
    plot_all_participant_ppc(summary)

    print("\nPPC by brain area (participant-level mean and 95% CI)")
    print(
        summary[
            [
                "brain_area",
                "n_participants",
                "mean_ppc",
                "ci95_low_ppc",
                "ci95_high_ppc",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
    )
    print(f"\nHighest observed mean PPC: brain area {highest_area}")
    print(
        "Friedman omnibus test: "
        f"chi-square={friedman.statistic:.6g}, p={friedman.pvalue:.6g}"
    )
    print("\nPaired Wilcoxon tests (two-sided, Holm corrected)")
    print(pairwise.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    return participant_ppc, summary, pairwise


def _print_summary(
    session: CON2PHYSSession,
    filtered_lfp: Mapping[int, nap.TsdFrame],
    phases_at_spikes: pd.DataFrame | None = None,
) -> None:
    if session.spikes is None:
        raise RuntimeError("participant 1 spike data were not loaded")

    area_counts = session.spikes.metadata["brain_area"].value_counts(sort=False)
    print(f"Participant {session.subject}")
    print(f"  Spike units: {len(session.spikes)}")
    for area, count in area_counts.items():
        print(f"    {area}: {count} units")
    for area, lfp in filtered_lfp.items():
        duration = len(lfp) / session.sampling_rate
        print(
            f"  LFP area {area}: {lfp.shape[1]} channels, "
            f"{duration:.3f} s, filtered {LFP_BAND_HZ[0]:g}-"
            f"{LFP_BAND_HZ[1]:g} Hz"
        )
    if phases_at_spikes is not None:
        valid = phases_at_spikes["phase"].notna().sum()
        print(f"  Spike phases: {valid}/{len(phases_at_spikes)} calculated")


if __name__ == "__main__":
    participant_1, theta_lfp, theta_phase, neuron_spike_phases = (
        run_participant_1_phase_analysis()
    )
    plot_example_neuron_phase_distributions(neuron_spike_phases)
    plot_mean_phase_distribution_by_area(neuron_spike_phases)
    neuron_firing_rates = phase_binned_firing_rates(
        neuron_spike_phases,
        theta_phase,
        participant_1.sampling_rate,
    )
    plot_mean_phase_binned_firing_rate(neuron_firing_rates)
    neuron_plv = calculate_phase_locking_values(
        neuron_spike_phases,
        neuron_firing_rates,
    )
    area_plv = summarize_phase_locking_by_area(neuron_plv)
    save_phase_locking_results(neuron_plv, area_plv)
    print("\nPhase-locking values by brain area")
    print(area_plv.to_string(index=False))
    _print_summary(participant_1, theta_lfp, neuron_spike_phases)
    plt.close("all")
    del (
        participant_1,
        theta_lfp,
        theta_phase,
        neuron_spike_phases,
        neuron_firing_rates,
        neuron_plv,
        area_plv,
    )
    gc.collect()
    run_all_participants_ppc_analysis()
    
