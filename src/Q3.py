"""Compare consensus sharp-wave ripple density across all participants.

Candidate events are detected independently on every channel with Pynapple's
``detect_oscillatory_events``, following its sharp-wave ripple tutorial.
Only events detected on at least 10 channels in an area are retained.  Group
estimates and statistical tests use participants as observations.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
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
    from data_loader import (  # type: ignore[no-redef]
        CON2PHYSSession,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )
    from stat_test import holm_adjust


LFP_BANKS = (1, 2, 3)
RIPPLE_BAND_HZ = (120.0, 250.0)
MIN_CONSENSUS_CHANNELS = 10


def load_subject_lfp(
    subject: int = 1,
    lfp_banks: Sequence[int] = LFP_BANKS,
) -> CON2PHYSSession:
    """Load selected LFP banks for one participant without spike data."""
    return load_session(
        subject=subject,
        lfp_banks=lfp_banks,
        load_spikes=False,
        lfp_dtype="float32",
    )


def load_subject_1_lfp(
    lfp_banks: Sequence[int] = LFP_BANKS,
) -> CON2PHYSSession:
    """Load the LFP data used by this participant-1 analysis."""
    return load_subject_lfp(1, lfp_banks)


def print_lfp_summary(session: CON2PHYSSession) -> None:
    """Print channel counts and recording durations for the loaded banks."""
    print(f"Participant {session.subject} LFP data")
    for area, lfp in session.lfp.items():
        duration = len(lfp) / session.sampling_rate
        print(f"  Area {area}: {lfp.shape[1]} channels, {duration:.3f} seconds")


def detect_sharp_wave_ripples(
    lfp: nap.Tsd | np.ndarray,
    sampling_rate: float | None = None,
    *,
    band: tuple[float, float] = RIPPLE_BAND_HZ,
    threshold_band: tuple[float, float] = (3.0, 15.0),
    duration_band: tuple[float, float] = (0.030, 0.300),
    min_interval: float = 0.020,
    sliding_window_size: int = 10,
) -> nap.IntervalSet:
    """Detect ripple epochs using Pynapple's tutorial implementation.

    ``lfp`` may be a :class:`pynapple.Tsd` or, for convenience, a 1-D NumPy
    array accompanied by ``sampling_rate``.  The returned ``IntervalSet``
    includes Pynapple's power, amplitude, and peak-time metadata.
    """
    if isinstance(lfp, np.ndarray):
        values = np.asarray(lfp, dtype=float)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("lfp must be one-dimensional with at least 2 samples")
        if sampling_rate is None:
            raise ValueError("sampling_rate is required when lfp is a NumPy array")
        lfp = nap.Tsd(t=np.arange(values.size) / sampling_rate, d=values)
    elif not isinstance(lfp, nap.Tsd):
        raise TypeError("lfp must be a pynapple.Tsd or one-dimensional NumPy array")

    fs = float(lfp.rate if sampling_rate is None else sampling_rate)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling_rate must be positive")
    low, requested_high = band
    nyquist = fs / 2
    high = min(float(requested_high), nyquist * 0.99)
    if not (0 < low < high):
        raise ValueError(f"band must overlap frequencies below {nyquist:g} Hz")
    if not (
        0 <= threshold_band[0] < threshold_band[1]
        and 0 < duration_band[0] <= duration_band[1]
        and min_interval >= 0
        and isinstance(sliding_window_size, (int, np.integer))
        and sliding_window_size > 0
    ):
        raise ValueError("invalid ripple detection parameters")
    if not np.isfinite(np.asarray(lfp.values, dtype=float)).all():
        raise ValueError("Pynapple ripple detection requires finite LFP values")

    return nap.detect_oscillatory_events(
        data=lfp,
        epochs=lfp.time_support,
        frequency_band=(float(low), high),
        threshold_band=threshold_band,
        duration_band=duration_band,
        min_interval=min_interval,
        fs=fs,
        sliding_window_size=int(sliding_window_size),
    )


def consensus_ripple_events(
    channel_events: pd.DataFrame,
    *,
    min_channels: int = MIN_CONSENSUS_CHANNELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group overlapping detections and retain multi-channel SWR events.

    Events are grouped separately within each brain area.  A group is kept
    when detections from at least ``min_channels`` distinct channels overlap
    directly or transitively.  The second returned table contains the
    channel-level detections belonging to retained consensus events.
    """
    required = {"brain_area", "channel", "start", "end"}
    missing = required.difference(channel_events.columns)
    if missing:
        raise ValueError(f"channel_events is missing columns: {sorted(missing)}")
    if not isinstance(min_channels, (int, np.integer)) or min_channels < 1:
        raise ValueError("min_channels must be a positive integer")

    consensus_rows: list[dict[str, float | int]] = []
    retained_frames: list[pd.DataFrame] = []
    event_id = 0
    for area, area_events in channel_events.groupby("brain_area", sort=True):
        ordered = area_events.sort_values(["start", "end"])
        cluster_indices: list[int] = []
        cluster_end = -np.inf

        def keep_cluster(indices: list[int]) -> None:
            nonlocal event_id
            if not indices:
                return
            cluster = ordered.loc[indices]
            participating = cluster["channel"].nunique()
            if participating < min_channels:
                return
            event_id += 1
            retained = cluster.copy()
            retained.insert(0, "consensus_event_id", event_id)
            retained_frames.append(retained)
            row: dict[str, float | int] = {
                "consensus_event_id": event_id,
                "brain_area": int(area),
                "start": float(cluster["start"].min()),
                "end": float(cluster["end"].max()),
                "duration_seconds": float(
                    cluster["end"].max() - cluster["start"].min()
                ),
                "n_channels": int(participating),
            }
            for column, operation in (
                ("peak_time", "median"),
                ("power", "mean"),
                ("amplitude", "max"),
            ):
                if column in cluster:
                    row[column] = float(getattr(cluster[column], operation)())
            consensus_rows.append(row)

        for index, row in ordered.iterrows():
            if cluster_indices and float(row["start"]) > cluster_end:
                keep_cluster(cluster_indices)
                cluster_indices = []
                cluster_end = -np.inf
            cluster_indices.append(index)
            cluster_end = max(cluster_end, float(row["end"]))
        keep_cluster(cluster_indices)

    consensus = pd.DataFrame(
        consensus_rows,
        columns=[
            "consensus_event_id",
            "brain_area",
            "start",
            "end",
            "duration_seconds",
            "n_channels",
            "peak_time",
            "power",
            "amplitude",
        ],
    )
    retained = (
        pd.concat(retained_frames, ignore_index=True)
        if retained_frames
        else channel_events.iloc[0:0].copy()
    )
    return consensus, retained


def ripple_density_by_area(
    session: CON2PHYSSession,
    *,
    min_channels: int = MIN_CONSENSUS_CHANNELS,
    **detector_options: float | tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return consensus events, channel participation, and area densities."""
    event_frames: list[pd.DataFrame] = []
    channel_rows: list[dict[str, float | int | str]] = []
    for area, lfp in session.lfp.items():
        values = np.asarray(lfp.values)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError(f"brain area {area} has too few LFP samples")
        duration_minutes = values.shape[0] / session.sampling_rate / 60
        for index, channel in enumerate(lfp.columns):
            ripple_epochs = detect_sharp_wave_ripples(
                lfp[:, index],
                session.sampling_rate,
                **detector_options,
            )
            event_table = ripple_epochs.as_dataframe()
            event_table["duration_seconds"] = (
                event_table["end"] - event_table["start"]
            )
            event_table.insert(0, "channel", str(channel))
            event_table.insert(0, "brain_area", int(area))
            event_frames.append(event_table)
            channel_rows.append(
                {
                    "brain_area": int(area),
                    "channel": str(channel),
                    "n_ripples": len(ripple_epochs),
                    "duration_minutes": duration_minutes,
                    "ripple_density_per_minute": (
                        len(ripple_epochs) / duration_minutes
                    ),
                }
            )

    all_channel_events = pd.concat(event_frames, ignore_index=True)
    channels = pd.DataFrame(channel_rows)
    if channels.empty:
        raise ValueError("session contains no LFP channels")
    events, retained = consensus_ripple_events(
        all_channel_events, min_channels=min_channels
    )
    participation = (
        retained.drop_duplicates(
            ["consensus_event_id", "brain_area", "channel"]
        )
        .groupby(["brain_area", "channel"])
        .size()
        .rename("n_ripples")
    )
    channels = channels.drop(columns=["n_ripples", "ripple_density_per_minute"])
    channels = channels.merge(
        participation,
        how="left",
        left_on=["brain_area", "channel"],
        right_index=True,
    )
    channels["n_ripples"] = channels["n_ripples"].fillna(0).astype(int)
    channels["ripple_density_per_minute"] = (
        channels["n_ripples"] / channels["duration_minutes"]
    )

    area_means = channels.groupby("brain_area").agg(
        n_channels=("channel", "count"),
        duration_minutes=("duration_minutes", "first"),
    )
    consensus_counts = events.groupby("brain_area").size().rename("total_ripples")
    area_means = area_means.join(consensus_counts, how="left")
    area_means["total_ripples"] = area_means["total_ripples"].fillna(0).astype(int)
    area_means["ripple_density_per_minute"] = (
        area_means["total_ripples"] / area_means["duration_minutes"]
    )
    area_means["minimum_consensus_channels"] = min_channels
    return events, channels, area_means


def load_all_subject_ripple_density(
    *,
    min_channels: int = MIN_CONSENSUS_CHANNELS,
    save_event_tables: bool = False,
    **detector_options: float | tuple[float, float],
) -> pd.DataFrame:
    """Calculate one consensus-SWR density per participant and brain area."""
    results: list[pd.DataFrame] = []
    for subject in available_subjects():
        print(f"Calculating participant {subject}...")
        session = load_subject_lfp(subject)
        print_lfp_summary(session)
        events, channels, areas = ripple_density_by_area(
            session,
            min_channels=min_channels,
            **detector_options,
        )
        frame = areas.reset_index()
        frame.insert(0, "subject", subject)
        results.append(frame)
        if save_event_tables:
            DEFAULT_PLOT_PATH.mkdir(parents=True, exist_ok=True)
            events.to_csv(
                DEFAULT_PLOT_PATH / f"participant_{subject}_ripple_events.csv",
                index=False,
            )
            channels.to_csv(
                DEFAULT_PLOT_PATH
                / f"participant_{subject}_ripple_density_by_channel.csv",
                index=False,
            )
        del session, events, channels, areas
    if not results:
        raise ValueError("no participant LFP data were found")
    return pd.concat(results, ignore_index=True)


def summarize_subject_ripple_density(
    subject_density: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate means and Student-t 95% CIs across participants."""
    required = {"subject", "brain_area", "ripple_density_per_minute"}
    missing = required.difference(subject_density.columns)
    if missing:
        raise ValueError(f"subject_density is missing columns: {sorted(missing)}")
    summary = (
        subject_density.groupby("brain_area")["ripple_density_per_minute"]
        .agg(
            n_subjects="count",
            mean_density="mean",
            sd_density="std",
        )
    )
    summary["sem_density"] = (
        summary["sd_density"] / np.sqrt(summary["n_subjects"])
    )
    critical_t = summary["n_subjects"].map(
        lambda n: stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    )
    margin = critical_t * summary["sem_density"]
    summary["ci95_low"] = summary["mean_density"] - margin
    summary["ci95_high"] = summary["mean_density"] + margin
    return summary


def compare_ripple_density_by_area(
    subject_density: pd.DataFrame,
) -> tuple[int, pd.DataFrame, object]:
    """Compare the highest-mean area with other areas using paired tests."""
    summary = summarize_subject_ripple_density(subject_density)
    highest_area = int(summary["mean_density"].idxmax())
    complete = subject_density.pivot(
        index="subject",
        columns="brain_area",
        values="ripple_density_per_minute",
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
        differences = (
            complete[highest_area].to_numpy()
            - complete[other_area].to_numpy()
        )
        if np.allclose(differences, 0):
            statistic, p_value = 0.0, 1.0
        else:
            test = stats.wilcoxon(
                complete[highest_area],
                complete[other_area],
                alternative="greater",
            )
            statistic, p_value = float(test.statistic), float(test.pvalue)
        raw_p_values.append(p_value)
        comparisons.append(
            {
                "highest_area": highest_area,
                "comparison_area": int(other_area),
                "n_subjects": len(complete),
                "wilcoxon_statistic": statistic,
                "p_value_one_sided": p_value,
            }
        )
    for comparison, adjusted in zip(comparisons, holm_adjust(raw_p_values)):
        comparison["p_value_holm"] = adjusted
        comparison["significant_0.05"] = adjusted < 0.05
    return highest_area, pd.DataFrame(comparisons), friedman


def plot_ripple_density(
    summary: pd.DataFrame,
    path: str | Path = DEFAULT_PLOT_PATH / "ripple_density_all_participants.png",
) -> None:
    """Plot across-participant mean consensus-SWR density and 95% CIs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    means = summary["mean_density"].to_numpy()
    errors = np.vstack(
        [
            means - summary["ci95_low"].to_numpy(),
            summary["ci95_high"].to_numpy() - means,
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        summary.index.astype(str),
        means,
        yerr=errors,
        fmt="o",
        capsize=5,
        linewidth=1.5,
    )
    ax.set(
        xlabel="Brain area",
        ylabel="Ripple events per minute",
        title=(
            "Mean consensus sharp-wave ripple density\n"
            f"(detected on ≥{MIN_CONSENSUS_CHANNELS} channels)"
        ),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    subject_density = load_all_subject_ripple_density()
    summary = summarize_subject_ripple_density(subject_density)
    highest_area, pairwise, friedman = compare_ripple_density_by_area(
        subject_density
    )

    DEFAULT_PLOT_PATH.mkdir(parents=True, exist_ok=True)
    subject_density.to_csv(
        DEFAULT_PLOT_PATH / "ripple_density_by_participant.csv",
        index=False,
    )
    summary.to_csv(DEFAULT_PLOT_PATH / "ripple_density_group_summary.csv")
    pairwise.to_csv(
        DEFAULT_PLOT_PATH / "ripple_density_pairwise_tests.csv",
        index=False,
    )
    plot_ripple_density(summary)

    print("\nConsensus sharp-wave ripple density (mean and 95% CI)")
    print(
        summary[
            ["n_subjects", "mean_density", "ci95_low", "ci95_high"]
        ].to_string(float_format=lambda value: f"{value:.6g}")
    )
    print(f"\nHighest observed mean: brain area {highest_area}")
    print(
        "Friedman omnibus test: "
        f"chi-square={friedman.statistic:.6g}, p={friedman.pvalue:.6g}"
    )
    print("\nOne-sided paired Wilcoxon tests (highest area > comparison area)")
    print(pairwise.to_string(index=False, float_format=lambda value: f"{value:.6g}"))


if __name__ == "__main__":
    main()
