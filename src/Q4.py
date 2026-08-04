"""Pairwise spike cross-correlograms for participant 1.

The analysis is restricted to pairs of units recorded in the same brain
region.  Run it from the repository root with::

    python src/Q4.py

Cross-correlogram values are target-unit firing rates (Hz) relative to spikes
of the reference unit.  Positive lags mean that the target unit fired after
the reference unit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats

if __package__:
    from .data_loader import CON2PHYSSession, load_session, DEFAULT_PLOT_PATH
else:
    from data_loader import CON2PHYSSession, load_session, DEFAULT_PLOT_PATH


PARTICIPANT = 1
BIN_SIZE_MS = 1.0
WINDOW_SIZE_MS = 100.0
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "analysis" / "Q4"


def load_participant_spikes(participant: int = PARTICIPANT) -> CON2PHYSSession:
    """Load spike trains and unit metadata without loading any LFP banks."""
    return load_session(subject=participant, lfp_banks=())


def crosscorrelograms_by_region(
    session: CON2PHYSSession,
    *,
    bin_size_ms: float = BIN_SIZE_MS,
    window_size_ms: float = WINDOW_SIZE_MS,
) -> pd.DataFrame:
    """Calculate every within-region, unordered unit-pair correlogram.

    Returns a tidy table with one row per pair and lag.  Pairs follow
    ``itertools.combinations`` order: ``unit_reference`` is the first unit and
    ``unit_target`` is the second. The rate is not baseline-normalized.
    """
    if session.spikes is None:
        raise ValueError("session must contain spike trains")
    if not np.isfinite(bin_size_ms) or bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be positive")
    if not np.isfinite(window_size_ms) or window_size_ms <= 0:
        raise ValueError("window_size_ms must be positive")
    if window_size_ms < bin_size_ms:
        raise ValueError("window_size_ms must be at least bin_size_ms")
    if "brain_area" not in session.spikes.metadata:
        raise ValueError("spike metadata must contain brain_area")

    regional_groups = session.spikes.getby_category("brain_area")
    tables: list[pd.DataFrame] = []
    for brain_area in sorted(regional_groups):
        group = regional_groups[brain_area]
        if len(group) < 2:
            continue
        correlograms = nap.compute_crosscorrelogram(
            group,
            binsize=float(bin_size_ms),
            windowsize=float(window_size_ms),
            norm=False,
            time_units="ms",
        )
        # Pynapple always reports its result index in seconds, even when the
        # input bin/window units are milliseconds.
        pairs = np.asarray(correlograms.columns.to_list(), dtype=int)
        n_lags, n_pairs = correlograms.shape
        long = pd.DataFrame(
            {
                "lag_ms": np.repeat(
                    correlograms.index.to_numpy(dtype=float) * 1_000, n_pairs
                ),
                "brain_area": brain_area,
                "unit_reference": np.tile(pairs[:, 0], n_lags),
                "unit_target": np.tile(pairs[:, 1], n_lags),
                "rate_hz": correlograms.to_numpy().ravel(),
            }
        )
        tables.append(long)

    if not tables:
        raise ValueError("no brain region contains at least two units")
    result = pd.concat(tables, ignore_index=True)
    result.insert(0, "participant", session.subject)
    return result


def trialwise_crosscorrelogram_summary(
    session: CON2PHYSSession,
    *,
    bin_size_ms: float = BIN_SIZE_MS,
    window_size_ms: float = WINDOW_SIZE_MS,
) -> pd.DataFrame:
    """Calculate pairwise CCGs per stimulus-to-outcome trial and summarize them.

    Every within-region unit pair is calculated by Pynapple for every trial.
    Pair values are reduced immediately to the mean and SEM at each lag to
    avoid materializing roughly 160 million rows for participant 1.
    """
    if session.spikes is None:
        raise ValueError("session must contain spike trains")
    required = {"stim_start", "outcome"}
    missing = required.difference(session.trial_table.columns)
    if missing:
        raise ValueError(f"trial table is missing columns: {sorted(missing)}")
    if not np.isfinite(bin_size_ms) or bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be positive")
    if not np.isfinite(window_size_ms) or window_size_ms < bin_size_ms:
        raise ValueError("window_size_ms must be at least bin_size_ms")

    regional_groups = session.spikes.getby_category("brain_area")
    tables: list[pd.DataFrame] = []
    for trial_id, trial in session.trial_table.iterrows():
        if trial["outcome"] <= trial["stim_start"]:
            raise ValueError("every outcome must occur after stimulus presentation")
        epoch = nap.IntervalSet(
            start=[trial["stim_start"]],
            end=[trial["outcome"]],
            time_units="s",
        )
        for brain_area, group in sorted(regional_groups.items()):
            if len(group) < 2:
                continue
            pairwise = nap.compute_crosscorrelogram(
                group,
                binsize=float(bin_size_ms),
                windowsize=float(window_size_ms),
                ep=epoch,
                norm=False,
                time_units="ms",
            )
            values = pairwise.to_numpy(dtype=float)
            finite = np.isfinite(values)
            n_pairs = finite.sum(axis=1)
            means = np.divide(
                np.nansum(values, axis=1),
                n_pairs,
                out=np.full(values.shape[0], np.nan),
                where=n_pairs > 0,
            )
            squared_deviations = np.where(
                finite, (values - means[:, np.newaxis]) ** 2, 0.0
            ).sum(axis=1)
            sd = np.sqrt(
                np.divide(
                    squared_deviations,
                    n_pairs - 1,
                    out=np.full(values.shape[0], np.nan),
                    where=n_pairs > 1,
                )
            )
            sem = np.divide(
                sd,
                np.sqrt(n_pairs),
                out=np.full(values.shape[0], np.nan),
                where=n_pairs > 1,
            )
            tables.append(
                pd.DataFrame(
                    {
                        "participant": session.subject,
                        "trial": trial_id,
                        "brain_area": brain_area,
                        "stim_start_s": trial["stim_start"],
                        "outcome_s": trial["outcome"],
                        "lag_ms": pairwise.index.to_numpy(dtype=float) * 1_000,
                        "n_valid_pairs": n_pairs,
                        "mean_rate_hz": means,
                        "sem_rate_hz": sem,
                    }
                )
            )
    if not tables:
        raise ValueError("no brain region contains at least two units")
    return pd.concat(tables, ignore_index=True)


def plot_trialwise_crosscorrelograms(
    summary: pd.DataFrame,
    path: str | Path = DEFAULT_PLOT_PATH
    / "participant_1_trialwise_crosscorrelograms.png",
) -> None:
    """Plot trial-by-lag heatmaps of mean within-region CCG rates."""
    required = {"trial", "brain_area", "lag_ms", "mean_rate_hz"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")

    areas = sorted(summary["brain_area"].unique())
    fig, axes = plt.subplots(
        1, len(areas), figsize=(5 * len(areas), 5), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    image = None
    for ax, brain_area in zip(axes, areas):
        matrix = summary[summary["brain_area"] == brain_area].pivot(
            index="trial", columns="lag_ms", values="mean_rate_hz"
        )
        image = ax.imshow(
            matrix.to_numpy(),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[
                matrix.columns.min(),
                matrix.columns.max(),
                -0.5,
                len(matrix.index) - 0.5,
            ],
        )
        ax.axvline(0, color="white", linestyle=":", linewidth=0.8)
        ax.set_title(f"Area {brain_area}")
        ax.set_xlabel("Lag (ms)")
    axes[0].set_ylabel("Trial")
    if image is not None:
        fig.colorbar(image, ax=axes, label="Mean target firing rate (Hz)")
    fig.suptitle(
        f"Participant {PARTICIPANT}: trial-wise cross-correlograms "
        "(stimulus to outcome)"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def trial_normalized_firing_rates(
    session: CON2PHYSSession,
    *,
    n_bins: int = 100,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Calculate population firing rates over normalized full-trial time.

    Each trial is divided into equal-duration bins from ``trial_start`` to
    ``trial_end``. Within each region, rates are first averaged over neurons;
    the summary mean and confidence interval are then calculated over trials.
    Trials whose event times are not ordered start < stimulus < outcome < end
    are excluded.
    """
    if session.spikes is None:
        raise ValueError("session must contain spike trains")
    required = {"trial_start", "stim_start", "outcome", "trial_end"}
    missing = required.difference(session.trial_table.columns)
    if missing:
        raise ValueError(f"trial table is missing columns: {sorted(missing)}")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")

    trials = session.trial_table
    valid = (
        (trials["trial_start"] < trials["stim_start"])
        & (trials["stim_start"] < trials["outcome"])
        & (trials["outcome"] < trials["trial_end"])
    )
    trials = trials.loc[valid]
    if trials.empty:
        raise ValueError("no trials have correctly ordered event times")

    regional_groups = session.spikes.getby_category("brain_area")
    bin_centres_percent = (np.arange(n_bins) + 0.5) * 100 / n_bins
    records: list[pd.DataFrame] = []
    for trial_id, trial in trials.iterrows():
        bin_duration = (trial["trial_end"] - trial["trial_start"]) / n_bins
        epoch = nap.IntervalSet(
            start=[trial["trial_start"]], end=[trial["trial_end"]], time_units="s"
        )
        for brain_area, group in sorted(regional_groups.items()):
            counts = np.asarray(
                group.count(bin_size=bin_duration, ep=epoch), dtype=float
            )
            if counts.shape[0] != n_bins:
                raise RuntimeError("Pynapple returned an unexpected number of trial bins")
            population_rate = counts.mean(axis=1) / bin_duration
            records.append(
                pd.DataFrame(
                    {
                        "participant": session.subject,
                        "trial": trial_id,
                        "brain_area": brain_area,
                        "trial_time_percent": bin_centres_percent,
                        "mean_neuron_rate_hz": population_rate,
                    }
                )
            )
    trial_rates = pd.concat(records, ignore_index=True)
    summary = (
        trial_rates.groupby(["brain_area", "trial_time_percent"])[
            "mean_neuron_rate_hz"
        ]
        .agg(n_trials="count", mean_rate_hz="mean", sd_rate_hz="std")
        .reset_index()
    )
    summary["sem_rate_hz"] = summary["sd_rate_hz"] / np.sqrt(summary["n_trials"])
    critical_t = stats.t.ppf(
        0.5 + confidence / 2, summary["n_trials"] - 1
    )
    margin = critical_t * summary["sem_rate_hz"]
    summary["ci95_low_hz"] = summary["mean_rate_hz"] - margin
    summary["ci95_high_hz"] = summary["mean_rate_hz"] + margin

    durations = trials["trial_end"] - trials["trial_start"]
    timeline = {
        "n_trials": float(len(trials)),
        "stimulus_percent": float(
            np.median((trials["stim_start"] - trials["trial_start"]) / durations)
            * 100
        ),
        "outcome_percent": float(
            np.median((trials["outcome"] - trials["trial_start"]) / durations)
            * 100
        ),
        "median_duration_s": float(np.median(durations)),
    }
    return trial_rates, summary, timeline


def plot_trial_normalized_firing_rates(
    summary: pd.DataFrame,
    timeline: dict[str, float],
    path: str | Path = DEFAULT_PLOT_PATH
    / "participant_1_trial_normalized_firing_rate.png",
) -> None:
    """Plot across-trial mean neuronal firing rates with 95% CIs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for brain_area, area_data in summary.groupby("brain_area", sort=True):
        x = area_data["trial_time_percent"].to_numpy(dtype=float)
        mean = area_data["mean_rate_hz"].to_numpy(dtype=float)
        low = area_data["ci95_low_hz"].to_numpy(dtype=float)
        high = area_data["ci95_high_hz"].to_numpy(dtype=float)
        line = ax.plot(x, mean, label=f"Area {brain_area}", linewidth=1.6)[0]
        ax.fill_between(x, low, high, color=line.get_color(), alpha=0.2)

    event_positions = {
        "Stimulus": timeline["stimulus_percent"],
        "Outcome": timeline["outcome_percent"],
    }
    for label, position in event_positions.items():
        ax.axvline(position, color="0.3", linestyle="--", linewidth=1)
        ax.text(
            position,
            1.01,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    timeline_ax = ax.twiny()
    timeline_ax.set_xlim(0, 100)
    timeline_ax.set_xticks(
        [0, timeline["stimulus_percent"], timeline["outcome_percent"], 100],
        ["Trial start", "Stimulus", "Outcome", "Trial end"],
    )
    timeline_ax.tick_params(axis="x", length=7, direction="out", pad=3)
    ax.set_xlim(0, 100)
    ax.set(
        xlabel="Normalized trial time (%)",
        ylabel="Mean firing rate per neuron (Hz)",
        title=(
            f"Participant {PARTICIPANT}: firing rate across full trials\n"
            f"mean and 95% CI over {int(timeline['n_trials'])} trials; "
            f"median duration {timeline['median_duration_s']:.2f} s"
        ),
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_crosscorrelograms(correlograms: pd.DataFrame) -> pd.DataFrame:
    """Return the across-pair mean and SEM at every lag for each region."""
    required = {"brain_area", "lag_ms", "rate_hz"}
    missing = required.difference(correlograms.columns)
    if missing:
        raise ValueError(f"correlograms are missing columns: {sorted(missing)}")
    summary = (
        correlograms.groupby(["brain_area", "lag_ms"], sort=True)["rate_hz"]
        .agg(n_pairs="count", mean_rate_hz="mean", sd_rate_hz="std")
        .reset_index()
    )
    summary["sem_rate_hz"] = summary["sd_rate_hz"] / np.sqrt(summary["n_pairs"])
    return summary


def isi_distributions_by_region(
    session: CON2PHYSSession,
    *,
    n_bins: int = 80,
) -> pd.DataFrame:
    """Compute regional ISI distributions with Pynapple."""
    if session.spikes is None:
        raise ValueError("session must contain spike trains")
    if "brain_area" not in session.spikes.metadata:
        raise ValueError("spike metadata must contain brain_area")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")

    tables: list[pd.DataFrame] = []
    for brain_area, group in sorted(
        session.spikes.getby_category("brain_area").items()
    ):
        distribution = nap.compute_isi_distribution(
            group,
            bins=n_bins,
            log_scale=True,
        )
        counts = distribution.sum(axis=1).to_numpy(dtype=int)
        total = counts.sum()
        if total == 0:
            continue
        tables.append(
            pd.DataFrame(
                {
                    "participant": session.subject,
                    "brain_area": brain_area,
                    "isi_ms": np.exp(
                        distribution.index.to_numpy(dtype=float)
                    )
                    * 1_000,
                    "count": counts,
                    "probability": counts / total,
                }
            )
        )
    if not tables:
        raise ValueError("no region contains calculable inter-spike intervals")
    return pd.concat(tables, ignore_index=True)


def trialwise_isi_distributions(
    session: CON2PHYSSession,
    *,
    n_bins: int = 80,
    minimum_isi_ms: float = 0.1,
) -> pd.DataFrame:
    """Compute ISI distributions from stimulus presentation to outcome.

    A separate Pynapple epoch is used for each trial, ensuring that an ISI
    never crosses a trial boundary. Common logarithmic bins make distributions
    directly comparable across trials and brain regions.
    """
    if session.spikes is None:
        raise ValueError("session must contain spike trains")
    required = {"stim_start", "outcome"}
    missing = required.difference(session.trial_table.columns)
    if missing:
        raise ValueError(f"trial table is missing columns: {sorted(missing)}")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    if not np.isfinite(minimum_isi_ms) or minimum_isi_ms <= 0:
        raise ValueError("minimum_isi_ms must be positive")

    durations = (
        session.trial_table["outcome"] - session.trial_table["stim_start"]
    )
    if (durations <= 0).any():
        raise ValueError("every outcome must occur after stimulus presentation")
    maximum_isi_seconds = float(durations.max())
    minimum_isi_seconds = minimum_isi_ms / 1_000
    if maximum_isi_seconds <= minimum_isi_seconds:
        raise ValueError("trial windows are shorter than minimum_isi_ms")
    # compute_isi_distribution applies log() before histogramming when
    # log_scale=True, so its explicit bin edges must also be in log-seconds.
    log_bin_edges = np.linspace(
        np.log(minimum_isi_seconds),
        np.log(maximum_isi_seconds),
        n_bins + 1,
    )

    regional_groups = session.spikes.getby_category("brain_area")
    tables: list[pd.DataFrame] = []
    for trial_id, trial in session.trial_table.iterrows():
        epoch = nap.IntervalSet(
            start=[trial["stim_start"]],
            end=[trial["outcome"]],
            time_units="s",
        )
        for brain_area, group in sorted(regional_groups.items()):
            distribution = nap.compute_isi_distribution(
                group,
                bins=log_bin_edges,
                log_scale=True,
                epochs=epoch,
            )
            counts = distribution.sum(axis=1).to_numpy(dtype=int)
            total = counts.sum()
            tables.append(
                pd.DataFrame(
                    {
                        "participant": session.subject,
                        "trial": trial_id,
                        "brain_area": brain_area,
                        "stim_start_s": trial["stim_start"],
                        "outcome_s": trial["outcome"],
                        "isi_ms": np.exp(
                            distribution.index.to_numpy(dtype=float)
                        )
                        * 1_000,
                        "count": counts,
                        "probability": counts / total if total else 0.0,
                        "n_intervals": total,
                    }
                )
            )
    return pd.concat(tables, ignore_index=True)


def plot_trialwise_isi_distributions(
    distributions: pd.DataFrame,
    path: str | Path = DEFAULT_PLOT_PATH
    / "participant_1_trialwise_isi_distribution.png",
) -> None:
    """Plot trial-by-ISI probability heatmaps, one per brain region."""
    required = {"trial", "brain_area", "isi_ms", "probability"}
    missing = required.difference(distributions.columns)
    if missing:
        raise ValueError(f"distributions are missing columns: {sorted(missing)}")

    areas = sorted(distributions["brain_area"].unique())
    fig, axes = plt.subplots(
        1, len(areas), figsize=(5 * len(areas), 5), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    image = None
    for ax, brain_area in zip(axes, areas):
        area_data = distributions[distributions["brain_area"] == brain_area]
        matrix = area_data.pivot(
            index="trial", columns="isi_ms", values="probability"
        )
        image = ax.imshow(
            matrix.to_numpy(),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[
                np.log10(matrix.columns.min()),
                np.log10(matrix.columns.max()),
                -0.5,
                len(matrix.index) - 0.5,
            ],
        )
        tick_values = np.array([1, 10, 100, 1_000], dtype=float)
        visible = tick_values[
            (tick_values >= matrix.columns.min())
            & (tick_values <= matrix.columns.max())
        ]
        ax.set_xticks(np.log10(visible), [f"{value:g}" for value in visible])
        ax.set_title(f"Area {brain_area}")
        ax.set_xlabel("ISI (ms, log scale)")
    axes[0].set_ylabel("Trial")
    if image is not None:
        fig.colorbar(image, ax=axes, label="Probability per log-ISI bin")
    fig.suptitle(
        f"Participant {PARTICIPANT}: trial-wise ISIs (stimulus to outcome)"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_inter_spike_interval_distribution(
    distributions: pd.DataFrame,
    path: str | Path = DEFAULT_PLOT_PATH / "participant_1_isi_distribution.png",
) -> None:
    """Plot regional ISI probability distributions returned by Pynapple."""
    required = {"brain_area", "isi_ms", "probability"}
    missing = required.difference(distributions.columns)
    if missing:
        raise ValueError(f"distributions are missing columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for brain_area, area_data in distributions.groupby("brain_area", sort=True):
        ax.plot(
            area_data["isi_ms"],
            area_data["probability"],
            linewidth=1.6,
            label=f"Area {brain_area}",
        )
    ax.set_xscale("log")
    ax.set(
        xlabel="Inter-spike interval (ms)",
        ylabel="Probability per log-ISI bin",
        title=f"Participant {PARTICIPANT}: inter-spike interval distributions",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_region_crosscorrelograms(
    summary: pd.DataFrame,
    path: str | Path = DEFAULT_PLOT_PATH / "participant_1_crosscorrelograms.png",
) -> None:
    """Plot the mean within-region cross-correlogram with SEM bands."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for brain_area, area_data in summary.groupby("brain_area", sort=True):
        lag = area_data["lag_ms"].to_numpy(dtype=float)
        mean = area_data["mean_rate_hz"].to_numpy(dtype=float)
        sem = area_data["sem_rate_hz"].fillna(0).to_numpy(dtype=float)
        line = ax.plot(lag, mean, label=f"Area {brain_area}")[0]
        ax.fill_between(lag, mean - sem, mean + sem, color=line.get_color(), alpha=0.2)
    ax.axvline(0, color="0.35", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Lag (ms; target relative to reference)",
        ylabel="Target firing rate (Hz)",
        title=f"Participant {PARTICIPANT}: within-region cross-correlograms",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    session = load_participant_spikes()
    correlograms = crosscorrelograms_by_region(session)
    summary = summarize_crosscorrelograms(correlograms)
    isi_distributions = isi_distributions_by_region(session)
    trialwise_isi = trialwise_isi_distributions(session)
    trialwise_ccg = trialwise_crosscorrelogram_summary(session)
    trial_rates, firing_rate_summary, trial_timeline = (
        trial_normalized_firing_rates(session)
    )

    DEFAULT_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    correlograms.to_csv(
        DEFAULT_OUTPUT_PATH / "participant_1_pairwise_crosscorrelograms.csv",
        index=False,
    )
    summary.to_csv(
        DEFAULT_OUTPUT_PATH / "participant_1_crosscorrelogram_summary.csv",
        index=False,
    )
    plot_region_crosscorrelograms(summary)
    isi_distributions.to_csv(
        DEFAULT_OUTPUT_PATH / "participant_1_isi_distribution.csv",
        index=False,
    )
    plot_inter_spike_interval_distribution(isi_distributions)
    trialwise_isi.to_csv(
        DEFAULT_OUTPUT_PATH / "participant_1_trialwise_isi_distribution.csv",
        index=False,
    )
    plot_trialwise_isi_distributions(trialwise_isi)
    trialwise_ccg.to_csv(
        DEFAULT_OUTPUT_PATH / "participant_1_trialwise_crosscorrelogram_summary.csv",
        index=False,
    )
    plot_trialwise_crosscorrelograms(trialwise_ccg)
    trial_rates.to_csv(
        DEFAULT_OUTPUT_PATH / "participant_1_trial_normalized_firing_rates.csv",
        index=False,
    )
    firing_rate_summary.to_csv(
        DEFAULT_OUTPUT_PATH / "participant_1_trial_firing_rate_summary.csv",
        index=False,
    )
    plot_trial_normalized_firing_rates(firing_rate_summary, trial_timeline)

    pairs = correlograms[
        ["brain_area", "unit_reference", "unit_target"]
    ].drop_duplicates()
    counts = pairs.groupby("brain_area").size().rename("n_unit_pairs")
    unit_counts = session.spikes.metadata.groupby("brain_area").size().rename("n_units")
    print("\nParticipant 1 within-region cross-correlograms")
    print(pd.concat([unit_counts, counts], axis=1).to_string())
    print(
        f"\nBin size: {BIN_SIZE_MS:g} ms; lag window: "
        f"±{WINDOW_SIZE_MS:g} ms; values: target firing rate (Hz)"
    )
    print(f"Results written to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
