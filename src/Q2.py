"""Compare broadband LFP power between brain areas across all subjects.

For each subject, 1–100 Hz power is calculated per channel with Welch's
method and averaged across channels within each brain area. Group estimates
and statistical tests use subjects, rather than channels, as observations.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

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
BROADBAND_HZ = (1.0, 100.0)


def load_subject_lfp(
    subject: int = 1,
    lfp_banks: Sequence[int] = LFP_BANKS,
) -> CON2PHYSSession:
    """Load selected LFP banks for one subject without spike data."""
    return load_session(
        subject=subject,
        lfp_banks=lfp_banks,
        load_spikes=False,
        lfp_dtype="float32",
    )


def load_subject_1_lfp(
    lfp_banks: Sequence[int] = LFP_BANKS,
) -> CON2PHYSSession:
    """Backward-compatible subject-1 loader."""
    return load_subject_lfp(1, lfp_banks)


def print_lfp_summary(session: CON2PHYSSession) -> None:
    """Print the dimensions and duration of each LFP bank."""
    print(f"Subject {session.subject} LFP data")
    for bank, lfp in session.lfp.items():
        duration = len(lfp) / session.sampling_rate
        print(
            f"  Area {bank}: {lfp.shape[1]} channels, "
            f"{duration:.3f} seconds"
        )


def broadband_power_by_area(
    session: CON2PHYSSession,
    band: tuple[float, float] = BROADBAND_HZ,
    *,
    window_seconds: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return channel-level power and the channel mean for each brain area."""
    low, high = band
    nyquist = session.sampling_rate / 2
    if not (0 <= low < high <= nyquist):
        raise ValueError(f"band must satisfy 0 <= low < high <= {nyquist:g} Hz")
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    rows: list[dict[str, float | int | str]] = []
    for area, lfp in session.lfp.items():
        values = np.asarray(lfp.values)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError(f"brain area {area} has too few LFP samples")
        nperseg = min(
            values.shape[0],
            max(2, round(window_seconds * session.sampling_rate)),
        )
        frequencies, psd = signal.welch(
            values,
            fs=session.sampling_rate,
            axis=0,
            nperseg=nperseg,
            detrend="constant",
            scaling="density",
        )
        in_band = (frequencies >= low) & (frequencies <= high)
        if in_band.sum() < 2:
            raise ValueError("frequency resolution leaves fewer than two bins in band")
        powers = np.trapezoid(psd[in_band], frequencies[in_band], axis=0)
        for channel, power in zip(lfp.columns, powers):
            rows.append(
                {
                    "brain_area": int(area),
                    "channel": str(channel),
                    "broadband_power": float(power),
                }
            )

    channels = pd.DataFrame(rows)
    if channels.empty:
        raise ValueError("session contains no LFP channels")
    area_means = (
        channels.groupby("brain_area")["broadband_power"]
        .agg(n_channels="count", mean_power="mean")
    )
    return channels, area_means


def load_all_subject_power() -> pd.DataFrame:
    """Calculate one channel-averaged value per subject and brain area."""
    results: list[pd.DataFrame] = []
    for subject in available_subjects():
        print(f"Calculating subject {subject}...")
        session = load_subject_lfp(subject)
        print_lfp_summary(session)
        _, area_means = broadband_power_by_area(session)
        frame = area_means.reset_index()
        frame.insert(0, "subject", subject)
        results.append(frame)
        del session
    if not results:
        raise ValueError("no subject LFP data were found")
    return pd.concat(results, ignore_index=True)


def summarize_subject_power(subject_power: pd.DataFrame) -> pd.DataFrame:
    """Calculate means and Student-t 95% CIs across subjects."""
    required = {"subject", "brain_area", "mean_power"}
    missing = required.difference(subject_power.columns)
    if missing:
        raise ValueError(f"subject_power is missing columns: {sorted(missing)}")
    summary = (
        subject_power.groupby("brain_area")["mean_power"]
        .agg(n_subjects="count", mean_power="mean", sd_power="std")
    )
    summary["sem_power"] = summary["sd_power"] / np.sqrt(summary["n_subjects"])
    critical_t = summary["n_subjects"].map(
        lambda n: stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    )
    margin = critical_t * summary["sem_power"]
    summary["ci95_low"] = summary["mean_power"] - margin
    summary["ci95_high"] = summary["mean_power"] + margin
    return summary


def compare_brain_areas(
    subject_power: pd.DataFrame,
) -> tuple[int, pd.DataFrame, object]:
    """Compare the highest-mean area with the other areas using paired tests."""
    summary = summarize_subject_power(subject_power)
    highest_area = int(summary["mean_power"].idxmax())
    complete = subject_power.pivot(
        index="subject", columns="brain_area", values="mean_power"
    ).dropna()
    if complete.shape[1] < 3:
        raise ValueError("the Friedman test requires at least three brain areas")
    if complete.shape[0] < 2:
        raise ValueError("at least two complete subjects are required")

    friedman = stats.friedmanchisquare(
        *(complete[area].to_numpy() for area in complete.columns)
    )
    comparisons: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for other_area in complete.columns:
        if other_area == highest_area:
            continue
        result = stats.wilcoxon(
            complete[highest_area],
            complete[other_area],
            alternative="greater",
        )
        raw_p_values.append(float(result.pvalue))
        comparisons.append(
            {
                "highest_area": highest_area,
                "comparison_area": int(other_area),
                "n_subjects": len(complete),
                "wilcoxon_statistic": float(result.statistic),
                "p_value_one_sided": float(result.pvalue),
            }
        )
    for comparison, adjusted in zip(comparisons, holm_adjust(raw_p_values)):
        comparison["p_value_holm"] = adjusted
        comparison["significant_0.05"] = adjusted < 0.05
    return highest_area, pd.DataFrame(comparisons), friedman


def plot_broadband_power(
    summary: pd.DataFrame,
    path=DEFAULT_PLOT_PATH / "lfp_broadband_power_all_subjects.png",
) -> None:
    """Plot across-subject means and 95% confidence intervals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    means = summary["mean_power"].to_numpy()
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
        ylabel="Broadband LFP power (signal units²)",
        title="Mean 1–100 Hz LFP power across subjects",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    subject_power = load_all_subject_power()
    summary = summarize_subject_power(subject_power)
    highest_area, pairwise, friedman = compare_brain_areas(subject_power)

    DEFAULT_PLOT_PATH.mkdir(parents=True, exist_ok=True)
    subject_power.to_csv(
        DEFAULT_PLOT_PATH / "lfp_broadband_power_by_subject.csv", index=False
    )
    summary.to_csv(DEFAULT_PLOT_PATH / "lfp_broadband_power_group_summary.csv")
    pairwise.to_csv(
        DEFAULT_PLOT_PATH / "lfp_broadband_power_pairwise_tests.csv", index=False
    )
    plot_broadband_power(summary)

    print("\nBroadband LFP power by area (subject mean and 95% CI)")
    print(
        summary[["n_subjects", "mean_power", "ci95_low", "ci95_high"]].to_string(
            float_format=lambda value: f"{value:.6g}"
        )
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
