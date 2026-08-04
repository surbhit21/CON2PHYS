"""Calculate waveform-defined excitatory-to-inhibitory firing-rate ratios.

Putative inhibitory neurons are the fast-spiking units classified by Q7 from
waveform trough-to-peak duration. Per the analysis definition, every other
unit is labelled putative excitatory. Firing rates are calculated in complete
10-ms bins within trials and averaged across neurons in each class.

Run participant 1 from the repository root with::

    python src/Q9.py --participant 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if __package__:
    from .data_loader import (
        CON2PHYSSession,
        DEFAULT_DATA_PATH,
        available_subjects,
        load_session,
    )
    from .Q7 import (
        FAST_SPIKING_THRESHOLD_MS,
        WAVEFORM_SAMPLING_RATE,
        waveform_features,
    )
    from .stat_test import holm_adjust
else:
    from data_loader import (
        CON2PHYSSession,
        DEFAULT_DATA_PATH,
        available_subjects,
        load_session,
    )
    from Q7 import (
        FAST_SPIKING_THRESHOLD_MS,
        WAVEFORM_SAMPLING_RATE,
        waveform_features,
    )
    from stat_test import holm_adjust


PARTICIPANT = 1
BIN_SIZE_S = 0.010
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "analysis" / "Q9"


def load_participant_spikes(
    participant: int = PARTICIPANT,
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> CON2PHYSSession:
    """Load spike times, waveform data, trial intervals, and unit metadata."""
    return load_session(subject=participant, data_path=data_path, lfp_banks=())


def classify_and_calculate_unit_rates(
    session: CON2PHYSSession,
    *,
    bin_size_s: float = BIN_SIZE_S,
    waveform_sampling_rate: float = WAVEFORM_SAMPLING_RATE,
    fast_spiking_threshold_ms: float = FAST_SPIKING_THRESHOLD_MS,
    baseline_samples: int = 10,
) -> pd.DataFrame:
    """Classify units and calculate their trial-period mean binned rates.

    Only complete bins are used separately within each trial. Thus no bin
    crosses a trial boundary and inter-trial periods do not affect the rates.
    The mean of each neuron's 10-ms bin rates equals its spike count divided by
    the total included bin occupancy.
    """
    if session.spikes is None:
        raise ValueError("session must contain spike data")
    if not np.isfinite(bin_size_s) or bin_size_s <= 0:
        raise ValueError("bin_size_s must be positive")
    features = waveform_features(
        session,
        sampling_rate=waveform_sampling_rate,
        threshold_ms=fast_spiking_threshold_ms,
        baseline_samples=baseline_samples,
    )
    intervals = session.trial_table[["trial_start", "trial_end"]].to_numpy(float)
    valid_trials = (
        np.isfinite(intervals).all(axis=1) & (intervals[:, 1] > intervals[:, 0])
    )
    intervals = intervals[valid_trials]
    if not len(intervals):
        raise ValueError("participant has no valid finite trial intervals")
    n_bins_by_trial = np.floor(
        (intervals[:, 1] - intervals[:, 0]) / bin_size_s + 1e-9
    ).astype(int)
    total_bins = int(n_bins_by_trial.sum())
    if total_bins < 1:
        raise ValueError("trials do not contain any complete firing-rate bins")
    duration_s = total_bins * bin_size_s

    records: list[dict[str, object]] = []
    for unit_id, feature in features.iterrows():
        spike_times = np.asarray(session.spikes[int(unit_id)].t, dtype=float)
        spike_count = 0
        for (start, _), n_bins in zip(intervals, n_bins_by_trial):
            if n_bins == 0:
                continue
            # Histogramming explicitly applies the requested 10-ms bins.
            edges = start + np.arange(n_bins + 1) * bin_size_s
            spike_count += int(np.histogram(spike_times, bins=edges)[0].sum())
        inhibitory = bool(feature["putative_fast_spiking"])
        records.append(
            {
                "cluster_id": int(unit_id),
                "brain_area": feature["brain_area"],
                "trough_to_peak_ms": feature["trough_to_peak_ms"],
                "putative_fast_spiking": inhibitory,
                "cell_class": (
                    "putative_inhibitory" if inhibitory else "putative_excitatory"
                ),
                "n_bins": total_bins,
                "n_valid_trials": int(valid_trials.sum()),
                "n_excluded_trials": int((~valid_trials).sum()),
                "bin_size_s": bin_size_s,
                "duration_s": duration_s,
                "n_spikes": spike_count,
                "mean_firing_rate_hz": spike_count / duration_s,
            }
        )
    return pd.DataFrame.from_records(records).set_index("cluster_id")


def excitation_inhibition_ratio(unit_rates: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean excitatory/inhibitory firing-rate ratios by area and overall."""
    required = {"brain_area", "cell_class", "mean_firing_rate_hz"}
    missing = required.difference(unit_rates.columns)
    if missing:
        raise ValueError(f"unit_rates is missing columns: {sorted(missing)}")

    records: list[dict[str, object]] = []
    groups: list[tuple[object, pd.DataFrame]] = list(
        unit_rates.groupby("brain_area", sort=True)
    )
    groups.append(("all", unit_rates))
    for area, group in groups:
        excitatory = group.loc[
            group["cell_class"] == "putative_excitatory", "mean_firing_rate_hz"
        ]
        inhibitory = group.loc[
            group["cell_class"] == "putative_inhibitory", "mean_firing_rate_hz"
        ]
        excitatory_mean = float(excitatory.mean()) if len(excitatory) else np.nan
        inhibitory_mean = float(inhibitory.mean()) if len(inhibitory) else np.nan
        if np.isfinite(inhibitory_mean) and inhibitory_mean > 0:
            ratio = excitatory_mean / inhibitory_mean
        else:
            ratio = np.nan
        records.append(
            {
                "brain_area": area,
                "n_putative_excitatory": len(excitatory),
                "n_putative_inhibitory": len(inhibitory),
                "mean_excitatory_firing_rate_hz": excitatory_mean,
                "mean_inhibitory_firing_rate_hz": inhibitory_mean,
                "excitation_to_inhibition_ratio": ratio,
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_ratios_across_participants(
    participant_ratios: pd.DataFrame,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Calculate mean E/I ratio and Student-t CI across participants by area."""
    required = {"participant", "brain_area", "excitation_to_inhibition_ratio"}
    missing = required.difference(participant_ratios.columns)
    if missing:
        raise ValueError(f"participant_ratios is missing columns: {sorted(missing)}")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    area_values = participant_ratios.loc[
        participant_ratios["brain_area"].astype(str) != "all"
    ].copy()
    grouped = area_values.groupby("brain_area")["excitation_to_inhibition_ratio"]
    summary = grouped.agg(n_participants="count", mean_ratio="mean", sd="std")
    summary["sem"] = summary["sd"] / np.sqrt(summary["n_participants"])
    critical = summary["n_participants"].map(
        lambda n: stats.t.ppf(0.5 + confidence / 2, n - 1) if n > 1 else np.nan
    )
    margin = critical * summary["sem"]
    summary["confidence"] = confidence
    summary["ci_low"] = summary["mean_ratio"] - margin
    summary["ci_high"] = summary["mean_ratio"] + margin
    return summary


def compare_brain_area_ratios(
    participant_ratios: pd.DataFrame,
) -> tuple[object, pd.DataFrame, object]:
    """Test whether the highest-mean brain area has larger E/I ratios.

    A Friedman test assesses any repeated-measures area difference. The area
    with the largest across-participant mean is then compared with every other
    area using one-sided paired Wilcoxon signed-rank tests and Holm correction.
    """
    summary = summarize_ratios_across_participants(participant_ratios)
    highest_area = summary["mean_ratio"].idxmax()
    area_values = participant_ratios.loc[
        participant_ratios["brain_area"].astype(str) != "all"
    ]
    complete = area_values.pivot(
        index="participant",
        columns="brain_area",
        values="excitation_to_inhibition_ratio",
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
        test = stats.wilcoxon(
            complete[highest_area], complete[other_area], alternative="greater"
        )
        raw_p_values.append(float(test.pvalue))
        comparisons.append(
            {
                "highest_area": highest_area,
                "comparison_area": other_area,
                "n_complete_participants": len(complete),
                "wilcoxon_statistic": float(test.statistic),
                "p_value_one_sided": float(test.pvalue),
            }
        )
    for comparison, adjusted in zip(comparisons, holm_adjust(raw_p_values)):
        comparison["p_value_holm"] = adjusted
        comparison["significant_0.05"] = adjusted < 0.05
    return highest_area, pd.DataFrame(comparisons), friedman


def analyze_participant(
    participant: int = PARTICIPANT,
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_PATH,
    bin_size_s: float = BIN_SIZE_S,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Q9 for one participant and save unit-level and ratio tables."""
    session = load_participant_spikes(participant, data_path=data_path)
    unit_rates = classify_and_calculate_unit_rates(session, bin_size_s=bin_size_s)
    ratios = excitation_inhibition_ratio(unit_rates)
    unit_rates = unit_rates.reset_index()
    unit_rates.insert(0, "participant", participant)
    ratios.insert(0, "participant", participant)
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    unit_rates.to_csv(path / f"participant_{participant}_unit_firing_rates.csv", index=False)
    ratios.to_csv(path / f"participant_{participant}_excitation_inhibition_ratio.csv", index=False)
    return unit_rates, ratios


def analyze_all_participants(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_PATH,
    bin_size_s: float = BIN_SIZE_S,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Q9 for every available participant and save combined tables."""
    all_rates: list[pd.DataFrame] = []
    all_ratios: list[pd.DataFrame] = []
    for participant in available_subjects(data_path):
        rates, ratios = analyze_participant(
            participant,
            data_path=data_path,
            output_dir=output_dir,
            bin_size_s=bin_size_s,
        )
        all_rates.append(rates)
        all_ratios.append(ratios)
        print(f"Participant {participant}: analyzed {len(rates)} neurons", flush=True)
    if not all_rates:
        raise ValueError("no participants were found")
    rates = pd.concat(all_rates, ignore_index=True)
    ratios = pd.concat(all_ratios, ignore_index=True)
    path = Path(output_dir)
    rates.to_csv(path / "all_participants_unit_firing_rates.csv", index=False)
    ratios.to_csv(path / "all_participants_excitation_inhibition_ratio.csv", index=False)
    summary = summarize_ratios_across_participants(ratios)
    highest_area, pairwise, friedman = compare_brain_area_ratios(ratios)
    summary.to_csv(path / "excitation_inhibition_ratio_group_summary.csv")
    pairwise.to_csv(path / "excitation_inhibition_ratio_pairwise_tests.csv", index=False)
    pd.DataFrame(
        [
            {
                "n_complete_participants": int(
                    ratios.loc[ratios["brain_area"].astype(str) != "all"].pivot(
                        index="participant", columns="brain_area",
                        values="excitation_to_inhibition_ratio",
                    ).dropna().shape[0]
                ),
                "friedman_statistic": float(friedman.statistic),
                "p_value": float(friedman.pvalue),
                "highest_mean_area": highest_area,
            }
        ]
    ).to_csv(path / "excitation_inhibition_ratio_friedman_test.csv", index=False)
    return rates, ratios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", type=int, default=PARTICIPANT)
    parser.add_argument("--all-participants", action="store_true")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--bin-size-ms", type=float, default=BIN_SIZE_S * 1_000)
    args = parser.parse_args()
    bin_size_s = args.bin_size_ms / 1_000
    if args.all_participants:
        _, ratios = analyze_all_participants(
            data_path=args.data_path,
            output_dir=args.output_dir,
            bin_size_s=bin_size_s,
        )
        summary = summarize_ratios_across_participants(ratios)
        highest_area, pairwise, friedman = compare_brain_area_ratios(ratios)
        print("\nMean E/I ratio and 95% CI across participants")
        print(summary.to_string(float_format=lambda value: f"{value:.6g}"))
        print(
            f"\nFriedman test: chi-square={friedman.statistic:.6g}, "
            f"p={friedman.pvalue:.6g}"
        )
        print(f"Highest mean E/I ratio: brain area {highest_area}")
        print("\nOne-sided paired Wilcoxon tests with Holm correction")
        print(pairwise.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    else:
        _, ratios = analyze_participant(
            args.participant,
            data_path=args.data_path,
            output_dir=args.output_dir,
            bin_size_s=bin_size_s,
        )
        print("\nExcitatory-to-inhibitory firing-rate ratios")
        print(ratios.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nResults: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
