"""Decode behavioral variable C from three complete trial epochs.

The analysis is deliberately participant-specific: neurons from the three
regions are concatenated within a participant, but never across participants.
For each participant, the same stratified 60/20/20 trial split is used to
compare firing rates during:

* trial start to stimulus onset
* stimulus onset to outcome
* outcome to trial end

Variable C is encoded as 0/2 in completed trials.  The few value-1 trials are
roughly 60-second timeout trials and are excluded by default.  An L2 logistic
regression is selected on validation balanced accuracy and evaluated once on
the held-out test set.

Run from the project root, for example::

    python src/Q12.py --repeats 10 --n-jobs -1
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import friedmanchisquare, t, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "con2phys-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .data_loader import DEFAULT_DATA_PATH, DEFAULT_PLOT_PATH, available_subjects, load_session
else:
    from data_loader import DEFAULT_DATA_PATH, DEFAULT_PLOT_PATH, available_subjects, load_session


LABEL_COLUMN = "variable_C"
COMPLETED_LABELS = (0, 2)
EPOCHS = {
    "trial_start_to_stim": ("trial_start", "stim_start"),
    "stim_to_outcome": ("stim_start", "outcome"),
    "outcome_to_trial_end": ("outcome", "trial_end"),
}
DECODING_METRICS = ("accuracy", "balanced_accuracy", "roc_auc")
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_OUTPUT_PATH = DEFAULT_PLOT_PATH.parent / "analysis" / "Q12_decoding"
EPOCH_COLORS = {
    "trial_start_to_stim": "#4C78A8",
    "stim_to_outcome": "#F58518",
    "outcome_to_trial_end": "#54A24B",
}


def stratified_train_val_test_split(
    labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint, exhaustive, class-stratified 60/20/20 indices."""
    y = np.asarray(labels)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("variable_C must contain exactly two completed-trial classes")
    if counts.min() < 5:
        raise ValueError("each variable_C class needs at least five trials")

    parts = [[], [], []]
    for cls in classes:
        indices = np.flatnonzero(y == cls)
        rng.shuffle(indices)
        # Splitting each class into five near-equal folds makes the requested
        # fractions exact whenever its size is divisible by five.
        folds = np.array_split(indices, 5)
        parts[0].extend(np.concatenate(folds[:3]))
        parts[1].extend(folds[3])
        parts[2].extend(folds[4])
    result = tuple(np.asarray(part, dtype=int) for part in parts)
    for part in result:
        rng.shuffle(part)
    if len(np.unique(np.concatenate(result))) != len(y):
        raise RuntimeError("train/validation/test split is not disjoint and exhaustive")
    return result


def epoch_firing_rates(
    spikes, trials: pd.DataFrame, start_column: str, end_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Compute trial x neuron firing rates for one variable-duration epoch."""
    starts = trials[start_column].to_numpy(float)
    ends = trials[end_column].to_numpy(float)
    durations = ends - starts
    valid = np.isfinite(starts) & np.isfinite(ends) & (durations > 0)
    unit_ids = list(spikes.keys())
    rates = np.empty((len(trials), len(unit_ids)), dtype=np.float64)
    rates.fill(np.nan)
    for neuron, unit_id in enumerate(unit_ids):
        times = np.asarray(spikes[unit_id].index, dtype=float)
        counts = (
            np.searchsorted(times, ends[valid], side="left")
            - np.searchsorted(times, starts[valid], side="left")
        )
        rates[valid, neuron] = counts / durations[valid]
    return rates, valid


def _model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=2_000)),
        ]
    )


def _decode_participant(
    participant: int,
    data_path: str | os.PathLike[str],
    repeats: int,
    seed: int,
    c_grid: Sequence[float],
) -> list[dict[str, int | float | str]]:
    session = load_session(participant, data_path=data_path, lfp_banks=(), load_spikes=True)
    if session.spikes is None:
        raise RuntimeError(f"participant {participant}: spikes were not loaded")
    table = session.trial_table
    required = {LABEL_COLUMN, *(column for pair in EPOCHS.values() for column in pair)}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"participant {participant}: missing columns {sorted(missing)}")

    completed = table[LABEL_COLUMN].isin(COMPLETED_LABELS).to_numpy()
    epoch_data = {}
    valid_all = completed.copy()
    for epoch, (start, end) in EPOCHS.items():
        rates, valid = epoch_firing_rates(session.spikes, table, start, end)
        epoch_data[epoch] = rates
        valid_all &= valid
    labels_original = table.loc[valid_all, LABEL_COLUMN].to_numpy()
    # Map 0/2 to the consecutive targets expected by binary metrics.
    labels = (labels_original == COMPLETED_LABELS[1]).astype(int)
    epoch_data = {name: values[valid_all] for name, values in epoch_data.items()}

    records: list[dict[str, int | float | str]] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(np.random.SeedSequence([seed, participant, repeat]))
        train, val, test = stratified_train_val_test_split(labels, rng)
        for epoch, values in epoch_data.items():
            best_c, best_val = None, -np.inf
            for c_value in c_grid:
                candidate = _model(float(c_value)).fit(values[train], labels[train])
                score = balanced_accuracy_score(labels[val], candidate.predict(values[val]))
                if score > best_val:
                    best_c, best_val = float(c_value), float(score)
            fitted = _model(float(best_c)).fit(
                values[np.concatenate([train, val])], labels[np.concatenate([train, val])]
            )
            prediction = fitted.predict(values[test])
            probability = fitted.predict_proba(values[test])[:, 1]
            records.append(
                {
                    "participant": participant,
                    "repeat": repeat,
                    "epoch": epoch,
                    "accuracy": accuracy_score(labels[test], prediction),
                    "balanced_accuracy": balanced_accuracy_score(labels[test], prediction),
                    "roc_auc": roc_auc_score(labels[test], probability),
                    "validation_balanced_accuracy": best_val,
                    "selected_C": best_c,
                    "n_neurons": values.shape[1],
                    "n_trials": len(labels),
                    "n_train": len(train),
                    "n_validation": len(val),
                    "n_test": len(test),
                }
            )
    return records


def decode_variable_c(
    *,
    data_path: str | os.PathLike[str] = DEFAULT_DATA_PATH,
    participants: Sequence[int] | None = None,
    repeats: int = 10,
    seed: int = 12345,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run participant-independent epoch decoding and return test scores."""
    if repeats < 1 or n_jobs == 0:
        raise ValueError("repeats must be positive and n_jobs must not be zero")
    if not c_grid or any(not np.isfinite(c) or c <= 0 for c in c_grid):
        raise ValueError("all c_grid values must be positive and finite")
    selected = list(available_subjects(data_path) if participants is None else participants)
    if not selected:
        raise ValueError("no participants were selected")
    chunks = Parallel(n_jobs=n_jobs)(
        delayed(_decode_participant)(p, data_path, repeats, seed, c_grid) for p in selected
    )
    return pd.DataFrame(row for chunk in chunks for row in chunk)


def summarize_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize participant means and paired epoch comparisons."""
    participant_scores = (
        results.groupby(["participant", "epoch"], as_index=False)["balanced_accuracy"].mean()
    )
    summary = participant_scores.groupby("epoch")["balanced_accuracy"].agg(
        mean="mean", std="std", n_participants="count"
    ).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["n_participants"])
    critical = t.ppf(0.975, summary["n_participants"] - 1)
    margin = critical * summary["sem"]
    summary["ci95_low"] = summary["mean"] - margin
    summary["ci95_high"] = summary["mean"] + margin
    summary = summary.sort_values("mean", ascending=False, ignore_index=True)

    wide = participant_scores.pivot(index="participant", columns="epoch", values="balanced_accuracy")
    comparisons = []
    names = list(EPOCHS)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            paired = wide[[first, second]].dropna()
            difference = paired[first] - paired[second]
            if len(paired) == 0:
                statistic = pvalue = np.nan
            elif np.allclose(difference, 0):
                statistic, pvalue = 0.0, 1.0
            else:
                statistic, pvalue = wilcoxon(paired[first], paired[second])
            difference_sem = difference.sem()
            difference_margin = (
                t.ppf(0.975, len(difference) - 1) * difference_sem
                if len(difference) > 1 else np.nan
            )
            comparisons.append({"epoch_1": first, "epoch_2": second,
                                "mean_difference": difference.mean(),
                                "difference_ci95_low": difference.mean() - difference_margin,
                                "difference_ci95_high": difference.mean() + difference_margin,
                                "wilcoxon_statistic": statistic,
                                "p_value": pvalue, "n_participants": len(paired)})
    comparisons = pd.DataFrame(comparisons)
    comparisons["p_value_holm"] = _holm_correction(comparisons["p_value"].to_numpy())
    comparisons["significant_holm_0.05"] = comparisons["p_value_holm"] < 0.05
    return summary, comparisons


def _holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Return Holm family-wise-error-corrected p-values."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return adjusted
    order = finite[np.argsort(values[finite])]
    scaled = np.array([(len(order) - rank) * values[index]
                       for rank, index in enumerate(order)])
    scaled = np.maximum.accumulate(scaled)
    adjusted[order] = np.minimum(scaled, 1.0)
    return adjusted


def epoch_statistical_tests(
    results: pd.DataFrame, metric: str = "balanced_accuracy"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test the three epochs using participants as paired observations.

    Returns a Friedman omnibus test and Holm-corrected pairwise Wilcoxon
    signed-rank tests. Repeated splits are averaged within participant first.
    """
    required = {"participant", "epoch", metric}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results are missing columns: {sorted(missing)}")
    wide = (
        results.groupby(["participant", "epoch"])[metric]
        .mean()
        .unstack()
        .reindex(columns=EPOCHS)
        .dropna()
    )
    if len(wide) < 2:
        raise ValueError("at least two participants with all three epochs are required")
    statistic, p_value = friedmanchisquare(*(wide[epoch] for epoch in EPOCHS))
    omnibus = pd.DataFrame(
        [{"test": "Friedman", "metric": metric, "statistic": statistic,
          "p_value": p_value, "n_participants": len(wide)}]
    )
    pairwise_input = results[["participant", "epoch", metric]].rename(
        columns={metric: "balanced_accuracy"}
    )
    _, pairwise = summarize_results(pairwise_input)
    pairwise.insert(0, "metric", metric)
    return omnibus, pairwise


def plot_results(results: pd.DataFrame, output_path: str | os.PathLike[str]) -> Path:
    """Plot participant-level held-out balanced accuracy for each epoch."""
    scores = results.groupby(["participant", "epoch"])["balanced_accuracy"].mean().unstack()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(EPOCHS))
    for _, row in scores.iterrows():
        ax.plot(x, row[list(EPOCHS)], color="0.78", linewidth=0.8, alpha=0.7)
    means = scores[list(EPOCHS)].mean()
    sems = scores[list(EPOCHS)].sem()
    ax.errorbar(x, means, yerr=sems, fmt="o", color="black", capsize=4, linewidth=2)
    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x, [name.replace("_", "\n") for name in EPOCHS])
    ax.set_ylabel("Held-out balanced accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Decoding variable C from complete trial epochs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_mean_decoding_metrics(
    results: pd.DataFrame, output_path: str | os.PathLike[str]
) -> Path:
    """Plot mean held-out accuracy, balanced accuracy, and ROC AUC.

    Repeats are first averaged within each participant, so participants—not
    repeated random splits—are the independent observations. Error bars show
    the standard error across participants.
    """
    required = {"participant", "epoch", *DECODING_METRICS}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results are missing columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("results must not be empty")

    participant_means = results.groupby(["participant", "epoch"])[
        list(DECODING_METRICS)
    ].mean()
    means = participant_means.groupby("epoch").mean().reindex(EPOCHS)
    grouped = participant_means.groupby("epoch")
    sems = grouped.sem().reindex(EPOCHS).fillna(0.0)
    counts = grouped.count().reindex(EPOCHS)
    ci95 = sems * t.ppf(0.975, counts - 1)
    ci95 = ci95.fillna(0.0)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(EPOCHS), dtype=float)
    width = 0.24
    labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "roc_auc": "ROC AUC",
    }
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for metric_index, (metric, color) in enumerate(zip(DECODING_METRICS, colors)):
        offset = (metric_index - 1) * width
        ax.bar(
            x + offset,
            means[metric],
            width,
            yerr=ci95[metric],
            capsize=3,
            label=labels[metric],
            color=color,
            alpha=0.9,
        )
    ax.axhline(0.5, color="0.35", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x, [name.replace("_", "\n") for name in EPOCHS])
    ax.set_ylabel("Mean held-out decoding score")
    ax.set_ylim(0, 1)
    ax.set_title("Variable C decoding by trial epoch (mean and 95% CI)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def summarize_decoding_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Return participant-level means and t-based 95% CIs for every metric."""
    required = {"participant", "epoch", *DECODING_METRICS}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results are missing columns: {sorted(missing)}")
    participant_means = (
        results.groupby(["participant", "epoch"])[list(DECODING_METRICS)]
        .mean()
        .reset_index()
        .melt(id_vars=["participant", "epoch"], var_name="metric", value_name="score")
    )
    summary = participant_means.groupby(["epoch", "metric"])["score"].agg(
        mean="mean", std="std", n_participants="count"
    ).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["n_participants"])
    margin = t.ppf(0.975, summary["n_participants"] - 1) * summary["sem"]
    summary["ci95_low"] = summary["mean"] - margin
    summary["ci95_high"] = summary["mean"] + margin
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--participants", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = decode_variable_c(data_path=args.data_path, participants=args.participants,
                                repeats=args.repeats, seed=args.seed, n_jobs=args.n_jobs)
    summary, comparisons = summarize_results(results)
    results.to_csv(args.output / "test_scores.csv", index=False)
    summary.to_csv(args.output / "epoch_summary.csv", index=False)
    summarize_decoding_metrics(results).to_csv(
        args.output / "metric_summary.csv", index=False
    )
    omnibus, corrected_comparisons = epoch_statistical_tests(results)
    omnibus.to_csv(args.output / "omnibus_test.csv", index=False)
    corrected_comparisons.to_csv(args.output / "paired_comparisons.csv", index=False)
    plot_results(results, args.output / "epoch_decoding.png")
    plot_mean_decoding_metrics(results, args.output / "mean_decoding_metrics.png")
    print(summary.to_string(index=False))
    print("\nStatistical test:")
    print(omnibus.to_string(index=False))
    print(corrected_comparisons.to_string(index=False))
    print(f"\nBest epoch: {summary.iloc[0]['epoch']}")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
