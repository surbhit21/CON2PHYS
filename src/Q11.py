"""Time-resolved decoding of behavioral variable A from regional spikes.

Participant data are pooled at the result level: identical decoding is run
within every participant and brain region, then fold accuracies are combined
across participants.  This avoids treating neurons recorded from different
participants as if they were simultaneous observations.

Run a reasonably quick analysis from the repository root with::

    python src/Q11.py --repeats 5 --label-shuffles 20
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap

# Spawned workers should share a writable Matplotlib cache rather than each
# constructing a private fallback cache under a read-only home directory.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "con2phys-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .data_loader import (
        CON2PHYSSession,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )
else:
    from data_loader import (
        CON2PHYSSession,
        DEFAULT_PLOT_PATH,
        available_subjects,
        load_session,
    )


BRAIN_REGIONS = (1, 2, 3)
EVENTS = ("stim_start", "outcome")
LABEL_COLUMN = "variable_A"
WINDOW_SIZE_S = 0.100
STEP_SIZE_S = 0.020
WINDOW_START_S = -0.200
WINDOW_STOP_S = 1.000
N_SPLITS = 5
DEFAULT_OUTPUT_PATH = DEFAULT_PLOT_PATH.parent / "analysis" / "Q11_decoding"
REGION_COLORS = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B"}


@dataclass(frozen=True)
class RegionalSpikes:
    """One participant's session and three separately stored regional groups."""

    session: CON2PHYSSession
    by_region: dict[int, nap.TsGroup]


def load_all_spikes_by_region() -> dict[int, RegionalSpikes]:
    """Load spikes for every participant, without loading any LFP arrays."""
    participants: dict[int, RegionalSpikes] = {}
    for participant in available_subjects():
        session = load_session(participant, lfp_banks=(), load_spikes=True)
        if session.spikes is None:
            raise RuntimeError(f"participant {participant} spikes were not loaded")
        grouped = session.spikes.getby_category("brain_area")
        missing = [region for region in BRAIN_REGIONS if region not in grouped]
        if missing:
            raise ValueError(
                f"participant {participant} lacks brain region(s) {missing}"
            )
        participants[participant] = RegionalSpikes(
            session=session,
            by_region={region: grouped[region] for region in BRAIN_REGIONS},
        )
    return participants


def window_starts() -> np.ndarray:
    """Return starts of bins fully contained between -200 and +1000 ms."""
    count = int(
        round((WINDOW_STOP_S - WINDOW_START_S - WINDOW_SIZE_S) / STEP_SIZE_S)
    ) + 1
    return WINDOW_START_S + np.arange(count) * STEP_SIZE_S


def valid_event_trials(trials: pd.DataFrame, event: str) -> np.ndarray:
    """Select trials for which every requested window stays inside the trial."""
    required = {"trial_start", "trial_end", event, LABEL_COLUMN}
    missing = required.difference(trials.columns)
    if missing:
        raise ValueError(f"trial table is missing columns: {sorted(missing)}")
    first_start = trials[event].to_numpy(float) + WINDOW_START_S
    # WINDOW_STOP_S denotes the requested right edge of the entire analysis
    # interval, not the start of an additional bin extending to +1100 ms.
    last_end = trials[event].to_numpy(float) + WINDOW_STOP_S
    labels = trials[LABEL_COLUMN].to_numpy()
    return (
        np.isfinite(first_start)
        & np.isfinite(last_end)
        & pd.notna(labels)
        & (first_start >= trials["trial_start"].to_numpy(float))
        & (last_end <= trials["trial_end"].to_numpy(float))
    )


def event_spike_counts(
    spikes: nap.TsGroup,
    event_times: np.ndarray,
    *,
    starts: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Return counts shaped ``trials x windows x neurons``."""
    starts = window_starts() if starts is None else np.asarray(starts, float)
    events = np.asarray(event_times, float)
    unit_ids = [int(unit_id) for unit_id in spikes.keys()]
    counts = np.empty((len(events), len(starts), len(unit_ids)), dtype=np.int32)
    left = events[:, None] + starts[None, :]
    right = left + WINDOW_SIZE_S
    for neuron, unit_id in enumerate(unit_ids):
        times = np.asarray(spikes[unit_id].index, dtype=float)
        counts[:, :, neuron] = (
            np.searchsorted(times, right, side="left")
            - np.searchsorted(times, left, side="left")
        )
    return counts, unit_ids


def stratified_folds(
    labels: np.ndarray, n_splits: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create shuffled stratified folds with approximately 80/20 splits."""
    y = np.asarray(labels, dtype=int)
    classes, class_counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("variable_A must contain exactly two classes")
    if class_counts.min() < n_splits:
        raise ValueError(f"each class needs at least {n_splits} trials")
    class_folds: dict[int, list[np.ndarray]] = {}
    for cls in classes:
        indices = np.flatnonzero(y == cls)
        rng.shuffle(indices)
        class_folds[int(cls)] = list(np.array_split(indices, n_splits))
    all_indices = np.arange(len(y))
    result = []
    for fold in range(n_splits):
        test = np.concatenate([class_folds[int(cls)][fold] for cls in classes])
        rng.shuffle(test)
        train = np.setdiff1d(all_indices, test, assume_unique=False)
        if np.intersect1d(train, test).size:
            raise RuntimeError("training and test indices overlap")
        if len(train) + len(test) != len(y):
            raise RuntimeError("training/test split does not cover every trial")
        result.append((train, test))
    return result


def l2_logistic_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    regularization: float = 1.0,
) -> dict[str, float]:
    """Fit on training data and score untouched test data three ways.

    A scikit-learn pipeline learns one mean and standard deviation per neuron
    exclusively from the training fold, then fits an L2 logistic regression.
    The test matrix and labels are used only by ``Pipeline.score`` after fit.
    """
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be positive")
    model = Pipeline(
        steps=[
            ("standardize_neurons", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    penalty="l2",
                    C=1.0 / regularization,
                    solver="lbfgs",
                    max_iter=1_000,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
    }


def l2_logistic_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    regularization: float = 1.0,
) -> float:
    """Backward-compatible raw-accuracy wrapper around the full scorer."""
    return l2_logistic_scores(
        x_train, y_train, x_test, y_test, regularization=regularization
    )["accuracy"]


def _decode_participant(
    participant: int,
    *,
    repeats: int,
    label_shuffles: int,
    seed: int,
    regularization: float,
) -> list[dict[str, int | float | str | bool]]:
    """Decode one participant; designed as an isolated parallel worker."""
    rng = np.random.default_rng(seed)
    starts = window_starts()
    records: list[dict[str, int | float | str | bool]] = []
    session = load_session(participant, lfp_banks=(), load_spikes=True)
    if session.spikes is None:
        raise RuntimeError(f"participant {participant} spikes were not loaded")
    grouped = session.spikes.getby_category("brain_area")
    missing = [region for region in BRAIN_REGIONS if region not in grouped]
    if missing:
        raise ValueError(f"participant {participant} lacks brain region(s) {missing}")
    by_region = {region: grouped[region] for region in BRAIN_REGIONS}
    n_matched = min(len(group) for group in by_region.values())

    for event in EVENTS:
        valid = valid_event_trials(session.trial_table, event)
        trials = session.trial_table.loc[valid]
        labels = trials[LABEL_COLUMN].to_numpy(dtype=int)
        matrices = {
            region: event_spike_counts(
                spikes, trials[event].to_numpy(float), starts=starts
            )[0]
            for region, spikes in by_region.items()
        }
        for repeat in range(repeats):
            neuron_samples = {
                region: rng.choice(matrix.shape[2], n_matched, replace=False)
                for region, matrix in matrices.items()
            }
            permutations = [labels]
            permutations.extend(rng.permutation(labels) for _ in range(label_shuffles))
            for shuffle, permuted_labels in enumerate(permutations):
                is_null = shuffle > 0
                folds = stratified_folds(permuted_labels, N_SPLITS, rng)
                for fold, (train, test) in enumerate(folds):
                    for region, matrix in matrices.items():
                        selected = matrix[:, :, neuron_samples[region]]
                        for time_index, start in enumerate(starts):
                            scores = l2_logistic_scores(
                                selected[train, time_index],
                                permuted_labels[train],
                                selected[test, time_index],
                                permuted_labels[test],
                                regularization=regularization,
                            )
                            records.append(
                                {
                                    "participant": participant,
                                    "event": event,
                                    "region": region,
                                    "window_start_s": start,
                                    "window_center_s": start + WINDOW_SIZE_S / 2,
                                    "repeat": repeat,
                                    "fold": fold,
                                    "shuffle": shuffle,
                                    "is_null": is_null,
                                    **scores,
                                    "n_neurons": n_matched,
                                    "n_trials": len(labels),
                                }
                            )
    return records


def decode_all_participants(
    *,
    repeats: int = 5,
    label_shuffles: int = 20,
    seed: int = 12345,
    regularization: float = 1.0,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Decode A over time for each participant/event/region.

    The smallest region determines the neuron count separately for each
    participant. Larger regions are independently subsampled on every repeat.
    Null decoding repeats the complete fit after permuting labels.

    ``n_jobs`` follows joblib convention: 1 is serial, and -1 uses all CPUs.
    Every worker loads and analyzes one participant independently.
    """
    if repeats < 1 or label_shuffles < 0:
        raise ValueError("repeats must be positive and label_shuffles non-negative")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero")
    participants = available_subjects()
    participant_seeds = np.random.SeedSequence(seed).spawn(len(participants))
    chunks = Parallel(n_jobs=n_jobs, prefer="processes", verbose=5)(
        delayed(_decode_participant)(
            participant,
            repeats=repeats,
            label_shuffles=label_shuffles,
            seed=int(participant_seed.generate_state(1)[0]),
            regularization=regularization,
        )
        for participant, participant_seed in zip(participants, participant_seeds)
    )
    records = [record for chunk in chunks for record in chunk]
    return pd.DataFrame.from_records(records)


def accuracy_statistics(
    results: pd.DataFrame, *, metric: str = "balanced_accuracy"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test a held-out metric against null and between regions over time."""
    if metric not in results:
        raise ValueError(f"results do not contain metric {metric!r}")
    real = results.loc[~results["is_null"]]
    null = results.loc[results["is_null"]]
    keys = ["participant", "event", "region", "window_start_s"]
    observed = real.groupby(keys, as_index=False)[metric].mean()
    null_mean = null.groupby(keys, as_index=False)[metric].mean()
    above_null = observed.merge(null_mean, on=keys, suffixes=("_real", "_null"))
    above_null["difference"] = above_null[f"{metric}_real"] - above_null[f"{metric}_null"]

    null_tests = []
    for (event, region, start), group in above_null.groupby(
        ["event", "region", "window_start_s"]
    ):
        statistic, pvalue = wilcoxon(group["difference"], alternative="greater")
        null_tests.append(
            {"metric": metric, "event": event, "region": region,
             "window_start_s": start,
             "mean_real_score": group[f"{metric}_real"].mean(),
             "mean_null_score": group[f"{metric}_null"].mean(),
             "wilcoxon_statistic": statistic, "pvalue": pvalue,
             "n_participants": len(group)}
        )

    paired_tests = []
    wide = observed.pivot(
        index=["participant", "event", "window_start_s"],
        columns="region", values=metric
    )
    for event in EVENTS:
        event_wide = wide.xs(event, level="event")
        for start, at_time in event_wide.groupby(level="window_start_s"):
            for first, second in combinations(BRAIN_REGIONS, 2):
                difference = at_time[first] - at_time[second]
                statistic, pvalue = wilcoxon(difference, alternative="two-sided")
                paired_tests.append(
                    {"metric": metric, "event": event, "window_start_s": start,
                     "region_1": first, "region_2": second,
                     "mean_score_difference": difference.mean(),
                     "wilcoxon_statistic": statistic, "pvalue": pvalue,
                     "n_participants": difference.notna().sum()}
                )
    return pd.DataFrame(null_tests), pd.DataFrame(paired_tests)


def save_results(results: pd.DataFrame, output: Path) -> None:
    """Save fold distributions, tests, and plots for all three metrics."""
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "fold_accuracies.csv", index=False)
    for metric in ("balanced_accuracy", "roc_auc", "accuracy"):
        null_tests, region_tests = accuracy_statistics(results, metric=metric)
        null_tests.to_csv(output / f"real_vs_shuffled_tests_{metric}.csv", index=False)
        region_tests.to_csv(output / f"paired_region_tests_{metric}.csv", index=False)
        plot_decoding_accuracy(
            results, output / f"decoding_{metric}_over_time.png", metric=metric
        )


def _accuracy_timecourse(
    results: pd.DataFrame, *, is_null: bool, metric: str
) -> pd.DataFrame:
    """Give each participant equal weight in a regional accuracy time course."""
    selected = results.loc[results["is_null"] == is_null]
    participant_means = selected.groupby(
        ["participant", "event", "region", "window_center_s"], as_index=False
    )[metric].mean()
    summary = participant_means.groupby(
        ["event", "region", "window_center_s"]
    )[metric].agg(["mean", "sem", "count"]).reset_index()
    summary["sem"] = summary["sem"].fillna(0.0)
    return summary


def plot_decoding_accuracy(
    results: pd.DataFrame,
    path: str | Path,
    *,
    metric: str = "accuracy",
) -> Path:
    """Plot a real and label-shuffled decoding metric over time."""
    path = Path(path)
    real = _accuracy_timecourse(results, is_null=False, metric=metric)
    null = _accuracy_timecourse(results, is_null=True, metric=metric)
    if real.empty:
        raise ValueError("results contain no real decoding accuracies")

    figure, axes = plt.subplots(1, len(EVENTS), figsize=(13, 4.8), sharey=True)
    for axis, event in zip(np.atleast_1d(axes), EVENTS):
        for region in BRAIN_REGIONS:
            color = REGION_COLORS[region]
            curve = real.loc[(real["event"] == event) & (real["region"] == region)]
            time_ms = curve["window_center_s"].to_numpy(float) * 1_000
            mean = curve["mean"].to_numpy(float)
            sem = curve["sem"].to_numpy(float)
            axis.plot(time_ms, mean, color=color, linewidth=2, label=f"Region {region}")
            axis.fill_between(time_ms, mean - sem, mean + sem, color=color, alpha=0.18)

            shuffled = null.loc[
                (null["event"] == event) & (null["region"] == region)
            ]
            if not shuffled.empty:
                axis.plot(
                    shuffled["window_center_s"].to_numpy(float) * 1_000,
                    shuffled["mean"].to_numpy(float),
                    color=color,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                )
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.axhline(0.5, color="0.35", linestyle="--", linewidth=1)
        axis.set_title("Stimulus aligned" if event == "stim_start" else "Outcome aligned")
        axis.set_xlabel("Time from event (ms; window center)")
        axis.grid(alpha=0.2)
    ylabel = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
              "roc_auc": "ROC AUC"}.get(metric, metric)
    axes[0].set_ylabel(f"Held-out {ylabel}")
    axes[-1].legend(frameon=False, title="Solid: real\nDotted: shuffled")
    figure.suptitle(
        f"Variable A decoding by brain region: {ylabel} "
        "(mean ± SEM across participants)"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--label-shuffles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="parallel participant workers; use -1 for all available CPUs",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    results = decode_all_participants(
        repeats=args.repeats,
        label_shuffles=args.label_shuffles,
        seed=args.seed,
        regularization=args.regularization,
        n_jobs=args.n_jobs,
    )
    save_results(results, args.output)
    print(f"Saved {len(results):,} fold accuracies and tests to {args.output}")


if __name__ == "__main__":
    main()
