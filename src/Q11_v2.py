"""Multi-session, multi-region reduced-rank decoding of variable A.

This is a lightweight NumPy implementation of the multi-region reduced-rank
regression (RRR) model introduced by Zhang et al. (Neuron, 2026).  It learns
participant/region-specific neural bases, region-specific temporal factors,
and a global temporal basis shared across all recordings.

Example
-------
Run five-fold decoding for both events::

    python src/Q11_v2.py --epochs 200 --rank 2 --global-rank 2

Add label-permutation null models (computationally expensive)::

    python src/Q11_v2.py --label-shuffles 20
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
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import t, wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "con2phys-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

if __package__:
    from .Q11 import (
        BRAIN_REGIONS,
        EVENTS,
        LABEL_COLUMN,
        event_spike_counts,
        stratified_folds,
        valid_event_trials,
        window_starts,
    )
    from .data_loader import DEFAULT_PLOT_PATH, available_subjects, load_session
else:
    from Q11 import (
        BRAIN_REGIONS,
        EVENTS,
        LABEL_COLUMN,
        event_spike_counts,
        stratified_folds,
        valid_event_trials,
        window_starts,
    )
    from data_loader import DEFAULT_PLOT_PATH, available_subjects, load_session


N_SPLITS = 5
DEFAULT_OUTPUT_PATH = DEFAULT_PLOT_PATH.parent / "analysis" / "Q11_RRR"
REGION_COLORS = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B"}


@dataclass(frozen=True)
class Recording:
    """Trial tensor and labels for one participant-region recording."""

    participant: int
    region: int
    counts: np.ndarray  # trials x time x neurons
    labels: np.ndarray


@dataclass(frozen=True)
class SplitRecording:
    """A standardized recording with a strictly held-out test set."""

    participant: int
    region: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_event_recordings(event: str) -> list[Recording]:
    """Load every participant and preserve each brain region separately."""
    if event not in EVENTS:
        raise ValueError(f"event must be one of {EVENTS}")
    starts = window_starts()
    recordings: list[Recording] = []
    for participant in available_subjects():
        session = load_session(participant, lfp_banks=(), load_spikes=True)
        if session.spikes is None:
            raise RuntimeError(f"participant {participant} spikes were not loaded")
        valid = valid_event_trials(session.trial_table, event)
        trials = session.trial_table.loc[valid]
        labels = trials[LABEL_COLUMN].to_numpy(dtype=np.int64)
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) != 2 or counts.min() < N_SPLITS:
            raise ValueError(
                f"participant {participant}, {event}: insufficient binary classes"
            )
        grouped = session.spikes.getby_category("brain_area")
        for region in BRAIN_REGIONS:
            if region not in grouped:
                raise ValueError(f"participant {participant} lacks region {region}")
            tensor, _ = event_spike_counts(
                grouped[region], trials[event].to_numpy(float), starts=starts
            )
            recordings.append(
                Recording(participant, region, tensor.astype(np.float32), labels)
            )
    return recordings


def make_fold_splits(
    recordings: list[Recording],
    *,
    fold: int,
    seed: int,
    shuffled: bool = False,
) -> list[SplitRecording]:
    """Create participant-stratified splits and fit scaling on training only."""
    if not 0 <= fold < N_SPLITS:
        raise ValueError(f"fold must be in [0, {N_SPLITS})")
    rng = np.random.default_rng(seed)
    by_participant: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for participant in sorted({item.participant for item in recordings}):
        labels = next(r.labels for r in recordings if r.participant == participant)
        fitted_labels = rng.permutation(labels) if shuffled else labels.copy()
        train, test = stratified_folds(fitted_labels, N_SPLITS, rng)[fold]
        if np.intersect1d(train, test).size:
            raise RuntimeError("train/test leakage detected")
        by_participant[participant] = (fitted_labels, train, test)

    result: list[SplitRecording] = []
    for item in recordings:
        labels, train, test = by_participant[item.participant]
        # One scaler per neuron. No held-out observation contributes to it.
        mean = item.counts[train].mean(axis=(0, 1), keepdims=True)
        scale = item.counts[train].std(axis=(0, 1), keepdims=True)
        scale[scale == 0] = 1.0
        standardized = (item.counts - mean) / scale
        result.append(
            SplitRecording(
                participant=item.participant,
                region=item.region,
                x_train=standardized[train].astype(np.float32),
                y_train=labels[train],
                x_test=standardized[test].astype(np.float32),
                y_test=labels[test],
            )
        )
    return result


class MultiRegionRRR(nn.Module):
    """Paper-style multi-session and multi-region reduced-rank classifier.

    For recording s from region j, the coefficient tensor is
    ``W_s = U_s @ A_j @ B``. ``U_s`` is recording-specific, ``A_j`` is
    region-specific, and ``B`` is shared by all participants and regions.
    """

    def __init__(
        self,
        neuron_counts: dict[tuple[int, int], int],
        n_time: int,
        *,
        rank: int = 2,
        global_rank: int = 2,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if rank < 1 or global_rank < 1:
            raise ValueError("ranks must be positive")
        self.rank = rank
        self.global_rank = global_rank
        self.n_time = n_time
        self.device = torch.device(device)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.U = nn.ParameterDict(
            {
                self._key(*key): nn.Parameter(
                    torch.randn(n, rank) / np.sqrt(max(n, 1))
                )
                for key, n in neuron_counts.items()
            }
        )
        self.A = nn.Parameter(
            torch.randn(len(BRAIN_REGIONS), rank, global_rank) / np.sqrt(rank)
        )
        self.B = nn.Parameter(
            torch.randn(global_rank, n_time, 2) / np.sqrt(global_rank)
        )
        self.bias = nn.ParameterDict(
            {self._key(*key): nn.Parameter(torch.zeros(2)) for key in neuron_counts}
        )
        self.loss_history: list[float] = []
        self.to(self.device)

    @staticmethod
    def _key(participant: int, region: int) -> str:
        return f"p{participant}_r{region}"

    def forward(
        self, x: torch.Tensor, participant: int, region: int
    ) -> torch.Tensor:
        key = self._key(participant, region)
        temporal = torch.einsum("rl,ltd->rtd", self.A[region - 1], self.B)
        latent = torch.einsum("ktn,nr->ktr", x, self.U[key])
        return torch.einsum("ktr,rtd->kd", latent, temporal) + self.bias[key]

    def predict_proba(
        self, x: np.ndarray, participant: int, region: int
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            values = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            probability = torch.softmax(self(values, participant, region), dim=1)
        return probability.cpu().numpy()

    def fit(
        self,
        recordings: list[SplitRecording],
        *,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-3,
        batch_size: int = 128,
        seed: int = 0,
    ) -> "MultiRegionRRR":
        """Jointly fit all recordings with PyTorch AdamW on ``self.device``."""
        if epochs < 1 or learning_rate <= 0 or weight_decay < 0 or batch_size < 1:
            raise ValueError("invalid optimizer setting")
        rng = np.random.default_rng(seed)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        tensors = {
            (item.participant, item.region): (
                torch.as_tensor(item.x_train, dtype=torch.float32, device=self.device),
                torch.as_tensor(item.y_train, dtype=torch.long, device=self.device),
            )
            for item in recordings
        }
        class_weights = {}
        for item in recordings:
            counts = np.bincount(item.y_train, minlength=2).astype(np.float32)
            weights = len(item.y_train) / (2.0 * counts)
            class_weights[(item.participant, item.region)] = torch.as_tensor(
                weights, dtype=torch.float32, device=self.device
            )

        self.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for item in recordings:
                key = (item.participant, item.region)
                x_all, y_all = tensors[key]
                take = rng.choice(
                    len(item.y_train), min(batch_size, len(item.y_train)), replace=False
                )
                take_tensor = torch.as_tensor(take, dtype=torch.long, device=self.device)
                logits = self(x_all[take_tensor], item.participant, item.region)
                loss = F.cross_entropy(
                    logits, y_all[take_tensor], weight=class_weights[key]
                ) / len(recordings)
                loss.backward()
                total_loss += float(loss.detach().cpu())
            optimizer.step()
            self.loss_history.append(total_loss)
        return self


def score_model(
    model: MultiRegionRRR,
    recordings: list[SplitRecording],
    *,
    event: str,
    fold: int,
    shuffle: int,
) -> pd.DataFrame:
    """Score every participant-region using only its held-out test trials."""
    rows = []
    for item in recordings:
        probability = model.predict_proba(
            item.x_test, item.participant, item.region
        )[:, 1]
        prediction = (probability >= 0.5).astype(int)
        rows.append(
            {
                "participant": item.participant,
                "event": event,
                "region": item.region,
                "fold": fold,
                "shuffle": shuffle,
                "is_null": shuffle > 0,
                "accuracy": accuracy_score(item.y_test, prediction),
                "balanced_accuracy": balanced_accuracy_score(
                    item.y_test, prediction
                ),
                "roc_auc": roc_auc_score(item.y_test, probability),
                "n_train": len(item.y_train),
                "n_test": len(item.y_test),
                "n_neurons": item.x_train.shape[2],
            }
        )
    return pd.DataFrame(rows)


def _fit_fold_task(
    recordings: list[Recording],
    *,
    event: str,
    fold: int,
    shuffle: int,
    split_seed: int,
    model_seed: int,
    rank: int,
    global_rank: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """Fit and score one independent fold/shuffle task in a worker process."""
    neuron_counts = {
        (item.participant, item.region): item.counts.shape[2]
        for item in recordings
    }
    splits = make_fold_splits(
        recordings,
        fold=fold,
        seed=split_seed,
        shuffled=shuffle > 0,
    )
    model = MultiRegionRRR(
        neuron_counts,
        recordings[0].counts.shape[1],
        rank=rank,
        global_rank=global_rank,
        seed=model_seed,
        device=device,
    ).fit(
        splits,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=model_seed,
    )
    result = score_model(model, splits, event=event, fold=fold, shuffle=shuffle)
    print(
        f"Finished {event}, shuffle {shuffle}, fold {fold + 1}/{N_SPLITS}",
        flush=True,
    )
    return result


def available_devices(requested: str = "auto") -> list[str]:
    """Resolve a requested PyTorch device into safe parallel worker devices."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return ["mps"]
        return ["cpu"]
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        index = int(requested.split(":", 1)[1])
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        return [requested]
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available")
        return ["mps"]
    if requested == "cpu":
        return ["cpu"]
    raise ValueError("device must be auto, cpu, cuda, cuda:<index>, or mps")


def run_rrr_decoding(
    *,
    rank: int = 2,
    global_rank: int = 2,
    epochs: int = 200,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-3,
    batch_size: int = 128,
    label_shuffles: int = 0,
    seed: int = 12345,
    n_jobs: int = 1,
    device: str = "auto",
) -> pd.DataFrame:
    """Fit the RRR with stratified CV, parallelizing independent model fits.

    ``n_jobs`` follows joblib conventions: 1 runs serially and -1 uses all
    CPUs. Each worker needs its own standardized tensors and optimizer state,
    so a bounded value such as 2 or 4 is usually preferable to -1.
    """
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero")
    devices = available_devices(device)
    accelerator = devices[0] != "cpu"
    effective_n_jobs = n_jobs
    if accelerator:
        requested_jobs = len(devices) if n_jobs == -1 else n_jobs
        effective_n_jobs = min(requested_jobs, len(devices))
        if effective_n_jobs != requested_jobs:
            print(
                f"Capping n_jobs at {effective_n_jobs}: one RRR worker per accelerator",
                flush=True,
            )
    print(f"RRR compute device(s): {', '.join(devices)}", flush=True)
    rows = []
    seed_sequence = np.random.SeedSequence(seed)
    model_seeds = iter(
        seed_sequence.generate_state(
            len(EVENTS) * N_SPLITS * (label_shuffles + 1) * 2
        )
    )
    for event in EVENTS:
        recordings = load_event_recordings(event)
        tasks = []
        task_index = 0
        for shuffle in range(label_shuffles + 1):
            for fold in range(N_SPLITS):
                split_seed = int(next(model_seeds))
                model_seed = int(next(model_seeds))
                tasks.append(
                    delayed(_fit_fold_task)(
                    recordings,
                    event=event,
                    fold=fold,
                    shuffle=shuffle,
                    split_seed=split_seed,
                    model_seed=model_seed,
                    rank=rank,
                    global_rank=global_rank,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    device=devices[task_index % len(devices)],
                    )
                )
                task_index += 1
        rows.extend(
            Parallel(
                n_jobs=effective_n_jobs,
                prefer="processes",
                verbose=5,
                max_nbytes="10M",
                mmap_mode="r",
            )(tasks)
        )
    return pd.concat(rows, ignore_index=True)


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    result = np.full(pvalues.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(pvalues))
    if not finite_indices.size:
        return result
    finite = pvalues[finite_indices]
    order = np.argsort(finite)
    adjusted = finite[order] * len(finite) / np.arange(1, len(finite) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    finite_result = np.empty_like(adjusted)
    finite_result[order] = np.minimum(adjusted, 1.0)
    result[finite_indices] = finite_result
    return result


def _mean_ci95(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    """Return a finite-sample t interval without warning on small groups."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {"mean": np.nan, "sem": np.nan, "n": 0,
                "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(array.mean())
    if n == 1:
        return {"mean": mean, "sem": np.nan, "n": 1,
                "ci95_low": np.nan, "ci95_high": np.nan}
    sem = float(array.std(ddof=1) / np.sqrt(n))
    margin = float(t.ppf(0.975, n - 1) * sem)
    # Accuracy, balanced accuracy, and AUC are all bounded metrics.
    return {"mean": mean, "sem": sem, "n": n,
            "ci95_low": max(0.0, mean - margin),
            "ci95_high": min(1.0, mean + margin)}


def _safe_wilcoxon(difference: pd.Series | np.ndarray) -> tuple[float, float, int]:
    """Run Wilcoxon safely for missing, singleton, or all-zero differences."""
    values = np.asarray(difference, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, 0
    if np.allclose(values, 0.0):
        return 0.0, 1.0, int(values.size)
    statistic, pvalue = wilcoxon(values, zero_method="zsplit")
    return float(statistic), float(pvalue), int(values.size)


def regional_statistics(
    results: pd.DataFrame, *, metric: str = "roc_auc"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return regional summaries and participant-paired regional tests."""
    real = results.loc[~results["is_null"]]
    participant = real.groupby(["participant", "event", "region"])[metric].mean()
    summary_rows = []
    for (event, region), values in participant.groupby(["event", "region"]):
        summary_rows.append(
            {"event": event, "region": region, **_mean_ci95(values)}
        )
    summary = pd.DataFrame(summary_rows)

    tests = []
    for event in EVENTS:
        wide = participant.xs(event, level="event").unstack("region")
        for first, second in combinations(BRAIN_REGIONS, 2):
            if first not in wide or second not in wide:
                difference = pd.Series(dtype=float)
            else:
                difference = wide[first] - wide[second]
            statistic, pvalue, n_paired = _safe_wilcoxon(difference)
            tests.append(
                {
                    "event": event,
                    "metric": metric,
                    "region_1": first,
                    "region_2": second,
                    "mean_difference": difference.mean(),
                    "wilcoxon_statistic": statistic,
                    "pvalue": pvalue,
                    "n_participants": n_paired,
                }
            )
    tests = pd.DataFrame(tests)
    tests["qvalue"] = tests.groupby("event")["pvalue"].transform(
        lambda values: _benjamini_hochberg(values.to_numpy())
    )
    return summary, tests


def plot_regional_performance(
    results: pd.DataFrame,
    path: str | Path,
    *,
    metric: str = "roc_auc",
) -> Path:
    """Plot participant-level test performance for the three regions."""
    path = Path(path)
    real = results.loc[~results["is_null"]]
    participant = real.groupby(["participant", "event", "region"])[metric].mean()
    figure, axes = plt.subplots(1, len(EVENTS), figsize=(10, 4.5), sharey=True)
    rng = np.random.default_rng(9)
    for axis, event in zip(axes, EVENTS):
        wide = participant.xs(event, level="event").unstack("region")
        for _, row in wide.iterrows():
            axis.plot(BRAIN_REGIONS, row.loc[list(BRAIN_REGIONS)], color="0.8", alpha=0.5)
        for region in BRAIN_REGIONS:
            if region not in wide:
                continue
            values = wide[region].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            x = region + rng.normal(0, 0.025, len(values))
            axis.scatter(x, values, color=REGION_COLORS[region], alpha=0.65, s=22)
            interval = _mean_ci95(values)
            if not np.isfinite(interval["ci95_low"]):
                error = np.array([[0.0], [0.0]])
            else:
                error = np.array(
                    [[float(interval["mean"] - interval["ci95_low"])],
                     [float(interval["ci95_high"] - interval["mean"])]]
                )
            axis.errorbar(
                region,
                interval["mean"],
                yerr=error,
                color="black",
                marker="o",
                capsize=4,
                linewidth=2,
            )
        axis.axhline(0.5, color="0.35", linestyle="--", linewidth=1)
        axis.set_xticks(BRAIN_REGIONS, [f"Region {r}" for r in BRAIN_REGIONS])
        axis.set_title("Stimulus aligned" if event == "stim_start" else "Outcome aligned")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Held-out " + metric.replace("_", " ").upper())
    figure.suptitle("Multi-region reduced-rank decoding of variable A")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(figure)
    return path


def save_results(results: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "fold_test_metrics.csv", index=False)
    for metric in ("balanced_accuracy", "roc_auc", "accuracy"):
        summary, tests = regional_statistics(results, metric=metric)
        summary.to_csv(output / f"regional_summary_{metric}.csv", index=False)
        tests.to_csv(output / f"paired_region_tests_{metric}.csv", index=False)
        plot_regional_performance(
            results, output / f"regional_{metric}.png", metric=metric
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--global-rank", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--label-shuffles", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="parallel fold/shuffle workers; 2-4 is recommended, -1 uses all CPUs",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, cuda, cuda:<index>, or mps",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    results = run_rrr_decoding(
        rank=args.rank,
        global_rank=args.global_rank,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        label_shuffles=args.label_shuffles,
        seed=args.seed,
        n_jobs=args.n_jobs,
        device=args.device,
    )
    save_results(results, args.output)
    print(f"Saved RRR results to {args.output}")


if __name__ == "__main__":
    main()
