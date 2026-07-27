"""Load the CON2PHYS consortium data as Pynapple objects.

The distributed ``py-files.zip`` is a ZIP64 archive containing one zip file
per subject.  This module reads those nested archives directly; extracting
roughly 32 GB of data a second time is not necessary.

Examples
--------
Load spikes, trials, and only the first LFP bank::

    from src.data_loader import load_session

    session = load_session(1, lfp_banks=(1,))
    session.spikes              # pynapple.TsGroup
    session.lfp[1]              # pynapple.TsdFrame
    session.trials              # pynapple.IntervalSet
    session.events["stim_start"]  # pynapple.Ts

Skip the large LFP arrays when doing spike-only analyses::

    session = load_session(1, lfp_banks=())
"""

from __future__ import annotations

import io
import os
import struct
import zipfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

import numpy as np
import pandas as pd
import pynapple as nap

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data"
DEFAULT_LFP_RATE = 500.0


@dataclass(frozen=True)
class CON2PHYSSession:
    """The Pynapple objects and accompanying unit/trial metadata for a subject."""

    subject: int
    lfp: dict[int, nap.TsdFrame]
    spikes: nap.TsGroup | None
    trials: nap.IntervalSet
    events: dict[str, nap.Ts]
    trial_table: pd.DataFrame
    waveforms: pd.DataFrame | None
    sampling_rate: float


class _FileSlice(io.RawIOBase):
    """Seekable view onto a byte range of a file.

    ``ZipExtFile.seek`` reaches an offset by reading from the beginning.  Since
    the outer consortium members are stored (not compressed), this small view
    lets ``zipfile`` jump directly to an inner archive's central directory.
    """

    def __init__(self, path: Path, start: int, size: int):
        self._file = path.open("rb", buffering=0)
        self._start = start
        self._size = size
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def readinto(self, buffer: bytearray) -> int:
        count = min(len(buffer), self._size - self._position)
        if count <= 0:
            return 0
        data = os.pread(
            self._file.fileno(), count, self._start + self._position
        )
        buffer[: len(data)] = data
        self._position += len(data)
        return len(data)

    def close(self) -> None:
        self._file.close()
        super().close()


def _member_data_offset(archive: Path, info: zipfile.ZipInfo) -> int:
    """Return the on-disk start of a stored zip member."""
    with archive.open("rb") as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
    if len(header) != 30:
        raise zipfile.BadZipFile("truncated local file header")
    values = struct.unpack("<IHHHHHIIIHH", header)
    if values[0] != 0x04034B50:
        raise zipfile.BadZipFile("invalid local file header")
    filename_length, extra_length = values[-2:]
    return info.header_offset + 30 + filename_length + extra_length


def _find_outer_archive(data_path: Path) -> Path | None:
    if data_path.is_file() and data_path.name == "py-files.zip":
        return data_path
    candidate = data_path / "py-files.zip"
    return candidate if candidate.is_file() else None


@contextmanager
def _subject_source(
    subject: int, data_path: str | os.PathLike[str]
) -> Iterator[tuple[zipfile.ZipFile | None, Path | None]]:
    """Yield either an open subject archive or an extracted subject directory."""
    path = Path(data_path).expanduser().resolve()
    extracted = path / str(subject)
    if extracted.is_dir():
        yield None, extracted
        return

    subject_zip = path if path.is_file() and path.name == f"{subject}.zip" else path / f"{subject}.zip"
    if subject_zip.is_file():
        with zipfile.ZipFile(subject_zip) as archive:
            yield archive, None
        return

    outer_path = _find_outer_archive(path)
    if outer_path is None:
        raise FileNotFoundError(
            f"Could not find subject {subject}, {subject}.zip, or py-files.zip "
            f"under {path}"
        )

    with ExitStack() as stack:
        outer = stack.enter_context(zipfile.ZipFile(outer_path))
        member_name = f"py-files/{subject}.zip"
        try:
            info = outer.getinfo(member_name)
        except KeyError as error:
            raise ValueError(f"subject {subject} is not present in {outer_path}") from error
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{member_name} is compressed in the outer archive; extract it first"
            )
        raw_slice = _FileSlice(
            outer_path, _member_data_offset(outer_path, info), info.file_size
        )
        buffered = stack.enter_context(io.BufferedReader(raw_slice))
        archive = stack.enter_context(zipfile.ZipFile(buffered))
        yield archive, None


@contextmanager
def _open_member(
    archive: zipfile.ZipFile | None, directory: Path | None, subject: int, name: str
) -> Iterator[BinaryIO]:
    relative = f"{subject}/{name}"
    if archive is not None:
        with archive.open(relative) as stream:
            yield stream
    else:
        assert directory is not None
        with (directory / name).open("rb") as stream:
            yield stream


def _load_npy(
    archive: zipfile.ZipFile | None,
    directory: Path | None,
    subject: int,
    name: str,
    *,
    allow_pickle: bool = False,
) -> np.ndarray:
    with _open_member(archive, directory, subject, name) as stream:
        return np.load(stream, allow_pickle=allow_pickle)


def _load_trials(
    archive: zipfile.ZipFile | None, directory: Path | None, subject: int
) -> pd.DataFrame:
    with _open_member(archive, directory, subject, "trial_data.csv") as stream:
        trials = pd.read_csv(stream, encoding="utf-8-sig", index_col=0)
    trials.index.name = "trial"
    return trials


def _make_spikes(
    spike_times: np.ndarray,
    cluster_ids: np.ndarray,
    brain_area: dict,
) -> nap.TsGroup:
    if spike_times.shape != cluster_ids.shape:
        raise ValueError("spikes.npy and clusters.npy have different shapes")
    unit_ids = np.asarray(brain_area["cluster_id"], dtype=int)
    areas = np.asarray(brain_area["brain_area"])
    if len(unit_ids) != len(areas):
        raise ValueError("brain_area cluster_id and brain_area arrays differ in length")

    order = np.argsort(cluster_ids, kind="stable")
    sorted_clusters = cluster_ids[order]
    boundaries = np.searchsorted(sorted_clusters, unit_ids)
    ends = np.searchsorted(sorted_clusters, unit_ids, side="right")
    units = {
        int(unit): nap.Ts(t=np.sort(spike_times[order[start:end]]), time_units="s")
        for unit, start, end in zip(unit_ids, boundaries, ends)
    }
    metadata = pd.DataFrame({"brain_area": areas}, index=unit_ids)
    metadata.index.name = "cluster_id"
    return nap.TsGroup(units, time_units="s", metadata=metadata)


def load_session(
    subject: int,
    data_path: str | os.PathLike[str] = DEFAULT_DATA_PATH,
    *,
    lfp_banks: Sequence[int] = (1, 2, 3),
    load_spikes: bool = True,
    sampling_rate: float = DEFAULT_LFP_RATE,
    lfp_dtype: str | np.dtype | None = None,
) -> CON2PHYSSession:
    """Load one CON2PHYS subject.

    Parameters
    ----------
    subject:
        Subject number (1 through 18 in the consortium release).
    data_path:
        The data directory, the outer ``py-files.zip``, an extracted subject
        directory's parent, or an individual ``<subject>.zip``.
    lfp_banks:
        Which of ``lfp_1.npy`` through ``lfp_3.npy`` to load. Pass ``()`` for
        spike/trial-only work. Each bank can occupy several hundred MB.
    load_spikes:
        Load spike times, cluster identities, brain-area metadata, and
        waveforms.
    sampling_rate:
        LFP sampling rate in Hz. The distributed files are sampled at 500 Hz.
    lfp_dtype:
        Optional dtype conversion, e.g. ``"float32"`` to halve LFP memory.

    Returns
    -------
    CON2PHYSSession
        LFP banks are ``TsdFrame`` objects, spikes are a ``TsGroup``, trial
        extents are an ``IntervalSet``, and stimulus/outcome times are ``Ts``.
    """
    if not isinstance(subject, (int, np.integer)) or subject < 1:
        raise ValueError("subject must be a positive integer")
    banks = tuple(dict.fromkeys(int(bank) for bank in lfp_banks))
    if any(bank not in (1, 2, 3) for bank in banks):
        raise ValueError("lfp_banks may only contain 1, 2, and 3")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    with _subject_source(int(subject), data_path) as (archive, directory):
        trial_table = _load_trials(archive, directory, int(subject))
        required = {"trial_start", "trial_end", "stim_start", "outcome"}
        missing = required.difference(trial_table.columns)
        if missing:
            raise ValueError(f"trial_data.csv is missing columns: {sorted(missing)}")

        metadata = trial_table.drop(columns=["trial_start", "trial_end"])
        trials = nap.IntervalSet(
            start=trial_table["trial_start"].to_numpy(),
            end=trial_table["trial_end"].to_numpy(),
            time_units="s",
            metadata=metadata,
        )
        events = {
            column: nap.Ts(t=trial_table[column].to_numpy(), time_units="s")
            for column in ("stim_start", "outcome")
        }

        lfp: dict[int, nap.TsdFrame] = {}
        for bank in banks:
            values = _load_npy(
                archive, directory, int(subject), f"lfp_{bank}.npy"
            )
            if values.ndim != 2:
                raise ValueError(f"lfp_{bank}.npy must be a two-dimensional array")
            if lfp_dtype is not None:
                values = values.astype(lfp_dtype, copy=False)
            times = np.arange(values.shape[1], dtype=np.float64) / sampling_rate
            columns = [f"bank_{bank}_ch_{channel}" for channel in range(values.shape[0])]
            lfp[bank] = nap.TsdFrame(
                t=times, d=values.T, time_units="s", columns=columns
            )

        spikes: nap.TsGroup | None = None
        waveforms: pd.DataFrame | None = None
        if load_spikes:
            spike_times = _load_npy(
                archive, directory, int(subject), "spikes.npy"
            )
            cluster_ids = _load_npy(
                archive, directory, int(subject), "clusters.npy"
            )
            brain_area_array = _load_npy(
                archive,
                directory,
                int(subject),
                "brain_area.npy",
                allow_pickle=True,
            )
            brain_area = brain_area_array.item()
            spikes = _make_spikes(spike_times, cluster_ids, brain_area)

            waveform_values = _load_npy(
                archive, directory, int(subject), "waveforms.npy"
            )
            unit_ids = np.asarray(brain_area["cluster_id"], dtype=int)
            if waveform_values.shape[0] != len(unit_ids):
                raise ValueError("waveforms and brain-area metadata have different unit counts")
            waveforms = pd.DataFrame(waveform_values, index=unit_ids)
            waveforms.index.name = "cluster_id"
            waveforms.columns.name = "sample"

    return CON2PHYSSession(
        subject=int(subject),
        lfp=lfp,
        spikes=spikes,
        trials=trials,
        events=events,
        trial_table=trial_table,
        waveforms=waveforms,
        sampling_rate=float(sampling_rate),
    )


def available_subjects(
    data_path: str | os.PathLike[str] = DEFAULT_DATA_PATH,
) -> list[int]:
    """Return subject numbers discoverable at ``data_path``."""
    path = Path(data_path).expanduser().resolve()
    outer_path = _find_outer_archive(path)
    if outer_path is not None:
        with zipfile.ZipFile(outer_path) as archive:
            subjects = []
            for name in archive.namelist():
                if name.startswith("py-files/") and name.endswith(".zip"):
                    stem = Path(name).stem
                    if stem.isdigit():
                        subjects.append(int(stem))
            return sorted(subjects)

    base = path if path.is_dir() else path.parent
    subjects = {
        int(item.stem)
        for item in base.iterdir()
        if item.stem.isdigit() and (item.is_dir() or item.suffix == ".zip")
    }
    return sorted(subjects)


# A short, discoverable alias for notebook use.
load_con2phys = load_session
