"""Small, generated CON2PHYS archives shared by the loader tests."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SUBJECT = 7


def _npy_bytes(value: object, *, allow_pickle: bool = True) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=allow_pickle)
    return stream.getvalue()


def _subject_zip_bytes(subject: int = SUBJECT) -> bytes:
    trials = pd.DataFrame(
        {
            "variable_A": [1, 0],
            "variable_B": [0, 1],
            "variable_C": [0, 2],
            "trial_start": [0.1, 1.1],
            "stim_start": [0.2, 1.2],
            "outcome": [0.8, 1.8],
            "trial_end": [1.0, 2.0],
        }
    )
    arrays = {
        "brain_area.npy": np.array(
            {
                "cluster_id": np.array([10, 20], dtype=np.int32),
                "brain_area": np.array([1, 3], dtype=np.int32),
            },
            dtype=object,
        ),
        # Deliberately interleave clusters to exercise grouping.
        "clusters.npy": np.array([20, 10, 20, 10], dtype=np.int32),
        "spikes.npy": np.array([0.4, 0.1, 1.4, 1.1], dtype=np.float64),
        "waveforms.npy": np.arange(8, dtype=np.float32).reshape(2, 4),
        "lfp_1.npy": np.arange(12, dtype=np.float64).reshape(2, 6),
        "lfp_2.npy": np.arange(6, dtype=np.float64).reshape(1, 6),
        "lfp_3.npy": np.arange(18, dtype=np.float64).reshape(3, 6),
    }

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in arrays.items():
            archive.writestr(f"{subject}/{name}", _npy_bytes(value))
        archive.writestr(f"{subject}/trial_data.csv", trials.to_csv())
    return output.getvalue()


@pytest.fixture
def consortium_data(tmp_path: Path) -> Path:
    """Directory containing the same nested ZIP structure as the release."""
    outer = tmp_path / "py-files.zip"
    with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(f"py-files/{SUBJECT}.zip", _subject_zip_bytes())
    return tmp_path


@pytest.fixture
def subject_zip(tmp_path: Path) -> Path:
    """Standalone subject ZIP accepted by ``load_session``."""
    path = tmp_path / f"{SUBJECT}.zip"
    path.write_bytes(_subject_zip_bytes())
    return path


@pytest.fixture
def extracted_data(tmp_path: Path) -> Path:
    """Parent directory containing an extracted subject directory."""
    with zipfile.ZipFile(io.BytesIO(_subject_zip_bytes())) as archive:
        archive.extractall(tmp_path)
    return tmp_path
