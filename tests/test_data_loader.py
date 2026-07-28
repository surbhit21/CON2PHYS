"""Tests for the public CON2PHYS data-loader API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
import pytest

from src import (
    available_subjects,
    firing_rate_by_brain_area,
    load_con2phys,
    load_session,
)
from src.data_loader import DEFAULT_DATA_PATH

from .conftest import SUBJECT


def test_available_subjects_reads_nested_archive(consortium_data: Path) -> None:
    assert available_subjects(consortium_data) == [SUBJECT]
    assert available_subjects(consortium_data / "py-files.zip") == [SUBJECT]


def test_loads_nested_archive_as_pynapple_objects(consortium_data: Path) -> None:
    session = load_session(
        SUBJECT,
        consortium_data,
        lfp_banks=(1, 3),
        sampling_rate=2.0,
        lfp_dtype="float32",
    )

    assert session.subject == SUBJECT
    assert session.sampling_rate == 2.0
    assert set(session.lfp) == {1, 3}
    assert isinstance(session.lfp[1], nap.TsdFrame)
    assert session.lfp[1].shape == (6, 2)
    assert session.lfp[1].values.dtype == np.float32
    np.testing.assert_allclose(session.lfp[1].index, np.arange(6) / 2.0)
    assert session.lfp[1].columns.tolist() == ["bank_1_ch_0", "bank_1_ch_1"]
    np.testing.assert_array_equal(
        session.lfp[1].values,
        np.arange(12, dtype=np.float32).reshape(2, 6).T,
    )

    assert isinstance(session.spikes, nap.TsGroup)
    assert list(session.spikes.keys()) == [10, 20]
    np.testing.assert_allclose(session.spikes[10].index, [0.1, 1.1])
    np.testing.assert_allclose(session.spikes[20].index, [0.4, 1.4])
    assert session.spikes.metadata.loc[10, "brain_area"] == 1
    assert session.spikes.metadata.loc[20, "brain_area"] == 3

    assert isinstance(session.trials, nap.IntervalSet)
    assert len(session.trials) == 2
    np.testing.assert_allclose(session.trials.start, [0.1, 1.1])
    np.testing.assert_allclose(session.trials.end, [1.0, 2.0])
    assert session.trials.metadata["variable_A"].tolist() == [1, 0]
    assert set(session.events) == {"stim_start", "outcome"}
    assert all(isinstance(event, nap.Ts) for event in session.events.values())
    np.testing.assert_allclose(session.events["stim_start"].index, [0.2, 1.2])

    assert isinstance(session.trial_table, pd.DataFrame)
    assert session.waveforms is not None
    assert session.waveforms.index.tolist() == [10, 20]
    np.testing.assert_array_equal(
        session.waveforms.to_numpy(), np.arange(8, dtype=np.float32).reshape(2, 4)
    )


def test_selective_loading_skips_lfp_and_spikes(consortium_data: Path) -> None:
    session = load_con2phys(
        SUBJECT, consortium_data, lfp_banks=(), load_spikes=False
    )

    assert session.lfp == {}
    assert session.spikes is None
    assert session.waveforms is None
    assert len(session.trials) == 2


@pytest.mark.parametrize("fixture_name", ["subject_zip", "extracted_data"])
def test_supports_alternate_storage_layouts(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    data_path = request.getfixturevalue(fixture_name)
    session = load_session(SUBJECT, data_path, lfp_banks=(), load_spikes=False)
    assert len(session.trials) == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lfp_banks": (4,)}, "lfp_banks"),
        ({"sampling_rate": 0}, "sampling_rate"),
    ],
)
def test_rejects_invalid_options(
    consortium_data: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_session(SUBJECT, consortium_data, **kwargs)


def test_reports_missing_subject(consortium_data: Path) -> None:
    with pytest.raises(ValueError, match="subject 99"):
        load_session(99, consortium_data, lfp_banks=(), load_spikes=False)


def test_firing_rate_by_brain_area(consortium_data: Path) -> None:
    session = load_session(SUBJECT, consortium_data, lfp_banks=())
    assert session.spikes is not None

    rates = firing_rate_by_brain_area(session.spikes)

    assert rates.index.tolist() == [1, 3]
    assert rates["n_units"].tolist() == [1, 1]
    assert rates["n_spikes"].tolist() == [2, 2]
    np.testing.assert_allclose(rates["duration_s"], 1.3)
    np.testing.assert_allclose(rates["population_firing_rate_hz"], 2 / 1.3)
    np.testing.assert_allclose(rates["mean_unit_firing_rate_hz"], 2 / 1.3)


def test_firing_rate_can_be_restricted_to_epoch(consortium_data: Path) -> None:
    session = load_session(SUBJECT, consortium_data, lfp_banks=())
    assert session.spikes is not None
    epoch = nap.IntervalSet(start=[0.1], end=[1.0], time_units="s")

    rates = firing_rate_by_brain_area(session.spikes, epoch)

    assert rates["n_spikes"].tolist() == [1, 1]
    np.testing.assert_allclose(rates["duration_s"], 0.9)
    np.testing.assert_allclose(rates["mean_unit_firing_rate_hz"], 1 / 0.9)


def test_firing_rate_requires_brain_area_metadata() -> None:
    spikes = nap.TsGroup({1: nap.Ts(t=[0.1, 0.2])})

    with pytest.raises(ValueError, match="brain_area"):
        firing_rate_by_brain_area(spikes)


@pytest.mark.integration
def test_participant_1_firing_rates_match_reference_data() -> None:
    """Check participant 1 against rates calculated from the consortium release."""
    archive = DEFAULT_DATA_PATH / "py-files.zip"
    if not archive.is_file():
        pytest.skip("the consortium data archive is not available")

    session = load_session(
        subject=1,
        data_path=DEFAULT_DATA_PATH,
        lfp_banks=(),
    )
    assert session.spikes is not None
    rates = firing_rate_by_brain_area(session.spikes)

    expected_hz = pd.Series(
        {
            1: 9.31268669,
            2: 5.69492681,
            3: 11.35885574,
        },
        name="mean_unit_firing_rate_hz",
    ).rename_axis("brain_area")
    actual_hz = rates["mean_unit_firing_rate_hz"].sort_index()

    pd.testing.assert_series_equal(
        actual_hz,
        expected_hz,
        check_exact=False,
        rtol=1e-7,
        atol=1e-8,
    )
