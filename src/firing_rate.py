"""Firing-rate summaries for CON2PHYS spike data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pynapple as nap


def firing_rate_by_brain_area(
    spikes: nap.TsGroup,
    epoch: nap.IntervalSet | None = None,
    *,
    area_column: str = "brain_area",
) -> pd.DataFrame:
    """Calculate firing rates for each brain area.

    Parameters
    ----------
    spikes:
        A Pynapple ``TsGroup`` whose unit metadata contains ``area_column``.
        ``load_session(...).spikes`` is directly compatible.
    epoch:
        Optional intervals over which to count spikes. For example, pass
        ``session.trials`` to calculate rates during trials only. If omitted,
        the full time support of ``spikes`` is used.
    area_column:
        Name of the unit metadata column containing brain-area labels.

    Returns
    -------
    pandas.DataFrame
        One row per brain area with the number of units, total spike count,
        analysed duration, population firing rate, and mean firing rate per
        unit. Rates are expressed in Hz.

    Notes
    -----
    ``population_firing_rate_hz`` is the sum of the constituent units' rates.
    ``mean_unit_firing_rate_hz`` divides that value by the number of units and
    is usually the appropriate value for comparing areas with different unit
    counts.
    """
    if not isinstance(spikes, nap.TsGroup):
        raise TypeError("spikes must be a pynapple.TsGroup")
    if area_column not in spikes.metadata.columns:
        raise ValueError(
            f"TsGroup metadata does not contain an {area_column!r} column"
        )
    if epoch is not None and not isinstance(epoch, nap.IntervalSet):
        raise TypeError("epoch must be a pynapple.IntervalSet or None")

    analysis_epoch = spikes.time_support if epoch is None else epoch
    duration = float(analysis_epoch.tot_length("s"))
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("the analysis epoch must have positive duration")

    selected = spikes if epoch is None else spikes.restrict(analysis_epoch)
    records: list[dict[str, object]] = []

    # sort=False preserves the consortium's area order and also supports a
    # mixture of numeric and textual area labels.
    for area, unit_metadata in spikes.metadata.groupby(area_column, sort=False):
        unit_ids = unit_metadata.index.tolist()
        spike_count = sum(len(selected[unit_id]) for unit_id in unit_ids)
        unit_count = len(unit_ids)
        population_rate = spike_count / duration
        records.append(
            {
                area_column: area,
                "n_units": unit_count,
                "n_spikes": spike_count,
                "duration_s": duration,
                "population_firing_rate_hz": population_rate,
                "mean_unit_firing_rate_hz": population_rate / unit_count,
            }
        )

    columns = [
        "n_units",
        "n_spikes",
        "duration_s",
        "population_firing_rate_hz",
        "mean_unit_firing_rate_hz",
    ]
    if not records:
        return pd.DataFrame(columns=columns).rename_axis(area_column)
    return pd.DataFrame.from_records(records).set_index(area_column)
