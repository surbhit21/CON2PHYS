from __future__ import annotations

import numpy as np

from src.Q6 import (
    build_glm_design,
    fit_connectivity_glms,
    plot_coupling_strength_histogram,
    plot_inter_region_coupling,
    run_all_participants,
    summarize_coupling_strengths,
)
from src.data_loader import load_session

from .conftest import SUBJECT


def test_design_has_causal_trial_reset_histories(consortium_data) -> None:
    session = load_session(SUBJECT, consortium_data, lfp_banks=())
    design = build_glm_design(
        session,
        source_area=3,
        target_area=1,
        bin_size=0.1,
        target_history=0.2,
        source_history=0.3,
    )
    assert design.design.shape == (18, 11)
    assert design.counts.shape == (18, 1)
    assert set(design.feature_table["feature_group"]) == {
        "target_history", "source_history", "trial"
    }
    # First bin of every trial has no spike-history contribution.
    history_columns = 2 + 3
    starts = np.flatnonzero(np.r_[True, design.trial_ids[1:] != design.trial_ids[:-1]])
    assert design.design[starts, :history_columns].nnz == 0


def test_fit_returns_lag_resolved_connectivity(consortium_data) -> None:
    session = load_session(SUBJECT, consortium_data, lfp_banks=())
    design = build_glm_design(
        session,
        source_area=3,
        target_area=1,
        bin_size=0.1,
        target_history=0.2,
        source_history=0.3,
    )
    result = fit_connectivity_glms(design, ridge=1.0, max_iter=20)
    source = result.coefficients.query("feature_group == 'source_history'")
    assert source["lag_ms"].tolist() == [100.0, 200.0, 300.0]
    assert result.source_connectivity.shape[0] == 1
    assert result.model_scores["target_unit_id"].tolist() == [10]


def test_coupling_plot_is_saved(consortium_data, tmp_path) -> None:
    session = load_session(SUBJECT, consortium_data, lfp_banks=())
    design = build_glm_design(
        session, source_area=3, target_area=1, bin_size=0.1,
        target_history=0.2, source_history=0.3,
    )
    result = fit_connectivity_glms(design, ridge=1.0, max_iter=20)
    output = plot_inter_region_coupling(result, tmp_path / "coupling.png")
    assert output.is_file()
    assert output.stat().st_size > 0
    histogram = plot_coupling_strength_histogram(
        result, tmp_path / "histogram.png", bins=20
    )
    assert histogram.is_file()
    assert histogram.stat().st_size > 0


def test_all_participants_runner_writes_combined_outputs(consortium_data, tmp_path) -> None:
    output = tmp_path / "analysis"
    plots = tmp_path / "plots"
    status = run_all_participants(
        source_area=3,
        target_area=1,
        data_path=consortium_data,
        output_root=output,
        plot_root=plots,
        max_iter=2,
    )
    assert status[["participant", "status"]].to_dict("records") == [
        {"participant": SUBJECT, "status": "completed"}
    ]
    assert (output / f"participant_{SUBJECT}" / "glm_coefficients.csv").is_file()
    assert (output / "all_participants_source_target_connectivity.csv").is_file()
    assert (output / "all_participants_glm_model_scores.csv").is_file()
    assert (output / "coupling_strength_by_participant.csv").is_file()
    assert (output / "coupling_strength_group_summary.csv").is_file()
    assert (plots / "all_participants_coupling_strength_histogram.png").is_file()


def test_coupling_summary_uses_participant_means() -> None:
    import pandas as pd

    connectivity = pd.DataFrame(
        {
            "participant": [1, 1, 2, 2],
            "summed_coefficient": [0.0, 2.0, 2.0, 4.0],
        }
    )
    summary = summarize_coupling_strengths(
        connectivity, group_column="participant"
    ).iloc[0]
    assert summary["observation_unit"] == "participant_means"
    assert summary["n"] == 2
    assert summary["mean_coupling_strength"] == 2.0
    assert summary["ci_low"] < 2.0 < summary["ci_high"]
