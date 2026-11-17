"""Statistical summary checks and a temporary-directory pipeline test."""

import math

import matplotlib

# Select a noninteractive backend before importing plotting code.
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src import analyze, figures, simulation
from src.scenarios import generate_scenarios


def example_results(successes, scenario_id="example"):
    """Construct replication records with known statistical outcomes."""
    return pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "regime": "linear",
            "p": 128,
            "k": 52,
            "n": 451,
            "theta": 1.0,
            "replication_id": range(len(successes)),
            "success": successes,
        }
    )


def test_summary_matches_hand_calculation():
    """Two successes in four trials give probability 1/2 and MCSE sqrt(1/12)."""
    summary = analyze.summarize_results(
        example_results([1, 1, 0, 0])
    )

    assert len(summary) == 1
    row = summary.iloc[0]

    assert row["n_replications"] == 4
    assert row["n_success"] == 2
    assert row["recovery_probability"] == pytest.approx(0.5)
    assert row["mcse"] == pytest.approx(math.sqrt(1 / 12))


def test_summary_keeps_scenarios_separate():
    """Different scenario probabilities must not be pooled."""
    first = example_results([1, 1, 0, 0], "first")
    second = example_results([1, 0, 0], "second")

    summary = analyze.summarize_results(
        pd.concat([first, second], ignore_index=True)
    ).set_index("scenario_id")

    assert set(summary.index) == {"first", "second"}

    assert summary.loc["first", "n_replications"] == 4
    assert summary.loc["first", "n_success"] == 2
    assert summary.loc[
        "first", "recovery_probability"
    ] == pytest.approx(0.5)

    assert summary.loc["second", "n_replications"] == 3
    assert summary.loc["second", "n_success"] == 1
    assert summary.loc[
        "second", "recovery_probability"
    ] == pytest.approx(1 / 3)
    assert summary.loc["second", "mcse"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("success", [0, 1])
def test_summary_constant_outcomes(success):
    """Constant observed outcomes have zero sample-based MCSE."""
    row = analyze.summarize_results(
        example_results([success] * 4)
    ).iloc[0]

    assert row["n_replications"] == 4
    assert row["n_success"] == 4 * success
    assert row["recovery_probability"] == success
    assert row["mcse"] == 0.0


def test_summary_single_replication_has_undefined_mcse():
    """One replication cannot estimate between-replication variance."""
    row = analyze.summarize_results(
        example_results([1])
    ).iloc[0]

    assert row["n_replications"] == 1
    assert row["recovery_probability"] == 1.0
    assert np.isnan(row["mcse"])


def test_summary_rejects_nonbinary_success():
    """Nonbinary outcomes are invalid for a recovery probability."""
    with pytest.raises(ValueError, match="binary"):
        analyze.summarize_results(
            example_results([1, 0, 0.5])
        )


def test_summary_rejects_empty_results():
    """An empty experiment cannot produce a recovery estimate."""
    empty = example_results([1]).iloc[:0]

    with pytest.raises(ValueError, match="empty"):
        analyze.summarize_results(empty)


def test_summary_rejects_missing_columns():
    """Missing grouping or outcome columns must produce a clear error."""
    incomplete = example_results([1, 0]).drop(columns="success")

    with pytest.raises(ValueError, match="Missing required columns"):
        analyze.summarize_results(incomplete)


def test_small_pipeline_in_temporary_directory(
    tmp_path, monkeypatch, capsys
):
    """Exercise simulation, CSV analysis, and figures without changing results."""
    design = generate_scenarios()
    selected = []

    # One threshold-region scenario from each regime keeps this test small.
    for regime in ("linear", "sublinear", "fractional"):
        match = design.loc[
            design["regime"].eq(regime)
            & design["p"].eq(128)
            & np.isclose(design["theta"], 1.0)
        ]
        assert len(match) == 1
        selected.append(match)

    scenarios = pd.concat(selected, ignore_index=True)
    scenarios["n_replications"] = 4

    raw_path = tmp_path / "results/raw/simulation_results.csv"
    summary_path = tmp_path / "results/recovery_summary.csv"
    figure_dir = tmp_path / "results/figures"

    # Direct every pipeline output to pytest's temporary directory.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        simulation, "generate_scenarios", lambda: scenarios.copy()
    )
    monkeypatch.setattr(simulation, "OUTPUT_PATH", raw_path)
    monkeypatch.setattr("sys.argv", ["simulation"])

    simulation.main()

    assert raw_path.is_file()
    raw = pd.read_csv(raw_path)

    assert len(raw) == 12
    assert not raw.duplicated(
        ["scenario_id", "replication_id"]
    ).any()
    assert raw.groupby("scenario_id").size().eq(4).all()
    assert raw["success"].isin([0, 1]).all()
    assert np.isfinite(raw["lambda_n"]).all()
    assert np.isfinite(raw["dual_gap"]).all()

    monkeypatch.setattr(analyze, "RAW_PATH", raw_path)
    monkeypatch.setattr(analyze, "SUMMARY_PATH", summary_path)
    analyze.main()

    assert summary_path.is_file()
    summary = pd.read_csv(summary_path).set_index("scenario_id")

    assert len(summary) == 3
    assert summary["n_replications"].eq(4).all()

    # Validate saved summaries independently using sample SD / sqrt(B).
    for scenario_id, group in raw.groupby("scenario_id"):
        observed = group["success"].to_numpy(dtype=float)
        row = summary.loc[scenario_id]

        assert row["n_success"] == int(observed.sum())
        assert row["recovery_probability"] == pytest.approx(
            observed.mean()
        )
        assert row["mcse"] == pytest.approx(
            observed.std(ddof=1) / math.sqrt(len(observed))
        )

    monkeypatch.setattr(figures, "RAW_PATH", raw_path)
    monkeypatch.setattr(figures, "SUMMARY_PATH", summary_path)
    monkeypatch.setattr(figures, "FIGURE_DIR", figure_dir)
    figures.main()

    for filename in ("recovery_curves.png", "solver_diagnostics.png"):
        output = figure_dir / filename
        assert output.is_file()

        # Decode the PNG to verify that the plotting output is readable.
        image = matplotlib.image.imread(output)
        assert image.ndim == 3
        assert image.shape[0] > 100
        assert image.shape[1] > 100
        assert np.isfinite(image).all()

    output = capsys.readouterr().out
    assert "Completed 12 replications." in output
    assert "Summarized 3 scenarios." in output