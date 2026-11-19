from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from energy_monitor.experiments.figures import (
    FIGURES,
    generate_figures,
    load_effect_rows,
    load_reports,
    p_value_text,
    plot_residual_scale,
    plot_structural_break,
)

REPORTS_DIR = Path("artifacts/experiments")

pytestmark = pytest.mark.skipif(
    not (REPORTS_DIR / "e5_forecasting_test_d1.json").exists(),
    reason="relatórios dos experimentos ausentes",
)


@pytest.fixture(scope="module")
def reports():
    return load_reports(REPORTS_DIR)


@pytest.mark.parametrize("name", list(FIGURES))
def test_each_figure_is_built_from_reports(reports, name):
    figure = FIGURES[name](reports)

    assert isinstance(figure, Figure)
    assert figure.axes


def test_load_effect_rows_match_reported_reductions(reports):
    rows = load_effect_rows(reports.e5).set_index(["label", "model"])

    expected = reports.e5["d1"]["horizons"]["1h"]["comparisons"]
    reduction = expected["lightgbm_with_load_vs_history_only"]["mae_reduction_percent"]
    assert rows.loc[("D1, 1h", "lightgbm"), "reduction"] == pytest.approx(reduction)
    assert len(rows) == 8


def test_p_value_text_formats_small_and_regular_values():
    assert p_value_text(0.0001) == "p < 0,001"
    assert p_value_text(0.3102) == "p = 0,310"


def test_residual_scale_reports_zero_mad_hosts():
    statistics = {
        "a": {"median": 0.0, "mad": 1e-7},
        "b": {"median": 0.1, "mad": 0.5},
        "c": {"median": -0.1, "mad": 2.0},
    }

    figure = plot_residual_scale(statistics, 20.0)

    assert "MAD < 0,001 W em 1 servidores" in figure.axes[0].get_title()


def test_structural_break_uses_daily_local_means(reports):
    timestamps = pd.date_range("2019-02-01", "2019-06-20", freq="6h", tz="UTC")
    grid = pd.DataFrame(
        {
            "ts": timestamps,
            "cpu_load": np.linspace(20, 2, len(timestamps)),
            "power": np.linspace(250, 180, len(timestamps)),
            "observed": True,
        }
    )

    figure = plot_structural_break(grid, reports.e1)

    assert len(figure.axes) == 2
    assert (
        len(figure.axes[0].lines[0].get_xdata())
        == grid["ts"].dt.tz_convert("Australia/Melbourne").dt.date.nunique()
    )


def test_generate_figures_writes_png_files(tmp_path):
    paths = generate_figures(REPORTS_DIR, tmp_path, None, None)

    assert len(paths) == len(FIGURES)
    assert all(path.suffix == ".png" and path.stat().st_size > 0 for path in paths)
