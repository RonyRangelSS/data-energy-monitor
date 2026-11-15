import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from energy_monitor.common.schema import SOURCE_TIMEZONE
from energy_monitor.common.splitting import compute_boundaries
from energy_monitor.experiments import e2_expected_consumption as e2
from energy_monitor.experiments import e3_detection as e3
from energy_monitor.experiments import protocol_freeze
from energy_monitor.experiments.protocol_freeze import (
    ProtocolNotFrozenError,
    freeze_path_for,
    freeze_protocol,
    selection_paths_for,
)
from energy_monitor.experiments.run_record import write_json
from tests.unit.forecast_helpers import make_grid

SLOTS = 144 * 20


def build_workspace(root: Path) -> dict[str, Any]:
    grid = make_grid(SLOTS)
    grid["power"] = 100 + 3 * grid["cpu_load"] + np.random.default_rng(1).normal(0, 1, len(grid))
    processed = root / "processed"
    processed.mkdir(parents=True)
    grid.to_parquet(processed / "d1_grid.parquet", index=False)
    (root / "raw.csv").write_text("x\n1\n", encoding="utf-8")
    start = grid["ts"].min()
    boundaries = compute_boundaries(start, grid["ts"].max(), start + pd.Timedelta(days=2))
    split = {
        name: {"utc": getattr(boundaries, name).isoformat()}
        for name in ("start", "train_end", "validation_end", "end")
    }
    metrics = {"split": split, "coverage": {"eligible_hosts": ["h1", "h2"]}}
    write_json({"metrics": metrics}, root / "artifacts/e1_data_quality_d1.json")
    return {
        "experiment": "e2_e3",
        "seed": 42,
        "source_timezone": SOURCE_TIMEZONE,
        "processed_dir": str(processed),
        "e1_report_dir": str(root / "artifacts"),
        "output_dir": str(root / "artifacts"),
        "models_dir": str(root / "models"),
        "datasets": {"d1": str(root / "raw.csv")},
        "freeze_file": "freeze_e2_e3.json",
        "selection_files": [
            "e2_expected_consumption_selection_d1.json",
            "e3_detection_selection_d1.json",
        ],
        "training_start_candidates_local": ["2019-03-03T11:00:00", None],
        "statistics_start_local": "2019-03-03T11:00:00",
        "out_of_fold_blocks": 3,
        "operational_expected_model": "lightgbm_quantile",
        "default_hyperparameters": {"lightgbm_quantile": {"n_estimators": 10, "num_leaves": 7}},
        "hyperparameter_grid": {"lightgbm_quantile": {"num_leaves": [7, 15]}},
        "detection": {
            "consecutive": 2,
            "events_per_seed": 6,
            "magnitudes": [0.2, 0.4],
            "durations_slots": [3, 6],
            "validation_seeds": [1, 2],
            "test_seeds": [3, 4],
            "tuning_kinds": ["offset", "zero_load_high_power"],
            "families": {
                "peaks": ["legit_peak"],
                "excess": ["offset", "drift", "stuck"],
            },
            "grids": {
                "deviation": [
                    {"open_threshold": 3.0, "close_threshold": 2.0},
                    {"open_threshold": 3.5, "close_threshold": 2.5},
                ],
                "fixed_threshold": [{"percentile": 95.0}],
                "isolation_forest": [{"percentile": 99.0}],
            },
            "top_episodes": 3,
        },
    }


def read_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    config = build_workspace(tmp_path_factory.mktemp("e2e3"))
    e2_path = e2.run_selection("d1", config)
    e3_path = e3.run_selection("d1", config)
    return {"config": config, "e2": e2_path, "e3": e3_path}


def test_e2_selection_compares_all_expected_models(workspace: dict[str, Any]) -> None:
    metrics = read_metrics(workspace["e2"])

    assert set(metrics["models"]) == {"cpu_bin_mean", "linear_regression", "lightgbm_quantile"}
    lightgbm = metrics["models"]["lightgbm_quantile"]
    assert len(lightgbm["window_search"]) == 2
    assert 0 <= lightgbm["validation"]["coverage_5_95"] <= 1
    assert "coverage_5_95" not in metrics["models"]["linear_regression"]["validation"]


def test_e2_selection_compares_out_of_fold_and_in_sample_statistics(
    workspace: dict[str, Any],
) -> None:
    rates = read_metrics(workspace["e2"])["residual_statistics_validation"]

    assert set(rates) == {"out_of_fold", "in_sample"}
    assert rates["out_of_fold"]["median_host_mad"] >= rates["in_sample"]["median_host_mad"]


def test_e3_selection_tunes_every_detector(workspace: dict[str, Any]) -> None:
    tuning = read_metrics(workspace["e3"])["tuning"]

    assert set(tuning) == {"deviation", "fixed_threshold", "isolation_forest"}
    assert len(tuning["deviation"]["grid"]) == 2
    assert len(tuning["deviation"]["grid"][0]["f1_by_seed"]) == 2


def test_final_stages_require_freeze(workspace: dict[str, Any]) -> None:
    with pytest.raises(ProtocolNotFrozenError):
        e2.run_final("d1", workspace["config"])
    with pytest.raises(ProtocolNotFrozenError):
        e3.run_final("d1", workspace["config"])


def test_final_stages_report_test_results(
    workspace: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = workspace["config"]
    clean = {"commit": "abc", "has_uncommitted_changes": False}
    monkeypatch.setattr(protocol_freeze, "describe_code_version", lambda: clean)
    freeze_protocol(selection_paths_for(config), freeze_path_for(config))

    e2_metrics = read_metrics(e2.run_final("d1", config))
    e3_metrics = read_metrics(e3.run_final("d1", config))

    assert "lightgbm_quantile_vs_cpu_bin_mean" in e2_metrics["comparisons"]
    assert set(e2_metrics["models"]["lightgbm_quantile"]["per_host_mae"]) == {"h1", "h2"}
    families = e3_metrics["families"]
    assert families["peaks"]["summary"]["deviation"]["false_positive_rate_on_peaks"]["mean"] >= 0
    assert "detection_rate_by_kind" in families["excess"]
    assert "p_value_holm" in families["excess"]["comparisons"]["deviation_vs_fixed_threshold"]
    normal = e3_metrics["normal_operation"]["deviation"]
    assert normal["alert_episodes_per_host_day"] >= 0
    assert len(normal["strongest_episodes"]) <= 3
