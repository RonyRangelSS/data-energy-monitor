import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
import pytest
import yaml

from energy_monitor.common.schema import SOURCE_TIMEZONE
from energy_monitor.common.splitting import compute_boundaries
from energy_monitor.experiments import protocol_freeze
from energy_monitor.experiments.e5_forecasting import main, run_final, run_selection
from energy_monitor.experiments.protocol_freeze import ProtocolNotFrozenError, freeze_protocol
from energy_monitor.experiments.run_record import write_json
from energy_monitor.models.base import ForecastModel
from energy_monitor.models.forecast import forecast_registry
from tests.unit.forecast_helpers import make_grid

SLOTS = 144 * 30


def build_workspace(root: Path, grid: pd.DataFrame | None = None) -> dict[str, Any]:
    frame = make_grid(SLOTS) if grid is None else grid
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed / "d1_grid.parquet", index=False)
    (root / "raw.csv").write_text("x\n1\n", encoding="utf-8")
    start = frame["ts"].min()
    boundaries = compute_boundaries(start, frame["ts"].max(), start + pd.Timedelta(days=4))
    split = {
        name: {"utc": getattr(boundaries, name).isoformat()}
        for name in ("start", "train_end", "validation_end", "end")
    }
    metrics = {"split": split, "coverage": {"eligible_hosts": ["h1", "h2"]}}
    write_json({"metrics": metrics}, root / "artifacts/e1_data_quality_d1.json")
    return {
        "experiment": "e5_forecasting",
        "seed": 42,
        "source_timezone": SOURCE_TIMEZONE,
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "processed_dir": str(processed),
        "e1_report_dir": str(root / "artifacts"),
        "output_dir": str(root / "artifacts"),
        "models_dir": str(root / "models"),
        "datasets": {"d1": str(root / "raw.csv")},
        "horizons": {"1h": 6, "24h": 144},
        "training_start_candidates_local": ["2019-03-05T11:00:00", None],
        "motivation_full_period": True,
        "default_hyperparameters": {"lightgbm": {"n_estimators": 5, "num_leaves": 7}},
        "hyperparameter_grid": {"lightgbm": {"num_leaves": [7, 15]}},
    }


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    return build_workspace(tmp_path)


@pytest.fixture
def selection_path(config: dict[str, Any]) -> Path:
    return run_selection("d1", config)


def freeze(config: dict[str, Any], paths: list[Path], monkeypatch: pytest.MonkeyPatch) -> None:
    clean = {"commit": "abc", "has_uncommitted_changes": False}
    monkeypatch.setattr(protocol_freeze, "describe_code_version", lambda: clean)
    freeze_protocol(paths, Path(config["output_dir"]) / protocol_freeze.FREEZE_FILE_NAME)


def read_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def test_selection_compares_every_registered_model_and_variant(selection_path: Path) -> None:
    horizon = read_metrics(selection_path)["horizons"]["1h"]

    assert set(horizon["models"]) == set(forecast_registry.names())
    assert set(horizon["models"]["seasonal_naive"]) == {"history_only"}
    assert set(horizon["models"]["lightgbm"]) == {"with_load", "history_only"}


def test_selection_chooses_window_and_hyperparameters_by_validation(selection_path: Path) -> None:
    lightgbm = read_metrics(selection_path)["horizons"]["1h"]["models"]["lightgbm"]["with_load"]

    assert len(lightgbm["window_search"]) == 2
    best = min(item["validation_mae"] for item in lightgbm["window_search"])
    assert lightgbm["chosen_window"]["validation_mae"] == best
    assert len(lightgbm["hyperparameter_search"]) == 2
    assert 0 <= lightgbm["interval_coverage_5_95"] <= 1


def test_selection_reports_operational_choice_and_load_effect(selection_path: Path) -> None:
    horizon = read_metrics(selection_path)["horizons"]["24h"]

    assert horizon["operational_choice"]["model"] in forecast_registry.names()
    assert set(horizon["load_effect_percent_validation"]) == {"lightgbm", "linear_regression"}
    assert "with_load" in horizon["motivation_full_period"]


def test_selection_is_not_affected_by_test_period(tmp_path: Path) -> None:
    grid = make_grid(SLOTS)
    first = build_workspace(tmp_path / "first", grid)
    original = read_metrics(run_selection("d1", first))
    limit = pd.Timestamp(
        json.loads(
            (tmp_path / "first/artifacts/e1_data_quality_d1.json").read_text(encoding="utf-8")
        )["metrics"]["split"]["validation_end"]["utc"]
    )
    changed = grid.copy()
    changed.loc[changed["ts"] >= limit, ["power", "cpu_load"]] = 9999.0
    second = build_workspace(tmp_path / "second", changed)

    assert read_metrics(run_selection("d1", second)) == original


def test_registered_fake_model_appears_in_selection(tmp_path: Path) -> None:
    class FakeForecaster(ForecastModel):
        name: ClassVar[str] = "fake_test_model"
        version: ClassVar[str] = "0.0.1"

        def fit(self, frame: pd.DataFrame, target: pd.Series) -> Self:
            self.mean_ = float(target.mean())
            return self

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(frame), self.mean_)

    forecast_registry.register(FakeForecaster)
    try:
        metrics = read_metrics(run_selection("d1", build_workspace(tmp_path)))
    finally:
        forecast_registry.unregister("fake_test_model")

    assert "fake_test_model" in metrics["horizons"]["1h"]["models"]


def test_final_stage_is_blocked_without_freeze(
    config: dict[str, Any], selection_path: Path
) -> None:
    with pytest.raises(ProtocolNotFrozenError, match="não congelado"):
        run_final("d1", config)


def test_final_stage_is_blocked_when_selection_changes_after_freeze(
    config: dict[str, Any], selection_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze(config, [selection_path], monkeypatch)
    selection_path.write_text('{"metrics": {}}\n', encoding="utf-8")

    with pytest.raises(ProtocolNotFrozenError, match="mudou"):
        run_final("d1", config)


@pytest.fixture
def final_result(
    config: dict[str, Any], selection_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, Any]]:
    freeze(config, [selection_path], monkeypatch)
    yield read_metrics(run_final("d1", config))


def test_final_stage_reports_details_and_comparisons(final_result: dict[str, Any]) -> None:
    horizon = final_result["horizons"]["1h"]
    lightgbm = horizon["models"]["lightgbm"]["with_load"]

    assert set(lightgbm["per_host"]) == {"h1", "h2"}
    assert lightgbm["weekly_blocks"]
    assert 0 <= lightgbm["interval_coverage_5_95"] <= 1
    assert "interval_coverage_5_95" not in horizon["models"]["linear_regression"]["with_load"]
    assert "lightgbm_with_load_vs_history_only" in horizon["comparisons"]
    assert "lightgbm_vs_linear_with_load" in horizon["comparisons"]
    corrected = [item for item in horizon["comparisons"].values() if item["p_value"] is not None]
    assert all(item["p_value_holm"] >= item["p_value"] for item in corrected)


def test_final_stage_saves_traceable_models(
    final_result: dict[str, Any], config: dict[str, Any]
) -> None:
    models_dir = Path(config["models_dir"])
    metadata_path = models_dir / "d1_lightgbm_1h_with_load_1.0.0.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert (models_dir / "d1_lightgbm_1h_with_load_1.0.0.joblib").exists()
    assert metadata["version"] == "1.0.0"
    assert metadata["horizon"] == "1h"
    assert metadata["feature_columns"]
    assert metadata["training_end_utc"]


def test_freeze_requires_committed_code(
    config: dict[str, Any], selection_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dirty = {"commit": "abc", "has_uncommitted_changes": True}
    monkeypatch.setattr(protocol_freeze, "describe_code_version", lambda: dirty)

    with pytest.raises(RuntimeError, match="commit"):
        freeze_protocol([selection_path], Path(config["output_dir"]) / "freeze.json")


def test_freeze_requires_existing_selection(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        freeze_protocol([tmp_path / "missing.json"], tmp_path / "freeze.json")


def test_command_runs_selection_for_one_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = build_workspace(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    main(["--stage", "selection", "--config", str(config_path), "--dataset", "d1"])

    assert (tmp_path / "artifacts/e5_forecasting_selection_d1.json").exists()
    assert "selection d1 6 lightgbm with_load" in capsys.readouterr().out
