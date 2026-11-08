from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pandas as pd
import pytest

from energy_monitor.common.features import SEASONAL_REFERENCE, feature_columns
from energy_monitor.models.base import ForecastModel
from energy_monitor.models.forecast import forecast_registry
from energy_monitor.models.registry import ModelRegistry


def make_frame(rows: int = 1000) -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(3)
    columns = feature_columns(6, True)
    data = {column: generator.normal(100, 10, rows).astype(np.float32) for column in columns}
    frame = pd.DataFrame(data)
    frame["target_hour"] = generator.integers(0, 24, rows).astype(np.int8)
    frame["target_day_of_week"] = generator.integers(0, 7, rows).astype(np.int8)
    frame["host_id"] = pd.Categorical(generator.choice(["h1", "h2"], rows))
    frame[SEASONAL_REFERENCE] = generator.normal(100, 10, rows).astype(np.float32)
    target = 3 * frame["power_lag_0"].astype(float) - frame["cpu_load"].astype(float) + 7
    target = target + frame["host_id"].eq("h2").astype(float) * 20
    return frame, target


class ConstantForecaster(ForecastModel):
    name: ClassVar[str] = "constant_test"
    version: ClassVar[str] = "0.1"

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> Self:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(frame))


def create(name: str, **hyperparameters: object) -> ForecastModel:
    return forecast_registry.create(name, feature_columns(6, True), 42, **hyperparameters)


def test_minimum_models_are_registered() -> None:
    assert {"seasonal_naive", "linear_regression", "lightgbm"} <= set(forecast_registry.names())


def test_seasonal_naive_repeats_value_from_previous_day() -> None:
    frame, target = make_frame()

    predicted = create("seasonal_naive").fit(frame, target).predict(frame)

    np.testing.assert_allclose(predicted, frame[SEASONAL_REFERENCE].astype(float))


def test_linear_regression_recovers_linear_relation() -> None:
    frame, target = make_frame()

    predicted = create("linear_regression").fit(frame, target).predict(frame)

    np.testing.assert_allclose(predicted, target, atol=1e-2)


def test_lightgbm_learns_and_produces_ordered_interval() -> None:
    frame, target = make_frame()
    model = create("lightgbm", n_estimators=30, num_leaves=7, with_interval=True).fit(frame, target)

    lower, upper = model.predict_interval(frame)

    assert np.mean(np.abs(model.predict(frame) - target)) < target.std()
    assert np.mean(lower <= upper) > 0.99


def test_lightgbm_without_interval_refuses_to_report_one() -> None:
    frame, target = make_frame()
    model = create("lightgbm", n_estimators=5).fit(frame, target)

    with pytest.raises(ValueError, match="with_interval"):
        model.predict_interval(frame)


def test_point_forecast_models_do_not_produce_interval() -> None:
    frame, _ = make_frame()

    with pytest.raises(NotImplementedError):
        create("seasonal_naive").predict_interval(frame)


@pytest.mark.parametrize("name", ["seasonal_naive", "linear_regression", "lightgbm"])
def test_saved_and_loaded_model_predicts_identically(name: str, tmp_path: Path) -> None:
    frame, target = make_frame()
    hyperparameters = {"n_estimators": 5} if name == "lightgbm" else {}
    model = create(name, **hyperparameters).fit(frame, target)
    path = tmp_path / "nested" / "model.joblib"

    model.save(path)

    np.testing.assert_array_equal(ForecastModel.load(path).predict(frame), model.predict(frame))


def test_load_rejects_file_of_another_model_type(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    ConstantForecaster(["a"]).save(path)

    with pytest.raises(TypeError):
        create("seasonal_naive").__class__.load(path)


def test_models_declare_name_and_version() -> None:
    for name in forecast_registry.names():
        model_class = forecast_registry.get(name)
        assert model_class.name == name
        assert model_class.version


def test_registry_rejects_duplicates_and_unknown_names() -> None:
    registry: ModelRegistry[ForecastModel] = ModelRegistry("forecast")
    registry.register(ConstantForecaster)

    with pytest.raises(ValueError):
        registry.register(ConstantForecaster)
    with pytest.raises(KeyError):
        registry.get("missing")


def test_registry_creates_and_removes_models() -> None:
    registry: ModelRegistry[ForecastModel] = ModelRegistry("forecast")
    registry.register(ConstantForecaster)

    model = registry.create("constant_test", ["a", "b"], 11, alpha=1)
    registry.unregister("constant_test")

    assert model.feature_columns == ["a", "b"]
    assert model.seed == 11
    assert model.hyperparameters == {"alpha": 1}
    assert registry.names() == []
