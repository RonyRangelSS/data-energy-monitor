from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_monitor.common.expected_features import (
    build_expected_frame,
    expected_feature_columns,
)
from energy_monitor.models.base import ExpectedConsumptionModel
from energy_monitor.models.expected import expected_registry
from energy_monitor.models.expected.residuals import (
    ResidualStatistics,
    compute_residual_statistics,
    contiguous_time_folds,
    out_of_fold_predictions,
    standardized_deviation,
)
from tests.unit.detection_helpers import make_expected_rows
from tests.unit.forecast_helpers import make_grid


def create(name: str, **hyperparameters: object) -> ExpectedConsumptionModel:
    return expected_registry.create(name, expected_feature_columns(), 42, **hyperparameters)


def test_minimum_expected_models_are_registered() -> None:
    assert set(expected_registry.names()) == {
        "cpu_bin_mean",
        "linear_regression",
        "lightgbm_quantile",
    }


def test_expected_frame_uses_same_instant_load_and_melbourne_calendar() -> None:
    grid = make_grid(200)
    grid.loc[5, "cpu_load"] = np.nan

    frame = build_expected_frame(grid, ["h1", "h2"])

    assert len(frame) == 399
    local = frame["ts"].iloc[0].tz_convert("Australia/Melbourne")
    assert frame["hour"].iloc[0] == local.hour
    assert "power_lag_1" not in frame.columns


def test_cpu_bin_mean_uses_host_bin_then_falls_back() -> None:
    rows = make_expected_rows()
    model = create("cpu_bin_mean").fit(rows, rows["power"])
    unseen = rows.head(1).copy()
    unseen["host_id"] = pd.Categorical(["h3"], categories=["h1", "h2", "h3"])

    lower, upper = model.predict_interval(rows)

    assert np.all(lower <= upper)
    assert np.isfinite(model.predict(unseen)).all()


def test_linear_expected_model_recovers_linear_relation() -> None:
    rows = make_expected_rows()

    predicted = create("linear_regression").fit(rows, rows["power"]).predict(rows)

    np.testing.assert_allclose(predicted, rows["power"], atol=1e-2)


def test_quantile_lightgbm_orders_median_and_interval() -> None:
    rows = make_expected_rows()
    model = create("lightgbm_quantile", n_estimators=30, num_leaves=7, with_interval=True)

    model.fit(rows, rows["power"])
    lower, upper = model.predict_interval(rows)

    assert np.mean(lower <= upper) > 0.99
    assert np.mean(np.abs(model.predict(rows) - rows["power"])) < rows["power"].std()


def test_quantile_lightgbm_without_interval_refuses_to_report_one() -> None:
    rows = make_expected_rows()
    model = create("lightgbm_quantile", n_estimators=5).fit(rows, rows["power"])

    with pytest.raises(ValueError, match="with_interval"):
        model.predict_interval(rows)


def test_expected_model_round_trip(tmp_path: Path) -> None:
    rows = make_expected_rows()
    model = create("cpu_bin_mean").fit(rows, rows["power"])

    model.save(tmp_path / "model.joblib")

    loaded = ExpectedConsumptionModel.load(tmp_path / "model.joblib")
    np.testing.assert_array_equal(loaded.predict(rows), model.predict(rows))


def test_residual_statistics_match_manual_calculation() -> None:
    hosts = pd.Series(["a", "a", "a", "b", "b", "b"])
    residuals = np.array([1.0, 2.0, 10.0, -1.0, 0.0, 1.0])

    statistics = compute_residual_statistics(hosts, residuals)

    assert statistics.median["a"] == 2.0
    assert statistics.mad["a"] == 1.0
    assert statistics.mad["b"] == 1.0


def test_standardized_deviation_uses_scaled_mad() -> None:
    statistics = ResidualStatistics(median=pd.Series({"a": 1.0}), mad=pd.Series({"a": 2.0}))
    frame = pd.DataFrame({"host_id": ["a"], "power": [110.0]})

    z = standardized_deviation(frame, np.array([100.0]), statistics)

    assert z[0] == pytest.approx((110 - 100 - 1) / (1.4826 * 2))


def test_zero_mad_produces_missing_score() -> None:
    statistics = ResidualStatistics(median=pd.Series({"a": 0.0}), mad=pd.Series({"a": 0.0}))
    frame = pd.DataFrame({"host_id": ["a"], "power": [110.0]})

    assert np.isnan(standardized_deviation(frame, np.array([100.0]), statistics)[0])


def test_statistics_round_trip_through_dictionary() -> None:
    statistics = ResidualStatistics(median=pd.Series({"a": 1.5}), mad=pd.Series({"a": 0.5}))

    restored = ResidualStatistics.from_dict(statistics.to_dict())

    assert restored.median["a"] == 1.5
    assert restored.mad["a"] == 0.5


def test_time_folds_are_contiguous_and_balanced() -> None:
    timestamps = pd.Series(pd.date_range("2019-01-01", periods=10, freq="10min", tz="UTC"))

    folds = contiguous_time_folds(pd.concat([timestamps, timestamps]), 5)

    assert folds[:10].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert folds[10:].tolist() == folds[:10].tolist()


class MeanModel(ExpectedConsumptionModel):
    name = "mean_test"
    version = "0"

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "MeanModel":
        self.mean = float(target.mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.mean)


def test_out_of_fold_predictions_never_use_the_held_out_block() -> None:
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2019-01-01", periods=4, freq="10min", tz="UTC"),
            "power": [0.0, 0.0, 100.0, 100.0],
        }
    )

    predictions = out_of_fold_predictions(MeanModel, frame, np.array([True] * 4), folds=2)

    assert predictions.tolist() == [100.0, 100.0, 0.0, 0.0]
