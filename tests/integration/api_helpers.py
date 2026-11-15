from typing import Any

import numpy as np
import pandas as pd

from energy_monitor.api.operational import ForecastSpec, OperationalModels
from energy_monitor.common.expected_features import build_expected_frame, expected_feature_columns
from energy_monitor.common.features import build_forecast_frame, feature_columns
from energy_monitor.models.detection import detector_registry
from energy_monitor.models.expected import expected_registry
from energy_monitor.models.expected.residuals import compute_residual_statistics
from energy_monitor.models.forecast import forecast_registry
from tests.unit.forecast_helpers import make_grid

HOSTS = ["h1", "h2"]
TRAINING_SLOTS = 600


def training_grid() -> pd.DataFrame:
    grid = make_grid(TRAINING_SLOTS, tuple(HOSTS))
    grid["power"] = 100 + 3 * grid["cpu_load"] + np.where(grid["host_id"].eq("h2"), 40.0, 0.0)
    grid["power"] += np.random.default_rng(5).normal(0, 0.5, len(grid))
    return grid


def build_operational_models() -> OperationalModels:
    grid = training_grid()
    rows = build_expected_frame(grid, HOSTS)
    expected = expected_registry.create(
        "lightgbm_quantile",
        expected_feature_columns(),
        42,
        n_estimators=30,
        num_leaves=7,
        with_interval=True,
    ).fit(rows, rows["power"])
    statistics = compute_residual_statistics(
        rows["host_id"], rows["power"].to_numpy() - expected.predict(rows)
    )
    detector = detector_registry.create(
        "deviation", (), 42, open_threshold=3.5, close_threshold=2.5, consecutive=2
    ).fit_statistics(statistics)
    forecasts = []
    for horizon, steps in (("1h", 6), ("24h", 144)):
        frame = build_forecast_frame(grid, HOSTS, steps).dropna(subset=["target"])
        model = forecast_registry.create(
            "lightgbm",
            feature_columns(steps, True),
            42,
            n_estimators=10,
            num_leaves=7,
            with_interval=True,
        ).fit(frame, frame["target"].astype(float))
        forecasts.append(ForecastSpec(horizon, steps, model, "with_load"))
    return OperationalModels(HOSTS, expected, detector, forecasts)


def readings_payload(grid: pd.DataFrame) -> list[dict[str, Any]]:
    columns = ["power", "cpu_load", "mem_pct", "n_vms", "cores", "cores_used", "net_rx", "net_tx"]
    return [
        {
            "host_id": row["host_id"],
            "ts": row["ts"].isoformat(),
            **{column: float(row[column]) for column in columns},
        }
        for _, row in grid.iterrows()
    ]
