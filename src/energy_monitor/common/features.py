import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from energy_monitor.common.preparation import is_valid_reading
from energy_monitor.common.schema import GRID_STEP, SOURCE_TIMEZONE

STEPS_PER_DAY: Final = 144
LAG_STEPS: Final = (0, 1, 2, 3, 6)
MEAN_1H_STEPS: Final = 6
MEAN_6H_STEPS: Final = 36
MIN_HISTORY_READINGS: Final = math.ceil(0.8 * STEPS_PER_DAY)
LOAD_FEATURES: Final = ("cpu_load", "mem_pct", "n_vms", "cores", "cores_used", "net_rx", "net_tx")
CALENDAR_FEATURES: Final = ("target_hour", "target_day_of_week")
SEASONAL_REFERENCE: Final = "power_target_minus_24h"
ORIGIN_MINUS_24H: Final = "power_origin_minus_24h"
GRID_COLUMNS: Final = ("host_id", "ts", "power", "observed", "quality_mask", *LOAD_FEATURES)


def lag_column(step: int) -> str:
    return f"power_lag_{step}"


def history_columns(horizon_steps: int) -> list[str]:
    columns = [lag_column(step) for step in LAG_STEPS] + ["power_mean_1h", "power_mean_6h"]
    daily_lag = ORIGIN_MINUS_24H if horizon_steps == STEPS_PER_DAY else SEASONAL_REFERENCE
    return [*columns, daily_lag]


def feature_columns(horizon_steps: int, with_load: bool) -> list[str]:
    columns = [*history_columns(horizon_steps), *CALENDAR_FEATURES, "host_id"]
    return [*columns, *LOAD_FEATURES] if with_load else columns


def required_columns(horizon_steps: int) -> list[str]:
    return sorted({*feature_columns(horizon_steps, with_load=True), SEASONAL_REFERENCE})


def select_valid_readings(grid: pd.DataFrame, hosts: Sequence[str]) -> pd.DataFrame:
    frame = grid.loc[grid["host_id"].isin(hosts), list(GRID_COLUMNS)].copy()
    frame = frame.sort_values(["host_id", "ts"], kind="stable").reset_index(drop=True)
    measurements = ["power", *LOAD_FEATURES]
    frame.loc[~is_valid_reading(frame), measurements] = np.nan
    return frame[["host_id", "ts", *measurements]]


def grouped_rolling(
    values: pd.Series, hosts: pd.Series, window: int, minimum: int, statistic: str
) -> pd.Series:
    rolling = values.groupby(hosts, sort=False).rolling(window, min_periods=minimum)
    return getattr(rolling, statistic)().reset_index(level=0, drop=True).reindex(values.index)


def add_power_history(frame: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    hosts = frame["host_id"]
    power = frame.groupby("host_id", sort=False)["power"]
    new_columns = {lag_column(step): power.shift(step) for step in LAG_STEPS}
    new_columns["power_mean_1h"] = grouped_rolling(
        frame["power"], hosts, MEAN_1H_STEPS, MEAN_1H_STEPS // 2, "mean"
    )
    new_columns["power_mean_6h"] = grouped_rolling(
        frame["power"], hosts, MEAN_6H_STEPS, MEAN_6H_STEPS // 2, "mean"
    )
    new_columns[SEASONAL_REFERENCE] = power.shift(STEPS_PER_DAY - horizon_steps)
    new_columns[ORIGIN_MINUS_24H] = power.shift(STEPS_PER_DAY)
    observed = frame["power"].notna().astype(np.float64)
    new_columns["readings_last_24h"] = grouped_rolling(observed, hosts, STEPS_PER_DAY, 1, "sum")
    new_columns["target"] = power.shift(-horizon_steps)
    return frame.assign(**new_columns)


def add_target_calendar(
    frame: pd.DataFrame, horizon_steps: int, timezone: str = SOURCE_TIMEZONE
) -> pd.DataFrame:
    target_ts = frame["ts"] + horizon_steps * GRID_STEP
    local_target = target_ts.dt.tz_convert(timezone)
    return frame.assign(
        target_ts=target_ts,
        target_hour=local_target.dt.hour.astype(np.int8),
        target_day_of_week=local_target.dt.day_of_week.astype(np.int8),
    )


def has_sufficient_history(frame: pd.DataFrame) -> pd.Series:
    return (
        frame[lag_column(0)].notna()
        & frame[SEASONAL_REFERENCE].notna()
        & frame["readings_last_24h"].ge(MIN_HISTORY_READINGS)
    )


def reduce_precision(frame: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    numeric = [
        column
        for column in feature_columns(horizon_steps, with_load=True)
        if column not in {"host_id", *CALENDAR_FEATURES}
    ]
    numeric = sorted({*numeric, SEASONAL_REFERENCE})
    return frame.assign(**{column: frame[column].astype(np.float32) for column in numeric})


def build_forecast_frame(
    grid: pd.DataFrame,
    hosts: Sequence[str],
    horizon_steps: int,
    timezone: str = SOURCE_TIMEZONE,
) -> pd.DataFrame:
    frame = add_power_history(select_valid_readings(grid, hosts), horizon_steps)
    frame = add_target_calendar(frame, horizon_steps, timezone)
    required = required_columns(horizon_steps)
    complete = has_sufficient_history(frame) & frame[required].notna().all(axis=1)
    kept = ["host_id", "ts", "target_ts", "target", *required]
    frame = frame.loc[complete, list(dict.fromkeys(kept))].reset_index(drop=True)
    frame["host_id"] = pd.Categorical(frame["host_id"], categories=sorted(hosts))
    return reduce_precision(frame, horizon_steps)


def compute_seasonal_scale(
    grid: pd.DataFrame, hosts: Sequence[str], start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.Series, float]:
    frame = select_valid_readings(grid, hosts)[["host_id", "ts", "power"]]
    frame["previous"] = frame.groupby("host_id", sort=False)["power"].shift(STEPS_PER_DAY)
    window = frame[(frame["ts"] >= start) & (frame["ts"] < end)].dropna(
        subset=["power", "previous"]
    )
    per_host = (window["power"] - window["previous"]).abs().groupby(window["host_id"]).mean()
    per_instant = window.groupby("ts")[["power", "previous"]].sum()
    total = float((per_instant["power"] - per_instant["previous"]).abs().mean())
    return per_host, total
