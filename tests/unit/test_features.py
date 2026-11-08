import numpy as np
import pandas as pd
import pytest

from energy_monitor.common.features import (
    MIN_HISTORY_READINGS,
    ORIGIN_MINUS_24H,
    SEASONAL_REFERENCE,
    add_power_history,
    build_forecast_frame,
    compute_seasonal_scale,
    feature_columns,
    history_columns,
    select_valid_readings,
)
from tests.unit.forecast_helpers import GRID_START, make_grid

STEP = pd.Timedelta(minutes=10)
HOSTS = ["h1", "h2"]


def ramp_grid(slots: int = 400) -> pd.DataFrame:
    grid = make_grid(slots, hosts=("h1",))
    grid["power"] = np.arange(slots, dtype=float) + 100
    return grid


def row_at(frame: pd.DataFrame, slot: int) -> pd.Series:
    return frame[frame["ts"] == GRID_START + slot * STEP].iloc[0]


def test_lags_are_relative_to_origin_and_means_end_at_origin() -> None:
    row = row_at(build_forecast_frame(ramp_grid(), ["h1"], 6), 200)

    assert row["power_lag_0"] == 300
    assert row["power_lag_1"] == 299
    assert row["power_lag_3"] == 297
    assert row["power_lag_6"] == 294
    assert row["power_mean_1h"] == pytest.approx(297.5)
    assert row["target"] == 306
    assert row["target_ts"] == GRID_START + 206 * STEP


def test_seasonal_reference_is_relative_to_target() -> None:
    row = row_at(build_forecast_frame(ramp_grid(), ["h1"], 6), 200)

    assert row[SEASONAL_REFERENCE] == 300 + 6 - 144


def test_daily_horizon_does_not_duplicate_current_power() -> None:
    frame = build_forecast_frame(ramp_grid(), ["h1"], 144)
    row = row_at(frame, 200)

    columns = history_columns(144)
    assert SEASONAL_REFERENCE not in columns
    assert ORIGIN_MINUS_24H in columns
    assert row[SEASONAL_REFERENCE] == row["power_lag_0"]
    assert row[ORIGIN_MINUS_24H] == 300 - 144


def test_short_horizon_uses_seasonal_reference_as_feature() -> None:
    assert SEASONAL_REFERENCE in history_columns(6)
    assert ORIGIN_MINUS_24H not in history_columns(6)


def test_load_variant_only_adds_load_columns() -> None:
    added = set(feature_columns(6, True)) - set(feature_columns(6, False))

    assert added == {"cpu_load", "mem_pct", "n_vms", "cores", "cores_used", "net_rx", "net_tx"}


def test_changing_future_values_does_not_change_features_of_earlier_origins() -> None:
    columns = feature_columns(6, True)
    grid = make_grid()
    original = build_forecast_frame(grid, HOSTS, 6)
    changed = grid.copy()
    future = changed["ts"] > GRID_START + 300 * STEP
    changed.loc[future, ["power", "cpu_load", "net_rx"]] = 9999.0

    modified = build_forecast_frame(changed, HOSTS, 6)

    limit = GRID_START + 300 * STEP
    pd.testing.assert_frame_equal(
        original.loc[original["ts"] <= limit, columns].reset_index(drop=True),
        modified.loc[modified["ts"] <= limit, columns].reset_index(drop=True),
    )


def test_calendar_uses_melbourne_time_of_target() -> None:
    grid = make_grid(400)

    row = row_at(build_forecast_frame(grid, HOSTS, 6), 200)

    target_local = (GRID_START + 206 * STEP).tz_convert("Australia/Melbourne")
    assert row["target_hour"] == target_local.hour
    assert row["target_day_of_week"] == target_local.day_of_week


def test_origins_without_daily_history_are_left_out() -> None:
    frame = build_forecast_frame(ramp_grid(), ["h1"], 6)

    assert frame["ts"].min() == GRID_START + 138 * STEP


def test_flagged_reading_is_treated_as_missing() -> None:
    grid = make_grid(400, hosts=("h1",))
    grid.loc[200, "quality_mask"] = 2

    frame = build_forecast_frame(grid, ["h1"], 6)

    assert GRID_START + 200 * STEP not in set(frame["ts"])


def test_origin_is_dropped_when_recent_history_is_below_minimum() -> None:
    grid = make_grid(400, hosts=("h1",))
    missing = 144 - MIN_HISTORY_READINGS + 1
    grid.loc[150 : 150 + missing - 1, ["power"]] = np.nan
    grid.loc[150 : 150 + missing - 1, ["observed"]] = False

    frame = build_forecast_frame(grid, ["h1"], 6)

    assert GRID_START + 200 * STEP not in set(frame["ts"])


def test_origin_is_kept_when_recent_history_reaches_minimum() -> None:
    grid = make_grid(400, hosts=("h1",))
    missing = 144 - MIN_HISTORY_READINGS
    grid.loc[150 : 150 + missing - 1, ["power"]] = np.nan
    grid.loc[150 : 150 + missing - 1, ["observed"]] = False

    frame = build_forecast_frame(grid, ["h1"], 6)

    assert GRID_START + 200 * STEP in set(frame["ts"])


@pytest.mark.parametrize(("last_missing", "is_available"), [(198, True), (199, False)])
def test_rolling_mean_requires_half_of_the_window(last_missing: int, is_available: bool) -> None:
    grid = make_grid(400, hosts=("h1",))
    grid.loc[196:last_missing, "power"] = np.nan
    grid.loc[196:last_missing, "observed"] = False

    frame = add_power_history(select_valid_readings(grid, ["h1"]), 6)

    assert pd.notna(frame.loc[200, "power_mean_1h"]) is is_available


def test_origin_with_missing_load_is_dropped_for_all_variants() -> None:
    grid = make_grid(400, hosts=("h1",))
    grid.loc[200, "cpu_load"] = np.nan

    frame = build_forecast_frame(grid, ["h1"], 6)

    assert GRID_START + 200 * STEP not in set(frame["ts"])


def test_host_categories_cover_every_eligible_host() -> None:
    frame = build_forecast_frame(make_grid(), HOSTS, 6)

    assert list(frame["host_id"].cat.categories) == HOSTS


def test_seasonal_scale_is_computed_by_hand() -> None:
    grid = make_grid(300, hosts=("h1", "h2"))
    grid["power"] = np.where(grid["host_id"].eq("h1"), np.arange(600) % 300 * 1.0, 50.0)
    start, end = GRID_START + 144 * STEP, GRID_START + 200 * STEP

    per_host, total = compute_seasonal_scale(grid, HOSTS, start, end)

    assert per_host["h1"] == pytest.approx(144.0)
    assert per_host["h2"] == pytest.approx(0.0)
    assert total == pytest.approx(144.0)


def test_total_scale_uses_only_hosts_valid_at_both_instants() -> None:
    grid = make_grid(300, hosts=("h1", "h2"))
    grid["power"] = 100.0
    grid.loc[grid["host_id"].eq("h2") & grid["ts"].eq(GRID_START + 150 * STEP), "power"] = np.nan
    grid.loc[grid["host_id"].eq("h2") & grid["ts"].eq(GRID_START + 150 * STEP), "observed"] = False

    _, total = compute_seasonal_scale(grid, HOSTS, GRID_START + 144 * STEP, GRID_START + 200 * STEP)

    assert total == pytest.approx(0.0)
