from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from energy_monitor.common.features import LOAD_FEATURES, select_valid_readings
from energy_monitor.common.schema import SOURCE_TIMEZONE

LOCAL_CALENDAR: Final = ("hour", "day_of_week")


def expected_feature_columns() -> list[str]:
    return [*LOAD_FEATURES, *LOCAL_CALENDAR, "host_id"]


def add_local_calendar(frame: pd.DataFrame, timezone: str = SOURCE_TIMEZONE) -> pd.DataFrame:
    local = frame["ts"].dt.tz_convert(timezone)
    return frame.assign(
        hour=local.dt.hour.astype(np.int8), day_of_week=local.dt.day_of_week.astype(np.int8)
    )


def prepare_expected_rows(
    frame: pd.DataFrame, hosts: Sequence[str], timezone: str = SOURCE_TIMEZONE
) -> pd.DataFrame:
    complete = frame[["power", *LOAD_FEATURES]].notna().all(axis=1)
    rows = add_local_calendar(frame.loc[complete].reset_index(drop=True), timezone)
    rows["host_id"] = pd.Categorical(rows["host_id"].astype(str), categories=sorted(hosts))
    rows["power"] = rows["power"].astype(np.float64)
    return rows.astype({column: np.float32 for column in LOAD_FEATURES})


def build_expected_frame(
    grid: pd.DataFrame, hosts: Sequence[str], timezone: str = SOURCE_TIMEZONE
) -> pd.DataFrame:
    return prepare_expected_rows(select_valid_readings(grid, hosts), hosts, timezone)
