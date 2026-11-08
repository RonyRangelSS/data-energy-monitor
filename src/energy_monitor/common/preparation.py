from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy_monitor.common.schema import (
    COLUMN_MAP,
    COLUMNS_DROPPED_FROM_GRID,
    GRID_FREQUENCY,
    MAX_CPU_LOAD,
    MIN_CPU_LOAD,
    RAW_TIME_FORMAT,
    REQUIRED_RAW_COLUMNS,
    SOURCE_TIMEZONE,
    QualityFlag,
)


@dataclass(frozen=True)
class PreparedDataset:
    standardized: pd.DataFrame
    grid: pd.DataFrame


def read_raw_csv(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    missing = REQUIRED_RAW_COLUMNS - set(header)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes em {path}: {sorted(missing)}")
    columns = [column for column in header if column in COLUMN_MAP]
    return pd.read_csv(path, usecols=columns, low_memory=False)


def compute_memory_percent(ram: pd.Series, ram_used: pd.Series) -> pd.Series:
    return ram_used / ram.where(ram > 0) * 100


def standardize(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.rename(columns=COLUMN_MAP).copy()
    frame["host_id"] = frame["host_id"].str.strip()
    frame["ts_local"] = pd.to_datetime(
        frame["ts_local"], format=RAW_TIME_FORMAT, errors="coerce"
    ).dt.as_unit("us")
    frame["mem_pct"] = compute_memory_percent(frame["ram"], frame["ram_used"])
    return frame


def localize_to_utc(frame: pd.DataFrame, timezone: str = SOURCE_TIMEZONE) -> pd.Series:
    is_first_occurrence = frame.groupby(["host_id", "ts_local"], dropna=False).cumcount().eq(0)
    localized = frame["ts_local"].dt.tz_localize(
        timezone, ambiguous=is_first_occurrence.to_numpy(), nonexistent="NaT"
    )
    return localized.dt.tz_convert("UTC")


def is_off_grid(timestamps: pd.Series) -> pd.Series:
    return timestamps.isna() | timestamps.ne(timestamps.dt.floor(GRID_FREQUENCY))


def flag_quality(frame: pd.DataFrame) -> pd.Series:
    conditions = {
        QualityFlag.DUPLICATE: frame["ts"].notna()
        & frame.duplicated(["host_id", "ts"], keep="first"),
        QualityFlag.NON_POSITIVE_POWER: frame["power"].le(0),
        QualityFlag.CPU_OUT_OF_RANGE: frame["cpu_load"].notna()
        & ~frame["cpu_load"].between(MIN_CPU_LOAD, MAX_CPU_LOAD),
        QualityFlag.MISSING_TARGET: frame["power"].isna(),
        QualityFlag.OFF_GRID: is_off_grid(frame["ts"]),
    }
    mask = np.zeros(len(frame), dtype=np.int32)
    for flag, condition in conditions.items():
        mask[condition.to_numpy()] |= int(flag)
    return pd.Series(mask, index=frame.index, name="quality_mask")


def has_any_flag(mask: pd.Series, flags: QualityFlag) -> pd.Series:
    return (mask.to_numpy() & int(flags)) != 0


def build_regular_grid(frame: pd.DataFrame) -> pd.DataFrame:
    placeable = ~has_any_flag(frame["quality_mask"], QualityFlag.DUPLICATE | QualityFlag.OFF_GRID)
    placed = frame.loc[placeable].drop(columns=list(COLUMNS_DROPPED_FROM_GRID))
    if placed.empty:
        return placed.assign(observed=pd.Series(dtype=bool)).reset_index(drop=True)
    slots = pd.date_range(
        placed["ts"].min(), placed["ts"].max(), freq=GRID_FREQUENCY, name="ts", unit="us"
    )
    hosts = sorted(placed["host_id"].unique())
    index = pd.MultiIndex.from_product([hosts, slots], names=["host_id", "ts"])
    grid = placed.set_index(["host_id", "ts"]).reindex(index).reset_index()
    grid["observed"] = grid["quality_mask"].notna()
    grid["quality_mask"] = grid["quality_mask"].fillna(0).astype(np.int32)
    return grid


def is_valid_reading(grid: pd.DataFrame) -> pd.Series:
    return grid["observed"] & grid["quality_mask"].eq(0)


def prepare_raw_frame(raw: pd.DataFrame, timezone: str = SOURCE_TIMEZONE) -> PreparedDataset:
    standardized = standardize(raw)
    standardized["ts"] = localize_to_utc(standardized, timezone).dt.as_unit("us")
    standardized["quality_mask"] = flag_quality(standardized)
    return PreparedDataset(standardized=standardized, grid=build_regular_grid(standardized))


def prepare_dataset(path: Path, timezone: str = SOURCE_TIMEZONE) -> PreparedDataset:
    return prepare_raw_frame(read_raw_csv(path), timezone)
