from typing import Any

import pandas as pd

from energy_monitor.common.preparation import PreparedDataset, is_valid_reading
from energy_monitor.common.schema import (
    GRID_STEP,
    NOT_REPORTED_AS_MISSING,
    SOURCE_TIMEZONE,
    QualityFlag,
)
from energy_monitor.common.splitting import (
    PARTITIONS,
    SplitBoundaries,
    compute_host_coverage,
    select_eligible_hosts,
)

LONG_GAP = pd.Timedelta(hours=6)
SENSOR_FLOOR_INLET = 4.0
SENSOR_FLOOR_FAN = 280.0


def format_timestamp(value: pd.Timestamp, timezone: str = "UTC") -> str:
    return value.tz_convert(timezone).isoformat()


def summarize_raw(standardized: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(standardized)),
        "hosts": int(standardized["host_id"].nunique()),
        "first_utc": format_timestamp(standardized["ts"].min()),
        "last_utc": format_timestamp(standardized["ts"].max()),
    }


def summarize_missing(standardized: pd.DataFrame) -> dict[str, dict[str, float]]:
    rows = len(standardized)
    summary = {}
    for column in standardized.columns.difference(sorted(NOT_REPORTED_AS_MISSING)):
        count = int(standardized[column].isna().sum())
        summary[column] = {"count": count, "percent": round(count / rows * 100, 4)}
    return summary


def summarize_flags(standardized: pd.DataFrame) -> dict[str, int]:
    mask = standardized["quality_mask"].to_numpy()
    counts = {flag.name.lower(): int(((mask & int(flag)) != 0).sum()) for flag in QualityFlag}
    counts["any_flag"] = int((mask != 0).sum())
    return counts


def count_repeated_local_times(standardized: pd.DataFrame) -> int:
    subset = standardized.dropna(subset=["ts_local"])
    return int(subset.duplicated(["host_id", "ts_local"], keep="first").sum())


def summarize_gaps(grid: pd.DataFrame) -> dict[str, Any]:
    observed = grid.loc[grid["observed"], ["host_id", "ts"]].sort_values(["host_id", "ts"])
    steps = observed.groupby("host_id")["ts"].diff()
    gaps = observed.assign(step=steps).loc[lambda frame: frame["step"] > GRID_STEP]
    long_gaps = gaps[gaps["step"] >= LONG_GAP]
    return {
        "unobserved_grid_slots": int((~grid["observed"]).sum()),
        "grid_slots": int(len(grid)),
        "gaps_between_readings": int(len(gaps)),
        "gaps_of_6h_or_more": int(len(long_gaps)),
        "hosts_with_6h_gap": int(long_gaps["host_id"].nunique()),
        "longest_gap_hours": float(gaps["step"].max() / pd.Timedelta(hours=1)) if len(gaps) else 0,
    }


def summarize_inconsistencies(standardized: pd.DataFrame) -> dict[str, int]:
    summary = {
        "cores_used_above_cores": int((standardized["cores_used"] > standardized["cores"]).sum()),
        "memory_or_vm_metrics_missing": int(standardized["ram"].isna().sum()),
    }
    if {"inlet_temp", "fan_speed1"} <= set(standardized.columns):
        at_floor = standardized["inlet_temp"].eq(SENSOR_FLOOR_INLET) & standardized[
            "fan_speed1"
        ].eq(SENSOR_FLOOR_FAN)
        summary["sensor_floor_readings"] = int(at_floor.sum())
    return summary


def summarize_core_changes(standardized: pd.DataFrame) -> dict[str, int]:
    ordered = standardized.dropna(subset=["cores", "ts"]).sort_values(
        ["host_id", "ts"], kind="stable"
    )
    changed = ordered.groupby("host_id")["cores"].diff().fillna(0).ne(0)
    dates = ordered.loc[changed, "ts"].dt.tz_convert(SOURCE_TIMEZONE).dt.strftime("%Y-%m-%d")
    return {date: int(count) for date, count in dates.value_counts().sort_index().items()}


def summarize_split(boundaries: SplitBoundaries) -> dict[str, dict[str, str]]:
    moments = {
        "start": boundaries.start,
        "train_end": boundaries.train_end,
        "validation_end": boundaries.validation_end,
        "end": boundaries.end,
    }
    return {
        name: {
            "utc": format_timestamp(moment),
            "local": format_timestamp(moment, SOURCE_TIMEZONE),
        }
        for name, moment in moments.items()
    }


def summarize_coverage(
    grid: pd.DataFrame, boundaries: SplitBoundaries, min_coverage: float
) -> dict[str, Any]:
    coverage = compute_host_coverage(grid, boundaries)
    eligible = select_eligible_hosts(coverage, min_coverage)
    return {
        "min_coverage": min_coverage,
        "eligible_hosts": eligible,
        "eligible_count": len(eligible),
        "excluded_hosts": sorted(set(coverage.index) - set(eligible)),
        "per_host": coverage.round(4).to_dict(orient="index"),
        "partitions": list(PARTITIONS),
    }


def build_quality_report(
    prepared: PreparedDataset, boundaries: SplitBoundaries, min_coverage: float
) -> dict[str, Any]:
    standardized, grid = prepared.standardized, prepared.grid
    flags = summarize_flags(standardized)
    return {
        "raw": summarize_raw(standardized),
        "missing_by_column": summarize_missing(standardized),
        "quality_flags": flags,
        "repeated_local_times": count_repeated_local_times(standardized),
        "true_duplicates_after_utc": flags["duplicate"],
        "valid_grid_readings": int(is_valid_reading(grid).sum()),
        "gaps": summarize_gaps(grid),
        "inconsistencies_without_flag": summarize_inconsistencies(standardized),
        "core_count_changes_by_local_date": summarize_core_changes(standardized),
        "split": summarize_split(boundaries),
        "coverage": summarize_coverage(grid, boundaries, min_coverage),
    }
