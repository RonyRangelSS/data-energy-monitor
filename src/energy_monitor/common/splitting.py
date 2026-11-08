from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from energy_monitor.common.preparation import is_valid_reading
from energy_monitor.common.schema import GRID_FREQUENCY, GRID_STEP

PRE_ANALYSIS: Final = "pre_analysis"
TRAIN: Final = "train"
VALIDATION: Final = "validation"
TEST: Final = "test"
PARTITIONS: Final = (TRAIN, VALIDATION, TEST)


@dataclass(frozen=True)
class SplitBoundaries:
    start: pd.Timestamp
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    end: pd.Timestamp


def local_to_utc(local_time: str, timezone: str) -> pd.Timestamp:
    return pd.Timestamp(local_time).tz_localize(timezone).tz_convert("UTC")


def compute_boundaries(
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
    analysis_start: pd.Timestamp | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> SplitBoundaries:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("As frações de treino e validação devem ser positivas")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Treino e validação devem deixar uma parte para o teste")
    requested = grid_start if analysis_start is None else max(analysis_start, grid_start)
    start = requested.ceil(GRID_FREQUENCY)
    if start > grid_end:
        raise ValueError("O início da análise é posterior ao fim da grade")
    slot_count = int((grid_end - start) / GRID_STEP) + 1
    train_slots = int(slot_count * train_fraction)
    validation_slots = int(slot_count * validation_fraction)
    return SplitBoundaries(
        start=start,
        train_end=start + train_slots * GRID_STEP,
        validation_end=start + (train_slots + validation_slots) * GRID_STEP,
        end=grid_end,
    )


def assign_partition(timestamps: pd.Series, boundaries: SplitBoundaries) -> pd.Series:
    conditions = [
        timestamps < boundaries.start,
        timestamps < boundaries.train_end,
        timestamps < boundaries.validation_end,
    ]
    labels = np.select(conditions, [PRE_ANALYSIS, TRAIN, VALIDATION], default=TEST)
    return pd.Series(labels, index=timestamps.index, name="partition")


def compute_host_coverage(grid: pd.DataFrame, boundaries: SplitBoundaries) -> pd.DataFrame:
    partition = assign_partition(grid["ts"], boundaries)
    frame = pd.DataFrame(
        {"host_id": grid["host_id"], "partition": partition, "valid": is_valid_reading(grid)}
    )
    frame = frame[frame["partition"].isin(PARTITIONS)]
    coverage = frame.groupby(["host_id", "partition"], observed=True)["valid"].mean().unstack()
    return coverage.reindex(columns=list(PARTITIONS)).fillna(0.0)


def select_eligible_hosts(coverage: pd.DataFrame, min_coverage: float) -> list[str]:
    return sorted(coverage.index[coverage.min(axis=1) >= min_coverage])
