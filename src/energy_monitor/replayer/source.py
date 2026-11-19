import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow.parquet as pq

from energy_monitor.common.features import GRID_COLUMNS, select_valid_readings
from energy_monitor.common.splitting import SplitBoundaries
from energy_monitor.replayer.injection import (
    InjectedEvent,
    InjectionPlan,
    apply_events,
    compute_host_reference,
    plan_events,
)
from energy_monitor.replayer.sender import READING_FIELDS

SOURCE_COLUMNS: Final = ("host_id", "ts", "observed", "quality_mask", *READING_FIELDS)


def load_period(
    grid_path: Path, hosts: Sequence[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    available = set(pq.read_schema(grid_path).names)
    grid = pd.read_parquet(grid_path, columns=[c for c in SOURCE_COLUMNS if c in available])
    selected = grid[grid["host_id"].isin(hosts) & (grid["ts"] >= start) & (grid["ts"] < end)]
    rows = selected[selected["observed"] & selected["quality_mask"].eq(0)]
    return rows.drop(columns=["observed", "quality_mask"]).reset_index(drop=True)


def training_reference_rows(
    grid_path: Path,
    hosts: Sequence[str],
    boundaries: SplitBoundaries,
    statistics_start: pd.Timestamp,
) -> pd.DataFrame:
    grid = pd.read_parquet(grid_path, columns=list(GRID_COLUMNS))
    valid = select_valid_readings(grid, hosts)
    window = valid[(valid["ts"] >= statistics_start) & (valid["ts"] < boundaries.train_end)]
    return window.dropna(subset=["power", "cpu_load"])


def ensure_outside_training(plan: InjectionPlan, boundaries: SplitBoundaries) -> None:
    if plan.start < boundaries.train_end:
        raise ValueError("As anomalias só podem ser injetadas na validação e no teste")


def inject(
    rows: pd.DataFrame, reference_rows: pd.DataFrame, hosts: Sequence[str], plan: InjectionPlan
) -> tuple[pd.DataFrame, list[InjectedEvent]]:
    events = plan_events(hosts, plan)
    injected = apply_events(rows, events, compute_host_reference(reference_rows))
    return injected, events


def write_event_log(
    path: Path, scenario: str, plan: InjectionPlan, events: Sequence[InjectedEvent]
) -> None:
    content: dict[str, Any] = {
        "scenario": scenario,
        "seed": plan.seed,
        "kinds": list(plan.kinds),
        "events": [event.to_dict() for event in events],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
