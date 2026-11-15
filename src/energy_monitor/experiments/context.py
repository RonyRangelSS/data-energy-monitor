import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from energy_monitor.common.splitting import SplitBoundaries


def read_quality_report(name: str, config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["e1_report_dir"]) / f"e1_data_quality_{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def boundaries_from_report(report: dict[str, Any]) -> SplitBoundaries:
    moments = {name: pd.Timestamp(values["utc"]) for name, values in report["split"].items()}
    return SplitBoundaries(
        start=moments["start"],
        train_end=moments["train_end"],
        validation_end=moments["validation_end"],
        end=moments["end"],
    )


def load_grid(
    name: str,
    config: dict[str, Any],
    columns: Sequence[str],
    before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    grid_path = Path(config["processed_dir"]) / f"{name}_grid.parquet"
    grid = pd.read_parquet(grid_path, columns=list(columns))
    if before is not None:
        grid = grid[grid["ts"] < before].reset_index(drop=True)
    return grid


def read_json_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]
