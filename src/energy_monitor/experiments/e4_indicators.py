import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml

from energy_monitor.common.features import GRID_COLUMNS, select_valid_readings
from energy_monitor.common.indicators import compute_indicators, energy_kwh, idle_mask
from energy_monitor.common.schema import GRID_STEP, SOURCE_TIMEZONE
from energy_monitor.experiments.context import (
    boundaries_from_report,
    load_grid,
    read_quality_report,
)
from energy_monitor.experiments.run_record import build_run_record, write_json

EXPERIMENT_NAME: Final = "e4_indicators"
DEFAULT_CONFIG_PATH: Final = Path("configs/e4_indicators.yaml")
DailyStatistic = Callable[[pd.DataFrame], float]


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def daily_totals(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    data = frame.assign(
        idle=idle_mask(frame, threshold),
        energy=frame["power"].to_numpy(np.float64) * (GRID_STEP / pd.Timedelta(hours=1)) / 1000,
        day=frame["ts"].dt.tz_convert(SOURCE_TIMEZONE).dt.date,
    )
    data["idle_energy"] = data["energy"].where(data["idle"], 0.0)
    return data.groupby("day").agg(
        energy=("energy", "sum"),
        idle_energy=("idle_energy", "sum"),
        idle_readings=("idle", "sum"),
        readings=("idle", "size"),
    )


def idle_energy_share(days: pd.DataFrame) -> float:
    return float(days["idle_energy"].sum() / days["energy"].sum())


def idle_time_share(days: pd.DataFrame) -> float:
    return float(days["idle_readings"].sum() / days["readings"].sum())


def mean_daily_energy(days: pd.DataFrame) -> float:
    return float(days["energy"].mean())


STATISTICS: Final[dict[str, DailyStatistic]] = {
    "idle_energy_share": idle_energy_share,
    "idle_time_share": idle_time_share,
    "mean_daily_energy_kwh": mean_daily_energy,
}


def daily_block_bootstrap(
    days: pd.DataFrame, statistic: DailyStatistic, resamples: int, seed: int, confidence: float
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    values = np.array(
        [
            statistic(days.iloc[generator.integers(0, len(days), len(days))])
            for _ in range(resamples)
        ]
    )
    tail = (1 - confidence) / 2 * 100
    return {
        "estimate": statistic(days),
        "lower": float(np.percentile(values, tail)),
        "upper": float(np.percentile(values, 100 - tail)),
    }


def evaluate_threshold(
    frame: pd.DataFrame, threshold: float, config: dict[str, Any]
) -> dict[str, Any]:
    days = daily_totals(frame, threshold)
    indicators = compute_indicators(frame, config["tariff_kwh"], threshold)
    intervals = {
        name: daily_block_bootstrap(
            days, statistic, config["bootstrap_resamples"], config["seed"], config["confidence"]
        )
        for name, statistic in STATISTICS.items()
    }
    return {"days": int(len(days)), "total": indicators["total"], "bootstrap": intervals}


def run(name: str, config: dict[str, Any]) -> Path:
    report = read_quality_report(name, config)
    boundaries = boundaries_from_report(report)
    grid = load_grid(name, config, GRID_COLUMNS)
    readings = select_valid_readings(grid, report["coverage"]["eligible_hosts"])
    test = readings[(readings["ts"] >= boundaries.validation_end) & readings["power"].notna()]
    test = test.dropna(subset=["cpu_load"]).reset_index(drop=True)
    metrics = {
        "dataset": name,
        "period_utc": [test["ts"].min().isoformat(), test["ts"].max().isoformat()],
        "energy_kwh": energy_kwh(test["power"]),
        "thresholds": {
            f"{threshold:g}": evaluate_threshold(test, threshold, config)
            for threshold in config["idle_thresholds"]
        },
    }
    record = build_run_record(
        EXPERIMENT_NAME, config, Path(config["datasets"][name]), config["seed"], metrics
    )
    path = Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_{name}.json"
    write_json(record, path)
    return path


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="E4: indicadores de consumo e ociosidade")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config = load_config(parser.parse_args(arguments).config)
    for name in config["datasets"]:
        print(f"Relatório gravado em {run(name, config)}")


if __name__ == "__main__":
    main()
