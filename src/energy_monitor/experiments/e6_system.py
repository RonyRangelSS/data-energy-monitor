import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import httpx
import numpy as np
import pandas as pd
import yaml
from sqlalchemy import func, select

from energy_monitor.common.schema import GRID_STEP
from energy_monitor.common.settings import get_settings
from energy_monitor.database.models import Reading
from energy_monitor.database.session import create_db_engine, create_session_factory
from energy_monitor.experiments.context import boundaries_from_report, read_quality_report
from energy_monitor.experiments.run_record import build_run_record, write_json
from energy_monitor.replayer.sender import iterate_batches, replay, send_batch
from energy_monitor.replayer.source import load_period

EXPERIMENT_NAME: Final = "e6_system"
DEFAULT_CONFIG_PATH: Final = Path("configs/e6_system.yaml")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def stored_readings(
    session_factory: Any, scenario: str, start: pd.Timestamp, end: pd.Timestamp
) -> int:
    with session_factory() as session:
        statement = (
            select(func.count())
            .select_from(Reading)
            .where(
                Reading.scenario == scenario,
                Reading.ts >= start.to_pydatetime(),
                Reading.ts < end.to_pydatetime(),
            )
        )
        return int(session.scalar(statement))


def run_speed(
    client: httpx.Client,
    session_factory: Any,
    warmup: pd.DataFrame,
    window: pd.DataFrame,
    speed: float,
    window_bounds: tuple[pd.Timestamp, pd.Timestamp],
    prefix: str,
) -> dict[str, Any]:
    scenario = f"{prefix}_{int(speed)}x"
    client.delete(f"/scenarios/{scenario}")
    replay(warmup, client, scenario, None)
    measured = replay(window, client, scenario, speed)
    _, first_batch = next(iterate_batches(window))
    repeated = send_batch(client, scenario, first_batch) or {"accepted": -1, "duplicates": -1}
    stored = stored_readings(session_factory, scenario, *window_bounds)
    summary = measured.summary()
    summary.update(
        stored_readings=stored,
        lost_readings_percent=100 * (1 - stored / measured.sent_readings)
        if measured.sent_readings
        else None,
        resend_accepted=repeated["accepted"],
        resend_duplicates=repeated["duplicates"],
    )
    return summary


def measure_series_queries(
    client: httpx.Client,
    scenario: str,
    host_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    requests: int,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    parameters = {"scenario": scenario, "start": start.isoformat(), "end": end.isoformat()}
    latencies, points = [], 0
    for _ in range(requests):
        began = clock()
        response = client.get(f"/servers/{host_id}/series", params=parameters)
        latencies.append((clock() - began) * 1000)
        response.raise_for_status()
        points = len(response.json())
    values = np.array(latencies)
    return {
        "requests": requests,
        "points_per_response": points,
        "latency_ms_p50": float(np.percentile(values, 50)),
        "latency_ms_p95": float(np.percentile(values, 95)),
        "latency_ms_max": float(values.max()),
        "rnf03_p95_below_1s": bool(np.percentile(values, 95) < 1000),
    }


def run_query_benchmark(config: dict[str, Any]) -> Path:
    settings = get_settings()
    name = config["dataset"]
    report = read_quality_report(name, config)
    boundaries = boundaries_from_report(report)
    hosts = report["coverage"]["eligible_hosts"]
    grid_path = Path(config["processed_dir"]) / f"{name}_grid.parquet"
    start = boundaries.validation_end
    end = start + pd.Timedelta(days=int(config["query_days"]))
    scenario = f"{config['scenario_prefix']}_queries"
    headers = {"X-API-Key": settings.api_key.get_secret_value()}
    with httpx.Client(base_url=settings.api_url, headers=headers, timeout=120) as client:
        client.delete(f"/scenarios/{scenario}")
        loaded = replay(load_period(grid_path, hosts, start, end), client, scenario, None)
        metrics = {
            "dataset": name,
            "loaded_readings": loaded.accepted,
            "period_utc": [start.isoformat(), end.isoformat()],
            **measure_series_queries(
                client, scenario, hosts[0], start, end, int(config["query_requests"])
            ),
        }
    record = build_run_record(
        EXPERIMENT_NAME, config, Path(config["data_file"]), config["seed"], metrics
    )
    path = Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_queries_{name}.json"
    write_json(record, path)
    return path


def run(config: dict[str, Any]) -> Path:
    settings = get_settings()
    name = config["dataset"]
    report = read_quality_report(name, config)
    boundaries = boundaries_from_report(report)
    hosts = report["coverage"]["eligible_hosts"]
    grid_path = Path(config["processed_dir"]) / f"{name}_grid.parquet"
    window_start = boundaries.validation_end
    window_end = window_start + int(config["window_readings"]) * GRID_STEP
    warmup = load_period(
        grid_path, hosts, window_start - int(config["warmup_readings"]) * GRID_STEP, window_start
    )
    window = load_period(grid_path, hosts, window_start, window_end)
    session_factory = create_session_factory(create_db_engine(settings.database_url))
    headers = {"X-API-Key": settings.api_key.get_secret_value()}
    results = {}
    with httpx.Client(base_url=settings.api_url, headers=headers, timeout=120) as client:
        for speed in config["speeds"]:
            print(f"e6 speed {speed}", flush=True)
            results[f"{speed:g}x"] = run_speed(
                client,
                session_factory,
                warmup,
                window,
                float(speed),
                (window_start, window_end),
                config["scenario_prefix"],
            )
    reference = results.get(f"{config['reference_speed']:g}x", {})
    metrics = {
        "dataset": name,
        "hosts": len(hosts),
        "window_utc": [window_start.isoformat(), window_end.isoformat()],
        "speeds": results,
        "rnf01_latency_p95_below_500ms": (reference.get("latency_ms_p95") or 1e9) < 500,
        "rnf02_no_lost_readings": reference.get("lost_readings_percent") == 0,
    }
    record = build_run_record(
        EXPERIMENT_NAME, config, Path(config["data_file"]), config["seed"], metrics
    )
    path = Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_{name}.json"
    write_json(record, path)
    return path


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="E6: latência, vazão e perdas do sistema")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--queries", action="store_true", help="Mede só as consultas (RNF03)")
    parsed = parser.parse_args(arguments)
    runner = run_query_benchmark if parsed.queries else run
    print(f"Relatório gravado em {runner(load_config(parsed.config))}")


if __name__ == "__main__":
    main()
