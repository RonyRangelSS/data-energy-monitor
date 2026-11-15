import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import httpx
import pandas as pd

from energy_monitor.common.settings import get_settings
from energy_monitor.common.splitting import local_to_utc
from energy_monitor.experiments.context import boundaries_from_report, read_quality_report
from energy_monitor.replayer.injection import load_injection_plan
from energy_monitor.replayer.sender import replay
from energy_monitor.replayer.source import (
    inject,
    load_period,
    training_reference_rows,
    write_event_log,
)

STATISTICS_START_LOCAL = "2019-03-15T00:00:00"
SOURCE_TIMEZONE = "Australia/Melbourne"


def parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reprodutor acelerado do dataset")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--start", required=True, help="Instante inicial com fuso (ISO 8601)")
    parser.add_argument("--end", required=True, help="Instante final com fuso (ISO 8601)")
    parser.add_argument("--speed", type=float, default=600.0)
    parser.add_argument("--dataset", default="d1")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--injection", type=Path, help="Arquivo YAML do plano de injeção")
    parser.add_argument("--event-log", type=Path, help="Onde gravar as anomalias injetadas")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = parse_arguments(arguments)
    settings = get_settings()
    config = {"e1_report_dir": str(parsed.reports_dir)}
    report = read_quality_report(parsed.dataset, config)
    hosts = report["coverage"]["eligible_hosts"]
    grid_path = parsed.processed_dir / f"{parsed.dataset}_grid.parquet"
    rows = load_period(grid_path, hosts, pd.Timestamp(parsed.start), pd.Timestamp(parsed.end))
    if parsed.injection is not None:
        plan = load_injection_plan(parsed.injection)
        reference = training_reference_rows(
            grid_path,
            hosts,
            boundaries_from_report(report),
            local_to_utc(STATISTICS_START_LOCAL, SOURCE_TIMEZONE),
        )
        rows, events = inject(rows, reference, hosts, plan)
        log_path = parsed.event_log or parsed.reports_dir / f"replay_events_{parsed.scenario}.json"
        write_event_log(log_path, parsed.scenario, plan, events)
    headers = {"X-API-Key": settings.api_key.get_secret_value()}
    with httpx.Client(base_url=settings.api_url, headers=headers, timeout=60) as client:
        result = replay(rows, client, parsed.scenario, parsed.speed)
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
