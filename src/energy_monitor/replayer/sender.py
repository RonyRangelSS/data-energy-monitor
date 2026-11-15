import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import httpx
import numpy as np
import pandas as pd

from energy_monitor.common.schema import GRID_STEP

READING_FIELDS: Final = (
    "power",
    "cpu_load",
    "mem_pct",
    "n_vms",
    "cores",
    "cores_used",
    "net_rx",
    "net_tx",
    "cpu1_temp",
    "cpu2_temp",
    "inlet_temp",
)
SECONDS_PER_READING: Final = GRID_STEP.total_seconds()


@dataclass
class ReplayReport:
    scenario: str
    speed: float | None
    batches: int = 0
    sent_readings: int = 0
    accepted: int = 0
    duplicates: int = 0
    rejected: int = 0
    failed_batches: int = 0
    latencies_seconds: list[float] = field(default_factory=list)
    schedule_lag_seconds: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        latencies = np.array(self.latencies_seconds) * 1000
        lags = np.array(self.schedule_lag_seconds)
        return {
            "scenario": self.scenario,
            "speed": self.speed,
            "batches": self.batches,
            "sent_readings": self.sent_readings,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "rejected": self.rejected,
            "failed_batches": self.failed_batches,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput_readings_per_second": (
                self.sent_readings / self.elapsed_seconds if self.elapsed_seconds else None
            ),
            "latency_ms_p50": percentile(latencies, 50),
            "latency_ms_p95": percentile(latencies, 95),
            "latency_ms_max": float(latencies.max()) if len(latencies) else None,
            "schedule_lag_seconds_max": float(lags.max()) if len(lags) else None,
        }


def percentile(values: np.ndarray, level: float) -> float | None:
    return float(np.percentile(values, level)) if len(values) else None


def json_value(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def reading_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = {name: json_value(row.get(name)) for name in READING_FIELDS}
    payload["host_id"] = str(row["host_id"])
    payload["ts"] = pd.Timestamp(row["ts"]).isoformat()
    return payload


def iterate_batches(frame: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, list[dict[str, Any]]]]:
    for moment, rows in frame.sort_values(["ts", "host_id"]).groupby("ts", sort=True):
        yield pd.Timestamp(moment), [reading_payload(row) for row in rows.to_dict("records")]


def interval_seconds(speed: float | None) -> float:
    return 0.0 if speed is None else SECONDS_PER_READING / speed


def send_batch(
    client: httpx.Client, scenario: str, readings: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    response = client.post("/readings", json={"scenario": scenario, "readings": list(readings)})
    if response.status_code != httpx.codes.OK:
        return None
    return response.json()


def replay(
    frame: pd.DataFrame,
    client: httpx.Client,
    scenario: str,
    speed: float | None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> ReplayReport:
    report = ReplayReport(scenario=scenario, speed=speed)
    interval = interval_seconds(speed)
    started = clock()
    for position, (_, readings) in enumerate(iterate_batches(frame)):
        scheduled = started + position * interval
        wait = scheduled - clock()
        if wait > 0:
            sleep(wait)
        sent_at = clock()
        report.schedule_lag_seconds.append(max(0.0, sent_at - scheduled))
        result = send_batch(client, scenario, readings)
        report.latencies_seconds.append(clock() - sent_at)
        report.batches += 1
        report.sent_readings += len(readings)
        if result is None:
            report.failed_batches += 1
            continue
        report.accepted += result["accepted"]
        report.duplicates += result["duplicates"]
        report.rejected += len(result["rejected"])
    report.elapsed_seconds = clock() - started
    return report
