import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from energy_monitor.replayer.injection import InjectionPlan
from energy_monitor.replayer.sender import (
    interval_seconds,
    iterate_batches,
    reading_payload,
    replay,
)
from energy_monitor.replayer.source import inject, load_period, write_event_log
from tests.unit.forecast_helpers import GRID_START, make_grid

STEP = pd.Timedelta(minutes=10)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def accepting_transport(calls: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200, json={"accepted": len(body["readings"]), "duplicates": 0, "rejected": []}
        )

    return httpx.MockTransport(handler)


def test_payload_replaces_missing_values_with_null() -> None:
    payload = reading_payload({"host_id": "h1", "ts": GRID_START, "power": np.nan, "cpu_load": 3.0})

    assert payload["power"] is None
    assert payload["cpu_load"] == 3.0
    assert payload["ts"] == GRID_START.isoformat()


def test_batches_group_readings_by_instant_in_order() -> None:
    grid = make_grid(3)

    batches = list(iterate_batches(grid))

    assert [moment for moment, _ in batches] == [GRID_START + step * STEP for step in range(3)]
    assert [item["host_id"] for item in batches[0][1]] == ["h1", "h2"]


def test_speed_600_sends_one_instant_per_second() -> None:
    assert interval_seconds(600.0) == 1.0
    assert interval_seconds(None) == 0.0


def test_replay_follows_schedule_and_counts_results() -> None:
    calls: list[dict] = []
    clock = FakeClock()
    with httpx.Client(base_url="http://api", transport=accepting_transport(calls)) as client:
        report = replay(make_grid(3), client, "s1", 600.0, clock=clock, sleep=clock.sleep)

    assert len(calls) == 3
    assert calls[0]["scenario"] == "s1"
    assert clock.sleeps == [1.0, 1.0]
    summary = report.summary()
    assert summary["sent_readings"] == 6
    assert summary["accepted"] == 6
    assert summary["failed_batches"] == 0


def test_replay_counts_failed_batches() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    with httpx.Client(base_url="http://api", transport=transport) as client:
        report = replay(make_grid(2), client, "s1", None)

    assert report.failed_batches == 2
    assert report.accepted == 0


def test_load_period_keeps_only_valid_readings_of_selected_hosts(tmp_path: Path) -> None:
    grid = make_grid(10)
    grid.loc[1, "quality_mask"] = 2
    grid.loc[2, "observed"] = False
    path = tmp_path / "grid.parquet"
    grid.to_parquet(path, index=False)

    rows = load_period(path, ["h1"], GRID_START, GRID_START + 5 * STEP)

    assert rows["host_id"].unique().tolist() == ["h1"]
    assert len(rows) == 3


def test_injection_is_logged_separately_from_readings(tmp_path: Path) -> None:
    grid = make_grid(200)
    plan = InjectionPlan(("offset",), (0.4,), (3,), 2, 5, GRID_START, GRID_START + 150 * STEP)

    injected, events = inject(grid, grid, ["h1", "h2"], plan)
    write_event_log(tmp_path / "events.json", "s1", plan, events)

    assert len(events) == 2
    assert (injected["power"] >= grid["power"]).all()
    assert "kind" not in injected.columns
    log = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    assert log["seed"] == 5
    assert len(log["events"]) == 2


@pytest.mark.parametrize("latencies", [[0.1, 0.2, 0.3]])
def test_summary_reports_percentiles_in_milliseconds(latencies: list[float]) -> None:
    from energy_monitor.replayer.sender import ReplayReport

    report = ReplayReport(
        scenario="s",
        speed=600.0,
        latencies_seconds=latencies,
        elapsed_seconds=1.0,
        sent_readings=10,
    )

    summary = report.summary()

    assert summary["latency_ms_p50"] == pytest.approx(200.0)
    assert summary["throughput_readings_per_second"] == 10.0
