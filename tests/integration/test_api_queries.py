import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from energy_monitor.database.session import create_session_factory
from energy_monitor.experiments import e6_system
from tests.integration.api_helpers import readings_payload, training_grid

pytestmark = pytest.mark.integration
STEP = pd.Timedelta(minutes=10)


@pytest.fixture
def loaded_client(api_client: TestClient) -> TestClient:
    grid = training_grid().sort_values(["ts", "host_id"])
    selected = grid[grid["ts"] < grid["ts"].min() + 152 * STEP].copy()
    selected.loc[
        selected["host_id"].eq("h1")
        & selected["ts"].between(grid["ts"].min() + 100 * STEP, grid["ts"].min() + 104 * STEP),
        "power",
    ] += 80
    response = api_client.post(
        "/readings", json={"scenario": "demo", "readings": readings_payload(selected)}
    )
    assert response.status_code == 200
    return api_client


def test_health(api_client: TestClient) -> None:
    assert api_client.get("/health").json() == {"status": "ok"}


def test_servers_lists_last_reading_per_host(loaded_client: TestClient) -> None:
    servers = loaded_client.get("/servers", params={"scenario": "demo"}).json()

    assert [item["host_id"] for item in servers] == ["h1", "h2"]
    assert all(item["readings"] == 152 for item in servers)


def test_series_joins_real_expected_and_forecast(loaded_client: TestClient) -> None:
    series = loaded_client.get("/servers/h1/series", params={"scenario": "demo"}).json()

    assert len(series) == 152
    assert series[-1]["expected"] is not None
    assert any(point["forecast_1h"] is not None for point in series)


def test_alerts_can_be_filtered(loaded_client: TestClient) -> None:
    alerts = loaded_client.get("/alerts", params={"scenario": "demo"}).json()
    only_h1 = loaded_client.get("/alerts", params={"scenario": "demo", "host_id": "h1"}).json()
    closed = loaded_client.get("/alerts", params={"scenario": "demo", "status": "closed"}).json()

    assert any(item["host_id"] == "h1" for item in alerts)
    assert only_h1 and all(item["host_id"] == "h1" for item in only_h1)
    assert all(item["status"] == "closed" for item in closed)


def test_indicators_use_default_or_requested_tariff(loaded_client: TestClient) -> None:
    default = loaded_client.get("/indicators", params={"scenario": "demo"}).json()
    custom = loaded_client.get("/indicators", params={"scenario": "demo", "tariff": 1.6}).json()

    assert default["tariff_kwh"] == 0.8
    assert custom["total"]["associated_cost"] == pytest.approx(
        2 * default["total"]["associated_cost"]
    )
    assert custom["total"]["energy_kwh"] == default["total"]["energy_kwh"]


def test_indicators_reject_invalid_tariff(api_client: TestClient) -> None:
    assert (
        api_client.get("/indicators", params={"scenario": "demo", "tariff": 0}).status_code == 422
    )


def test_total_forecast_reports_hosts_left_out(loaded_client: TestClient) -> None:
    forecasts = loaded_client.get("/forecasts", params={"scenario": "demo", "horizon": "1h"}).json()

    available = [item for item in forecasts if item["status"] == "ok"]
    assert available
    assert all(item["host_id"] == "total" for item in forecasts)
    assert available[0]["hosts_included"] + available[0]["hosts_left_out"] == 2


def test_models_describe_operational_choices(api_client: TestClient) -> None:
    description = api_client.get("/models").json()

    assert description["hosts"] == 2
    assert [item["horizon"] for item in description["forecasts"]] == ["1h", "24h"]


def test_delete_scenario_requires_key_and_removes_everything(loaded_client: TestClient) -> None:
    assert (
        loaded_client.delete("/scenarios/demo", headers={"X-API-Key": "wrong"}).status_code == 401
    )

    removed = loaded_client.delete("/scenarios/demo").json()["removed"]

    assert removed["readings"] == 304
    assert loaded_client.get("/servers", params={"scenario": "demo"}).json() == []
    assert loaded_client.get("/forecasts", params={"scenario": "demo"}).json() == []


def test_e6_speed_run_measures_latency_and_losses(
    api_client: TestClient, clean_engine: Engine
) -> None:
    grid = training_grid().sort_values(["ts", "host_id"])
    start = grid["ts"].min()
    warmup = grid[grid["ts"] < start + 3 * STEP]
    window = grid[(grid["ts"] >= start + 3 * STEP) & (grid["ts"] < start + 6 * STEP)]

    summary = e6_system.run_speed(
        api_client,
        create_session_factory(clean_engine),
        warmup,
        window,
        60000.0,
        (start + 3 * STEP, start + 6 * STEP),
        "e6test",
    )

    assert summary["sent_readings"] == 6
    assert summary["stored_readings"] == 6
    assert summary["lost_readings_percent"] == 0
    assert summary["resend_duplicates"] == 2
    assert summary["latency_ms_p95"] > 0


def test_series_query_benchmark_reports_percentiles(loaded_client: TestClient) -> None:
    start = training_grid()["ts"].min()

    result = e6_system.measure_series_queries(
        loaded_client, "demo", "h1", start, start + 200 * STEP, requests=5
    )

    assert result["requests"] == 5
    assert result["points_per_response"] == 152
    assert result["latency_ms_p95"] >= result["latency_ms_p50"] > 0
