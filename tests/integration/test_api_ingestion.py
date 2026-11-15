import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from energy_monitor.api.app import create_app
from energy_monitor.api.operational import OperationalModels
from energy_monitor.common.expected_features import prepare_expected_rows
from energy_monitor.common.settings import Settings
from energy_monitor.database.models import Alert, Forecast, Reading
from tests.integration.api_helpers import HOSTS, readings_payload, training_grid
from tests.integration.conftest import API_KEY

pytestmark = pytest.mark.integration
STEP = pd.Timedelta(minutes=10)


def post(client: TestClient, readings: list[dict], scenario: str = "test") -> dict:
    response = client.post("/readings", json={"scenario": scenario, "readings": readings})
    assert response.status_code == 200, response.text
    return response.json()


def first_readings(count: int) -> list[dict]:
    grid = training_grid().sort_values(["ts", "host_id"])
    return readings_payload(grid.head(count))


def test_ingestion_requires_api_key(api_client: TestClient) -> None:
    body = {"scenario": "test", "readings": first_readings(1)}

    assert (
        api_client.post("/readings", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    )
    assert api_client.post("/readings", json=body, headers={"X-API-Key": ""}).status_code == 401


def test_mixed_batch_reports_accepted_duplicates_and_rejections(api_client: TestClient) -> None:
    valid = first_readings(2)
    invalid_power = {**valid[0], "ts": "2019-03-01T00:10:00+00:00", "power": 0.0}
    invalid_cpu = {**valid[1], "ts": "2019-03-01T00:10:00+00:00", "cpu_load": 150.0}
    naive = {**valid[0], "ts": "2019-03-01T00:20:00"}
    garbage = {**valid[0], "ts": "not a date"}
    off_grid = {**valid[0], "ts": "2019-03-01T00:25:00+00:00"}

    result = post(
        api_client, [*valid, valid[0], invalid_power, invalid_cpu, naive, garbage, off_grid]
    )

    assert result["accepted"] == 2
    assert result["duplicates"] == 1
    reasons = {item["index"]: item["reasons"] for item in result["rejected"]}
    assert reasons[3] == ["non_positive_power"]
    assert reasons[4] == ["cpu_out_of_range"]
    assert reasons[5] == ["invalid_timestamp"]
    assert reasons[6] == ["invalid_timestamp"]
    assert reasons[7] == ["off_grid"]


def test_resending_batch_does_not_change_stored_count(
    api_client: TestClient, clean_engine: Engine
) -> None:
    readings = first_readings(10)
    post(api_client, readings)

    again = post(api_client, readings)

    assert again == {**again, "accepted": 0, "duplicates": 10}
    with Session(clean_engine) as session:
        assert len(session.scalars(select(Reading)).all()) == 10


def test_stored_expected_and_z_match_offline_calculation(
    api_client: TestClient, clean_engine: Engine, operational_models: OperationalModels
) -> None:
    grid = training_grid().sort_values(["ts", "host_id"]).head(40)
    post(api_client, readings_payload(grid))

    rows = prepare_expected_rows(grid.reset_index(drop=True), HOSTS)
    expected = operational_models.expected_model.predict(rows)
    z = operational_models.detector.score(rows.assign(expected=expected))
    with Session(clean_engine) as session:
        stored = session.execute(
            select(Reading.host_id, Reading.ts, Reading.expected, Reading.z).order_by(
                Reading.ts, Reading.host_id
            )
        ).all()
    np.testing.assert_allclose([item.expected for item in stored], expected, atol=1e-6)
    np.testing.assert_allclose([item.z for item in stored], z, atol=1e-6)


def anomalous_series(start: int, high: list[bool]) -> list[dict]:
    grid = training_grid()
    host = grid[grid["host_id"] == "h1"].sort_values("ts").iloc[start : start + len(high)].copy()
    host.loc[np.array(high), "power"] += 80.0
    return readings_payload(host)


def test_alert_opens_and_closes_with_hysteresis(
    api_client: TestClient, clean_engine: Engine
) -> None:
    result = post(api_client, anomalous_series(0, [False, True, True, True, False, False, False]))

    assert result["alerts_opened"] == 1
    assert result["alerts_closed"] == 1
    with Session(clean_engine) as session:
        alert = session.scalars(select(Alert)).one()
    assert alert.status == "closed"
    assert alert.readings == 3
    assert alert.unexplained_energy_kwh > 0
    assert "acima do esperado" in alert.justification


def test_open_alert_survives_api_restart(
    settings: Settings, operational_models: OperationalModels, clean_engine: Engine
) -> None:
    readings = anomalous_series(10, [True, True, True, False, False])
    with TestClient(
        create_app(settings, operational_models, clean_engine), headers={"X-API-Key": API_KEY}
    ) as first:
        assert post(first, readings[:3])["alerts_opened"] == 1
    with TestClient(
        create_app(settings, operational_models, clean_engine), headers={"X-API-Key": API_KEY}
    ) as second:
        assert post(second, readings[3:])["alerts_closed"] == 1
    with Session(clean_engine) as session:
        assert session.scalars(select(Alert)).one().status == "closed"


def test_forecasts_need_a_day_of_history(api_client: TestClient, clean_engine: Engine) -> None:
    grid = training_grid().sort_values(["ts", "host_id"])
    first_instants = grid[grid["ts"] < grid["ts"].min() + 150 * STEP]

    result = post(api_client, readings_payload(first_instants))

    assert result["forecast_instants"] == 150
    with Session(clean_engine) as session:
        early = session.scalars(
            select(Forecast).where(
                Forecast.origin_ts == grid["ts"].min().to_pydatetime(), Forecast.host_id == "h1"
            )
        ).all()
        late = session.scalars(
            select(Forecast).where(
                Forecast.origin_ts == (grid["ts"].min() + 149 * STEP).to_pydatetime(),
                Forecast.host_id == "total",
            )
        ).all()
    assert {item.status for item in early} == {"insufficient_history"}
    assert {item.status for item in late} == {"ok"}
    assert all(item.hosts_included == 2 and item.hosts_left_out == 0 for item in late)
