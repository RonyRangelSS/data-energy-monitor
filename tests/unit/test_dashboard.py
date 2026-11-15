import logging

import httpx
import pytest

from energy_monitor.api.structured_logging import JsonFormatter
from energy_monitor.dashboard.client import MonitorClient
from energy_monitor.dashboard.views import (
    alerts_frame,
    series_frame,
    servers_frame,
    total_forecast_frame,
)

TIMEZONE = "Australia/Melbourne"


def test_client_sends_only_given_parameters() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"total": {}})

    client = MonitorClient("http://api", transport=httpx.MockTransport(handler))
    client.indicators("demo", tariff=1.2)

    assert requests[0].url.path == "/indicators"
    assert dict(requests[0].url.params) == {"scenario": "demo", "tariff": "1.2"}


def test_client_raises_on_error_status() -> None:
    client = MonitorClient(
        "http://api", transport=httpx.MockTransport(lambda request: httpx.Response(500))
    )

    with pytest.raises(httpx.HTTPStatusError):
        client.servers("demo")


def test_series_frame_uses_local_time_and_portuguese_labels() -> None:
    payload = [
        {
            "ts": "2019-05-01T00:00:00+00:00",
            "power": 100.0,
            "expected": 98.0,
            "upper": 110.0,
            "forecast_1h": None,
            "forecast_24h": 101.0,
        }
    ]

    frame = series_frame(payload, TIMEZONE)

    assert list(frame.columns) == [
        "Real",
        "Esperado",
        "Limite superior",
        "Previsto (1 h)",
        "Previsto (24 h)",
    ]
    assert frame.index[0].hour == 10


def test_alerts_frame_orders_newest_first() -> None:
    payload = [
        {
            "host_id": "h1",
            "status": "closed",
            "start_ts": "2019-05-01T00:00:00+00:00",
            "end_ts": "2019-05-01T01:00:00+00:00",
            "peak_z": 5.0,
            "unexplained_energy_kwh": 0.1,
            "justification": "x",
        },
        {
            "host_id": "h2",
            "status": "open",
            "start_ts": "2019-05-02T00:00:00+00:00",
            "end_ts": None,
            "peak_z": 4.0,
            "unexplained_energy_kwh": 0.0,
            "justification": "y",
        },
    ]

    frame = alerts_frame(payload, TIMEZONE)

    assert frame["Servidor"].tolist() == ["h2", "h1"]


def test_total_forecast_frame_pivots_horizons_and_skips_missing() -> None:
    payload = [
        {
            "horizon": "1h",
            "target_ts": "2019-05-01T01:00:00+00:00",
            "prediction": 500.0,
            "hosts_included": 2,
            "hosts_left_out": 0,
        },
        {
            "horizon": "24h",
            "target_ts": "2019-05-01T01:00:00+00:00",
            "prediction": None,
            "hosts_included": 0,
            "hosts_left_out": 2,
        },
    ]

    frame = total_forecast_frame(payload, TIMEZONE)

    assert list(frame.columns) == ["1h"]
    assert frame.iloc[0, 0] == 500.0


def test_servers_frame_renames_columns() -> None:
    payload = [
        {
            "host_id": "h1",
            "last_ts": "2019-05-01T00:00:00+00:00",
            "last_power": 100.0,
            "last_z": 0.5,
            "open_alert": False,
            "readings": 3,
        }
    ]

    assert "Alerta aberto" in servers_frame(payload, TIMEZONE).columns


def test_json_formatter_includes_extra_fields() -> None:
    record = logging.makeLogRecord(
        {"msg": "reading_rejected", "levelname": "INFO", "host_id": "h1"}
    )

    text = JsonFormatter().format(record)

    assert '"event": "reading_rejected"' in text
    assert '"host_id": "h1"' in text
