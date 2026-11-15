from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from energy_monitor.common.indicators import compute_indicators
from energy_monitor.database.models import TOTAL_HOST_ID, Alert, AlertState, Forecast, Reading


def period_filters(column: Any, start: datetime | None, end: datetime | None) -> list[Any]:
    filters = []
    if start is not None:
        filters.append(column >= start)
    if end is not None:
        filters.append(column <= end)
    return filters


def list_servers(session: Session, scenario: str) -> list[dict[str, Any]]:
    latest = (
        select(
            Reading.host_id, func.max(Reading.ts).label("last_ts"), func.count().label("readings")
        )
        .where(Reading.scenario == scenario)
        .group_by(Reading.host_id)
        .subquery()
    )
    statement = (
        select(latest.c.host_id, latest.c.last_ts, latest.c.readings, Reading.power, Reading.z)
        .join(Reading, (Reading.host_id == latest.c.host_id) & (Reading.ts == latest.c.last_ts))
        .where(Reading.scenario == scenario)
        .order_by(latest.c.host_id)
    )
    open_hosts = set(
        session.scalars(
            select(Alert.host_id).where(Alert.scenario == scenario, Alert.status == "open")
        )
    )
    return [
        {
            "host_id": host_id,
            "last_ts": last_ts,
            "readings": readings,
            "last_power": power,
            "last_z": z,
            "open_alert": host_id in open_hosts,
        }
        for host_id, last_ts, readings, power, z in session.execute(statement).all()
    ]


def server_series(
    session: Session, scenario: str, host_id: str, start: datetime | None, end: datetime | None
) -> list[dict[str, Any]]:
    readings = session.execute(
        select(
            Reading.ts, Reading.power, Reading.expected, Reading.upper, Reading.z, Reading.cpu_load
        )
        .where(
            Reading.scenario == scenario,
            Reading.host_id == host_id,
            *period_filters(Reading.ts, start, end),
        )
        .order_by(Reading.ts)
    ).all()
    forecasts = session.execute(
        select(Forecast.target_ts, Forecast.horizon, Forecast.prediction).where(
            Forecast.scenario == scenario,
            Forecast.host_id == host_id,
            Forecast.status == "ok",
            *period_filters(Forecast.target_ts, start, end),
        )
    ).all()
    predicted = {(target, horizon): value for target, horizon, value in forecasts}
    return [
        {
            "ts": ts,
            "power": power,
            "expected": expected,
            "upper": upper,
            "z": z,
            "cpu_load": cpu,
            "forecast_1h": predicted.get((ts, "1h")),
            "forecast_24h": predicted.get((ts, "24h")),
        }
        for ts, power, expected, upper, z, cpu in readings
    ]


def list_alerts(
    session: Session,
    scenario: str,
    status: str | None,
    host_id: str | None,
    start: datetime | None,
    end: datetime | None,
) -> list[Alert]:
    filters = [Alert.scenario == scenario, *period_filters(Alert.start_ts, start, end)]
    if status is not None:
        filters.append(Alert.status == status)
    if host_id is not None:
        filters.append(Alert.host_id == host_id)
    return list(session.scalars(select(Alert).where(*filters).order_by(Alert.start_ts)))


def list_forecasts(
    session: Session,
    scenario: str,
    host_id: str | None,
    horizon: str | None,
    start: datetime | None,
    end: datetime | None,
    limit: int,
) -> list[Forecast]:
    filters = [Forecast.scenario == scenario, *period_filters(Forecast.target_ts, start, end)]
    filters.append(Forecast.host_id == (host_id or TOTAL_HOST_ID))
    if horizon is not None:
        filters.append(Forecast.horizon == horizon)
    statement = select(Forecast).where(*filters).order_by(Forecast.origin_ts.desc()).limit(limit)
    return list(session.scalars(statement))


def indicators(
    session: Session,
    scenario: str,
    start: datetime | None,
    end: datetime | None,
    tariff_kwh: float,
    idle_threshold: float,
) -> dict[str, Any]:
    rows = session.execute(
        select(Reading.host_id, Reading.ts, Reading.power, Reading.cpu_load).where(
            Reading.scenario == scenario, *period_filters(Reading.ts, start, end)
        )
    ).all()
    frame = pd.DataFrame(rows, columns=["host_id", "ts", "power", "cpu_load"])
    return compute_indicators(frame, tariff_kwh, idle_threshold)


def delete_scenario(session: Session, scenario: str) -> dict[str, int]:
    removed = {}
    for table in (Reading, Alert, AlertState, Forecast):
        result = session.execute(delete(table).where(table.scenario == scenario))
        removed[table.__tablename__] = int(result.rowcount)
    return removed
