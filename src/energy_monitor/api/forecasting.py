from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from energy_monitor.api.operational import ForecastSpec, OperationalModels, model_label
from energy_monitor.common.features import LOAD_FEATURES, STEPS_PER_DAY, build_forecast_frame
from energy_monitor.common.schema import GRID_FREQUENCY, GRID_STEP
from energy_monitor.database.models import TOTAL_HOST_ID, Forecast, Reading

AVAILABLE: Final = "ok"
INSUFFICIENT_HISTORY: Final = "insufficient_history"
HISTORY_COLUMNS: Final = ("host_id", "ts", "power", *LOAD_FEATURES)


def load_history(
    session: Session, scenario: str, hosts: Sequence[str], origin: pd.Timestamp
) -> pd.DataFrame:
    start = origin - STEPS_PER_DAY * GRID_STEP
    columns = [getattr(Reading, name) for name in HISTORY_COLUMNS]
    statement = select(*columns).where(
        Reading.scenario == scenario,
        Reading.host_id.in_(list(hosts)),
        Reading.ts >= start.to_pydatetime(),
        Reading.ts <= origin.to_pydatetime(),
    )
    frame = pd.DataFrame(session.execute(statement).all(), columns=list(HISTORY_COLUMNS))
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True).dt.as_unit("us")
    return frame


def history_grid(history: pd.DataFrame, hosts: Sequence[str], origin: pd.Timestamp) -> pd.DataFrame:
    slots = pd.date_range(
        origin - STEPS_PER_DAY * GRID_STEP, origin, freq=GRID_FREQUENCY, name="ts", unit="us"
    )
    index = pd.MultiIndex.from_product([sorted(hosts), slots], names=["host_id", "ts"])
    grid = history.set_index(["host_id", "ts"]).reindex(index).reset_index()
    grid["observed"] = grid["power"].notna()
    grid["quality_mask"] = 0
    return grid


def host_records(
    spec: ForecastSpec,
    scenario: str,
    origin: pd.Timestamp,
    hosts: Sequence[str],
    rows: pd.DataFrame,
) -> list[dict]:
    target = (origin + spec.steps * GRID_STEP).to_pydatetime()
    prediction = dict(zip(rows["host_id"].astype(str), spec.model.predict(rows), strict=True))
    lower, upper = {}, {}
    if spec.model.has_interval() and len(rows):
        low, high = spec.model.predict_interval(rows)
        lower = dict(zip(rows["host_id"].astype(str), low, strict=True))
        upper = dict(zip(rows["host_id"].astype(str), high, strict=True))
    base = {
        "scenario": scenario,
        "horizon": spec.horizon,
        "origin_ts": origin.to_pydatetime(),
        "target_ts": target,
        "model": model_label(spec.model),
        "variant": spec.variant,
    }
    return [
        {
            **base,
            "host_id": host,
            "status": AVAILABLE if host in prediction else INSUFFICIENT_HISTORY,
            "prediction": float(prediction[host]) if host in prediction else None,
            "lower": float(lower[host]) if host in lower else None,
            "upper": float(upper[host]) if host in upper else None,
            "hosts_included": None,
            "hosts_left_out": None,
        }
        for host in hosts
    ]


def total_record(records: list[dict], hosts: Sequence[str]) -> dict:
    available = [record["prediction"] for record in records if record["prediction"] is not None]
    total = {**records[0], "host_id": TOTAL_HOST_ID, "lower": None, "upper": None}
    total.update(
        status=AVAILABLE if available else INSUFFICIENT_HISTORY,
        prediction=float(np.sum(available)) if available else None,
        hosts_included=len(available),
        hosts_left_out=len(hosts) - len(available),
    )
    return total


def forecast_instant(
    session: Session, models: OperationalModels, scenario: str, origin: pd.Timestamp
) -> int:
    history = load_history(session, scenario, models.hosts, origin)
    grid = history_grid(history, models.hosts, origin)
    records: list[dict] = []
    for spec in models.forecasts:
        frame = build_forecast_frame(grid, models.hosts, spec.steps)
        rows = frame[frame["ts"] == origin]
        host_rows = host_records(spec, scenario, origin, models.hosts, rows)
        records.extend([*host_rows, total_record(host_rows, models.hosts)])
    statement = (
        insert(Forecast)
        .values(records)
        .on_conflict_do_nothing(index_elements=["scenario", "host_id", "origin_ts", "horizon"])
    )
    session.execute(statement)
    return len(records)


def forecast_instants(
    session: Session, models: OperationalModels, scenario: str, instants: Sequence[pd.Timestamp]
) -> int:
    return sum(
        forecast_instant(session, models, scenario, pd.Timestamp(origin)) > 0 for origin in instants
    )
