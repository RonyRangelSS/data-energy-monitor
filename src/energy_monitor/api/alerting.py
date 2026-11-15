from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from energy_monitor.api.operational import OperationalModels, model_label
from energy_monitor.api.structured_logging import get_logger
from energy_monitor.database.models import Alert, AlertState, Reading
from energy_monitor.models.detection.alerts import summarize_episode
from energy_monitor.models.detection.base import DEFAULT_CONSECUTIVE
from energy_monitor.models.detection.hysteresis import HysteresisState, step

OPEN: str = "open"
CLOSED: str = "closed"


@dataclass
class AlertUpdate:
    opened: int = 0
    closed: int = 0


def load_states(session: Session, scenario: str, hosts: list[str]) -> dict[str, AlertState]:
    statement = (
        select(AlertState)
        .where(AlertState.scenario == scenario, AlertState.host_id.in_(hosts))
        .with_for_update()
    )
    states = {state.host_id: state for state in session.scalars(statement)}
    for host_id in hosts:
        if host_id not in states:
            states[host_id] = AlertState(
                scenario=scenario, host_id=host_id, active=False, above=0, below=0
            )
            session.add(states[host_id])
    return states


def episode_rows(session: Session, alert: Alert) -> pd.DataFrame:
    statement = (
        select(
            Reading.host_id,
            Reading.ts,
            Reading.power,
            Reading.expected,
            Reading.upper,
            Reading.cpu_load,
            Reading.z,
        )
        .where(
            Reading.scenario == alert.scenario,
            Reading.host_id == alert.host_id,
            Reading.ts >= alert.start_ts,
            Reading.ts <= alert.last_ts,
        )
        .order_by(Reading.ts)
    )
    return pd.DataFrame(
        session.execute(statement).all(),
        columns=[
            "host_id",
            "ts",
            "power",
            "expected",
            "upper",
            "cpu_load",
            "z",
        ],
    )


def refresh_alert(session: Session, alert: Alert) -> None:
    session.flush()
    rows = episode_rows(session, alert)
    episode = summarize_episode(rows, rows["z"].to_numpy(dtype=np.float64))
    alert.readings = episode.readings
    alert.peak_z = episode.peak_score
    alert.unexplained_energy_kwh = episode.unexplained_energy_kwh
    alert.justification = episode.justification


def open_alert(session: Session, models: OperationalModels, scenario: str, row: pd.Series) -> Alert:
    alert = Alert(
        scenario=scenario,
        host_id=row["host_id"],
        status=OPEN,
        start_ts=row["ts"].to_pydatetime(),
        last_ts=row["ts"].to_pydatetime(),
        readings=0,
        peak_z=0.0,
        unexplained_energy_kwh=0.0,
        justification="",
        detector=f"{models.detector.name}:{models.detector.version}",
        expected_model=model_label(models.expected_model),
    )
    session.add(alert)
    session.flush()
    return alert


def close_alert(alert: Alert) -> None:
    alert.status = CLOSED
    alert.end_ts = alert.last_ts


def process_reading(
    session: Session,
    models: OperationalModels,
    scenario: str,
    state: AlertState,
    row: pd.Series,
    update: AlertUpdate,
) -> None:
    opening_threshold, closing_threshold = models.detector.thresholds()
    consecutive = int(models.detector.hyperparameters.get("consecutive", DEFAULT_CONSECUTIVE))
    z = row["z"]
    valid = pd.notna(z)
    current = HysteresisState(active=state.active, above=state.above, below=state.below)
    following = step(
        current, valid and z >= opening_threshold, (not valid) or z < closing_threshold, consecutive
    )
    if following.active and not current.active:
        alert = open_alert(session, models, scenario, row)
        state.open_alert_id = alert.id
        update.opened += 1
        get_logger().info(
            "alert_opened",
            extra={"scenario": scenario, "host_id": row["host_id"], "ts": str(row["ts"])},
        )
    if following.active:
        alert = session.get(Alert, state.open_alert_id)
        alert.last_ts = row["ts"].to_pydatetime()
        refresh_alert(session, alert)
    if current.active and not following.active:
        alert = session.get(Alert, state.open_alert_id)
        close_alert(alert)
        state.open_alert_id = None
        update.closed += 1
        get_logger().info(
            "alert_closed",
            extra={"scenario": scenario, "host_id": row["host_id"], "alert_id": alert.id},
        )
    state.active, state.above, state.below = following.active, following.above, following.below


def update_alerts(
    session: Session, models: OperationalModels, scenario: str, inserted: pd.DataFrame
) -> AlertUpdate:
    update = AlertUpdate()
    if inserted.empty:
        return update
    ordered = inserted.sort_values(["host_id", "ts"])
    states = load_states(session, scenario, sorted(ordered["host_id"].astype(str).unique()))
    for host_id, rows in ordered.groupby("host_id", sort=True):
        state = states[str(host_id)]
        for _, row in rows.iterrows():
            process_reading(session, models, scenario, state, row, update)
    return update
