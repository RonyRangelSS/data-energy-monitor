from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from energy_monitor.api.alerting import AlertUpdate, update_alerts
from energy_monitor.api.operational import OperationalModels, model_label
from energy_monitor.api.schemas import ReadingIn, RejectedReading
from energy_monitor.api.structured_logging import get_logger
from energy_monitor.common.expected_features import prepare_expected_rows
from energy_monitor.common.preparation import flag_quality
from energy_monitor.common.schema import QualityFlag
from energy_monitor.database.models import Reading

MEASUREMENTS: Final = (
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
INVALID_TIMESTAMP: Final = "invalid_timestamp"


@dataclass
class IngestionResult:
    accepted: int = 0
    duplicates: int = 0
    rejected: list[RejectedReading] = field(default_factory=list)
    instants: list[pd.Timestamp] = field(default_factory=list)
    alerts: AlertUpdate = field(default_factory=AlertUpdate)


def parse_timestamp(value: str) -> pd.Timestamp | None:
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return pd.Timestamp(moment).tz_convert("UTC")


def readings_frame(readings: Sequence[ReadingIn]) -> pd.DataFrame:
    frame = pd.DataFrame([reading.model_dump() for reading in readings])
    frame["raw_ts"] = frame["ts"]
    parsed = [parse_timestamp(value) for value in frame["ts"]]
    frame["ts"] = pd.to_datetime(pd.Series(parsed, dtype="object"), utc=True).dt.as_unit("us")
    for column in MEASUREMENTS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float64)
    return frame


def rejection_reasons(mask: int, has_timestamp: bool) -> list[str]:
    if not has_timestamp:
        return [INVALID_TIMESTAMP]
    reasons = [flag.name.lower() for flag in QualityFlag if mask & int(flag)]
    return [reason for reason in reasons if reason != QualityFlag.DUPLICATE.name.lower()]


def add_expected_values(frame: pd.DataFrame, models: OperationalModels) -> pd.DataFrame:
    frame = frame.assign(expected=np.nan, upper=np.nan, z=np.nan, position=np.arange(len(frame)))
    rows = prepare_expected_rows(frame.drop(columns=["expected", "upper", "z"]), models.hosts)
    if rows.empty:
        return frame
    rows["expected"] = models.expected_model.predict(rows)
    _, rows["upper"] = models.expected_model.predict_interval(rows)
    rows["z"] = models.detector.score(rows)
    positions = rows["position"].to_numpy()
    for column in ("expected", "upper", "z"):
        frame.loc[positions, column] = rows[column].to_numpy()
    return frame


def to_record(row: pd.Series, scenario: str, model: str, received_at: datetime) -> dict:
    record = {column: none_if_missing(row[column]) for column in MEASUREMENTS}
    record.update(
        scenario=scenario,
        host_id=row["host_id"],
        ts=row["ts"].to_pydatetime(),
        expected=none_if_missing(row["expected"]),
        upper=none_if_missing(row["upper"]),
        z=none_if_missing(row["z"]),
        expected_model=model,
        received_at=received_at,
    )
    return record


def none_if_missing(value: object) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def insert_readings(
    session: Session, frame: pd.DataFrame, scenario: str, models: OperationalModels
) -> pd.DataFrame:
    if frame.empty:
        return frame
    received_at = datetime.now(UTC)
    label = model_label(models.expected_model)
    records = [to_record(row, scenario, label, received_at) for _, row in frame.iterrows()]
    statement = (
        insert(Reading)
        .values(records)
        .on_conflict_do_nothing(index_elements=["host_id", "ts", "scenario"])
        .returning(Reading.host_id, Reading.ts)
    )
    inserted = {(host, pd.Timestamp(ts)) for host, ts in session.execute(statement).all()}
    keys = list(zip(frame["host_id"], frame["ts"], strict=True))
    return frame[[key in inserted for key in keys]]


def ingest_batch(
    session: Session, models: OperationalModels, scenario: str, readings: Sequence[ReadingIn]
) -> IngestionResult:
    frame = readings_frame(readings)
    frame["quality_mask"] = flag_quality(frame)
    result = IngestionResult()
    duplicate = (frame["quality_mask"].to_numpy() & int(QualityFlag.DUPLICATE)) != 0
    for index, row in frame.iterrows():
        reasons = rejection_reasons(int(row["quality_mask"]), pd.notna(row["ts"]))
        if reasons:
            result.rejected.append(
                RejectedReading(
                    index=index, host_id=row["host_id"], ts=row["raw_ts"], reasons=reasons
                )
            )
    rejected_indexes = {item.index for item in result.rejected}
    candidates = frame[~duplicate & ~frame.index.isin(rejected_indexes)]
    result.duplicates = int(duplicate.sum())
    candidates = add_expected_values(candidates.reset_index(drop=True), models)
    inserted = insert_readings(session, candidates, scenario, models)
    result.accepted = int(len(inserted))
    result.duplicates += int(len(candidates) - len(inserted))
    result.instants = sorted(inserted["ts"].unique()) if len(inserted) else []
    result.alerts = update_alerts(session, models, scenario, inserted)
    log_rejections(scenario, result.rejected)
    return result


def log_rejections(scenario: str, rejected: Sequence[RejectedReading]) -> None:
    logger = get_logger()
    for item in rejected:
        logger.info(
            "reading_rejected",
            extra={
                "scenario": scenario,
                "host_id": item.host_id,
                "ts": item.ts,
                "reasons": item.reasons,
            },
        )
