from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from energy_monitor.database.base import Base

TOTAL_HOST_ID = "total"


class Reading(Base):
    __tablename__ = "readings"
    __table_args__ = (
        UniqueConstraint("host_id", "ts", "scenario"),
        Index("ix_readings_scenario_ts", "scenario", "ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scenario: Mapped[str] = mapped_column(String(100))
    host_id: Mapped[str] = mapped_column(String(64))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    power: Mapped[float] = mapped_column(Float)
    cpu_load: Mapped[float | None] = mapped_column(Float)
    mem_pct: Mapped[float | None] = mapped_column(Float)
    n_vms: Mapped[float | None] = mapped_column(Float)
    cores: Mapped[float | None] = mapped_column(Float)
    cores_used: Mapped[float | None] = mapped_column(Float)
    net_rx: Mapped[float | None] = mapped_column(Float)
    net_tx: Mapped[float | None] = mapped_column(Float)
    cpu1_temp: Mapped[float | None] = mapped_column(Float)
    cpu2_temp: Mapped[float | None] = mapped_column(Float)
    inlet_temp: Mapped[float | None] = mapped_column(Float)
    expected: Mapped[float | None] = mapped_column(Float)
    upper: Mapped[float | None] = mapped_column(Float)
    z: Mapped[float | None] = mapped_column(Float)
    expected_model: Mapped[str | None] = mapped_column(String(100))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AlertState(Base):
    __tablename__ = "alert_states"

    scenario: Mapped[str] = mapped_column(String(100), primary_key=True)
    host_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    active: Mapped[bool] = mapped_column(default=False)
    above: Mapped[int] = mapped_column(Integer, default=0)
    below: Mapped[int] = mapped_column(Integer, default=0)
    open_alert_id: Mapped[int | None] = mapped_column(BigInteger)


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alerts_scenario_start", "scenario", "start_ts"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scenario: Mapped[str] = mapped_column(String(100))
    host_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(10))
    start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    readings: Mapped[int] = mapped_column(Integer, default=0)
    peak_z: Mapped[float] = mapped_column(Float, default=0.0)
    unexplained_energy_kwh: Mapped[float] = mapped_column(Float, default=0.0)
    justification: Mapped[str] = mapped_column(Text, default="")
    detector: Mapped[str] = mapped_column(String(100))
    expected_model: Mapped[str] = mapped_column(String(100))


class Forecast(Base):
    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint("scenario", "host_id", "origin_ts", "horizon"),
        Index("ix_forecasts_scenario_target", "scenario", "target_ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scenario: Mapped[str] = mapped_column(String(100))
    host_id: Mapped[str] = mapped_column(String(64))
    horizon: Mapped[str] = mapped_column(String(8))
    origin_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    target_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32))
    prediction: Mapped[float | None] = mapped_column(Float)
    lower: Mapped[float | None] = mapped_column(Float)
    upper: Mapped[float | None] = mapped_column(Float)
    hosts_included: Mapped[int | None] = mapped_column(Integer)
    hosts_left_out: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str] = mapped_column(String(100))
    variant: Mapped[str] = mapped_column(String(32))


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    file_name: Mapped[str] = mapped_column(String(200), unique=True)
    experiment: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    commit: Mapped[str | None] = mapped_column(String(64))
    seed: Mapped[int | None] = mapped_column(Integer)
    data_sha256: Mapped[str | None] = mapped_column(String(64))
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
