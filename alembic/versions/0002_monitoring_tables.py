"""monitoring tables

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-15 22:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MEASUREMENTS = (
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
    "expected",
    "upper",
    "z",
)


def upgrade() -> None:
    op.create_table(
        "readings",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("scenario", sa.String(100), nullable=False),
        sa.Column("host_id", sa.String(64), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("power", sa.Float(), nullable=False),
        *[sa.Column(name, sa.Float(), nullable=True) for name in MEASUREMENTS],
        sa.Column("expected_model", sa.String(100), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_readings")),
        sa.UniqueConstraint("host_id", "ts", "scenario", name=op.f("uq_readings_host_id")),
    )
    op.create_index("ix_readings_scenario_ts", "readings", ["scenario", "ts"])
    op.create_table(
        "alert_states",
        sa.Column("scenario", sa.String(100), nullable=False),
        sa.Column("host_id", sa.String(64), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("above", sa.Integer(), nullable=False),
        sa.Column("below", sa.Integer(), nullable=False),
        sa.Column("open_alert_id", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("scenario", "host_id", name=op.f("pk_alert_states")),
    )
    op.create_table(
        "alerts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("scenario", sa.String(100), nullable=False),
        sa.Column("host_id", sa.String(64), nullable=False),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("start_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("readings", sa.Integer(), nullable=False),
        sa.Column("peak_z", sa.Float(), nullable=False),
        sa.Column("unexplained_energy_kwh", sa.Float(), nullable=False),
        sa.Column("justification", sa.Text(), nullable=False),
        sa.Column("detector", sa.String(100), nullable=False),
        sa.Column("expected_model", sa.String(100), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alerts")),
    )
    op.create_index("ix_alerts_scenario_start", "alerts", ["scenario", "start_ts"])
    op.create_table(
        "forecasts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("scenario", sa.String(100), nullable=False),
        sa.Column("host_id", sa.String(64), nullable=False),
        sa.Column("horizon", sa.String(8), nullable=False),
        sa.Column("origin_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("prediction", sa.Float(), nullable=True),
        sa.Column("lower", sa.Float(), nullable=True),
        sa.Column("upper", sa.Float(), nullable=True),
        sa.Column("hosts_included", sa.Integer(), nullable=True),
        sa.Column("hosts_left_out", sa.Integer(), nullable=True),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("variant", sa.String(32), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_forecasts")),
        sa.UniqueConstraint(
            "scenario", "host_id", "origin_ts", "horizon", name=op.f("uq_forecasts_scenario")
        ),
    )
    op.create_index("ix_forecasts_scenario_target", "forecasts", ["scenario", "target_ts"])
    op.create_table(
        "experiment_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("file_name", sa.String(200), nullable=False),
        sa.Column("experiment", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("commit", sa.String(64), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("data_sha256", sa.String(64), nullable=True),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_experiment_runs")),
        sa.UniqueConstraint("file_name", name=op.f("uq_experiment_runs_file_name")),
    )


def downgrade() -> None:
    op.drop_table("experiment_runs")
    op.drop_index("ix_forecasts_scenario_target", table_name="forecasts")
    op.drop_table("forecasts")
    op.drop_index("ix_alerts_scenario_start", table_name="alerts")
    op.drop_table("alerts")
    op.drop_table("alert_states")
    op.drop_index("ix_readings_scenario_ts", table_name="readings")
    op.drop_table("readings")
