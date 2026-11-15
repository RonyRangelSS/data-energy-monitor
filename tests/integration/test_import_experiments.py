from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from energy_monitor.database.import_experiments import import_experiments
from energy_monitor.database.models import ExperimentRun
from energy_monitor.experiments.run_record import write_json

pytestmark = pytest.mark.integration


def test_experiment_reports_are_imported_and_updated(tmp_path: Path, clean_engine: Engine) -> None:
    record = {
        "experiment": "e5_forecasting",
        "created_at": "2026-11-15T21:00:00+00:00",
        "configuration": {"seed": 42},
        "seed": 42,
        "code_version": {"commit": "abc123"},
        "data_sha256": "f" * 64,
        "metrics": {"mae": 3.6},
    }
    write_json(record, tmp_path / "e5_forecasting_test_d1.json")
    write_json({"frozen_at": "x"}, tmp_path / "protocol_freeze.json")

    with Session(clean_engine) as session:
        assert import_experiments(session, tmp_path) == 1
        write_json({**record, "metrics": {"mae": 3.5}}, tmp_path / "e5_forecasting_test_d1.json")
        import_experiments(session, tmp_path)
        runs = session.scalars(select(ExperimentRun)).all()

    assert len(runs) == 1
    assert runs[0].commit == "abc123"
    assert runs[0].metrics == {"mae": 3.5}
