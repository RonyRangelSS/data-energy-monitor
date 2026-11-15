import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_monitor.common.indicators import compute_indicators, energy_kwh, idle_mask
from energy_monitor.common.splitting import compute_boundaries
from energy_monitor.experiments import e4_indicators
from energy_monitor.experiments.run_record import write_json
from tests.unit.forecast_helpers import make_grid

START = pd.Timestamp("2019-05-01 00:00:00", tz="UTC")
STEP = pd.Timedelta(minutes=10)


def frame_with_cpu(cpu: list[float], power: float = 120.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "host_id": "h1",
            "ts": [START + position * STEP for position in range(len(cpu))],
            "cpu_load": cpu,
            "power": power,
        }
    )


def test_energy_of_one_hour_at_600_watts_is_point_six_kwh() -> None:
    assert energy_kwh(np.full(6, 600.0)) == pytest.approx(0.6)


def test_idle_requires_three_consecutive_low_readings() -> None:
    frame = frame_with_cpu([1, 1, 50, 1, 1, 1, 50])

    assert idle_mask(frame, 5.0).tolist() == [False, False, False, True, True, True, False]


def test_idle_runs_do_not_cross_hosts() -> None:
    frame = pd.concat(
        [frame_with_cpu([1, 1]), frame_with_cpu([1]).assign(host_id="h2")], ignore_index=True
    )

    assert not idle_mask(frame, 5.0).any()


def test_indicators_apply_tariff_to_energy() -> None:
    frame = frame_with_cpu([1, 1, 1, 50, 50, 50], power=600.0)

    result = compute_indicators(frame, tariff_kwh=2.0, idle_threshold=5.0)

    assert result["total"]["energy_kwh"] == pytest.approx(0.6)
    assert result["total"]["idle_energy_kwh"] == pytest.approx(0.3)
    assert result["total"]["idle_energy_share"] == pytest.approx(0.5)
    assert result["total"]["associated_cost"] == pytest.approx(1.2)
    assert result["per_host"]["h1"]["idle_time_share"] == pytest.approx(0.5)


def test_daily_block_bootstrap_is_reproducible_and_brackets_estimate() -> None:
    days = pd.DataFrame(
        {
            "energy": [10.0, 12.0, 11.0, 13.0],
            "idle_energy": [5.0, 4.0, 6.0, 5.0],
            "idle_readings": [50, 40, 60, 50],
            "readings": [100, 100, 100, 100],
        }
    )

    first = e4_indicators.daily_block_bootstrap(days, e4_indicators.idle_energy_share, 200, 3, 0.95)
    second = e4_indicators.daily_block_bootstrap(
        days, e4_indicators.idle_energy_share, 200, 3, 0.95
    )

    assert first == second
    assert first["lower"] <= first["estimate"] <= first["upper"]


def test_e4_runs_on_test_period(tmp_path: Path) -> None:
    grid = make_grid(144 * 10)
    processed = tmp_path / "processed"
    processed.mkdir()
    grid.to_parquet(processed / "d1_grid.parquet", index=False)
    (tmp_path / "raw.csv").write_text("x\n", encoding="utf-8")
    boundaries = compute_boundaries(grid["ts"].min(), grid["ts"].max())
    split = {
        name: {"utc": getattr(boundaries, name).isoformat()}
        for name in ("start", "train_end", "validation_end", "end")
    }
    write_json(
        {"metrics": {"split": split, "coverage": {"eligible_hosts": ["h1", "h2"]}}},
        tmp_path / "e1_data_quality_d1.json",
    )
    config = {
        "seed": 1,
        "processed_dir": str(processed),
        "e1_report_dir": str(tmp_path),
        "output_dir": str(tmp_path / "out"),
        "datasets": {"d1": str(tmp_path / "raw.csv")},
        "idle_thresholds": [5.0, 15.0],
        "tariff_kwh": 0.8,
        "bootstrap_resamples": 50,
        "confidence": 0.95,
    }

    metrics = json.loads(e4_indicators.run("d1", config).read_text(encoding="utf-8"))["metrics"]

    assert set(metrics["thresholds"]) == {"5", "15"}
    share = metrics["thresholds"]["15"]["bootstrap"]["idle_time_share"]
    assert share["lower"] <= share["estimate"] <= share["upper"]
    assert metrics["period_utc"][0] >= boundaries.validation_end.isoformat()
