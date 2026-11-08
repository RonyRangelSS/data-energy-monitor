import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from energy_monitor.common.preparation import prepare_raw_frame
from energy_monitor.common.quality_report import build_quality_report
from energy_monitor.common.splitting import compute_boundaries
from energy_monitor.experiments.e1_data_quality import main
from energy_monitor.experiments.run_record import hash_file


def build_rows(row: Any, host: str, slots: int, start: str = "2019-01-01 00:00:00") -> list[dict]:
    moments = pd.date_range(start, periods=slots, freq="10min")
    return [row(host, moment.strftime("%Y-%m-%d %H:%M:%S")) for moment in moments]


def build_report(raw: pd.DataFrame, min_coverage: float = 0.8) -> dict[str, Any]:
    prepared = prepare_raw_frame(raw)
    slots = prepared.grid["ts"]
    boundaries = compute_boundaries(slots.min(), slots.max())
    return build_quality_report(prepared, boundaries, min_coverage)


def test_report_counts_flags_and_inconsistencies(raw_frame, row) -> None:
    rows = build_rows(row, "h1", 10)
    rows[1]["Power"] = 0.0
    rows[2]["CPU_Load"] = 500.0
    rows[3]["Power"] = None
    rows[4]["CPU_cores_used"] = 99.0

    report = build_report(raw_frame(rows))

    assert report["raw"]["rows"] == 10
    assert report["quality_flags"]["non_positive_power"] == 1
    assert report["quality_flags"]["cpu_out_of_range"] == 1
    assert report["quality_flags"]["missing_target"] == 1
    assert report["quality_flags"]["any_flag"] == 3
    assert report["inconsistencies_without_flag"]["cores_used_above_cores"] == 1
    assert report["missing_by_column"]["power"]["count"] == 1


def test_report_separates_daylight_saving_repeats_from_true_duplicates(raw_frame, row) -> None:
    moments = ["2019-04-07 01:50:00", "2019-04-07 02:00:00", "2019-04-07 02:50:00"]
    moments += ["2019-04-07 02:00:00"]

    report = build_report(raw_frame([row("h1", moment) for moment in moments]))

    assert report["repeated_local_times"] == 1
    assert report["true_duplicates_after_utc"] == 0


def test_report_measures_gaps(raw_frame, row) -> None:
    rows = build_rows(row, "h1", 2)
    rows += build_rows(row, "h1", 2, start="2019-01-01 07:00:00")

    gaps = build_report(raw_frame(rows))["gaps"]

    assert gaps["gaps_between_readings"] == 1
    assert gaps["gaps_of_6h_or_more"] == 1
    assert gaps["longest_gap_hours"] == pytest.approx(6.83, abs=0.01)
    assert gaps["unobserved_grid_slots"] == 40


def test_report_lists_excluded_hosts(raw_frame, row) -> None:
    rows = build_rows(row, "complete", 100) + build_rows(row, "partial", 60)

    coverage = build_report(raw_frame(rows))["coverage"]

    assert coverage["eligible_hosts"] == ["complete"]
    assert coverage["excluded_hosts"] == ["partial"]


def test_report_records_sensor_floor_only_when_columns_exist(raw_frame, row) -> None:
    without_sensors = build_report(raw_frame(build_rows(row, "h1", 10)))
    rows = build_rows(row, "h1", 10)
    for item in rows:
        item["Inlet_Temperature"], item["Fan_speed1"] = 4.0, 280.0

    with_sensors = build_report(raw_frame(rows))

    assert "sensor_floor_readings" not in without_sensors["inconsistencies_without_flag"]
    assert with_sensors["inconsistencies_without_flag"]["sensor_floor_readings"] == 10


def test_report_finds_core_count_changes(raw_frame, row) -> None:
    rows = build_rows(row, "h1", 4)
    for item in rows[2:]:
        item["CPU_cores"] = 128.0

    changes = build_report(raw_frame(rows))["core_count_changes_by_local_date"]

    assert changes == {"2019-01-01": 1}


def test_command_generates_grid_and_traceable_report(raw_frame, row, tmp_path: Path) -> None:
    data_path = tmp_path / "raw.csv"
    raw_frame(build_rows(row, "h1", 100) + build_rows(row, "h2", 100)).to_csv(
        data_path, index=False
    )
    config_path = tmp_path / "config.yaml"
    config = {
        "experiment": "e1_data_quality",
        "source_timezone": "Australia/Melbourne",
        "analysis_start_local": "2019-01-01T00:00:00",
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "min_host_coverage": 0.8,
        "processed_dir": str(tmp_path / "processed"),
        "output_dir": str(tmp_path / "artifacts"),
        "datasets": {"d1": str(data_path)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    main(["--config", str(config_path)])

    record = json.loads((tmp_path / "artifacts/e1_data_quality_d1.json").read_text("utf-8"))
    grid = pd.read_parquet(tmp_path / "processed/d1_grid.parquet")
    assert record["data_sha256"] == hash_file(data_path)
    assert record["configuration"]["min_host_coverage"] == 0.8
    assert {"commit", "has_uncommitted_changes"} <= set(record["code_version"])
    assert "pandas" in record["package_versions"]
    assert record["metrics"]["coverage"]["eligible_count"] == 2
    assert len(grid) == 200
