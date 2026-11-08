from pathlib import Path

import pandas as pd
import pytest

from energy_monitor.common.preparation import (
    compute_memory_percent,
    flag_quality,
    has_any_flag,
    is_valid_reading,
    prepare_raw_frame,
    read_raw_csv,
    standardize,
)
from energy_monitor.common.schema import QualityFlag


def flags_of(prepared_row: pd.Series) -> QualityFlag:
    return QualityFlag(int(prepared_row["quality_mask"]))


def test_standardize_strips_host_and_computes_memory_percent(raw_frame, row) -> None:
    frame = standardize(raw_frame([row("rcc120 ", "2019-01-01 00:00:00")]))

    assert frame.loc[0, "host_id"] == "rcc120"
    assert frame.loc[0, "mem_pct"] == pytest.approx(50.0)


def test_memory_percent_is_missing_when_ram_is_zero() -> None:
    result = compute_memory_percent(pd.Series([0.0, 64.0]), pd.Series([10.0, 32.0]))

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(50.0)


def test_read_raw_csv_rejects_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("Series,Time\nh1,2019-01-01 00:00:00\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Power"):
        read_raw_csv(path)


def test_read_raw_csv_accepts_dataset_without_optional_columns(raw_frame, row, tmp_path) -> None:
    path = tmp_path / "d2.csv"
    raw_frame([row("h1", "2019-01-01 00:00:00")]).to_csv(path, index=False)

    assert "Inlet_Temperature" not in read_raw_csv(path).columns


def test_repeated_daylight_saving_hour_becomes_distinct_utc_instants(raw_frame, row) -> None:
    times = ["2019-04-07 01:50:00"]
    times += [f"2019-04-07 02:{minute:02d}:00" for minute in range(0, 60, 10)] * 2
    times += ["2019-04-07 03:00:00"]

    prepared = prepare_raw_frame(raw_frame([row("h1", time) for time in times]))

    timestamps = prepared.standardized["ts"]
    assert timestamps.diff().dropna().eq(pd.Timedelta(minutes=10)).all()
    assert timestamps.iloc[1] == pd.Timestamp("2019-04-06 15:00:00", tz="UTC")
    assert timestamps.iloc[7] == pd.Timestamp("2019-04-06 16:00:00", tz="UTC")
    assert prepared.standardized["quality_mask"].eq(0).all()


def test_true_duplicate_is_flagged_and_first_reading_is_kept(raw_frame, row) -> None:
    rows = [row("h1", "2019-01-01 00:00:00", power=100.0)]
    rows.append(row("h1", "2019-01-01 00:00:00", power=999.0))

    prepared = prepare_raw_frame(raw_frame(rows))

    assert flags_of(prepared.standardized.iloc[1]) == QualityFlag.DUPLICATE
    assert prepared.grid["power"].tolist() == [100.0]


@pytest.mark.parametrize(
    ("power", "cpu", "expected"),
    [
        (0.0, 10.0, QualityFlag.NON_POSITIVE_POWER),
        (-5.0, 10.0, QualityFlag.NON_POSITIVE_POWER),
        (100.0, 100.5, QualityFlag.CPU_OUT_OF_RANGE),
        (100.0, -0.1, QualityFlag.CPU_OUT_OF_RANGE),
        (None, 10.0, QualityFlag.MISSING_TARGET),
        (0.0, 150.0, QualityFlag.NON_POSITIVE_POWER | QualityFlag.CPU_OUT_OF_RANGE),
    ],
)
def test_invalid_readings_receive_coded_reasons(raw_frame, row, power, cpu, expected) -> None:
    prepared = prepare_raw_frame(raw_frame([row("h1", "2019-01-01 00:00:00", power, cpu)]))

    assert flags_of(prepared.standardized.iloc[0]) == expected


def test_valid_boundaries_of_cpu_are_accepted(raw_frame, row) -> None:
    rows = [row("h1", "2019-01-01 00:00:00", cpu=0.0), row("h1", "2019-01-01 00:10:00", cpu=100.0)]

    prepared = prepare_raw_frame(raw_frame(rows))

    assert prepared.standardized["quality_mask"].eq(0).all()


def test_off_grid_reading_is_flagged_and_left_out_of_grid(raw_frame, row) -> None:
    rows = [
        row("h1", "2019-01-01 00:00:00"),
        row("h1", "2019-01-01 00:07:00"),
        row("h1", "2019-01-01 00:10:00"),
    ]

    prepared = prepare_raw_frame(raw_frame(rows))

    assert flags_of(prepared.standardized.iloc[1]) == QualityFlag.OFF_GRID
    assert len(prepared.grid) == 2


def test_invalid_timestamp_is_flagged_as_off_grid(raw_frame, row) -> None:
    rows = [row("h1", "2019-01-01 00:00:00"), row("h1", "not-a-date")]

    prepared = prepare_raw_frame(raw_frame(rows))

    assert flags_of(prepared.standardized.iloc[1]) == QualityFlag.OFF_GRID
    assert len(prepared.grid) == 1


def test_gaps_stay_missing_and_are_never_interpolated(raw_frame, row) -> None:
    rows = [
        row("h1", "2019-01-01 00:00:00", power=100.0),
        row("h1", "2019-01-01 00:30:00", power=200.0),
    ]

    grid = prepare_raw_frame(raw_frame(rows)).grid

    assert len(grid) == 4
    assert grid["power"].isna().tolist() == [False, True, True, False]
    assert grid["observed"].tolist() == [True, False, False, True]


def test_all_hosts_share_the_same_grid(raw_frame, row) -> None:
    rows = [
        row("h1", "2019-01-01 00:00:00"),
        row("h1", "2019-01-01 00:10:00"),
        row("h2", "2019-01-01 00:10:00"),
        row("h2", "2019-01-01 00:20:00"),
    ]

    grid = prepare_raw_frame(raw_frame(rows)).grid

    assert grid.groupby("host_id").size().tolist() == [3, 3]


def test_flagged_reading_stays_on_grid_but_is_not_valid(raw_frame, row) -> None:
    rows = [row("h1", "2019-01-01 00:00:00"), row("h1", "2019-01-01 00:10:00", power=0.0)]

    grid = prepare_raw_frame(raw_frame(rows)).grid

    assert is_valid_reading(grid).tolist() == [True, False]


def test_batch_and_single_reading_validation_agree(raw_frame, row) -> None:
    rows = [
        row("h1", "2019-01-01 00:00:00", power=0.0),
        row("h2", "2019-01-01 00:00:00", cpu=120.0),
        row("h3", "2019-01-01 00:00:00", power=None),
        row("h4", "2019-01-01 00:03:00"),
        row("h5", "2019-01-01 00:00:00"),
    ]
    batch = prepare_raw_frame(raw_frame(rows)).standardized

    single = [
        int(prepare_raw_frame(raw_frame([item])).standardized["quality_mask"].iloc[0])
        for item in rows
    ]

    assert batch["quality_mask"].tolist() == single


def test_flag_quality_output_is_aligned_with_input_index(raw_frame, row) -> None:
    standardized = prepare_raw_frame(
        raw_frame([row("h1", "2019-01-01 00:00:00", power=0.0)])
    ).standardized

    mask = flag_quality(standardized.set_axis([7]))

    assert mask.index.tolist() == [7]
    assert has_any_flag(mask, QualityFlag.NON_POSITIVE_POWER).tolist() == [True]
