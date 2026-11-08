import pandas as pd
import pytest

from energy_monitor.common.splitting import (
    PRE_ANALYSIS,
    TEST,
    TRAIN,
    VALIDATION,
    SplitBoundaries,
    assign_partition,
    compute_boundaries,
    compute_host_coverage,
    local_to_utc,
    select_eligible_hosts,
)

START = pd.Timestamp("2019-01-01 00:00:00", tz="UTC")
STEP = pd.Timedelta(minutes=10)


def slots(count: int) -> pd.Series:
    return pd.Series(pd.date_range(START, periods=count, freq="10min"))


def make_grid(valid_by_host: dict[str, list[bool]]) -> pd.DataFrame:
    frames = []
    for host, valid in valid_by_host.items():
        frame = pd.DataFrame({"host_id": host, "ts": slots(len(valid))})
        frame["observed"] = valid
        frame["quality_mask"] = 0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def boundaries_for_ten_slots() -> SplitBoundaries:
    return compute_boundaries(START, START + 9 * STEP)


def test_boundaries_follow_fractions() -> None:
    boundaries = boundaries_for_ten_slots()

    assert boundaries.train_end == START + 6 * STEP
    assert boundaries.validation_end == START + 8 * STEP


def test_no_validation_or_test_reading_precedes_end_of_train() -> None:
    timestamps = slots(1000)
    boundaries = compute_boundaries(timestamps.min(), timestamps.max())

    partition = assign_partition(timestamps, boundaries)

    last_train = timestamps[partition == TRAIN].max()
    assert timestamps[partition == VALIDATION].min() > last_train
    assert timestamps[partition == TEST].min() > timestamps[partition == VALIDATION].max()


def test_partitions_cover_every_slot_without_overlap() -> None:
    timestamps = slots(10)

    partition = assign_partition(timestamps, boundaries_for_ten_slots())

    assert partition.value_counts().to_dict() == {TRAIN: 6, VALIDATION: 2, TEST: 2}


def test_analysis_start_excludes_earlier_slots_from_all_partitions() -> None:
    timestamps = slots(20)
    analysis_start = START + 10 * STEP

    boundaries = compute_boundaries(timestamps.min(), timestamps.max(), analysis_start)
    partition = assign_partition(timestamps, boundaries)

    assert (partition.iloc[:10] == PRE_ANALYSIS).all()
    assert boundaries.start == analysis_start
    assert boundaries.train_end == analysis_start + 6 * STEP


def test_analysis_start_is_aligned_to_the_next_slot() -> None:
    boundaries = compute_boundaries(START, START + 99 * STEP, START + pd.Timedelta(minutes=3))

    assert boundaries.start == START + STEP


def test_analysis_start_before_grid_uses_grid_start() -> None:
    boundaries = compute_boundaries(START, START + 9 * STEP, START - 100 * STEP)

    assert boundaries.start == START


@pytest.mark.parametrize(("train", "validation"), [(0.8, 0.2), (0.0, 0.2), (0.6, 0.0), (0.9, 0.3)])
def test_invalid_fractions_are_rejected(train: float, validation: float) -> None:
    with pytest.raises(ValueError):
        compute_boundaries(START, START + 9 * STEP, None, train, validation)


def test_analysis_start_after_grid_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_boundaries(START, START + 9 * STEP, START + 100 * STEP)


def test_local_to_utc_applies_melbourne_offset() -> None:
    converted = local_to_utc("2019-03-15T00:00:00", "Australia/Melbourne")

    assert converted == pd.Timestamp("2019-03-14 13:00:00", tz="UTC")


def test_host_is_eligible_only_when_every_partition_reaches_minimum() -> None:
    complete = [True] * 10
    weak_test = [True] * 8 + [False] * 2
    grid = make_grid({"complete": complete, "weak_test": weak_test})

    coverage = compute_host_coverage(grid, boundaries_for_ten_slots())

    assert coverage.loc["weak_test", TEST] == pytest.approx(0.0)
    assert select_eligible_hosts(coverage, 0.8) == ["complete"]


def test_coverage_ignores_flagged_readings() -> None:
    grid = make_grid({"flagged": [True] * 10})
    grid.loc[0:5, "quality_mask"] = 8

    coverage = compute_host_coverage(grid, boundaries_for_ten_slots())

    assert coverage.loc["flagged", TRAIN] == pytest.approx(0.0)


def test_host_at_exact_minimum_coverage_is_eligible() -> None:
    coverage = pd.DataFrame(
        {TRAIN: [0.8], VALIDATION: [1.0], TEST: [1.0]}, index=pd.Index(["limit"], name="host_id")
    )

    assert select_eligible_hosts(coverage, 0.8) == ["limit"]
