import numpy as np
import pandas as pd
import pytest

from energy_monitor.models.detection import detector_registry
from energy_monitor.models.detection.alerts import (
    describe_episode,
    extract_episodes,
    run_boundaries,
    unexplained_energy_kwh,
)
from energy_monitor.models.detection.base import Detector
from energy_monitor.models.detection.hysteresis import HysteresisState, apply_hysteresis, step
from energy_monitor.models.expected.residuals import ResidualStatistics
from tests.unit.detection_helpers import make_expected_rows


def hysteresis_for(
    scores: list[float], open_at: float = 3.5, close_below: float = 2.5
) -> list[bool]:
    values = np.array(scores)
    groups = np.array(["h1"] * len(values))
    return apply_hysteresis(groups, values >= open_at, values < close_below, 2).tolist()


def test_alert_opens_after_two_consecutive_high_readings() -> None:
    assert hysteresis_for([4, 1, 4, 4, 3]) == [False, False, False, True, True]


def test_alert_closes_after_two_consecutive_low_readings() -> None:
    assert hysteresis_for([4, 4, 2, 3, 2, 2, 4]) == [False, True, True, True, True, False, False]


def test_hysteresis_restarts_for_each_host() -> None:
    groups = np.array(["h1", "h1", "h2", "h2"])
    high = np.array([False, True, True, False])

    active = apply_hysteresis(groups, high, ~high, 2)

    assert active.tolist() == [False, False, False, False]


def test_step_keeps_counters_between_readings() -> None:
    state = step(HysteresisState(), True, False, 2)

    assert state == HysteresisState(active=False, above=1, below=0)
    assert step(state, True, False, 2).active


def test_all_detectors_are_registered() -> None:
    assert set(detector_registry.names()) == {"deviation", "fixed_threshold", "isolation_forest"}


def deviation_rows() -> pd.DataFrame:
    rows = make_expected_rows(20, hosts=("h1",))
    return rows.assign(expected=rows["power"] - 1.0, upper=rows["power"] + 5)


def test_deviation_detector_flags_power_above_expected() -> None:
    rows = deviation_rows()
    rows.loc[5:8, "power"] += 50
    statistics = ResidualStatistics(median=pd.Series({"h1": 1.0}), mad=pd.Series({"h1": 1.0}))
    detector = detector_registry.create("deviation", (), 42).fit_statistics(statistics)

    active = detector.detect(rows)

    assert active[6:9].all()
    assert not active[:6].any()


def test_deviation_detector_learns_statistics_from_residuals() -> None:
    frame = pd.DataFrame({"host_id": ["h1"] * 3, "residual": [1.0, 2.0, 3.0]})

    detector = detector_registry.create("deviation", (), 42).fit(frame)

    assert detector.statistics.median["h1"] == 2.0


def test_fixed_threshold_uses_percentile_of_training_power() -> None:
    rows = make_expected_rows(200, hosts=("h1",))
    detector = detector_registry.create("fixed_threshold", (), 42, percentile=95.0).fit(rows)
    limit = rows["power"].quantile(0.95)

    scores = detector.score(rows)

    np.testing.assert_allclose(scores, rows["power"] / limit)
    assert detector.thresholds() == (1.0, 1.0)


def test_fixed_threshold_rejects_percentile_outside_grid() -> None:
    rows = make_expected_rows(50, hosts=("h1",))
    detector = detector_registry.create("fixed_threshold", (), 42, percentile=95.3).fit(rows)

    with pytest.raises(ValueError):
        detector.score(rows)


def test_isolation_forest_scores_extreme_reading_higher() -> None:
    rows = make_expected_rows(300, hosts=("h1",))
    detector = detector_registry.create("isolation_forest", (), 42, n_estimators=50).fit(rows)
    extreme = rows.head(1).assign(power=rows["power"].max() * 3, cpu_load=0.0)

    assert detector.score(extreme)[0] > np.median(detector.score(rows))
    low, high = detector.with_parameters(percentile=50.0).thresholds()
    assert low == high


def test_with_parameters_does_not_change_original() -> None:
    detector = detector_registry.create("deviation", (), 42, open_threshold=3.5)

    variant = detector.with_parameters(open_threshold=5.0)

    assert detector.thresholds()[0] == 3.5
    assert variant.thresholds()[0] == 5.0


def test_detector_round_trip(tmp_path) -> None:
    rows = make_expected_rows(100, hosts=("h1",))
    detector = detector_registry.create("fixed_threshold", (), 42).fit(rows)
    detector.save(tmp_path / "detector.joblib")

    loaded = Detector.load(tmp_path / "detector.joblib")

    np.testing.assert_array_equal(loaded.detect(rows), detector.detect(rows))


def test_unexplained_energy_counts_only_power_above_upper_limit() -> None:
    energy = unexplained_energy_kwh(np.array([160.0, 90.0]), np.array([100.0, 100.0]))

    assert energy == pytest.approx(60 * (10 / 60) / 1000)


def test_episode_justification_is_readable() -> None:
    assert describe_episode(41.2, 8.0, 40) == (
        "potência 41% acima do esperado para CPU de 8%, durante 40 min"
    )


def test_run_boundaries_split_by_host_and_gaps() -> None:
    groups = np.array(["a", "a", "a", "b", "b"])
    active = np.array([True, True, False, True, True])

    assert run_boundaries(groups, active) == [(0, 1), (3, 4)]


def test_extract_episodes_summarizes_each_run() -> None:
    rows = deviation_rows()
    rows["expected"] = 100.0
    rows["upper"] = 110.0
    rows["power"] = 140.0
    active = np.zeros(len(rows), dtype=bool)
    active[2:6] = True

    episodes = extract_episodes(rows, active, np.arange(len(rows), dtype=float))

    assert len(episodes) == 1
    assert episodes[0].readings == 4
    assert episodes[0].peak_score == 5.0
    assert episodes[0].unexplained_energy_kwh == pytest.approx(4 * 30 * (10 / 60) / 1000)
    assert episodes[0].justification.startswith("potência 40% acima do esperado")
    assert episodes[0].to_dict()["start"] == rows["ts"].iloc[2].isoformat()
