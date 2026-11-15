from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from energy_monitor.experiments.detection_metrics import (
    episodes_per_host_day,
    evaluate_detection,
    evaluate_negative_control,
    pointwise_scores,
)
from energy_monitor.replayer.injection import (
    ANOMALY_KINDS,
    InjectedEvent,
    InjectionPlan,
    apply_events,
    compute_host_reference,
    label_rows,
    load_injection_plan,
    plan_events,
)
from tests.unit.detection_helpers import make_expected_rows

START = pd.Timestamp("2019-03-01 00:00:00", tz="UTC")
STEP = pd.Timedelta(minutes=10)


def make_plan(
    seed: int = 1, count: int = 10, kinds: tuple[str, ...] = ANOMALY_KINDS
) -> InjectionPlan:
    return InjectionPlan(kinds, (0.1, 0.4), (3, 12), count, seed, START, START + 500 * STEP)


def event(kind: str, magnitude: float = 0.4, first: int = 10, slots: int = 4) -> InjectedEvent:
    start = START + first * STEP
    return InjectedEvent("h1", start, start + (slots - 1) * STEP, kind, magnitude, slots)


def test_same_seed_gives_same_events() -> None:
    assert plan_events(["h1", "h2"], make_plan(7)) == plan_events(["h1", "h2"], make_plan(7))
    assert plan_events(["h1", "h2"], make_plan(7)) != plan_events(["h1", "h2"], make_plan(8))


def test_events_on_same_host_never_overlap_and_stay_in_period() -> None:
    events = plan_events(["h1"], make_plan(count=15))

    for first, second in zip(events, events[1:], strict=False):
        assert second.start > first.end + 6 * STEP
    assert all(item.start >= START and item.end < START + 500 * STEP for item in events)


def injected(kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = make_expected_rows(60, hosts=("h1",))
    reference = compute_host_reference(rows)
    return rows, apply_events(rows, [event(kind)], reference)


def test_offset_raises_power_without_changing_load() -> None:
    rows, changed = injected("offset")
    mean = rows["power"].mean()

    np.testing.assert_allclose(changed["power"].iloc[10:14], rows["power"].iloc[10:14] + 0.4 * mean)
    pd.testing.assert_series_equal(changed["cpu_load"], rows["cpu_load"])
    pd.testing.assert_series_equal(changed["power"].iloc[:10], rows["power"].iloc[:10])


def test_drift_grows_linearly_until_full_magnitude() -> None:
    rows, changed = injected("drift")
    increase = (changed["power"] - rows["power"]).iloc[10:14].to_numpy()

    np.testing.assert_allclose(increase / increase[-1], [0.25, 0.5, 0.75, 1.0])


def test_stuck_freezes_power_at_first_value() -> None:
    rows, changed = injected("stuck")

    assert changed["power"].iloc[10:14].nunique() == 1
    assert changed["power"].iloc[13] == rows["power"].iloc[10]


def test_zero_load_high_power_clears_load_and_raises_power() -> None:
    rows, changed = injected("zero_load_high_power")

    assert (changed[["cpu_load", "cores_used", "n_vms"]].iloc[10:14] == 0).all().all()
    assert (changed["power"].iloc[10:14] > rows["power"].iloc[10:14]).all()


def test_legit_peak_raises_load_and_power_together() -> None:
    rows, changed = injected("legit_peak")

    assert (changed["cpu_load"].iloc[10:14] > rows["cpu_load"].iloc[10:14]).all()
    assert (changed["power"].iloc[10:14] > rows["power"].iloc[10:14]).all()
    assert not event("legit_peak").is_anomaly


def test_unknown_kind_is_rejected() -> None:
    rows = make_expected_rows(60, hosts=("h1",))

    with pytest.raises(ValueError):
        apply_events(rows, [event("unknown")], compute_host_reference(rows))


def test_label_rows_marks_only_event_readings() -> None:
    rows = make_expected_rows(30, hosts=("h1",))

    labels = label_rows(rows, [event("offset")])

    assert np.flatnonzero(labels).tolist() == [10, 11, 12, 13]


def test_plan_is_loaded_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "scenario.yaml"
    values = {
        "kinds": ["offset"],
        "magnitudes": [0.2],
        "durations_slots": [3],
        "count": 2,
        "seed": 9,
        "start": "2019-05-01T00:00:00+00:00",
        "end": "2019-05-02T00:00:00+00:00",
    }
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    plan = load_injection_plan(path)

    assert plan.seed == 9
    assert plan.kinds == ("offset",)
    assert plan.end - plan.start == pd.Timedelta(days=1)


def test_detection_metrics_count_events_episodes_and_delay() -> None:
    rows = make_expected_rows(60, hosts=("h1",))
    active = np.zeros(len(rows), dtype=bool)
    active[12:14] = True
    active[40:42] = True

    result = evaluate_detection(rows, active, [event("offset")])

    assert result["detected_events"] == 1
    assert result["alert_episodes"] == 2
    assert result["false_alert_episodes"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["mean_delay_minutes"] == 20.0


def test_missed_event_has_zero_recall_and_no_delay() -> None:
    rows = make_expected_rows(60, hosts=("h1",))

    result = evaluate_detection(rows, np.zeros(len(rows), dtype=bool), [event("offset")])

    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["mean_delay_minutes"] is None


def test_pointwise_scores_do_not_adjust_whole_event() -> None:
    labels = np.array([True, True, True, True, False])
    active = np.array([True, False, False, False, True])

    scores = pointwise_scores(labels, active)

    assert scores["point_precision"] == 0.5
    assert scores["point_recall"] == 0.25


def test_negative_control_counts_peaks_with_alert() -> None:
    rows = make_expected_rows(60, hosts=("h1",))
    active = np.zeros(len(rows), dtype=bool)
    active[11] = True

    result = evaluate_negative_control(rows, active, [event("legit_peak")])

    assert result["peaks_with_alert"] == 1
    assert result["false_positive_rate_on_peaks"] == 1.0


def test_episodes_per_host_day() -> None:
    rows = make_expected_rows(145, hosts=("h1",))
    active = np.zeros(len(rows), dtype=bool)
    active[5:7] = True

    assert episodes_per_host_day(rows, active) == pytest.approx(1.0)
