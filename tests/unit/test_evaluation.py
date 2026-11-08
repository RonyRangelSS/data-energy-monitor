import numpy as np
import pandas as pd
import pytest

from energy_monitor.experiments.evaluation import (
    SeasonalScale,
    build_predictions,
    compare_paired_errors,
    evaluate_forecast,
    interval_coverage,
    mean_absolute_error,
    percent_reduction,
    root_mean_squared_error,
    summarize_total,
    summarize_weekly_blocks,
)

MOMENT = pd.Timestamp("2019-05-01 00:00:00", tz="UTC")
STEP = pd.Timedelta(minutes=10)


def predictions_for(
    hosts: list[str], slots: list[int], actual: list[float], forecast: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "host_id": hosts,
            "target_ts": [MOMENT + slot * STEP for slot in slots],
            "target": actual,
            "prediction": forecast,
        }
    )


def test_error_metrics_match_manual_calculation() -> None:
    actual, forecast = np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0])

    assert mean_absolute_error(actual, forecast) == pytest.approx(7 / 3)
    assert root_mean_squared_error(actual, forecast) == pytest.approx(np.sqrt(17 / 3))


def test_interval_coverage_counts_bounds_as_inside() -> None:
    actual = np.array([1.0, 5.0, 9.0, 12.0])

    assert interval_coverage(actual, np.full(4, 1.0), np.full(4, 9.0)) == pytest.approx(0.75)


def test_evaluate_forecast_averages_mase_across_hosts() -> None:
    predictions = predictions_for(
        ["h1", "h1", "h2", "h2"], [0, 1, 0, 1], [10.0, 10.0, 100.0, 100.0], [8.0, 12.0, 95.0, 105.0]
    )
    scale = SeasonalScale(per_host=pd.Series({"h1": 4.0, "h2": 2.5}), total=5.0)

    result = evaluate_forecast(predictions, scale, eligible_hosts=2)

    assert result["mae"] == pytest.approx(3.5)
    assert result["mase"] == pytest.approx((2 / 4 + 5 / 2.5) / 2)


def test_total_compares_only_hosts_with_forecast_and_counts_the_rest() -> None:
    predictions = predictions_for(
        ["h1", "h2", "h1"], [0, 0, 1], [10.0, 20.0, 15.0], [11.0, 21.0, 14.0]
    )

    total = summarize_total(predictions, total_scale=2.0, eligible_hosts=3)

    assert total["mae"] == pytest.approx((2.0 + 1.0) / 2)
    assert total["mase"] == pytest.approx(0.75)
    assert total["mean_hosts_left_out"] == pytest.approx(1.5)
    assert total["max_hosts_left_out"] == 2


def test_total_mase_is_absent_when_scale_is_zero() -> None:
    predictions = predictions_for(["h1"], [0], [10.0], [11.0])

    assert summarize_total(predictions, 0.0, 1)["mase"] is None


def test_details_are_included_only_when_requested() -> None:
    predictions = predictions_for(["h1", "h1"], [0, 1], [10.0, 10.0], [8.0, 12.0])
    scale = SeasonalScale(per_host=pd.Series({"h1": 4.0}), total=5.0)

    assert "per_host" not in evaluate_forecast(predictions, scale, 1)
    detailed = evaluate_forecast(predictions, scale, 1, include_details=True)
    assert detailed["per_host"]["h1"]["mae"] == pytest.approx(2.0)
    assert detailed["weekly_blocks"][0]["rows"] == 2


def test_weekly_blocks_split_by_seven_days() -> None:
    week_slots = int(pd.Timedelta(days=7) / STEP)
    predictions = predictions_for(
        ["h1", "h1", "h1"], [0, week_slots - 1, week_slots], [10.0, 10.0, 10.0], [9.0, 9.0, 5.0]
    )

    blocks = summarize_weekly_blocks(predictions)

    assert [block["rows"] for block in blocks] == [2, 1]
    assert blocks[1]["mae"] == pytest.approx(5.0)


def test_build_predictions_keeps_rows_aligned() -> None:
    rows = pd.DataFrame(
        {
            "host_id": ["h1"],
            "target_ts": [MOMENT],
            "target": np.array([3.5], dtype=np.float32),
            "extra": [1],
        }
    )

    predictions = build_predictions(rows, np.array([4.0]))

    assert list(predictions.columns) == ["host_id", "target_ts", "target", "prediction"]
    assert predictions["target"].dtype == np.float64


def test_paired_comparison_detects_the_better_model() -> None:
    first = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=list("abcdef"))
    second = first + pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=list("abcdef"))

    result = compare_paired_errors(first, second)

    assert result["first_better_hosts"] == 6
    assert result["rank_biserial"] == pytest.approx(1.0)
    assert result["p_value"] < 0.05


def test_paired_comparison_of_identical_errors_has_no_effect() -> None:
    errors = pd.Series([1.0, 2.0], index=["a", "b"])

    result = compare_paired_errors(errors, errors)

    assert result["p_value"] is None
    assert result["rank_biserial"] == 0.0


def test_percent_reduction_is_positive_when_candidate_is_better() -> None:
    assert percent_reduction(10.0, 8.0) == pytest.approx(20.0)
