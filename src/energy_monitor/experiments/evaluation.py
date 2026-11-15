from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

WEEK: Final = pd.Timedelta(days=7)
ACTUAL: Final = "target"
FORECAST: Final = "prediction"


@dataclass(frozen=True)
class SeasonalScale:
    per_host: pd.Series
    total: float


def mean_absolute_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def root_mean_squared_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((actual >= lower) & (actual <= upper)))


def build_predictions(rows: pd.DataFrame, forecast: np.ndarray) -> pd.DataFrame:
    frame = rows[["host_id", "target_ts", ACTUAL]].copy()
    frame[ACTUAL] = frame[ACTUAL].astype(np.float64)
    frame[FORECAST] = forecast
    return frame


def summarize_hosts(predictions: pd.DataFrame, per_host_scale: pd.Series) -> pd.DataFrame:
    errors = predictions[ACTUAL] - predictions[FORECAST]
    hosts = predictions["host_id"]
    summary = pd.DataFrame(
        {
            "mae": errors.abs().groupby(hosts, observed=True).mean(),
            "rmse": np.sqrt((errors**2).groupby(hosts, observed=True).mean()),
        }
    )
    summary["mase"] = summary["mae"] / per_host_scale.reindex(summary.index).replace(0, np.nan)
    return summary


def summarize_total(
    predictions: pd.DataFrame, total_scale: float, eligible_hosts: int
) -> dict[str, Any]:
    per_instant = predictions.groupby("target_ts").agg(
        target=(ACTUAL, "sum"), prediction=(FORECAST, "sum"), hosts=("host_id", "nunique")
    )
    actual, forecast = per_instant[ACTUAL].to_numpy(), per_instant[FORECAST].to_numpy()
    mae = mean_absolute_error(actual, forecast)
    left_out = eligible_hosts - per_instant["hosts"]
    return {
        "mae": mae,
        "rmse": root_mean_squared_error(actual, forecast),
        "mase": mae / total_scale if total_scale > 0 else None,
        "instants": int(len(per_instant)),
        "mean_hosts_left_out": float(left_out.mean()),
        "max_hosts_left_out": int(left_out.max()),
    }


def summarize_weekly_blocks(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    start = predictions["target_ts"].min()
    block = ((predictions["target_ts"] - start) / WEEK).astype(np.int64)
    errors = (predictions[ACTUAL] - predictions[FORECAST]).abs()
    weekly = errors.groupby(block).agg(["mean", "size"])
    return [
        {"week": int(week), "mae": float(row["mean"]), "rows": int(row["size"])}
        for week, row in weekly.iterrows()
    ]


def evaluate_forecast(
    predictions: pd.DataFrame,
    scale: SeasonalScale,
    eligible_hosts: int,
    include_details: bool = False,
) -> dict[str, Any]:
    actual, forecast = predictions[ACTUAL].to_numpy(), predictions[FORECAST].to_numpy()
    hosts = summarize_hosts(predictions, scale.per_host)
    result = {
        "rows": int(len(predictions)),
        "mae": mean_absolute_error(actual, forecast),
        "rmse": root_mean_squared_error(actual, forecast),
        "mase": float(hosts["mase"].mean()),
        "mase_median": float(hosts["mase"].median()),
        "total": summarize_total(predictions, scale.total, eligible_hosts),
    }
    if include_details:
        result["per_host"] = hosts.round(6).to_dict(orient="index")
        result["weekly_blocks"] = summarize_weekly_blocks(predictions)
    return result


def compare_paired_errors(first: pd.Series, second: pd.Series) -> dict[str, Any]:
    paired = pd.concat([first, second], axis=1, keys=["first", "second"]).dropna()
    differences = paired["first"] - paired["second"]
    nonzero = differences[differences != 0]
    if nonzero.empty:
        return {
            "hosts": int(len(paired)),
            "statistic": None,
            "p_value": None,
            "rank_biserial": 0.0,
            "first_better_hosts": 0,
            "second_better_hosts": 0,
        }
    test = wilcoxon(nonzero)
    ranks = nonzero.abs().rank()
    positive, negative = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    return {
        "hosts": int(len(paired)),
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "rank_biserial": float((negative - positive) / (negative + positive)),
        "first_better_hosts": int((nonzero < 0).sum()),
        "second_better_hosts": int((nonzero > 0).sum()),
    }


def percent_reduction(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline * 100


def holm_adjust(p_values: dict[str, float | None]) -> dict[str, float | None]:
    tested = sorted((value, key) for key, value in p_values.items() if value is not None)
    adjusted: dict[str, float | None] = {key: None for key in p_values}
    running_max = 0.0
    for position, (value, key) in enumerate(tested):
        running_max = max(running_max, (len(tested) - position) * value)
        adjusted[key] = min(1.0, running_max)
    return adjusted
