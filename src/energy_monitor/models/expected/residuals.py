from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from energy_monitor.models.base import ExpectedConsumptionModel

MAD_SCALE: Final = 1.4826


@dataclass(frozen=True)
class ResidualStatistics:
    median: pd.Series
    mad: pd.Series

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            host: {"median": float(self.median[host]), "mad": float(self.mad[host])}
            for host in self.median.index
        }

    @classmethod
    def from_dict(cls, values: dict[str, dict[str, float]]) -> "ResidualStatistics":
        frame = pd.DataFrame.from_dict(values, orient="index")
        return cls(median=frame["median"], mad=frame["mad"])


def compute_residual_statistics(hosts: pd.Series, residuals: np.ndarray) -> ResidualStatistics:
    frame = pd.DataFrame({"host": hosts.astype(str).to_numpy(), "residual": residuals})
    median = frame.groupby("host")["residual"].median()
    deviation = (frame["residual"] - frame["host"].map(median)).abs()
    return ResidualStatistics(median=median, mad=deviation.groupby(frame["host"]).median())


def standardized_deviation(
    frame: pd.DataFrame, expected: np.ndarray, statistics: ResidualStatistics
) -> np.ndarray:
    hosts = frame["host_id"].astype(str)
    median = hosts.map(statistics.median).to_numpy(dtype=np.float64)
    scale = MAD_SCALE * hosts.map(statistics.mad).to_numpy(dtype=np.float64)
    scale = np.where(scale > 0, scale, np.nan)
    return (frame["power"].to_numpy(dtype=np.float64) - expected - median) / scale


def contiguous_time_folds(timestamps: pd.Series, folds: int) -> np.ndarray:
    instants = np.sort(timestamps.unique())
    block = np.minimum(np.arange(len(instants)) * folds // len(instants), folds - 1)
    lookup = pd.Series(block, index=pd.DatetimeIndex(instants))
    return lookup.reindex(pd.DatetimeIndex(timestamps)).to_numpy()


def out_of_fold_predictions(
    model_factory: Callable[[], ExpectedConsumptionModel],
    fit_frame: pd.DataFrame,
    statistics_mask: np.ndarray,
    folds: int,
) -> np.ndarray:
    statistics_index = np.flatnonzero(statistics_mask)
    fold_of_row = contiguous_time_folds(fit_frame["ts"].iloc[statistics_index], folds)
    predictions = np.full(len(statistics_index), np.nan)
    for fold in range(folds):
        held_out = statistics_index[fold_of_row == fold]
        training = fit_frame.drop(index=fit_frame.index[held_out])
        model = model_factory().fit(training, training["power"])
        predictions[fold_of_row == fold] = model.predict(fit_frame.iloc[held_out])
    return predictions
