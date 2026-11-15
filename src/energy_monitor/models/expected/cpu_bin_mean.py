from typing import ClassVar, Final, Self

import numpy as np
import pandas as pd

from energy_monitor.models.base import ExpectedConsumptionModel
from energy_monitor.models.expected.catalog import expected_registry

CPU_BIN_EDGES: Final = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, float("inf"))
LOWER_QUANTILE: Final = 0.05
UPPER_QUANTILE: Final = 0.95


def summarize_power(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = data.groupby(keys)["power"]
    return pd.DataFrame(
        {
            "mean": grouped.mean(),
            "lower": grouped.quantile(LOWER_QUANTILE),
            "upper": grouped.quantile(UPPER_QUANTILE),
        }
    )


@expected_registry.register
class CpuBinMeanModel(ExpectedConsumptionModel):
    name: ClassVar[str] = "cpu_bin_mean"
    version: ClassVar[str] = "1.0.0"
    supports_interval: ClassVar[bool] = True

    def _keys(self, frame: pd.DataFrame) -> pd.DataFrame:
        cpu_bin = pd.cut(frame["cpu_load"], CPU_BIN_EDGES, right=False, labels=False)
        return pd.DataFrame(
            {"host": frame["host_id"].astype(str).to_numpy(), "cpu_bin": cpu_bin.to_numpy()}
        )

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> Self:
        data = self._keys(frame).assign(power=np.asarray(target, dtype=np.float64))
        self._by_host_and_bin = summarize_power(data, ["host", "cpu_bin"])
        self._by_bin = summarize_power(data, ["cpu_bin"])
        self._overall = summarize_power(data.assign(all=0), ["all"]).iloc[0]
        return self

    def _lookup(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        keys = self._keys(frame)
        by_host = keys.join(self._by_host_and_bin[column], on=["host", "cpu_bin"])[column]
        by_bin = keys.join(self._by_bin[column], on="cpu_bin")[column]
        return by_host.fillna(by_bin).fillna(self._overall[column]).to_numpy(dtype=np.float64)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._lookup(frame, "mean")

    def predict_interval(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self._lookup(frame, "lower"), self._lookup(frame, "upper")
