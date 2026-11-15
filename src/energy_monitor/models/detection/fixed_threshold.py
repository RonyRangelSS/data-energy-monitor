from typing import ClassVar, Final, Self

import numpy as np
import pandas as pd

from energy_monitor.models.detection.base import Detector
from energy_monitor.models.detection.catalog import detector_registry

PERCENTILE_GRID: Final = np.round(np.arange(0, 100.5, 0.5), 1)


@detector_registry.register
class FixedThresholdDetector(Detector):
    name: ClassVar[str] = "fixed_threshold"
    version: ClassVar[str] = "1.0.0"
    scores_depend_on_parameters: ClassVar[bool] = True

    def fit(self, frame: pd.DataFrame) -> Self:
        power = frame["power"].astype(np.float64).groupby(frame["host_id"].astype(str))
        self._quantiles = power.quantile(PERCENTILE_GRID / 100).unstack()
        self._quantiles.columns = PERCENTILE_GRID
        return self

    def host_thresholds(self) -> pd.Series:
        percentile = round(float(self.hyperparameters.get("percentile", 95.0)), 1)
        if percentile not in self._quantiles.columns:
            raise ValueError(f"Percentil fora da grade de 0,5: {percentile}")
        return self._quantiles[percentile]

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        limit = frame["host_id"].astype(str).map(self.host_thresholds()).to_numpy(np.float64)
        return frame["power"].to_numpy(dtype=np.float64) / limit

    def thresholds(self) -> tuple[float, float]:
        return 1.0, 1.0
