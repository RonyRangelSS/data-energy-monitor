from typing import ClassVar, Self

import numpy as np
import pandas as pd

from energy_monitor.models.detection.base import Detector
from energy_monitor.models.detection.catalog import detector_registry
from energy_monitor.models.expected.residuals import (
    ResidualStatistics,
    compute_residual_statistics,
    standardized_deviation,
)


@detector_registry.register
class DeviationDetector(Detector):
    name: ClassVar[str] = "deviation"
    version: ClassVar[str] = "1.0.0"

    def fit(self, frame: pd.DataFrame) -> Self:
        residuals = frame["residual"].to_numpy(dtype=np.float64)
        return self.fit_statistics(compute_residual_statistics(frame["host_id"], residuals))

    def fit_statistics(self, statistics: ResidualStatistics) -> Self:
        self.statistics = statistics
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        expected = frame["expected"].to_numpy(dtype=np.float64)
        return standardized_deviation(frame, expected, self.statistics)

    def thresholds(self) -> tuple[float, float]:
        return (
            float(self.hyperparameters.get("open_threshold", 3.5)),
            float(self.hyperparameters.get("close_threshold", 2.5)),
        )
