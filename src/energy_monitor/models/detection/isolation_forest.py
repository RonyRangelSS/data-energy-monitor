from typing import ClassVar, Final, Self

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from energy_monitor.common.features import LOAD_FEATURES
from energy_monitor.models.detection.base import Detector
from energy_monitor.models.detection.catalog import detector_registry

FEATURES: Final = ("power_ratio", *LOAD_FEATURES)


@detector_registry.register
class IsolationForestDetector(Detector):
    name: ClassVar[str] = "isolation_forest"
    version: ClassVar[str] = "1.0.0"

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        reference = frame["host_id"].astype(str).map(self._median_power).to_numpy(np.float64)
        features = frame[list(LOAD_FEATURES)].astype(np.float64)
        return features.assign(power_ratio=frame["power"].to_numpy(np.float64) / reference)[
            list(FEATURES)
        ]

    def fit(self, frame: pd.DataFrame) -> Self:
        self._median_power = (
            frame["power"].astype(np.float64).groupby(frame["host_id"].astype(str)).median()
        )
        features = self._features(frame)
        self._forest = IsolationForest(
            n_estimators=int(self.hyperparameters.get("n_estimators", 100)),
            random_state=self.seed,
        ).fit(features)
        self._training_scores = -self._forest.score_samples(features)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return -self._forest.score_samples(self._features(frame))

    def thresholds(self) -> tuple[float, float]:
        percentile = float(self.hyperparameters.get("percentile", 99.0))
        limit = float(np.percentile(self._training_scores, percentile))
        return limit, limit
