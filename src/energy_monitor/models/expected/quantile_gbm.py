from typing import ClassVar, Final, Self

import lightgbm as lgb
import numpy as np
import pandas as pd

from energy_monitor.models.base import ExpectedConsumptionModel
from energy_monitor.models.expected.catalog import expected_registry

MEDIAN: Final = 0.5
LOWER_QUANTILE: Final = 0.05
UPPER_QUANTILE: Final = 0.95


@expected_registry.register
class QuantileLightGBMModel(ExpectedConsumptionModel):
    name: ClassVar[str] = "lightgbm_quantile"
    version: ClassVar[str] = "1.0.0"
    supports_interval: ClassVar[bool] = True

    def _fit_quantile(
        self, frame: pd.DataFrame, target: pd.Series, alpha: float
    ) -> lgb.LGBMRegressor:
        parameters = {
            key: value for key, value in self.hyperparameters.items() if key != "with_interval"
        }
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha, random_state=self.seed, verbose=-1, **parameters
        )
        return model.fit(frame[self.feature_columns], target)

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> Self:
        self._median = self._fit_quantile(frame, target, MEDIAN)
        self._lower = self._upper = None
        if self.hyperparameters.get("with_interval", False):
            self._lower = self._fit_quantile(frame, target, LOWER_QUANTILE)
            self._upper = self._fit_quantile(frame, target, UPPER_QUANTILE)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._median.predict(frame[self.feature_columns])

    def has_interval(self) -> bool:
        return self._lower is not None and self._upper is not None

    def predict_interval(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._lower is None or self._upper is None:
            raise ValueError("Modelo treinado sem with_interval=True")
        features = frame[self.feature_columns]
        return self._lower.predict(features), self._upper.predict(features)
