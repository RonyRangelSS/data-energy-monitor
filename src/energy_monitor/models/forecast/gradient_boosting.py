from typing import Any, ClassVar, Final, Self

import lightgbm as lgb
import numpy as np
import pandas as pd

from energy_monitor.models.base import ForecastModel
from energy_monitor.models.forecast.catalog import forecast_registry

LOWER_QUANTILE: Final = 0.05
UPPER_QUANTILE: Final = 0.95


@forecast_registry.register
class LightGBMForecaster(ForecastModel):
    name: ClassVar[str] = "lightgbm"
    version: ClassVar[str] = "1.0.0"
    supports_interval: ClassVar[bool] = True

    def _fit_booster(
        self, frame: pd.DataFrame, target: pd.Series, **objective: Any
    ) -> lgb.LGBMRegressor:
        parameters = {
            key: value for key, value in self.hyperparameters.items() if key != "with_interval"
        }
        model = lgb.LGBMRegressor(random_state=self.seed, verbose=-1, **parameters, **objective)
        return model.fit(frame[self.feature_columns], target)

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> Self:
        self._point = self._fit_booster(frame, target, objective="l1")
        self._lower = self._upper = None
        if self.hyperparameters.get("with_interval", False):
            self._lower = self._fit_booster(
                frame, target, objective="quantile", alpha=LOWER_QUANTILE
            )
            self._upper = self._fit_booster(
                frame, target, objective="quantile", alpha=UPPER_QUANTILE
            )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._point.predict(frame[self.feature_columns])

    def has_interval(self) -> bool:
        return self._lower is not None and self._upper is not None

    def predict_interval(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._lower is None or self._upper is None:
            raise ValueError("Modelo treinado sem with_interval=True")
        features = frame[self.feature_columns]
        return self._lower.predict(features), self._upper.predict(features)
