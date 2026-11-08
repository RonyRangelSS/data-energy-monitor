from typing import ClassVar, Self

import numpy as np
import pandas as pd

from energy_monitor.common.features import SEASONAL_REFERENCE
from energy_monitor.models.base import ForecastModel
from energy_monitor.models.forecast.catalog import forecast_registry


@forecast_registry.register
class SeasonalNaiveForecaster(ForecastModel):
    name: ClassVar[str] = "seasonal_naive"
    version: ClassVar[str] = "1.0.0"
    requires_training: ClassVar[bool] = False
    uses_load_features: ClassVar[bool] = False

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> Self:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[SEASONAL_REFERENCE].to_numpy(dtype=np.float64)
