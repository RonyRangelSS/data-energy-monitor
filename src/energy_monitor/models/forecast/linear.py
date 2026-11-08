from typing import ClassVar, Final, Self

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from energy_monitor.common.features import CALENDAR_FEATURES
from energy_monitor.models.base import ForecastModel
from energy_monitor.models.forecast.catalog import forecast_registry

CATEGORICAL_COLUMNS: Final = frozenset({"host_id", *CALENDAR_FEATURES})


@forecast_registry.register
class LinearForecaster(ForecastModel):
    name: ClassVar[str] = "linear_regression"
    version: ClassVar[str] = "1.0.0"

    def _build_pipeline(self) -> Pipeline:
        numeric = [c for c in self.feature_columns if c not in CATEGORICAL_COLUMNS]
        categorical = [c for c in self.feature_columns if c in CATEGORICAL_COLUMNS]
        encoder = OneHotEncoder(drop="first", sparse_output=True, dtype=np.float32)
        preprocessor = ColumnTransformer(
            [("numeric", StandardScaler(), numeric), ("categorical", encoder, categorical)],
            sparse_threshold=1.0,
        )
        return Pipeline([("preprocessor", preprocessor), ("regression", LinearRegression())])

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> Self:
        self._pipeline = self._build_pipeline().fit(frame[self.feature_columns], target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict(frame[self.feature_columns])
