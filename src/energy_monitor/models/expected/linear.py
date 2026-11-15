from typing import ClassVar, Final, Self

import numpy as np
import pandas as pd

from energy_monitor.common.expected_features import LOCAL_CALENDAR
from energy_monitor.models.base import ExpectedConsumptionModel
from energy_monitor.models.expected.catalog import expected_registry
from energy_monitor.models.linear_pipeline import build_linear_pipeline

CATEGORICAL_COLUMNS: Final = frozenset({"host_id", *LOCAL_CALENDAR})


@expected_registry.register
class LinearExpectedModel(ExpectedConsumptionModel):
    name: ClassVar[str] = "linear_regression"
    version: ClassVar[str] = "1.0.0"

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> Self:
        pipeline = build_linear_pipeline(self.feature_columns, CATEGORICAL_COLUMNS)
        self._pipeline = pipeline.fit(frame[self.feature_columns], target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict(frame[self.feature_columns])
