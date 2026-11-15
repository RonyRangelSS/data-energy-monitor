from collections.abc import Collection, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_linear_pipeline(
    feature_columns: Sequence[str], categorical_columns: Collection[str]
) -> Pipeline:
    numeric = [column for column in feature_columns if column not in categorical_columns]
    categorical = [column for column in feature_columns if column in categorical_columns]
    encoder = OneHotEncoder(drop="first", sparse_output=True, dtype=np.float32)
    preprocessor = ColumnTransformer(
        [("numeric", StandardScaler(), numeric), ("categorical", encoder, categorical)],
        sparse_threshold=1.0,
    )
    return Pipeline([("preprocessor", preprocessor), ("regression", LinearRegression())])
