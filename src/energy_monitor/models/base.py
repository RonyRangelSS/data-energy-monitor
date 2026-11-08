from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import joblib
import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """Interface comum dos modelos de previsão do registro.

    Cada subclasse define `name` e `version` e recebe na construção as colunas
    de variáveis que deve usar. `fit` e `predict` recebem o quadro completo de
    previsão e selecionam suas próprias colunas.
    """

    name: ClassVar[str]
    version: ClassVar[str]
    requires_training: ClassVar[bool] = True
    uses_load_features: ClassVar[bool] = True
    supports_interval: ClassVar[bool] = False

    def __init__(
        self, feature_columns: Sequence[str], seed: int = 42, **hyperparameters: Any
    ) -> None:
        """Guarda as colunas de entrada, a semente e os hiperparâmetros."""
        self.feature_columns = list(feature_columns)
        self.seed = seed
        self.hyperparameters = hyperparameters

    @abstractmethod
    def fit(self, frame: pd.DataFrame, target: pd.Series) -> Self:
        """Treina o modelo e retorna a própria instância."""

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Retorna a previsão pontual para cada linha de `frame`."""

    def predict_interval(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Retorna os limites 5% e 95%; modelos pontuais não implementam."""
        raise NotImplementedError(f"{self.name} não produz intervalo de previsão")

    def save(self, path: Path) -> None:
        """Grava o modelo em `path`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Carrega um modelo gravado com `save`."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} não contém um {cls.__name__}")
        return model
