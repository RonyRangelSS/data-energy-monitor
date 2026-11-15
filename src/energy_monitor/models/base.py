from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import joblib
import numpy as np
import pandas as pd


class RegisteredModel(ABC):
    """Base comum dos modelos do registro.

    Cada subclasse define `name` e `version` e recebe na construção as colunas
    de entrada, a semente e os hiperparâmetros.
    """

    name: ClassVar[str]
    version: ClassVar[str]

    def __init__(
        self, feature_columns: Sequence[str] = (), seed: int = 42, **hyperparameters: Any
    ) -> None:
        """Guarda as colunas de entrada, a semente e os hiperparâmetros."""
        self.feature_columns = list(feature_columns)
        self.seed = seed
        self.hyperparameters = hyperparameters

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


class RegressionModel(RegisteredModel):
    """Modelo que estima a potência a partir de um quadro de variáveis.

    `fit` e `predict` recebem o quadro completo e selecionam suas próprias
    colunas.
    """

    supports_interval: ClassVar[bool] = False

    @abstractmethod
    def fit(self, frame: pd.DataFrame, target: pd.Series) -> Self:
        """Treina o modelo e retorna a própria instância."""

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Retorna a estimativa pontual para cada linha de `frame`."""

    def has_interval(self) -> bool:
        """Indica se o modelo treinado pode produzir intervalo."""
        return self.supports_interval

    def predict_interval(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Retorna os limites 5% e 95%; modelos pontuais não implementam."""
        raise NotImplementedError(f"{self.name} não produz intervalo")


class ForecastModel(RegressionModel):
    """Interface comum dos modelos de previsão do registro.

    `requires_training` indica se o modelo precisa de treino e
    `uses_load_features` se ele tem variantes com e sem variáveis de carga.
    """

    requires_training: ClassVar[bool] = True
    uses_load_features: ClassVar[bool] = True


class ExpectedConsumptionModel(RegressionModel):
    """Interface comum dos modelos de consumo esperado do registro.

    Estimam a potência a partir da carga do mesmo instante, sem defasagens de
    potência.
    """
