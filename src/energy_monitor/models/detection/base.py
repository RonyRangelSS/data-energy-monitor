import copy
from abc import abstractmethod
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

from energy_monitor.models.base import RegisteredModel
from energy_monitor.models.detection.hysteresis import apply_hysteresis

DEFAULT_CONSECUTIVE = 2


class Detector(RegisteredModel):
    """Interface comum dos detectores do registro.

    `score` atribui uma intensidade a cada leitura e `detect` aplica os
    limiares de abertura e fechamento com histerese por servidor. O quadro
    deve estar ordenado por servidor e instante.
    """

    scores_depend_on_parameters: ClassVar[bool] = False

    @abstractmethod
    def fit(self, frame: pd.DataFrame) -> Self:
        """Ajusta o detector ao período de treino."""

    @abstractmethod
    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Retorna a intensidade de anomalia de cada leitura."""

    @abstractmethod
    def thresholds(self) -> tuple[float, float]:
        """Retorna os limiares de abertura e de fechamento."""

    def with_parameters(self, **parameters: Any) -> Self:
        """Retorna uma cópia com hiperparâmetros alterados, sem novo ajuste."""
        variant = copy.copy(self)
        variant.hyperparameters = {**self.hyperparameters, **parameters}
        return variant

    def detect_from_scores(self, frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
        """Aplica a histerese a escores já calculados."""
        opening_threshold, closing_threshold = self.thresholds()
        consecutive = int(self.hyperparameters.get("consecutive", DEFAULT_CONSECUTIVE))
        valid = ~np.isnan(scores)
        opening = valid & (scores >= opening_threshold)
        closing = ~valid | (scores < closing_threshold)
        groups = frame["host_id"].astype(str).to_numpy()
        return apply_hysteresis(groups, opening, closing, consecutive)

    def detect(self, frame: pd.DataFrame) -> np.ndarray:
        """Retorna, para cada leitura, se há alerta ativo."""
        return self.detect_from_scores(frame, self.score(frame))
