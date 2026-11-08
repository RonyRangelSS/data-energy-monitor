from typing import Final

from energy_monitor.models.base import ForecastModel
from energy_monitor.models.registry import ModelRegistry

forecast_registry: Final[ModelRegistry[ForecastModel]] = ModelRegistry("forecast")
