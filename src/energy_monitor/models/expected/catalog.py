from typing import Final

from energy_monitor.models.base import ExpectedConsumptionModel
from energy_monitor.models.registry import ModelRegistry

expected_registry: Final[ModelRegistry[ExpectedConsumptionModel]] = ModelRegistry(
    "expected_consumption"
)
