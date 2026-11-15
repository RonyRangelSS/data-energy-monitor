from typing import Final

from energy_monitor.models.detection.base import Detector
from energy_monitor.models.registry import ModelRegistry

detector_registry: Final[ModelRegistry[Detector]] = ModelRegistry("detector")
