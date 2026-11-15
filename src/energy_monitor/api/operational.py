import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from energy_monitor.models.base import ExpectedConsumptionModel, ForecastModel
from energy_monitor.models.detection.base import Detector
from energy_monitor.models.forecast import forecast_registry


@dataclass(frozen=True)
class ForecastSpec:
    horizon: str
    steps: int
    model: ForecastModel
    variant: str


@dataclass(frozen=True)
class OperationalModels:
    hosts: list[str]
    expected_model: ExpectedConsumptionModel
    detector: Detector
    forecasts: list[ForecastSpec]

    def describe(self) -> dict[str, Any]:
        return {
            "hosts": len(self.hosts),
            "expected_consumption": model_label(self.expected_model),
            "detector": {
                "name": self.detector.name,
                "version": self.detector.version,
                "parameters": self.detector.hyperparameters,
            },
            "forecasts": [
                {
                    "horizon": spec.horizon,
                    "steps": spec.steps,
                    "model": model_label(spec.model),
                    "variant": spec.variant,
                }
                for spec in self.forecasts
            ],
        }


def model_label(model: ExpectedConsumptionModel | ForecastModel) -> str:
    return f"{model.name}:{model.version}"


def read_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def load_forecast_specs(
    dataset: str, models_dir: Path, experiments_dir: Path
) -> list[ForecastSpec]:
    selection = read_metrics(experiments_dir / f"e5_forecasting_selection_{dataset}.json")
    configuration = json.loads(
        (experiments_dir / f"e5_forecasting_selection_{dataset}.json").read_text(encoding="utf-8")
    )["configuration"]
    specs = []
    for horizon, steps in configuration["horizons"].items():
        choice = selection["horizons"][horizon]["operational_choice"]
        version = forecast_registry.get(choice["model"]).version
        file_name = f"{dataset}_{choice['model']}_{horizon}_{choice['variant']}_{version}.joblib"
        model = ForecastModel.load(models_dir / file_name)
        specs.append(ForecastSpec(horizon, int(steps), model, choice["variant"]))
    return specs


def load_operational_models(
    dataset: str, models_dir: Path, experiments_dir: Path
) -> OperationalModels:
    quality = read_metrics(experiments_dir / f"e1_data_quality_{dataset}.json")
    expected = read_metrics(experiments_dir / f"e2_expected_consumption_selection_{dataset}.json")
    detection = read_metrics(experiments_dir / f"e3_detection_selection_{dataset}.json")
    expected_file = expected["models"][expected["operational_model"]]["model_file"]
    return OperationalModels(
        hosts=quality["coverage"]["eligible_hosts"],
        expected_model=ExpectedConsumptionModel.load(models_dir / expected_file),
        detector=Detector.load(models_dir / detection["detector_files"]["deviation"]),
        forecasts=load_forecast_specs(dataset, models_dir, experiments_dir),
    )
