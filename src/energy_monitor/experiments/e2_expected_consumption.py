import argparse
import gc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid

from energy_monitor.common.expected_features import build_expected_frame, expected_feature_columns
from energy_monitor.common.features import GRID_COLUMNS
from energy_monitor.common.schema import GRID_STEP
from energy_monitor.common.splitting import SplitBoundaries, local_to_utc
from energy_monitor.experiments.context import (
    boundaries_from_report,
    load_grid,
    read_json_metrics,
    read_quality_report,
)
from energy_monitor.experiments.evaluation import (
    compare_paired_errors,
    holm_adjust,
    mean_absolute_error,
    root_mean_squared_error,
)
from energy_monitor.experiments.protocol_freeze import (
    freeze_path_for,
    require_frozen_protocol,
    selection_paths_for,
)
from energy_monitor.experiments.run_record import build_run_record, write_json
from energy_monitor.models.base import ExpectedConsumptionModel
from energy_monitor.models.expected import expected_registry
from energy_monitor.models.expected.residuals import (
    ResidualStatistics,
    compute_residual_statistics,
    out_of_fold_predictions,
    standardized_deviation,
)

EXPERIMENT_NAME: Final = "e2_expected_consumption"
DEFAULT_CONFIG_PATH: Final = Path("configs/e2_e3_detection.yaml")
FILE_START: Final = "file_start"
ALERT_LEVEL: Final = 3.5


@dataclass(frozen=True)
class ExpectedContext:
    name: str
    frame: pd.DataFrame
    boundaries: SplitBoundaries
    hosts: list[str]
    statistics_start: pd.Timestamp


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_expected_context(
    name: str, config: dict[str, Any], stop_at_validation_end: bool
) -> ExpectedContext:
    report = read_quality_report(name, config)
    boundaries = boundaries_from_report(report)
    hosts = report["coverage"]["eligible_hosts"]
    before = boundaries.validation_end if stop_at_validation_end else None
    grid = load_grid(name, config, GRID_COLUMNS, before)
    frame = build_expected_frame(grid, hosts, config["source_timezone"])
    statistics_start = local_to_utc(config["statistics_start_local"], config["source_timezone"])
    return ExpectedContext(name, frame, boundaries, hosts, statistics_start)


def rows_between(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame["ts"] >= start) & (frame["ts"] < end)]


def candidate_windows(context: ExpectedContext, config: dict[str, Any]) -> list[dict[str, Any]]:
    windows = []
    for local_start in config["training_start_candidates_local"]:
        if local_start is None:
            windows.append({"label": FILE_START, "start": context.frame["ts"].min()})
        else:
            start = local_to_utc(local_start, config["source_timezone"])
            windows.append({"label": local_start, "start": start})
    return windows


def fit_expected_model(
    name: str, train: pd.DataFrame, seed: int, hyperparameters: dict[str, Any]
) -> ExpectedConsumptionModel:
    model = expected_registry.create(name, expected_feature_columns(), seed, **hyperparameters)
    return model.fit(train, train["power"])


def evaluate_expected(
    model: ExpectedConsumptionModel, rows: pd.DataFrame, details: bool = False
) -> dict[str, Any]:
    actual = rows["power"].to_numpy(np.float64)
    predicted = model.predict(rows)
    result: dict[str, Any] = {
        "rows": int(len(rows)),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": root_mean_squared_error(actual, predicted),
    }
    if model.has_interval():
        lower, upper = model.predict_interval(rows)
        result["coverage_5_95"] = float(np.mean((actual >= lower) & (actual <= upper)))
        result["coverage_upper_95"] = float(np.mean(actual <= upper))
    if details:
        errors = pd.Series(np.abs(actual - predicted)).groupby(
            rows["host_id"].astype(str).to_numpy()
        )
        result["per_host_mae"] = errors.mean().round(6).to_dict()
    return result


def default_hyperparameters(name: str, config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("default_hyperparameters", {}).get(name, {}))


def search_windows(
    name: str, context: ExpectedContext, config: dict[str, Any], validation: pd.DataFrame
) -> list[dict[str, Any]]:
    results = []
    for window in candidate_windows(context, config):
        train = rows_between(context.frame, window["start"], context.boundaries.train_end)
        model = fit_expected_model(
            name, train, config["seed"], default_hyperparameters(name, config)
        )
        results.append(
            {
                "label": window["label"],
                "start_utc": window["start"].isoformat(),
                "train_rows": int(len(train)),
                "validation_mae": evaluate_expected(model, validation)["mae"],
            }
        )
        del model
        gc.collect()
    return results


def search_hyperparameters(
    name: str, train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
    grid = config.get("hyperparameter_grid", {}).get(name)
    if not grid:
        return []
    results = []
    for candidate in ParameterGrid(grid):
        hyperparameters = {**default_hyperparameters(name, config), **candidate}
        model = fit_expected_model(name, train, config["seed"], hyperparameters)
        results.append(
            {
                "hyperparameters": hyperparameters,
                "validation_mae": evaluate_expected(model, validation)["mae"],
            }
        )
        del model
        gc.collect()
    return results


def select_model(
    name: str, context: ExpectedContext, config: dict[str, Any]
) -> tuple[dict[str, Any], ExpectedConsumptionModel]:
    print(f"selection {context.name} expected {name}", flush=True)
    boundaries = context.boundaries
    validation = rows_between(context.frame, boundaries.train_end, boundaries.validation_end)
    window_search = search_windows(name, context, config, validation)
    chosen_window = min(window_search, key=lambda item: item["validation_mae"])
    train = rows_between(
        context.frame, pd.Timestamp(chosen_window["start_utc"]), boundaries.train_end
    )
    hyperparameter_search = search_hyperparameters(name, train, validation, config)
    hyperparameters = default_hyperparameters(name, config)
    if hyperparameter_search:
        hyperparameters = min(hyperparameter_search, key=lambda item: item["validation_mae"])[
            "hyperparameters"
        ]
    final_parameters = {**hyperparameters, "with_interval": True}
    model = fit_expected_model(name, train, config["seed"], final_parameters)
    result = {
        "window_search": window_search,
        "chosen_window": chosen_window,
        "hyperparameter_search": hyperparameter_search,
        "chosen_hyperparameters": hyperparameters,
        "validation": evaluate_expected(model, validation),
    }
    return result, model


def compute_out_of_fold_statistics(
    name: str,
    hyperparameters: dict[str, Any],
    window_start: pd.Timestamp,
    context: ExpectedContext,
    config: dict[str, Any],
) -> ResidualStatistics:
    fit_frame = rows_between(context.frame, window_start, context.boundaries.train_end)
    fit_frame = fit_frame.reset_index(drop=True)
    mask = (fit_frame["ts"] >= context.statistics_start).to_numpy()
    point_parameters = {**hyperparameters, "with_interval": False}

    def factory() -> ExpectedConsumptionModel:
        return expected_registry.create(
            name, expected_feature_columns(), config["seed"], **point_parameters
        )

    predictions = out_of_fold_predictions(factory, fit_frame, mask, config["out_of_fold_blocks"])
    rows = fit_frame.loc[mask]
    return compute_residual_statistics(rows["host_id"], rows["power"].to_numpy() - predictions)


def compute_in_sample_statistics(
    model: ExpectedConsumptionModel, context: ExpectedContext
) -> ResidualStatistics:
    rows = rows_between(context.frame, context.statistics_start, context.boundaries.train_end)
    return compute_residual_statistics(
        rows["host_id"], rows["power"].to_numpy() - model.predict(rows)
    )


def alert_level_rates(
    model: ExpectedConsumptionModel, rows: pd.DataFrame, statistics: ResidualStatistics
) -> dict[str, float]:
    z = standardized_deviation(rows, model.predict(rows), statistics)
    return {
        "share_z_at_least_3_5": float(np.nanmean(z >= ALERT_LEVEL)),
        "share_abs_z_at_least_3_5": float(np.nanmean(np.abs(z) >= ALERT_LEVEL)),
        "median_host_mad": float(statistics.mad.median()),
    }


def model_path(context_name: str, model: ExpectedConsumptionModel, config: dict[str, Any]) -> Path:
    return (
        Path(config["models_dir"]) / f"{context_name}_expected_{model.name}_{model.version}.joblib"
    )


def statistics_path(context_name: str, config: dict[str, Any]) -> Path:
    return Path(config["models_dir"]) / f"{context_name}_residual_statistics.json"


def save_model(
    model: ExpectedConsumptionModel, path: Path, window_start: str, end: pd.Timestamp
) -> None:
    model.save(path)
    metadata = {
        "model": model.name,
        "version": model.version,
        "feature_columns": model.feature_columns,
        "hyperparameters": model.hyperparameters,
        "seed": model.seed,
        "training_start_utc": window_start,
        "training_end_utc": end.isoformat(),
    }
    write_json(metadata, path.with_suffix(".json"))


def run_selection(name: str, config: dict[str, Any]) -> Path:
    context = load_expected_context(name, config, stop_at_validation_end=True)
    models: dict[str, Any] = {}
    operational_name = config["operational_expected_model"]
    for model_name in expected_registry.names():
        result, model = select_model(model_name, context, config)
        path = model_path(name, model, config)
        save_model(model, path, result["chosen_window"]["start_utc"], context.boundaries.train_end)
        result["model_file"] = path.name
        models[model_name] = result
        if model_name == operational_name:
            operational_model = model
        del model
        gc.collect()
    chosen = models[operational_name]
    print(f"selection {name} residual statistics", flush=True)
    out_of_fold = compute_out_of_fold_statistics(
        operational_name,
        chosen["chosen_hyperparameters"],
        pd.Timestamp(chosen["chosen_window"]["start_utc"]),
        context,
        config,
    )
    in_sample = compute_in_sample_statistics(operational_model, context)
    validation = rows_between(
        context.frame, context.boundaries.train_end, context.boundaries.validation_end
    )
    write_json({"statistics": out_of_fold.to_dict()}, statistics_path(name, config))
    metrics = {
        "stage": "selection",
        "dataset": name,
        "eligible_hosts": len(context.hosts),
        "models": models,
        "operational_model": operational_name,
        "residual_statistics_file": statistics_path(name, config).name,
        "residual_statistics_validation": {
            "out_of_fold": alert_level_rates(operational_model, validation, out_of_fold),
            "in_sample": alert_level_rates(operational_model, validation, in_sample),
        },
    }
    record = build_run_record(
        EXPERIMENT_NAME, config, Path(config["datasets"][name]), config["seed"], metrics
    )
    path = Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_selection_{name}.json"
    write_json(record, path)
    return path


def load_selected_model(file_name: str, config: dict[str, Any]) -> ExpectedConsumptionModel:
    return ExpectedConsumptionModel.load(Path(config["models_dir"]) / file_name)


def compare_with_operational(models: dict[str, Any], operational: str) -> dict[str, Any]:
    reference = pd.Series(models[operational]["per_host_mae"])
    comparisons = {
        f"{operational}_vs_{other}": compare_paired_errors(
            reference, pd.Series(models[other]["per_host_mae"])
        )
        for other in models
        if other != operational
    }
    adjusted = holm_adjust({key: value["p_value"] for key, value in comparisons.items()})
    for key, value in adjusted.items():
        comparisons[key]["p_value_holm"] = value
    return comparisons


def run_final(name: str, config: dict[str, Any]) -> Path:
    require_frozen_protocol(selection_paths_for(config), freeze_path_for(config))
    selection = read_json_metrics(
        Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_selection_{name}.json"
    )
    context = load_expected_context(name, config, stop_at_validation_end=False)
    test = rows_between(
        context.frame, context.boundaries.validation_end, context.boundaries.end + GRID_STEP
    )
    models = {
        model_name: evaluate_expected(
            load_selected_model(result["model_file"], config), test, details=True
        )
        for model_name, result in selection["models"].items()
    }
    metrics = {
        "stage": "test",
        "dataset": name,
        "models": models,
        "comparisons": compare_with_operational(models, selection["operational_model"]),
    }
    record = build_run_record(
        EXPERIMENT_NAME, config, Path(config["datasets"][name]), config["seed"], metrics
    )
    path = Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_test_{name}.json"
    write_json(record, path)
    return path


def parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2: consumo esperado")
    parser.add_argument("--stage", choices=["selection", "final"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = parse_arguments(arguments)
    config = load_config(parsed.config)
    runner = run_selection if parsed.stage == "selection" else run_final
    for name in config["datasets"]:
        print(f"Relatório gravado em {runner(name, config)}")


if __name__ == "__main__":
    main()
