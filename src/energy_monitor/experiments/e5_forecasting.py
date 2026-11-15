import argparse
import gc
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid

from energy_monitor.common.features import (
    GRID_COLUMNS,
    build_forecast_frame,
    compute_seasonal_scale,
    feature_columns,
)
from energy_monitor.common.schema import GRID_STEP
from energy_monitor.common.splitting import SplitBoundaries, compute_boundaries, local_to_utc
from energy_monitor.experiments.context import (
    boundaries_from_report,
    load_grid,
    read_quality_report,
)
from energy_monitor.experiments.evaluation import (
    SeasonalScale,
    build_predictions,
    compare_paired_errors,
    evaluate_forecast,
    holm_adjust,
    interval_coverage,
    percent_reduction,
)
from energy_monitor.experiments.protocol_freeze import (
    freeze_path_for,
    require_frozen_protocol,
    selection_paths_for,
)
from energy_monitor.experiments.run_record import build_run_record, write_json
from energy_monitor.models.base import ForecastModel
from energy_monitor.models.forecast import forecast_registry

EXPERIMENT_NAME: Final = "e5_forecasting"
DEFAULT_CONFIG_PATH: Final = Path("configs/e5_forecasting.yaml")
WITH_LOAD: Final = "with_load"
HISTORY_ONLY: Final = "history_only"
VARIANTS: Final = {WITH_LOAD: True, HISTORY_ONLY: False}
FILE_START: Final = "file_start"
LIGHTGBM: Final = "lightgbm"
LINEAR_REGRESSION: Final = "linear_regression"


@dataclass(frozen=True)
class DatasetContext:
    name: str
    grid: pd.DataFrame
    boundaries: SplitBoundaries
    hosts: list[str]
    scale: SeasonalScale


def report_progress(stage: str, dataset: str, horizon: object, model: str, variant: str) -> None:
    moment = datetime.now().strftime("%H:%M:%S")
    print(f"[{moment}] {stage} {dataset} {horizon} {model} {variant}", flush=True)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_context(name: str, config: dict[str, Any], stop_at_validation_end: bool) -> DatasetContext:
    report = read_quality_report(name, config)
    boundaries = boundaries_from_report(report)
    hosts = report["coverage"]["eligible_hosts"]
    before = boundaries.validation_end if stop_at_validation_end else None
    grid = load_grid(name, config, GRID_COLUMNS, before)
    per_host, total = compute_seasonal_scale(grid, hosts, boundaries.start, boundaries.train_end)
    scale = SeasonalScale(per_host=per_host, total=total)
    return DatasetContext(name, grid, boundaries, hosts, scale)


def training_windows(context: DatasetContext, config: dict[str, Any]) -> list[dict[str, Any]]:
    windows = []
    for local_start in config["training_start_candidates_local"]:
        if local_start is None:
            windows.append({"label": FILE_START, "start": context.grid["ts"].min()})
        else:
            start = local_to_utc(local_start, config["source_timezone"])
            windows.append({"label": local_start, "start": start})
    return windows


def training_rows(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame["ts"] >= start) & (frame["target_ts"] < end)].dropna(subset=["target"])


def evaluation_rows(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    selected = (frame["target_ts"] >= start) & (frame["target_ts"] < end)
    return frame[selected].dropna(subset=["target"])


def fit_model(
    name: str,
    train: pd.DataFrame,
    columns: Sequence[str],
    seed: int,
    hyperparameters: dict[str, Any],
) -> ForecastModel:
    model = forecast_registry.create(name, columns, seed, **hyperparameters)
    return model.fit(train, train["target"].astype(float))


def variant_names(model_name: str) -> list[str]:
    uses_load = forecast_registry.get(model_name).uses_load_features
    return [WITH_LOAD, HISTORY_ONLY] if uses_load else [HISTORY_ONLY]


def default_hyperparameters(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("default_hyperparameters", {}).get(model_name, {}))


def score(
    model: ForecastModel, rows: pd.DataFrame, context: DatasetContext, details: bool = False
) -> dict[str, Any]:
    predictions = build_predictions(rows, model.predict(rows))
    return evaluate_forecast(predictions, context.scale, len(context.hosts), details)


def search_windows(
    model_name: str,
    columns: Sequence[str],
    frame: pd.DataFrame,
    context: DatasetContext,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    validation = evaluation_rows(
        frame, context.boundaries.train_end, context.boundaries.validation_end
    )
    results = []
    for window in training_windows(context, config):
        train = training_rows(frame, window["start"], context.boundaries.train_end)
        model = fit_model(
            model_name, train, columns, config["seed"], default_hyperparameters(model_name, config)
        )
        results.append(
            {
                "label": window["label"],
                "start_utc": window["start"].isoformat(),
                "train_rows": int(len(train)),
                "validation_mae": score(model, validation, context)["mae"],
            }
        )
        del model
        gc.collect()
    return results


def search_hyperparameters(
    model_name: str,
    columns: Sequence[str],
    window_start: pd.Timestamp,
    frame: pd.DataFrame,
    context: DatasetContext,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    grid = config.get("hyperparameter_grid", {}).get(model_name)
    if not grid:
        return []
    train = training_rows(frame, window_start, context.boundaries.train_end)
    validation = evaluation_rows(
        frame, context.boundaries.train_end, context.boundaries.validation_end
    )
    base = default_hyperparameters(model_name, config)
    results = []
    for candidate in ParameterGrid(grid):
        hyperparameters = {**base, **candidate}
        model = fit_model(model_name, train, columns, config["seed"], hyperparameters)
        results.append(
            {
                "hyperparameters": hyperparameters,
                "validation_mae": score(model, validation, context)["mae"],
            }
        )
        del model
        gc.collect()
    return results


def select_variant(
    model_name: str,
    with_load: bool,
    frame: pd.DataFrame,
    horizon_steps: int,
    context: DatasetContext,
    config: dict[str, Any],
) -> dict[str, Any]:
    variant = WITH_LOAD if with_load else HISTORY_ONLY
    report_progress("selection", context.name, horizon_steps, model_name, variant)
    columns = feature_columns(horizon_steps, with_load)
    model_class = forecast_registry.get(model_name)
    boundaries = context.boundaries
    validation = evaluation_rows(frame, boundaries.train_end, boundaries.validation_end)
    hyperparameters = default_hyperparameters(model_name, config)
    if not model_class.requires_training:
        model = forecast_registry.create(model_name, columns, config["seed"], **hyperparameters)
        return {"validation": score(model, validation, context), "columns": columns}

    window_search = search_windows(model_name, columns, frame, context, config)
    chosen_window = min(window_search, key=lambda item: item["validation_mae"])
    window_start = pd.Timestamp(chosen_window["start_utc"])
    hyperparameter_search = search_hyperparameters(
        model_name, columns, window_start, frame, context, config
    )
    if hyperparameter_search:
        best = min(hyperparameter_search, key=lambda item: item["validation_mae"])
        hyperparameters = best["hyperparameters"]
    train = training_rows(frame, window_start, boundaries.train_end)
    fit_parameters = dict(hyperparameters)
    if model_class.supports_interval:
        fit_parameters["with_interval"] = True
    model = fit_model(model_name, train, columns, config["seed"], fit_parameters)
    result = {
        "columns": columns,
        "window_search": window_search,
        "chosen_window": chosen_window,
        "hyperparameter_search": hyperparameter_search,
        "chosen_hyperparameters": hyperparameters,
        "validation": score(model, validation, context),
    }
    if model_class.supports_interval:
        lower, upper = model.predict_interval(validation)
        result["interval_coverage_5_95"] = interval_coverage(
            validation["target"].astype(float).to_numpy(), lower, upper
        )
    return result


def motivation_full_period(
    frame: pd.DataFrame,
    horizon_steps: int,
    context: DatasetContext,
    config: dict[str, Any],
) -> dict[str, Any]:
    full_grid_start = context.grid["ts"].min()
    full = compute_boundaries(
        full_grid_start,
        context.boundaries.end,
        None,
        config["train_fraction"],
        config["validation_fraction"],
    )
    train = training_rows(frame, full_grid_start, full.train_end)
    boundaries = context.boundaries
    validation = evaluation_rows(frame, boundaries.train_end, boundaries.validation_end)
    results: dict[str, Any] = {"train_end_utc": full.train_end.isoformat()}
    for variant, with_load in VARIANTS.items():
        columns = feature_columns(horizon_steps, with_load)
        model = fit_model(
            LIGHTGBM, train, columns, config["seed"], default_hyperparameters(LIGHTGBM, config)
        )
        results[variant] = score(model, validation, context)
        del model
        gc.collect()
    return results


def choose_operational_model(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        (result["validation"]["mae"], model, variant)
        for model, variants in models.items()
        for variant, result in variants.items()
    ]
    mae, model, variant = min(candidates)
    return {"model": model, "variant": variant, "validation_mae": mae}


def load_effect(models: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        model: percent_reduction(
            variants[HISTORY_ONLY]["validation"]["mae"], variants[WITH_LOAD]["validation"]["mae"]
        )
        for model, variants in models.items()
        if {WITH_LOAD, HISTORY_ONLY} <= set(variants)
    }


def select_horizon(
    horizon_steps: int, context: DatasetContext, config: dict[str, Any]
) -> dict[str, Any]:
    frame = build_forecast_frame(
        context.grid, context.hosts, horizon_steps, config["source_timezone"]
    )
    models: dict[str, dict[str, Any]] = {}
    for model_name in forecast_registry.names():
        models[model_name] = {
            variant: select_variant(
                model_name, VARIANTS[variant], frame, horizon_steps, context, config
            )
            for variant in variant_names(model_name)
        }
    result = {
        "models": models,
        "operational_choice": choose_operational_model(models),
        "load_effect_percent_validation": load_effect(models),
    }
    if config.get("motivation_full_period", False) and LIGHTGBM in models:
        result["motivation_full_period"] = motivation_full_period(
            frame, horizon_steps, context, config
        )
    return result


def run_selection(name: str, config: dict[str, Any]) -> Path:
    context = load_context(name, config, stop_at_validation_end=True)
    horizons = {
        label: select_horizon(steps, context, config) for label, steps in config["horizons"].items()
    }
    data_path = Path(config["datasets"][name])
    metrics = {
        "stage": "selection",
        "dataset": name,
        "eligible_hosts": len(context.hosts),
        "horizons": horizons,
    }
    record = build_run_record(EXPERIMENT_NAME, config, data_path, config["seed"], metrics)
    path = Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_selection_{name}.json"
    write_json(record, path)
    return path


def read_selection(name: str, config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_selection_{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def save_trained_model(
    model: ForecastModel,
    context: DatasetContext,
    label: str,
    variant: str,
    training_start: pd.Timestamp,
    config: dict[str, Any],
) -> Path:
    name = f"{context.name}_{model.name}_{label}_{variant}_{model.version}"
    directory = Path(config["models_dir"])
    path = directory / f"{name}.joblib"
    model.save(path)
    metadata = {
        "model": model.name,
        "version": model.version,
        "dataset": context.name,
        "horizon": label,
        "variant": variant,
        "feature_columns": model.feature_columns,
        "hyperparameters": model.hyperparameters,
        "seed": model.seed,
        "training_start_utc": training_start.isoformat(),
        "training_end_utc": context.boundaries.validation_end.isoformat(),
    }
    write_json(metadata, directory / f"{name}.json")
    return path


def evaluate_final_variant(
    model_name: str,
    variant: str,
    selected: dict[str, Any],
    label: str,
    frame: pd.DataFrame,
    context: DatasetContext,
    config: dict[str, Any],
) -> dict[str, Any]:
    report_progress("final", context.name, label, model_name, variant)
    model_class = forecast_registry.get(model_name)
    boundaries = context.boundaries
    test = evaluation_rows(frame, boundaries.validation_end, boundaries.end + GRID_STEP)
    start = boundaries.start
    hyperparameters = default_hyperparameters(model_name, config)
    if model_class.requires_training:
        start = pd.Timestamp(selected["chosen_window"]["start_utc"])
        hyperparameters = dict(selected["chosen_hyperparameters"])
    if model_class.supports_interval:
        hyperparameters["with_interval"] = True
    train = training_rows(frame, start, boundaries.validation_end)
    model = fit_model(model_name, train, selected["columns"], config["seed"], hyperparameters)
    save_trained_model(model, context, label, variant, start, config)
    result = score(model, test, context, details=True)
    if model_class.supports_interval:
        lower, upper = model.predict_interval(test)
        result["interval_coverage_5_95"] = interval_coverage(
            test["target"].astype(float).to_numpy(), lower, upper
        )
    return result


def host_errors(result: dict[str, Any]) -> pd.Series:
    return pd.Series({host: row["mae"] for host, row in result["per_host"].items()})


def compare_final(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for model, variants in models.items():
        if {WITH_LOAD, HISTORY_ONLY} <= set(variants):
            comparisons[f"{model}_with_load_vs_history_only"] = {
                **compare_paired_errors(
                    host_errors(variants[WITH_LOAD]), host_errors(variants[HISTORY_ONLY])
                ),
                "mae_reduction_percent": percent_reduction(
                    variants[HISTORY_ONLY]["mae"], variants[WITH_LOAD]["mae"]
                ),
            }
    if {LIGHTGBM, LINEAR_REGRESSION} <= set(models):
        for variant in VARIANTS:
            comparisons[f"lightgbm_vs_linear_{variant}"] = compare_paired_errors(
                host_errors(models[LIGHTGBM][variant]),
                host_errors(models[LINEAR_REGRESSION][variant]),
            )
    return comparisons


def add_holm_correction(horizons: dict[str, Any]) -> None:
    family = {
        (label, comparison): result["p_value"]
        for label, horizon in horizons.items()
        for comparison, result in horizon["comparisons"].items()
    }
    adjusted = holm_adjust(family)
    for (label, comparison), value in adjusted.items():
        horizons[label]["comparisons"][comparison]["p_value_holm"] = value


def run_final(name: str, config: dict[str, Any]) -> Path:
    require_frozen_protocol(selection_paths_for(config), freeze_path_for(config))
    context = load_context(name, config, stop_at_validation_end=False)
    selection = read_selection(name, config)
    horizons: dict[str, Any] = {}
    for label, steps in config["horizons"].items():
        frame = build_forecast_frame(context.grid, context.hosts, steps, config["source_timezone"])
        models = {
            model: {
                variant: evaluate_final_variant(
                    model, variant, selected, label, frame, context, config
                )
                for variant, selected in selection["horizons"][label]["models"][model].items()
            }
            for model in selection["horizons"][label]["models"]
        }
        horizons[label] = {"models": models, "comparisons": compare_final(models)}
        del frame
        gc.collect()
    add_holm_correction(horizons)
    metrics = {"stage": "test", "dataset": name, "horizons": horizons}
    data_path = Path(config["datasets"][name])
    record = build_run_record(EXPERIMENT_NAME, config, data_path, config["seed"], metrics)
    path = Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_test_{name}.json"
    write_json(record, path)
    return path


def parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E5: previsão de consumo")
    parser.add_argument("--stage", choices=["selection", "final"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", action="append")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = parse_arguments(arguments)
    config = load_config(parsed.config)
    runner = run_selection if parsed.stage == "selection" else run_final
    for name in parsed.dataset or list(config["datasets"]):
        print(f"Relatório gravado em {runner(name, config)}")


if __name__ == "__main__":
    main()
