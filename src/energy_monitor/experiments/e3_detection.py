import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from energy_monitor.common.schema import GRID_STEP
from energy_monitor.experiments.context import read_json_metrics
from energy_monitor.experiments.detection_metrics import (
    episodes_per_host_day,
    evaluate_detection,
    evaluate_negative_control,
    event_detection,
)
from energy_monitor.experiments.e2_expected_consumption import (
    EXPERIMENT_NAME as E2_EXPERIMENT,
)
from energy_monitor.experiments.e2_expected_consumption import (
    ExpectedContext,
    load_config,
    load_expected_context,
    rows_between,
    statistics_path,
)
from energy_monitor.experiments.evaluation import compare_paired_errors, holm_adjust
from energy_monitor.experiments.protocol_freeze import (
    freeze_path_for,
    require_frozen_protocol,
    selection_paths_for,
)
from energy_monitor.experiments.run_record import build_run_record, write_json
from energy_monitor.models.base import ExpectedConsumptionModel
from energy_monitor.models.detection import detector_registry
from energy_monitor.models.detection.alerts import extract_episodes
from energy_monitor.models.detection.base import Detector
from energy_monitor.models.detection.deviation import DeviationDetector
from energy_monitor.models.expected.residuals import ResidualStatistics
from energy_monitor.replayer.injection import (
    NEGATIVE_KINDS,
    HostReference,
    InjectedEvent,
    InjectionPlan,
    apply_events,
    compute_host_reference,
    plan_events,
)

EXPERIMENT_NAME: Final = "e3_detection"
DEFAULT_CONFIG_PATH: Final = Path("configs/e2_e3_detection.yaml")
REFERENCE_DETECTOR: Final = "deviation"


def read_e2_selection(name: str, config: dict[str, Any]) -> dict[str, Any]:
    return read_json_metrics(Path(config["output_dir"]) / f"{E2_EXPERIMENT}_selection_{name}.json")


def load_operational_model(
    selection: dict[str, Any], config: dict[str, Any]
) -> ExpectedConsumptionModel:
    file_name = selection["models"][selection["operational_model"]]["model_file"]
    return ExpectedConsumptionModel.load(Path(config["models_dir"]) / file_name)


def load_statistics(name: str, config: dict[str, Any]) -> ResidualStatistics:
    values = json.loads(statistics_path(name, config).read_text(encoding="utf-8"))
    return ResidualStatistics.from_dict(values["statistics"])


def with_expected(rows: pd.DataFrame, model: ExpectedConsumptionModel) -> pd.DataFrame:
    _, upper = model.predict_interval(rows)
    return rows.assign(expected=model.predict(rows), upper=upper)


def training_rows(context: ExpectedContext) -> pd.DataFrame:
    return rows_between(context.frame, context.statistics_start, context.boundaries.train_end)


def fit_detectors(
    train: pd.DataFrame, statistics: ResidualStatistics, config: dict[str, Any]
) -> dict[str, Detector]:
    detectors = {}
    consecutive = config["detection"]["consecutive"]
    for name in detector_registry.names():
        detector = detector_registry.create(name, (), config["seed"], consecutive=consecutive)
        if isinstance(detector, DeviationDetector):
            detectors[name] = detector.fit_statistics(statistics)
        else:
            detectors[name] = detector.fit(train)
    return detectors


def scenario_events(
    hosts: Sequence[str],
    kinds: Sequence[str],
    seed: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: dict[str, Any],
) -> list[InjectedEvent]:
    detection = config["detection"]
    plan = InjectionPlan(
        kinds=tuple(kinds),
        magnitudes=tuple(detection["magnitudes"]),
        durations_slots=tuple(detection["durations_slots"]),
        count=int(detection["events_per_seed"]),
        seed=int(seed),
        start=start,
        end=end,
    )
    return plan_events(hosts, plan)


def injected_rows(
    rows: pd.DataFrame,
    events: Sequence[InjectedEvent],
    reference: HostReference,
    model: ExpectedConsumptionModel,
) -> pd.DataFrame:
    return with_expected(apply_events(rows, events, reference), model)


def parameter_grid(name: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    return config["detection"]["grids"].get(name, [{}])


def detect_variants(
    detector: Detector, rows: pd.DataFrame, grid: Sequence[dict[str, Any]]
) -> list[np.ndarray]:
    shared_scores = None if detector.scores_depend_on_parameters else detector.score(rows)
    results = []
    for parameters in grid:
        variant = detector.with_parameters(**parameters)
        scores = variant.score(rows) if shared_scores is None else shared_scores
        results.append(variant.detect_from_scores(rows, scores))
    return results


def tune_detectors(
    detectors: dict[str, Detector],
    rows: pd.DataFrame,
    context: ExpectedContext,
    reference: HostReference,
    model: ExpectedConsumptionModel,
    config: dict[str, Any],
) -> dict[str, Any]:
    scores: dict[str, list[list[float]]] = {
        name: [[] for _ in parameter_grid(name, config)] for name in detectors
    }
    boundaries = context.boundaries
    for seed in config["detection"]["validation_seeds"]:
        print(f"selection {context.name} detection seed {seed}", flush=True)
        events = scenario_events(
            context.hosts,
            config["detection"]["tuning_kinds"],
            seed,
            boundaries.train_end,
            boundaries.validation_end,
            config,
        )
        injected = injected_rows(rows, events, reference, model)
        for name, detector in detectors.items():
            grid = parameter_grid(name, config)
            for position, active in enumerate(detect_variants(detector, injected, grid)):
                scores[name][position].append(evaluate_detection(injected, active, events)["f1"])
    tuning = {}
    for name in detectors:
        grid = parameter_grid(name, config)
        table = [
            {"parameters": parameters, "mean_f1": float(np.mean(values)), "f1_by_seed": values}
            for parameters, values in zip(grid, scores[name], strict=True)
        ]
        best = max(table, key=lambda item: item["mean_f1"])
        tuning[name] = {"grid": table, "chosen_parameters": best["parameters"]}
    return tuning


def detector_path(name: str, detector: Detector, config: dict[str, Any]) -> Path:
    return Path(config["models_dir"]) / f"{name}_detector_{detector.name}_{detector.version}.joblib"


def run_selection(name: str, config: dict[str, Any]) -> Path:
    selection = read_e2_selection(name, config)
    context = load_expected_context(name, config, stop_at_validation_end=True)
    model = load_operational_model(selection, config)
    train = training_rows(context)
    reference = compute_host_reference(train)
    detectors = fit_detectors(train, load_statistics(name, config), config)
    validation = rows_between(
        context.frame, context.boundaries.train_end, context.boundaries.validation_end
    )
    tuning = tune_detectors(detectors, validation, context, reference, model, config)
    normal = with_expected(validation, model)
    files = {}
    for detector_name, detector in detectors.items():
        chosen = detector.with_parameters(**tuning[detector_name]["chosen_parameters"])
        tuning[detector_name]["validation_alert_episodes_per_host_day"] = episodes_per_host_day(
            normal, chosen.detect(normal)
        )
        path = detector_path(name, chosen, config)
        chosen.save(path)
        files[detector_name] = path.name
    metrics = {"stage": "selection", "dataset": name, "tuning": tuning, "detector_files": files}
    record = build_run_record(
        EXPERIMENT_NAME, config, Path(config["datasets"][name]), config["seed"], metrics
    )
    path = Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_selection_{name}.json"
    write_json(record, path)
    return path


def load_detectors(selection: dict[str, Any], config: dict[str, Any]) -> dict[str, Detector]:
    return {
        name: Detector.load(Path(config["models_dir"]) / file_name)
        for name, file_name in selection["detector_files"].items()
    }


def normal_operation(
    detectors: dict[str, Detector], rows: pd.DataFrame, top: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, detector in detectors.items():
        scores = detector.score(rows)
        active = detector.detect_from_scores(rows, scores)
        result[name] = {"alert_episodes_per_host_day": episodes_per_host_day(rows, active)}
        if name == REFERENCE_DETECTOR:
            episodes = sorted(
                extract_episodes(rows, active, scores),
                key=lambda item: item.peak_score,
                reverse=True,
            )
            result[name]["episodes"] = len(episodes)
            result[name]["unexplained_energy_kwh"] = float(
                sum(item.unexplained_energy_kwh for item in episodes)
            )
            result[name]["strongest_episodes"] = [item.to_dict() for item in episodes[:top]]
    return result


def is_negative_family(kinds: Sequence[str]) -> bool:
    return all(kind in NEGATIVE_KINDS for kind in kinds)


def family_runs(
    kinds: Sequence[str],
    detectors: dict[str, Detector],
    rows: pd.DataFrame,
    context: ExpectedContext,
    reference: HostReference,
    model: ExpectedConsumptionModel,
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {name: [] for name in detectors}
    negative = is_negative_family(kinds)
    for seed in config["detection"]["test_seeds"]:
        events = scenario_events(
            context.hosts,
            kinds,
            seed,
            context.boundaries.validation_end,
            context.boundaries.end + GRID_STEP,
            config,
        )
        injected = injected_rows(rows, events, reference, model)
        for name, detector in detectors.items():
            active = detector.detect(injected)
            evaluation = (
                evaluate_negative_control(injected, active, events)
                if negative
                else evaluate_detection(injected, active, events)
            )
            if not negative:
                evaluation["detected_by_kind"] = detected_by_kind(injected, active, events)
            runs[name].append({"seed": int(seed), **evaluation})
    return runs


def detected_by_kind(
    rows: pd.DataFrame, active: np.ndarray, events: Sequence[InjectedEvent]
) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = {}
    for event in events:
        key = f"{event.kind}_{event.magnitude:g}"
        found, _ = event_detection(rows, active, event)
        detected, total = counts.get(key, [0, 0])
        counts[key] = [detected + int(found), total + 1]
    return counts


def summarize_runs(runs: list[dict[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    summary = {}
    for metric in metrics:
        values = [run[metric] for run in runs if run.get(metric) is not None]
        summary[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    return summary


def merge_kind_counts(runs: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, list[int]] = {}
    for run in runs:
        for key, (detected, total) in run.get("detected_by_kind", {}).items():
            current = totals.get(key, [0, 0])
            totals[key] = [current[0] + detected, current[1] + total]
    return {key: detected / total for key, (detected, total) in sorted(totals.items())}


def compare_detectors(runs: dict[str, list[dict[str, Any]]], negative: bool) -> dict[str, Any]:
    metric = "false_positive_rate_on_peaks" if negative else "f1"

    def errors(name: str) -> pd.Series:
        values = pd.Series({run["seed"]: run[metric] for run in runs[name]})
        return values if negative else 1 - values

    return {
        f"{REFERENCE_DETECTOR}_vs_{other}": compare_paired_errors(
            errors(REFERENCE_DETECTOR), errors(other)
        )
        for other in runs
        if other != REFERENCE_DETECTOR
    }


def run_final(name: str, config: dict[str, Any]) -> Path:
    require_frozen_protocol(selection_paths_for(config), freeze_path_for(config))
    e2_selection = read_e2_selection(name, config)
    selection = read_json_metrics(
        Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_selection_{name}.json"
    )
    context = load_expected_context(name, config, stop_at_validation_end=False)
    model = load_operational_model(e2_selection, config)
    reference = compute_host_reference(training_rows(context))
    detectors = load_detectors(selection, config)
    test = rows_between(
        context.frame, context.boundaries.validation_end, context.boundaries.end + GRID_STEP
    )
    families: dict[str, Any] = {}
    for family, kinds in config["detection"]["families"].items():
        print(f"final {name} detection {family}", flush=True)
        negative = is_negative_family(kinds)
        runs = family_runs(kinds, detectors, test, context, reference, model, config)
        metrics_names = (
            ["false_positive_rate_on_peaks", "alert_episodes"]
            if negative
            else [
                "f1",
                "precision",
                "recall",
                "point_f1",
                "mean_delay_minutes",
                "false_alert_episodes",
            ]
        )
        families[family] = {
            "kinds": list(kinds),
            "summary": {
                detector: summarize_runs(items, metrics_names) for detector, items in runs.items()
            },
            "comparisons": compare_detectors(runs, negative),
            "runs": runs,
        }
        if not negative:
            families[family]["detection_rate_by_kind"] = {
                detector: merge_kind_counts(items) for detector, items in runs.items()
            }
    family_comparisons = {
        (family, key): value["p_value"]
        for family, result in families.items()
        for key, value in result["comparisons"].items()
    }
    for (family, key), value in holm_adjust(family_comparisons).items():
        families[family]["comparisons"][key]["p_value_holm"] = value
    metrics = {
        "stage": "test",
        "dataset": name,
        "normal_operation": normal_operation(
            detectors, with_expected(test, model), config["detection"]["top_episodes"]
        ),
        "families": families,
    }
    record = build_run_record(
        EXPERIMENT_NAME, config, Path(config["datasets"][name]), config["seed"], metrics
    )
    path = Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_test_{name}.json"
    write_json(record, path)
    return path


def parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E3: detecção de consumo não explicado")
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
