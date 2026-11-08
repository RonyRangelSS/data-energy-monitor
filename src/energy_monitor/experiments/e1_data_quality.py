import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import yaml

from energy_monitor.common.preparation import PreparedDataset, prepare_dataset
from energy_monitor.common.quality_report import build_quality_report
from energy_monitor.common.splitting import SplitBoundaries, compute_boundaries, local_to_utc
from energy_monitor.experiments.run_record import build_run_record, write_json

EXPERIMENT_NAME: Final = "e1_data_quality"
DEFAULT_CONFIG_PATH: Final = Path("configs/e1_data_quality.yaml")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def boundaries_for(prepared: PreparedDataset, config: dict[str, Any]) -> SplitBoundaries:
    slots = prepared.grid["ts"]
    return compute_boundaries(
        grid_start=slots.min(),
        grid_end=slots.max(),
        analysis_start=local_to_utc(config["analysis_start_local"], config["source_timezone"]),
        train_fraction=config["train_fraction"],
        validation_fraction=config["validation_fraction"],
    )


def run_dataset(name: str, data_path: Path, config: dict[str, Any]) -> Path:
    prepared = prepare_dataset(data_path, config["source_timezone"])
    boundaries = boundaries_for(prepared, config)
    metrics = build_quality_report(prepared, boundaries, config["min_host_coverage"])
    grid_path = Path(config["processed_dir"]) / f"{name}_grid.parquet"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.grid.to_parquet(grid_path, index=False)
    record = build_run_record(EXPERIMENT_NAME, config, data_path, seed=None, metrics=metrics)
    report_path = Path(config["output_dir"]) / f"{EXPERIMENT_NAME}_{name}.json"
    write_json(record, report_path)
    return report_path


def run(config: dict[str, Any]) -> list[Path]:
    return [
        run_dataset(name, Path(data_path), config) for name, data_path in config["datasets"].items()
    ]


def parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1: qualidade dos dados e divisão temporal")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    for report_path in run(load_config(parse_arguments(arguments).config)):
        print(f"Relatório gravado em {report_path}")


if __name__ == "__main__":
    main()
