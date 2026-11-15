import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import yaml

from energy_monitor.experiments.run_record import describe_code_version, hash_file, write_json

FREEZE_FILE_NAME: Final = "protocol_freeze.json"
DEFAULT_CONFIG_PATH: Final = Path("configs/e5_forecasting.yaml")


class ProtocolNotFrozenError(RuntimeError):
    pass


def freeze_protocol(selection_paths: Sequence[Path], freeze_path: Path) -> dict[str, Any]:
    missing = [str(path) for path in selection_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Seleções ausentes: {missing}")
    version = describe_code_version()
    if version["has_uncommitted_changes"] is not False:
        raise RuntimeError("Faça o commit do código antes de congelar o protocolo")
    freeze = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "code_version": version,
        "selection_sha256": {path.name: hash_file(path) for path in selection_paths},
    }
    write_json(freeze, freeze_path)
    return freeze


def require_frozen_protocol(selection_paths: Sequence[Path], freeze_path: Path) -> None:
    if not freeze_path.exists():
        raise ProtocolNotFrozenError(f"Protocolo não congelado: {freeze_path} não existe")
    recorded = json.loads(freeze_path.read_text(encoding="utf-8"))["selection_sha256"]
    for path in selection_paths:
        if recorded.get(path.name) != hash_file(path):
            raise ProtocolNotFrozenError(f"{path.name} mudou depois do congelamento")


def selection_paths_for(config: dict[str, Any]) -> list[Path]:
    output_dir = Path(config["output_dir"])
    if "selection_files" in config:
        return [output_dir / name for name in config["selection_files"]]
    return [
        output_dir / f"{config['experiment']}_selection_{name}.json" for name in config["datasets"]
    ]


def freeze_path_for(config: dict[str, Any]) -> Path:
    return Path(config["output_dir"]) / config.get("freeze_file", FREEZE_FILE_NAME)


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Congela o protocolo antes do teste")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config = yaml.safe_load(parser.parse_args(arguments).config.read_text(encoding="utf-8"))
    freeze_protocol(selection_paths_for(config), freeze_path_for(config))
    print(f"Protocolo congelado em {freeze_path_for(config)}")


if __name__ == "__main__":
    main()
