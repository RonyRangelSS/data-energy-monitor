import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from energy_monitor.common.settings import get_settings
from energy_monitor.database.models import ExperimentRun
from energy_monitor.database.session import create_db_engine

REPORT_PATTERN = "*.json"


def run_values(path: Path) -> dict[str, Any] | None:
    content = json.loads(path.read_text(encoding="utf-8"))
    if "experiment" not in content or "metrics" not in content:
        return None
    return {
        "file_name": path.name,
        "experiment": content["experiment"],
        "created_at": content.get("created_at"),
        "commit": (content.get("code_version") or {}).get("commit"),
        "seed": content.get("seed"),
        "data_sha256": content.get("data_sha256"),
        "configuration": content.get("configuration", {}),
        "metrics": content["metrics"],
    }


def import_experiments(session: Session, directory: Path) -> int:
    imported = 0
    for path in sorted(directory.glob(REPORT_PATTERN)):
        values = run_values(path)
        if values is None:
            continue
        statement = insert(ExperimentRun).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=["file_name"],
            set_={key: statement.excluded[key] for key in values if key != "file_name"},
        )
        session.execute(statement)
        imported += 1
    session.commit()
    return imported


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Importa os registros de experimentos para o banco"
    )
    parser.add_argument("--directory", type=Path, default=Path("artifacts/experiments"))
    directory = parser.parse_args(arguments).directory
    with Session(create_db_engine(get_settings().database_url)) as session:
        print(f"Registros importados: {import_experiments(session, directory)}")


if __name__ == "__main__":
    main()
