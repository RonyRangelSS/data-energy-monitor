from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.integration


def test_connects_to_test_database(test_engine: Engine) -> None:
    with test_engine.connect() as connection:
        assert connection.execute(text("SELECT 1")).scalar_one() == 1


def test_alembic_upgrade_head_runs(test_database_url: str, test_engine: Engine) -> None:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", test_database_url)
    config.attributes["configure_logging"] = False

    command.upgrade(config, "head")

    assert "alembic_version" in inspect(test_engine).get_table_names()
