import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from energy_monitor.api.app import create_app
from energy_monitor.api.operational import OperationalModels
from energy_monitor.common.settings import Settings, get_settings
from energy_monitor.database.session import create_db_engine
from tests.integration.api_helpers import build_operational_models

ROOT = Path(__file__).resolve().parents[2]
API_KEY = "integration-key"
TABLES = ("readings", "alerts", "alert_states", "forecasts", "experiment_runs")


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL") or get_settings().test_database_url
    if not url:
        pytest.fail("TEST_DATABASE_URL não configurada (defina no .env ou no ambiente)")
    return url


@pytest.fixture(scope="session")
def test_engine(test_database_url: str) -> Iterator[Engine]:
    engine = create_db_engine(test_database_url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def migrated_engine(test_database_url: str, test_engine: Engine) -> Engine:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", test_database_url)
    config.attributes["configure_logging"] = False
    command.upgrade(config, "head")
    return test_engine


@pytest.fixture
def clean_engine(migrated_engine: Engine) -> Engine:
    with migrated_engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY"))
    return migrated_engine


@pytest.fixture(scope="session")
def operational_models() -> OperationalModels:
    return build_operational_models()


@pytest.fixture
def settings(test_database_url: str) -> Settings:
    return Settings(_env_file=None, database_url=test_database_url, api_key=API_KEY, tariff_kwh=0.8)


@pytest.fixture
def api_client(
    settings: Settings, operational_models: OperationalModels, clean_engine: Engine
) -> Iterator[TestClient]:
    app = create_app(settings, operational_models, clean_engine)
    with TestClient(app, headers={"X-API-Key": API_KEY}) as client:
        yield client
