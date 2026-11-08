import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine

from energy_monitor.common.settings import get_settings
from energy_monitor.database.session import create_db_engine


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
