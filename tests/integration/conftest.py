import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine

from energia_monitor.banco.sessao import criar_engine


@pytest.fixture(scope="session")
def url_banco_teste() -> str:
    url = os.environ.get("DATABASE_URL_TESTE")
    if not url:
        from energia_monitor.comum.configuracao import obter_configuracao

        url = obter_configuracao().database_url_teste
    if not url:
        pytest.fail("DATABASE_URL_TESTE não configurada (defina no .env ou no ambiente)")
    return url


@pytest.fixture(scope="session")
def engine_teste(url_banco_teste: str) -> Iterator[Engine]:
    engine = criar_engine(url_banco_teste)
    yield engine
    engine.dispose()
