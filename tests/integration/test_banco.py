from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect, text

RAIZ = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.integration


def test_conecta_ao_banco_de_teste(engine_teste: Engine) -> None:
    with engine_teste.connect() as conexao:
        assert conexao.execute(text("SELECT 1")).scalar_one() == 1


def test_alembic_upgrade_head_executa(url_banco_teste: str, engine_teste: Engine) -> None:
    configuracao = Config(str(RAIZ / "alembic.ini"))
    configuracao.set_main_option("sqlalchemy.url", url_banco_teste)
    configuracao.attributes["configurar_logs"] = False

    command.upgrade(configuracao, "head")

    assert "alembic_version" in inspect(engine_teste).get_table_names()
