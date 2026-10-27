import pytest
from pydantic import ValidationError

from energia_monitor.comum.configuracao import Configuracao


@pytest.fixture
def ambiente_minimo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:s@localhost:5432/energia")
    monkeypatch.setenv("API_KEY", "chave-teste")


def test_carrega_valores_do_ambiente(ambiente_minimo: None) -> None:
    configuracao = Configuracao(_env_file=None)

    assert configuracao.database_url.endswith("/energia")
    assert configuracao.api_key.get_secret_value() == "chave-teste"


def test_usa_padroes_quando_opcionais_ausentes(ambiente_minimo: None) -> None:
    configuracao = Configuracao(_env_file=None)

    assert configuracao.fuso_exibicao == "Australia/Melbourne"
    assert configuracao.database_url_teste is None


def test_falha_sem_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("API_KEY", "chave-teste")

    with pytest.raises(ValidationError):
        Configuracao(_env_file=None)


def test_falha_sem_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:s@localhost:5432/energia")
    monkeypatch.delenv("API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Configuracao(_env_file=None)


def test_rejeita_tarifa_nao_positiva(
    ambiente_minimo: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TARIFA_KWH", "0")

    with pytest.raises(ValidationError):
        Configuracao(_env_file=None)


def test_api_key_nao_aparece_na_representacao(ambiente_minimo: None) -> None:
    configuracao = Configuracao(_env_file=None)

    assert "chave-teste" not in repr(configuracao)
