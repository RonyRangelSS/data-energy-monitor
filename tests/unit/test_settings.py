import pytest
from pydantic import ValidationError

from energy_monitor.common.settings import Settings

DATABASE_URL = "postgresql+psycopg://u:s@localhost:5434/energy"


@pytest.fixture
def minimal_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("API_KEY", "test-key")


def test_loads_values_from_environment(minimal_environment: None) -> None:
    settings = Settings(_env_file=None)

    assert settings.database_url == DATABASE_URL
    assert settings.api_key.get_secret_value() == "test-key"


def test_uses_defaults_for_optional_values(minimal_environment: None) -> None:
    settings = Settings(_env_file=None)

    assert settings.display_timezone == "Australia/Melbourne"
    assert settings.test_database_url is None


def test_fails_without_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("API_KEY", "test-key")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_fails_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.delenv("API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rejects_non_positive_tariff(
    minimal_environment: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TARIFF_KWH", "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_api_key_is_hidden_in_repr(minimal_environment: None) -> None:
    assert "test-key" not in repr(Settings(_env_file=None))
