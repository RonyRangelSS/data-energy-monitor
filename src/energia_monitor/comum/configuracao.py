from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuracao(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str
    database_url_teste: str | None = None
    api_key: SecretStr
    tarifa_kwh: float = Field(default=0.80, gt=0)
    fuso_exibicao: str = "Australia/Melbourne"
    log_level: str = "INFO"


@lru_cache
def obter_configuracao() -> Configuracao:
    return Configuracao()
