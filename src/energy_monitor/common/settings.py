from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str
    test_database_url: str | None = None
    api_key: SecretStr
    tariff_kwh: float = Field(default=0.80, gt=0)
    display_timezone: str = "Australia/Melbourne"
    log_level: str = "INFO"
    models_dir: str = "artifacts/models"
    experiments_dir: str = "artifacts/experiments"
    operational_dataset: str = "d1"
    api_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
