from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from energia_monitor.banco.base import Base
from energia_monitor.comum.configuracao import obter_configuracao

config = context.config

if config.config_file_name is not None and config.attributes.get("configurar_logs", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def obter_url() -> str:
    return config.get_main_option("sqlalchemy.url") or obter_configuracao().database_url


def executar_offline() -> None:
    context.configure(url=obter_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def executar_online() -> None:
    engine = create_engine(obter_url(), poolclass=pool.NullPool)
    with engine.connect() as conexao:
        context.configure(connection=conexao, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    executar_offline()
else:
    executar_online()
