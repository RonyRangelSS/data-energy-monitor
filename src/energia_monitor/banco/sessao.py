from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker


def criar_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def criar_fabrica_sessoes(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)
