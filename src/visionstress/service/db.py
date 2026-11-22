"""Database connection. Runtime uses PostgreSQL; tests may use SQLite."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def database_url() -> str:
    return os.getenv(
        "VISIONSTRESS_DATABASE_URL",
        "postgresql+psycopg://visionstress:visionstress@localhost:5432/visionstress",
    )


def make_engine(url: str | None = None):
    url = url or database_url()
    if url.startswith("sqlite") and not os.getenv("VISIONSTRESS_TESTING"):
        raise RuntimeError("SQLite is reserved for automated tests; configure PostgreSQL")
    return create_engine(url, pool_pre_ping=True)


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
