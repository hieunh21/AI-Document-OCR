"""Alembic environment. URL priority: ``-x url=...`` > ``sqlalchemy.url`` set by the caller
(tests) > ``VIETDOCAI_DATABASE_URL`` / ``.env``."""

from __future__ import annotations

import sys
from pathlib import Path

from alembic import context

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]  # backend/ (the ``app`` package)

from app import models  # noqa: E402,F401  (registers the tables on Base.metadata)
from app.core.config import get_settings  # noqa: E402
from app.db import Base, make_engine  # noqa: E402

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    return (
        context.get_x_argument(as_dictionary=True).get("url")
        or config.get_main_option("sqlalchemy.url")
        or get_settings().database_url
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=_url().startswith("sqlite"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=connection.dialect.name == "sqlite",
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
