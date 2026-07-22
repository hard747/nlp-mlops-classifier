from src.infrastructure.config import Settings


def test_forces_asyncpg_driver_on_bare_postgres_url():
    settings = Settings(database_url="postgresql://user:pw@host:5432/db")

    assert settings.database_url.startswith("postgresql+asyncpg://")


def test_translates_neon_style_sslmode_to_asyncpg_ssl_param():
    settings = Settings(
        database_url="postgresql://user:pw@ep-example.neon.tech/db?sslmode=require&channel_binding=require"
    )

    assert "ssl=require" in settings.database_url
    assert "sslmode" not in settings.database_url
    assert "channel_binding" not in settings.database_url


def test_leaves_url_without_query_params_unchanged():
    settings = Settings(database_url="postgresql+asyncpg://user:pw@host:5432/db")

    assert settings.database_url == "postgresql+asyncpg://user:pw@host:5432/db"
