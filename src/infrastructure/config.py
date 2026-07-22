from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/nlp_audit_db"
    model_path: str = "./src/infrastructure/ml_model/weights"

    @field_validator("database_url")
    @classmethod
    def _force_asyncpg_driver(cls, value: str) -> str:
        # Render's managed Postgres (and most providers) hand back a bare
        # postgresql:// DSN; the app requires the asyncpg driver explicitly.
        if value.startswith("postgresql://"):
            value = value.replace("postgresql://", "postgresql+asyncpg://", 1)

        # Connection strings copied from managed providers (Neon, Supabase, ...)
        # use libpq-style query params. asyncpg.connect() takes the same
        # 'require'/'verify-full'/etc. values under the name `ssl`, not
        # `sslmode`, and doesn't recognize `channel_binding` at all - passed
        # through unchanged they surface as "connect() got an unexpected
        # keyword argument" at the first real connection attempt.
        parts = urlsplit(value)
        query = dict(parse_qsl(parts.query))
        if "sslmode" in query:
            query["ssl"] = query.pop("sslmode")
        query.pop("channel_binding", None)
        return urlunsplit(parts._replace(query=urlencode(query)))

    db_timeout_ms: int = 500

    batch_max_size: int = 50
    batch_max_interval_ms: int = 2000
    batch_queue_max_size: int = 1000

    breaker_failure_threshold: int = 5
    breaker_cooldown_seconds: float = 30.0

    drift_check_interval_seconds: int = 300
    drift_window_minutes: int = 60
    # Mean prediction confidence on the Phase 2 held-out validation set; override
    # via env once a real training run reports its own eval confidence.
    drift_baseline_confidence: float = 0.95
    drift_alert_threshold: float = 0.05


settings = Settings()
