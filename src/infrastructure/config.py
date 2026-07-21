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
            return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        return value
    db_timeout_ms: int = 500

    batch_max_size: int = 50
    batch_max_interval_ms: int = 2000
    batch_queue_max_size: int = 1000

    breaker_failure_threshold: int = 5
    breaker_cooldown_seconds: float = 30.0


settings = Settings()
