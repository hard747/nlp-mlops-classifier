from sqlalchemy.ext.asyncio import async_sessionmaker

from src.domain.models import AuditLogEntry
from src.infrastructure.database.models import AuditLog


class SqlAlchemyAuditLogRepository:
    """Implements AuditLogRepositoryPort with a single batched INSERT per call."""

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._session_factory = session_factory

    async def save_batch(self, entries: list[AuditLogEntry]) -> None:
        if not entries:
            return
        async with self._session_factory() as session:
            session.add_all(
                [
                    AuditLog(
                        request_id=entry.request_id,
                        input_text=entry.input_text,
                        predicted_intent=entry.predicted_intent,
                        confidence=entry.confidence,
                        latency_ms=entry.latency_ms,
                        created_at=entry.created_at,
                    )
                    for entry in entries
                ]
            )
            await session.commit()
