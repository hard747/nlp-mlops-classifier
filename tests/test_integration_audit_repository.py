import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.domain.models import AuditLogEntry
from src.infrastructure.database.audit_repository import SqlAlchemyAuditLogRepository
from src.infrastructure.database.models import AuditLog

pytestmark = pytest.mark.integration


@pytest.fixture
async def session_factory():
    database_url = os.environ["DATABASE_URL"]
    engine = create_async_engine(database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def test_save_batch_persists_and_is_readable_back(session_factory):
    repository = SqlAlchemyAuditLogRepository(session_factory)
    request_id = str(uuid.uuid4())
    entry = AuditLogEntry(
        request_id=request_id,
        input_text="I need a refund",
        predicted_intent="refund_request",
        confidence=0.87,
        latency_ms=15.2,
        created_at=datetime.now(timezone.utc),
    )

    await repository.save_batch([entry])

    async with session_factory() as session:
        result = await session.execute(select(AuditLog).where(AuditLog.request_id == request_id))
        row = result.scalar_one()
        assert row.predicted_intent == "refund_request"
        assert row.confidence == pytest.approx(0.87)
