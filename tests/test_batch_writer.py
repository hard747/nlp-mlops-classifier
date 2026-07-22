import asyncio
from datetime import datetime, timezone

import pytest

from src.domain.models import AuditLogEntry
from src.infrastructure.database.batch_writer import FanInBatchWriter
from src.infrastructure.resilience.circuit_breaker import CircuitBreaker


class FakeRepository:
    def __init__(self, fail_times: int = 0) -> None:
        self.saved_batches: list[list[AuditLogEntry]] = []
        self._fail_times = fail_times

    async def save_batch(self, entries: list[AuditLogEntry]) -> None:
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("db unavailable")
        self.saved_batches.append(entries)


def make_entry(request_id: str = "r1") -> AuditLogEntry:
    return AuditLogEntry(
        request_id=request_id,
        input_text="hello",
        predicted_intent="greet",
        confidence=0.9,
        latency_ms=5.0,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def breaker():
    return CircuitBreaker(failure_threshold=5, cooldown_seconds=30.0)


async def test_flush_writes_batched_entries(breaker):
    repo = FakeRepository()
    writer = FanInBatchWriter(
        repository=repo, breaker=breaker, max_batch_size=10, max_interval_ms=50, db_timeout_seconds=1.0
    )
    writer.enqueue(make_entry("a"))
    writer.enqueue(make_entry("b"))

    await writer._flush_once()

    assert len(repo.saved_batches) == 1
    assert [e.request_id for e in repo.saved_batches[0]] == ["a", "b"]


async def test_background_loop_flushes_on_interval(breaker):
    repo = FakeRepository()
    writer = FanInBatchWriter(
        repository=repo, breaker=breaker, max_batch_size=10, max_interval_ms=20, db_timeout_seconds=1.0
    )
    writer.start()
    writer.enqueue(make_entry("a"))

    await asyncio.sleep(0.1)
    await writer.stop(grace_period_seconds=1.0)

    assert any(e.request_id == "a" for batch in repo.saved_batches for e in batch)


async def test_failed_write_falls_back_to_deque_and_retries(breaker):
    repo = FakeRepository(fail_times=1)
    writer = FanInBatchWriter(
        repository=repo, breaker=breaker, max_batch_size=10, max_interval_ms=50, db_timeout_seconds=1.0
    )
    writer.enqueue(make_entry("a"))

    await writer._flush_once()
    assert repo.saved_batches == []
    assert len(writer._fallback) == 1

    writer.enqueue(make_entry("b"))
    await writer._flush_once()

    assert len(repo.saved_batches) == 1
    assert [e.request_id for e in repo.saved_batches[0]] == ["a", "b"]
    assert len(writer._fallback) == 0


async def test_queue_full_drops_entry_without_raising(breaker):
    repo = FakeRepository()
    writer = FanInBatchWriter(
        repository=repo, breaker=breaker, max_queue_size=1, max_batch_size=10, max_interval_ms=50
    )
    writer.enqueue(make_entry("a"))
    writer.enqueue(make_entry("b"))  # queue full, should be dropped silently

    assert writer._queue.qsize() == 1
