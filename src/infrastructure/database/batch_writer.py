import asyncio
import logging
from collections import deque

from src.domain.models import AuditLogEntry
from src.domain.ports import AuditLogRepositoryPort
from src.infrastructure.observability.metrics import set_circuit_breaker_state
from src.infrastructure.resilience.circuit_breaker import CircuitBreaker, call_with_timeout

logger = logging.getLogger(__name__)


class FanInBatchWriter:
    """Collects AuditLogEntry items from many requests and flushes them in batches.

    /predict calls `enqueue()` which is non-blocking (queue.put_nowait). A single
    background task drains the queue, batching up to `max_batch_size` items or
    every `max_interval_ms`, whichever comes first, and writes them through a
    circuit breaker. If the breaker is open or the write fails, entries fall back
    to a bounded, oldest-evicting in-memory deque and are retried on the next flush.
    """

    def __init__(
        self,
        repository: AuditLogRepositoryPort,
        breaker: CircuitBreaker,
        max_queue_size: int = 1000,
        max_batch_size: int = 50,
        max_interval_ms: int = 2000,
        db_timeout_seconds: float = 0.5,
        fallback_maxlen: int = 5000,
    ) -> None:
        self._repository = repository
        self._breaker = breaker
        self._queue: asyncio.Queue[AuditLogEntry] = asyncio.Queue(maxsize=max_queue_size)
        self._max_batch_size = max_batch_size
        self._max_interval_seconds = max_interval_ms / 1000
        self._db_timeout_seconds = db_timeout_seconds
        self._fallback: deque[AuditLogEntry] = deque(maxlen=fallback_maxlen)
        self._task: asyncio.Task | None = None
        self._stopping = False

    def enqueue(self, entry: AuditLogEntry) -> None:
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            logger.warning("audit queue full, dropping entry request_id=%s", entry.request_id)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self, grace_period_seconds: float = 5.0) -> None:
        self._stopping = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=grace_period_seconds)
            except asyncio.TimeoutError:
                self._task.cancel()
        await self._flush_once()

    async def _run(self) -> None:
        while not self._stopping:
            batch = await self._collect_batch()
            if batch:
                await self._write_batch(batch)

    async def _collect_batch(self) -> list[AuditLogEntry]:
        batch: list[AuditLogEntry] = []
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=self._max_interval_seconds)
            batch.append(first)
        except asyncio.TimeoutError:
            return batch

        while len(batch) < self._max_batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _flush_once(self) -> None:
        batch: list[AuditLogEntry] = []
        while not self._queue.empty() and len(batch) < self._max_batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if batch:
            await self._write_batch(batch)

    async def _write_batch(self, batch: list[AuditLogEntry]) -> None:
        # Oldest-first: flush anything stranded from a previous outage before the new batch.
        to_write = list(self._fallback) + batch

        async def _do_write() -> None:
            await self._repository.save_batch(to_write)

        try:
            await call_with_timeout(self._breaker, _do_write, self._db_timeout_seconds)
            self._fallback.clear()
        except Exception as exc:
            logger.warning("audit batch write failed (%s), buffering %d entries", exc, len(to_write))
            self._fallback.extend(to_write)
        finally:
            set_circuit_breaker_state(self._breaker.state.value)
