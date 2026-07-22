import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.infrastructure.database.models import AuditLog
from src.infrastructure.observability.metrics import set_drift_metrics

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Periodically compares recent prediction confidence against the training-time baseline.

    A confidence gap beyond `alert_threshold` is a cheap, label-free proxy for
    distribution shift - real accuracy drift can't be measured in production
    without ground-truth labels, but a sustained drop in the model's own
    confidence is usually the first visible symptom of it.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker,
        baseline_confidence: float,
        window_minutes: int,
        check_interval_seconds: int,
        alert_threshold: float,
    ) -> None:
        self._session_factory = session_factory
        self._baseline_confidence = baseline_confidence
        self._window = timedelta(minutes=window_minutes)
        self._interval_seconds = check_interval_seconds
        self._alert_threshold = alert_threshold
        self._task: asyncio.Task | None = None
        self._stopping = False

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()

    async def _run(self) -> None:
        while not self._stopping:
            try:
                await self._check_once()
            except Exception:
                logger.exception("drift check failed")
            await asyncio.sleep(self._interval_seconds)

    async def _check_once(self) -> None:
        window_start = datetime.now(timezone.utc) - self._window
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.avg(AuditLog.confidence)).where(AuditLog.created_at >= window_start)
            )
            avg_confidence = result.scalar()

        if avg_confidence is None:
            return

        drift_score = abs(avg_confidence - self._baseline_confidence)
        set_drift_metrics(avg_confidence, drift_score)

        if drift_score >= self._alert_threshold:
            logger.warning(
                "prediction confidence drift detected: avg=%.4f baseline=%.4f score=%.4f",
                avg_confidence, self._baseline_confidence, drift_score,
            )
