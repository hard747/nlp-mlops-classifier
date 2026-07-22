import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.domain.services import ClassificationService
from src.infrastructure.config import settings
from src.infrastructure.database.audit_repository import SqlAlchemyAuditLogRepository
from src.infrastructure.database.batch_writer import FanInBatchWriter
from src.infrastructure.database.session import async_session_factory
from src.infrastructure.ml_model.classifier_adapter import TransformerClassifierAdapter
from src.infrastructure.observability.drift import DriftMonitor
from src.infrastructure.resilience.circuit_breaker import CircuitBreaker
from src.infrastructure.api.routers import health, predict

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = TransformerClassifierAdapter(settings.model_path)
    app.state.classification_service = ClassificationService(classifier)
    app.state.model_ready = True

    repository = SqlAlchemyAuditLogRepository(async_session_factory)
    breaker = CircuitBreaker(
        failure_threshold=settings.breaker_failure_threshold,
        cooldown_seconds=settings.breaker_cooldown_seconds,
    )
    batch_writer = FanInBatchWriter(
        repository=repository,
        breaker=breaker,
        max_queue_size=settings.batch_queue_max_size,
        max_batch_size=settings.batch_max_size,
        max_interval_ms=settings.batch_max_interval_ms,
        db_timeout_seconds=settings.db_timeout_ms / 1000,
    )
    app.state.batch_writer = batch_writer
    batch_writer.start()

    drift_monitor = DriftMonitor(
        session_factory=async_session_factory,
        baseline_confidence=settings.drift_baseline_confidence,
        window_minutes=settings.drift_window_minutes,
        check_interval_seconds=settings.drift_check_interval_seconds,
        alert_threshold=settings.drift_alert_threshold,
    )
    app.state.drift_monitor = drift_monitor
    drift_monitor.start()

    logger.info("startup complete: model loaded, batch writer running")
    yield

    await drift_monitor.stop()
    await batch_writer.stop()
    logger.info("shutdown complete: audit queue drained")


def create_app() -> FastAPI:
    app = FastAPI(title="NLP MLOps Classifier", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(predict.router)
    # Exposes GET /metrics in Prometheus text format (request counts, latency
    # histograms per route) for the Prometheus/Grafana stack in infra/.
    Instrumentator().instrument(app).expose(app)
    return app


app = create_app()
