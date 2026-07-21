import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from src.domain.models import AuditLogEntry
from src.domain.services import ClassificationService
from src.infrastructure.api.dependencies import get_batch_writer, get_classification_service
from src.infrastructure.api.schemas import PredictRequest, PredictResponse
from src.infrastructure.database.batch_writer import FanInBatchWriter

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    service: ClassificationService = Depends(get_classification_service),
    batch_writer: FanInBatchWriter = Depends(get_batch_writer),
) -> PredictResponse:
    request_id = str(uuid.uuid4())

    # Model forward pass is CPU-bound/blocking; keep it off the event loop.
    prediction = await run_in_threadpool(service.classify, payload.text)

    batch_writer.enqueue(
        AuditLogEntry(
            request_id=request_id,
            input_text=payload.text,
            predicted_intent=prediction.intent,
            confidence=prediction.confidence,
            latency_ms=prediction.latency_ms,
            created_at=datetime.now(timezone.utc),
        )
    )

    return PredictResponse(
        intent=prediction.intent,
        confidence=prediction.confidence,
        latency_ms=prediction.latency_ms,
        request_id=request_id,
    )
