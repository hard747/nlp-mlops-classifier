from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float
    latency_ms: float


@dataclass(frozen=True)
class AuditLogEntry:
    request_id: str
    input_text: str
    predicted_intent: str
    confidence: float
    latency_ms: float
    created_at: datetime
