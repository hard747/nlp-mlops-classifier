from typing import Protocol

from src.domain.models import AuditLogEntry, IntentPrediction


class IntentClassifierPort(Protocol):
    """Adapters implementing this run inference over raw text."""

    def predict(self, text: str) -> IntentPrediction: ...


class AuditLogRepositoryPort(Protocol):
    """Adapters implementing this persist audit entries in a batch."""

    async def save_batch(self, entries: list[AuditLogEntry]) -> None: ...
