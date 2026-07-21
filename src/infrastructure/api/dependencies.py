from fastapi import Request

from src.domain.services import ClassificationService
from src.infrastructure.database.batch_writer import FanInBatchWriter


def get_classification_service(request: Request) -> ClassificationService:
    return request.app.state.classification_service


def get_batch_writer(request: Request) -> FanInBatchWriter:
    return request.app.state.batch_writer
