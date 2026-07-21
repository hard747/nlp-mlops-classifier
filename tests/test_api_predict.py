from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.domain.models import AuditLogEntry, IntentPrediction
from src.domain.services import ClassificationService
from src.infrastructure.api.dependencies import get_batch_writer, get_classification_service
from src.infrastructure.api.routers import health, predict


class FakeClassifier:
    def predict(self, text: str) -> IntentPrediction:
        return IntentPrediction(intent="cancel_order", confidence=0.99, latency_ms=1.0)


class FakeBatchWriter:
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def enqueue(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


@pytest.fixture
def fake_batch_writer():
    return FakeBatchWriter()


@pytest.fixture
def app(fake_batch_writer):
    @asynccontextmanager
    async def noop_lifespan(app: FastAPI):
        app.state.model_ready = True
        yield

    test_app = FastAPI(lifespan=noop_lifespan)
    test_app.include_router(health.router)
    test_app.include_router(predict.router)

    test_app.dependency_overrides[get_classification_service] = lambda: ClassificationService(
        FakeClassifier()
    )
    test_app.dependency_overrides[get_batch_writer] = lambda: fake_batch_writer
    return test_app


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def test_health_always_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_200_when_model_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"model_ready": True}


def test_predict_returns_intent_and_confidence(client):
    response = client.post("/predict", json={"text": "I want to cancel my order"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "cancel_order"
    assert body["confidence"] == 0.99
    assert "request_id" in body
    assert "latency_ms" in body


def test_predict_enqueues_audit_entry(client, fake_batch_writer):
    client.post("/predict", json={"text": "I want to cancel my order"})
    assert len(fake_batch_writer.entries) == 1
    assert fake_batch_writer.entries[0].predicted_intent == "cancel_order"


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
