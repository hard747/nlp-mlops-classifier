import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_fastapi_instrumentator import Instrumentator

from src.infrastructure.observability.metrics import set_circuit_breaker_state


@pytest.fixture
def instrumented_client():
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"pong": True}

    Instrumentator().instrument(app).expose(app)
    with TestClient(app) as client:
        yield client


def test_metrics_endpoint_exposes_prometheus_format(instrumented_client):
    instrumented_client.get("/ping")

    response = instrumented_client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_set_circuit_breaker_state_updates_gauge(instrumented_client):
    set_circuit_breaker_state("open")

    response = instrumented_client.get("/metrics")

    assert "circuit_breaker_state 1.0" in response.text


def test_set_circuit_breaker_state_rejects_unknown_state():
    with pytest.raises(KeyError):
        set_circuit_breaker_state("not_a_real_state")
