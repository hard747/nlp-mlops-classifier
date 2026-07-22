import pytest
from pydantic import ValidationError

from src.infrastructure.api.schemas import PredictRequest, PredictResponse


def test_predict_request_accepts_valid_text():
    request = PredictRequest(text="I want to cancel my order")
    assert request.text == "I want to cancel my order"


def test_predict_request_rejects_empty_text():
    with pytest.raises(ValidationError):
        PredictRequest(text="")


def test_predict_request_rejects_text_over_max_length():
    with pytest.raises(ValidationError):
        PredictRequest(text="a" * 2001)


def test_predict_request_rejects_missing_text():
    with pytest.raises(ValidationError):
        PredictRequest()


def test_predict_response_round_trip():
    response = PredictResponse(
        intent="cancel_order", confidence=0.98, latency_ms=12.3, request_id="abc-123"
    )
    assert response.model_dump() == {
        "intent": "cancel_order",
        "confidence": 0.98,
        "latency_ms": 12.3,
        "request_id": "abc-123",
    }
