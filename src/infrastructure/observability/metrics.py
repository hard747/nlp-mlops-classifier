from prometheus_client import Gauge

circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Audit DB circuit breaker state: 0=closed, 1=open, 2=half_open",
)

_STATE_VALUES = {"closed": 0, "open": 1, "half_open": 2}


def set_circuit_breaker_state(state_name: str) -> None:
    circuit_breaker_state.set(_STATE_VALUES[state_name])


prediction_avg_confidence = Gauge(
    "prediction_avg_confidence_1h",
    "Rolling average model confidence over the drift monitoring window",
)

prediction_drift_score = Gauge(
    "prediction_drift_score",
    "Absolute deviation between rolling average confidence and the training-time baseline",
)


def set_drift_metrics(avg_confidence: float, drift_score: float) -> None:
    prediction_avg_confidence.set(avg_confidence)
    prediction_drift_score.set(drift_score)
