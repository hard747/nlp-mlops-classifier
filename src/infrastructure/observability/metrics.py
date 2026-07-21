from prometheus_client import Gauge

circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Audit DB circuit breaker state: 0=closed, 1=open, 2=half_open",
)

_STATE_VALUES = {"closed": 0, "open": 1, "half_open": 2}


def set_circuit_breaker_state(state_name: str) -> None:
    circuit_breaker_state.set(_STATE_VALUES[state_name])
