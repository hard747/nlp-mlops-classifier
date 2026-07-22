import pytest

from src.infrastructure.observability import metrics
from src.infrastructure.observability.drift import DriftMonitor


class FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class FakeSession:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, _query):
        return FakeResult(self._value)


def make_monitor(avg_confidence, baseline=0.95, alert_threshold=0.05):
    return DriftMonitor(
        session_factory=lambda: FakeSession(avg_confidence),
        baseline_confidence=baseline,
        window_minutes=60,
        check_interval_seconds=300,
        alert_threshold=alert_threshold,
    )


async def test_check_once_sets_gauges_and_flags_drift(caplog):
    monitor = make_monitor(avg_confidence=0.70)

    with caplog.at_level("WARNING"):
        await monitor._check_once()

    assert metrics.prediction_avg_confidence._value.get() == 0.70
    assert metrics.prediction_drift_score._value.get() == pytest.approx(0.25)
    assert "drift detected" in caplog.text


async def test_check_once_no_alert_when_within_threshold(caplog):
    monitor = make_monitor(avg_confidence=0.94)

    with caplog.at_level("WARNING"):
        await monitor._check_once()

    assert "drift detected" not in caplog.text


async def test_check_once_skips_when_no_data_yet():
    monitor = make_monitor(avg_confidence=None)

    await monitor._check_once()
