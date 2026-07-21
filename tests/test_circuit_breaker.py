import pytest

from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    call_with_timeout,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def ok() -> str:
    return "ok"


async def fail() -> str:
    raise RuntimeError("boom")


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def breaker(clock):
    return CircuitBreaker(failure_threshold=3, cooldown_seconds=10.0, clock=clock)


async def test_starts_closed(breaker):
    assert breaker.state is CircuitState.CLOSED
    assert await breaker.call(ok) == "ok"


async def test_stays_closed_below_threshold(breaker):
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(fail)
    assert breaker.state is CircuitState.CLOSED


async def test_trips_open_after_consecutive_failures(breaker):
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(fail)
    assert breaker.state is CircuitState.OPEN


async def test_open_short_circuits_without_calling_downstream(breaker):
    calls = 0

    async def counting_fail():
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(counting_fail)
    assert calls == 3

    with pytest.raises(CircuitOpenError):
        await breaker.call(counting_fail)
    assert calls == 3  # short-circuited, downstream never invoked


async def test_success_resets_failure_count(breaker):
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(fail)
    await breaker.call(ok)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(fail)
    assert breaker.state is CircuitState.CLOSED  # threshold never hit again


async def test_transitions_to_half_open_after_cooldown(breaker, clock):
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(fail)
    assert breaker.state is CircuitState.OPEN

    clock.advance(10.0)
    assert breaker.state is CircuitState.HALF_OPEN


async def test_half_open_success_closes_breaker(breaker, clock):
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(fail)
    clock.advance(10.0)

    assert await breaker.call(ok) == "ok"
    assert breaker.state is CircuitState.CLOSED


async def test_half_open_failure_reopens_breaker(breaker, clock):
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(fail)
    clock.advance(10.0)

    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    assert breaker.state is CircuitState.OPEN


async def test_call_with_timeout_times_out_and_counts_as_failure(breaker):
    import asyncio

    async def slow():
        await asyncio.sleep(10)

    with pytest.raises(asyncio.TimeoutError):
        await call_with_timeout(breaker, slow, timeout_seconds=0.01)
    with pytest.raises(asyncio.TimeoutError):
        await call_with_timeout(breaker, slow, timeout_seconds=0.01)
    with pytest.raises(asyncio.TimeoutError):
        await call_with_timeout(breaker, slow, timeout_seconds=0.01)

    assert breaker.state is CircuitState.OPEN
