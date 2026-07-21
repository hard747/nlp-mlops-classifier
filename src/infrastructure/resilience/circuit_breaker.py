import asyncio
import enum
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is short-circuited because the breaker is OPEN."""


class CircuitBreaker:
    """Hand-rolled CLOSED -> OPEN -> HALF_OPEN -> CLOSED state machine.

    - CLOSED: calls pass through. `failure_threshold` consecutive failures trip it OPEN.
    - OPEN: calls are short-circuited immediately (no downstream call at all) until
      `cooldown_seconds` elapses, then the next call is allowed through as a trial (HALF_OPEN).
    - HALF_OPEN: exactly one trial call is allowed. Success closes the breaker and resets
      the failure count; failure reopens it and restarts the cooldown.

    Not thread-safe across event loops by design — one breaker per asyncio app.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_trial_in_flight = False

    @property
    def state(self) -> CircuitState:
        if self._state is CircuitState.OPEN and self._cooldown_elapsed():
            return CircuitState.HALF_OPEN
        return self._state

    def _cooldown_elapsed(self) -> bool:
        return self._opened_at is not None and (self._clock() - self._opened_at) >= self._cooldown_seconds

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        current = self.state

        if current is CircuitState.OPEN:
            raise CircuitOpenError("circuit breaker is OPEN, call short-circuited")

        if current is CircuitState.HALF_OPEN:
            if self._half_open_trial_in_flight:
                raise CircuitOpenError("circuit breaker is HALF_OPEN, trial call already in flight")
            self._half_open_trial_in_flight = True
            try:
                result = await func()
            except Exception:
                self._trip_open()
                raise
            else:
                self._reset_closed()
                return result
            finally:
                self._half_open_trial_in_flight = False

        # CLOSED
        try:
            result = await func()
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._trip_open()
            raise
        else:
            self._consecutive_failures = 0
            return result

    def _trip_open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()

    def _reset_closed(self) -> None:
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None


async def call_with_timeout(
    breaker: CircuitBreaker, func: Callable[[], Awaitable[T]], timeout_seconds: float
) -> T:
    async def _bounded() -> T:
        return await asyncio.wait_for(func(), timeout=timeout_seconds)

    return await breaker.call(_bounded)
