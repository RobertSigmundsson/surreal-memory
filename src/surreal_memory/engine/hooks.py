"""Generic event hook system for Surreal-Memory operations.

Provides a typed event bus that plugins, extensions, and internal
components can use to react to memory operations without modifying
core code.

Usage:
    registry = HookRegistry()
    registry.on(HookEvent.POST_REMEMBER, my_listener)
    await registry.emit(HookEvent.POST_REMEMBER, {"fiber_id": "..."})
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class HookEvent(StrEnum):
    """Events emitted during Surreal-Memory operations."""

    PRE_REMEMBER = "pre_remember"
    POST_REMEMBER = "post_remember"
    PRE_RECALL = "pre_recall"
    POST_RECALL = "post_recall"
    PRE_ENCODE = "pre_encode"
    POST_ENCODE = "post_encode"
    CONFLICT_DETECTED = "conflict_detected"
    PRE_CONSOLIDATE = "pre_consolidate"
    POST_CONSOLIDATE = "post_consolidate"
    MEMORY_EXPIRED = "memory_expired"


@dataclass(frozen=True)
class HookPayload:
    """Immutable payload delivered to hook listeners.

    Attributes:
        event: The event that triggered this payload
        data: Event-specific data dictionary
    """

    event: HookEvent
    data: dict[str, Any] = field(default_factory=dict)


HookListener = Callable[[HookPayload], Awaitable[None]]


class HookRegistry:
    """Central event bus for Surreal-Memory operations.

    Listeners are async callables that receive a HookPayload.
    Errors in listeners are logged but never propagate to callers,
    ensuring hooks cannot break core operations.
    """

    def __init__(self) -> None:
        self._listeners: dict[HookEvent, list[HookListener]] = {}

    def on(self, event: HookEvent, listener: HookListener) -> None:
        """Register a listener for an event."""
        self._listeners.setdefault(event, []).append(listener)

    def off(self, event: HookEvent, listener: HookListener) -> None:
        """Remove a listener for an event."""
        if event in self._listeners:
            self._listeners[event] = [
                existing for existing in self._listeners[event] if existing is not listener
            ]

    def has_listeners(self, event: HookEvent) -> bool:
        """Check if an event has any registered listeners."""
        return bool(self._listeners.get(event))

    def listener_count(self, event: HookEvent) -> int:
        """Return the number of listeners for an event."""
        return len(self._listeners.get(event, []))

    async def emit(self, event: HookEvent, data: dict[str, Any] | None = None) -> None:
        """Emit an event to all registered listeners.

        Listeners are called sequentially. Errors are caught and logged
        but never re-raised — hooks must not break core operations.
        """
        listeners = self._listeners.get(event)
        if not listeners:
            return

        payload = HookPayload(event=event, data=data or {})
        for listener in listeners:
            try:
                await listener(payload)
            except Exception:
                logger.error(
                    "Hook listener %s failed for %s",
                    getattr(listener, "__name__", repr(listener)),
                    event,
                    exc_info=True,
                )

    def clear(self, event: HookEvent | None = None) -> None:
        """Remove all listeners, or all listeners for a specific event."""
        if event is None:
            self._listeners.clear()
        elif event in self._listeners:
            del self._listeners[event]
