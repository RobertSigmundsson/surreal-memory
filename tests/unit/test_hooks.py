"""Tests for the generic event hook system."""

from __future__ import annotations

import pytest

from surreal_memory.engine.hooks import HookEvent, HookPayload, HookRegistry


@pytest.mark.asyncio
async def test_emit_fires_registered_listener() -> None:
    """Registered listener receives the correct payload."""
    registry = HookRegistry()
    received: list[HookPayload] = []

    async def listener(payload: HookPayload) -> None:
        received.append(payload)

    registry.on(HookEvent.POST_REMEMBER, listener)
    await registry.emit(HookEvent.POST_REMEMBER, {"fiber_id": "test-123"})

    assert len(received) == 1
    assert received[0].event == HookEvent.POST_REMEMBER
    assert received[0].data["fiber_id"] == "test-123"


@pytest.mark.asyncio
async def test_emit_no_listeners_is_noop() -> None:
    """Emitting an event with no listeners does nothing."""
    registry = HookRegistry()
    await registry.emit(HookEvent.PRE_RECALL, {"query": "test"})


@pytest.mark.asyncio
async def test_multiple_listeners_all_called() -> None:
    """All registered listeners for an event are called."""
    registry = HookRegistry()
    calls: list[str] = []

    async def listener_a(payload: HookPayload) -> None:
        calls.append("a")

    async def listener_b(payload: HookPayload) -> None:
        calls.append("b")

    registry.on(HookEvent.POST_REMEMBER, listener_a)
    registry.on(HookEvent.POST_REMEMBER, listener_b)
    await registry.emit(HookEvent.POST_REMEMBER)

    assert calls == ["a", "b"]


@pytest.mark.asyncio
async def test_listener_error_does_not_propagate() -> None:
    """A failing listener does not break emission or subsequent listeners."""
    registry = HookRegistry()
    calls: list[str] = []

    async def failing_listener(payload: HookPayload) -> None:
        raise ValueError("boom")

    async def good_listener(payload: HookPayload) -> None:
        calls.append("good")

    registry.on(HookEvent.POST_REMEMBER, failing_listener)
    registry.on(HookEvent.POST_REMEMBER, good_listener)

    await registry.emit(HookEvent.POST_REMEMBER)

    assert calls == ["good"]


@pytest.mark.asyncio
async def test_off_removes_listener() -> None:
    """off() removes a specific listener."""
    registry = HookRegistry()
    calls: list[str] = []

    async def listener(payload: HookPayload) -> None:
        calls.append("called")

    registry.on(HookEvent.PRE_RECALL, listener)
    registry.off(HookEvent.PRE_RECALL, listener)
    await registry.emit(HookEvent.PRE_RECALL)

    assert calls == []


@pytest.mark.asyncio
async def test_off_nonexistent_listener_is_safe() -> None:
    """off() with unregistered listener does not raise."""
    registry = HookRegistry()

    async def listener(payload: HookPayload) -> None:
        pass

    registry.off(HookEvent.PRE_RECALL, listener)


@pytest.mark.asyncio
async def test_different_events_are_isolated() -> None:
    """Listeners for one event are not triggered by another."""
    registry = HookRegistry()
    calls: list[str] = []

    async def recall_listener(payload: HookPayload) -> None:
        calls.append("recall")

    registry.on(HookEvent.POST_RECALL, recall_listener)
    await registry.emit(HookEvent.POST_REMEMBER)

    assert calls == []


@pytest.mark.asyncio
async def test_has_listeners() -> None:
    """has_listeners reports correctly."""
    registry = HookRegistry()

    assert not registry.has_listeners(HookEvent.POST_REMEMBER)

    async def listener(payload: HookPayload) -> None:
        pass

    registry.on(HookEvent.POST_REMEMBER, listener)
    assert registry.has_listeners(HookEvent.POST_REMEMBER)
    assert not registry.has_listeners(HookEvent.PRE_RECALL)


@pytest.mark.asyncio
async def test_listener_count() -> None:
    """listener_count returns correct count."""
    registry = HookRegistry()

    assert registry.listener_count(HookEvent.POST_REMEMBER) == 0

    async def listener_a(payload: HookPayload) -> None:
        pass

    async def listener_b(payload: HookPayload) -> None:
        pass

    registry.on(HookEvent.POST_REMEMBER, listener_a)
    registry.on(HookEvent.POST_REMEMBER, listener_b)
    assert registry.listener_count(HookEvent.POST_REMEMBER) == 2


@pytest.mark.asyncio
async def test_clear_all() -> None:
    """clear() removes all listeners."""
    registry = HookRegistry()

    async def listener(payload: HookPayload) -> None:
        pass

    registry.on(HookEvent.POST_REMEMBER, listener)
    registry.on(HookEvent.PRE_RECALL, listener)
    registry.clear()

    assert not registry.has_listeners(HookEvent.POST_REMEMBER)
    assert not registry.has_listeners(HookEvent.PRE_RECALL)


@pytest.mark.asyncio
async def test_clear_specific_event() -> None:
    """clear(event) removes only that event's listeners."""
    registry = HookRegistry()

    async def listener(payload: HookPayload) -> None:
        pass

    registry.on(HookEvent.POST_REMEMBER, listener)
    registry.on(HookEvent.PRE_RECALL, listener)
    registry.clear(HookEvent.POST_REMEMBER)

    assert not registry.has_listeners(HookEvent.POST_REMEMBER)
    assert registry.has_listeners(HookEvent.PRE_RECALL)


@pytest.mark.asyncio
async def test_payload_is_frozen() -> None:
    """HookPayload is immutable."""
    payload = HookPayload(event=HookEvent.POST_REMEMBER, data={"key": "value"})

    with pytest.raises(AttributeError):
        payload.event = HookEvent.PRE_RECALL  # type: ignore[misc]


@pytest.mark.asyncio
async def test_emit_with_none_data_defaults_to_empty_dict() -> None:
    """emit() with None data creates payload with empty dict."""
    registry = HookRegistry()
    received: list[HookPayload] = []

    async def listener(payload: HookPayload) -> None:
        received.append(payload)

    registry.on(HookEvent.PRE_ENCODE, listener)
    await registry.emit(HookEvent.PRE_ENCODE)

    assert len(received) == 1
    assert received[0].data == {}


def test_hook_event_values() -> None:
    """All expected hook events exist."""
    expected = {
        "pre_remember",
        "post_remember",
        "pre_recall",
        "post_recall",
        "pre_encode",
        "post_encode",
        "conflict_detected",
        "pre_consolidate",
        "post_consolidate",
        "memory_expired",
    }
    actual = {e.value for e in HookEvent}
    assert actual == expected
