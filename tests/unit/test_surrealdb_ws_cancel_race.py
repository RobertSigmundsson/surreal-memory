"""The SDK's pending-request cleanup turns a cancelled query into ``KeyError``.

Measured on the UserPromptSubmit hook (``~/.surrealmemory/prompt_recall_slad_bledy.jsonl``):
38 rows ``status=wyjatek powod=recall-KeyError`` between 2026-10-03 and 2026-11-03, every one
20.6-24.9 s after the prompt — i.e. AFTER the hook's 20 s cap, never before it. A real
SurrealDB 3.2.3 + surrealdb 2.0.0 reproduce it 10/10 whenever the event loop is busy past
the deadline while a query is in flight (and 0/300 without the busy stretch).

Mechanism (surrealdb 2.0.0, ``connections/async_ws.py``):

1. the cap cancels the waiter's future; the response for it was already buffered;
2. the receive loop wakes first, calls ``fut.set_result`` on the cancelled future ->
   ``InvalidStateError`` -> its ``finally`` cancels every pending future and CLEARS ``qry``;
3. the waiter's own ``finally: del self.qry[query_id]`` hits a missing key -> ``KeyError``
   replaces the ``CancelledError`` it was woken with, so ``asyncio.timeout`` never converts
   it into ``TimeoutError``;
4. the socket reference stays set, so ``connect()`` never rebuilds the dead receiver and
   every later request on that connection waits forever.

These tests drive the REAL SDK class over an in-memory socket, with the loop ordering of
production (the buffered read is serviced before the overdue timer's waiter wakes).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

async_ws = pytest.importorskip("surrealdb.connections.async_ws")
sdk_cbor = pytest.importorskip("surrealdb.data.cbor")

from tests.unit.test_consolidation_timeouts_and_reconnect import (  # noqa: E402
    _ConnFactory,
    _store_with,
    _StubConn,
)


class _InMemorySocket:
    """What the SDK expects from a websockets connection: ``send``, async iteration, ``close``."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._inbox: asyncio.Queue[bytes] = asyncio.Queue()
        self.closed = False

    async def send(self, data: bytes) -> None:
        self.sent.append(sdk_cbor.decode(data))

    def deliver(self, frame: bytes) -> None:
        self._inbox.put_nowait(frame)

    def __aiter__(self) -> _InMemorySocket:
        return self

    async def __anext__(self) -> bytes:
        return await self._inbox.get()

    async def close(self) -> None:
        self.closed = True


def _query_ok_frame(request_id: str, rows: list[dict[str, Any]]) -> bytes:
    return sdk_cbor.encode(
        {"id": request_id, "result": [{"status": "OK", "time": "1ms", "result": rows}]}
    )


def _sdk_connection() -> tuple[Any, _InMemorySocket]:
    """A real ``AsyncWsSurrealConnection`` already 'connected' to an in-memory socket."""
    conn = async_ws.AsyncWsSurrealConnection("ws://localhost:1")
    sock = _InMemorySocket()
    conn.socket = sock
    conn.loop = asyncio.get_running_loop()
    conn.recv_task = asyncio.create_task(conn._recv_task())
    return conn, sock


async def _until(predicate: Any, *, limit: int = 200) -> None:
    for _ in range(limit):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never became true")


def _answer_then_stall(
    sock: _InMemorySocket, request_id: str, *, after: float, stall: float
) -> None:
    """The response lands `after` s from now, but the loop is busy for `stall` s (> the cap).

    Busy = a synchronous stretch (an urllib call, CPU-bound decoding) — exactly what made the
    production timer fire 0.6-4.9 s late.
    """
    loop = asyncio.get_running_loop()
    loop.call_at(loop.time() + after, sock.deliver, _query_ok_frame(request_id, [{"late": True}]))
    time.sleep(stall)


class TestSdkMechanism:
    """Characterises the upstream defect itself; stays green until the SDK is fixed."""

    @pytest.mark.asyncio
    async def test_a_timed_out_query_surfaces_as_keyerror_not_timeout(self) -> None:
        conn, sock = _sdk_connection()
        task = asyncio.create_task(asyncio.wait_for(conn.query("SELECT 1"), timeout=0.05))
        await _until(lambda: sock.sent)

        _answer_then_stall(sock, sock.sent[0]["id"], after=0.01, stall=0.1)

        with pytest.raises(KeyError):
            await task
        assert conn.recv_task.done(), "the receive loop died on the cancelled future"
        assert conn.socket is not None, (
            "...but the socket reference survives, so nothing rebuilds it"
        )


class TestStoreRestoresTheRealOutcome:
    @pytest.mark.asyncio
    async def test_the_hook_cap_is_reported_as_timeout(self, monkeypatch: Any) -> None:
        """The UserPromptSubmit cap must read as `timeout`, not as `recall-KeyError`."""
        conn, sock = _sdk_connection()
        store = _store_with(conn)
        monkeypatch.setattr("surrealdb.AsyncSurreal", _ConnFactory(alive=True), raising=False)
        task = asyncio.create_task(asyncio.wait_for(store._query("SELECT 1"), timeout=0.05))
        await _until(lambda: sock.sent)

        _answer_then_stall(sock, sock.sent[0]["id"], after=0.01, stall=0.1)

        with pytest.raises(TimeoutError):
            await task

    @pytest.mark.asyncio
    async def test_an_innocent_sibling_is_retried_on_a_fresh_connection(
        self, monkeypatch: Any
    ) -> None:
        """A request nobody cancelled must not fail because a sibling's cap killed the receiver."""
        conn, sock = _sdk_connection()
        store = _store_with(conn)
        factory = _ConnFactory(alive=True)
        monkeypatch.setattr("surrealdb.AsyncSurreal", factory, raising=False)

        capped = asyncio.create_task(asyncio.wait_for(store._query("SELECT 1"), timeout=0.05))
        sibling = asyncio.create_task(store._query("SELECT 2"))
        await _until(lambda: len(sock.sent) == 2)
        capped_id = next(m["id"] for m in sock.sent if "SELECT 1" in str(m))

        _answer_then_stall(sock, capped_id, after=0.01, stall=0.1)

        with pytest.raises(TimeoutError):
            await capped
        assert await asyncio.wait_for(sibling, timeout=5) == [{"ok": True}]
        assert factory.calls == 1, "exactly one rebuild of the dead connection"

    @pytest.mark.asyncio
    async def test_a_connection_whose_receiver_died_is_rebuilt_not_waited_on(
        self, monkeypatch: Any
    ) -> None:
        """Without a receiver every request waits forever; the next query must rebuild instead."""
        conn, _sock = _sdk_connection()
        conn.recv_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await conn.recv_task
        assert conn.socket is not None
        store = _store_with(conn)
        factory = _ConnFactory(alive=True)
        monkeypatch.setattr("surrealdb.AsyncSurreal", factory, raising=False)

        rows = await asyncio.wait_for(store._query("SELECT 1"), timeout=5)

        assert rows == [{"ok": True}]
        assert factory.calls == 1

    @pytest.mark.asyncio
    async def test_a_keyerror_from_anywhere_else_is_not_masked(self, monkeypatch: Any) -> None:
        """Only the SDK's `_send` cleanup is translated; a real KeyError stays a KeyError."""

        class _KeyErrorConn(_StubConn):
            async def query(self, _sql: str, _params: Any) -> list[Any]:
                await asyncio.sleep(0)
                raise KeyError("result")

        store = _store_with(_KeyErrorConn(alive=True))
        factory = _ConnFactory(alive=True)
        monkeypatch.setattr("surrealdb.AsyncSurreal", factory, raising=False)

        with pytest.raises(KeyError):
            await store._query("SELECT 1")
        assert factory.calls == 0, "no reconnect for an error that is not about the transport"

    @pytest.mark.asyncio
    async def test_a_healthy_sdk_connection_answers_without_a_rebuild(
        self, monkeypatch: Any
    ) -> None:
        """Positive control: the pre-flight check must not fire on a live receiver."""
        conn, sock = _sdk_connection()
        store = _store_with(conn)
        factory = _ConnFactory(alive=True)
        monkeypatch.setattr("surrealdb.AsyncSurreal", factory, raising=False)
        task = asyncio.create_task(store._query("SELECT 1"))
        await _until(lambda: sock.sent)

        sock.deliver(_query_ok_frame(sock.sent[0]["id"], [{"live": True}]))

        assert await asyncio.wait_for(task, timeout=5) == [{"live": True}]
        assert factory.calls == 0
