"""Tests for the autouse aiosqlite leak guard in tests/conftest.py.

Leaked (never-closed) aiosqlite connections keep NON-daemon worker threads
blocked on their queue forever; any such thread still referenced at interpreter
exit wedges pytest in ``threading._shutdown`` after a green summary. The guard
closes stragglers at each test's teardown.

The two tests below cooperate and rely on within-file definition order (which
pytest preserves): the first deliberately leaks a connection, the second
asserts the guard stopped its worker thread during teardown.
"""

from __future__ import annotations

from typing import Any

import aiosqlite

_handoff: dict[str, Any] = {}


async def test_leak_a_connection_on_purpose() -> None:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("CREATE TABLE t (x INTEGER)")  # prove the worker is live
    thread = conn._thread
    assert thread.is_alive()
    _handoff["thread"] = thread
    # Deliberately NOT closed — the autouse guard must stop it at teardown.


async def test_the_guard_stopped_the_leaked_worker_thread() -> None:
    thread = _handoff.pop("thread")
    thread.join(timeout=2.0)
    assert not thread.is_alive(), (
        "the leak guard should have stopped the worker thread leaked by the previous test"
    )


async def test_a_properly_closed_connection_is_untouched() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        cursor = await conn.execute("SELECT 1")
        row = await cursor.fetchone()
        assert row == (1,)
    thread = conn._thread
    thread.join(timeout=2.0)
    assert not thread.is_alive()
