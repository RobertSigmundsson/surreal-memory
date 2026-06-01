"""Tests for SurrealDBStorage auth fail-fast and default handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestInitializeAuthFailFast:
    """signin on bad credentials → StorageAuthError, not raw NotAllowedError."""

    @pytest.mark.asyncio
    async def test_signin_credential_error_raises_storage_auth_error(self):
        from surreal_memory.storage.surrealdb.connection import StorageAuthError
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        class FakeNotAllowedError(Exception):
            pass

        mock_conn = AsyncMock()
        mock_conn.signin.side_effect = FakeNotAllowedError(
            "There was a problem with authentication"
        )

        storage = SurrealDBStorage(url="http://localhost:8001", password="wrongpass")  # noqa: S106

        with patch("surrealdb.AsyncSurreal", return_value=mock_conn):
            with pytest.raises(StorageAuthError) as exc_info:
                await storage.initialize()

        err = exc_info.value
        assert "wrongpass" not in str(err), "Password must not appear in error message"
        assert err.hint != "", "hint must be non-empty"
        assert "SURREALDB_PASS" in err.hint

    @pytest.mark.asyncio
    async def test_signin_credential_error_includes_user_and_url(self):
        from surreal_memory.storage.surrealdb.connection import StorageAuthError
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        class NotAllowedError(Exception):  # name triggers class-name fallback
            pass

        mock_conn = AsyncMock()
        mock_conn.signin.side_effect = NotAllowedError("not allowed")

        storage = SurrealDBStorage(url="http://myhost:8001", user="myuser", password="bad")  # noqa: S106

        with patch("surrealdb.AsyncSurreal", return_value=mock_conn):
            with pytest.raises(StorageAuthError) as exc_info:
                await storage.initialize()

        msg = str(exc_info.value)
        assert "myuser" in msg
        assert "myhost" in msg

    @pytest.mark.asyncio
    async def test_non_credential_exception_propagates_unchanged(self):
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        mock_conn = AsyncMock()
        mock_conn.signin.side_effect = ConnectionRefusedError("connection refused")

        storage = SurrealDBStorage()

        with patch("surrealdb.AsyncSurreal", return_value=mock_conn):
            with pytest.raises(ConnectionRefusedError):
                await storage.initialize()

    @pytest.mark.asyncio
    async def test_initialize_success_does_not_raise(self):
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        mock_conn = AsyncMock()
        mock_conn.signin.return_value = None
        mock_conn.use.return_value = None

        storage = SurrealDBStorage()

        with (
            patch("surrealdb.AsyncSurreal", return_value=mock_conn),
            patch("surreal_memory.storage.surrealdb.store.ensure_schema", new_callable=AsyncMock),
        ):
            await storage.initialize()  # must not raise


class TestReconnectAuthFailFast:
    """_reconnect on bad credentials → StorageAuthError (not a loop)."""

    @pytest.mark.asyncio
    async def test_reconnect_credential_error_raises_storage_auth_error(self):
        from surreal_memory.storage.surrealdb.connection import StorageAuthError
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        class NotAllowedError(Exception):  # name triggers class-name fallback
            pass

        mock_conn = AsyncMock()
        mock_conn.signin.side_effect = NotAllowedError("not allowed")

        storage = SurrealDBStorage()

        with patch("surrealdb.AsyncSurreal", return_value=mock_conn):
            with pytest.raises(StorageAuthError):
                await storage._reconnect()


class TestDefaultPasswordDry:
    """Default password comes from connection.py (surrealmemory), not 'root'."""

    def test_default_password_is_surrealmemory(self, monkeypatch):
        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        s = SurrealDBStorage()
        assert s._password == "surrealmemory"  # noqa: S105

    def test_explicit_password_overrides_default(self, monkeypatch):
        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        s = SurrealDBStorage(password="explicit")  # noqa: S106
        assert s._password == "explicit"  # noqa: S105

    def test_env_password_overrides_default(self, monkeypatch):
        monkeypatch.setenv("SURREALDB_PASS", "envpass")
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        s = SurrealDBStorage()
        assert s._password == "envpass"  # noqa: S105
