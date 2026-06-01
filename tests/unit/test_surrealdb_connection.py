"""Tests for surreal_memory.storage.surrealdb.connection module."""

from __future__ import annotations

import pytest


class TestSurrealSettings:
    def test_from_env_empty_returns_defaults(self, monkeypatch):
        """Empty env → all defaults; password must be 'surrealmemory', not 'root'."""
        from surreal_memory.storage.surrealdb.connection import SurrealSettings

        monkeypatch.delenv("SURREALDB_URL", raising=False)
        monkeypatch.delenv("SURREALDB_USER", raising=False)
        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        monkeypatch.delenv("SURREALDB_NS", raising=False)
        monkeypatch.delenv("SURREALDB_DB", raising=False)

        s = SurrealSettings.from_env()
        assert s.url == "http://localhost:8001"
        assert s.user == "root"
        assert s.password == "surrealmemory"  # noqa: S105
        assert s.namespace == "surreal_memory"
        assert s.database == "default"

    def test_from_env_reads_env_vars(self, monkeypatch):
        from surreal_memory.storage.surrealdb.connection import SurrealSettings

        monkeypatch.setenv("SURREALDB_URL", "http://custom:9999")
        monkeypatch.setenv("SURREALDB_USER", "myuser")
        monkeypatch.setenv("SURREALDB_PASS", "mypass")
        monkeypatch.setenv("SURREALDB_NS", "mynamespace")
        monkeypatch.setenv("SURREALDB_DB", "mydb")

        s = SurrealSettings.from_env()
        assert s.url == "http://custom:9999"
        assert s.user == "myuser"
        assert s.password == "mypass"  # noqa: S105
        assert s.namespace == "mynamespace"
        assert s.database == "mydb"

    def test_settings_immutable(self, monkeypatch):
        from surreal_memory.storage.surrealdb.connection import SurrealSettings

        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        s = SurrealSettings.from_env()
        with pytest.raises((AttributeError, TypeError)):
            s.password = "changed"  # type: ignore[misc]  # noqa: S105


class TestStorageAuthError:
    def test_carries_hint(self):
        from surreal_memory.storage.surrealdb.connection import StorageAuthError

        err = StorageAuthError("auth failed", hint="set SURREALDB_PASS")
        assert err.hint == "set SURREALDB_PASS"

    def test_str_includes_message_and_hint(self):
        from surreal_memory.storage.surrealdb.connection import StorageAuthError

        err = StorageAuthError("auth failed", hint="fix this")
        assert "auth failed" in str(err)
        assert "fix this" in str(err)

    def test_str_without_hint(self):
        from surreal_memory.storage.surrealdb.connection import StorageAuthError

        err = StorageAuthError("auth failed")
        assert "auth failed" in str(err)
        assert err.hint == ""

    def test_is_exception(self):
        from surreal_memory.storage.surrealdb.connection import StorageAuthError

        with pytest.raises(StorageAuthError):
            raise StorageAuthError("test")


class TestIsCredentialError:
    def test_true_for_class_named_not_allowed_error(self):
        """Object whose class is named NotAllowedError → True."""
        from surreal_memory.storage.surrealdb.connection import is_credential_error

        class NotAllowedError(Exception):
            pass

        assert is_credential_error(NotAllowedError("some problem")) is True

    def test_true_for_problem_with_authentication_string(self):
        from surreal_memory.storage.surrealdb.connection import is_credential_error

        assert is_credential_error(Exception("There was a problem with authentication")) is True

    def test_true_for_not_allowed_with_auth_context(self):
        from surreal_memory.storage.surrealdb.connection import is_credential_error

        # "not allowed" + auth keyword → credential error
        assert is_credential_error(Exception("signin not allowed for this user")) is True
        assert is_credential_error(Exception("credentials not allowed")) is True

    def test_false_for_not_allowed_without_auth_context(self):
        """Pure permission/schema error — 'not allowed' without auth keyword must NOT match."""
        from surreal_memory.storage.surrealdb.connection import is_credential_error

        assert is_credential_error(Exception("Operation not allowed on this table")) is False
        assert is_credential_error(Exception("SELECT not allowed on namespace::table")) is False

    def test_false_for_plain_value_error(self):
        from surreal_memory.storage.surrealdb.connection import is_credential_error

        assert is_credential_error(ValueError("bad value")) is False

    def test_false_for_401_token_error(self):
        """401 token-expiry errors are handled by _is_auth_error; is_credential_error must not match them."""
        from surreal_memory.storage.surrealdb.connection import is_credential_error

        class TokenExpiredError(Exception):
            status = 401

        assert is_credential_error(TokenExpiredError("token expired")) is False

    def test_false_for_runtime_error(self):
        from surreal_memory.storage.surrealdb.connection import is_credential_error

        assert is_credential_error(RuntimeError("connection refused")) is False


class TestBuildMcpEnv:
    def test_returns_required_keys(self, monkeypatch):
        from surreal_memory.storage.surrealdb.connection import build_mcp_env

        monkeypatch.delenv("SURREALDB_URL", raising=False)
        monkeypatch.delenv("SURREALDB_USER", raising=False)
        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        monkeypatch.delenv("SURREALDB_NS", raising=False)
        monkeypatch.delenv("SURREALDB_DB", raising=False)

        env = build_mcp_env()
        assert "SURREAL_MEMORY_STORAGE" in env
        assert env["SURREAL_MEMORY_STORAGE"] == "surrealdb"
        assert "SURREALDB_URL" in env
        assert "SURREALDB_USER" in env
        assert "SURREALDB_PASS" in env
        assert "SURREALDB_NS" in env
        assert "SURREALDB_DB" in env

    def test_defaults_password_is_surrealmemory(self, monkeypatch):
        from surreal_memory.storage.surrealdb.connection import build_mcp_env

        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        env = build_mcp_env()
        assert env["SURREALDB_PASS"] == "surrealmemory"  # noqa: S105

    def test_respects_env_override(self, monkeypatch):
        from surreal_memory.storage.surrealdb.connection import build_mcp_env

        monkeypatch.setenv("SURREALDB_PASS", "custompass")
        env = build_mcp_env()
        assert env["SURREALDB_PASS"] == "custompass"  # noqa: S105

    def test_all_values_are_strings(self, monkeypatch):
        from surreal_memory.storage.surrealdb.connection import build_mcp_env

        monkeypatch.delenv("SURREALDB_URL", raising=False)
        env = build_mcp_env()
        for k, v in env.items():
            assert isinstance(v, str), f"Key {k!r} has non-string value {v!r}"
