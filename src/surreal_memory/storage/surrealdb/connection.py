"""SurrealDB connection settings and auth error types.

Single source of truth for:
- Default connection parameters (URL, user, password, namespace, database)
- Credential-error detection (distinct from 401 token-expiry errors)
- MCP env dict generation for client config backfill
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

DEFAULT_URL = "http://localhost:8001"
DEFAULT_USER = "root"
DEFAULT_PASS = "surrealmemory"  # noqa: S105  # nosec B105 — udokumentowany default dev; prod nadpisuje SURREALDB_PASS

AUTH_HINT = (
    "The MCP server env is likely empty or missing SURREALDB_PASS, so it fell back to the "
    "default password. Set SURREALDB_PASS (and SURREALDB_URL/USER) in your MCP client config "
    "(Claude Code: ~/.claude.json; Claude Desktop: claude_desktop_config.json) or run "
    "`smem doctor --fix`."
)
DEFAULT_NS = "surreal_memory"
DEFAULT_DB = "default"


# Minimum SurrealDB server version required (v2.6.0+): the synapse table is a
# native RELATION and internal GQL path search needs 3.2.0's ISO GQL. The
# synapse->RELATE auto-migration also gates on this.
MIN_SERVER_VERSION = (3, 2, 0)


def parse_server_version(raw: str) -> tuple[int, int, int] | None:
    """Parse a SurrealDB version string into a (major, minor, patch) tuple.

    Tolerates the ``surrealdb-`` prefix the HTTP ``/version`` endpoint / SDK
    ``version()`` return (e.g. ``surrealdb-3.2.0`` -> ``(3, 2, 0)``). Returns None
    when no ``X.Y.Z`` can be found, so callers can warn-and-continue rather than
    hard-fail on an unrecognised build string.
    """
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", str(raw))
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


@dataclass(frozen=True)
class SurrealSettings:
    """Immutable SurrealDB connection settings resolved from environment."""

    url: str
    user: str
    password: str
    namespace: str
    database: str

    @classmethod
    def from_env(cls) -> SurrealSettings:
        """Build settings from environment variables, falling back to defaults."""
        return cls(
            url=os.getenv("SURREALDB_URL", DEFAULT_URL),
            user=os.getenv("SURREALDB_USER", DEFAULT_USER),
            password=os.getenv("SURREALDB_PASS", DEFAULT_PASS),
            namespace=os.getenv("SURREALDB_NS", DEFAULT_NS),
            database=os.getenv("SURREALDB_DB", DEFAULT_DB),
        )


class StorageAuthError(Exception):
    """Raised when SurrealDB rejects credentials (wrong password / user).

    Distinct from token-expiry 401 errors handled by _is_auth_error in store.py.
    Carries an actionable hint so MCP clients surface the root cause instead of
    a generic -32000 failure.
    """

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} {self.hint}".strip()


class StorageVersionError(Exception):
    """Raised when the connected SurrealDB server is older than MIN_SERVER_VERSION.

    Carries an actionable upgrade hint (message + hint in English, no secrets) so
    the failure surfaces the root cause instead of a downstream schema/migration
    error. Only raised on a CONFIRMED old version — an unparsable/failed version
    probe warns and continues.
    """

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} {self.hint}".strip()


def is_credential_error(exc: Exception) -> bool:
    """True if *exc* signals bad credentials (wrong password / user not allowed).

    Distinct from _is_auth_error (token-expiry 401) in store.py — do NOT merge.
    Detects NotAllowedError from surrealdb SDK by three means (in order):
      1. isinstance check when the SDK is importable
      2. class-name match (works when import is not available)
      3. string content match as last resort

    Returns False for 401 token-expiry errors so the reconnect loop is not
    triggered on a bad password.
    """
    # Try real isinstance first — guard against MagicMock stubs in tests.
    try:
        from surrealdb.errors import NotAllowedError  # type: ignore[import-untyped,unused-ignore]

        if isinstance(NotAllowedError, type) and isinstance(exc, NotAllowedError):
            return True
    except (ImportError, ModuleNotFoundError, TypeError):
        pass

    # Class-name fallback (SDK installed but import path changed)
    if type(exc).__name__ == "NotAllowedError":
        return True

    # String-content fallback — match auth-specific phrases only.
    # "not allowed" alone is too broad (SurrealDB reuses it for permission errors
    # on tables/operations); require an auth-context indicator alongside it.
    msg = str(exc).lower()
    if "problem with authentication" in msg:
        return True
    if "not allowed" in msg and any(
        kw in msg for kw in ("signin", "authenticate", "credentials", "login", "password")
    ):
        return True

    return False


def build_mcp_env() -> dict[str, str]:
    """Return an env dict suitable for embedding in MCP client configs.

    Reads actual env / .env values so a developer's local overrides are
    preserved; falls back to defaults so clean installs get a working config.
    """
    s = SurrealSettings.from_env()
    return {
        "SURREAL_MEMORY_STORAGE": "surrealdb",
        "SURREALDB_URL": s.url,
        "SURREALDB_USER": s.user,
        "SURREALDB_PASS": s.password,
        "SURREALDB_NS": s.namespace,
        "SURREALDB_DB": s.database,
    }
