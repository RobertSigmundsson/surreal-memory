"""Tests for embedding-pipeline hardening.

Covers the handover fixes: transient-error retry/backoff, the dead-model
fallback guard in semantic discovery, and the embedding capability probe.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from surreal_memory.engine.embedding import capability as capability_mod
from surreal_memory.engine.embedding.capability import probe_embedding_capability
from surreal_memory.engine.embedding.retry import call_with_retry, is_transient


class _APIError(Exception):
    """Fake provider error carrying a numeric status code, like google-genai."""

    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"api error {code}")


async def _instant_sleep(_delay: float) -> None:
    """Drop-in for asyncio.sleep so retry tests don't actually wait."""
    return None


# ── is_transient ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_is_transient_true_for_transient_status(code: int) -> None:
    assert is_transient(_APIError(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_is_transient_false_for_client_errors(code: int) -> None:
    assert is_transient(_APIError(code)) is False


def test_is_transient_message_markers() -> None:
    assert is_transient(Exception("RESOURCE_EXHAUSTED: rate limit hit")) is True
    assert is_transient(Exception("connection reset by peer")) is True
    assert is_transient(Exception("invalid argument")) is False


# ── call_with_retry ─────────────────────────────────────────────────────────


async def test_retry_returns_on_first_success() -> None:
    calls = {"n": 0}

    async def factory() -> str:
        calls["n"] += 1
        return "ok"

    assert await call_with_retry(factory) == "ok"
    assert calls["n"] == 1


async def test_retry_succeeds_after_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    attempts = {"n": 0}

    async def factory() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _APIError(503)
        return "recovered"

    assert await call_with_retry(factory) == "recovered"
    assert attempts["n"] == 3


async def test_retry_reraises_non_transient_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    attempts = {"n": 0}

    async def factory() -> str:
        attempts["n"] += 1
        raise _APIError(400)

    with pytest.raises(_APIError):
        await call_with_retry(factory)
    assert attempts["n"] == 1  # non-transient → no retries


async def test_retry_exhausts_and_reraises_last(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    attempts = {"n": 0}

    async def factory() -> str:
        attempts["n"] += 1
        raise _APIError(429)

    with pytest.raises(_APIError):
        await call_with_retry(factory)
    assert attempts["n"] == 4  # _MAX_ATTEMPTS


# ── _auto_detect_provider never returns a decommissioned model ───────────────


def test_auto_detect_never_returns_decommissioned_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def _no_ollama(*_args: object, **_kwargs: object) -> object:
        raise ConnectionError("no ollama")

    monkeypatch.setattr(httpx, "get", _no_ollama)
    # Make `import sentence_transformers` raise ImportError so gemini is chosen.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    from surreal_memory.engine.embedding.gemini_embedding import _DEFAULT_MODEL
    from surreal_memory.engine.semantic_discovery import _auto_detect_provider

    provider, model = _auto_detect_provider()
    assert provider == "gemini"
    assert model == _DEFAULT_MODEL == "gemini-embedding-001"
    assert "text-embedding-004" not in model


# ── capability probe ─────────────────────────────────────────────────────────


def test_capability_disabled_reports_unavailable() -> None:
    cfg = SimpleNamespace(
        embedding_enabled=False, embedding_provider="gemini", embedding_model="m"
    )
    info = probe_embedding_capability(cfg)
    assert info["enabled"] is False
    assert info["available"] is False


def test_capability_missing_package_has_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capability_mod.importlib.util, "find_spec", lambda _name: None)
    cfg = SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="gemini",
        embedding_model="gemini-embedding-001",
    )
    info = probe_embedding_capability(cfg)
    assert info["available"] is False
    assert "embeddings-gemini" in info["detail"]


def test_capability_available_reports_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capability_mod.importlib.util, "find_spec", lambda _name: object()
    )
    cfg = SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="gemini",
        embedding_model="gemini-embedding-001",
    )
    info = probe_embedding_capability(cfg)
    assert info["available"] is True
    assert info["dimension"] == 3072


def test_capability_accepts_nested_unified_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capability_mod.importlib.util, "find_spec", lambda _name: object()
    )
    cfg = SimpleNamespace(
        embedding=SimpleNamespace(
            enabled=True, provider="gemini", model="gemini-embedding-001"
        )
    )
    info = probe_embedding_capability(cfg)
    assert info["enabled"] is True
    assert info["available"] is True
