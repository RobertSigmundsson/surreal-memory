"""Embedding capability probe — turns the silent "embeddings configured but the
provider package is missing" failure into a visible, actionable state.

Used by ``smem_health`` (to report ``embedding`` availability) and at MCP
startup (to log a loud, actionable warning). The probe is cheap: it checks
package availability via ``importlib.util.find_spec`` and never imports heavy
models or makes API calls.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

logger = logging.getLogger(__name__)

# provider name -> (import module to check, pip extra that provides it)
_PROVIDER_IMPORT: dict[str, tuple[str, str]] = {
    "gemini": ("google.genai", "embeddings-gemini"),
    "openai": ("openai", "embeddings-openai"),
    "openrouter": ("openai", "embeddings-openrouter"),
    "sentence_transformer": ("sentence_transformers", "embeddings"),
}


def _dimension_for(provider: str, model: str) -> int | None:
    """Best-effort dimension lookup without loading models or calling APIs."""
    try:
        if provider == "gemini":
            from surreal_memory.engine.embedding.gemini_embedding import _MODEL_DIMENSIONS

            return _MODEL_DIMENSIONS.get(model, 3072)
        if provider == "openai":
            from surreal_memory.engine.embedding.openai_embedding import _MODEL_DIMENSIONS

            return _MODEL_DIMENSIONS.get(model, 1536)
        if provider == "openrouter":
            from surreal_memory.engine.embedding.openrouter_embedding import _MODEL_DIMENSIONS

            return _MODEL_DIMENSIONS.get(model, 1536)
    except Exception:
        return None
    return None


def probe_embedding_capability(config: Any) -> dict[str, Any]:
    """Report whether the configured embedding provider is usable.

    Returns a dict with: ``enabled``, ``provider``, ``model``, ``available``
    (bool) and ``detail`` (a human-readable note, including the exact install
    command when a package is missing). Never raises.
    """
    # Accept either a flat BrainConfig (embedding_enabled/provider/model) or a
    # nested unified config (config.embedding.enabled/provider/model).
    nested = getattr(config, "embedding", None)
    enabled = getattr(config, "embedding_enabled", None)
    provider = getattr(config, "embedding_provider", None)
    model = getattr(config, "embedding_model", None)
    if nested is not None and not isinstance(nested, (str, bool)):
        if enabled is None:
            enabled = getattr(nested, "enabled", False)
        if provider is None:
            provider = getattr(nested, "provider", "")
        if model is None:
            model = getattr(nested, "model", "")
    enabled = bool(enabled)
    provider = (provider or "").strip()
    model = model or ""
    result: dict[str, Any] = {
        "enabled": enabled,
        "provider": provider,
        "model": model,
        "available": False,
        "dimension": None,
        "detail": None,
    }

    if not enabled:
        result["detail"] = "embeddings disabled"
        return result

    if provider == "ollama":
        # Local server — availability depends on the daemon, not a Python package.
        result["available"] = True
        result["detail"] = "ollama (local server — ensure it is running)"
        return result

    if provider == "auto":
        # Provider is detected at runtime from whatever is installed/keyed.
        result["available"] = True
        result["detail"] = "auto-detect at runtime"
        return result

    entry = _PROVIDER_IMPORT.get(provider)
    if entry is None:
        result["detail"] = f"unknown embedding provider {provider!r}"
        return result

    module_name, extra = entry
    if importlib.util.find_spec(module_name) is None:
        result["detail"] = (
            f"{provider} provider configured but '{module_name}' is not installed — "
            f"install it with: pip install 'surreal-memory[{extra}]'"
        )
        return result

    result["available"] = True
    result["dimension"] = _dimension_for(provider, model)
    return result


def warn_if_embedding_unavailable(config: Any) -> None:
    """Log a loud, actionable warning at startup if embeddings are configured
    but the provider package is missing. Recall/remember still fail-soft to
    keyword mode — this just makes the cause visible instead of silent."""
    info = probe_embedding_capability(config)
    if info["enabled"] and not info["available"]:
        logger.warning(
            "Embedding provider unavailable — running in degraded keyword mode. %s",
            info["detail"],
        )
