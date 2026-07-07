"""Regression tests for cloud-sync config persistence (RUN-004).

Pins the three defects fixed in RUN-004 so the panel "Cloud Sync: not
configured" bug cannot regress:

  1. Dead env vars: SyncConfig now honors SURREAL_MEMORY_HUB_URL / _API_KEY /
     _SYNC_ENABLED / _SYNC_AUTO via ``_load_sync_settings`` (was TOML-only, so a
     hub configured only via compose/MCP env was invisible to Overview).
  2. Stale singleton: a caller that ``save()``s must also ``set_config()`` so
     ``get_config()`` in the SAME process serves fresh values without a restart.
  3. On-disk persistence survives a reload.

No mocks: real ``save()``/``load()``/``get_config()``/``set_config()`` against an
isolated ``SURREAL_MEMORY_DIR`` (tmp). Sync config is a config.toml + in-process
singleton concern, independent of the storage backend, so it is tested at the
config layer (the storage-backed stories are covered by the live-DB suite).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from surreal_memory.unified_config import (
    SyncConfig,
    _load_sync_settings,
    get_config,
    set_config,
)

_SYNC_ENV_VARS = (
    "SURREAL_MEMORY_HUB_URL",
    "SURREAL_MEMORY_API_KEY",
    "SURREAL_MEMORY_SYNC_ENABLED",
    "SURREAL_MEMORY_SYNC_AUTO",
)


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point config dir + singleton at a throwaway tmp dir so the real
    ``~/.surrealmemory/config.toml`` is never touched, and clear sync env vars
    so an ambient hub config does not leak into the assertions."""
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    for var in _SYNC_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Prime the module singleton bound to the tmp dir.
    set_config(get_config(reload=True))
    return tmp_path


class TestSyncEnvOverride:
    """Defect 1: sync config must honor environment variables."""

    def test_hub_url_from_env_when_toml_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_HUB_URL", "https://hub.example.workers.dev")
        assert _load_sync_settings({}).hub_url == "https://hub.example.workers.dev"

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_API_KEY", "nmk_env_key_123")
        assert _load_sync_settings({}).api_key == "nmk_env_key_123"

    def test_env_wins_over_toml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_HUB_URL", "https://env.example.workers.dev")
        cfg = _load_sync_settings({"hub_url": "https://toml.example.workers.dev"})
        assert cfg.hub_url == "https://env.example.workers.dev"

    def test_toml_used_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SURREAL_MEMORY_HUB_URL", raising=False)
        cfg = _load_sync_settings({"hub_url": "https://toml.example.workers.dev"})
        assert cfg.hub_url == "https://toml.example.workers.dev"

    def test_sync_enabled_truthy_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_SYNC_ENABLED", "true")
        assert _load_sync_settings({}).enabled is True

    def test_invalid_env_hub_url_sanitized_to_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # SyncConfig.from_dict rejects non-http(s) hub_url — env values are
        # re-validated through it, so a bad env value becomes "".
        monkeypatch.setenv("SURREAL_MEMORY_HUB_URL", "ftp://bad-scheme")
        assert _load_sync_settings({}).hub_url == ""


class TestSyncSingletonAndPersistence:
    """Defects 2 & 3: same-process freshness after save()+set_config(), and
    on-disk persistence across a reload."""

    def test_set_config_makes_get_config_fresh_in_process(self, isolated_config: Path) -> None:
        cfg = get_config()
        assert cfg.sync.hub_url == ""  # not configured initially

        new_sync = SyncConfig.from_dict(
            {
                "hub_url": "https://hub.example.workers.dev",
                "api_key": "nmk_persist_123",
                "enabled": True,
            }
        )
        updated = dataclasses.replace(cfg, sync=new_sync)
        updated.save()
        set_config(updated)  # the fix — without this, get_config() stayed stale

        # SAME process, no reload: must serve the fresh value.
        assert get_config().sync.hub_url == "https://hub.example.workers.dev"
        assert get_config().sync.api_key == "nmk_persist_123"
        # Overview's "configured" predicate is bool(sync.hub_url):
        assert bool(get_config().sync.hub_url) is True

    def test_persists_across_reload(self, isolated_config: Path) -> None:
        cfg = get_config()
        new_sync = SyncConfig.from_dict(
            {"hub_url": "https://hub.example.workers.dev", "api_key": "nmk_reload_123"}
        )
        dataclasses.replace(cfg, sync=new_sync).save()

        reloaded = get_config(reload=True)  # fresh read from disk
        assert reloaded.sync.hub_url == "https://hub.example.workers.dev"
        assert reloaded.sync.api_key == "nmk_reload_123"
