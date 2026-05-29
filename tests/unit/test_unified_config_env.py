"""UnifiedConfig.load must honor SURREAL_MEMORY_* env on a fresh install.

Regression coverage for the "active brain reports 0/F" bug: when no
config.toml exists yet, load() built a default config that silently ignored
SURREAL_MEMORY_STORAGE (defaulting to sqlite) and SURREAL_MEMORY_BRAIN. A
server process that started before config.toml existed therefore cached a
sqlite-backed singleton and read an empty store, while data-plane processes
(which had a config.toml) wrote to SurrealDB. The env must win in both
branches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from surreal_memory.unified_config import UnifiedConfig

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_load_without_config_file_honors_storage_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")
    cfg = UnifiedConfig.load(config_path=tmp_path / "config.toml")
    assert cfg.storage_backend == "surrealdb"


def test_load_without_config_file_honors_brain_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_BRAIN", "l260639")
    cfg = UnifiedConfig.load(config_path=tmp_path / "config.toml")
    assert cfg.current_brain == "l260639"
