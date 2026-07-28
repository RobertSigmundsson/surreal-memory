"""Tests for CLI brain-name resolution priority (explicit arg > env var > config).

Regression coverage for the bug where ``brain_name = brain or config.current_brain``
at CLI command call sites always produced a non-None name, causing ``get_storage``
to skip its own env-var check and silently ignore ``SURREAL_MEMORY_BRAIN``.
"""

from __future__ import annotations

from surreal_memory.cli._helpers import resolve_brain
from surreal_memory.cli.config import CLIConfig


def _config(current_brain: str = "default") -> CLIConfig:
    return CLIConfig(current_brain=current_brain)


class TestResolveBrain:
    def test_explicit_arg_wins_over_everything(self, monkeypatch) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_BRAIN", "env-brain")
        config = _config("config-brain")

        assert resolve_brain("explicit-brain", config) == "explicit-brain"

    def test_env_var_wins_when_no_explicit_arg(self, monkeypatch) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_BRAIN", "env-brain")
        config = _config("config-brain")

        assert resolve_brain(None, config) == "env-brain"

    def test_config_current_brain_used_when_no_explicit_arg_or_env(self, monkeypatch) -> None:
        monkeypatch.delenv("SURREAL_MEMORY_BRAIN", raising=False)
        config = _config("config-brain")

        assert resolve_brain(None, config) == "config-brain"

    def test_empty_env_var_treated_as_unset(self, monkeypatch) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_BRAIN", "")
        config = _config("config-brain")

        assert resolve_brain(None, config) == "config-brain"

    def test_default_param_used_over_config_when_no_explicit_arg_or_env(self, monkeypatch) -> None:
        monkeypatch.delenv("SURREAL_MEMORY_BRAIN", raising=False)
        config = _config("config-brain")

        assert resolve_brain(None, config, default="imported-brain") == "imported-brain"

    def test_env_var_wins_over_default_param(self, monkeypatch) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_BRAIN", "env-brain")
        config = _config("config-brain")

        assert resolve_brain(None, config, default="imported-brain") == "env-brain"

    def test_explicit_arg_wins_over_default_param(self, monkeypatch) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_BRAIN", "env-brain")
        config = _config("config-brain")

        assert resolve_brain("explicit-brain", config, default="imported-brain") == "explicit-brain"
