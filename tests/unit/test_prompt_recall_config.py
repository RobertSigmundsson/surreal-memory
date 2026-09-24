"""Regression tests: the [prompt_recall] section must survive save()/load().

The dataclass and the ``from_dict`` wiring shipped without the two halves that
make a section persistent — ``SECTION_NAMES`` and the ``save()`` emitter. The
result was silent: a hand-written ``[prompt_recall]`` block kept working until
something called ``save()`` (``smem doctor --fix`` does), which rewrote
config.toml without the section and reverted prompt recall to its ``enabled =
false`` default. Nothing logged, nothing raised — the feature simply stopped.

``save()`` cross-checks emitted headers against ``SECTION_NAMES`` and raises on
drift, so the two halves can only be added together. These tests pin both.
"""

from __future__ import annotations

from pathlib import Path

from surreal_memory.unified_config import PromptRecallConfig, UnifiedConfig


class TestPromptRecallConfig:
    def test_neutral_defaults(self) -> None:
        c = PromptRecallConfig()
        assert c.enabled is False
        assert c.min_prompt_chars == 40
        assert c.max_tokens == 600
        assert c.timeout_seconds == 5.0

    def test_round_trip(self) -> None:
        c = PromptRecallConfig(
            enabled=True, min_prompt_chars=200, max_tokens=900, timeout_seconds=20.0
        )
        assert PromptRecallConfig.from_dict(c.to_dict()) == c

    def test_from_dict_defaults_on_missing_keys(self) -> None:
        assert PromptRecallConfig.from_dict({}) == PromptRecallConfig()


class TestUnifiedConfigPromptRecallWiring:
    def test_section_is_declared(self) -> None:
        assert "prompt_recall" in UnifiedConfig.SECTION_NAMES

    def test_save_emits_the_section_header(self, tmp_path: Path) -> None:
        UnifiedConfig(data_dir=tmp_path).save()
        assert "[prompt_recall]" in (tmp_path / "config.toml").read_text(encoding="utf-8")

    def test_save_load_round_trip_preserves_prompt_recall(self, tmp_path: Path) -> None:
        cfg = UnifiedConfig(data_dir=tmp_path)
        cfg.prompt_recall = PromptRecallConfig(
            enabled=True, min_prompt_chars=200, max_tokens=900, timeout_seconds=20.0
        )
        cfg.save()

        loaded = UnifiedConfig.load(config_path=tmp_path / "config.toml")
        assert loaded.prompt_recall.enabled is True
        assert loaded.prompt_recall.min_prompt_chars == 200
        assert loaded.prompt_recall.max_tokens == 900
        assert loaded.prompt_recall.timeout_seconds == 20.0

    def test_resaving_does_not_drop_an_enabled_section(self, tmp_path: Path) -> None:
        """The exact failure: an enabled section silently reverting on rewrite."""
        cfg = UnifiedConfig(data_dir=tmp_path)
        cfg.prompt_recall = PromptRecallConfig(enabled=True, min_prompt_chars=200)
        cfg.save()

        UnifiedConfig.load(config_path=tmp_path / "config.toml").save()

        reloaded = UnifiedConfig.load(config_path=tmp_path / "config.toml")
        assert reloaded.prompt_recall.enabled is True
        assert reloaded.prompt_recall.min_prompt_chars == 200


class TestSystemPrefixes:
    """R1 (jev-uzycie-wdrozenie): `system_prefixes` survives save()/load() and rejects junk."""

    def test_default_is_the_measured_list(self) -> None:
        assert PromptRecallConfig().system_prefixes == (
            "<task-notification",
            "<bash-input",
            "<local-command",
            "<command-",
        )

    def test_missing_key_means_default(self) -> None:
        assert PromptRecallConfig.from_dict({"enabled": True}).system_prefixes == (
            PromptRecallConfig().system_prefixes
        )

    def test_explicit_empty_list_disables(self) -> None:
        assert PromptRecallConfig.from_dict({"system_prefixes": []}).system_prefixes == ()

    def test_invalid_entries_are_dropped_with_a_count(self, caplog) -> None:
        c = PromptRecallConfig.from_dict(
            {"system_prefixes": ["", "  ", "task", "<ok", '<zle"cudzyslow', "<Wielkie"]}
        )
        assert c.system_prefixes == ("<ok",)
        assert "dropped 5 invalid" in caplog.text

    def test_save_load_round_trip_keeps_prefixes(self, tmp_path: Path) -> None:
        for prefixes in (
            (),
            ("<task-notification", "<bash-input"),
            PromptRecallConfig().system_prefixes,
        ):
            cfg = UnifiedConfig(data_dir=tmp_path)
            cfg.prompt_recall = PromptRecallConfig(enabled=True, system_prefixes=prefixes)
            cfg.save()
            loaded = UnifiedConfig.load(config_path=tmp_path / "config.toml")
            assert loaded.prompt_recall.system_prefixes == prefixes
        tekst = (tmp_path / "config.toml").read_text(encoding="utf-8")
        assert (
            'system_prefixes = ["<task-notification", "<bash-input", "<local-command", "<command-"]'
            in tekst
        )

    def test_resaving_twice_does_not_drop_prefixes(self, tmp_path: Path) -> None:
        cfg = UnifiedConfig(data_dir=tmp_path)
        cfg.prompt_recall = PromptRecallConfig(enabled=True, system_prefixes=("<bash-input",))
        cfg.save()
        UnifiedConfig.load(config_path=tmp_path / "config.toml").save()
        UnifiedConfig.load(config_path=tmp_path / "config.toml").save()
        reloaded = UnifiedConfig.load(config_path=tmp_path / "config.toml")
        assert reloaded.prompt_recall.system_prefixes == ("<bash-input",)

    def test_file_without_the_key_loads_default(self, tmp_path: Path) -> None:
        """The live config.toml has no `system_prefixes` line — the filter must still be on."""
        (tmp_path / "config.toml").write_text(
            "[prompt_recall]\nenabled = true\nmin_prompt_chars = 200\n", encoding="utf-8"
        )
        loaded = UnifiedConfig.load(config_path=tmp_path / "config.toml")
        assert loaded.prompt_recall.system_prefixes == PromptRecallConfig().system_prefixes
