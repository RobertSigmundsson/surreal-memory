"""Tests for MCP auto-configuration (setup_mcp_claude)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from surreal_memory.cli.setup import (
    _add_via_claude_json,
    _claude_json_has_server,
    _cleanup_stale_mcp_servers_json,
    setup_mcp_claude,
)


class TestClaudeJsonHasServer:
    def test_file_not_exists(self, tmp_path: Path) -> None:
        assert _claude_json_has_server(tmp_path / "nope.json", "surreal-memory") is False

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        f.write_text("{}")
        assert _claude_json_has_server(f, "surreal-memory") is False

    def test_server_present(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        f.write_text(json.dumps({"mcpServers": {"surreal-memory": {"command": "smem-mcp"}}}))
        assert _claude_json_has_server(f, "surreal-memory") is True

    def test_different_server(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        f.write_text(json.dumps({"mcpServers": {"other": {"command": "other"}}}))
        assert _claude_json_has_server(f, "surreal-memory") is False

    def test_corrupt_json(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        f.write_text("not json")
        assert _claude_json_has_server(f, "surreal-memory") is False


class TestAddViaClaudeJson:
    def test_creates_new_file(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        assert _add_via_claude_json(f, {"command": "smem-mcp"}) is True
        data = json.loads(f.read_text())
        assert data["mcpServers"]["surreal-memory"]["command"] == "smem-mcp"

    def test_preserves_existing_data(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        f.write_text(json.dumps({"numStartups": 5, "mcpServers": {"other": {"command": "x"}}}))
        assert _add_via_claude_json(f, {"command": "smem-mcp"}) is True
        data = json.loads(f.read_text())
        assert data["numStartups"] == 5
        assert data["mcpServers"]["other"]["command"] == "x"
        assert data["mcpServers"]["surreal-memory"]["command"] == "smem-mcp"


class TestCleanupStaleMcpServersJson:
    def test_removes_stale_file(self, tmp_path: Path) -> None:
        stale = tmp_path / ".claude" / "mcp_servers.json"
        stale.parent.mkdir(parents=True)
        stale.write_text('{"surreal-memory": {}}')
        with patch("surreal_memory.cli.setup.Path.home", return_value=tmp_path):
            _cleanup_stale_mcp_servers_json()
        assert not stale.exists()

    def test_no_error_if_missing(self, tmp_path: Path) -> None:
        with patch("surreal_memory.cli.setup.Path.home", return_value=tmp_path):
            _cleanup_stale_mcp_servers_json()  # Should not raise


class TestSetupMcpClaude:
    def test_not_found_no_claude_dir(self, tmp_path: Path) -> None:
        with patch("surreal_memory.cli.setup.Path.home", return_value=tmp_path):
            assert setup_mcp_claude() == "not_found"

    def test_already_exists(self, tmp_path: Path) -> None:
        (tmp_path / ".claude").mkdir()
        claude_json = tmp_path / ".claude.json"
        # Entry with env.SURREALDB_PASS → considered complete, returns "exists"
        claude_json.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "surreal-memory": {
                            "command": "smem-mcp",
                            "env": {"SURREALDB_PASS": "surrealmemory"},
                        }
                    }
                }
            )
        )
        with patch("surreal_memory.cli.setup.Path.home", return_value=tmp_path):
            assert setup_mcp_claude() == "exists"

    def test_adds_via_fallback(self, tmp_path: Path) -> None:
        (tmp_path / ".claude").mkdir()
        with (
            patch("surreal_memory.cli.setup.Path.home", return_value=tmp_path),
            patch("surreal_memory.cli.setup._add_via_claude_cli", return_value=False),
            patch("surreal_memory.cli.setup.shutil.which", return_value=None),
        ):
            result = setup_mcp_claude()
        assert result == "added"
        data = json.loads((tmp_path / ".claude.json").read_text())
        assert "surreal-memory" in data["mcpServers"]

    def test_cleans_stale_on_success(self, tmp_path: Path) -> None:
        (tmp_path / ".claude").mkdir()
        stale = tmp_path / ".claude" / "mcp_servers.json"
        stale.write_text('{"old": true}')
        with (
            patch("surreal_memory.cli.setup.Path.home", return_value=tmp_path),
            patch("surreal_memory.cli.setup._add_via_claude_cli", return_value=False),
            patch("surreal_memory.cli.setup.shutil.which", return_value=None),
        ):
            result = setup_mcp_claude()
        assert result == "added"
        assert not stale.exists()

    def test_adds_via_cli_when_available(self, tmp_path: Path) -> None:
        (tmp_path / ".claude").mkdir()
        with (
            patch("surreal_memory.cli.setup.Path.home", return_value=tmp_path),
            patch("surreal_memory.cli.setup._add_via_claude_cli", return_value=True),
            patch("surreal_memory.cli.setup.shutil.which", return_value=None),
        ):
            result = setup_mcp_claude()
        assert result == "added"


class TestFindSmemCommandEnv:
    """find_smem_command() must include an 'env' key with SurrealDB config."""

    def test_returns_env_key(self, monkeypatch):
        from surreal_memory.cli.setup import find_smem_command

        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        result = find_smem_command()
        assert "env" in result, "find_smem_command must return an 'env' key"

    def test_env_contains_surrealdb_pass(self, monkeypatch):
        from surreal_memory.cli.setup import find_smem_command

        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        result = find_smem_command()
        assert "SURREALDB_PASS" in result["env"]

    def test_env_contains_storage_type(self, monkeypatch):
        from surreal_memory.cli.setup import find_smem_command

        result = find_smem_command()
        assert result["env"].get("SURREAL_MEMORY_STORAGE") == "surrealdb"

    def test_env_default_password_is_surrealmemory(self, monkeypatch):
        from surreal_memory.cli.setup import find_smem_command

        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        result = find_smem_command()
        assert result["env"]["SURREALDB_PASS"] == "surrealmemory"  # noqa: S105

    def test_env_respects_surrealdb_pass_override(self, monkeypatch):
        from surreal_memory.cli.setup import find_smem_command

        monkeypatch.setenv("SURREALDB_PASS", "custompass")
        result = find_smem_command()
        assert result["env"]["SURREALDB_PASS"] == "custompass"  # noqa: S105


class TestAddViaClaudeJsonWithEnv:
    """_add_via_claude_json writes env into the MCP entry."""

    def test_writes_env_field(self, tmp_path: Path) -> None:
        f = tmp_path / ".claude.json"
        entry = {"command": "smem-mcp", "env": {"SURREALDB_PASS": "surrealmemory"}}
        assert _add_via_claude_json(f, entry) is True
        data = json.loads(f.read_text())
        server = data["mcpServers"]["surreal-memory"]
        assert server["env"]["SURREALDB_PASS"] == "surrealmemory"  # noqa: S105


class TestSetupMcpClaudeDesktop:
    """setup_mcp_claude_desktop() — new function for Claude Desktop support."""

    def test_not_found_when_no_desktop_config_dir(self, tmp_path: Path) -> None:
        from surreal_memory.cli.setup import setup_mcp_claude_desktop

        with patch("surreal_memory.cli.setup.Path.home", return_value=tmp_path):
            result = setup_mcp_claude_desktop()
        assert result == "not_found"

    def test_adds_entry_with_env(self, tmp_path: Path) -> None:
        import sys

        from surreal_memory.cli.setup import setup_mcp_claude_desktop

        if sys.platform == "darwin":
            config_dir = tmp_path / "Library" / "Application Support" / "Claude"
        else:
            config_dir = tmp_path / ".config" / "Claude"
        config_dir.mkdir(parents=True)

        with (
            patch("surreal_memory.cli.setup.Path.home", return_value=tmp_path),
            patch("surreal_memory.cli.setup.shutil.which", return_value=None),
        ):
            result = setup_mcp_claude_desktop()

        assert result == "added"
        config_file = config_dir / "claude_desktop_config.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert "surreal-memory" in data["mcpServers"]
        assert "env" in data["mcpServers"]["surreal-memory"]
        assert "SURREALDB_PASS" in data["mcpServers"]["surreal-memory"]["env"]

    def test_idempotent_returns_exists(self, tmp_path: Path) -> None:
        import sys

        from surreal_memory.cli.setup import setup_mcp_claude_desktop

        if sys.platform == "darwin":
            config_dir = tmp_path / "Library" / "Application Support" / "Claude"
        else:
            config_dir = tmp_path / ".config" / "Claude"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "claude_desktop_config.json"
        config_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "surreal-memory": {"command": "smem-mcp", "env": {"SURREALDB_PASS": "x"}}
                    }
                }
            )
        )

        with (
            patch("surreal_memory.cli.setup.Path.home", return_value=tmp_path),
            patch("surreal_memory.cli.setup.shutil.which", return_value=None),
        ):
            result = setup_mcp_claude_desktop()

        assert result == "exists"

    def test_backfills_env_when_missing(self, tmp_path: Path) -> None:
        """Entry exists but lacks env → should update env (returns 'added')."""
        import sys

        from surreal_memory.cli.setup import setup_mcp_claude_desktop

        if sys.platform == "darwin":
            config_dir = tmp_path / "Library" / "Application Support" / "Claude"
        else:
            config_dir = tmp_path / ".config" / "Claude"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "claude_desktop_config.json"
        # Entry without env
        config_file.write_text(
            json.dumps({"mcpServers": {"surreal-memory": {"command": "smem-mcp"}}})
        )

        with (
            patch("surreal_memory.cli.setup.Path.home", return_value=tmp_path),
            patch("surreal_memory.cli.setup.shutil.which", return_value=None),
        ):
            result = setup_mcp_claude_desktop()

        assert result == "added"
        data = json.loads(config_file.read_text())
        assert "env" in data["mcpServers"]["surreal-memory"]

    def test_creates_backup_of_existing_file(self, tmp_path: Path) -> None:
        import sys

        from surreal_memory.cli.setup import setup_mcp_claude_desktop

        if sys.platform == "darwin":
            config_dir = tmp_path / "Library" / "Application Support" / "Claude"
        else:
            config_dir = tmp_path / ".config" / "Claude"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "claude_desktop_config.json"
        config_file.write_text(json.dumps({"mcpServers": {}}))

        with (
            patch("surreal_memory.cli.setup.Path.home", return_value=tmp_path),
            patch("surreal_memory.cli.setup.shutil.which", return_value=None),
        ):
            setup_mcp_claude_desktop()

        backup = config_dir / "claude_desktop_config.json.bak"
        assert backup.exists()
