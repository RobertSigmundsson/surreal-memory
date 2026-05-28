"""Unit tests for hooks.project_context.derive_project_name."""

from __future__ import annotations

from unittest.mock import patch

from surreal_memory.git_context import GitContext
from surreal_memory.hooks.project_context import derive_project_name


def test_env_override_takes_precedence(monkeypatch) -> None:
    """SMEM_PROJECT wins over git/cwd and is stripped."""
    monkeypatch.setenv("SMEM_PROJECT", "  my-override  ")
    # Even with a valid git context present, the override is returned.
    with patch(
        "surreal_memory.hooks.project_context.detect_git_context",
        return_value=GitContext(branch="main", commit="abc", repo_root="/x/repo", repo_name="repo"),
    ):
        assert derive_project_name({"cwd": "/x/repo"}) == "my-override"


def test_blank_env_override_is_ignored(monkeypatch) -> None:
    """A whitespace-only SMEM_PROJECT does not override git resolution."""
    monkeypatch.setenv("SMEM_PROJECT", "   ")
    with patch(
        "surreal_memory.hooks.project_context.detect_git_context",
        return_value=GitContext(branch="main", commit="abc", repo_root="/x/repo", repo_name="repo"),
    ):
        assert derive_project_name({"cwd": "/x/repo"}) == "repo"


def test_git_repo_name_from_hook_cwd(monkeypatch) -> None:
    """Resolves to the git repo basename when inside a repo."""
    monkeypatch.delenv("SMEM_PROJECT", raising=False)
    with patch(
        "surreal_memory.hooks.project_context.detect_git_context",
        return_value=GitContext(
            branch="main",
            commit="abc",
            repo_root="/home/u/surreal-memory",
            repo_name="surreal-memory",
        ),
    ) as mock_detect:
        assert derive_project_name({"cwd": "/home/u/surreal-memory/src"}) == "surreal-memory"
    mock_detect.assert_called_once()


def test_cwd_basename_fallback_when_not_in_git(monkeypatch) -> None:
    """Falls back to cwd basename when not inside a git repository."""
    monkeypatch.delenv("SMEM_PROJECT", raising=False)
    with patch(
        "surreal_memory.hooks.project_context.detect_git_context",
        return_value=None,
    ):
        assert derive_project_name({"cwd": "/tmp/some-dir"}) == "some-dir"


def test_uses_process_cwd_when_hook_input_missing(monkeypatch) -> None:
    """With no hook_input, derives from the process working directory."""
    monkeypatch.delenv("SMEM_PROJECT", raising=False)
    monkeypatch.setattr("surreal_memory.hooks.project_context.os.getcwd", lambda: "/tmp/proc-dir")
    with patch(
        "surreal_memory.hooks.project_context.detect_git_context",
        return_value=None,
    ):
        assert derive_project_name(None) == "proc-dir"
