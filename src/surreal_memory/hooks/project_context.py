"""Project-scope resolution for Claude Code hooks.

All Claude Code hooks run against a single shared brain, but memories should
be scoped to the project (repository) the agent is currently working in.
This module derives a stable project name from the hook's working directory;
that name is used directly as the ``project_id`` on captured memories and to
filter recall, so scoping works on any storage backend without requiring a
separate project registry.

Project name resolution order:
    1. ``SMEM_PROJECT`` environment variable (explicit override)
    2. basename of the git repository root containing ``cwd``
    3. basename of ``cwd`` itself (when not inside a git repository)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from surreal_memory.git_context import detect_git_context


def derive_project_name(hook_input: dict[str, Any] | None = None) -> str | None:
    """Derive a stable project name from the hook working directory.

    Args:
        hook_input: Parsed Claude Code hook JSON. The ``cwd`` field is used
            when present; otherwise the process working directory is used.

    Returns:
        A project name (git repo basename, else cwd basename), or None if no
        usable directory could be determined.
    """
    override = os.environ.get("SMEM_PROJECT")
    if override and override.strip():
        return override.strip()

    cwd = ""
    if hook_input:
        cwd = str(hook_input.get("cwd") or "").strip()
    if not cwd:
        try:
            cwd = os.getcwd()
        except OSError:
            return None

    ctx = detect_git_context(Path(cwd))
    if ctx and ctx.repo_name:
        return ctx.repo_name

    base = Path(cwd).name
    return base or None
