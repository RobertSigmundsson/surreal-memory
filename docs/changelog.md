# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Source of truth:** [`CHANGELOG.md`](https://github.com/acidkill/surreal-memory/blob/main/CHANGELOG.md) in the repository root.
> This page mirrors that file for the MkDocs site.

## [Unreleased]

### Maintenance

- **Repository file-permission normalization** (`a4f27d0`, 2026-05-05): 963 tracked files had
  their mode bits drift from `100644` (regular) to `100755` (executable), likely due to a
  `core.fileMode` mismatch on the host filesystem. Zero content changes — this commit restores
  the intended permission state so future diffs reflect only real code changes.

## [1.0.0] — 2026-05-04

First stable release of **surreal-memory** as an independent PyPI package. This version forks from
surreal-memory 4.24.0, replaces SQLite with SurrealDB, and unlocks all Pro-tier features for free
via the bundled community plugin.

### Added

- **SurrealDB storage backend — fully implemented** (163/163 methods): Ten mixin classes covering
  typed memory, sources, alerts, cognitive state, review schedules, versioning, keyword/entity
  extraction, compression, activity tracking, and depth priors.
- **`get_project_memories` parity**: `SurrealDBTypedMemoryMixin` implements `get_project_memories`
  matching the `SQLiteTypedMemoryMixin` signature.
- **Memory type classifier expansion**: `suggest_memory_type()` extended from 9 to 12 covered
  types. New branches: `BOUNDARY`, `TOOL`, `CONTEXT`.
- **`INSTALL_PROMPT.md`**: 9-step Claude Code installation prompt covering prerequisites, Docker
  setup, pipx install, env config, MCP registration, doctor verification, and CLAUDE.md injection.
- **Community plugin** (`src/surreal_memory/plugins/community.py`): Bypasses Pro feature gates.
  Provides cone queries (HNSW vector search), smart merge, and directional compression.

### Fixed

- **`ensure_schema` never applied on connect** (F821 runtime bug): Schema now correctly
  initialises on every SurrealDB connection.
- **Mypy / ruff clean build**: Removed unused locals, typed `_max()`, added `# noqa` for
  intentionally naive datetime sentinels.
- **SQLite FK constraint in tests**: Test suite now creates `projects` rows before seeding
  `typed_memory` rows.
- **Taskmaster project-locality**: CWD-walking resolver — each project uses its own
  `.taskmaster/tasks.json`.

### Improved

- **`docs/getting-started/installation.md`** rewritten with accurate surreal-memory instructions.
- **`pyproject.toml` project URLs** corrected to `acidkill/surreal-memory-surrealdb-version`.
- **`README.md`** Quick Start: automated setup via Claude Code listed first; badge URLs corrected.

### Tests

- **150+ new parametrised tests** across four new test files covering `get_project_memories`
  parity, `suggest_memory_type` coverage (128 tests), remember-handler all-types (19 tests), and
  SurrealDB typed-memory integration (31 tests, skipped without `SURREALDB_URL`).
