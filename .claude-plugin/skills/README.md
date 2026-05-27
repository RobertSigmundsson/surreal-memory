# Surreal-Memory Skills

Composable AI agent skills for Surreal-Memory, following the [ship-faster](https://github.com/Heyvhuang/ship-faster) pattern.

## Available Skills

| Skill | Stage | Description |
|-------|-------|-------------|
| **memory-intake** | workflow | Structured memory creation from messy notes — 1-question-at-a-time clarification, batch storage with preview |
| **memory-audit** | review | 6-dimension quality review (purity, freshness, coverage, clarity, relevance, structure) with graded findings |
| **memory-evolution** | workflow | Evidence-based optimization from usage patterns — consolidation, enrichment, pruning, tag normalization |

## Install

### Claude Code (Plugin — Recommended)

Skills are included automatically when you install the plugin:

```bash
/plugin marketplace add acidkill/surreal-memory
/plugin install surreal-memory@surreal-memory-marketplace
```

### Manual Install

If not using the plugin, install skills separately:

```bash
smem install-skills
```

Options:

```bash
smem install-skills --list     # Show available skills
smem install-skills --force    # Overwrite with latest versions
```

## Usage

In Claude Code, invoke by name:

```
/memory-intake "Messy meeting notes: decided to use Redis for caching,
Bob will handle the migration, deadline is next Friday, also need to
update the API docs"

/memory-audit

/memory-evolution "Focus on the auth topic"
```

## Skill Format

Each skill follows the ship-faster SKILL.md format:

```yaml
---
name: skill-name
description: What it does
metadata:
  stage: workflow|review|tool
  tags: [relevant, tags]
agent: Agent Role Name
allowed-tools:
  - smem_remember
  - smem_recall
---
```

## Prerequisites

- Surreal-Memory installed (`pip install surreal-memory`)
- Surreal-Memory MCP server configured in Claude Code
- At least one brain with some memories (for audit/evolution)
