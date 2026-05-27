# OpenClaw Plugin Setup

Surreal-Memory replaces OpenClaw's built-in memory system (`memory-core`) with a
neural graph that survives context compaction, detects contradictions, and learns
from usage patterns.

## Why Replace memory-core?

OpenClaw memory is plain Markdown — `MEMORY.md` + `memory/YYYY-MM-DD.md`.
When a session hits the context window limit, compaction summarizes older messages
and **discards** what hasn't been written to disk. Any insight the agent didn't
explicitly save is lost.

Surreal-Memory stores everything in a persistent SQLite neural graph **outside** the
context window. Memories survive compaction, session restarts, and device changes.

| Feature | memory-core | Surreal-Memory |
|---------|-------------|--------------|
| Storage | Markdown files | SQLite neural graph |
| Search | Vector + BM25 (needs embedding API) | Spreading activation (zero cost) |
| Compaction-safe | No — unsaved context is lost | Yes — memories live outside context |
| Conflict detection | No | Auto-detect + resolution |
| Temporal reasoning | No | Causal chains + event sequences |
| Memory lifecycle | Static (keep forever or delete) | 4-stage (STM → Working → Episodic → Semantic) |
| Cross-session | Per-workspace only | Portable SQLite brains |
| Embedding cost | ~$0.02/1K queries | $0.00 |

## Prerequisites

- **OpenClaw** installed and running (`openclaw gateway`)
- **Python 3.11+** with pip
- **Node.js 18+** with npm

## Setup

> ⚠️ **Order matters strictly.** Steps 1 and 2 must be completed and verified
> before editing the config in Step 3. Adding `slots.memory = "surrealmemory"`
> before the plugin is loaded causes OpenClaw to refuse to start with
> `plugin not found: surrealmemory`.


### Step 1 — Install packages

Install the Python backend and the npm plugin package:

```bash
pip install surreal-memory
npm install -g surrealmemory
```

Verify both are working before continuing:

```bash
smem --help       # should show Surreal-Memory CLI commands
smem-mcp --help   # should show MCP server help
```

### Step 2 — Register the plugin with OpenClaw

Installing via `npm install -g` alone is not enough — OpenClaw does not scan the
global npm registry automatically. Copy the plugin into OpenClaw's extensions
directory:

```bash
cp -r $(npm root -g)/surrealmemory ~/.openclaw/extensions/surrealmemory
```

**Verify before continuing:**

```bash
openclaw plugins list | grep -i neural
```

You must see `surrealmemory` in the output before proceeding to Step 3. If it does
not appear, do **not** continue — adding `slots.memory` without the plugin loaded
will cause the gateway to fail on every restart.

---

### Step 3 — Configure openclaw.json

Only after the plugin is confirmed visible in `openclaw plugins list`, edit
`~/.openclaw/openclaw.json` and add or merge the following into the `plugins`
section:

```json
{
  "plugins": {
    "allow": ["surrealmemory"],
    "load": {
      "paths": ["~/.openclaw/extensions/surrealmemory"]
    },
    "slots": {
      "memory": "surrealmemory"
    }
  }
}
```

**Notes on this config:**

- **`allow`** — lists all non-bundled plugins you trust. Any plugin not listed
  here will emit a warning and may be disabled. Add all other plugins you use
  (e.g. `"telegram"`, `"llm-task"`).

- **`load.paths`** — explicitly tells OpenClaw where to find the plugin. Required
  when the plugin was registered via manual copy in Step 2.

- **`slots.memory`** — disables `memory-core` and activates Surreal-Memory as the
  exclusive memory provider. Plugin slots are exclusive — only one plugin can own
  a slot at a time.

### Step 4 — Restart the gateway

```bash
# If running as a daemon
openclaw gateway restart

# Or stop and start manually
openclaw gateway stop
openclaw gateway
```

### Verify

```bash
openclaw doctor
```

There should be no errors about `surrealmemory`. Then ask your agent:

```
What memory tools do you have?
```

The agent should list `smem_remember`, `smem_recall`, `smem_context`, `smem_todo`,
`smem_stats`, and `smem_health`. If it lists `memory_search` or `memory_get`
instead, the slot config is not applied — recheck Step 3.


## How It Works

```
OpenClaw Agent
    │
    ▼ (tool call: smem_recall)
OpenClaw Plugin (TypeScript, in-process)
    │
    ▼ JSON-RPC over stdio
Surreal-Memory MCP Server (Python subprocess)
    │
    ▼
SQLite Neural Graph (~/.surrealmemory/brains/)
```

The plugin:

1. **Starts** a Python MCP subprocess (`python -m surreal_memory.mcp`) when the
   gateway boots
2. **Registers 6 tools** directly into OpenClaw's tool system
3. **Before each agent run**: queries relevant memories and injects them as context
4. **After each agent run**: auto-captures decisions, errors, and insights from
   the conversation

## Plugin Configuration

Optional config under `plugins.entries.surrealmemory.config` in `openclaw.json`:

> **Important:** The entry name **must** be `surrealmemory` (no hyphen) — this
> matches the plugin ID. Using any other name (e.g. `surreal-memory`, `telegram`)
> will cause an "Invalid value" error.

```json
{
  "plugins": {
    "allow": ["surrealmemory"],
    "load": {
      "paths": ["~/.openclaw/extensions/surrealmemory"]
    },
    "slots": {
      "memory": "surrealmemory"
    },
    "entries": {
      "surrealmemory": {
        "config": {
          "pythonPath": "python",
          "brain": "default",
          "autoContext": true,
          "autoCapture": true,
          "contextDepth": 1,
          "maxContextTokens": 500,
          "timeout": 30000,
          "initTimeout": 90000
        }
      }
    }
  }
}
```

| Option | Default | Description |
|--------|---------|-------------|
| `pythonPath` | `"python"` | Path to Python executable with `surreal-memory` installed |
| `brain` | `"default"` | Brain name for this workspace |
| `autoContext` | `true` | Inject relevant memories before each agent run |
| `autoCapture` | `true` | Extract and store memories after each agent run |
| `contextDepth` | `1` | Recall depth: 0=instant, 1=context, 2=habit, 3=deep |
| `maxContextTokens` | `500` | Maximum tokens for auto-context injection |
| `timeout` | `30000` | MCP request timeout in milliseconds |
| `initTimeout` | `90000` | MCP initialize handshake timeout (increase if first boot is slow) |

> **Only these keys are allowed in `config`.** The schema uses
> `additionalProperties: false` — any extra key (e.g. `"enabled"`, `"url"`)
> will trigger an "Invalid value" error from OpenClaw.

## Available Tools

Once configured, the agent has access to these tools:

| Tool | Description |
|------|-------------|
| `smem_remember` | Store a memory (fact, decision, error, preference, etc.) |
| `smem_recall` | Query memories via spreading activation |
| `smem_context` | Get recent memories for context |
| `smem_todo` | Quick TODO with 30-day expiry |
| `smem_stats` | Brain statistics |
| `smem_health` | Brain health diagnostics |

The plugin also injects a system prompt telling the agent to use `smem_*` tools
exclusively and **not** use `memory_search` or `memory_get` from the disabled
`memory-core` plugin.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Invalid value` in plugin config | Unknown key or wrong entry name | Only use documented config keys. Entry name must be `surrealmemory` (no hyphen) |
| `no MCP Client` | Using SKILL.md with `mcp:` block | Skills don't support MCP. Use the Plugin approach (this guide) |
| `ENOENT: python not found` | Wrong Python path | Set `pythonPath` in plugin config to your Python binary |
| `MCP process exited with code 1` | `surreal-memory` not installed | Run `pip install surreal-memory` |
| Agent still uses `memory_search` | Slot not configured | Set `plugins.slots.memory = "surrealmemory"` in `openclaw.json` |
| Agent uses both `smem_*` and `memory_*` | `memory-core` still active | Check slot config — only one memory plugin can be active |
| `MCP timeout` | Slow machine or large brain | Increase `timeout` in plugin config (default: 30000ms) |
| Plugin not found | Not installed globally | Run `npm install -g surrealmemory` |

## Common Mistakes

### Adding unknown keys to config

```json
// WRONG — "enabled" is not a valid config key → Invalid value
{
  "plugins": {
    "enabled": true,
    "slots": { "memory": "surrealmemory" },
    "entries": {
      "surrealmemory": {
        "config": { "enabled": true }
      }
    }
  }
}

// CORRECT — only use documented keys
{
  "plugins": {
    "slots": { "memory": "surrealmemory" },
    "entries": {
      "surrealmemory": {
        "config": { "pythonPath": "python", "brain": "default" }
      }
    }
  }
}
```

The plugin schema is strict (`additionalProperties: false`). Only these config
keys are accepted: `pythonPath`, `brain`, `autoContext`, `autoCapture`,
`contextDepth`, `maxContextTokens`, `timeout`.

### Using wrong entry name

```json
// WRONG — entry name must be "surrealmemory" (the plugin ID)
{ "plugins": { "entries": { "surreal-memory": { "config": {} } } } }
{ "plugins": { "entries": { "telegram": { "enabled": true } } } }

// CORRECT
{ "plugins": { "entries": { "surrealmemory": { "config": {} } } } }
```

### Using `"memory": "none"`

```json
// WRONG — disables ALL memory plugins including Surreal-Memory
{ "plugins": { "slots": { "memory": "none" } } }

// CORRECT — activates Surreal-Memory, disables memory-core
{ "plugins": { "slots": { "memory": "surrealmemory" } } }
```

### Using SKILL.md with `mcp:` block

```markdown
# WRONG — OpenClaw skills don't have an MCP client
---
mcp:
  surreal-memory:
    command: smem-mcp
---
```

OpenClaw skills provide instructions to the LLM but cannot spawn MCP server
processes. The plugin approach bundles its own MCP client that communicates with
the Surreal-Memory Python process over stdio.

### Adding rules to AGENTS.MD

```markdown
# WRONG — AGENTS.MD rules can't disable registered tools
Do NOT use memory_search. Use smem_recall instead.
```

AGENTS.MD is an instruction to the model, not a tool access control. The model
may still call `memory_search` if `memory-core` is registered. The correct fix
is the slot config in Step 2 — it prevents `memory-core` from loading entirely.

## Further Reading

- [Quick Start](../getting-started/quickstart.md) — Basic Surreal-Memory usage
- [CLI Reference](../getting-started/cli.md) — All commands and options
- [Integration Guide](integration.md) — Setup for Claude Code, Cursor, and other editors
- [MCP Server Guide](mcp-server.md) — MCP configuration for 20+ editors
