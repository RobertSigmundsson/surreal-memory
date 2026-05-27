# Integration Guide

Integrate Surreal-Memory with AI assistants, IDEs, and development tools.

## Claude Code Integration

### Option A: MCP Server (Recommended)

Surreal-Memory provides a native MCP (Model Context Protocol) server.

#### 1. Install Surreal-Memory

```bash
pip install surreal-memory
```

#### 2. Configure MCP Server

Add to `~/.claude/mcp_servers.json`:

=== "CLI (recommended)"
    ```json
    {
      "surreal-memory": {
        "command": "smem",
        "args": ["mcp"]
      }
    }
    ```

=== "Entry point"
    ```json
    {
      "surreal-memory": {
        "command": "smem-mcp"
      }
    }
    ```

=== "Python module"
    ```json
    {
      "surreal-memory": {
        "command": "python",
        "args": ["-m", "surreal_memory.mcp"]
      }
    }
    ```

#### 3. Restart Claude Code

After restarting, Claude has access to:

| Tool | Description |
|------|-------------|
| `smem_remember` | Store a memory with type, priority, tags |
| `smem_recall` | Query memories with depth and confidence |
| `smem_context` | Get recent context for injection |
| `smem_todo` | Quick TODO with 30-day expiry |
| `smem_stats` | Get brain statistics |
| `smem_auto` | Auto-capture memories from text |

#### 4. Usage

Claude automatically uses these tools:

```
You: Remember that we decided to use PostgreSQL
Claude: [uses smem_remember tool]
       Stored the decision about PostgreSQL.

You: What database did we choose?
Claude: [uses smem_recall tool]
       Based on my memory, you decided to use PostgreSQL.
```

### Option B: CLAUDE.md Instructions

Add to your project's `CLAUDE.md`:

```markdown
## Memory Instructions

At session start, get context:
```bash
smem context --limit 20 --json
```

When learning something important:
```bash
smem remember "Important info" --type decision
```

When recalling past information:
```bash
smem recall "query"
```
```

### Option C: Manual Context Injection

```bash
# Get context and inject at session start
CONTEXT=$(smem context --json --limit 20)
echo "Recent project context: $CONTEXT"
```

---

## VS Code Extension

Surreal-Memory has a dedicated VS Code extension with visual brain exploration and inline memory tools.

### Installation

```bash
cd vscode-extension
npm install && npm run build
```

Then install the generated `.vsix` file via **Extensions > Install from VSIX** or use Extension Developer Host (`F5`).

### Features

| Feature | Description |
|---------|-------------|
| **Memory Tree** | Activity bar sidebar with neurons grouped by type |
| **Graph Explorer** | Cytoscape.js force-directed graph with sub-graph navigation |
| **Encode** | Store selected text or typed input as memories |
| **Recall** | Query memories with depth selection |
| **CodeLens** | Memory counts on functions/classes, comment triggers |
| **Status Bar** | Live brain stats (neurons, synapses, fibers) |
| **WebSocket Sync** | Real-time updates across all views |

### Configuration

In VS Code settings (`surrealmemory.*`):

| Setting | Default | Description |
|---------|---------|-------------|
| `serverUrl` | `http://localhost:8000` | Surreal-Memory server URL |
| `pythonPath` | `python` | Python executable path |
| `graphNodeLimit` | `200` | Max nodes shown in graph |
| `codeLensTriggers` | `remember,note,decision,todo` | Comment triggers |

### Usage

1. Start the Surreal-Memory server: `smem serve`
2. Open VS Code — the extension connects automatically
3. Use command palette (`Ctrl+Shift+P`) for:
   - `Surreal-Memory: Encode Selection as Memory`
   - `Surreal-Memory: Recall Memory`
   - `Surreal-Memory: Open Graph Explorer`
   - `Surreal-Memory: Switch Brain`

---

## Cursor Integration

### Cursor Rules

Add to `.cursorrules` in your project:

```markdown
## Memory System

This project uses Surreal-Memory for persistent context.

### Getting Context
Before starting work:
```bash
smem context --limit 10
```

### Storing Information
```bash
smem remember "description" --type decision
smem remember "error fix" --type error
smem todo "task description"
```

### Recalling
```bash
smem recall "query"
```
```

### Cursor Commands

Create custom commands in Cursor settings:

```json
{
  "cursor.commands": [
    {
      "name": "Memory: Get Context",
      "command": "smem context --limit 10"
    },
    {
      "name": "Memory: Remember Selection",
      "command": "smem remember \"${selectedText}\""
    }
  ]
}
```

---

## Windsurf Integration

### Windsurf Rules

Create `.windsurfrules` in your project:

```markdown
## Surreal-Memory Integration

### Session Start
```bash
smem context --fresh-only --limit 10
```

### During Development
- Decisions: `smem remember "X" --type decision`
- Errors: `smem remember "X" --type error`
- TODOs: `smem todo "X" --priority 7`

### Querying
```bash
smem recall "your query" --depth 2
```
```

### AI Flow Integration

```yaml
name: "With Memory Context"
steps:
  - run: "smem context --json --limit 10"
    output: memory_context
  - prompt: |
      Recent project context:
      {{memory_context}}

      Now, {{user_request}}
```

---

## Aider Integration

### Shell Wrapper

Create `aider-with-memory.sh`:

```bash
#!/bin/bash
echo "Loading memory context..."
CONTEXT=$(smem context --json --limit 15)

aider --message "Project context from memory:
$CONTEXT

Remember to use 'smem remember' for important decisions." "$@"
```

### In-Session Commands

```
> /run smem context
> /run smem remember "We decided to use FastAPI" --type decision
> /run smem recall "API framework decision"
```

### Git Hook Integration

Create `.git/hooks/post-commit`:

```bash
#!/bin/bash
MSG=$(git log -1 --pretty=%B)
smem remember "Git commit: $MSG" --tag git --tag auto
```

---

## GitHub Copilot

Add to `.github/copilot-instructions.md`:

```markdown
## Memory Context

Get project context: `smem context`
Store decisions: `smem remember "X" --type decision`
Query past info: `smem recall "X"`
```

---

## VS Code with Continue.dev

In `.continue/config.json`:

```json
{
  "customCommands": [
    {
      "name": "memory-context",
      "description": "Get Surreal-Memory context",
      "prompt": "{{#if output}}Project memory context:\n\n{{output}}{{/if}}",
      "command": "smem context --limit 10"
    }
  ]
}
```

---

## Shell Integration

### Auto-Remember Git Commits

Add to `~/.bashrc` or `~/.zshrc`:

```bash
git() {
    command git "$@"
    if [[ "$1" == "commit" ]]; then
        local msg=$(command git log -1 --pretty=%B)
        smem remember "Git commit: $msg" --tag git --type workflow &
    fi
}
```

### Session Start Hook

```bash
smem-session() {
    echo "Recent Memory Context:"
    smem context --limit 5
}

cd() {
    builtin cd "$@"
    if [[ -f ".surreal-memory" ]]; then
        smem-session
    fi
}
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: CI with Memory

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Surreal-Memory
        run: pip install surreal-memory

      - name: Remember deployment
        if: github.ref == 'refs/heads/main'
        run: |
          smem remember "Deployed ${{ github.sha }} to main" \
            --type workflow \
            --tag deploy

      - name: Remember test results
        if: always()
        run: |
          smem remember "CI: ${{ job.status }} for ${{ github.sha }}" \
            --type workflow \
            --tag ci
```

---

## Best Practices

### 1. Semantic Commit Messages

```bash
# BAD
git commit -m "fix bug"

# GOOD
git commit -m "fix(auth): handle null email in validateUser

- Added null check at login.py:42
- Prevents crash on empty form submission"
```

### 2. Structured Memories

```bash
# BAD
smem remember "fixed it"

# GOOD
smem remember "Fixed auth bug: null email in validateUser(). Added null check at login.py:42." --tag auth --tag bugfix
```

### 3. Decision Records

```bash
smem remember "DECISION: JWT over sessions. REASON: Stateless scaling. ALTERNATIVE: Redis sessions" --type decision
```

### 4. Error-Solution Pairs

```bash
smem remember "ERROR: 'Cannot read id of undefined'. SOLUTION: Add null check before user.id" --type error
```

---

## Troubleshooting

### Memory Not Found

1. Check if content was stored: `smem stats`
2. Try broader query terms
3. Use `--depth 3` for deeper search

### MCP Server Not Working

1. Check Python path: `which python`
2. Test manually: `python -m surreal_memory.mcp`
3. Check Claude Code logs for errors

### Slow Queries

1. Use specific queries
2. Limit context: `smem context --limit 5`
3. Create separate brains per project
