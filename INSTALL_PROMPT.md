# Surreal-Memory — Claude Code Installation Prompt

## How to Use This File

On the **target machine**, open Claude Code in a terminal and run:

```
Please read INSTALL_PROMPT.md and follow the instructions to set up Surreal-Memory on this machine.
```

Or paste this file's contents directly into Claude Code as a message.

Claude Code will execute each step, ask for your API keys, and verify the full setup before finishing.

---

## Instructions for Claude Code

You are performing a complete setup of **Surreal-Memory** on this machine. Surreal-Memory is a self-hostable neural graph memory system for AI agents, backed by SurrealDB (document + graph + vector in one database). Follow every step below, in order. Do not skip steps or assume something is already done — verify each one.

> **SurrealDB ≥ 3.2.0 is required.** The bundled `docker-compose.surrealdb.yml` already uses
> `surrealdb/surrealdb:v3.2.0`. If this machine already runs an **older** SurrealDB for
> Surreal-Memory, the upgrade is in-place: **back up the `surrealdb_data` volume first**, then
> `docker compose -f docker-compose.surrealdb.yml pull && docker compose -f docker-compose.surrealdb.yml up -d`.
> On the first connect after the upgrade the `synapse` graph auto-migrates to native RELATE
> edges (synapse ids, fibers and the Merkle root are preserved; pre-migration rows are kept in
> a `synapse_migration_backup` table).

---

### Step 0 — Collect Required Values

Before running any commands, ask the user for the following values. You will use them in later steps.

| Value | Required | Description |
|-------|----------|-------------|
| `GEMINI_API_KEY` | **Recommended** | Enables semantic recall via Gemini embeddings (`gemini-embedding-001`). Free at https://aistudio.google.com/apikey. No key? Skip it and use the local offline embedder instead (see Step 3). |
| `SURREALDB_PASS` | No | SurrealDB password. Default: `surrealmemory`. Change for production. |
| `SURREAL_MEMORY_API_KEY` | No | Multi-device sync key. Leave empty to skip cloud sync. |

---

### Step 1 — Check Prerequisites

Run each check. If anything is missing, install it before continuing.

```bash
python3 --version   # must be 3.11 or higher
docker --version
docker compose version
git --version
pipx --version
```

**Install missing tools:**

```bash
# pipx (if missing)
python3 -m pip install --user pipx
python3 -m pipx ensurepath
source ~/.bashrc   # or restart the shell
```

For Docker on Ubuntu/Debian: follow https://docs.docker.com/engine/install/ubuntu/
For Docker on macOS: install Docker Desktop from https://www.docker.com/products/docker-desktop/

---

### Step 2 — Clone the Repository

```bash
git clone https://github.com/acidkill/surreal-memory.git \
    ~/repos/surreal-memory
```

If the directory already exists, skip the clone and verify it is up to date:

```bash
cd ~/repos/surreal-memory && git pull
```

---

### Step 3 — Configure Environment

```bash
cd ~/repos/surreal-memory
cp .env.example .env
```

Now edit `.env` and fill in the values collected in Step 0. At minimum, set:

```
GEMINI_API_KEY=<value from user>
SURREALDB_PASS=<value from user, or leave as: surrealmemory>
SURREAL_MEMORY_STORAGE=surrealdb
SURREAL_MEMORY_EMBEDDING_ENABLED=true
SURREAL_MEMORY_EMBEDDING_PROVIDER=gemini
```

**No Gemini key?** Use the local, offline embedder instead — set
`SURREAL_MEMORY_EMBEDDING_PROVIDER=sentence_transformer` (no API key; install the
`[embeddings]` extra in Step 5 instead of `[embeddings-gemini]`). Or set
`SURREAL_MEMORY_EMBEDDING_PROVIDER=auto` to auto-detect the best available provider.

If the user provided `SURREAL_MEMORY_API_KEY`, also set:

```
SURREAL_MEMORY_SYNC_ENABLED=true
SURREAL_MEMORY_API_KEY=<value from user>
```

---

### Step 4 — Start Services via Docker

```bash
cd ~/repos/surreal-memory
docker compose -f docker-compose.surrealdb.yml up -d --build
```

Wait ~30 seconds, then verify both services are healthy:

```bash
docker compose -f docker-compose.surrealdb.yml ps
```

Both `surrealdb` and `surreal-memory` must show `Up (healthy)`. If `surreal-memory` is not healthy, check logs:

```bash
docker compose -f docker-compose.surrealdb.yml logs surreal-memory --tail=50
```

Common fixes:
- `Port already in use` — Stop whatever is using port 8000 or 8001
- `Permission denied on /data` — Already fixed in Dockerfile.surrealdb; rebuild with `--build`

---

### Step 5 — Install smem CLI via pipx

Install the local fork into an isolated pipx environment with all required extras:

```bash
pipx install \
  "surreal-memory[surrealdb,embeddings-gemini,server] @ git+https://github.com/acidkill/surreal-memory.git"
```

Verify the CLI is available:

```bash
smem --version
smem-mcp --help
```

If `smem` is not found after install, reload the shell:

```bash
source ~/.bashrc   # or: exec $SHELL
```

> **Note:** v2.0.0 is a clean break from upstream NeuralMemory. The CLI entry points are `smem`, `smem-mcp`, and the `smem-hook-*` hooks — there are no other binaries. For the local no-key embedder, swap `embeddings-gemini` for `embeddings` in the extras above. See `CHANGELOG.md` for the full v2.0.0 BREAKING change list.

---

### Step 6 — Configure Shell Environment for smem-mcp

The `smem-mcp` process (the MCP server Claude Code will call) needs these environment variables every time it starts. Add them to your shell profile.

On Linux (bash):

```bash
cat >> ~/.bashrc << 'ENVBLOCK'

# Surreal-Memory MCP configuration
export SURREAL_MEMORY_STORAGE=surrealdb
export SURREALDB_URL=http://localhost:8001
export SURREALDB_USER=root
export SURREALDB_PASS=surrealmemory
export SURREALDB_NS=surreal_memory
export SURREALDB_DB=default
export SURREAL_MEMORY_EMBEDDING_PROVIDER=gemini
export SURREAL_MEMORY_EMBEDDING_ENABLED=true
ENVBLOCK
```

Then add the Gemini API key (replace `YOUR_KEY_HERE` with the value the user provided):

```bash
echo 'export GEMINI_API_KEY=YOUR_KEY_HERE' >> ~/.bashrc
source ~/.bashrc
```

On macOS (zsh), use `~/.zshrc` instead of `~/.bashrc`.

---

### Step 7 — Register MCP Server with Claude Code

```bash
claude mcp add --scope user surreal-memory -- smem-mcp
```

Verify it appears:

```bash
claude mcp list
```

`surreal-memory` must be in the list. If not, restart Claude Code after registration.

---

### Step 8 — Verify Full Setup

Run the diagnostics tool:

```bash
smem doctor
```

All checks must pass. Common failures and fixes:

| Failure | Fix |
|---------|-----|
| `SurrealDB not reachable` | Re-run Step 4; check `docker compose ps` |
| `Embedding provider error` | Verify `GEMINI_API_KEY` is set and valid |
| `smem-mcp not found` | Re-run Step 5; reload shell |
| `Schema not initialized` | Run `smem init` once |

Store and recall a test memory to confirm end-to-end:

```bash
smem remember "Surreal-Memory setup complete on this machine"
smem recall "setup"
```

The recall must return the stored memory.

---

### Step 9 — Add Agent Instructions to Claude Code

Copy the memory usage instructions into your global Claude Code configuration so future sessions know how to use the memory tools proactively:

```bash
cat >> ~/.claude/CLAUDE.md << 'EOF'

# Memory System

This machine uses Surreal-Memory (`smem_*` MCP tools) for persistent memory across sessions.

## Session Start (always do this first)
- `smem_recap()` — resume context from the last session
- `smem_context(limit=20)` — load recent memories

## During Work
- After every decision, bug fix, or important finding: call `smem_remember(content="...", type="decision|error|insight|fact")`
- Before asking the user a question: call `smem_recall(query="<topic>")` first

## Session End / Before Compaction
- `smem_auto(action="process", text="<session summary>")`
EOF
```

---

### Step 10 — Configure SessionStart Hook

Add the SessionStart hook so Claude Code automatically injects recent memories at
the beginning of every new session — no need to call `smem_context` manually.

Register the hook via the Claude Code CLI:

```bash
claude hooks add --event SessionStart --command "smem-hook-session-start"
```

Or add it manually to `~/.claude/settings.json` under the `hooks` key:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "smem-hook-session-start"
          }
        ]
      }
    ]
  }
}
```

Verify the hook works by starting a new Claude Code session — you should see
a `## Recent Memories` block in the initial context.

---

### Final Verification Checklist

Confirm all of the following before reporting setup complete:

- [ ] `docker compose -f ~/repos/surreal-memory/docker-compose.surrealdb.yml ps` shows both services `Up (healthy)`
- [ ] `smem --version` prints a version number
- [ ] `smem doctor` shows all checks passing
- [ ] `smem remember "test"` returns success with a fiber ID
- [ ] `smem recall "test"` returns the stored memory
- [ ] `claude mcp list` shows `surreal-memory`
- [ ] Dashboard accessible at http://localhost:8000/ui

Report the results of each checklist item to the user.

---

### Troubleshooting Reference

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `smem: command not found` | pipx install failed or PATH not updated | `source ~/.bashrc` then `pipx list` |
| MCP tools not appearing in Claude Code | smem-mcp not registered | Run Step 7; restart Claude Code |
| `Connection refused` on port 8001 | SurrealDB not running | `docker compose -f ... up -d` |
| `Embedding failed` | GEMINI_API_KEY missing or invalid | Check `echo $GEMINI_API_KEY`; re-run Step 6 |
| Build errors after code change | Docker uses stale image | `docker compose -f ... up -d --build` |
| `ModuleNotFoundError: surreal_memory` in tests | Editable install missing in pytest venv | `pipx runpip surreal-memory install -e .` |
