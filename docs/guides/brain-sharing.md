# Brain Sharing

Share knowledge between agents and team members.

## Overview

Surreal-Memory supports multiple ways to share brains:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Export/Import** | File-based transfer | Backup, offline sharing |
| **Shared Server** | Real-time HTTP sync | Team collaboration |
| **Fork** | Create copy of brain | Start from template |
| **Merge** | Combine two brains | Aggregate knowledge |

## Export & Import

### Export a Brain

```bash
# Export current brain
smem brain export -o backup.json

# Export specific brain
smem brain export --name work -o work-backup.json

# Export without sensitive content
smem brain export --exclude-sensitive -o safe-share.json
```

### Import a Brain

```bash
# Import as new brain
smem brain import backup.json

# Import with custom name
smem brain import backup.json --name imported-brain

# Import and switch to it
smem brain import backup.json --use

# Merge into existing brain
smem brain import additional.json --merge

# Scan for sensitive content first
smem brain import untrusted.json --scan
```

### Export Format

The export is a JSON file containing:

```json
{
  "brain_id": "brain-123",
  "exported_at": "2026-02-05T10:00:00Z",
  "version": "0.4.0",
  "neurons": [...],
  "synapses": [...],
  "fibers": [...],
  "typed_memories": [...],
  "neuron_states": [...],
  "metadata": {
    "neuron_count": 150,
    "synapse_count": 280,
    "fiber_count": 45
  }
}
```

## Shared Server Mode

### Enable Shared Mode

Connect to a Surreal-Memory server:

```bash
# Enable with server URL
smem shared enable http://localhost:8000

# With API key authentication
smem shared enable https://memory.example.com --api-key YOUR_KEY

# With custom timeout
smem shared enable http://localhost:8000 --timeout 60
```

### Check Status

```bash
smem shared status
```

Output:
```
Shared mode: ENABLED
Server: http://localhost:8000
Connection: OK
Last sync: 2 minutes ago
```

### Test Connection

```bash
smem shared test
```

### Use Shared Storage

Once enabled, commands automatically use remote storage:

```bash
# Store to remote
smem remember "Shared team knowledge"

# Query from remote
smem recall "team decisions"
```

### Per-Command Sharing

Use `--shared` flag for single commands without enabling globally:

```bash
smem remember "Team insight" --shared
smem recall "project status" --shared
```

### Sync Local with Remote

```bash
# Full bidirectional sync
smem shared sync

# Push local to server only
smem shared sync --direction push

# Pull from server only
smem shared sync --direction pull
```

### Disable Shared Mode

```bash
smem shared disable
```

## Running a Server

### Start Server

```bash
pip install surreal-memory[server]
smem serve --host 0.0.0.0 --port 8000
```

### Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/memory/encode` | POST | Store memory |
| `/memory/query` | POST | Query memories |
| `/brain/create` | POST | Create brain |
| `/brain/{id}` | GET | Get brain info |
| `/brain/{id}/export` | GET | Export brain |
| `/sync/ws` | WS | Real-time sync |
| `/ui` | GET | Web visualization |
| `/api/graph` | GET | Graph data for UI |

### Docker Deployment

```dockerfile
FROM python:3.11-slim

RUN pip install surreal-memory[server]

EXPOSE 8000

CMD ["uvicorn", "surreal_memory.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t surreal-memory-server .
docker run -p 8000:8000 surreal-memory-server
```

## Use Cases

### Team Knowledge Base

1. One team member runs the server
2. All team members connect with `smem shared enable`
3. Decisions, patterns, and errors are automatically shared

```bash
# Team member 1
smem remember "API rate limit is 1000/hour" --type fact --shared

# Team member 2 (sees the same knowledge)
smem recall "rate limit" --shared
```

### Brain Templates

Create template brains for common setups:

```bash
# Create template
smem brain create python-project-template
smem remember "Use black for formatting" --type instruction
smem remember "Run pytest before commit" --type workflow
smem brain export -o python-template.json

# Share with team
# Each person imports as starting point
smem brain import python-template.json --name my-project
```

### Knowledge Transfer

When onboarding or handing off:

```bash
# Expert exports their brain
smem brain export --name auth-expertise -o auth-brain.json

# New team member imports
smem brain import auth-brain.json --name auth-learning
smem recall "authentication best practices"
```

### Multi-Agent Collaboration

Multiple AI agents share knowledge:

```bash
# Agent 1 learns something
smem remember "User prefers detailed explanations" --type preference --shared

# Agent 2 uses that knowledge
smem recall "user preferences" --shared
```

## Security Considerations

### Before Sharing

!!! warning "Check for Sensitive Content"
    Always check brain health before sharing:
    ```bash
    smem brain health
    ```

### Safe Export

```bash
# Exclude sensitive content
smem brain export --exclude-sensitive -o safe.json

# Scan import for issues
smem brain import untrusted.json --scan
```

### Brain Isolation

Use separate brains for different security levels:

```bash
smem brain create public-knowledge    # Safe to share
smem brain create internal-only       # Team only
smem brain create personal            # Never share
```

### Server Security

For production deployments:

- Use HTTPS
- Implement authentication (API keys)
- Set up proper CORS
- Use rate limiting
- Monitor for abuse

## Merge Strategies

When importing with `--merge`:

| Strategy | Behavior |
|----------|----------|
| Keep newer | Conflicting memories keep newer timestamp |
| Keep both | Both versions preserved with tags |
| Ask | Prompt for each conflict |

```bash
# Merge with existing brain
smem brain import updates.json --merge
```

## Troubleshooting

### Connection Failed

```bash
# Check server is running
curl http://localhost:8000/health

# Check firewall/network
ping memory.example.com

# Increase timeout
smem shared enable http://slow-server.com --timeout 120
```

### Sync Conflicts

```bash
# Check current status
smem shared status

# Force push local
smem shared sync --direction push

# Force pull remote
smem shared sync --direction pull
```

### Large Exports

For very large brains:

```bash
# Export with compression
smem brain export -o brain.json
gzip brain.json

# Import compressed
gunzip brain.json.gz
smem brain import brain.json
```
