# Surreal-Memory — Instructions for Claude Code

> Copy this section into your project's `CLAUDE.md` or `~/.claude/CLAUDE.md` (global).

## Memory System

This workspace uses **Surreal-Memory** for persistent memory across sessions.
You have access to `smem_*` MCP tools. Use them **proactively** — do not wait for the user to ask.

All features are free. The community plugin provides vector search, smart merge, and directional compression at no cost.

### Recommended Backend

**SurrealDB** is the recommended backend for Surreal-Memory. It provides native vector search, graph relations, and real-time sync — all the advanced memory features work out of the box with zero extra configuration. The SQLite backend works for local-only use but lacks vector search.

### Session Start (ALWAYS do this)

```
smem_recap()                          # Resume context from last session
smem_context(limit=20)                # Load recent memories
smem_session(action="get")            # Check current task/feature/progress
```

If `gap_detected: true`, run `smem_auto(action="flush", text="<recent context>")` to recover lost content.

### During Work — REMEMBER automatically

| Event | Action |
|-------|--------|
| Decision made | `smem_remember(content="...", type="decision", priority=7)` |
| Bug fixed | `smem_remember(content="...", type="error", priority=7)` |
| User preference stated | `smem_remember(content="...", type="preference", priority=6)` |
| Important fact learned | `smem_remember(content="...", type="fact", priority=5)` |
| TODO identified | `smem_todo(task="...", priority=6)` |
| Workflow discovered | `smem_remember(content="...", type="workflow", priority=6)` |

### During Work — RECALL before asking

Before asking the user a question, check memory first:

```
smem_recall(query="<topic>", depth=1)
```

Depth guide: 0=instant lookup, 1=context (default), 2=patterns, 3=deep graph traversal.

### Session End / Before Compaction

```
smem_auto(action="process", text="<summary of session>")
smem_session(action="set", feature="...", task="...", progress=0.8)
```

Before `/compact` or `/new`:
```
smem_auto(action="flush", text="<recent conversation>")
```

### Project Context

```
smem_eternal(action="save", project_name="MyProject", tech_stack=["React", "Node.js"])
smem_eternal(action="save", decision="Use PostgreSQL", reason="Team expertise")
```

### Codebase Indexing

First time on a project:
```
smem_index(action="scan", path="./src")
```

Then `smem_recall(query="authentication")` finds related code through the neural graph.

### Knowledge Base Training

Train permanent knowledge from documentation files:
```
# Train from docs directory (PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV, MD, TXT, RST)
smem_train(action="train", path="docs/", domain_tag="react")

# Train a single file
smem_train(action="train", path="api-spec.pdf")

# Check training status
smem_train(action="status")
```

Trained knowledge is **pinned** — permanent, no decay, no pruning. Re-training same file is skipped (SHA-256 dedup).

For non-text formats: `pip install surreal-memory[extract]`

### Pin/Unpin Memories

```
smem_pin(fiber_ids=["id1", "id2"], pinned=true)   # Make permanent
smem_pin(fiber_ids=["id1"], pinned=false)           # Resume lifecycle
```

### Health & Diagnostics

```
smem_health()                              # Brain health score + warnings
smem_stats()                               # Memory counts and freshness
smem_alerts(action="list")                 # Active health alerts
smem_conflicts(action="list")              # Conflicting memories
smem_evolution()                           # Brain maturation + plasticity
```

### Spaced Repetition

```
smem_review(action="queue")                # Get memories due for review
smem_review(action="mark", fiber_id="...", success=true)  # Record result
```

### Brain Versioning & Transplant

```
smem_version(action="create", name="pre-refactor")  # Snapshot
smem_version(action="rollback", version_id="...")    # Restore
smem_transplant(source_brain="other", tags=["react"])  # Import from another brain
```

### Multi-Device Sync

```
smem_sync(action="full")                   # Bi-directional sync with hub
smem_sync_status()                         # Check sync status
smem_sync_config(action="set", hub_url="https://hub:8080", enabled=true)
```

### Import External Data

```
smem_import(source="chromadb", connection="/path/to/chroma")
smem_import(source="mem0", user_id="user123")
```

### Ephemeral Memories

For scratch notes, debugging context, or temporary reasoning:
```
smem_remember(content="Debugging: token expires at step 3", ephemeral=true)
```

Auto-expires after 24h, never synced to cloud, excluded from consolidation.
Filter them out: `smem_recall(query="...", permanent_only=true)`

### Compact Mode

All tools support `compact=true` to reduce response tokens by 60-80%.
Use `token_budget=N` to cap response size.
```
smem_recall(query="auth decisions", compact=true)
smem_stats(token_budget=200)
```

### Edit & Forget — Correct Mistakes

```
# Fix wrong type (auto-detector got it wrong)
smem_edit(memory_id="fiber-abc", type="insight")

# Fix wrong content
smem_edit(memory_id="fiber-abc", content="Corrected: the bug was in auth.py, not login.py")

# Adjust priority
smem_edit(memory_id="fiber-abc", priority=9)

# Soft delete — memory decays naturally (recommended for outdated info)
smem_forget(memory_id="fiber-abc", reason="outdated")

# Hard delete — permanent removal (for sensitive data, test garbage)
smem_forget(memory_id="fiber-abc", hard=true)
```

### Cognitive Reasoning

```
smem_hypothesize(action="create", content="Redis is the bottleneck", confidence=0.6)
smem_evidence(hypothesis_id="h-1", evidence_type="for", content="Redis latency 200ms")
smem_predict(action="create", content="Fix will drop latency 50%", hypothesis_id="h-1", deadline="2026-04-01")
smem_verify(prediction_id="p-1", outcome="correct")
smem_cognitive(action="summary")           # Hot index
smem_gaps(action="detect", topic="...")    # Track unknowns
smem_schema(action="evolve", hypothesis_id="h-1", content="...", reason="...")
```

### Connection Tracing

```
smem_explain(entity_a="Redis", entity_b="auth outage")
```

Traces shortest path with evidence. Use to debug recall or verify connections.

### Rules

1. **Be proactive** — remember important info without being asked
2. **Store 3-5 memories per task** — a bug fix has: root cause, fix, insight, prevention
3. **Use rich language** — "Chose X over Y because Z" not just "X". Mix causal, temporal, relational, comparative
4. **Check memory first** — recall before asking questions the user may have answered before
5. **Use diverse types** — fact, decision, error, preference, todo, workflow, insight, instruction, context
6. **Set priority** — critical=7-10, normal=5, trivial=1-3
7. **Add tags** — always include project name + topic for better retrieval
8. **Recap on start** — always call `smem_recap()` at session beginning
9. **Train KB first** — if project has docs/, train them into memory for permanent context
10. **Fix mistakes** — use `smem_edit` for wrong types/content, `smem_forget` for outdated info
11. **Health weekly** — `smem_health()` and fix the highest penalty first
