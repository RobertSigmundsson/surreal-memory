# Anthropic Discord — #mcp-servers or #show-your-work

**Message:**

Hey! I built **Surreal-Memory** — an MCP server that gives Claude Code persistent memory across sessions.

Instead of RAG/vector search, it uses **spreading activation on a neural graph** — memories are typed neurons connected by typed synapses (CAUSED_BY, LEADS_TO, RESOLVED_BY, etc.), and recall works by activating related concepts through the graph.

**Quick setup:**
```
/plugin marketplace add acidkill/surreal-memory
```
Or manual:
```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "uvx",
      "args": ["surreal-memory"]
    }
  }
}
```

**Features:** 53 MCP tools, proactive auto-save, habits tracking, connection explainer, multi-brain, local SQLite, optional embeddings (Ollama/Gemini/OpenAI).

**No API keys needed** for core recall — it's pure graph traversal.

GitHub: https://github.com/acidkill/surreal-memory

Happy to answer any questions!
