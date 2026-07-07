# RUN-005 U8 — tonis-api-tester report (release surface: dashboard/MCP API returns migrated edges)

**Verdict: PASS.** The v2.6.0 release surface serves the native-RELATE synapse edges through
every synapse-bearing API endpoint. Each verdict below is a REAL request + captured response
(anti-fraud rule "call it, don't read it").

- Target: QA stack app `http://localhost:8030` (image `smemv26qa-surreal-memory:latest`, v2.6.0)
  on a live SurrealDB **v3.2.0** backend (`:8031`, `curl /version` → `surrealdb-3.2.0`).
- Data: default brain seeded with 5 neurons + 5 **native-RELATE** synapses
  (`INFO FOR DB` synapse def = `DEFINE TABLE synapse TYPE RELATION IN neuron OUT neuron SCHEMAFULL`).
- The app reached this state via the real `store.initialize()` path (version gate ≥3.2.0 →
  ensure_schema → apply_migrations → GQL probe). App health: `{"status":"ok","version":"2.6.0",...}`.

| Endpoint | Method | Exercises | HTTP | Result | Evidence |
|---|---|---|---|---|---|
| `/api/dashboard/stats` | GET | brain synapse aggregate | 200 | `total_synapses=5`, brain `default` `synapse_count=5` | dashboard-stats.json |
| `/api/graph` | GET | `get_all_synapses` (dashboard graph data) | 200 | 5 neurons + 5 synapses, each `{id, source_id, target_id, type, weight, direction}` — native in/out mapped back to the stable wire format | graph.json |
| `/memory/neurons/{id}/neighbors` | GET | `get_neighbors` (native in/out filter + in.*/out.* inline) | 200 | api-gateway -> rate-limiter, auth-service (neuron + synapse each) | neighbors.json |
| `/memory/neurons/{id}/path` | GET | `get_path` (GQL fast-path -> BFS fallback) | 200 | api-gateway -> auth-service -> ... -> user-db (2-hop path) | path.json |

Negative check: `/api/dashboard/*` correctly returns **403** from an untrusted client network
(require_local_request gate); it returns 200 only once the QA network (172.21.0.0/16) is trusted —
i.e. the gate actually rejects, it isn't open-by-accident.

Conclusion: the dashboard/MCP API returns the migrated native-RELATE edges end-to-end. Combined with
the U6 real-db-test-runner PASS (migration produces exactly this data), the release API surface is verified.
