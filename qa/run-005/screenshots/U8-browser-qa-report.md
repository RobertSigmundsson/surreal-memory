# RUN-005 U8 — tonis-browser-qa-tester report (dashboard graph renders the migrated graph)

**Verdict: PASS** (pixel-backed).

- Target: dashboard SPA at `http://localhost:8030/ui` (QA stack, app v2.6.0 on SurrealDB v3.2.0).
- Data: default brain with 5 neurons + 5 native-RELATE synapses (served by `GET /api/graph`).
- Tool: Playwright + system chromium (`/usr/bin/chromium`), headless, 1440x900.

## Evidence
- `01_dashboard_loaded.png` — dashboard loads (v2.6.0, brain `default`, full nav).
- `02_graph_view.png` — the **Graph** tab: header "Neural Graph", panel
  "**Network Visualization — 5 nodes, 5 edges**", with 5 node dots and the connecting
  edges drawn on the SVG canvas (visually confirmed, not just DOM).

## Observed vs expected
| Check | Expected | Observed | Verdict |
|---|---|---|---|
| Graph view reachable | Graph nav renders a visualization | "Neural Graph" view rendered | PASS |
| Graph renders nodes/edges | 5 nodes + 5 edges (the seeded native-RELATE graph) | "5 nodes, 5 edges" + drawn nodes/edges | PASS |
| Not empty / not error | non-empty canvas, no crash | 5 node dots + edge lines drawn | PASS |
| Version surface | v2.6.0 | header shows v2.6.0 | PASS |

Harness console: 1 non-fatal 404 (a peripheral asset, not the graph data — `/api/graph`
returned 200 with 5 nodes + 5 edges).

Conclusion: the dashboard graph view renders the migrated native-RELATE graph. The synapses
that reach the browser were read via the v8 RELATE queries (`get_all_synapses` -> `/api/graph`),
so this is end-to-end proof of the release surface on SurrealDB 3.2.0.
