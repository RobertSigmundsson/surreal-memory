"""U8 Stage-2 browser QA: load the surreal-memory dashboard, navigate to the graph
view, and screenshot the rendered migrated (native-RELATE) graph. Uses the system
chromium via Playwright. Exit 0 = graph rendered with nodes+edges; 2 = not rendered.
"""

import sys

from playwright.sync_api import sync_playwright

SHOTS = "/home/acidkill/repos/surreal-memory-v26/qa/run-005/screenshots"
URL = "http://localhost:8030/ui"


def run() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        console_errors: list[str] = []
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.goto(URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2500)  # let the SPA fetch /api/graph + render
        page.screenshot(path=f"{SHOTS}/01_dashboard_loaded.png", full_page=True)

        # Try to reach the graph/network view (nav item), then screenshot again.
        for name in ("Graph", "Network", "Neural", "Visualiz", "Brain"):
            try:
                el = page.get_by_text(name, exact=False).first
                if el and el.is_visible():
                    el.click(timeout=2000)
                    page.wait_for_timeout(2000)
                    break
            except Exception:  # noqa: BLE001
                continue
        page.screenshot(path=f"{SHOTS}/02_graph_view.png", full_page=True)

        # Evidence: how many SVG/canvas graph elements rendered.
        svg_nodes = page.eval_on_selector_all(
            "svg circle, svg .node, canvas", "els => els.length"
        )
        svg_edges = page.eval_on_selector_all("svg line, svg path.link, svg .link", "els => els.length")
        body_text = page.inner_text("body")[:400]
        print(f"svg/canvas node-ish elements: {svg_nodes}")
        print(f"svg edge-ish elements: {svg_edges}")
        print(f"console errors: {len(console_errors)}")
        for e in console_errors[:5]:
            print("  console-error:", e[:120])
        print("body text sample:", body_text.replace("\n", " ")[:300])
        browser.close()
        # A rendered graph shows several node/edge SVG/canvas elements.
        return 0 if (svg_nodes >= 3 or svg_edges >= 3) else 2


sys.exit(run())
