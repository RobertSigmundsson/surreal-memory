/**
 * The palette renders with `shouldFilter={false}` because fibers and neurons are
 * matched by hand (neurons server-side, fibers by summary). That also switched
 * off cmdk's filtering for the static items, so pages and Pro hints stayed on
 * screen whatever was typed. Every static item goes through this instead.
 */
export function matchesQuery(label: string, query: string): boolean {
  const q = query.trim().toLowerCase()
  return q.length === 0 || label.toLowerCase().includes(q)
}

/**
 * Neuron search lives on the memory router (`/memory/neurons`); the palette used
 * to call a bare `/neurons`, which the server never exposed, so every top-bar
 * search ended in a 404 that the catch block turned into "no results".
 */
export function neuronSearchPath(query: string, limit = 8): string {
  return `/memory/neurons?content_contains=${encodeURIComponent(query)}&limit=${limit}`
}
