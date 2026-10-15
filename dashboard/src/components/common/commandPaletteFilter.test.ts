import { describe, expect, it } from "vitest"

import { matchesQuery, neuronSearchPath } from "./commandPaletteFilter"

describe("matchesQuery", () => {
  it("keeps everything for an empty or blank query", () => {
    expect(matchesQuery("Graph", "")).toBe(true)
    expect(matchesQuery("Graph", "   ")).toBe(true)
  })

  it("matches case-insensitively on a substring of the label", () => {
    expect(matchesQuery("Graph", "gra")).toBe(true)
    expect(matchesQuery("Tool Stats", "STATS")).toBe(true)
  })

  it("drops labels that do not contain the query", () => {
    expect(matchesQuery("Overview", "graph")).toBe(false)
    expect(matchesQuery("Settings", "zzz")).toBe(false)
  })
})

describe("neuronSearchPath", () => {
  it("points at the memory router, where /neurons is actually served", () => {
    expect(neuronSearchPath("graf")).toBe("/memory/neurons?content_contains=graf&limit=8")
  })

  it("encodes the query and honours the limit", () => {
    expect(neuronSearchPath("a b&c", 3)).toBe("/memory/neurons?content_contains=a%20b%26c&limit=3")
  })
})
