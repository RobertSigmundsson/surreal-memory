/**
 * Type definitions for the Surreal-Memory REST API.
 *
 * Mirrors the server response shapes from `src/surreal_memory/server/`.
 * Only public-API fields are typed here; internal fields are passed through
 * as `unknown` so the SDK does not break when the server adds new fields.
 */

export type MemoryType =
  | "fact"
  | "decision"
  | "error"
  | "insight"
  | "preference"
  | "workflow"
  | "instruction"
  | "concept"
  | "context"
  | "todo"

export type SynapseType =
  | "CAUSED_BY"
  | "LEADS_TO"
  | "CONTRADICTS"
  | "SIMILAR_TO"
  | "PART_OF"
  | "USED_BY"
  | "DEPENDS_ON"
  | string

export type NeuronType = "entity" | "concept" | "time" | "action" | "intent" | "state"

export type LifecycleStage = "full" | "summary" | "essence" | "ghost" | "metadata"

export interface Neuron {
  id: string
  type: NeuronType
  content: string
  content_hash?: number
  created_at: string
  metadata?: Record<string, unknown>
}

export interface Synapse {
  id: string
  type: SynapseType
  source_id: string
  target_id: string
  weight?: number
  created_at: string
}

export interface Fiber {
  id: string
  content: string
  type?: MemoryType
  priority?: number
  tags?: string[]
  stage?: LifecycleStage
  created_at: string
  updated_at?: string
  metadata?: Record<string, unknown>
}

export interface Brain {
  id: string
  name: string
  created_at: string
  stats?: BrainStats
}

export interface BrainStats {
  neurons: number
  synapses: number
  fibers: number
  active_neurons?: number
}

export interface RememberRequest {
  content: string
  type?: MemoryType
  priority?: number
  tags?: string[]
  ephemeral?: boolean
  metadata?: Record<string, unknown>
}

export interface RememberResponse {
  fiber_id: string
  type: MemoryType
  saved: true
}

export interface RecallRequest {
  query: string
  limit?: number
  type?: MemoryType
  tags?: string[]
  min_priority?: number
}

export interface RecallResult {
  fiber: Fiber
  score: number
  activation?: number
}

export interface RecallResponse {
  results: RecallResult[]
  query: string
  brain: string
}

export interface ContextRequest {
  limit?: number
  since?: string
}

export interface ContextResponse {
  fibers: Fiber[]
  brain: string
  count: number
}

export interface HealthResponse {
  status: "ok" | "degraded" | "down"
  version: string
  storage: string
  brain?: string
}

export interface ApiErrorPayload {
  detail?: string
  error?: string
  status?: number
}
