/**
 * @acidkill/surreal-memory-client
 *
 * Typed REST client for Surreal-Memory. See README.md for usage examples.
 */

export { SurrealMemoryClient, ApiError } from "./client"
export type { ClientOptions, RequestOptions } from "./client"
export type {
  ApiErrorPayload,
  Brain,
  BrainStats,
  ContextRequest,
  ContextResponse,
  Fiber,
  HealthResponse,
  LifecycleStage,
  MemoryType,
  Neuron,
  NeuronType,
  RecallRequest,
  RecallResponse,
  RecallResult,
  RememberRequest,
  RememberResponse,
  Synapse,
  SynapseType,
} from "./types"
