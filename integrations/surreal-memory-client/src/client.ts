/**
 * SurrealMemoryClient — typed REST client for a running Surreal-Memory server.
 *
 * @example
 * ```ts
 * import { SurrealMemoryClient } from "@acidkill/surreal-memory-client"
 *
 * const client = new SurrealMemoryClient({
 *   baseUrl: "http://localhost:8000",
 *   brain: "myproject",
 * })
 *
 * const { fiber_id } = await client.remember({
 *   content: "Fixed auth bug with null check in login.py:42",
 *   type: "fix",
 *   priority: 7,
 *   tags: ["auth"],
 * })
 *
 * const { results } = await client.recall({ query: "auth bug", limit: 5 })
 * ```
 */

import type {
  ApiErrorPayload,
  Brain,
  BrainStats,
  ContextRequest,
  ContextResponse,
  Fiber,
  HealthResponse,
  RecallRequest,
  RecallResponse,
  RememberRequest,
  RememberResponse,
} from "./types"

export interface ClientOptions {
  /** REST server URL, e.g. `http://localhost:8000`. No trailing slash. */
  baseUrl: string
  /** Default brain name. Can be overridden per-request. Optional. */
  brain?: string
  /** Optional bearer token sent as `Authorization: Bearer <token>`. */
  apiKey?: string
  /** Custom `fetch` implementation (defaults to global `fetch`). */
  fetch?: typeof fetch
  /** Default timeout in ms (defaults to 30000). */
  timeoutMs?: number
  /** Extra headers attached to every request. */
  headers?: Record<string, string>
}

export interface RequestOptions {
  /** Override the brain for this single call. */
  brain?: string
  /** Override the timeout for this single call. */
  timeoutMs?: number
  /** AbortSignal for cancellation. */
  signal?: AbortSignal
  /** Additional headers merged on top of client defaults. */
  headers?: Record<string, string>
}

export class ApiError extends Error {
  public readonly status: number
  public readonly payload: ApiErrorPayload | undefined

  constructor(status: number, message: string, payload?: ApiErrorPayload) {
    super(message)
    this.name = "ApiError"
    this.status = status
    this.payload = payload
  }
}

export class SurrealMemoryClient {
  private readonly baseUrl: string
  private readonly defaultBrain: string | undefined
  private readonly apiKey: string | undefined
  private readonly fetchImpl: typeof fetch
  private readonly defaultTimeoutMs: number
  private readonly defaultHeaders: Record<string, string>

  constructor(options: ClientOptions) {
    if (!options.baseUrl) {
      throw new Error("SurrealMemoryClient: baseUrl is required")
    }
    this.baseUrl = options.baseUrl.replace(/\/+$/, "")
    this.defaultBrain = options.brain
    this.apiKey = options.apiKey
    this.fetchImpl = options.fetch ?? globalThis.fetch
    this.defaultTimeoutMs = options.timeoutMs ?? 30_000
    this.defaultHeaders = options.headers ?? {}
  }

  // ── Memory operations ─────────────────────────────────────

  async remember(req: RememberRequest, options?: RequestOptions): Promise<RememberResponse> {
    return this.request<RememberResponse>("POST", "/api/remember", req, options)
  }

  async recall(req: RecallRequest, options?: RequestOptions): Promise<RecallResponse> {
    return this.request<RecallResponse>("POST", "/api/recall", req, options)
  }

  async context(req?: ContextRequest, options?: RequestOptions): Promise<ContextResponse> {
    const query = req ? this.buildQuery(req as Record<string, unknown>) : ""
    return this.request<ContextResponse>("GET", `/api/context${query}`, undefined, options)
  }

  async getFiber(fiberId: string, options?: RequestOptions): Promise<Fiber> {
    return this.request<Fiber>("GET", `/api/fibers/${encodeURIComponent(fiberId)}`, undefined, options)
  }

  async forget(fiberId: string, options?: RequestOptions): Promise<{ deleted: true }> {
    return this.request("DELETE", `/api/fibers/${encodeURIComponent(fiberId)}`, undefined, options)
  }

  // ── Brain operations ──────────────────────────────────────

  async listBrains(options?: RequestOptions): Promise<Brain[]> {
    return this.request<Brain[]>("GET", "/api/brains", undefined, options)
  }

  async getBrainStats(options?: RequestOptions): Promise<BrainStats> {
    return this.request<BrainStats>("GET", "/api/stats", undefined, options)
  }

  // ── Health / diagnostics ─────────────────────────────────

  async health(options?: RequestOptions): Promise<HealthResponse> {
    return this.request<HealthResponse>("GET", "/api/health", undefined, options)
  }

  // ── Internal request plumbing ────────────────────────────

  private async request<T>(
    method: string,
    path: string,
    body: unknown,
    options?: RequestOptions,
  ): Promise<T> {
    const brain = options?.brain ?? this.defaultBrain
    const timeoutMs = options?.timeoutMs ?? this.defaultTimeoutMs

    const headers = new Headers(this.defaultHeaders)
    for (const [k, v] of Object.entries(options?.headers ?? {})) {
      headers.set(k, v)
    }
    if (brain) {
      headers.set("X-Brain-ID", brain)
    }
    if (this.apiKey) {
      headers.set("Authorization", `Bearer ${this.apiKey}`)
    }
    if (body !== undefined && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json")
    }
    if (!headers.has("Accept")) {
      headers.set("Accept", "application/json")
    }

    const controller = new AbortController()
    const timeoutHandle =
      timeoutMs > 0
        ? setTimeout(() => controller.abort(new Error("Request timeout")), timeoutMs)
        : undefined

    const externalSignal = options?.signal
    if (externalSignal) {
      if (externalSignal.aborted) {
        controller.abort(externalSignal.reason)
      } else {
        externalSignal.addEventListener("abort", () => controller.abort(externalSignal.reason), {
          once: true,
        })
      }
    }

    try {
      const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      })

      if (!response.ok) {
        const payload = await this.safeReadErrorPayload(response)
        const message = payload?.detail ?? payload?.error ?? `HTTP ${response.status}`
        throw new ApiError(response.status, message, payload)
      }

      if (response.status === 204) {
        return undefined as T
      }

      return (await response.json()) as T
    } finally {
      if (timeoutHandle) {
        clearTimeout(timeoutHandle)
      }
    }
  }

  private async safeReadErrorPayload(response: Response): Promise<ApiErrorPayload | undefined> {
    try {
      const contentType = response.headers.get("Content-Type") ?? ""
      if (contentType.includes("application/json")) {
        return (await response.json()) as ApiErrorPayload
      }
      const text = await response.text()
      return { detail: text || undefined }
    } catch {
      return undefined
    }
  }

  private buildQuery(params: Record<string, unknown>): string {
    const search = new URLSearchParams()
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined || value === null) continue
      if (Array.isArray(value)) {
        for (const v of value) {
          search.append(key, String(v))
        }
      } else {
        search.set(key, String(value))
      }
    }
    const qs = search.toString()
    return qs ? `?${qs}` : ""
  }
}
