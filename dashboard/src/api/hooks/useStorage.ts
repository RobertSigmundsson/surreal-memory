import { useQuery } from "@tanstack/react-query"
import { api } from "@/api/client"
import type { SurrealDBStorageStatus, TierDistribution } from "@/api/types"

const keys = {
  status: ["storage", "status"] as const,
  tierStats: ["storage", "tier-stats"] as const,
}

/**
 * Fetch SurrealDB storage status (backend, connection, live counts).
 * Replaces the old SQLite/InfinityDB useStorageStatus hook — SurrealDB-only.
 */
export function useStorageStatus() {
  return useQuery({
    queryKey: keys.status,
    queryFn: () => api.get<SurrealDBStorageStatus>("/api/dashboard/storage/status"),
  })
}

export function useTierStats() {
  return useQuery({
    queryKey: keys.tierStats,
    queryFn: () => api.get<TierDistribution>("/api/dashboard/tier-stats"),
  })
}
