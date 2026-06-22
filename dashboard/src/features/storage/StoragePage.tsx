import { useStorageStatus, useTierStats } from "@/api/hooks/useStorage"
import { StorageStatusCard } from "./StorageStatusCard"
import { TierDistributionCard } from "./TierDistributionCard"
import { Skeleton } from "@/components/ui/skeleton"
import { useTranslation } from "react-i18next"

/**
 * SurrealDB-only storage view.
 *
 * Shows live SurrealDB connection status (backend, URL, health grade, counts)
 * from GET /api/dashboard/storage/status, plus the tier distribution card.
 * The legacy SQLite/InfinityDB migration flow was removed in the SurrealDB-only
 * release — no backend switching is needed or available.
 */
export default function StoragePage() {
  const { data: status, isLoading } = useStorageStatus()
  const { data: tierStats } = useTierStats()
  const { t } = useTranslation()

  if (isLoading || !status) {
    return (
      <div className="space-y-6 p-6">
        <h1 className="text-2xl font-bold">{t("storage.title")}</h1>
        <div className="grid gap-6 md:grid-cols-2">
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-bold">{t("storage.title")}</h1>
      <div className="grid gap-6 md:grid-cols-2">
        <StorageStatusCard status={status} />
        {tierStats && tierStats.total > 0 && (
          <TierDistributionCard data={tierStats} />
        )}
      </div>
    </div>
  )
}
