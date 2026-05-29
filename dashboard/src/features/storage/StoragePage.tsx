import { useStats } from "@/api/hooks/useDashboard"
import { useTierStats } from "@/api/hooks/useStorage"
import { TierDistributionCard } from "./TierDistributionCard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { useTranslation } from "react-i18next"
import { Database } from "@phosphor-icons/react"

/**
 * SurrealDB-only storage view.
 *
 * The legacy SQLite↔InfinityDB backend switch and migration flow were removed
 * in the SurrealDB-only release (#28), along with the /storage/status endpoint.
 * This page now reports the active SurrealDB store using the live /stats and
 * /tier-stats endpoints — no backend migration UI.
 */
export default function StoragePage() {
  const { data: stats, isLoading } = useStats()
  const { data: tierStats } = useTierStats()
  const { t } = useTranslation()

  if (isLoading || !stats) {
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

  const brain = stats.active_brain ?? "default"

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-bold">{t("storage.title")}</h1>
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Database className="size-5" aria-hidden="true" />
              {t("storage.currentBackend", "Storage Backend")}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <Badge variant="default" className="text-sm px-3 py-1">
                <span className="flex items-center gap-1.5">
                  <Database className="size-3.5" aria-hidden="true" />
                  SurrealDB
                </span>
              </Badge>
              <span className="text-sm text-muted-foreground">
                {t("storage.brain", "Brain")}:{" "}
                <span className="font-medium text-foreground">{brain}</span>
              </span>
              <Badge variant="outline" className="text-xs">
                {t("storage.grade", "Grade")}: {stats.health_grade}
              </Badge>
            </div>

            <dl className="grid grid-cols-3 gap-3 text-center">
              <div>
                <dt className="text-xs text-muted-foreground">
                  {t("storage.neurons", "Neurons")}
                </dt>
                <dd className="text-lg font-semibold">
                  {stats.total_neurons.toLocaleString()}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">
                  {t("storage.synapses", "Synapses")}
                </dt>
                <dd className="text-lg font-semibold">
                  {stats.total_synapses.toLocaleString()}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">
                  {t("storage.fibers", "Fibers")}
                </dt>
                <dd className="text-lg font-semibold">
                  {stats.total_fibers.toLocaleString()}
                </dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        {tierStats && tierStats.total > 0 && (
          <TierDistributionCard data={tierStats} />
        )}
      </div>
    </div>
  )
}
