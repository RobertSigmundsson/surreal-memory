import type { SurrealDBStorageStatus } from "@/api/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { useTranslation } from "react-i18next"
import { Database, CheckCircle, XCircle } from "@phosphor-icons/react"

interface StorageStatusCardProps {
  status: SurrealDBStorageStatus
}

export function StorageStatusCard({ status }: StorageStatusCardProps) {
  const { t } = useTranslation()

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Database className="size-5" aria-hidden="true" />
          {t("storage.currentBackend", "Storage Backend")}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Backend badge + connection status */}
        <div className="flex flex-wrap items-center gap-3">
          <Badge variant="default" className="text-sm px-3 py-1">
            <span className="flex items-center gap-1.5">
              <Database className="size-3.5" aria-hidden="true" />
              SurrealDB
            </span>
          </Badge>
          <Badge
            variant={status.healthy ? "outline" : "destructive"}
            className="text-xs"
          >
            {status.healthy
              ? t("storage.connected", "Connected")
              : t("storage.disconnected", "Disconnected")}
          </Badge>
          {status.health_grade && (
            <Badge variant="secondary" className="text-xs">
              {t("storage.grade", "Grade")}: {status.health_grade}
            </Badge>
          )}
        </div>

        {/* Connection URL */}
        <div className="flex items-center gap-2 text-sm">
          {status.healthy ? (
            <CheckCircle className="size-4 shrink-0 text-green-500" aria-hidden="true" />
          ) : (
            <XCircle className="size-4 shrink-0 text-destructive" aria-hidden="true" />
          )}
          <span className="text-muted-foreground break-all">
            {status.url || t("storage.urlUnknown", "URL unknown")}
          </span>
        </div>

        {/* Namespace / Database */}
        {(status.namespace || status.database) && (
          <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
            <div>
              <span className="font-medium text-foreground">
                {t("storage.namespace", "Namespace")}
              </span>
              <p className="truncate">{status.namespace || "—"}</p>
            </div>
            <div>
              <span className="font-medium text-foreground">
                {t("storage.database", "Database")}
              </span>
              <p className="truncate">{status.database || "—"}</p>
            </div>
          </div>
        )}

        {/* Live counts */}
        <dl className="grid grid-cols-3 gap-3 text-center">
          <div>
            <dt className="text-xs text-muted-foreground">
              {t("storage.neurons", "Neurons")}
            </dt>
            <dd className="text-lg font-semibold">
              {status.neuron_count.toLocaleString()}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">
              {t("storage.synapses", "Synapses")}
            </dt>
            <dd className="text-lg font-semibold">
              {status.synapse_count.toLocaleString()}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">
              {t("storage.fibers", "Fibers")}
            </dt>
            <dd className="text-lg font-semibold">
              {status.fiber_count.toLocaleString()}
            </dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  )
}
