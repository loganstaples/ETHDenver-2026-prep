import type { SystemMetrics } from "../../../lib/types";
import { Sparkline } from "../../ui/Sparkline";
import { formatBytes } from "../../../lib/utils";

interface ResourcesDetailProps {
  metrics: SystemMetrics;
  cpuHistory: number[];
  memHistory: number[];
  gpuHistory: number[];
}

export function ResourcesDetail({ metrics, cpuHistory, memHistory, gpuHistory }: ResourcesDetailProps) {
  const memPercent = metrics.memory_total_mb > 0
    ? (metrics.memory_used_mb / metrics.memory_total_mb) * 100
    : 0;

  return (
    <div className="space-y-5">
      {/* CPU */}
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-baseline gap-3 mb-2">
            <p className="text-label text-text-tertiary">CPU</p>
            <p className="text-metric-md text-text-primary">
              {metrics.cpu_usage_percent.toFixed(0)}%
            </p>
          </div>
          <Sparkline data={cpuHistory} width={200} height={32} />
        </div>
      </div>

      {/* Memory */}
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-baseline gap-3 mb-2">
            <p className="text-label text-text-tertiary">Memory</p>
            <p className="text-metric-md text-text-primary">
              {formatBytes(metrics.memory_used_mb)}
              <span className="text-text-tertiary text-xs ml-1">
                / {formatBytes(metrics.memory_total_mb)}
              </span>
            </p>
          </div>
          <Sparkline data={memHistory} width={200} height={32} color="#3B82F6" />
        </div>
        <span className="text-metric-sm text-text-tertiary ml-4">{memPercent.toFixed(0)}%</span>
      </div>

      {/* GPU (if available) */}
      {metrics.gpu_usage_percent !== null && (
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-baseline gap-3 mb-2">
              <p className="text-label text-text-tertiary">GPU</p>
              <p className="text-metric-md text-text-primary">
                {metrics.gpu_usage_percent?.toFixed(0) ?? "—"}%
              </p>
            </div>
            <Sparkline data={gpuHistory} width={200} height={32} color="#A855F7" />
          </div>
        </div>
      )}

      {metrics.gpu_usage_percent === null && (
        <div className="pt-2 border-t border-border-subtle">
          <p className="text-label-sm text-text-tertiary">No GPU detected</p>
        </div>
      )}
    </div>
  );
}
