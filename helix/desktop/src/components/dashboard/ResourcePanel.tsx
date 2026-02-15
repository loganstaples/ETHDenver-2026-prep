import { useEffect } from "react";
import type { SystemMetrics } from "../../lib/types";
import { PanelHeader } from "../ui/Panel";
import { Sparkline } from "../ui/Sparkline";
import { useMultiMetricHistory } from "../../hooks/useMetricHistory";
import { formatBytes, formatNumber } from "../../lib/utils";

interface ResourcePanelProps {
  metrics: SystemMetrics;
}

export function ResourcePanel({ metrics }: ResourcePanelProps) {
  const { cpu, memory, gpu, pushAll } = useMultiMetricHistory();

  useEffect(() => {
    pushAll(
      metrics.cpu_usage_percent,
      metrics.memory_total_mb > 0
        ? (metrics.memory_used_mb / metrics.memory_total_mb) * 100
        : 0,
      metrics.gpu_usage_percent ?? 0
    );
  }, [metrics, pushAll]);

  return (
    <div className="h-full bg-bg-panel p-3 flex flex-col">
      <PanelHeader>RESOURCES</PanelHeader>

      <div className="flex flex-col gap-3 flex-1">
        <ResourceRow
          label="CPU"
          value={`${formatNumber(metrics.cpu_usage_percent, 1)}%`}
          percent={metrics.cpu_usage_percent}
          sparkData={cpu}
        />
        <ResourceRow
          label="MEM"
          value={`${formatBytes(metrics.memory_used_mb)} / ${formatBytes(metrics.memory_total_mb)}`}
          percent={
            metrics.memory_total_mb > 0
              ? (metrics.memory_used_mb / metrics.memory_total_mb) * 100
              : 0
          }
          sparkData={memory}
        />
        <ResourceRow
          label="GPU"
          value={
            metrics.gpu_usage_percent != null
              ? `${formatNumber(metrics.gpu_usage_percent, 1)}%`
              : "N/A"
          }
          percent={metrics.gpu_usage_percent ?? 0}
          sparkData={gpu}
          dimmed={metrics.gpu_usage_percent == null}
        />
      </div>

      {/* Footer: absolute memory */}
      <div className="flex items-center gap-3 pt-2 border-t border-border-grid">
        <div className="flex items-center gap-1">
          <span className="text-label text-text-tertiary">RAM</span>
          <span className="text-caption text-text-secondary tabular-nums font-mono">
            {formatBytes(metrics.memory_used_mb)}
          </span>
        </div>
        {metrics.gpu_memory_used_mb != null && (
          <div className="flex items-center gap-1">
            <span className="text-label text-text-tertiary">VRAM</span>
            <span className="text-caption text-text-secondary tabular-nums font-mono">
              {formatBytes(metrics.gpu_memory_used_mb)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function ResourceRow({
  label,
  value,
  percent,
  sparkData,
  dimmed = false,
}: {
  label: string;
  value: string;
  percent: number;
  sparkData: number[];
  dimmed?: boolean;
}) {
  const clampedPct = Math.min(100, Math.max(0, percent));

  return (
    <div className={dimmed ? "opacity-40" : ""}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-label text-text-tertiary">{label}</span>
        <span className="text-mono-data text-text-primary tabular-nums">
          {value}
        </span>
      </div>
      <div className="flex items-center gap-2">
        {/* Progress bar */}
        <div className="flex-1 h-0.5 bg-border-primary">
          <div
            className="h-full bg-text-secondary transition-all duration-500"
            style={{ width: `${clampedPct}%` }}
          />
        </div>
        {/* Sparkline */}
        <Sparkline data={sparkData} width={80} height={24} />
      </div>
    </div>
  );
}
