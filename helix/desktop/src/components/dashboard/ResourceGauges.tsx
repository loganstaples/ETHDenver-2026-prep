import { Card, CardHeader } from "../ui/Card";
import { ProgressBar } from "../ui/ProgressBar";
import { Cpu, MemoryStick, Monitor, type LucideIcon } from "lucide-react";
import { formatBytes } from "../../lib/utils";
import type { SystemMetrics } from "../../lib/types";

interface ResourceGaugesProps {
  metrics: SystemMetrics;
}

function ResourceRow({
  icon: Icon,
  label,
  value,
  max,
  detail,
}: {
  icon: LucideIcon;
  label: string;
  value: number;
  max: number;
  detail: string;
}) {
  const percent = max > 0 ? (value / max) * 100 : 0;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon size={14} strokeWidth={1.5} />
          <span className="text-xs text-text-secondary">{label}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-tertiary">{detail}</span>
          <span className="text-xs text-text-primary tabular-nums w-8 text-right">
            {Math.round(percent)}%
          </span>
        </div>
      </div>
      <ProgressBar value={percent} />
    </div>
  );
}

export function ResourceGauges({ metrics }: ResourceGaugesProps) {
  return (
    <Card>
      <CardHeader title="Resources" />
      <div className="space-y-4">
        <ResourceRow
          icon={Cpu}
          label="CPU"
          value={metrics.cpu_usage_percent}
          max={100}
          detail={`${Math.round(metrics.cpu_usage_percent)}% used`}
        />
        <ResourceRow
          icon={MemoryStick}
          label="Memory"
          value={metrics.memory_used_mb}
          max={metrics.memory_total_mb}
          detail={`${formatBytes(metrics.memory_used_mb)} / ${formatBytes(metrics.memory_total_mb)}`}
        />
        {metrics.gpu_usage_percent !== null && (
          <ResourceRow
            icon={Monitor}
            label="GPU"
            value={metrics.gpu_usage_percent}
            max={100}
            detail={
              metrics.gpu_memory_used_mb !== null && metrics.gpu_memory_total_mb !== null
                ? `${formatBytes(metrics.gpu_memory_used_mb)} / ${formatBytes(metrics.gpu_memory_total_mb)}`
                : `${Math.round(metrics.gpu_usage_percent)}% used`
            }
          />
        )}
      </div>
    </Card>
  );
}
