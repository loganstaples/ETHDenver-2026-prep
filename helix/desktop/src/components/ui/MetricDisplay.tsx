import { cn } from "../../lib/utils";

interface MetricDisplayProps {
  label: string;
  value: string | number;
  unit?: string;
  className?: string;
}

export function MetricDisplay({
  label,
  value,
  unit,
  className,
}: MetricDisplayProps) {
  return (
    <div className={cn("flex flex-col", className)}>
      <span className="text-[11px] font-medium uppercase tracking-[0.05em] text-text-tertiary mb-1">
        {label}
      </span>
      <div className="flex items-baseline gap-1">
        <span className="text-xl font-semibold text-text-primary tabular-nums tracking-tight">
          {value}
        </span>
        {unit && (
          <span className="text-xs text-text-tertiary">{unit}</span>
        )}
      </div>
    </div>
  );
}
