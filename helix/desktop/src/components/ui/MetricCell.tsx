import { cn } from "../../lib/utils";

interface MetricCellProps {
  label: string;
  value: React.ReactNode;
  className?: string;
}

export function MetricCell({ label, value, className }: MetricCellProps) {
  return (
    <div className={cn("flex flex-col gap-0.5", className)}>
      <span className="text-label text-text-tertiary">{label}</span>
      <span className="text-metric-sm text-text-primary">{value}</span>
    </div>
  );
}

export function MetricCellLarge({ label, value, className }: MetricCellProps) {
  return (
    <div className={cn("flex flex-col gap-0.5", className)}>
      <span className="text-label text-text-tertiary">{label}</span>
      <span className="text-metric-md text-text-primary">{value}</span>
    </div>
  );
}
