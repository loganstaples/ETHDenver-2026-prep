import { cn } from "../../lib/utils";

type Status = "active" | "warning" | "error" | "idle";

const colorMap: Record<Status, string> = {
  active: "bg-status-active",
  warning: "bg-status-warning",
  error: "bg-status-error",
  idle: "bg-status-idle",
};

interface StatusIndicatorProps {
  status: Status;
  label?: string;
  pulse?: boolean;
  className?: string;
}

export function StatusIndicator({
  status,
  label,
  pulse = false,
  className,
}: StatusIndicatorProps) {
  return (
    <span className={cn("inline-flex items-center gap-1.5", className)}>
      <span
        className={cn(
          "w-1.5 h-1.5 rounded-full shrink-0",
          colorMap[status],
          pulse && "animate-pulse-dot"
        )}
      />
      {label && (
        <span className="text-caption text-text-secondary">{label}</span>
      )}
    </span>
  );
}
