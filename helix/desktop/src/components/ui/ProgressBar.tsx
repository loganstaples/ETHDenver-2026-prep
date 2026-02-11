import { cn } from "../../lib/utils";

interface ProgressBarProps {
  value: number;
  max?: number;
  className?: string;
  size?: "sm" | "md";
  showLabel?: boolean;
}

export function ProgressBar({
  value,
  max = 100,
  className,
  size = "sm",
  showLabel = false,
}: ProgressBarProps) {
  const percent = Math.min(100, Math.max(0, (value / max) * 100));

  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <div
        className={cn(
          "flex-1 rounded-full bg-border-primary overflow-hidden",
          size === "sm" ? "h-1" : "h-1.5"
        )}
      >
        <div
          className="h-full bg-accent rounded-full transition-[width] duration-400 ease-in-out"
          style={{ width: `${percent}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-xs text-text-tertiary tabular-nums w-8 text-right">
          {Math.round(percent)}%
        </span>
      )}
    </div>
  );
}
