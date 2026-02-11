import { cn } from "../../lib/utils";

interface StatusDotProps {
  status: "active" | "warning" | "error" | "idle";
  pulse?: boolean;
  size?: "sm" | "md";
}

export function StatusDot({
  status,
  pulse = false,
  size = "sm",
}: StatusDotProps) {
  return (
    <span
      className={cn(
        "inline-block rounded-full",
        size === "sm" ? "h-1.5 w-1.5" : "h-2 w-2",
        status === "active" && "bg-status-active",
        status === "warning" && "bg-status-warning",
        status === "error" && "bg-status-error",
        status === "idle" && "bg-status-idle",
        pulse && "animate-pulse-dot"
      )}
    />
  );
}
