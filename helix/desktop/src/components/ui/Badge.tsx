import { cn } from "../../lib/utils";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "default" | "active" | "warning" | "error" | "idle";
  className?: string;
}

export function Badge({
  children,
  variant = "default",
  className,
}: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2 py-0.5 text-[11px] font-medium rounded",
        variant === "default" &&
          "bg-bg-tertiary text-text-secondary border border-border-primary",
        variant === "active" &&
          "bg-bg-tertiary text-status-active border border-border-primary",
        variant === "warning" &&
          "bg-bg-tertiary text-status-warning border border-border-primary",
        variant === "error" &&
          "bg-bg-tertiary text-status-error border border-border-primary",
        variant === "idle" &&
          "bg-bg-tertiary text-text-tertiary border border-border-primary",
        className
      )}
    >
      {children}
    </span>
  );
}
