import { cn } from "../../lib/utils";

interface PanelProps {
  children: React.ReactNode;
  className?: string;
  padding?: "none" | "sm" | "md";
}

const paddingMap = {
  none: "",
  sm: "p-2",
  md: "p-3",
};

export function Panel({ children, className, padding = "md" }: PanelProps) {
  return (
    <div
      className={cn(
        "bg-bg-panel border border-border-primary rounded-none overflow-hidden",
        paddingMap[padding],
        className
      )}
    >
      {children}
    </div>
  );
}

export function PanelHeader({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "text-label text-text-tertiary mb-2",
        className
      )}
    >
      {children}
    </div>
  );
}
