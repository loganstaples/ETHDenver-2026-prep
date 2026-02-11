import { cn } from "../../lib/utils";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
}

export function Button({
  children,
  variant = "secondary",
  size = "md",
  className,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center font-medium rounded-md transition-colors duration-150 cursor-pointer",
        "disabled:opacity-40 disabled:cursor-not-allowed",
        variant === "primary" &&
          "bg-accent text-bg-primary hover:bg-accent/90",
        variant === "secondary" &&
          "bg-bg-tertiary text-text-primary border border-border-primary hover:bg-bg-hover",
        variant === "ghost" &&
          "text-text-secondary hover:text-text-primary hover:bg-bg-tertiary",
        size === "sm" && "text-xs px-2.5 py-1.5",
        size === "md" && "text-sm px-3.5 py-2",
        size === "lg" && "text-sm px-5 py-2.5",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
