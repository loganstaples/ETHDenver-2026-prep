import { cn } from "../../lib/utils";

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  label?: string;
}

export function Toggle({ checked, onChange, disabled, label }: ToggleProps) {
  return (
    <label className="inline-flex items-center gap-2.5 cursor-pointer">
      <button
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative inline-flex h-5 w-9 items-center rounded-full transition-colors duration-200",
          "disabled:opacity-40 disabled:cursor-not-allowed",
          checked ? "bg-accent" : "bg-border-secondary"
        )}
      >
        <span
          className={cn(
            "inline-block h-3.5 w-3.5 rounded-full transition-transform duration-200",
            checked
              ? "translate-x-[18px] bg-bg-primary"
              : "translate-x-[3px] bg-text-tertiary"
          )}
        />
      </button>
      {label && (
        <span className="text-sm text-text-secondary">{label}</span>
      )}
    </label>
  );
}
