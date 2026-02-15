import { Settings, Square } from "lucide-react";
import { formatSessionTimer } from "../../lib/utils";

interface HeaderProps {
  sessionSeconds: number;
  onStop: () => void;
  onOpenSettings: () => void;
}

export function Header({ sessionSeconds, onStop, onOpenSettings }: HeaderProps) {
  return (
    <header className="h-16 flex items-center justify-between px-8 flex-shrink-0 border-b border-border-subtle">
      {/* Left: wordmark */}
      <span className="text-text-tertiary tracking-[0.25em] text-sm font-light select-none">
        HELIX
      </span>

      {/* Right: timer + controls */}
      <div className="flex items-center gap-5">
        <span className="font-mono text-text-primary text-sm tabular-nums">
          {formatSessionTimer(sessionSeconds)}
        </span>
        <button
          onClick={onOpenSettings}
          className="p-1.5 text-text-tertiary hover:text-text-secondary transition-colors cursor-pointer"
        >
          <Settings size={18} />
        </button>
        <button
          onClick={onStop}
          className="p-1.5 text-text-tertiary hover:text-status-error transition-colors cursor-pointer"
        >
          <Square size={15} fill="currentColor" />
        </button>
      </div>
    </header>
  );
}
