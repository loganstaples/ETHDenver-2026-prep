import { Settings, Square } from "lucide-react";
import { formatSessionTimer } from "../../lib/utils";

interface HeaderProps {
  sessionSeconds: number;
  onStop: () => void;
  onOpenSettings: () => void;
}

export function Header({ sessionSeconds, onStop, onOpenSettings }: HeaderProps) {
  return (
    <header className="h-12 flex items-center justify-between px-5 flex-shrink-0">
      {/* Left: wordmark */}
      <span className="text-text-tertiary tracking-[0.25em] text-xs font-light select-none">
        HELIX
      </span>

      {/* Right: timer + controls */}
      <div className="flex items-center gap-4">
        <span className="font-mono text-text-secondary text-xs tabular-nums">
          {formatSessionTimer(sessionSeconds)}
        </span>
        <button
          onClick={onOpenSettings}
          className="p-1.5 text-text-tertiary hover:text-text-secondary transition-colors cursor-pointer"
        >
          <Settings size={15} />
        </button>
        <button
          onClick={onStop}
          className="p-1.5 text-text-tertiary hover:text-status-error transition-colors cursor-pointer"
        >
          <Square size={13} fill="currentColor" />
        </button>
      </div>
    </header>
  );
}
