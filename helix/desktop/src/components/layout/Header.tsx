import { Settings, Square } from "lucide-react";
import type { NodeStatus } from "../../lib/types";
import { useSessionTimer } from "../../hooks/useSessionTimer";
import { formatSessionTimer, formatNumber } from "../../lib/utils";
import { StatusIndicator } from "../ui/StatusIndicator";

interface HeaderProps {
  status: NodeStatus;
  earningsRate: number;
  onStop: () => void;
  onOpenSettings: () => void;
}

export function Header({
  status,
  earningsRate,
  onStop,
  onOpenSettings,
}: HeaderProps) {
  const sessionTime = useSessionTimer(status.running, status.uptime_secs);

  return (
    <header className="h-10 bg-bg-header border-b border-border-grid flex items-center px-4 gap-4 shrink-0 select-none">
      {/* Logo */}
      <span className="font-mono text-[14px] font-bold tracking-wider text-text-primary mr-1">
        HELIX
      </span>

      <StatusIndicator status="active" pulse />

      {/* Session timer */}
      <span className="text-mono-data text-text-secondary">
        SESSION {formatSessionTimer(sessionTime)}
      </span>

      <Separator />

      {/* Earnings rate */}
      <span className="text-mono-data text-status-active tabular-nums">
        {formatNumber(earningsRate, 3)} HLX/hr
      </span>

      <Separator />

      {/* Peers */}
      <span className="text-mono-data text-text-secondary tabular-nums">
        {status.connected_peers} peers
      </span>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Stop button */}
      <button
        onClick={onStop}
        className="flex items-center gap-1.5 text-mono-data text-text-secondary hover:text-status-error transition-colors px-2 py-1"
      >
        <Square size={10} />
        STOP
      </button>

      {/* Settings */}
      <button
        onClick={onOpenSettings}
        className="text-text-tertiary hover:text-text-primary transition-colors p-1"
      >
        <Settings size={14} />
      </button>
    </header>
  );
}

function Separator() {
  return <div className="w-px h-3 bg-border-primary" />;
}
