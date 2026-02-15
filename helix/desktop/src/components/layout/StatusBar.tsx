import type { NodeStatus, TrainingStatus } from "../../lib/types";
import { useSessionTimer } from "../../hooks/useSessionTimer";
import { formatSessionTimer } from "../../lib/utils";
import { StatusIndicator } from "../ui/StatusIndicator";

interface StatusBarProps {
  status: NodeStatus;
  training: TrainingStatus;
}

export function StatusBar({ status, training }: StatusBarProps) {
  const sessionTime = useSessionTimer(status.running, status.uptime_secs);

  return (
    <footer className="h-6 bg-bg-header border-t border-border-grid flex items-center px-4 gap-3 shrink-0 select-none">
      <StatusIndicator
        status={status.running ? "active" : "idle"}
        label={status.running ? "CONNECTED" : "OFFLINE"}
      />

      <Sep />
      <span className="text-caption text-text-tertiary tabular-nums">
        {status.connected_peers} peers
      </span>

      <Sep />
      <span className="text-caption text-text-tertiary tabular-nums font-mono">
        {formatSessionTimer(sessionTime)}
      </span>

      {training.active && (
        <>
          <Sep />
          <span className="text-caption text-text-tertiary">
            Round #{training.current_round} {training.phase}
          </span>
        </>
      )}

      <div className="flex-1" />
      <span className="text-caption text-text-muted">HELIX v0.1.0</span>
    </footer>
  );
}

function Sep() {
  return <div className="w-px h-2.5 bg-border-grid" />;
}
