import { StatusDot } from "../ui/StatusDot";

interface StatusBarProps {
  nodeRunning: boolean;
  peerCount: number;
  uptime: string;
}

export function StatusBar({ nodeRunning, peerCount, uptime }: StatusBarProps) {
  return (
    <div className="h-6 bg-bg-secondary border-t border-border-primary flex items-center px-4 gap-4 text-[11px] text-text-tertiary shrink-0">
      <div className="flex items-center gap-1.5">
        <StatusDot status={nodeRunning ? "active" : "idle"} />
        <span>{nodeRunning ? "Connected" : "Disconnected"}</span>
      </div>
      <span className="text-border-secondary">|</span>
      <span>{peerCount} peers</span>
      <span className="text-border-secondary">|</span>
      <span>{uptime}</span>
      <div className="flex-1" />
      <span>HELIX v0.1.0</span>
    </div>
  );
}
