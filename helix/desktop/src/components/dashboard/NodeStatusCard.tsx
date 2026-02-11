import { Card, CardHeader } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { StatusDot } from "../ui/StatusDot";
import { Copy } from "lucide-react";
import { formatUptime, truncateHash } from "../../lib/utils";
import type { NodeStatus } from "../../lib/types";

interface NodeStatusCardProps {
  status: NodeStatus;
}

export function NodeStatusCard({ status }: NodeStatusCardProps) {
  const copyPeerId = () => {
    navigator.clipboard.writeText(status.peer_id);
  };

  return (
    <Card>
      <CardHeader
        title="Node Status"
        action={
          <Badge variant={status.running ? "active" : "idle"}>
            <StatusDot
              status={status.running ? "active" : "idle"}
              pulse={status.running}
            />
            {status.running ? "Connected" : "Offline"}
          </Badge>
        }
      />
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-xs text-text-tertiary">Role</span>
          <span className="text-sm text-text-primary">{status.role}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-xs text-text-tertiary">Uptime</span>
          <span className="text-sm text-text-primary tabular-nums">
            {formatUptime(status.uptime_secs)}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-xs text-text-tertiary">Peers</span>
          <span className="text-sm text-text-primary tabular-nums">
            {status.connected_peers}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-xs text-text-tertiary">Peer ID</span>
          <button
            onClick={copyPeerId}
            className="flex items-center gap-1.5 text-sm text-text-secondary hover:text-text-primary transition-colors font-mono"
          >
            {truncateHash(status.peer_id)}
            <Copy size={12} strokeWidth={1.5} />
          </button>
        </div>
      </div>
    </Card>
  );
}
