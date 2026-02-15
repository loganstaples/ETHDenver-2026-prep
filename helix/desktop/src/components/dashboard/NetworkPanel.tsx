import type { PeerInfo } from "../../lib/types";
import { PanelHeader } from "../ui/Panel";
import { StatusIndicator } from "../ui/StatusIndicator";
import { truncateHash } from "../../lib/utils";

interface NetworkPanelProps {
  peers: PeerInfo[];
  connectedCount: number;
}

export function NetworkPanel({ peers, connectedCount }: NetworkPanelProps) {
  const displayPeers = peers.slice(0, 6);

  return (
    <div className="h-full bg-bg-panel p-3 flex flex-col overflow-hidden">
      <PanelHeader>NETWORK</PanelHeader>

      {/* Summary row */}
      <div className="flex items-center gap-4 mb-2">
        <div className="flex items-center gap-1">
          <span className="text-label text-text-tertiary">PEERS</span>
          <span className="text-metric-sm text-text-primary tabular-nums">
            {connectedCount}
          </span>
        </div>
      </div>

      {/* Peer list */}
      <div className="flex-1 overflow-y-auto">
        {displayPeers.length === 0 ? (
          <div className="text-body text-text-tertiary text-center py-4">
            No peers connected
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-border-grid">
                <th className="text-left text-label text-text-muted py-0.5 px-1">
                  PEER
                </th>
                <th className="text-left text-label text-text-muted py-0.5 px-1">
                  ROLE
                </th>
                <th className="text-right text-label text-text-muted py-0.5 px-1">
                  REP
                </th>
                <th className="text-right text-label text-text-muted py-0.5 px-1">
                  STATUS
                </th>
              </tr>
            </thead>
            <tbody>
              {displayPeers.map((peer) => (
                <tr
                  key={peer.id}
                  className="border-b border-border-grid hover:bg-bg-hover transition-colors"
                >
                  <td className="py-0.5 px-1 text-mono-data text-text-secondary">
                    {truncateHash(peer.id, 4)}
                  </td>
                  <td className="py-0.5 px-1 text-caption text-text-tertiary">
                    {peer.role}
                  </td>
                  <td className="py-0.5 px-1 text-mono-data text-text-secondary text-right tabular-nums">
                    {peer.reputation}
                  </td>
                  <td className="py-0.5 px-1 text-right">
                    <StatusIndicator
                      status={peer.last_seen_secs_ago < 10 ? "active" : "warning"}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
