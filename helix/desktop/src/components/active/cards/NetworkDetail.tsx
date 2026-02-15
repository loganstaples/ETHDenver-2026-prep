import type { PeerInfo } from "../../../lib/types";
import { truncateHash } from "../../../lib/utils";

interface NetworkDetailProps {
  peers: PeerInfo[];
}

export function NetworkDetail({ peers }: NetworkDetailProps) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-5 gap-2 text-label-sm text-text-tertiary pb-2 border-b border-border-subtle">
        <span>Peer</span>
        <span>Address</span>
        <span>Role</span>
        <span className="text-right">Reputation</span>
        <span className="text-right">Last seen</span>
      </div>
      {peers.length === 0 ? (
        <p className="text-body text-text-tertiary py-4 text-center">No peers connected</p>
      ) : (
        peers.map((peer) => (
          <div key={peer.id} className="grid grid-cols-5 gap-2 items-center py-1.5">
            <span className="text-metric-sm text-text-secondary font-mono">
              {truncateHash(peer.id)}
            </span>
            <span className="text-xs text-text-tertiary truncate">{peer.address}</span>
            <span className="text-xs text-text-secondary">{peer.role}</span>
            <span className="text-metric-sm text-text-secondary text-right">{peer.reputation}</span>
            <span className="text-xs text-text-tertiary text-right">
              {peer.last_seen_secs_ago < 5 ? "now" : `${peer.last_seen_secs_ago}s ago`}
            </span>
          </div>
        ))
      )}
    </div>
  );
}
