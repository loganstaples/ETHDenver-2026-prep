import { motion } from "framer-motion";
import { Card, CardHeader } from "../components/ui/Card";
import { MetricDisplay } from "../components/ui/MetricDisplay";
import { Badge } from "../components/ui/Badge";
import { useNetworkPeers } from "../hooks/useNetworkPeers";

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.03 },
  },
};

const item = {
  hidden: { opacity: 0, y: 4 },
  show: { opacity: 1, y: 0, transition: { duration: 0.15, ease: "easeOut" as const } },
};

export function Network() {
  const peers = useNetworkPeers();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold text-text-primary tracking-tight">
          Network
        </h1>
        <p className="text-sm text-text-tertiary">
          Connected peers and network status
        </p>
      </div>

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="space-y-4"
      >
        {/* Connection Summary */}
        <motion.div variants={item}>
          <div className="grid grid-cols-4 gap-4">
            <Card padding="sm">
              <MetricDisplay label="Peers" value={peers.length} />
            </Card>
            <Card padding="sm">
              <MetricDisplay label="Latency" value="--" unit="ms" />
            </Card>
            <Card padding="sm">
              <MetricDisplay label="Bandwidth" value="--" unit="KB/s" />
            </Card>
            <Card padding="sm">
              <MetricDisplay label="Discovery" value="mDNS" />
            </Card>
          </div>
        </motion.div>

        {/* Peer List */}
        <motion.div variants={item}>
          <Card>
            <CardHeader title="Peers" subtitle={`${peers.length} connected`} />
            {peers.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border-primary">
                      <th className="text-left text-[11px] text-text-tertiary font-medium uppercase tracking-wide py-2">
                        Peer ID
                      </th>
                      <th className="text-left text-[11px] text-text-tertiary font-medium uppercase tracking-wide py-2">
                        Address
                      </th>
                      <th className="text-left text-[11px] text-text-tertiary font-medium uppercase tracking-wide py-2">
                        Role
                      </th>
                      <th className="text-left text-[11px] text-text-tertiary font-medium uppercase tracking-wide py-2">
                        Reputation
                      </th>
                      <th className="text-right text-[11px] text-text-tertiary font-medium uppercase tracking-wide py-2">
                        Last Seen
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {peers.map((peer) => (
                      <tr
                        key={peer.id}
                        className="border-b border-border-primary last:border-0 hover:bg-bg-tertiary/30 transition-colors duration-150"
                      >
                        <td className="py-2.5 font-mono text-text-secondary text-xs">
                          {peer.id}
                        </td>
                        <td className="py-2.5 font-mono text-text-tertiary text-xs">
                          {peer.address}
                        </td>
                        <td className="py-2.5">
                          <Badge>{peer.role}</Badge>
                        </td>
                        <td className="py-2.5 tabular-nums text-text-secondary">
                          {peer.reputation}
                        </td>
                        <td className="py-2.5 text-right text-text-tertiary tabular-nums">
                          {peer.last_seen_secs_ago}s ago
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8">
                <p className="text-sm text-text-tertiary">
                  No peers connected
                </p>
                <p className="text-xs text-text-disabled mt-1">
                  Start mining to discover and connect to network peers
                </p>
              </div>
            )}
          </Card>
        </motion.div>
      </motion.div>
    </div>
  );
}
