import { motion } from "framer-motion";
import type { NodeStatus } from "../../lib/types";
import { useTrainingStatus } from "../../hooks/useTrainingStatus";
import { useSystemMetrics } from "../../hooks/useSystemMetrics";
import { useNetworkPeers } from "../../hooks/useNetworkPeers";
import { useActivityLog } from "../../hooks/useActivityLog";
import { Header } from "../layout/Header";
import { StatusBar } from "../layout/StatusBar";
import { EarningsPanel } from "./EarningsPanel";
import { TrainingPanel } from "./TrainingPanel";
import { ResourcePanel } from "./ResourcePanel";
import { NetworkPanel } from "./NetworkPanel";
import { ActivityLog } from "./ActivityLog";

interface CommandCenterProps {
  status: NodeStatus;
  onStop: () => void;
  onOpenSettings: () => void;
}

export function CommandCenter({
  status,
  onStop,
  onOpenSettings,
}: CommandCenterProps) {
  const training = useTrainingStatus();
  const metrics = useSystemMetrics();
  const peers = useNetworkPeers();
  const events = useActivityLog(status.running);

  const earningsRate =
    status.uptime_secs > 0
      ? (training.session_earned / status.uptime_secs) * 3600
      : 0;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      className="h-full w-full flex flex-col"
    >
      <Header
        status={status}
        earningsRate={earningsRate}
        onStop={onStop}
        onOpenSettings={onOpenSettings}
      />

      {/* 5-panel grid */}
      <div
        className="flex-1 grid gap-px bg-border-grid overflow-hidden"
        style={{
          gridTemplateColumns: "30fr 30fr 40fr",
          gridTemplateRows: "1fr 1fr",
        }}
      >
        {/* Top row */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
        >
          <EarningsPanel training={training} />
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <TrainingPanel training={training} />
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
        >
          <ResourcePanel metrics={metrics} />
        </motion.div>

        {/* Bottom row */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <NetworkPanel peers={peers} connectedCount={status.connected_peers} />
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="col-span-2"
        >
          <ActivityLog events={events} />
        </motion.div>
      </div>

      <StatusBar status={status} training={training} />
    </motion.div>
  );
}
