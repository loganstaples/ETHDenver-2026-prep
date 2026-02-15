import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Play, Square } from "lucide-react";
import { Button } from "../components/ui/Button";
import { NodeStatusCard } from "../components/dashboard/NodeStatusCard";
import { ResourceGauges } from "../components/dashboard/ResourceGauges";
import { TrainingOverview } from "../components/dashboard/TrainingOverview";
import { EarningsSummary } from "../components/dashboard/EarningsSummary";
import { RecentActivity } from "../components/dashboard/RecentActivity";
import { useNodeStatus } from "../hooks/useNodeStatus";
import { useSystemMetrics } from "../hooks/useSystemMetrics";
import { useTrainingStatus } from "../hooks/useTrainingStatus";
import type { ActivityEvent } from "../lib/types";
import { formatTimestamp } from "../lib/utils";

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

export function Dashboard() {
  const { status, start, stop } = useNodeStatus();
  const metrics = useSystemMetrics();
  const training = useTrainingStatus();
  const [toggling, setToggling] = useState(false);

  const handleToggle = async () => {
    setToggling(true);
    try {
      if (status.running) {
        await stop();
      } else {
        await start();
      }
    } finally {
      setToggling(false);
    }
  };

  const mockActivity: ActivityEvent[] = useMemo(
    () =>
      status.running
        ? [
            {
              id: 1,
              timestamp: formatTimestamp(new Date()),
              message: "Node started, connecting to network...",
              event_type: "info",
            },
          ]
        : [],
    [status.running]
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-text-primary tracking-tight">
            Dashboard
          </h1>
          <p className="text-sm text-text-tertiary">Compute Contributor</p>
        </div>
        <Button
          variant={status.running ? "secondary" : "primary"}
          size="lg"
          onClick={handleToggle}
          disabled={toggling}
        >
          {status.running ? (
            <>
              <Square size={14} strokeWidth={1.5} className="mr-2" />
              Stop Mining
            </>
          ) : (
            <>
              <Play size={14} strokeWidth={1.5} className="mr-2" />
              Start Mining
            </>
          )}
        </Button>
      </div>

      {/* Grid */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-2 gap-4"
      >
        <motion.div variants={item}>
          <NodeStatusCard status={status} />
        </motion.div>
        <motion.div variants={item}>
          <ResourceGauges metrics={metrics} />
        </motion.div>
        <motion.div variants={item}>
          <TrainingOverview training={training} />
        </motion.div>
        <motion.div variants={item}>
          <EarningsSummary training={training} />
        </motion.div>
        <motion.div variants={item} className="col-span-2">
          <RecentActivity events={mockActivity} />
        </motion.div>
      </motion.div>
    </div>
  );
}
