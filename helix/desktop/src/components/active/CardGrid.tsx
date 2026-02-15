import { LayoutGroup } from "framer-motion";
import { MetricCard } from "./cards/MetricCard";
import { EarningsDetail } from "./cards/EarningsDetail";
import { TrainingDetail } from "./cards/TrainingDetail";
import { ResourcesDetail } from "./cards/ResourcesDetail";
import { NetworkDetail } from "./cards/NetworkDetail";
import { LossDetail } from "./cards/LossDetail";
import { RoundsDetail } from "./cards/RoundsDetail";
import { ActivityFeed } from "./ActivityFeed";
import { useTrainingStatus } from "../../hooks/useTrainingStatus";
import { useSystemMetrics } from "../../hooks/useSystemMetrics";
import { useNetworkPeers } from "../../hooks/useNetworkPeers";
import { useActivityLog } from "../../hooks/useActivityLog";
import { useMetricHistory, useMultiMetricHistory } from "../../hooks/useMetricHistory";
import { formatNumber, formatBytes } from "../../lib/utils";
import type { CardId } from "../../hooks/useExpandedCard";
import type { SessionInfo } from "../../lib/types";
import { useEffect, useState } from "react";
import { getSessionInfo } from "../../lib/tauri";

interface CardGridProps {
  expandedCard: CardId;
  onExpand: (id: CardId) => void;
}

export function CardGrid({ expandedCard, onExpand }: CardGridProps) {
  const training = useTrainingStatus();
  const metrics = useSystemMetrics();
  const peers = useNetworkPeers();
  const events = useActivityLog(true);
  const { history: earningsHistory, push: pushEarnings } = useMetricHistory();
  const { history: lossHistory, push: pushLoss } = useMetricHistory();
  const { cpu, memory, gpu, pushAll } = useMultiMetricHistory();
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);

  // Push metric history on updates
  useEffect(() => {
    pushEarnings(training.session_earned);
  }, [training.session_earned, pushEarnings]);

  useEffect(() => {
    if (training.current_loss !== null) pushLoss(training.current_loss);
  }, [training.current_loss, pushLoss]);

  useEffect(() => {
    const memPercent = metrics.memory_total_mb > 0
      ? (metrics.memory_used_mb / metrics.memory_total_mb) * 100
      : 0;
    pushAll(metrics.cpu_usage_percent, memPercent, metrics.gpu_usage_percent ?? 0);
  }, [metrics, pushAll]);

  useEffect(() => {
    getSessionInfo().then(setSessionInfo).catch(() => {});
    const interval = setInterval(() => {
      getSessionInfo().then(setSessionInfo).catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const cards = [
    {
      id: "earnings" as CardId,
      value: formatNumber(training.session_earned),
      label: "HLX earned",
      detail: (
        <EarningsDetail
          training={training}
          sessionInfo={sessionInfo}
          earningsHistory={earningsHistory}
        />
      ),
    },
    {
      id: "training" as CardId,
      value: training.current_round !== null ? `#${training.current_round}` : "—",
      label: training.phase,
      detail: <TrainingDetail training={training} lossHistory={lossHistory} />,
    },
    {
      id: "resources" as CardId,
      value: `${metrics.cpu_usage_percent.toFixed(0)}%`,
      label: `CPU · ${formatBytes(metrics.memory_used_mb)} mem`,
      detail: (
        <ResourcesDetail
          metrics={metrics}
          cpuHistory={cpu}
          memHistory={memory}
          gpuHistory={gpu}
        />
      ),
    },
    {
      id: "network" as CardId,
      value: `${peers.length}`,
      label: "peers",
      detail: <NetworkDetail peers={peers} />,
    },
    {
      id: "loss" as CardId,
      value: training.current_loss?.toFixed(4) ?? "—",
      label: "loss",
      detail: <LossDetail training={training} lossHistory={lossHistory} />,
    },
    {
      id: "rounds" as CardId,
      value: `${training.rounds_completed}`,
      label: "rounds",
      detail: <RoundsDetail training={training} />,
    },
  ];

  return (
    <div className="flex-1 flex flex-col gap-3 px-5 pb-4 overflow-hidden">
      <LayoutGroup>
        <div className="grid grid-cols-3 gap-3 flex-1 min-h-0">
          {cards.map((card) => (
            <MetricCard
              key={card.id}
              id={card.id}
              expandedCard={expandedCard}
              onExpand={onExpand}
              value={card.value}
              label={card.label}
              expandedContent={card.detail}
            />
          ))}
        </div>
      </LayoutGroup>
      <ActivityFeed events={events} />
    </div>
  );
}
