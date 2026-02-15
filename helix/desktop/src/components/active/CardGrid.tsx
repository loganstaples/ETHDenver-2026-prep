import { WindowCard } from "./WindowCard";
import { FocusOverlay } from "./FocusOverlay";
import { MinimizedDock } from "./MinimizedDock";
import { Sparkline } from "../ui/Sparkline";
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
import type { CardId, WindowState } from "../../hooks/useWindowManager";
import type { SessionInfo } from "../../lib/types";
import type { LucideIcon } from "lucide-react";
import { Coins, Brain, Cpu, Network, TrendingDown, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { getSessionInfo } from "../../lib/tauri";

interface CardGridProps {
  windows: WindowState[];
  focusedCard: CardId | null;
  onFocus: (id: CardId) => void;
  onUnfocus: () => void;
  onMinimize: (id: CardId) => void;
  onRestore: (id: CardId) => void;
}

interface CardDef {
  id: CardId;
  title: string;
  icon: LucideIcon;
  value: string;
  label: string;
  secondaries: { label: string; value: string }[];
  sparkData: number[];
  detail: React.ReactNode;
}

export function CardGrid({ windows, focusedCard, onFocus, onUnfocus, onMinimize, onRestore }: CardGridProps) {
  const training = useTrainingStatus();
  const metrics = useSystemMetrics();
  const peers = useNetworkPeers();
  const events = useActivityLog(true);
  const { history: earningsHistory, push: pushEarnings } = useMetricHistory();
  const { history: lossHistory, push: pushLoss } = useMetricHistory();
  const { cpu, memory, gpu, pushAll } = useMultiMetricHistory();
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);

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

  const memPercent = metrics.memory_total_mb > 0
    ? ((metrics.memory_used_mb / metrics.memory_total_mb) * 100).toFixed(0)
    : "0";

  const cards: CardDef[] = [
    {
      id: "earnings",
      title: "Earnings",
      icon: Coins,
      value: formatNumber(training.session_earned),
      label: "HLX earned",
      secondaries: [
        { label: "Rate", value: sessionInfo ? `${sessionInfo.earnings_rate_per_hour.toFixed(2)}/hr` : "—" },
        { label: "Total", value: formatNumber(training.total_earned) },
        { label: "Proofs", value: `${training.proofs_generated}` },
      ],
      sparkData: earningsHistory,
      detail: <EarningsDetail training={training} sessionInfo={sessionInfo} earningsHistory={earningsHistory} />,
    },
    {
      id: "training",
      title: "Training",
      icon: Brain,
      value: training.current_round !== null ? `#${training.current_round}` : "—",
      label: training.phase,
      secondaries: [
        { label: "Step", value: training.current_step !== null ? `${training.current_step}/${training.total_steps ?? "?"}` : "—" },
        { label: "Progress", value: `${training.progress.toFixed(0)}%` },
        { label: "Model", value: training.model_name || "—" },
      ],
      sparkData: lossHistory,
      detail: <TrainingDetail training={training} lossHistory={lossHistory} />,
    },
    {
      id: "resources",
      title: "Resources",
      icon: Cpu,
      value: `${metrics.cpu_usage_percent.toFixed(0)}%`,
      label: "CPU",
      secondaries: [
        { label: "Mem", value: `${formatBytes(metrics.memory_used_mb)} (${memPercent}%)` },
        { label: "GPU", value: metrics.gpu_usage_percent !== null ? `${metrics.gpu_usage_percent.toFixed(0)}%` : "N/A" },
      ],
      sparkData: cpu,
      detail: <ResourcesDetail metrics={metrics} cpuHistory={cpu} memHistory={memory} gpuHistory={gpu} />,
    },
    {
      id: "network",
      title: "Network",
      icon: Network,
      value: `${peers.length}`,
      label: "peers connected",
      secondaries: [
        { label: "Healthy", value: `${peers.filter((p) => p.last_seen_secs_ago < 30).length}` },
        { label: "Avg rep", value: peers.length > 0 ? (peers.reduce((s, p) => s + p.reputation, 0) / peers.length).toFixed(0) : "—" },
      ],
      sparkData: [],
      detail: <NetworkDetail peers={peers} />,
    },
    {
      id: "loss",
      title: "Loss",
      icon: TrendingDown,
      value: training.current_loss?.toFixed(4) ?? "—",
      label: "current loss",
      secondaries: [
        { label: "Round", value: training.current_round !== null ? `#${training.current_round}` : "—" },
        { label: "Trend", value: lossHistory.length >= 2 ? (lossHistory[lossHistory.length - 1] <= lossHistory[lossHistory.length - 2] ? "decreasing" : "increasing") : "—" },
      ],
      sparkData: lossHistory,
      detail: <LossDetail training={training} lossHistory={lossHistory} />,
    },
    {
      id: "rounds",
      title: "Rounds",
      icon: RotateCcw,
      value: `${training.rounds_completed}`,
      label: "completed",
      secondaries: [
        { label: "Proofs", value: `${training.proofs_generated}` },
        { label: "Active", value: training.active ? "Yes" : "No" },
      ],
      sparkData: [],
      detail: <RoundsDetail training={training} />,
    },
  ];

  const cardMeta = Object.fromEntries(
    cards.map((c) => [c.id, { title: c.title, icon: c.icon }])
  ) as Record<CardId, { title: string; icon: LucideIcon }>;

  const focusedDef = focusedCard ? cards.find((c) => c.id === focusedCard) : null;
  const minimizedCards = windows.filter((w) => w.status === "minimized");

  return (
    <div className="flex-1 flex flex-col gap-3 px-8 pb-4 overflow-hidden">
      <div className="grid grid-cols-3 gap-4 flex-1 min-h-0">
        {cards.map((card) => {
          const ws = windows.find((w) => w.id === card.id);
          const status = ws?.status ?? "normal";
          return (
            <WindowCard
              key={card.id}
              id={card.id}
              status={status}
              title={card.title}
              icon={card.icon}
              onFocus={onFocus}
              onMinimize={onMinimize}
            >
              <div className="flex flex-col gap-3">
                <div className="flex items-baseline gap-2">
                  <span className="text-metric-window text-text-primary">{card.value}</span>
                  <span className="text-label text-text-tertiary">{card.label}</span>
                </div>
                <div className="flex items-center gap-4">
                  {card.secondaries.map((s) => (
                    <div key={s.label} className="flex items-center gap-1.5">
                      <span className="text-label-sm text-text-tertiary">{s.label}</span>
                      <span className="font-mono text-xs text-text-secondary tabular-nums">{s.value}</span>
                    </div>
                  ))}
                </div>
                {card.sparkData.length >= 2 && (
                  <Sparkline data={card.sparkData} width={180} height={32} />
                )}
              </div>
            </WindowCard>
          );
        })}
      </div>

      <MinimizedDock cards={minimizedCards} cardMeta={cardMeta} onRestore={onRestore} />

      <ActivityFeed events={events} />

      {focusedDef && (
        <FocusOverlay
          open
          onClose={onUnfocus}
          title={focusedDef.title}
          icon={focusedDef.icon}
          value={focusedDef.value}
          label={focusedDef.label}
        >
          {focusedDef.detail}
        </FocusOverlay>
      )}
    </div>
  );
}
