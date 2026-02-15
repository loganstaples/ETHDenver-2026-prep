import type { TrainingStatus } from "../../../lib/types";
import { Sparkline } from "../../ui/Sparkline";

interface LossDetailProps {
  training: TrainingStatus;
  lossHistory: number[];
}

export function LossDetail({ training, lossHistory }: LossDetailProps) {
  const recentLoss = lossHistory.slice(-10);
  const avgRecent = recentLoss.length > 0
    ? recentLoss.reduce((a, b) => a + b, 0) / recentLoss.length
    : 0;
  const trend = recentLoss.length >= 2
    ? recentLoss[recentLoss.length - 1] - recentLoss[0]
    : 0;

  return (
    <div className="space-y-5">
      {/* Loss curve — full width */}
      <div>
        <p className="text-label text-text-tertiary mb-2">Loss over training</p>
        <Sparkline data={lossHistory} width={460} height={100} color="#EAB308" fillOpacity={0.05} />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 pt-2 border-t border-border-subtle">
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Current</p>
          <p className="text-metric-md text-text-primary">
            {training.current_loss?.toFixed(4) ?? "—"}
          </p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Avg (last 10)</p>
          <p className="text-metric-md text-text-secondary">{avgRecent.toFixed(4)}</p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Trend</p>
          <p className={`text-metric-md ${trend <= 0 ? "text-status-healthy" : "text-status-warning"}`}>
            {trend <= 0 ? "↓" : "↑"} {Math.abs(trend).toFixed(4)}
          </p>
        </div>
      </div>
    </div>
  );
}
