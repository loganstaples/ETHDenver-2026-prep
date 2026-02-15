import type { TrainingStatus, SessionInfo } from "../../../lib/types";
import { Sparkline } from "../../ui/Sparkline";
import { formatNumber } from "../../../lib/utils";

interface EarningsDetailProps {
  training: TrainingStatus;
  sessionInfo: SessionInfo | null;
  earningsHistory: number[];
}

export function EarningsDetail({ training, sessionInfo, earningsHistory }: EarningsDetailProps) {
  return (
    <div className="space-y-5">
      {/* Earnings chart */}
      <div>
        <p className="text-label text-text-tertiary mb-2">Session earnings</p>
        <Sparkline data={earningsHistory} width={460} height={80} />
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-3 gap-4">
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Rate</p>
          <p className="text-metric-md text-text-primary">
            {sessionInfo ? formatNumber(sessionInfo.earnings_rate_per_hour) : "—"}
            <span className="text-text-tertiary text-xs ml-1">HLX/hr</span>
          </p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Rounds</p>
          <p className="text-metric-md text-text-primary">{training.rounds_completed}</p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Proofs</p>
          <p className="text-metric-md text-text-primary">{training.proofs_generated}</p>
        </div>
      </div>

      {/* Total vs session */}
      <div className="flex gap-6 pt-2 border-t border-border-subtle">
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Total earned</p>
          <p className="text-metric-sm text-text-secondary">{formatNumber(training.total_earned, 4)} HLX</p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">This session</p>
          <p className="text-metric-sm text-accent">{formatNumber(training.session_earned, 4)} HLX</p>
        </div>
      </div>
    </div>
  );
}
