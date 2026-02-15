import type { TrainingStatus } from "../../lib/types";
import { AnimatedNumber, AnimatedInteger } from "../ui/AnimatedNumber";
import { PanelHeader } from "../ui/Panel";

interface EarningsPanelProps {
  training: TrainingStatus;
}

export function EarningsPanel({ training }: EarningsPanelProps) {
  return (
    <div className="h-full bg-bg-panel p-3 flex flex-col">
      <PanelHeader>EARNINGS</PanelHeader>

      {/* Hero number */}
      <div className="flex items-baseline gap-2 mb-3">
        <AnimatedNumber
          value={training.total_earned}
          decimals={2}
          className="text-hero text-text-bright"
        />
        <span className="text-metric-sm text-text-tertiary">HLX</span>
      </div>

      {/* Session earned */}
      <div className="flex flex-col gap-1.5 mb-3">
        <div className="flex items-baseline gap-1.5">
          <AnimatedNumber
            value={training.session_earned}
            decimals={2}
            className="text-metric-md text-status-active"
          />
          <span className="text-body text-text-tertiary">HLX this session</span>
        </div>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Footer metrics */}
      <div className="flex items-center gap-4 pt-2 border-t border-border-grid">
        <div className="flex items-center gap-1.5">
          <span className="text-label text-text-tertiary">ROUNDS</span>
          <AnimatedInteger
            value={training.rounds_completed}
            className="text-metric-sm text-text-primary"
          />
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-label text-text-tertiary">PROOFS</span>
          <AnimatedInteger
            value={training.proofs_generated}
            className="text-metric-sm text-text-primary"
          />
        </div>
      </div>
    </div>
  );
}
