import type { TrainingStatus } from "../../../lib/types";

interface RoundsDetailProps {
  training: TrainingStatus;
}

export function RoundsDetail({ training }: RoundsDetailProps) {
  return (
    <div className="space-y-4">
      {/* Summary stats */}
      <div className="grid grid-cols-3 gap-4">
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Completed</p>
          <p className="text-metric-lg text-text-primary">{training.rounds_completed}</p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Proofs generated</p>
          <p className="text-metric-lg text-text-primary">{training.proofs_generated}</p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Current round</p>
          <p className="text-metric-lg text-accent">
            #{training.current_round ?? "—"}
          </p>
        </div>
      </div>

      {/* Status */}
      <div className="pt-3 border-t border-border-subtle">
        <div className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full"
            style={{
              backgroundColor: training.active ? "#22C55E" : "#505050",
            }}
          />
          <span className="text-body text-text-secondary">
            {training.active ? `Round #${training.current_round} — ${training.phase}` : "Idle"}
          </span>
        </div>
      </div>
    </div>
  );
}
