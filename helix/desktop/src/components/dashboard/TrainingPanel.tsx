import type { TrainingStatus } from "../../lib/types";
import { PanelHeader } from "../ui/Panel";
import { PhaseDots } from "../ui/PhaseDots";
import { formatNumber } from "../../lib/utils";

interface TrainingPanelProps {
  training: TrainingStatus;
}

export function TrainingPanel({ training }: TrainingPanelProps) {
  const progressPct = Math.round(training.progress * 100);

  return (
    <div className="h-full bg-bg-panel p-3 flex flex-col">
      <PanelHeader>TRAINING STATUS</PanelHeader>

      {training.active ? (
        <>
          {/* Round info */}
          <div className="flex items-baseline gap-2 mb-2">
            <span className="text-metric-lg text-text-bright tabular-nums">
              #{training.current_round}
            </span>
            <span className="text-body text-text-secondary">{training.phase}</span>
          </div>

          {/* Phase dots */}
          <PhaseDots currentPhase={training.phase} className="mb-3" />

          {/* Progress bar */}
          <div className="mb-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-label text-text-tertiary">PROGRESS</span>
              <span className="text-mono-data text-text-secondary tabular-nums">
                {progressPct}%
              </span>
            </div>
            <div className="h-0.5 bg-border-primary w-full">
              <div
                className="h-full bg-text-bright transition-all duration-500"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>

          {/* Model name */}
          {training.model_name && (
            <div className="flex items-center gap-1.5 mb-1">
              <span className="text-label text-text-tertiary">MODEL</span>
              <span className="text-mono-data text-text-secondary">
                {training.model_name}
              </span>
            </div>
          )}

          {/* Step info */}
          {training.current_step != null && training.total_steps != null && (
            <div className="flex items-center gap-1.5 mb-1">
              <span className="text-label text-text-tertiary">STEP</span>
              <span className="text-mono-data text-text-secondary tabular-nums">
                {training.current_step}/{training.total_steps}
              </span>
            </div>
          )}

          {/* Spacer */}
          <div className="flex-1" />

          {/* Bottom metrics */}
          <div className="flex items-center gap-4 pt-2 border-t border-border-grid">
            {training.current_loss != null && (
              <div className="flex items-center gap-1.5">
                <span className="text-label text-text-tertiary">LOSS</span>
                <span className="text-mono-data text-text-primary tabular-nums">
                  {formatNumber(training.current_loss, 4)}
                </span>
              </div>
            )}
          </div>
        </>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <span className="text-body text-text-tertiary">
            Waiting for assignment...
          </span>
        </div>
      )}
    </div>
  );
}
