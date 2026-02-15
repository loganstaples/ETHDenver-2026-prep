import type { TrainingStatus } from "../../../lib/types";
import { Sparkline } from "../../ui/Sparkline";

const PHASES = ["Assigned", "Training", "Proving", "Submitted", "Verified"];

interface TrainingDetailProps {
  training: TrainingStatus;
  lossHistory: number[];
}

export function TrainingDetail({ training, lossHistory }: TrainingDetailProps) {
  const phaseIndex = PHASES.findIndex(
    (p) => p.toLowerCase() === training.phase.toLowerCase()
  );

  return (
    <div className="space-y-5">
      {/* Phase timeline */}
      <div>
        <p className="text-label text-text-tertiary mb-3">Phase</p>
        <div className="flex items-center gap-2">
          {PHASES.map((p, i) => (
            <div key={p} className="flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full"
                style={{
                  backgroundColor:
                    i <= phaseIndex ? "#34D399" : "#252525",
                }}
              />
              <span
                className="text-xs"
                style={{
                  color: i <= phaseIndex ? "#A0A0A0" : "#505050",
                }}
              >
                {p}
              </span>
              {i < PHASES.length - 1 && (
                <div
                  className="w-4 h-px"
                  style={{
                    backgroundColor: i < phaseIndex ? "#34D399" : "#252525",
                  }}
                />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Progress */}
      <div>
        <div className="flex justify-between mb-1.5">
          <p className="text-label-sm text-text-tertiary">Progress</p>
          <p className="text-metric-sm text-text-secondary">
            {training.current_step ?? 0}/{training.total_steps ?? 0}
          </p>
        </div>
        <div className="h-1 bg-border-subtle rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${training.progress * 100}%`,
              backgroundColor: "#34D399",
            }}
          />
        </div>
      </div>

      {/* Loss curve */}
      <div>
        <p className="text-label text-text-tertiary mb-2">Loss curve</p>
        <Sparkline data={lossHistory} width={460} height={80} color="#EAB308" />
      </div>

      {/* Info grid */}
      <div className="grid grid-cols-2 gap-4 pt-2 border-t border-border-subtle">
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Model</p>
          <p className="text-body text-text-secondary">{training.model_name || "—"}</p>
        </div>
        <div>
          <p className="text-label-sm text-text-tertiary mb-1">Current loss</p>
          <p className="text-metric-sm text-text-secondary">
            {training.current_loss?.toFixed(4) ?? "—"}
          </p>
        </div>
      </div>
    </div>
  );
}
