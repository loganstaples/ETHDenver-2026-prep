import { Card, CardHeader } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { ProgressBar } from "../ui/ProgressBar";
import type { TrainingStatus } from "../../lib/types";

interface TrainingOverviewProps {
  training: TrainingStatus;
}

export function TrainingOverview({ training }: TrainingOverviewProps) {
  return (
    <Card>
      <CardHeader
        title="Current Training"
        action={
          training.active ? (
            <Badge variant="active">In Progress</Badge>
          ) : (
            <Badge variant="idle">Idle</Badge>
          )
        }
      />
      {training.active && training.current_round !== null ? (
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-xs text-text-tertiary">Round</span>
            <span className="text-sm text-text-primary font-mono tabular-nums">
              #{training.current_round}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-text-tertiary">Phase</span>
            <span className="text-sm text-text-primary">{training.phase}</span>
          </div>
          <div className="space-y-1.5">
            <div className="flex justify-between items-center">
              <span className="text-xs text-text-tertiary">Progress</span>
              <span className="text-xs text-text-primary tabular-nums">
                {Math.round(training.progress * 100)}%
              </span>
            </div>
            <ProgressBar value={training.progress * 100} size="md" />
          </div>
        </div>
      ) : (
        <div className="text-center py-6">
          <p className="text-sm text-text-tertiary">
            Waiting for training round...
          </p>
          <p className="text-xs text-text-disabled mt-1">
            Start mining to participate
          </p>
        </div>
      )}
    </Card>
  );
}
