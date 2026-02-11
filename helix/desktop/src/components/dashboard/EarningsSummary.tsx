import { Card, CardHeader } from "../ui/Card";
import { MetricDisplay } from "../ui/MetricDisplay";
import { formatNumber } from "../../lib/utils";
import type { TrainingStatus } from "../../lib/types";

interface EarningsSummaryProps {
  training: TrainingStatus;
}

export function EarningsSummary({ training }: EarningsSummaryProps) {
  return (
    <Card>
      <CardHeader title="Earnings" />
      <div className="grid grid-cols-2 gap-4">
        <MetricDisplay
          label="Total Earned"
          value={formatNumber(training.total_earned, 2)}
          unit="HLX"
        />
        <MetricDisplay
          label="This Session"
          value={formatNumber(training.session_earned, 2)}
          unit="HLX"
        />
        <MetricDisplay
          label="Rounds Done"
          value={training.rounds_completed}
        />
        <MetricDisplay
          label="Proofs Generated"
          value={training.proofs_generated}
        />
      </div>
    </Card>
  );
}
