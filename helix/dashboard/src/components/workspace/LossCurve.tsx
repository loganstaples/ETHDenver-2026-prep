'use client';

import { useMemo } from 'react';
import { useLossCurve } from '@/hooks/useTraining';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Chart } from '@/components/ui/Chart';
import { Skeleton } from '@/components/ui/Skeleton';

interface LossCurveProps {
  modelId: bigint;
}

const TREND_VARIANT: Record<string, 'default' | 'outline' | 'pulse'> = {
  improving: 'default',
  stable: 'outline',
  degrading: 'outline',
};

export function LossCurve({ modelId }: LossCurveProps) {
  const {
    lossHistory,
    smoothedLoss,
    accuracyHistory,
    trend,
    convergenceRate,
    isLoading,
  } = useLossCurve(modelId);

  const chartData = useMemo(() => {
    return smoothedLoss.map((point) => {
      const accuracyPoint = accuracyHistory.find((a) => a.epoch === point.epoch);
      return {
        epoch: point.epoch,
        loss: Number(point.loss.toFixed(4)),
        accuracy: accuracyPoint ? Number((accuracyPoint.accuracy * 100).toFixed(1)) : undefined,
      };
    });
  }, [smoothedLoss, accuracyHistory]);

  const latestLoss = lossHistory.length > 0
    ? lossHistory[lossHistory.length - 1].loss
    : null;

  if (isLoading) {
    return (
      <Card>
        <Skeleton height="24px" width="160px" className="mb-4" />
        <Skeleton height="400px" />
      </Card>
    );
  }

  if (chartData.length === 0) {
    return (
      <Card>
        <p className="text-sm text-helix-muted mb-3">
          Loss & Accuracy
        </p>
        <p className="text-sm text-helix-muted py-8 text-center">
          No training data available yet
        </p>
      </Card>
    );
  }

  return (
    <Card>
      <p className="text-sm text-helix-muted mb-4">
        Loss & Accuracy
      </p>

      <Chart
        data={chartData}
        xKey="epoch"
        yKeys={[
          { key: 'loss', label: 'Loss' },
          { key: 'accuracy', label: 'Accuracy (%)' },
        ]}
        height={400}
        type="line"
      />

      {/* Stats Row */}
      <div className="flex items-center gap-6 mt-4 pt-4 border-t border-helix-border">
        <div className="flex items-center gap-2">
          <span className="text-2xs text-helix-muted">Trend</span>
          <Badge variant={TREND_VARIANT[trend] ?? 'outline'}>
            {trend.toUpperCase()}
          </Badge>
        </div>
        <div>
          <span className="text-2xs text-helix-muted">Convergence Rate </span>
          <span className="text-2xs font-mono text-helix-text2">
            {(convergenceRate * 100).toFixed(1)}%
          </span>
        </div>
        {latestLoss !== null && (
          <div>
            <span className="text-2xs text-helix-muted">Latest Loss </span>
            <span className="text-2xs font-mono text-helix-text2">
              {latestLoss.toFixed(4)}
            </span>
          </div>
        )}
      </div>
    </Card>
  );
}
