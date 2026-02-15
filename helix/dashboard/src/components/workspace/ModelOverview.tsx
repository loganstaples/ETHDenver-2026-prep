'use client';

import { useMemo } from 'react';
import { Activity, Target, Shield, RotateCcw } from 'lucide-react';
import { useTraining } from '@/hooks/useTraining';
import { useProofs } from '@/hooks/useProofs';
import { StatCard } from '@/components/ui/StatCard';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Chart } from '@/components/ui/Chart';
import { Table } from '@/components/ui/Table';
import { TimeAgo } from '@/components/ui/TimeAgo';
import { Skeleton } from '@/components/ui/Skeleton';

interface ModelOverviewProps {
  modelId: bigint;
}

const ROUND_STATUS_VARIANT: Record<string, 'default' | 'outline' | 'pulse'> = {
  completed: 'default',
  in_progress: 'pulse',
  aggregating: 'pulse',
  proving: 'pulse',
  failed: 'outline',
  pending: 'outline',
};

const PROOF_STATUS_VARIANT: Record<string, 'default' | 'outline' | 'pulse'> = {
  verified: 'default',
  pending: 'outline',
  generating: 'pulse',
  failed: 'outline',
  challenged: 'outline',
};

export function ModelOverview({ modelId }: ModelOverviewProps) {
  const {
    metrics,
    lossHistory,
    currentRound,
    progress,
    isLoading: trainingLoading,
  } = useTraining({ modelId });

  const {
    proofs,
    isLoading: proofsLoading,
  } = useProofs({ modelId: Number(modelId), maxProofs: 5 });

  const chartData = useMemo(() => {
    return lossHistory.map((point) => ({
      epoch: point.epoch,
      loss: Number(point.loss.toFixed(4)),
    }));
  }, [lossHistory]);

  const recentProofRows = useMemo(() => {
    return proofs.slice(0, 5).map((proof) => ({
      hash: (
        <span className="font-mono text-2xs text-helix-text2">
          {proof.hash.slice(0, 10)}...{proof.hash.slice(-6)}
        </span>
      ),
      status: (
        <Badge variant={PROOF_STATUS_VARIANT[proof.status] ?? 'default'}>
          {proof.status.toUpperCase()}
        </Badge>
      ),
      time: <TimeAgo timestamp={proof.createdAt} />,
    }));
  }, [proofs]);

  if (trainingLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} height="5rem" />
          ))}
        </div>
        <Skeleton height="200px" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="Loss"
          value={metrics.loss.toFixed(4)}
          delta={metrics.loss < 1 ? 'converging' : undefined}
          deltaDirection={metrics.loss < 1 ? 'down' : 'neutral'}
          icon={<Activity size={16} />}
        />
        <StatCard
          label="Accuracy"
          value={`${(metrics.accuracy * 100).toFixed(1)}%`}
          delta={metrics.accuracy > 0.8 ? 'strong' : undefined}
          deltaDirection={metrics.accuracy > 0.8 ? 'up' : 'neutral'}
          icon={<Target size={16} />}
        />
        <StatCard
          label="Error Bound"
          value={metrics.accumulatedErrorBound.toExponential(2)}
          delta={metrics.accumulatedErrorBound < 0.01 ? 'within budget' : 'exceeding'}
          deltaDirection={metrics.accumulatedErrorBound < 0.01 ? 'neutral' : 'down'}
          icon={<Shield size={16} />}
        />
        <StatCard
          label="Current Round"
          value={progress.currentRound.toString()}
          delta={`Epoch ${progress.currentEpoch}/${progress.totalEpochs}`}
          deltaDirection="neutral"
          icon={<RotateCcw size={16} />}
        />
      </div>

      {/* Mini Loss Curve */}
      {chartData.length > 0 && (
        <Card>
          <p className="text-2xs font-mono uppercase tracking-wider text-helix-muted mb-3">
            Loss Curve
          </p>
          <Chart
            data={chartData}
            xKey="epoch"
            yKeys={[{ key: 'loss', label: 'Loss' }]}
            height={200}
            type="area"
          />
        </Card>
      )}

      <div className="grid grid-cols-2 gap-4">
        {/* Current Round Status */}
        <Card>
          <p className="text-2xs font-mono uppercase tracking-wider text-helix-muted mb-3">
            Current Round
          </p>
          {currentRound ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-helix-text font-mono">
                  Round #{currentRound.id.toString()}
                </span>
                <Badge variant={ROUND_STATUS_VARIANT[currentRound.status] ?? 'default'}>
                  {currentRound.status.replace('_', ' ').toUpperCase()}
                </Badge>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-2xs text-helix-muted">Participants</p>
                  <p className="text-sm text-helix-text2 font-mono">
                    {currentRound.participants.length}
                  </p>
                </div>
                <div>
                  <p className="text-2xs text-helix-muted">Started</p>
                  <TimeAgo timestamp={currentRound.startedAt} className="text-sm" />
                </div>
                <div>
                  <p className="text-2xs text-helix-muted">Error Bound</p>
                  <p className="text-sm text-helix-text2 font-mono">
                    {currentRound.errorBound.toExponential(2)}
                  </p>
                </div>
                <div>
                  <p className="text-2xs text-helix-muted">Deadline</p>
                  <TimeAgo timestamp={currentRound.deadline} className="text-sm" />
                </div>
              </div>
            </div>
          ) : (
            <p className="text-sm text-helix-muted">No active round</p>
          )}
        </Card>

        {/* Recent Proofs */}
        <Card>
          <p className="text-2xs font-mono uppercase tracking-wider text-helix-muted mb-3">
            Recent Proofs
          </p>
          {proofsLoading ? (
            <Skeleton height="120px" />
          ) : recentProofRows.length > 0 ? (
            <Table
              columns={[
                { key: 'hash', label: 'Hash', mono: true },
                { key: 'status', label: 'Status' },
                { key: 'time', label: 'Time', align: 'right' },
              ]}
              data={recentProofRows}
            />
          ) : (
            <p className="text-sm text-helix-muted">No proofs submitted yet</p>
          )}
        </Card>
      </div>
    </div>
  );
}
