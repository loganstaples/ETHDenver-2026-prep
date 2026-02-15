'use client';

import { useMemo } from 'react';
import { useTrainingRounds } from '@/hooks/useTraining';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Table } from '@/components/ui/Table';
import { Skeleton } from '@/components/ui/Skeleton';
import { EmptyState } from '@/components/ui/EmptyState';
import { RotateCcw } from 'lucide-react';

interface RoundHistoryProps {
  modelId: bigint;
}

const STATUS_VARIANT: Record<string, 'default' | 'outline' | 'pulse'> = {
  completed: 'default',
  in_progress: 'pulse',
  aggregating: 'pulse',
  proving: 'pulse',
  failed: 'outline',
  pending: 'outline',
};

function formatRoundDuration(startedAt: number, completedAt?: number): string {
  if (!completedAt) return '--';
  const durationMs = completedAt - startedAt;
  if (durationMs < 1000) return `${durationMs}ms`;
  const seconds = Math.floor(durationMs / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainSeconds = seconds % 60;
  return `${minutes}m ${remainSeconds}s`;
}

function formatGas(gas?: bigint): string {
  if (!gas) return '--';
  const num = Number(gas);
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(0)}K`;
  return num.toString();
}

function truncateHash(hash?: string): string {
  if (!hash) return '--';
  return `${hash.slice(0, 6)}...${hash.slice(-4)}`;
}

export function RoundHistory({ modelId }: RoundHistoryProps) {
  const {
    rounds,
    successRate,
    averageRoundTime,
    totalRounds,
    isLoading,
  } = useTrainingRounds(modelId);

  const sortedRounds = useMemo(() => {
    return [...rounds].sort((a, b) => {
      if (a.id > b.id) return -1;
      if (a.id < b.id) return 1;
      return 0;
    });
  }, [rounds]);

  const tableData = useMemo(() => {
    return sortedRounds.map((round) => ({
      round: (
        <span className="font-mono text-sm text-helix-text2">
          #{round.id.toString()}
        </span>
      ),
      status: (
        <Badge variant={STATUS_VARIANT[round.status] ?? 'default'}>
          {round.status.replace('_', ' ').toUpperCase()}
        </Badge>
      ),
      participants: (
        <span className="font-mono text-sm text-helix-text2">
          {round.participants.length}
        </span>
      ),
      duration: (
        <span className="font-mono text-2xs text-helix-text2">
          {formatRoundDuration(round.startedAt, round.completedAt)}
        </span>
      ),
      errorBound: (
        <span className="font-mono text-2xs text-helix-text2">
          {round.errorBound.toExponential(2)}
        </span>
      ),
      gas: (
        <span className="font-mono text-2xs text-helix-text2">
          {formatGas(round.gasUsed)}
        </span>
      ),
      tx: round.transactionHash ? (
        <a
          href={`https://etherscan.io/tx/${round.transactionHash}`}
          target="_blank"
          rel="noopener noreferrer"
          className="font-mono text-2xs text-helix-text2 hover:text-helix-text transition-colors underline decoration-helix-border"
        >
          {truncateHash(round.transactionHash)}
        </a>
      ) : (
        <span className="text-2xs text-helix-dim">--</span>
      ),
    }));
  }, [sortedRounds]);

  if (isLoading) {
    return (
      <Card>
        <Skeleton height="24px" width="160px" className="mb-4" />
        <Skeleton height="200px" />
      </Card>
    );
  }

  return (
    <Card>
      <div className="flex items-baseline justify-between mb-4">
        <p className="text-2xs font-mono uppercase tracking-wider text-helix-muted">
          Training Rounds
        </p>
        <div className="flex items-center gap-4">
          <span className="text-2xs text-helix-dim">
            {totalRounds} rounds
          </span>
          <span className="text-2xs text-helix-dim">
            {(successRate * 100).toFixed(0)}% success
          </span>
          {averageRoundTime > 0 && (
            <span className="text-2xs text-helix-dim">
              avg {formatRoundDuration(0, averageRoundTime)}
            </span>
          )}
        </div>
      </div>

      {tableData.length > 0 ? (
        <Table
          columns={[
            { key: 'round', label: 'Round', mono: true },
            { key: 'status', label: 'Status' },
            { key: 'participants', label: 'Participants', align: 'right' },
            { key: 'duration', label: 'Duration', align: 'right' },
            { key: 'errorBound', label: 'Error Bound', align: 'right' },
            { key: 'gas', label: 'Gas Used', align: 'right' },
            { key: 'tx', label: 'Tx', align: 'right' },
          ]}
          data={tableData}
        />
      ) : (
        <EmptyState
          icon={<RotateCcw size={24} />}
          title="No training rounds"
          description="Training rounds will appear here once the model begins training."
        />
      )}
    </Card>
  );
}
