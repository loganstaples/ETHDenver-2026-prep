'use client';

import { useMemo } from 'react';
import { Shield, Clock, CheckCircle, Zap } from 'lucide-react';
import { StatCard } from '@/components/ui/StatCard';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { Table } from '@/components/ui/Table';
import { TimeAgo } from '@/components/ui/TimeAgo';
import { Skeleton } from '@/components/ui/Skeleton';
import { useProofs } from '@/hooks/useProofs';
import { cn } from '@/lib/utils';

interface ProofPanelProps {
  modelId: bigint;
}

const STATUS_STYLES: Record<string, string> = {
  verified: 'bg-white/[0.08] text-white',
  pending: 'bg-white/[0.04] text-helix-text2',
  generating: 'bg-white/[0.04] text-helix-muted',
  failed: 'bg-white/[0.04] text-helix-dim',
  challenged: 'bg-white/[0.06] text-helix-text2',
};

const STAGE_STYLES: Record<string, string> = {
  witness: 'bg-white/[0.04] text-helix-muted',
  setup: 'bg-white/[0.06] text-helix-text2',
  proving: 'bg-white/[0.08] text-white',
  verifying: 'bg-white/[0.08] text-white',
  complete: 'bg-white/[0.10] text-white',
  failed: 'bg-white/[0.04] text-helix-dim',
};

function truncateHash(hash: string): string {
  if (hash.length <= 14) return hash;
  return `${hash.slice(0, 6)}...${hash.slice(-4)}`;
}

function formatScientific(value: number): string {
  if (value === 0) return '0';
  if (value < 0.0001) return value.toExponential(2);
  return value.toFixed(6);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatEta(seconds: number): string {
  if (seconds <= 0) return '--';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

export function ProofPanel({ modelId }: ProofPanelProps) {
  const {
    proofs,
    stats,
    activeGenerations,
    isLoading,
  } = useProofs({ modelId: Number(modelId) });

  const tableColumns = useMemo(
    () => [
      { key: 'hash', label: 'Hash', mono: true },
      { key: 'type', label: 'Type' },
      { key: 'status', label: 'Status' },
      { key: 'errorBound', label: 'Error Bound', align: 'right' as const, mono: true },
      { key: 'size', label: 'Size', align: 'right' as const },
      { key: 'time', label: 'Time', align: 'right' as const },
      { key: 'gas', label: 'Gas', align: 'right' as const, mono: true },
    ],
    [],
  );

  const tableData = useMemo(
    () =>
      proofs.map((proof) => ({
        hash: (
          <span className="font-mono text-2xs">{truncateHash(proof.hash)}</span>
        ),
        type: (
          <Badge className="bg-white/[0.06] text-helix-muted">
            {proof.type}
          </Badge>
        ),
        status: (
          <Badge className={cn(STATUS_STYLES[proof.status])}>
            {proof.status}
          </Badge>
        ),
        errorBound: (
          <span className="font-mono text-2xs text-helix-text2">
            {formatScientific(proof.errorBound)}
          </span>
        ),
        size: (
          <span className="text-2xs text-helix-text2">
            {formatBytes(proof.size)}
          </span>
        ),
        time: <TimeAgo timestamp={proof.createdAt} />,
        gas: proof.gasUsed ? (
          <span className="font-mono text-2xs text-helix-text2">
            {Number(proof.gasUsed).toLocaleString()}
          </span>
        ) : (
          <span className="text-2xs text-helix-dim">--</span>
        ),
      })),
    [proofs],
  );

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} height="5rem" />
          ))}
        </div>
        <Skeleton height="12rem" />
        <Skeleton height="20rem" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stat Cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="Total Proofs"
          value={stats.total}
          icon={<Shield size={16} />}
        />
        <StatCard
          label="Verified"
          value={stats.verified}
          icon={<CheckCircle size={16} />}
        />
        <StatCard
          label="Success Rate"
          value={`${(stats.successRate * 100).toFixed(1)}%`}
          icon={<Zap size={16} />}
        />
        <StatCard
          label="Avg Gen Time"
          value={`${Math.round(stats.averageGenerationTime)}ms`}
          icon={<Clock size={16} />}
        />
      </div>

      {/* Active Generations */}
      <div className="space-y-3">
        <h3 className="text-sm text-helix-muted">
          Active Generations
        </h3>

        {activeGenerations.length === 0 ? (
          <p className="text-2xs text-helix-dim py-4">
            No active generations
          </p>
        ) : (
          <div className="grid gap-3">
            {activeGenerations.map((gen) => (
              <Card
                key={gen.proofId}
                variant="default"
                className="space-y-3"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-2xs text-helix-text2">
                    {truncateHash(gen.proofId)}
                  </span>
                  <Badge className={cn(STAGE_STYLES[gen.stage])}>
                    {gen.stage}
                  </Badge>
                </div>

                <ProgressBar value={gen.progress} size="sm" />

                <div className="flex items-center justify-between">
                  <span className="text-2xs text-helix-muted">
                    {gen.currentStep}
                  </span>
                  <span className="text-2xs text-helix-dim font-mono">
                    ETA {formatEta(gen.estimatedTimeRemaining)}
                  </span>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Proof Explorer */}
      <div className="space-y-3">
        <h3 className="text-sm text-helix-muted">
          Proof Explorer
        </h3>
        <Card variant="default" className="p-0 overflow-hidden">
          <Table columns={tableColumns} data={tableData} />
        </Card>
      </div>
    </div>
  );
}
