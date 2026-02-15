'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { useTraining } from '@/hooks/useTraining';
import { useProofs } from '@/hooks/useProofs';
import type { ModelDetails } from '@/hooks/useContractState';

interface ModelCardProps {
  model: ModelDetails;
}

function truncateHash(hash: string, length: number): string {
  if (hash.length <= length) return hash;
  return hash.slice(0, length) + '...';
}

function deriveModelName(ipfsHash: string): string {
  return truncateHash(ipfsHash, 16);
}

function formatMetric(value: number | undefined | null, fallback = '--'): string {
  if (value === undefined || value === null || isNaN(value)) return fallback;
  if (value < 0.0001) return value.toExponential(2);
  if (value < 1) return value.toFixed(4);
  return value.toFixed(2);
}

export function ModelCard({ model }: ModelCardProps) {
  const { metrics, progress, isLoading: trainingLoading } = useTraining({
    modelId: model.id,
    autoRefresh: true,
    refreshInterval: 5000,
  });

  const { stats } = useProofs({
    modelId: Number(model.id),
    maxProofs: 5,
    autoRefresh: false,
  });

  return (
    <Link href={`/models/${model.id.toString()}`} className="block">
      <motion.div
        whileHover={{ y: -2 }}
        transition={{ duration: 0.15 }}
      >
        <Card variant="glass" hover className="space-y-4">
          {/* Header: Name + Status Badge */}
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-sm font-medium text-white truncate">
              {deriveModelName(model.ipfsHash)}
            </h3>
            {model.active ? (
              <Badge variant="pulse">Training</Badge>
            ) : (
              <Badge variant="default">Idle</Badge>
            )}
          </div>

          {/* IPFS Hash */}
          <p className="font-mono text-2xs text-helix-muted truncate">
            {truncateHash(model.ipfsHash, 20)}
          </p>

          {/* Metrics Row */}
          <div className="grid grid-cols-3 gap-3">
            <div>
              <p className="text-2xs text-helix-muted uppercase tracking-wider mb-0.5">
                Loss
              </p>
              <p className="font-mono text-xs text-helix-text">
                {trainingLoading ? '--' : formatMetric(metrics.loss)}
              </p>
            </div>
            <div>
              <p className="text-2xs text-helix-muted uppercase tracking-wider mb-0.5">
                Accuracy
              </p>
              <p className="font-mono text-xs text-helix-text">
                {trainingLoading
                  ? '--'
                  : metrics.accuracy !== undefined
                    ? `${(metrics.accuracy * 100).toFixed(1)}%`
                    : '--'}
              </p>
            </div>
            <div>
              <p className="text-2xs text-helix-muted uppercase tracking-wider mb-0.5">
                Error Bound
              </p>
              <p className="font-mono text-xs text-helix-text">
                {formatMetric(model.errorBoundFormatted)}
              </p>
            </div>
          </div>

          {/* Progress + Workers */}
          <div className="space-y-2">
            <ProgressBar value={progress.overallProgress} size="sm" />
            <div className="flex items-center justify-between">
              <span className="text-2xs text-helix-dim font-mono">
                {progress.overallProgress.toFixed(0)}% complete
              </span>
              <span className="text-2xs text-helix-dim font-mono">
                {stats.verified} proofs
              </span>
            </div>
          </div>
        </Card>
      </motion.div>
    </Link>
  );
}
