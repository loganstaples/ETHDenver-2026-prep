'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { formatEther } from 'viem';
import { Server } from 'lucide-react';
import { useNodes, type WorkerNode } from '@/hooks/useNodes';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { AddressDisplay } from '@/components/ui/AddressDisplay';
import { EmptyState } from '@/components/ui/EmptyState';
import { Skeleton } from '@/components/ui/Skeleton';
import { NodeDetailSheet } from './NodeDetailSheet';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const roleBadgeClass: Record<WorkerNode['type'], string> = {
  compute: 'bg-white/[0.08] text-helix-text2',
  aggregator: 'bg-white/[0.08] text-helix-text2',
  verifier: 'bg-white/[0.08] text-helix-text2',
};

function statusVariant(status: WorkerNode['status']): 'pulse' | 'default' | 'outline' {
  switch (status) {
    case 'active':
      return 'pulse';
    case 'offline':
      return 'outline';
    default:
      return 'default';
  }
}

function statusLabel(status: WorkerNode['status']): string {
  return status.charAt(0).toUpperCase() + status.slice(1);
}

// ---------------------------------------------------------------------------
// NodeCard
// ---------------------------------------------------------------------------

interface NodeCardProps {
  node: WorkerNode;
  index: number;
  onClick: () => void;
}

function NodeCard({ node, index, onClick }: NodeCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: index * 0.04 }}
    >
      <Card
        hover
        className="cursor-pointer"
        onClick={onClick}
        layout={false}
        // Disable the Card's own initial/animate so stagger controls it
        initial={false}
        animate={false}
      >
        {/* Header: address + badges */}
        <div className="flex items-start justify-between gap-2">
          <AddressDisplay address={node.address} />
          <div className="flex items-center gap-1.5">
            <Badge className={roleBadgeClass[node.type]}>
              {node.type}
            </Badge>
            <Badge variant={statusVariant(node.status)}>
              {statusLabel(node.status)}
            </Badge>
          </div>
        </div>

        {/* Metrics */}
        <div className="mt-4 space-y-2.5">
          <MetricRow label="CPU" value={node.metrics.cpu} />
          <MetricRow label="MEM" value={node.metrics.memory} />
          {node.metrics.gpu !== undefined && (
            <MetricRow label="GPU" value={node.metrics.gpu} />
          )}
        </div>

        {/* Stats row */}
        <div className="mt-4 pt-3 border-t border-helix-border flex items-center justify-between">
          <Stat label="Proofs" value={node.proofsSubmitted.toString()} />
          <Stat label="Rep." value={node.reputation.toFixed(1)} />
          <Stat
            label="Stake"
            value={`${Number(formatEther(node.stakedAmount)).toFixed(1)} ADI`}
          />
        </div>
      </Card>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Small sub-components
// ---------------------------------------------------------------------------

function MetricRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-2xs text-helix-muted w-7 flex-shrink-0 font-mono">
        {label}
      </span>
      <ProgressBar value={value} size="sm" className="flex-1" />
      <span className="text-2xs text-helix-text2 w-8 text-right font-mono">
        {Math.round(value)}%
      </span>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <p className="text-2xs text-helix-muted">{label}</p>
      <p className="font-mono text-xs text-white mt-0.5">{value}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// NodeGrid
// ---------------------------------------------------------------------------

export function NodeGrid() {
  const { filteredNodes, isLoading } = useNodes();
  const [detailNode, setDetailNode] = useState<WorkerNode | null>(null);

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} height="220px" />
        ))}
      </div>
    );
  }

  if (filteredNodes.length === 0) {
    return (
      <EmptyState
        icon={<Server size={28} strokeWidth={1.5} />}
        title="No nodes found"
        description="There are no nodes matching your current filter."
      />
    );
  }

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-4">
        {filteredNodes.map((node, i) => (
          <NodeCard
            key={node.id}
            node={node}
            index={i}
            onClick={() => setDetailNode(node)}
          />
        ))}
      </div>

      <NodeDetailSheet
        node={detailNode}
        onClose={() => setDetailNode(null)}
      />
    </>
  );
}
