'use client';

import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { formatEther } from 'viem';
import { Loader2, Server, ChevronRight } from 'lucide-react';
import { useNodes, useNetworkHealth, type WorkerNode } from '@/hooks/useNodes';
import { cn } from '@/lib/utils';

// ============================================================================
// Types
// ============================================================================

type FilterType = 'all' | 'compute' | 'aggregator' | 'verifier';

// ============================================================================
// Helpers
// ============================================================================

const STATUS_STYLES: Record<WorkerNode['status'], { label: string; dot: string; text: string }> = {
  active: { label: 'Active', dot: 'bg-green-400', text: 'text-green-400' },
  idle: { label: 'Idle', dot: 'bg-yellow-400', text: 'text-yellow-400' },
  offline: { label: 'Offline', dot: 'bg-helix-dim', text: 'text-helix-muted' },
  syncing: { label: 'Syncing', dot: 'bg-blue-400', text: 'text-blue-400' },
  proving: { label: 'Proving', dot: 'bg-purple-400', text: 'text-purple-400' },
  training: { label: 'Training', dot: 'bg-cyan-400', text: 'text-cyan-400' },
};

function formatAddr(address: string) {
  if (address.length < 12) return address;
  return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

// ============================================================================
// Page
// ============================================================================

export default function NetworkPage() {
  const { filteredNodes, networkStats, isLoading, updateFilter } = useNodes();
  const { healthScore } = useNetworkHealth();
  const [activeFilter, setActiveFilter] = useState<FilterType>('all');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const handleFilter = (f: FilterType) => {
    setActiveFilter(f);
    updateFilter({ type: f === 'all' ? undefined : f });
    setExpandedId(null);
  };

  const filters: readonly { key: FilterType; label: string }[] = [
    { key: 'all', label: 'All' },
    { key: 'compute', label: 'Compute' },
    { key: 'aggregator', label: 'Aggregator' },
    { key: 'verifier', label: 'Verifier' },
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 size={24} className="animate-spin text-helix-muted" />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="space-y-6"
    >
      {/* ── Header ─────────────────────────────────────────────── */}
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-4xl font-semibold tracking-tight text-white">Network</h1>
          <p className="text-base text-helix-muted mt-1">
            {networkStats.activeNodes} active · {networkStats.totalNodes} total workers
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.06]">
          <span className={cn(
            'w-2 h-2 rounded-full',
            networkStats.activeNodes > 0 ? 'bg-green-400 animate-pulse' : 'bg-helix-dim',
          )} />
          <span className="text-sm text-helix-text2">
            {networkStats.activeNodes > 0 ? 'Online' : 'No workers'}
          </span>
        </div>
      </div>

      {/* ── Stats row ──────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { label: 'Active Nodes', value: networkStats.activeNodes },
          { label: 'Health Score', value: healthScore },
          { label: 'Avg Reputation', value: Math.round(networkStats.averageReputation) },
          { label: 'Total Proofs', value: networkStats.totalProofs },
        ].map((stat) => (
          <div
            key={stat.label}
            className="rounded-2xl bg-helix-surface border border-helix-border px-5 py-4"
          >
            <p className="text-xs text-helix-dim uppercase tracking-wider">{stat.label}</p>
            <p className="text-2xl font-semibold text-white tabular-nums font-mono mt-1">
              {stat.value}
            </p>
          </div>
        ))}
      </div>

      {/* ── Filter tabs ────────────────────────────────────────── */}
      <div className="grid grid-cols-4 gap-2">
        {filters.map((f) => (
          <button
            key={f.key}
            type="button"
            onClick={() => handleFilter(f.key)}
            className={cn(
              'py-2.5 rounded-xl text-base font-medium transition-all',
              activeFilter === f.key
                ? 'bg-white text-black shadow-lg shadow-white/5'
                : 'bg-helix-surface border border-helix-border text-helix-muted hover:text-white hover:border-helix-border2',
            )}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* ── Worker list ────────────────────────────────────────── */}
      {filteredNodes.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 rounded-2xl bg-helix-surface border border-helix-border">
          <Server size={24} className="text-helix-dim mb-3" />
          <p className="text-base text-helix-text2">No workers found</p>
          <p className="text-sm text-helix-muted mt-1">
            {activeFilter !== 'all'
              ? 'Try a different filter'
              : 'No nodes have joined the network yet'}
          </p>
        </div>
      ) : (
        <div className="rounded-2xl bg-helix-surface border border-helix-border overflow-hidden">
          {/* Table header */}
          <div className="grid grid-cols-[minmax(0,1fr)_70px_50px_50px_50px_65px_50px_20px] gap-3 px-6 py-2.5 border-b border-helix-border bg-helix-bg/30">
            <span className="text-[10px] text-helix-muted uppercase tracking-wider">Worker</span>
            <span className="text-[10px] text-helix-muted uppercase tracking-wider text-right">Status</span>
            <span className="text-[10px] text-helix-muted uppercase tracking-wider text-right">Rep</span>
            <span className="text-[10px] text-helix-muted uppercase tracking-wider text-right">CPU</span>
            <span className="text-[10px] text-helix-muted uppercase tracking-wider text-right">Mem</span>
            <span className="text-[10px] text-helix-muted uppercase tracking-wider text-right">Stake</span>
            <span className="text-[10px] text-helix-muted uppercase tracking-wider text-right">Proofs</span>
            <span />
          </div>

          {filteredNodes.map((node, i) => {
            const status = STATUS_STYLES[node.status] ?? STATUS_STYLES.offline;
            const isExpanded = expandedId === node.id;
            const isLast = i === filteredNodes.length - 1;

            return (
              <div key={node.id}>
                <button
                  type="button"
                  onClick={() => setExpandedId(isExpanded ? null : node.id)}
                  className={cn(
                    'w-full grid grid-cols-[minmax(0,1fr)_70px_50px_50px_50px_65px_50px_20px] gap-3 items-center px-6 py-3.5 text-left transition-colors',
                    !isLast && !isExpanded && 'border-b border-helix-border/50',
                    isExpanded && 'border-b border-helix-border/50 bg-white/[0.01]',
                    'hover:bg-white/[0.015]',
                  )}
                >
                  {/* Worker: dot + address + role badge */}
                  <div className="flex items-center gap-3 min-w-0">
                    <div className={cn('w-2 h-2 rounded-full shrink-0', status.dot)} />
                    <span className="font-mono text-sm text-white truncate">
                      {formatAddr(node.address)}
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-white/[0.04] text-helix-text2 capitalize shrink-0">
                      {node.type}
                    </span>
                  </div>

                  {/* Status */}
                  <span className={cn('text-xs font-medium text-right truncate', status.text)}>
                    {status.label}
                  </span>

                  {/* Rep */}
                  <span className="font-mono text-xs text-white tabular-nums text-right">
                    {Math.round(node.reputation)}
                  </span>

                  {/* CPU */}
                  <span className="font-mono text-xs text-white tabular-nums text-right">
                    {Math.round(node.metrics.cpu)}%
                  </span>

                  {/* Mem */}
                  <span className="font-mono text-xs text-helix-text2 tabular-nums text-right">
                    {Math.round(node.metrics.memory)}%
                  </span>

                  {/* Stake */}
                  <span className="font-mono text-xs text-white tabular-nums text-right">
                    {Number(formatEther(node.stakedAmount)).toFixed(1)}
                  </span>

                  {/* Proofs */}
                  <span className="font-mono text-xs text-white tabular-nums text-right">
                    {node.proofsSubmitted}
                  </span>

                  {/* Chevron */}
                  <ChevronRight
                    size={12}
                    className={cn(
                      'text-helix-dim transition-transform duration-200',
                      isExpanded && 'rotate-90',
                    )}
                  />
                </button>

                {/* ── Expanded detail ─────────────────────── */}
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2, ease: 'easeOut' }}
                      className="overflow-hidden"
                    >
                      <div className={cn(
                        'px-6 py-5 bg-helix-bg/50',
                        !isLast && 'border-b border-helix-border/50',
                      )}>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                          {/* System Metrics */}
                          <div>
                            <h4 className="text-[10px] text-helix-muted uppercase tracking-wider mb-3">
                              System Metrics
                            </h4>
                            <div className="space-y-2.5">
                              <MetricBar label="CPU" value={node.metrics.cpu} />
                              <MetricBar label="Memory" value={node.metrics.memory} />
                              {node.metrics.gpu !== undefined && (
                                <MetricBar label="GPU" value={node.metrics.gpu} />
                              )}
                              {node.metrics.gpuMemory !== undefined && (
                                <MetricBar label="GPU Mem" value={node.metrics.gpuMemory} />
                              )}
                              {node.metrics.temperature !== undefined && (
                                <DetailRow
                                  label="Temp"
                                  value={`${node.metrics.temperature.toFixed(0)}°C`}
                                />
                              )}
                            </div>
                          </div>

                          {/* Capabilities */}
                          <div>
                            <h4 className="text-[10px] text-helix-muted uppercase tracking-wider mb-3">
                              Capabilities
                            </h4>
                            <div className="space-y-2">
                              <DetailRow label="Role" value={node.type} />
                              {node.capabilities.gpuModel && (
                                <DetailRow label="GPU Model" value={node.capabilities.gpuModel} />
                              )}
                              <DetailRow
                                label="GPU Memory"
                                value={`${node.capabilities.gpuMemoryMb} MB`}
                              />
                              <DetailRow
                                label="Max Batch"
                                value={String(node.capabilities.maxBatchSize)}
                              />
                              <div className="flex items-center gap-1.5 mt-2">
                                {node.capabilities.canTrain && (
                                  <span className="text-[10px] px-2 py-0.5 rounded bg-white/5 text-helix-text2">
                                    Train
                                  </span>
                                )}
                                {node.capabilities.canAggregate && (
                                  <span className="text-[10px] px-2 py-0.5 rounded bg-white/5 text-helix-text2">
                                    Aggregate
                                  </span>
                                )}
                                {node.capabilities.canProve && (
                                  <span className="text-[10px] px-2 py-0.5 rounded bg-white/5 text-helix-text2">
                                    Prove
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>

                          {/* Performance & Stake */}
                          <div>
                            <h4 className="text-[10px] text-helix-muted uppercase tracking-wider mb-3">
                              Performance
                            </h4>
                            <div className="space-y-2">
                              <DetailRow
                                label="Proofs Submitted"
                                value={String(node.proofsSubmitted)}
                              />
                              <DetailRow
                                label="Proofs Verified"
                                value={String(node.proofsVerified)}
                              />
                              <DetailRow
                                label="Proofs Failed"
                                value={String(node.proofsFailed)}
                              />
                              <DetailRow
                                label="Success Rate"
                                value={
                                  node.proofsSubmitted > 0
                                    ? `${((node.proofsVerified / node.proofsSubmitted) * 100).toFixed(1)}%`
                                    : '—'
                                }
                              />
                            </div>

                            <h4 className="text-[10px] text-helix-muted uppercase tracking-wider mt-4 mb-3">
                              Stake & Reputation
                            </h4>
                            <div className="space-y-2">
                              <DetailRow
                                label="Staked"
                                value={`${Number(formatEther(node.stakedAmount)).toFixed(2)} ADI`}
                              />
                              <DetailRow
                                label="Earnings"
                                value={`${Number(formatEther(node.earningsTotal)).toFixed(2)} ADI`}
                              />
                              <DetailRow
                                label="Reputation"
                                value={node.reputation.toFixed(1)}
                              />
                              {node.slashed && (
                                <span className="inline-flex text-[10px] px-2 py-0.5 rounded bg-red-500/10 text-red-400 mt-1">
                                  Slashed
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            );
          })}
        </div>
      )}
    </motion.div>
  );
}

// ============================================================================
// Inline sub-components
// ============================================================================

function MetricBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-helix-text2 w-16 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-helix-border overflow-hidden">
        <div
          className="h-full rounded-full bg-white"
          style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        />
      </div>
      <span className="font-mono text-xs text-helix-muted tabular-nums w-10 text-right shrink-0">
        {Math.round(value)}%
      </span>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-helix-text2">{label}</span>
      <span className="font-mono text-xs text-white">{value}</span>
    </div>
  );
}
