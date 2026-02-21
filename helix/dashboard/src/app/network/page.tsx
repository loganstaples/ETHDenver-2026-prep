'use client';

import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { formatEther } from 'viem';
import {
  Loader2,
  Server,
  Shield,
  Clock,
  Boxes,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Coins,
} from 'lucide-react';
import { useNodes, type WorkerNode } from '@/hooks/useNodes';
import { Modal } from '@/components/ui/Modal';
import { cn } from '@/lib/utils';

// ============================================================================
// Types
// ============================================================================

type StatusFilter = 'all' | 'active' | 'idle' | 'offline';
type SortKey = 'reputation' | 'staked' | 'steps' | 'uptime';

// ============================================================================
// Helpers
// ============================================================================

const STATUS_DOT: Record<string, string> = {
  active: 'bg-emerald-400',
  proving: 'bg-emerald-400',
  training: 'bg-emerald-400',
  syncing: 'bg-emerald-400',
  idle: 'bg-helix-muted',
  offline: 'bg-helix-dim',
};

const STATUS_LABEL: Record<string, string> = {
  active: 'Active',
  proving: 'Proving',
  training: 'Training',
  syncing: 'Syncing',
  idle: 'Idle',
  offline: 'Offline',
};

function isActive(status: WorkerNode['status']): boolean {
  return status === 'active' || status === 'proving' || status === 'training' || status === 'syncing';
}

function formatAddr(address: string) {
  if (address.length < 12) return address;
  return `${address.slice(0, 6)}\u2026${address.slice(-4)}`;
}

function formatStake(amount: bigint): string {
  const val = Number(formatEther(amount));
  if (val >= 1000) return `${(val / 1000).toFixed(1)}k`;
  return val.toFixed(1);
}

function formatCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return n.toString();
}

function formatTimeSince(timestamp: number): string {
  if (!timestamp) return '\u2014';
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 5) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}


// ============================================================================
// Page
// ============================================================================

export default function NetworkPage() {
  const { filteredNodes: allNodes, networkStats, isLoading, updateFilter } = useNodes();
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [sortKey, setSortKey] = useState<SortKey>('reputation');
  const [selectedNode, setSelectedNode] = useState<WorkerNode | null>(null);

  // ── Status counts
  const statusCounts = useMemo(() => {
    const counts = { all: allNodes.length, active: 0, idle: 0, offline: 0 };
    allNodes.forEach((n) => {
      if (isActive(n.status)) counts.active++;
      else if (n.status === 'idle') counts.idle++;
      else counts.offline++;
    });
    return counts;
  }, [allNodes]);

  // ── Filtered + sorted nodes
  const displayNodes = useMemo(() => {
    let result = allNodes;
    if (statusFilter === 'active') result = result.filter((n) => isActive(n.status));
    else if (statusFilter === 'idle') result = result.filter((n) => n.status === 'idle');
    else if (statusFilter === 'offline') result = result.filter((n) => n.status === 'offline');

    const sorted = [...result];
    switch (sortKey) {
      case 'reputation':
        sorted.sort((a, b) => b.reputation - a.reputation);
        break;
      case 'staked':
        sorted.sort((a, b) => Number(b.stakedAmount - a.stakedAmount));
        break;
      case 'steps':
        sorted.sort((a, b) => b.proofsSubmitted - a.proofsSubmitted);
        break;
      case 'uptime':
        sorted.sort((a, b) => b.lastSeen - a.lastSeen);
        break;
    }
    return sorted;
  }, [allNodes, statusFilter, sortKey]);

  // ── Derived stats
  const onlineCount = useMemo(
    () => allNodes.filter((n) => n.status !== 'offline').length,
    [allNodes],
  );

  const totalJobs = useMemo(
    () => allNodes.reduce((sum, n) => sum + n.roundsCompleted, 0),
    [allNodes],
  );

  const uptimeLabel = useMemo(() => {
    if (networkStats.totalNodes === 0) return '\u2014';
    return `${(networkStats.networkUptime * 100).toFixed(0)}%`;
  }, [networkStats.networkUptime, networkStats.totalNodes]);

  // ── Handlers
  const handleStatusFilter = (f: StatusFilter) => {
    setStatusFilter(f);
    updateFilter({ status: f === 'all' ? undefined : (f as WorkerNode['status']) });
  };

  // ── Loading
  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
        className="flex items-center justify-center py-40"
      >
        <Loader2 size={28} className="animate-spin text-helix-muted" />
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="space-y-8 max-w-7xl mx-auto"
    >
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-4xl font-semibold tracking-tight text-white">Network</h1>
          <p className="text-lg text-helix-muted mt-2">
            Workers powering the Helix protocol
          </p>
        </div>
        <div className="flex items-center gap-2.5 pb-1">
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-400" />
          </span>
          <span className="text-base text-helix-muted">Live</span>
        </div>
      </div>

      {/* ── Summary Stats ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          {
            label: 'Workers Online',
            value: `${onlineCount} / ${networkStats.totalNodes}`,
            icon: Server,
          },
          {
            label: 'Total Staked',
            value: `${formatStake(networkStats.totalStaked)} HLX`,
            icon: Coins,
          },
          {
            label: 'Network Uptime',
            value: uptimeLabel,
            icon: Clock,
          },
          {
            label: 'Avg Reputation',
            value: Math.round(networkStats.averageReputation).toString(),
            icon: Shield,
          },
        ].map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: i * 0.05 }}
            className="relative rounded-2xl border border-helix-border bg-helix-surface/50 p-6 overflow-hidden"
          >
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-helix-muted">{stat.label}</span>
              <div className="p-2 rounded-xl bg-white/[0.04]">
                <stat.icon size={16} className="text-helix-dim" />
              </div>
            </div>
            <p className="text-2xl font-semibold text-white tracking-tight tabular-nums">
              {stat.value}
            </p>
          </motion.div>
        ))}
      </div>

      {/* ── Filter / Sort Controls ─────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3, delay: 0.2 }}
        className="flex flex-col lg:flex-row lg:items-center gap-4"
      >
        {/* Status filter */}
        <div className="flex items-center gap-2">
          {(
            [
              { key: 'all', label: 'All' },
              { key: 'active', label: 'Active' },
              { key: 'idle', label: 'Idle' },
              { key: 'offline', label: 'Offline' },
            ] as const
          ).map((f) => (
            <button
              key={f.key}
              type="button"
              onClick={() => handleStatusFilter(f.key)}
              className={cn(
                'px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200',
                statusFilter === f.key
                  ? 'bg-white/10 text-white'
                  : 'text-helix-dim hover:text-helix-muted hover:bg-white/[0.03]',
              )}
            >
              {f.label}
              <span className="ml-2 tabular-nums opacity-50">
                {statusCounts[f.key]}
              </span>
            </button>
          ))}
        </div>

        {/* Divider */}
        <div className="hidden lg:block w-px h-8 bg-helix-border" />

        {/* Sort */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-helix-dim mr-1">Sort</span>
          {(
            [
              { key: 'reputation', label: 'Reputation' },
              { key: 'staked', label: 'Staked' },
              { key: 'steps', label: 'Proofs' },
              { key: 'uptime', label: 'Uptime' },
            ] as const
          ).map((s) => (
            <button
              key={s.key}
              type="button"
              onClick={() => setSortKey(s.key)}
              className={cn(
                'px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200',
                sortKey === s.key
                  ? 'bg-white/10 text-white'
                  : 'text-helix-dim hover:text-helix-muted hover:bg-white/[0.03]',
              )}
            >
              {s.label}
            </button>
          ))}
        </div>
      </motion.div>

      {/* ── Worker Cards ───────────────────────────────────────────────── */}
      {displayNodes.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="flex flex-col items-center justify-center py-28 rounded-2xl border border-helix-border bg-helix-surface/50"
        >
          <div className="p-5 rounded-2xl bg-white/[0.04] mb-5">
            <Server size={32} className="text-helix-dim" />
          </div>
          <p className="text-xl font-medium text-white">No workers found</p>
          <p className="text-base text-helix-dim mt-2">
            {statusFilter !== 'all'
              ? 'Try a different filter'
              : 'No nodes have joined the network yet'}
          </p>
        </motion.div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {displayNodes.map((node, i) => (
            <motion.div
              key={node.id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, delay: Math.min(i * 0.03, 0.3) }}
            >
              <WorkerCard
                node={node}
                onClick={() => setSelectedNode(node)}
              />
            </motion.div>
          ))}
        </div>
      )}

      {/* ── Network Metrics Footer ─────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.25 }}
        className="grid grid-cols-1 sm:grid-cols-3 gap-4"
      >
        {[
          {
            label: 'Total Rounds Completed',
            value: totalJobs.toLocaleString(),
            icon: Boxes,
          },
          {
            label: 'Total Proofs',
            value: formatCompact(networkStats.totalProofs),
            icon: ShieldCheck,
          },
          {
            label: 'Cheaters Detected',
            value: allNodes.filter((n) => n.slashed).length.toString(),
            icon: ShieldAlert,
            isAlert: allNodes.some((n) => n.slashed),
          },
        ].map((footer) => (
          <div
            key={footer.label}
            className="relative rounded-2xl border border-helix-border bg-helix-surface/50 p-6 overflow-hidden"
          >
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-helix-muted mb-2">{footer.label}</p>
                <p className="text-2xl font-semibold text-white tracking-tight tabular-nums">
                  {footer.value}
                </p>
              </div>
              <div className={cn(
                'p-2.5 rounded-xl',
                footer.isAlert ? 'bg-red-500/10' : 'bg-white/[0.04]',
              )}>
                <footer.icon size={18} className={cn(
                  footer.isAlert ? 'text-red-400/70' : 'text-helix-dim',
                )} />
              </div>
            </div>
          </div>
        ))}
      </motion.div>

      {/* ── Worker Detail Modal ─────────────────────────────────────────── */}
      <WorkerDetailModal
        node={selectedNode}
        onClose={() => setSelectedNode(null)}
      />
    </motion.div>
  );
}

// ============================================================================
// Sub-components
// ============================================================================

function WorkerCard({
  node,
  onClick,
}: {
  node: WorkerNode;
  onClick: () => void;
}) {
  const status = node.slashed ? 'offline' : node.status;
  const dotColor = STATUS_DOT[status] ?? 'bg-helix-dim';
  const badgeStyle = node.slashed
    ? 'bg-red-500/10 text-red-400/90'
    : isActive(node.status)
      ? 'bg-emerald-500/10 text-emerald-400/90'
      : status === 'idle'
        ? 'bg-white/[0.05] text-white/40'
        : 'bg-white/[0.03] text-white/25';
  const badgeLabel = node.slashed ? 'Slashed' : (STATUS_LABEL[status] ?? status);
  const isNodeActive = isActive(node.status) && !node.slashed;
  const repClamped = Math.max(0, Math.min(100, node.reputation));

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'w-full h-full relative rounded-3xl overflow-hidden',
        'bg-white/[0.04] backdrop-blur-sm',
        'hover:bg-white/[0.07] transition-all duration-300 ease-out group',
        'px-7 py-7 text-left',
      )}
    >
      {/* Soft emerald glow on hover */}
      <div className="absolute top-0 inset-x-0 h-28 bg-gradient-to-b from-emerald-500/[0.04] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

      {/* Row 1: Address + status pill */}
      <div className="relative flex items-center justify-between mb-6">
        <div className="flex items-center gap-3 min-w-0">
          <span className="relative flex shrink-0">
            {isNodeActive && (
              <span className={cn('absolute inline-flex h-2.5 w-2.5 rounded-full opacity-40 animate-ping', dotColor)} />
            )}
            <span className={cn('relative w-2.5 h-2.5 rounded-full shrink-0', dotColor)} />
          </span>
          <span className="text-[17px] font-medium text-white tracking-tight truncate">
            {formatAddr(node.address)}
          </span>
          {node.partyIndex != null && (
            <span className="text-sm text-white/30 shrink-0">P{node.partyIndex}</span>
          )}
        </div>
        <span className={cn(
          'text-sm font-medium px-4 py-1.5 rounded-full shrink-0 ml-3',
          badgeStyle,
        )}>
          {badgeLabel}
        </span>
      </div>

      {/* Row 2: Big reputation number + bar */}
      <div className="relative mb-7">
        <div className="flex items-end justify-between mb-3">
          <div>
            <p className="text-sm text-white/40 mb-1.5">Reputation</p>
            <p className={cn(
              'text-4xl font-semibold tabular-nums tracking-tight leading-none',
              repClamped >= 80 ? 'text-emerald-400' :
              repClamped >= 50 ? 'text-white' :
              'text-white/50',
            )}>
              {Math.round(repClamped)}
            </p>
          </div>
          <p className="text-sm text-white/30 pb-1">
            {formatTimeSince(node.lastHeartbeat)}
          </p>
        </div>
        <div className="h-2 bg-white/[0.06] rounded-full overflow-hidden">
          <motion.div
            className={cn(
              'h-full rounded-full',
              repClamped >= 80 ? 'bg-emerald-400' :
              repClamped >= 50 ? 'bg-white/30' :
              'bg-white/15',
            )}
            initial={{ width: 0 }}
            animate={{ width: `${repClamped}%` }}
            transition={{ duration: 0.8, ease: [0.25, 0.1, 0.25, 1] }}
          />
        </div>
      </div>

      {/* Row 3: Stats in soft containers */}
      <div className="relative flex items-center gap-3">
        <div className="flex-1 rounded-2xl bg-white/[0.04] px-5 py-3.5">
          <p className="text-sm text-white/35 mb-1">Rounds</p>
          <p className="text-xl font-semibold text-white tabular-nums">{node.roundsParticipated}</p>
        </div>
        <div className="flex-1 rounded-2xl bg-white/[0.04] px-5 py-3.5">
          <p className="text-sm text-white/35 mb-1">Proofs</p>
          <p className="text-xl font-semibold text-white tabular-nums">{node.proofsSubmitted}</p>
        </div>
      </div>
    </button>
  );
}

function WorkerDetailModal({
  node,
  onClose,
}: {
  node: WorkerNode | null;
  onClose: () => void;
}) {
  if (!node) return null;

  const status = node.slashed ? 'offline' : node.status;
  const dotColor = STATUS_DOT[status] ?? 'bg-helix-dim';
  const badgeStyle = node.slashed
    ? 'bg-red-500/10 text-red-400/90'
    : isActive(node.status)
      ? 'bg-emerald-500/10 text-emerald-400/90'
      : status === 'idle'
        ? 'bg-white/[0.05] text-white/40'
        : 'bg-white/[0.03] text-white/25';
  const badgeLabel = node.slashed ? 'Slashed' : (STATUS_LABEL[status] ?? status);
  const isNodeActive = isActive(node.status) && !node.slashed;
  const repClamped = Math.max(0, Math.min(100, node.reputation));
  const earnings = Number(formatEther(node.earningsTotal));
  const successPct = node.successRate != null && node.roundsParticipated > 0
    ? Math.round(node.successRate * 100)
    : null;

  return (
    <Modal isOpen={!!node} onClose={onClose} className="max-w-2xl">
      <div className="space-y-7">
        {/* Header: address + status */}
        <div className="flex items-center gap-3">
          <span className="relative flex shrink-0">
            {isNodeActive && (
              <span className={cn('absolute inline-flex h-3 w-3 rounded-full opacity-40 animate-ping', dotColor)} />
            )}
            <span className={cn('relative w-3 h-3 rounded-full shrink-0', dotColor)} />
          </span>
          <span className="text-base font-medium text-white tracking-tight flex-1 break-all">
            {node.address}
          </span>
          <span className={cn('text-sm font-medium px-4 py-1.5 rounded-full shrink-0', badgeStyle)}>
            {badgeLabel}
          </span>
        </div>

        {/* Metadata row */}
        <div className="flex items-center gap-3 flex-wrap">
          {node.partyIndex != null && (
            <span className="text-sm text-white/35">Party {node.partyIndex}</span>
          )}
          {node.endpoint && (
            <span className="text-sm text-white/25 ml-auto">{node.endpoint}</span>
          )}
        </div>

        {/* Slashed warning */}
        {node.slashed && (
          <div className="flex items-center gap-3 px-5 py-4 rounded-2xl bg-red-500/8">
            <AlertTriangle size={18} className="text-red-400/80 shrink-0" />
            <span className="text-[15px] text-red-300/90">
              This worker has been slashed for submitting invalid proofs.
            </span>
          </div>
        )}

        {/* Reputation — hero section */}
        <div>
          <p className="text-sm text-white/40 mb-2">Reputation</p>
          <div className="flex items-end justify-between mb-3">
            <p className={cn(
              'text-5xl font-semibold tabular-nums tracking-tight leading-none',
              repClamped >= 80 ? 'text-emerald-400' :
              repClamped >= 50 ? 'text-white' :
              'text-white/50',
            )}>
              {Math.round(repClamped)}
            </p>
            {successPct !== null && (
              <p className={cn(
                'text-base tabular-nums',
                successPct >= 90 ? 'text-emerald-400/70' :
                successPct < 50 ? 'text-red-400/70' :
                'text-white/35',
              )}>
                {successPct}% success
              </p>
            )}
          </div>
          <div className="h-2.5 bg-white/[0.06] rounded-full overflow-hidden">
            <motion.div
              className={cn(
                'h-full rounded-full',
                repClamped >= 80 ? 'bg-emerald-400' :
                repClamped >= 50 ? 'bg-white/30' :
                'bg-white/15',
              )}
              initial={{ width: 0 }}
              animate={{ width: `${repClamped}%` }}
              transition={{ duration: 0.8, ease: [0.25, 0.1, 0.25, 1] }}
            />
          </div>
        </div>

        {/* Stats — 2x2 grid in soft containers */}
        <div className="grid grid-cols-2 gap-3">
          <ModalStat label="Rounds" value={node.roundsParticipated.toString()} />
          <ModalStat label="Completed" value={node.roundsCompleted.toString()} />
          <ModalStat label="Proofs" value={node.proofsSubmitted.toString()} />
          <ModalStat
            label="Failed"
            value={node.proofsFailed.toString()}
            highlight={node.proofsFailed > 0 ? 'red' : undefined}
          />
        </div>

        {/* Staked + Earnings row */}
        <div className="flex items-center gap-3">
          <div className="flex-1 rounded-2xl bg-white/[0.04] px-5 py-4">
            <p className="text-sm text-white/35 mb-1">Staked</p>
            <p className="text-xl font-semibold text-white tabular-nums">
              {Number(formatEther(node.stakedAmount)).toFixed(2)} <span className="text-base font-normal text-white/35">HLX</span>
            </p>
          </div>
          {earnings > 0 && (
            <div className="flex-1 rounded-2xl bg-emerald-500/[0.06] px-5 py-4">
              <p className="text-sm text-emerald-400/50 mb-1">Earned</p>
              <p className="text-xl font-semibold text-emerald-400 tabular-nums">
                {earnings.toFixed(2)} <span className="text-base font-normal text-emerald-400/50">HLX</span>
              </p>
            </div>
          )}
        </div>

        {/* System metrics — only if there's data */}
        {(node.metrics.cpu > 0 || node.metrics.memory > 0) && (
          <div className="flex items-center gap-3">
            {node.metrics.cpu > 0 && (
              <div className="flex-1 rounded-2xl bg-white/[0.04] px-5 py-3.5">
                <p className="text-sm text-white/35 mb-1">CPU</p>
                <p className="text-lg font-semibold text-white tabular-nums">{node.metrics.cpu}%</p>
              </div>
            )}
            {node.metrics.memory > 0 && (
              <div className="flex-1 rounded-2xl bg-white/[0.04] px-5 py-3.5">
                <p className="text-sm text-white/35 mb-1">Memory</p>
                <p className="text-lg font-semibold text-white tabular-nums">{node.metrics.memory} MB</p>
              </div>
            )}
            <div className="flex-1 rounded-2xl bg-white/[0.04] px-5 py-3.5">
              <p className="text-sm text-white/35 mb-1">Last seen</p>
              <p className="text-lg font-semibold text-white">{formatTimeSince(node.lastHeartbeat)}</p>
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
}

function ModalStat({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: 'red' | 'green';
}) {
  return (
    <div className="rounded-2xl bg-white/[0.04] px-5 py-4">
      <p className="text-sm text-white/35 mb-1">{label}</p>
      <p className={cn(
        'text-xl font-semibold tabular-nums',
        highlight === 'red' ? 'text-red-400' :
        highlight === 'green' ? 'text-emerald-400' :
        'text-white',
      )}>
        {value}
      </p>
    </div>
  );
}
