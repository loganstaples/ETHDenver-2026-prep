'use client';

import { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { formatEther } from 'viem';
import {
  X,
  Cpu,
  HardDrive,
  MonitorDot,
  Network,
  Thermometer,
  ShieldCheck,
  MapPin,
  Timer,
  Layers,
  CheckCircle2,
  XCircle,
  BarChart3,
  Coins,
} from 'lucide-react';
import { type WorkerNode } from '@/hooks/useNodes';
import { Badge } from '@/components/ui/Badge';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { AddressDisplay } from '@/components/ui/AddressDisplay';
import { cn } from '@/lib/utils';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface NodeDetailSheetProps {
  node: WorkerNode | null;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

function formatBytes(mbPerSec: number): string {
  if (mbPerSec >= 1000) return `${(mbPerSec / 1000).toFixed(1)} GB/s`;
  return `${mbPerSec.toFixed(0)} MB/s`;
}

// ---------------------------------------------------------------------------
// Section
// ---------------------------------------------------------------------------

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h3 className="text-2xs text-helix-muted uppercase tracking-wider mb-3">
        {title}
      </h3>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// MetricRow
// ---------------------------------------------------------------------------

function MetricRow({
  icon: Icon,
  label,
  value,
  bar,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  bar?: number;
}) {
  return (
    <div className="flex items-center gap-3 py-1.5">
      <Icon size={14} className="text-helix-dim flex-shrink-0" strokeWidth={1.5} />
      <span className="text-2xs text-helix-muted flex-shrink-0 w-20">{label}</span>
      {bar !== undefined ? (
        <div className="flex-1 flex items-center gap-2">
          <ProgressBar value={bar} size="sm" className="flex-1" />
          <span className="font-mono text-2xs text-helix-text2 w-10 text-right">
            {Math.round(bar)}%
          </span>
        </div>
      ) : (
        <span className="font-mono text-2xs text-white">{value}</span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// StatRow
// ---------------------------------------------------------------------------

function StatRow({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
}) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <div className="flex items-center gap-2.5">
        <Icon size={14} className="text-helix-dim flex-shrink-0" strokeWidth={1.5} />
        <span className="text-2xs text-helix-muted">{label}</span>
      </div>
      <span className="font-mono text-2xs text-white">{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sheet
// ---------------------------------------------------------------------------

export function NodeDetailSheet({ node, onClose }: NodeDetailSheetProps) {
  // Close on Escape
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const successRate =
    node && node.proofsSubmitted > 0
      ? ((node.proofsVerified / node.proofsSubmitted) * 100).toFixed(1)
      : '0.0';

  return (
    <AnimatePresence>
      {node && (
        <>
          {/* Backdrop */}
          <motion.div
            key="sheet-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 bg-black/40"
            onClick={onClose}
          />

          {/* Panel */}
          <motion.aside
            key="sheet-panel"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className={cn(
              'fixed top-0 right-0 z-50 h-full w-[400px] max-w-full',
              'bg-helix-surface border-l border-helix-border',
              'flex flex-col overflow-hidden',
            )}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-helix-border flex-shrink-0">
              <div className="flex items-center gap-3 min-w-0">
                <AddressDisplay address={node.address} />
                <Badge variant={statusVariant(node.status)}>
                  {node.status.charAt(0).toUpperCase() + node.status.slice(1)}
                </Badge>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="text-helix-muted hover:text-white transition-colors p-1 -mr-1"
                aria-label="Close"
              >
                <X size={16} strokeWidth={1.5} />
              </button>
            </div>

            {/* Scrollable content */}
            <div className="flex-1 overflow-y-auto px-5 py-5 space-y-6">
              {/* Role */}
              <Section title="Role">
                <Badge className="bg-white/[0.08] text-helix-text2">
                  {node.type}
                </Badge>
              </Section>

              {/* System Metrics */}
              <Section title="System Metrics">
                <div className="space-y-0.5">
                  <MetricRow icon={Cpu} label="CPU" bar={node.metrics.cpu} value="" />
                  <MetricRow icon={HardDrive} label="Memory" bar={node.metrics.memory} value="" />
                  {node.metrics.gpu !== undefined && (
                    <MetricRow icon={MonitorDot} label="GPU" bar={node.metrics.gpu} value="" />
                  )}
                  {node.metrics.gpuMemory !== undefined && (
                    <MetricRow icon={MonitorDot} label="GPU Memory" bar={node.metrics.gpuMemory} value="" />
                  )}
                  <MetricRow
                    icon={Network}
                    label="Net In"
                    value={formatBytes(node.metrics.networkIn)}
                  />
                  <MetricRow
                    icon={Network}
                    label="Net Out"
                    value={formatBytes(node.metrics.networkOut)}
                  />
                  {node.metrics.temperature !== undefined && (
                    <MetricRow
                      icon={Thermometer}
                      label="Temperature"
                      value={`${node.metrics.temperature.toFixed(0)} C`}
                    />
                  )}
                </div>
              </Section>

              {/* Capabilities */}
              <Section title="Capabilities">
                <div className="space-y-0.5">
                  {node.capabilities.gpuModel && (
                    <StatRow icon={MonitorDot} label="GPU Model" value={node.capabilities.gpuModel} />
                  )}
                  <StatRow
                    icon={Layers}
                    label="GPU Memory"
                    value={`${node.capabilities.gpuMemoryMb} MB`}
                  />
                  <StatRow
                    icon={Layers}
                    label="Max Batch Size"
                    value={node.capabilities.maxBatchSize}
                  />
                </div>
              </Section>

              {/* Performance */}
              <Section title="Performance">
                <div className="space-y-0.5">
                  <StatRow
                    icon={CheckCircle2}
                    label="Proofs Submitted"
                    value={node.proofsSubmitted}
                  />
                  <StatRow
                    icon={CheckCircle2}
                    label="Proofs Verified"
                    value={node.proofsVerified}
                  />
                  <StatRow
                    icon={XCircle}
                    label="Proofs Failed"
                    value={node.proofsFailed}
                  />
                  <StatRow
                    icon={Timer}
                    label="Rounds Participated"
                    value={node.roundsParticipated}
                  />
                  <StatRow
                    icon={BarChart3}
                    label="Success Rate"
                    value={`${successRate}%`}
                  />
                </div>
              </Section>

              {/* Stake & Reputation */}
              <Section title="Stake & Reputation">
                <div className="space-y-0.5">
                  <StatRow
                    icon={Coins}
                    label="Staked"
                    value={`${Number(formatEther(node.stakedAmount)).toFixed(2)} ETH`}
                  />
                  <StatRow
                    icon={Coins}
                    label="Total Earnings"
                    value={`${Number(formatEther(node.earningsTotal)).toFixed(2)} ETH`}
                  />
                  <StatRow
                    icon={ShieldCheck}
                    label="Reputation"
                    value={node.reputation.toFixed(1)}
                  />
                  {node.slashed && (
                    <div className="mt-2">
                      <Badge variant="outline" className="text-helix-muted">
                        Slashed
                      </Badge>
                    </div>
                  )}
                </div>
              </Section>

              {/* Location */}
              {node.location && (
                <Section title="Location">
                  <div className="space-y-0.5">
                    <StatRow
                      icon={MapPin}
                      label="Region"
                      value={node.location.region}
                    />
                    <StatRow
                      icon={Timer}
                      label="Latency"
                      value={`${node.location.latency.toFixed(0)} ms`}
                    />
                  </div>
                </Section>
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
