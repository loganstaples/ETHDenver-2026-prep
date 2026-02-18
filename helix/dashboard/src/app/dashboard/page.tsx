'use client';

import { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChevronDown,
  Pause,
  Square,
  CheckCircle,
  AlertTriangle,
  Info,
  XCircle,
  X,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { cn } from '@/lib/utils';
import { useDashboardSessions, type TrainingEvent } from '@/hooks/useDashboardSessions';
import { useWorkerHealth, type WorkerHealth } from '@/hooks/useWorkerHealth';
import { formatEther } from 'viem';

// ============================================================================
// Alert icon/color helpers
// ============================================================================

const ALERT_CONFIG: Record<string, { icon: typeof Info; color: string }> = {
  info: { icon: Info, color: '#60a5fa' },
  success: { icon: CheckCircle, color: '#4ade80' },
  warning: { icon: AlertTriangle, color: '#fbbf24' },
  error: { icon: XCircle, color: '#f87171' },
};

function timeAgo(ts: number): string {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

/** Format milliseconds as "Xm Ys" or just "Ys" */
function formatDuration(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/** Live elapsed since timestamp, with seconds */
function elapsedSince(ts: number): string {
  return formatDuration(Date.now() - ts);
}

// ============================================================================
// Types (populated from API)
// ============================================================================

interface TrainingSession {
  id: string;
  name: string;
  modelId: bigint;
  status: 'training' | 'paused';
}

interface DashboardInferenceRequest {
  id: string;
  model: string;
  status: 'processing' | 'completed';
  progress: number;
  phase: string;
  workers: number;
  created: number;
  inputLabel: string;
  result: string | null;
  confidence: number | null;
  duration: number | null;
}

// ============================================================================
// Horseshoe Progress
// ============================================================================

function HorseshoeProgress({ progress, size = 150 }: { progress: number; size?: number }) {
  const pad = 16; // extra padding so glow never clips
  const full = size + pad * 2;
  const strokeWidth = 7;
  const radius = (size - strokeWidth) / 2;
  const center = full / 2;

  // 270° arc (gap at bottom)
  const arcDeg = 270;
  const circumference = 2 * Math.PI * radius;
  const arcLength = (arcDeg / 360) * circumference;
  const filledLength = (Math.min(100, Math.max(0, progress)) / 100) * arcLength;

  // Rotate so gap is at the bottom center: start at 135° (bottom-left)
  const rotation = 135;

  // The 90° gap at the bottom shifts the visual center upward.
  // Nudge the whole SVG down to compensate (~8% of size).
  const yOffset = size * 0.08;

  const filterId = `glow-${size}`;

  return (
    <svg
      width={full}
      height={full}
      viewBox={`0 0 ${full} ${full}`}
      style={{ margin: -pad, marginTop: -pad + yOffset, overflow: 'visible' }}
    >
      <defs>
        <filter id={filterId} x="-100%" y="-100%" width="400%" height="400%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Track (dark) */}
      <circle
        cx={center}
        cy={center}
        r={radius}
        fill="none"
        stroke="#1e1e22"
        strokeWidth={strokeWidth}
        strokeDasharray={`${arcLength} ${circumference}`}
        strokeLinecap="round"
        transform={`rotate(${rotation} ${center} ${center})`}
      />

      {/* Filled arc (white with glow) */}
      {progress > 0 && (
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="white"
          strokeWidth={strokeWidth}
          strokeDasharray={`${filledLength} ${circumference}`}
          strokeLinecap="round"
          transform={`rotate(${rotation} ${center} ${center})`}
          filter={`url(#${filterId})`}
          style={{ transition: 'stroke-dasharray 0.6s ease-out' }}
        />
      )}

      {/* Percentage text */}
      <text
        x={center}
        y={center - size * 0.02}
        textAnchor="middle"
        dominantBaseline="central"
        className="fill-white font-sans"
        style={{ fontSize: size * 0.26, fontWeight: 600, letterSpacing: '-0.02em' }}
      >
        {Math.round(progress)}%
      </text>
    </svg>
  );
}

// ============================================================================
// Worker Detail Modal
// ============================================================================

function WorkerModal({ worker, onClose }: { worker: WorkerHealth; onClose: () => void }) {
  const earnings = Number(formatEther(worker.performance.totalEarnings));
  const stake = Number(formatEther(worker.stake.amount));

  const resources = [
    { label: 'CPU', value: worker.metrics.cpu },
    { label: 'Memory', value: worker.metrics.memory },
    ...(worker.metrics.gpu != null ? [{ label: 'GPU', value: worker.metrics.gpu }] : []),
  ];

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 8 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.96, y: 8 }}
        transition={{ duration: 0.2 }}
        className="w-[480px] bg-helix-surface border border-helix-border rounded-2xl shadow-2xl overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-helix-border">
          <div className="flex items-center gap-3">
            <div className={cn(
              'w-2.5 h-2.5 rounded-full',
              worker.status === 'healthy' ? 'bg-green-400'
                : worker.status === 'degraded' ? 'bg-yellow-400'
                  : worker.status === 'unhealthy' ? 'bg-red-400'
                    : 'bg-helix-dim',
            )} />
            <span className="text-sm font-mono text-white">{worker.address}</span>
          </div>
          <button onClick={onClose} className="text-helix-muted hover:text-white transition-colors p-1">
            <X size={16} />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Status + Role */}
          <div className="flex gap-2">
            <span className={cn(
              'text-xs px-2.5 py-1 rounded-lg capitalize',
              worker.status === 'healthy' ? 'bg-green-400/10 text-green-400'
                : worker.status === 'degraded' ? 'bg-yellow-400/10 text-yellow-400'
                  : worker.status === 'unhealthy' ? 'bg-red-400/10 text-red-400'
                    : 'bg-helix-border text-helix-muted',
            )}>
              {worker.status}
            </span>
            <span className="text-xs px-2.5 py-1 rounded-lg bg-helix-border text-helix-text2 capitalize">
              {worker.role}
            </span>
            <span className="text-xs px-2.5 py-1 rounded-lg bg-helix-border text-helix-text2 capitalize">
              {worker.activity}
            </span>
          </div>

          {/* Resources */}
          <div>
            <div className="text-xs text-helix-muted mb-3">Resources</div>
            <div className="space-y-3">
              {resources.map((r) => (
                <div key={r.label} className="flex items-center gap-3">
                  <span className="text-xs text-helix-text2 w-14">{r.label}</span>
                  <div className="flex-1 h-1.5 bg-helix-border rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-white/80 transition-all"
                      style={{ width: `${Math.min(100, r.value)}%` }}
                    />
                  </div>
                  <span className="text-xs text-white font-mono tabular-nums w-10 text-right">
                    {r.value.toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-helix-bg rounded-xl px-4 py-3">
              <div className="text-xs text-helix-muted mb-1">Earnings</div>
              <div className="text-lg font-mono font-medium text-white tabular-nums">{earnings.toFixed(2)}</div>
              <div className="text-xs text-helix-muted">ADI</div>
            </div>
            <div className="bg-helix-bg rounded-xl px-4 py-3">
              <div className="text-xs text-helix-muted mb-1">Stake</div>
              <div className="text-lg font-mono font-medium text-white tabular-nums">{stake.toFixed(2)}</div>
              <div className="text-xs text-helix-muted">ADI</div>
            </div>
            <div className="bg-helix-bg rounded-xl px-4 py-3">
              <div className="text-xs text-helix-muted mb-1">Success</div>
              <div className="text-lg font-mono font-medium text-white tabular-nums">
                {(worker.performance.successRate * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-helix-muted">rate</div>
            </div>
          </div>

          {/* Performance details */}
          <div className="grid grid-cols-2 gap-x-8 gap-y-2">
            {[
              { label: 'Proofs submitted', value: worker.performance.proofsSubmitted },
              { label: 'Proofs verified', value: worker.performance.proofsVerified },
              { label: 'Avg proof time', value: `${worker.performance.averageProofTime.toFixed(0)}ms` },
              { label: 'Uptime', value: `${(worker.performance.uptime * 100).toFixed(1)}%` },
              { label: 'Latency', value: `${worker.metrics.latency.toFixed(0)}ms` },
              { label: 'Rounds', value: worker.performance.roundsParticipated },
            ].map((s) => (
              <div key={s.label} className="flex items-center justify-between py-1">
                <span className="text-xs text-helix-muted">{s.label}</span>
                <span className="text-xs text-white font-mono tabular-nums">{s.value}</span>
              </div>
            ))}
          </div>

          {/* Issues */}
          {worker.issues.filter(i => !i.resolved).length > 0 && (
            <div className="space-y-1.5">
              {worker.issues.filter(i => !i.resolved).map((issue) => (
                <div key={issue.id} className="flex items-center gap-2 px-3 py-2 rounded-xl bg-red-500/5 border border-red-500/10">
                  <div className="w-1.5 h-1.5 rounded-full bg-red-400 shrink-0" />
                  <span className="text-xs text-red-300">{issue.message}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export default function DashboardPage() {
  const [tab, setTab] = useState<'training' | 'inference'>('training');
  const [inferenceRequests, setInferenceRequests] = useState<DashboardInferenceRequest[]>([]);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [selectedWorker, setSelectedWorker] = useState<WorkerHealth | null>(null);

  // Config display values (read-only — backend doesn't support mid-training changes)
  const displayLR = '0.001';
  const displayBatch = '64';

  // Backend data via useDashboardSessions
  const {
    sessions: backendSessions,
    activeSession,
    losses,
    events,
    selectSession,
  } = useDashboardSessions();

  const { workers: healthWorkers } = useWorkerHealth();

  // Map backend sessions to dropdown items
  const sessions: TrainingSession[] = useMemo(
    () => backendSessions.map((s) => ({
      id: s.session_id,
      name: `Session ${s.session_id.slice(0, 8)}`,
      modelId: BigInt(s.job_id || 0),
      status: (s.status === 'running' || s.status === 'starting') ? 'training' as const : 'paused' as const,
    })),
    [backendSessions],
  );

  const selectedSession = useMemo(
    () => activeSession ? sessions.find((s) => s.id === activeSession.session_id) ?? null : null,
    [activeSession, sessions],
  );

  // Load inference history from localStorage
  useEffect(() => {
    const loadInferenceHistory = () => {
      try {
        const raw = localStorage.getItem('helix-inference-history');
        if (raw) {
          const parsed = JSON.parse(raw) as DashboardInferenceRequest[];
          setInferenceRequests(parsed);
        }
      } catch {
        // Ignore parse errors
      }
    };
    loadInferenceHistory();
    // Re-check periodically in case inference page writes new entries
    const interval = setInterval(loadInferenceHistory, 3000);
    return () => clearInterval(interval);
  }, []);

  // Chart data from losses array
  const lossData = useMemo(
    () => losses.slice(-80).map((l) => ({ step: l.step, value: Number(l.loss.toFixed(4)) })),
    [losses],
  );

  // Accuracy chart: single point at completion or empty during training
  const accData = useMemo(() => {
    if (activeSession?.accuracy != null) {
      return [{ epoch: 1, value: Number((activeSession.accuracy * 100).toFixed(1)) }];
    }
    return [];
  }, [activeSession]);

  // Current / delta values from activeSession
  const currentLoss = activeSession?.current_loss ?? 0;
  const currentAcc = activeSession?.accuracy ?? 0;
  const prevLoss = losses.length > 10 ? losses[losses.length - 11].loss : currentLoss;
  const lossDelta = currentLoss - prevLoss;
  const accDelta = 0; // No previous accuracy to compare

  const progress = activeSession
    ? activeSession.total_steps > 0
      ? (activeSession.current_step / activeSession.total_steps) * 100
      : 0
    : 0;

  // Generate activity feed from WebSocket events
  const activityFeed = useMemo(() => {
    type FeedItem = { id: string; type: string; title: string; message: string; timestamp: number };
    const items: FeedItem[] = [];

    events.forEach((e, i) => {
      const evt = e.event;
      const ts = (e as TrainingEvent & { receivedAt?: number }).receivedAt ?? (Date.now() - (events.length - i) * 100);

      if (evt.type === 'training_step') {
        const step = evt.step as number;
        // Show every 20 steps
        if (step > 0 && step % 20 === 0) {
          items.push({
            id: `step-${step}`,
            type: 'info',
            title: `Step ${step}`,
            message: `loss ${(evt.loss as number).toFixed(4)}`,
            timestamp: ts,
          });
        }
      } else if (evt.type === 'phase_started') {
        items.push({
          id: `phase-${evt.phase}`,
          type: 'info',
          title: `Phase ${evt.phase}`,
          message: (evt.description as string) || '',
          timestamp: ts,
        });
      } else if (evt.type === 'checkpoint_submitted') {
        items.push({
          id: `checkpoint-${i}`,
          type: 'success',
          title: 'Checkpoint submitted',
          message: 'On-chain attestation',
          timestamp: ts,
        });
      } else if (evt.type === 'cheater_detected') {
        items.push({
          id: `cheater-${i}`,
          type: 'error',
          title: 'Cheater detected',
          message: `Worker ${evt.party_index} at step ${evt.step}`,
          timestamp: ts,
        });
      } else if (evt.type === 'session_complete' || evt.type === 'training_complete') {
        items.push({
          id: `complete-${i}`,
          type: 'success',
          title: 'Training complete',
          message: evt.accuracy ? `${((evt.accuracy as number) * 100).toFixed(1)}% accuracy` : '',
          timestamp: ts,
        });
      } else if (evt.type === 'session_failed') {
        items.push({
          id: `failed-${i}`,
          type: 'error',
          title: 'Session failed',
          message: (evt.error as string) || (evt.reason as string) || '',
          timestamp: ts,
        });
      }
    });

    return items.sort((a, b) => b.timestamp - a.timestamp).slice(0, 20);
  }, [events]);

  // Workers sorted: active first by earnings, then offline
  const sortedWorkers = useMemo(() => {
    return [...healthWorkers].sort((a, b) => {
      if (a.status === 'offline' && b.status !== 'offline') return 1;
      if (a.status !== 'offline' && b.status === 'offline') return -1;
      return Number(formatEther(b.performance.totalEarnings)) - Number(formatEther(a.performance.totalEarnings));
    });
  }, [healthWorkers]);

  // Always exactly 3 Y-axis ticks spanning the data range (no 0 baseline)
  function niceTicks3(values: number[], decimals: number): number[] {
    if (values.length === 0) return [0, 0.5, 1];
    let min = Math.min(...values);
    let max = Math.max(...values);
    // If flat or single point, pad ±10% around the value
    if (max - min < 0.001) {
      const pad = Math.max(Math.abs(max) * 0.1, 0.5);
      min = min - pad;
      max = max + pad;
    }
    const step = (max - min) / 2;
    return [
      Number(min.toFixed(decimals)),
      Number((min + step).toFixed(decimals)),
      Number(max.toFixed(decimals)),
    ];
  }

  const lossTicks = useMemo(() => niceTicks3(lossData.map((d) => d.value), 2), [lossData]);
  const accTicks = useMemo(() => niceTicks3(accData.map((d) => d.value), 1), [accData]);

  const chartTooltipStyle = {
    backgroundColor: '#111113',
    border: '1px solid #1e1e22',
    borderRadius: 12,
    fontSize: 12,
    fontFamily: 'var(--font-geist-mono)',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="space-y-6 max-w-[1200px]"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-semibold tracking-tight text-white">Dashboard</h1>

        <div className="flex items-center gap-3">
          {/* Session Selector */}
          {tab === 'training' && selectedSession && (
            <div className="relative">
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                className="flex items-center gap-2.5 px-4 py-2 bg-helix-surface border border-helix-border rounded-xl text-sm text-white hover:border-helix-border2 transition-colors"
              >
                <div className={cn(
                  'w-2 h-2 rounded-full',
                  selectedSession.status === 'training' ? 'bg-green-400' : 'bg-yellow-400',
                )} />
                {selectedSession.name}
                <ChevronDown size={14} className="text-helix-muted" />
              </button>

              <AnimatePresence>
                {dropdownOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    className="absolute right-0 top-full mt-2 w-64 bg-helix-surface border border-helix-border rounded-xl shadow-xl z-50 overflow-hidden"
                  >
                    {sessions.map((s) => (
                      <button
                        key={s.id}
                        onClick={() => { selectSession(s.id); setDropdownOpen(false); }}
                        className={cn(
                          'w-full flex items-center gap-3 px-4 py-3 text-sm text-left hover:bg-white/[0.04] transition-colors',
                          s.id === selectedSession.id && 'bg-white/[0.06]',
                        )}
                      >
                        <div className={cn(
                          'w-2 h-2 rounded-full',
                          s.status === 'training' ? 'bg-green-400' : 'bg-yellow-400',
                        )} />
                        <div>
                          <div className="text-white">{s.name}</div>
                          <div className="text-xs text-helix-muted capitalize mt-0.5">{s.status}</div>
                        </div>
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* Tab Switcher */}
          <div className="flex bg-helix-surface rounded-xl p-1 border border-helix-border">
            {(['training', 'inference'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={cn(
                  'px-5 py-1.5 rounded-lg text-sm font-medium transition-all capitalize',
                  tab === t ? 'bg-white text-black' : 'text-helix-muted hover:text-white',
                )}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ================================================================ */}
      {/* Training Tab */}
      {/* ================================================================ */}
      {tab === 'training' && !selectedSession && (
        <motion.div
          key="training-empty"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.25 }}
        >
          <div className="bg-helix-surface border border-helix-border rounded-2xl p-16 text-center">
            <div className="text-sm text-helix-muted">No active training sessions</div>
            <div className="text-xs text-helix-dim mt-2">Start a training session from the Train page</div>
          </div>
        </motion.div>
      )}

      {tab === 'training' && selectedSession && (
        <motion.div
          key="training"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.25 }}
          className="space-y-4"
        >
          {/* Hero Metrics: Loss + Accuracy */}
          <div className="grid grid-cols-2 gap-4">
            {/* Loss */}
            <div className="bg-helix-surface border border-helix-border rounded-2xl p-6">
              <div className="text-xs text-helix-muted mb-1">Loss</div>
              <div className="flex items-baseline gap-3">
                <span className="text-4xl font-semibold text-white tabular-nums font-mono">
                  {currentLoss.toFixed(4)}
                </span>
                <span className={cn(
                  'text-sm font-medium tabular-nums',
                  lossDelta <= 0 ? 'text-green-400' : 'text-red-400',
                )}>
                  {lossDelta <= 0 ? '\u2193' : '\u2191'} {Math.abs(lossDelta).toFixed(4)}
                </span>
              </div>
              <div className="mt-4 h-36">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={lossData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                    <defs>
                      <linearGradient id="lG" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#ffffff" stopOpacity={0.08} />
                        <stop offset="100%" stopColor="#ffffff" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="step"
                      tick={{ fill: '#63636e', fontSize: 10, fontFamily: 'var(--font-geist-mono)' }}
                      axisLine={{ stroke: '#1e1e22' }}
                      tickLine={false}
                      tickMargin={6}
                    />
                    <YAxis
                      domain={[lossTicks[0], lossTicks[lossTicks.length - 1]]}
                      ticks={lossTicks}
                      tick={{ fill: '#63636e', fontSize: 10, fontFamily: 'var(--font-geist-mono)' }}
                      axisLine={false}
                      tickLine={false}
                      tickMargin={4}
                      width={38}
                    />
                    <Tooltip contentStyle={chartTooltipStyle} labelStyle={{ color: '#63636e' }} />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="rgba(255,255,255,0.5)"
                      fill="url(#lG)"
                      strokeWidth={1.5}
                      dot={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Accuracy */}
            <div className="bg-helix-surface border border-helix-border rounded-2xl p-6">
              <div className="text-xs text-helix-muted mb-1">Accuracy</div>
              <div className="flex items-baseline gap-3">
                <span className="text-4xl font-semibold text-white tabular-nums font-mono">
                  {(currentAcc * 100).toFixed(1)}%
                </span>
                <span className={cn(
                  'text-sm font-medium tabular-nums',
                  accDelta >= 0 ? 'text-green-400' : 'text-red-400',
                )}>
                  {accDelta >= 0 ? '\u2191' : '\u2193'} {Math.abs(accDelta).toFixed(1)}%
                </span>
              </div>
              <div className="mt-4 h-36">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={accData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                    <defs>
                      <linearGradient id="aG" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#4ade80" stopOpacity={0.1} />
                        <stop offset="100%" stopColor="#4ade80" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="epoch"
                      tick={{ fill: '#63636e', fontSize: 10, fontFamily: 'var(--font-geist-mono)' }}
                      axisLine={{ stroke: '#1e1e22' }}
                      tickLine={false}
                      tickMargin={6}
                    />
                    <YAxis
                      domain={[accTicks[0], accTicks[accTicks.length - 1]]}
                      ticks={accTicks}
                      tick={{ fill: '#63636e', fontSize: 10, fontFamily: 'var(--font-geist-mono)' }}
                      axisLine={false}
                      tickLine={false}
                      tickMargin={4}
                      width={42}
                      unit="%"
                    />
                    <Tooltip contentStyle={chartTooltipStyle} labelStyle={{ color: '#63636e' }} />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="#4ade80"
                      fill="url(#aG)"
                      strokeWidth={1.5}
                      dot={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Progress */}
          <div className="bg-helix-surface border border-helix-border rounded-2xl p-6">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-white">
                Step {activeSession?.current_step ?? 0} of {activeSession?.total_steps ?? 0}
                {activeSession?.phase_description && (
                  <><span className="text-helix-muted mx-2">&middot;</span>{activeSession.phase_description}</>
                )}
              </span>
              <span className="text-sm text-helix-muted tabular-nums font-mono">{progress.toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-helix-border rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full bg-white"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
              />
            </div>
            {activeSession?.status === 'complete' && (
              <div className="mt-2 text-xs text-green-400">Training complete</div>
            )}
            {activeSession?.status === 'failed' && (
              <div className="mt-2 text-xs text-red-400">Training failed</div>
            )}
          </div>

          {/* Workers + Settings */}
          <div className="grid grid-cols-5 gap-4">
            {/* Workers */}
            <div className="col-span-3 bg-helix-surface border border-helix-border rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-medium text-white">Workers</span>
                <span className="text-xs text-helix-muted">
                  {healthWorkers.filter((w) => w.status !== 'offline').length} active
                </span>
              </div>
              <div className="space-y-0.5 max-h-[280px] overflow-y-auto">
                {sortedWorkers.map((w) => {
                  const eth = Number(formatEther(w.performance.totalEarnings));
                  const offline = w.status === 'offline';
                  return (
                    <button
                      key={w.id}
                      type="button"
                      onClick={() => setSelectedWorker(w)}
                      className={cn(
                        'w-full flex items-center gap-4 px-4 py-3 rounded-xl text-left transition-colors',
                        offline ? 'opacity-35' : 'hover:bg-white/[0.03]',
                      )}
                    >
                      <div className={cn(
                        'w-2 h-2 rounded-full shrink-0',
                        w.status === 'healthy' ? 'bg-green-400'
                          : w.status === 'degraded' ? 'bg-yellow-400'
                            : w.status === 'unhealthy' ? 'bg-red-400'
                              : 'bg-helix-dim',
                      )} />
                      <span className="text-sm text-white font-mono flex-1">
                        {w.address.slice(0, 6)}...{w.address.slice(-4)}
                      </span>
                      <span className="text-xs text-helix-muted capitalize w-20">{w.activity}</span>
                      <span className="text-sm text-white font-mono tabular-nums w-20 text-right">
                        {eth.toFixed(2)} <span className="text-helix-muted text-xs">ADI</span>
                      </span>
                    </button>
                  );
                })}
                {sortedWorkers.length === 0 && (
                  <div className="text-sm text-helix-muted text-center py-8">No workers</div>
                )}
              </div>
            </div>

            {/* Settings & Controls */}
            <div className="col-span-2 bg-helix-surface border border-helix-border rounded-2xl p-6 flex flex-col">
              <span className="text-sm font-medium text-white mb-4">Configuration</span>

              <div className="space-y-4 flex-1">
                <div>
                  <label className="text-xs text-helix-muted block mb-1.5">Learning Rate</label>
                  <div className="px-3 py-2 bg-helix-bg border border-helix-border rounded-xl text-sm text-helix-text2 font-mono">
                    {displayLR}
                  </div>
                </div>
                <div>
                  <label className="text-xs text-helix-muted block mb-1.5">Batch Size</label>
                  <div className="px-3 py-2 bg-helix-bg border border-helix-border rounded-xl text-sm text-helix-text2 font-mono">
                    {displayBatch}
                  </div>
                </div>
                <div>
                  <label className="text-xs text-helix-muted block mb-1.5">Optimizer</label>
                  <div className="px-3 py-2 bg-helix-bg border border-helix-border rounded-xl text-sm text-helix-text2">
                    AdamW
                  </div>
                </div>
                <div>
                  <label className="text-xs text-helix-muted block mb-1.5">Loss Function</label>
                  <div className="px-3 py-2 bg-helix-bg border border-helix-border rounded-xl text-sm text-helix-text2">
                    CrossEntropy
                  </div>
                </div>
              </div>

              {/* Actions (disabled — backend doesn't support pause/resume) */}
              <div className="flex gap-3 mt-6">
                <button
                  disabled
                  title="Pause/resume not supported by backend"
                  className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-medium bg-white/5 text-helix-dim cursor-not-allowed"
                >
                  <Pause size={14} />
                  Pause
                </button>
                <button
                  disabled
                  title="Stop not supported by backend"
                  className="flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium bg-red-500/5 text-red-300/30 cursor-not-allowed"
                >
                  <Square size={14} />
                  Stop
                </button>
              </div>
            </div>
          </div>

          {/* Activity Feed */}
          <div className="bg-helix-surface border border-helix-border rounded-2xl p-6">
            <span className="text-sm font-medium text-white block mb-4">Recent Activity</span>
            <div className="space-y-0.5 max-h-[240px] overflow-y-auto">
              {activityFeed.length === 0 ? (
                <div className="text-sm text-helix-muted text-center py-6">No activity yet</div>
              ) : (
                activityFeed.map((item) => {
                  const cfg = ALERT_CONFIG[item.type] ?? ALERT_CONFIG.info;
                  const Icon = cfg.icon;
                  return (
                    <div key={item.id} className="flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/[0.02] transition-colors">
                      <Icon size={14} style={{ color: cfg.color }} className="shrink-0" />
                      <span className="text-sm text-white flex-1">{item.title}</span>
                      {item.message && item.message !== item.title && (
                        <span className="text-xs text-helix-muted">{item.message}</span>
                      )}
                      <span className="text-xs text-helix-dim tabular-nums font-mono w-14 text-right">
                        {timeAgo(item.timestamp)}
                      </span>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* ================================================================ */}
      {/* Inference Tab */}
      {/* ================================================================ */}
      {tab === 'inference' && (
        <motion.div
          key="inference"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.25 }}
          className="space-y-8"
        >
          {/* Active Requests */}
          {inferenceRequests.filter((r) => r.status === 'processing').length > 0 && (
            <div>
              <span className="text-xs uppercase tracking-wider text-helix-muted block mb-4">In Progress</span>
              <div className="space-y-4">
                {inferenceRequests.filter((r) => r.status === 'processing').map((req, i) => (
                  <motion.div
                    key={req.id}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.35, delay: i * 0.08 }}
                    className="relative bg-helix-surface border border-helix-border rounded-2xl p-8 overflow-hidden group hover:border-helix-border2 transition-colors"
                  >
                    {/* Subtle top-edge accent line */}
                    <div
                      className="absolute top-0 left-0 h-[1px] bg-gradient-to-r from-transparent via-white/20 to-transparent"
                      style={{ width: `${req.progress}%`, transition: 'width 0.6s ease-out' }}
                    />

                    <div className="flex items-center gap-8">
                      {/* Horseshoe with radial backdrop */}
                      <div className="relative shrink-0 flex items-center justify-center">
                        <div className="absolute inset-0 rounded-full bg-white/[0.02] blur-xl scale-110" />
                        <HorseshoeProgress progress={req.progress} size={150} />
                      </div>

                      {/* Details */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-3">
                            <span className="text-lg font-medium text-white">{req.model}</span>
                            <span className="text-[11px] uppercase tracking-wider px-2.5 py-0.5 rounded-md bg-white/[0.06] text-helix-text2">
                              {req.phase}
                            </span>
                          </div>
                          <button className="text-sm px-5 py-2 rounded-xl bg-white/[0.06] text-helix-text2 hover:bg-red-500/10 hover:text-red-400 transition-all font-medium">
                            Cancel
                          </button>
                        </div>

                        <div className="text-sm text-helix-muted mb-5">{req.inputLabel}</div>

                        <div className="grid grid-cols-3 gap-4">
                          <div>
                            <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-1">Workers</div>
                            <div className="text-lg font-medium text-white tabular-nums">{req.workers} nodes</div>
                          </div>
                          <div className="border-l border-helix-border pl-4">
                            <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-1">Elapsed</div>
                            <div className="text-lg font-medium text-white tabular-nums font-mono">{elapsedSince(req.created)}</div>
                          </div>
                          <div className="border-l border-helix-border pl-4">
                            <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-1">Status</div>
                            <div className="flex items-center gap-2">
                              <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                              <span className="text-lg font-medium text-white">Active</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {/* Completed Requests */}
          {inferenceRequests.filter((r) => r.status === 'completed').length > 0 && (
            <div>
              <span className="text-xs uppercase tracking-wider text-helix-muted block mb-4">Completed</span>
              <div className="space-y-4">
                {inferenceRequests.filter((r) => r.status === 'completed').map((req, i) => (
                  <motion.div
                    key={req.id}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.35, delay: i * 0.08 }}
                    className="relative bg-helix-surface border border-helix-border rounded-2xl p-8 overflow-hidden group hover:border-helix-border2 transition-colors opacity-50"
                  >
                    {/* Full accent line */}
                    <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent" />

                    <div className="flex items-center gap-8">
                      {/* Horseshoe with radial backdrop */}
                      <div className="relative shrink-0 flex items-center justify-center">
                        <div className="absolute inset-0 rounded-full bg-white/[0.02] blur-xl scale-110" />
                        <HorseshoeProgress progress={100} size={150} />
                      </div>

                      {/* Details */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-3">
                            <span className="text-lg font-medium text-white">{req.model}</span>
                            <span className="text-[11px] uppercase tracking-wider px-2.5 py-0.5 rounded-md bg-green-400/10 text-green-400">
                              Complete
                            </span>
                          </div>
                          <span className="text-sm text-helix-muted font-mono tabular-nums">{timeAgo(req.created)}</span>
                        </div>

                        <div className="text-sm text-helix-muted mb-5">
                          Predicted &ldquo;{req.result}&rdquo; with {req.confidence != null ? `${(req.confidence * 100).toFixed(1)}%` : '\u2014'} confidence
                        </div>

                        <div className="grid grid-cols-3 gap-4">
                          <div>
                            <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-1">Result</div>
                            <div className="text-lg font-medium text-white tabular-nums">{req.result}</div>
                          </div>
                          <div className="border-l border-helix-border pl-4">
                            <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-1">Confidence</div>
                            <div className="text-lg font-medium text-white tabular-nums font-mono">
                              {req.confidence != null ? `${(req.confidence * 100).toFixed(1)}%` : '\u2014'}
                            </div>
                          </div>
                          <div className="border-l border-helix-border pl-4">
                            <div className="text-[10px] uppercase tracking-wider text-helix-muted mb-1">Duration</div>
                            <div className="text-lg font-medium text-white tabular-nums font-mono">
                              {req.duration != null ? formatDuration(req.duration) : '\u2014'}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {inferenceRequests.length === 0 && (
            <div className="bg-helix-surface border border-helix-border rounded-2xl p-16 text-center">
              <div className="text-sm text-helix-muted">No inference requests yet</div>
              <div className="text-xs text-helix-dim mt-2">Run inference from the Inference page</div>
            </div>
          )}
        </motion.div>
      )}

      {/* Worker Detail Modal */}
      <AnimatePresence>
        {selectedWorker && (
          <WorkerModal worker={selectedWorker} onClose={() => setSelectedWorker(null)} />
        )}
      </AnimatePresence>
    </motion.div>
  );
}
