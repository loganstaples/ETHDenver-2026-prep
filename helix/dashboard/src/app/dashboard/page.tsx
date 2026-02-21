'use client';

import { useState, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useRouter } from 'next/navigation';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  Layers,
  DollarSign,
  Activity,
  Zap,
  Search,
  Plus,
  ChevronRight,
  Wallet,
  TrendingUp,
  CheckCircle,
  AlertTriangle,
  Info,
  XCircle,
  Globe,
  Lock,
  Tag,
  GitBranch,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useModelRegistry, type ModelWithVersions } from '@/hooks/useModelRegistry';
import { useDashboardSessions, type TrainingEvent } from '@/hooks/useDashboardSessions';
import { useAccount } from 'wagmi';
import { CreateModelModal } from '@/components/models/CreateModelModal';

// ============================================================================
// Constants & helpers
// ============================================================================

type SortOption = 'revenue' | 'accuracy' | 'newest' | 'name';

const SORT_OPTIONS: { id: SortOption; label: string }[] = [
  { id: 'revenue', label: 'Revenue' },
  { id: 'accuracy', label: 'Accuracy' },
  { id: 'newest', label: 'Newest' },
  { id: 'name', label: 'Name' },
];

function sortModels(models: ModelWithVersions[], sort: SortOption): ModelWithVersions[] {
  const sorted = [...models];
  switch (sort) {
    case 'revenue':
      return sorted.sort((a, b) => b.feesAccrued - a.feesAccrued);
    case 'accuracy': {
      const bestAcc = (m: ModelWithVersions) =>
        m.versions.reduce((best, v) => (v.accuracy > best ? v.accuracy : best), 0);
      return sorted.sort((a, b) => bestAcc(b) - bestAcc(a));
    }
    case 'newest':
      return sorted.sort((a, b) => b.createdAt - a.createdAt);
    case 'name':
      return sorted.sort((a, b) => a.name.localeCompare(b.name));
    default:
      return sorted;
  }
}

function formatADI(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return value.toFixed(2);
}

function timeAgo(ts: number): string {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

// ============================================================================
// Stacked area chart colors for per-model revenue
// ============================================================================

const STACK_COLORS = [
  { stroke: '#818cf8', fill: '#818cf8' },  // indigo
  { stroke: '#a78bfa', fill: '#a78bfa' },  // violet
  { stroke: '#38bdf8', fill: '#38bdf8' },  // sky
  { stroke: '#4ade80', fill: '#4ade80' },  // emerald
  { stroke: '#fbbf24', fill: '#fbbf24' },  // amber
  { stroke: '#fb7185', fill: '#fb7185' },  // rose
];

// ============================================================================
// Mock month-to-date revenue data (stacked per model)
// ============================================================================

function generateRevenueData(models: ModelWithVersions[]) {
  if (models.length === 0) return { data: [], modelNames: [] };

  const now = new Date();
  const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
  const days = Math.floor((now.getTime() - startOfMonth.getTime()) / 86400000) + 1;

  const stackModels = [...models].sort((a, b) => b.feesAccrued - a.feesAccrued).slice(0, 6);

  const data = [];
  for (let i = 0; i < days; i++) {
    const date = new Date(startOfMonth.getTime() + i * 86400000);
    const dayLabel = date.toLocaleDateString('en', { month: 'short', day: 'numeric' });
    const progress = days > 1 ? i / (days - 1) : 1;

    const entry: Record<string, string | number> = { day: dayLabel };

    stackModels.forEach((model, mi) => {
      const revenue = model.feesAccrued || 1;
      // Each model gets a unique growth curve with seeded variation
      const phase = 0.3 + mi * 0.08;
      const curve = 1 / (1 + Math.exp(-8 * (progress - phase)));
      const noise = 1 + (Math.sin(i * (2.3 + mi * 0.7)) * 0.2 + Math.sin(i * (1.1 + mi * 0.5)) * 0.1);
      const daily = (revenue / Math.max(days, 1)) * curve * 2.5 * noise;
      entry[model.name] = Math.max(0, Number(daily.toFixed(4)));
    });

    data.push(entry);
  }

  return { data, modelNames: stackModels.map((m) => m.name) };
}

// ============================================================================
// Alert icon/color helpers
// ============================================================================

const ALERT_CONFIG: Record<string, { icon: typeof Info; color: string; bg: string }> = {
  info: { icon: Info, color: '#818cf8', bg: 'bg-indigo-500/10' },
  success: { icon: CheckCircle, color: '#4ade80', bg: 'bg-green-500/10' },
  warning: { icon: AlertTriangle, color: '#fbbf24', bg: 'bg-yellow-500/10' },
  error: { icon: XCircle, color: '#f87171', bg: 'bg-red-500/10' },
};

// Accent dots for top earners
const EARNER_COLORS = [
  'bg-indigo-400',
  'bg-violet-400',
  'bg-sky-400',
  'bg-emerald-400',
  'bg-amber-400',
  'bg-rose-400',
];

// ============================================================================
// Stat Card
// ============================================================================

function StatCard({
  label,
  value,
  icon: Icon,
  delay = 0,
  accent,
}: {
  label: string;
  value: string;
  icon: typeof Layers;
  delay?: number;
  accent?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className="rounded-2xl border border-helix-border bg-helix-surface/50 p-6"
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-helix-muted">{label}</span>
        <div className={cn('p-2 rounded-xl', accent || 'bg-white/[0.04]')}>
          <Icon size={16} className="text-helix-muted" />
        </div>
      </div>
      <div className="text-2xl font-semibold text-helix-text tracking-tight">{value}</div>
    </motion.div>
  );
}

// ============================================================================
// Main Page
// ============================================================================

export default function DashboardPage() {
  const router = useRouter();
  const { isConnected } = useAccount();
  const {
    models,
    isLoading: isModelsLoading,
    isContractDeployed: _isContractDeployed,
    refetch: refetchModels,
  } = useModelRegistry();

  const {
    sessions,
    activeSession: _activeSession,
    events,
  } = useDashboardSessions();

  const [search, setSearch] = useState('');
  const [sort, setSort] = useState<SortOption>('revenue');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [totalInferences, setTotalInferences] = useState<number | null>(null);

  // ── Fetch inference stats from server ───────────────────
  useEffect(() => {
    async function fetchStats() {
      try {
        const res = await fetch('/api/models/stats');
        if (!res.ok) return;
        const data: Record<string, { inferenceCount: number }> = await res.json();
        const total = Object.values(data).reduce((sum, s) => sum + s.inferenceCount, 0);
        setTotalInferences(total);
      } catch {
        // Non-fatal
      }
    }
    fetchStats();
    const interval = setInterval(fetchStats, 15000);
    return () => clearInterval(interval);
  }, []);

  // ── Computed values ──────────────────────────────────────────

  const totalRevenue = useMemo(
    () => models.reduce((sum, m) => sum + m.feesAccrued, 0),
    [models],
  );

  const activeSessions = useMemo(
    () => sessions.filter((s) => s.status === 'running' || s.status === 'starting'),
    [sessions],
  );

  const { data: revenueData, modelNames: revenueModelNames } = useMemo(() => generateRevenueData(models), [models]);

  const topEarners = useMemo(
    () => [...models].sort((a, b) => b.feesAccrued - a.feesAccrued).slice(0, 6),
    [models],
  );

  const filteredModels = useMemo(() => {
    let list = models;
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(
        (m) =>
          m.name.toLowerCase().includes(q) ||
          m.slug.toLowerCase().includes(q) ||
          m.description.toLowerCase().includes(q),
      );
    }
    return sortModels(list, sort);
  }, [models, search, sort]);

  // ── Activity feed from events + sessions + models ──────

  const activityFeed = useMemo(() => {
    type FeedItem = {
      id: string;
      type: string;
      title: string;
      message: string;
      timestamp: number;
    };
    const items: FeedItem[] = [];

    // 1. Real-time WebSocket events (from current browser session)
    events.forEach((e, i) => {
      const evt = e.event;
      const ts = (e as TrainingEvent & { receivedAt?: number }).receivedAt ?? (Date.now() - (events.length - i) * 100);

      if (evt.type === 'training_step') {
        const step = evt.step as number;
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
      } else if (evt.type === 'cheater_slashed') {
        items.push({
          id: `slashed-${i}`,
          type: 'warning',
          title: 'Cheater slashed',
          message: `Worker ${evt.party_index} stake slashed`,
          timestamp: ts,
        });
      } else if (evt.type === 'recovery_completed') {
        items.push({
          id: `recovery-${i}`,
          type: 'success',
          title: 'Training recovered',
          message: `${evt.honest_workers} workers continuing`,
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

    // 2. Derive activity from training sessions (persisted on backend)
    sessions.forEach((session) => {
      const name = session.model_name || `Session ${session.session_id.slice(0, 8)}`;
      // started_at is unix seconds from backend
      const sessionTs = session.started_at > 0
        ? session.started_at * 1000
        : Date.now() - 60000;

      if (session.status === 'complete') {
        items.push({
          id: `session-complete-${session.session_id}`,
          type: 'success',
          title: 'Training completed',
          message: session.accuracy
            ? `${name} \u2014 ${(session.accuracy * 100).toFixed(1)}% accuracy`
            : name,
          timestamp: sessionTs,
        });
      } else if (session.status === 'failed') {
        items.push({
          id: `session-failed-${session.session_id}`,
          type: 'error',
          title: 'Training failed',
          message: name,
          timestamp: sessionTs,
        });
      } else if (session.status === 'running' || session.status === 'starting') {
        items.push({
          id: `session-active-${session.session_id}`,
          type: 'info',
          title: 'Training in progress',
          message: session.current_step > 0
            ? `${name} \u2014 step ${session.current_step}/${session.total_steps}`
            : name,
          timestamp: sessionTs,
        });
      }
    });

    // 3. Model creation events (from on-chain data)
    models.forEach((model) => {
      if (model.createdAt > 0) {
        items.push({
          id: `model-created-${model.tokenId}`,
          type: 'success',
          title: 'Model registered',
          message: model.name,
          // On-chain timestamps are unix seconds
          timestamp: model.createdAt * 1000,
        });
      }

      // Version additions
      model.versions.forEach((version, vi) => {
        if (version.timestamp > 0) {
          items.push({
            id: `version-added-${model.tokenId}-${vi}`,
            type: 'info',
            title: 'Version added',
            message: `${model.name} v${version.semver}${version.accuracy > 0 ? ` \u2014 ${(version.accuracy * 100).toFixed(1)}%` : ''}`,
            timestamp: version.timestamp * 1000,
          });
        }
      });
    });

    // Deduplicate by id (WS events take priority)
    const seen = new Set<string>();
    const deduped = items.filter((item) => {
      if (seen.has(item.id)) return false;
      seen.add(item.id);
      return true;
    });

    return deduped.sort((a, b) => b.timestamp - a.timestamp).slice(0, 20);
  }, [events, sessions, models]);

  const chartTooltipStyle = {
    backgroundColor: '#111113',
    border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: 12,
    fontSize: 12,
    fontFamily: 'var(--font-geist-mono)',
  };

  // ── Not connected state ──────────────────────────────────────

  if (!isConnected) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="space-y-8 max-w-7xl mx-auto"
      >
        <div>
          <h1 className="text-2xl font-semibold text-helix-text">Portfolio</h1>
          <p className="text-sm text-helix-muted mt-1">Your models, revenue, and activity</p>
        </div>
        <div className="rounded-2xl border border-helix-border bg-helix-surface/50 flex flex-col items-center justify-center py-24 px-8">
          <div className="p-4 rounded-2xl bg-white/[0.04] mb-5">
            <Wallet size={32} className="text-helix-dim" />
          </div>
          <p className="text-lg font-medium text-helix-text mb-2">Connect your wallet</p>
          <p className="text-sm text-helix-muted text-center max-w-sm">
            Connect your wallet to view your model portfolio, track revenue, and manage training sessions.
          </p>
        </div>
      </motion.div>
    );
  }

  // ── Main portfolio view ──────────────────────────────────────

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="space-y-8 max-w-7xl mx-auto"
      >
        {/* ── 1. Header ────────────────────────────────────────── */}
        <div>
          <h1 className="text-2xl font-semibold text-helix-text">Portfolio</h1>
          <p className="text-sm text-helix-muted mt-1">Your models, revenue, and activity</p>
        </div>

        {/* ── 2. Stat Cards ────────────────────────────────────── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Models Owned"
            value={isModelsLoading ? '--' : String(models.length)}
            icon={Layers}
            delay={0}
            accent="bg-indigo-500/10"
          />
          <StatCard
            label="Total Revenue"
            value={isModelsLoading ? '--' : `${formatADI(totalRevenue)} ADI`}
            icon={DollarSign}
            delay={0.05}
            accent="bg-emerald-500/10"
          />
          <StatCard
            label="Active Training"
            value={String(activeSessions.length)}
            icon={Activity}
            delay={0.1}
            accent="bg-sky-500/10"
          />
          <StatCard
            label="Inferences Served"
            value={totalInferences === null ? '--' : String(totalInferences)}
            icon={Zap}
            delay={0.15}
            accent="bg-amber-500/10"
          />
        </div>

        {/* ── 3. Revenue Section ───────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.15 }}
          className="grid grid-cols-1 lg:grid-cols-5 gap-4"
        >
          {/* Revenue chart (3/5) */}
          <div className="lg:col-span-3 rounded-2xl border border-helix-border bg-helix-surface/50 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <span className="text-sm font-medium text-helix-muted">Revenue</span>
                <p className="text-sm text-helix-dim mt-0.5">Month to date</p>
              </div>
              {revenueData.length > 0 && (
                <div className="flex items-center gap-2">
                  <TrendingUp size={14} className="text-indigo-400" />
                  <span className="text-sm font-medium text-indigo-300">{formatADI(totalRevenue)} ADI</span>
                </div>
              )}
            </div>
            {revenueData.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-64 text-center">
                <div className="p-4 rounded-2xl bg-white/[0.03] mb-4">
                  <TrendingUp size={24} className="text-white/20" />
                </div>
                <p className="text-[15px] text-white/40 font-medium">No revenue yet</p>
                <p className="text-sm text-white/20 mt-1">Register a model to start earning</p>
              </div>
            ) : (
              <>
                {/* Legend */}
                <div className="flex flex-wrap gap-x-4 gap-y-1 mb-4">
                  {revenueModelNames.map((name, i) => (
                    <div key={name} className="flex items-center gap-1.5">
                      <div
                        className="w-2.5 h-2.5 rounded-sm"
                        style={{ backgroundColor: STACK_COLORS[i % STACK_COLORS.length].fill, opacity: 0.8 }}
                      />
                      <span className="text-sm text-helix-dim truncate max-w-[100px]">{name}</span>
                    </div>
                  ))}
                </div>
                <div className="h-52">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={revenueData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                      <defs>
                        {revenueModelNames.map((name, i) => {
                          const color = STACK_COLORS[i % STACK_COLORS.length].fill;
                          return (
                            <linearGradient key={name} id={`stackGrad-${i}`} x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor={color} stopOpacity={0.4} />
                              <stop offset="100%" stopColor={color} stopOpacity={0.05} />
                            </linearGradient>
                          );
                        })}
                      </defs>
                      <XAxis
                        dataKey="day"
                        tick={{ fill: '#63636e', fontSize: 10, fontFamily: 'var(--font-geist-mono)' }}
                        axisLine={{ stroke: '#1e1e22' }}
                        tickLine={false}
                        tickMargin={6}
                        interval={Math.max(0, Math.floor(revenueData.length / 6) - 1)}
                      />
                      <YAxis
                        tick={{ fill: '#63636e', fontSize: 10, fontFamily: 'var(--font-geist-mono)' }}
                        axisLine={false}
                        tickLine={false}
                        tickMargin={4}
                        width={38}
                      />
                      <Tooltip
                        contentStyle={chartTooltipStyle}
                        labelStyle={{ color: '#63636e' }}
                        formatter={(value: number | undefined, name?: string) => [`${(value ?? 0).toFixed(4)} ADI`, name ?? '']}
                      />
                      {revenueModelNames.map((name, i) => {
                        const color = STACK_COLORS[i % STACK_COLORS.length];
                        return (
                          <Area
                            key={name}
                            type="monotone"
                            dataKey={name}
                            stackId="revenue"
                            stroke={color.stroke}
                            fill={`url(#stackGrad-${i})`}
                            strokeWidth={1.5}
                            dot={false}
                          />
                        );
                      })}
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </div>

          {/* Top earners (2/5) */}
          <div className="lg:col-span-2 rounded-2xl border border-helix-border bg-helix-surface/50 p-6">
            <span className="text-sm font-medium text-helix-muted block mb-4">Top Earners</span>
            {topEarners.length === 0 ? (
              <div className="flex items-center justify-center h-40 text-sm text-helix-dim">
                No models yet
              </div>
            ) : (
              <div className="space-y-1">
                {topEarners.map((model, i) => (
                  <button
                    key={model.tokenId}
                    onClick={() => router.push(`/models/${model.tokenId}`)}
                    className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/[0.03] transition-colors text-left group"
                  >
                    <div className={cn('w-2 h-2 rounded-full shrink-0', EARNER_COLORS[i % EARNER_COLORS.length])} />
                    <span className="text-sm text-helix-text flex-1 truncate group-hover:text-white transition-colors">
                      {model.name}
                    </span>
                    <span className="text-sm font-mono text-helix-muted tabular-nums">
                      {formatADI(model.feesAccrued)}
                    </span>
                    <span className="text-sm text-helix-dim">ADI</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </motion.div>

        {/* ── 4. My Models Section ─────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
        >
          {/* Models header */}
          <div className="flex flex-col sm:flex-row sm:items-center gap-4 mb-5">
            <span className="text-lg font-semibold text-helix-text flex-shrink-0">My Models</span>

            {/* Search */}
            <div className="relative flex-1 max-w-sm">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-helix-dim" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search models..."
                className="w-full pl-10 pr-4 py-2 bg-helix-bg border border-helix-border rounded-xl text-sm text-helix-text placeholder:text-helix-dim focus:outline-none focus:border-helix-border2 transition-colors"
              />
            </div>

            {/* Sort pills */}
            <div className="flex items-center gap-1">
              {SORT_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => setSort(opt.id)}
                  className={cn(
                    'px-4 py-2 rounded-xl text-sm font-medium transition-all',
                    sort === opt.id
                      ? 'bg-white/10 text-helix-text'
                      : 'text-helix-dim hover:text-helix-muted',
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            {/* Create button */}
            <button
              onClick={() => setShowCreateModal(true)}
              className="flex items-center gap-2 px-5 py-2 rounded-xl bg-white text-black text-sm font-semibold hover:bg-white/90 transition-colors flex-shrink-0"
            >
              <Plus size={16} />
              Create
            </button>
          </div>

          {/* Model cards grid */}
          {isModelsLoading ? (
            <div className="flex items-center justify-center py-20">
              <div className="w-6 h-6 border-2 border-helix-dim border-t-white rounded-full animate-spin" />
            </div>
          ) : filteredModels.length === 0 ? (
            <div className="rounded-2xl border border-helix-border bg-helix-surface/50 flex flex-col items-center justify-center py-20 px-8">
              {models.length === 0 ? (
                <>
                  <Layers size={28} className="text-helix-dim mb-4" />
                  <p className="text-base text-helix-muted mb-2">No models yet</p>
                  <p className="text-sm text-helix-dim text-center max-w-sm">
                    Create your first model to start training and earning revenue.
                  </p>
                </>
              ) : (
                <>
                  <Search size={24} className="text-helix-dim mb-3" />
                  <p className="text-base text-helix-muted">No models match your search</p>
                </>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {filteredModels.map((model, i) => {
                const bestVersion = model.versions.length > 0
                  ? model.versions.reduce((best, v) => (v.accuracy > best.accuracy ? v : best), model.versions[0])
                  : null;
                const bestAccuracy = bestVersion?.accuracy ?? 0;
                return (
                  <motion.button
                    key={model.tokenId}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: i * 0.04 }}
                    onClick={() => router.push(`/models/${model.tokenId}`)}
                    className="rounded-2xl border border-helix-border bg-helix-surface/50 p-5 hover:bg-helix-surface/80 hover:border-helix-border2 transition-all text-left group"
                  >
                    {/* Top row: name + badges */}
                    <div className="flex items-start justify-between gap-3 mb-3">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2.5">
                          <h3 className="text-base font-semibold text-helix-text group-hover:text-white transition-colors truncate">
                            {model.name}
                          </h3>
                          {bestVersion && (
                            <span className="text-sm font-mono px-2 py-0.5 rounded-md shrink-0 text-helix-muted bg-white/[0.04]">
                              v{bestVersion.semver}
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-helix-muted mt-1 font-mono">{model.slug}</p>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        {model.isPublic ? (
                          <span className="flex items-center gap-1 text-sm text-emerald-400 bg-emerald-500/10 px-2 py-1 rounded-lg">
                            <Globe size={12} />
                            Public
                          </span>
                        ) : (
                          <span className="flex items-center gap-1 text-sm text-helix-muted bg-white/[0.04] px-2 py-1 rounded-lg">
                            <Lock size={12} />
                            Private
                          </span>
                        )}
                        {model.forSale && (
                          <span className="flex items-center gap-1 text-sm text-amber-400 bg-amber-500/10 px-2 py-1 rounded-lg">
                            <Tag size={12} />
                            For Sale
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Description */}
                    {model.description && (
                      <p className="text-sm text-helix-muted mb-4 line-clamp-2 leading-relaxed">
                        {model.description}
                      </p>
                    )}

                    {/* Stats row */}
                    <div className="flex items-center gap-6 pt-3 border-t border-helix-border/50">
                      <div>
                        <span className="text-sm text-helix-dim block mb-0.5">Accuracy</span>
                        <span className="text-sm font-mono font-medium text-helix-text tabular-nums">
                          {bestAccuracy > 0 ? `${(bestAccuracy * 100).toFixed(1)}%` : '\u2014'}
                        </span>
                      </div>
                      <div>
                        <span className="text-sm text-helix-dim block mb-0.5">Revenue</span>
                        <span className="text-sm font-mono font-medium text-helix-text tabular-nums">
                          {formatADI(model.feesAccrued)} <span className="text-sm text-helix-dim">ADI</span>
                        </span>
                      </div>
                      <div>
                        <span className="text-sm text-helix-dim block mb-0.5">Versions</span>
                        <span className="text-sm font-mono font-medium text-helix-text tabular-nums flex items-center gap-1">
                          <GitBranch size={12} className="text-helix-dim" />
                          {model.versions.length}
                        </span>
                      </div>
                      {model.architecture && (
                        <div className="ml-auto">
                          <span className="text-sm text-helix-dim block mb-0.5">Architecture</span>
                          <span className="text-sm font-mono text-helix-muted truncate max-w-[120px] block">
                            {model.architecture}
                          </span>
                        </div>
                      )}
                      <ChevronRight
                        size={18}
                        className="text-helix-dim group-hover:text-helix-muted transition-colors shrink-0 ml-auto"
                      />
                    </div>
                  </motion.button>
                );
              })}
            </div>
          )}
        </motion.div>

        {/* ── 5. Active Training ───────────────────────────────── */}
        {activeSessions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.25 }}
          >
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-helix-muted">Active Training</span>
              <button
                onClick={() => router.push('/train')}
                className="text-sm text-indigo-400 hover:text-indigo-300 transition-colors"
              >
                View all
              </button>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {activeSessions.map((session, i) => {
                const progress = session.total_steps > 0
                  ? (session.current_step / session.total_steps) * 100
                  : 0;

                return (
                  <motion.div
                    key={session.session_id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: i * 0.05 }}
                    className="rounded-2xl border border-helix-border bg-helix-surface/50 p-5"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                        <span className="text-sm font-medium text-helix-text truncate">
                          {session.model_name || `Session ${session.session_id.slice(0, 8)}`}
                        </span>
                      </div>
                      <span className="text-sm text-helix-dim font-mono tabular-nums">
                        {progress.toFixed(0)}%
                      </span>
                    </div>

                    {/* Progress bar */}
                    <div className="h-1.5 bg-helix-border rounded-full overflow-hidden mb-3">
                      <motion.div
                        className="h-full rounded-full bg-indigo-400"
                        initial={{ width: 0 }}
                        animate={{ width: `${progress}%` }}
                        transition={{ duration: 0.5, ease: 'easeOut' }}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-sm text-helix-dim">
                        Step {session.current_step} / {session.total_steps}
                      </span>
                      <button
                        onClick={() => router.push('/train')}
                        className="text-sm text-indigo-400 hover:text-indigo-300 transition-colors"
                      >
                        View
                      </button>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        )}

        {/* ── 6. Recent Activity ───────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.3 }}
          className="rounded-2xl border border-helix-border bg-helix-surface/50 p-6"
        >
          <span className="text-sm font-medium text-helix-muted block mb-4">Recent Activity</span>
          <div className="space-y-0.5 max-h-[320px] overflow-y-auto">
            {activityFeed.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-10">
                <Activity size={20} className="text-helix-dim mb-2" />
                <p className="text-sm text-helix-dim">No recent activity</p>
                <p className="text-sm text-helix-dim mt-1">Activity from training sessions will appear here</p>
              </div>
            ) : (
              activityFeed.map((item, i) => {
                const cfg = ALERT_CONFIG[item.type] ?? ALERT_CONFIG.info;
                const Icon = cfg.icon;
                return (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, x: -4 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.2, delay: i * 0.02 }}
                    className="flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/[0.02] transition-colors"
                  >
                    <div className={cn('p-1.5 rounded-lg', cfg.bg)}>
                      <Icon size={12} style={{ color: cfg.color }} />
                    </div>
                    <span className="text-sm text-helix-text flex-1 truncate">{item.title}</span>
                    {item.message && item.message !== item.title && (
                      <span className="text-sm text-helix-muted truncate max-w-[200px]">{item.message}</span>
                    )}
                    <span className="text-sm text-helix-dim tabular-nums font-mono w-14 text-right shrink-0">
                      {timeAgo(item.timestamp)}
                    </span>
                  </motion.div>
                );
              })
            )}
          </div>
        </motion.div>
      </motion.div>

      {/* Create Model Modal */}
      <AnimatePresence>
        {showCreateModal && (
          <CreateModelModal
            onClose={() => setShowCreateModal(false)}
            onCreated={() => {
              setShowCreateModal(false);
              refetchModels();
            }}
          />
        )}
      </AnimatePresence>
    </>
  );
}
