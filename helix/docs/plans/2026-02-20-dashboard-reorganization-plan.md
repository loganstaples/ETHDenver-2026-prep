# Dashboard Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize the Helix dashboard from a scattered layout into a portfolio-focused dashboard for logged-in users, a public marketplace for logged-out users, a merged smart model detail page, and a public worker directory on the Network page.

**Architecture:** Next.js App Router pages. Hooks abstract contract/API data. Pages are `'use client'` components. Framer Motion for animations. Tailwind + custom helix-* CSS vars for styling. The reorganization touches 6 pages and the sidebar but keeps all existing hooks intact.

**Tech Stack:** Next.js 14, React 18, Tailwind CSS, Framer Motion, Recharts, wagmi/viem, lucide-react, RainbowKit

**Design doc:** `docs/plans/2026-02-20-dashboard-reorganization-design.md`

---

## Task 1: Update Sidebar Navigation

**Files:**
- Modify: `helix/dashboard/src/components/layout/Sidebar.tsx`

**Step 1: Update the nav items array**

Replace the current navigation items with the new structure. The sidebar should show different items based on wallet connection status. Read the file first, then replace the nav items.

Current nav items (approx lines 15-40):
```typescript
const navItems = [
  { label: 'Dashboard', icon: LayoutDashboard, href: '/dashboard' },
  { label: 'Models', icon: Boxes, href: '/' },
  { label: 'My Models', icon: Layers, href: '/my-models' },
  { label: 'Train', icon: Cpu, href: '/train' },
  { label: 'Inference', icon: Sparkles, href: '/inference' },
  { label: 'Network', icon: Globe, href: '/network' },
  { label: 'Settings', icon: Settings, href: '/settings' },
];
```

New nav items — add `useAccount` import from wagmi and conditionally render:
```typescript
import { useAccount } from 'wagmi';

// Inside component:
const { isConnected } = useAccount();

const navItems = [
  // Auth-required items shown only when connected
  ...(isConnected ? [
    { label: 'Dashboard', icon: LayoutDashboard, href: '/dashboard' },
  ] : []),
  { label: 'Marketplace', icon: Boxes, href: '/' },
  ...(isConnected ? [
    { label: 'Train', icon: Cpu, href: '/train' },
    { label: 'Inference', icon: Sparkles, href: '/inference' },
  ] : []),
  { label: 'Network', icon: Globe, href: '/network' },
  { label: 'Settings', icon: Settings, href: '/settings' },
];
```

Also update the `isActive` check — currently Models matches on `'/'` and `/models/*`. Change to Marketplace matching on `'/'` and `/models/*`:
```typescript
const isActive = item.href === '/'
  ? (pathname === '/' || pathname.startsWith('/models'))
  : pathname.startsWith(item.href);
```

Remove the `Layers` icon import since My Models is gone.

**Step 2: Verify the sidebar renders correctly**

Run: `cd helix/dashboard && npm run build`
Expected: No build errors. Sidebar should show new nav structure.

**Step 3: Commit**

```bash
git add helix/dashboard/src/components/layout/Sidebar.tsx
git commit -m "refactor(dashboard): update sidebar nav — portfolio dashboard, marketplace, remove My Models"
```

---

## Task 2: Build the New Dashboard (Portfolio) Page

This is the biggest task. The new dashboard replaces the training/inference monitor with a financial portfolio view.

**Files:**
- Modify: `helix/dashboard/src/app/dashboard/page.tsx` (complete rewrite)

**Step 1: Read the current dashboard page**

Read `helix/dashboard/src/app/dashboard/page.tsx` to understand the full current implementation before rewriting.

**Step 2: Write the new portfolio dashboard**

The new dashboard page needs these sections:
1. **Stat cards row** — Models Owned, Total Revenue, Active Training, Inferences Served
2. **Revenue chart** — 30-day line chart + commission breakdown by model
3. **My Models list** — sortable table with create button, links to `/models/[id]`
4. **Active Training** — compact cards for in-progress jobs
5. **Recent Activity feed** — unified chronological feed

Key imports and hooks needed:
```typescript
'use client';
import { useState, useMemo } from 'react';
import { useAccount } from 'wagmi';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard, TrendingUp, Cpu, Sparkles, Plus,
  ArrowUpRight, Clock, ChevronRight, Layers, Activity,
  ShoppingCart, Zap, BarChart3, Search
} from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { useModelRegistry } from '@/hooks/useModelRegistry';
import { useDashboardSessions } from '@/hooks/useDashboardSessions';
```

**Component structure outline:**

```typescript
export default function DashboardPage() {
  const { isConnected, address } = useAccount();
  const router = useRouter();
  const { models, isLoading: modelsLoading, isContractDeployed } = useModelRegistry();
  const { sessions, activeSession, losses, events } = useDashboardSessions();

  const [modelSort, setModelSort] = useState<'revenue' | 'accuracy' | 'newest' | 'name'>('revenue');
  const [modelSearch, setModelSearch] = useState('');
  const [createModalOpen, setCreateModalOpen] = useState(false);

  // Computed stats
  const totalRevenue = useMemo(() =>
    models.reduce((sum, m) => sum + Number(m.feesAccrued), 0), [models]);
  const totalInferences = useMemo(() => /* derive from events or models */, [events, models]);
  const activeTraining = useMemo(() =>
    sessions.filter(s => s.status === 'active'), [sessions]);

  // Revenue chart data (mock 30-day for demo, or derive from events)
  const revenueData = useMemo(() => generateRevenueData(models, events), [models, events]);

  // Commission breakdown by model
  const commissionBreakdown = useMemo(() =>
    models
      .filter(m => m.feesAccrued > 0)
      .sort((a, b) => Number(b.feesAccrued) - Number(a.feesAccrued))
      .slice(0, 5),
    [models]
  );

  // Sorted/filtered models
  const sortedModels = useMemo(() => {
    let filtered = models.filter(m =>
      m.name.toLowerCase().includes(modelSearch.toLowerCase()) ||
      m.slug.toLowerCase().includes(modelSearch.toLowerCase())
    );
    switch (modelSort) {
      case 'revenue': return filtered.sort((a, b) => Number(b.feesAccrued) - Number(a.feesAccrued));
      case 'accuracy': return filtered.sort((a, b) => {
        const aAcc = Math.max(...(a.versions.map(v => v.accuracy) || [0]));
        const bAcc = Math.max(...(b.versions.map(v => v.accuracy) || [0]));
        return bAcc - aAcc;
      });
      case 'newest': return filtered.sort((a, b) => b.createdAt - a.createdAt);
      case 'name': return filtered.sort((a, b) => a.name.localeCompare(b.name));
    }
  }, [models, modelSort, modelSearch]);

  // Not connected — prompt to connect
  if (!isConnected) {
    return (/* EmptyState: "Connect your wallet to view your portfolio" */);
  }

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-helix-text">Portfolio</h1>
          <p className="text-sm text-helix-muted mt-1">Your models, revenue, and activity</p>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Layers} label="Models Owned" value={models.length} />
        <StatCard icon={TrendingUp} label="Total Revenue" value={formatAdi(totalRevenue)} suffix="ADI" />
        <StatCard icon={Cpu} label="Active Training" value={activeTraining.length} />
        <StatCard icon={Sparkles} label="Inferences Served" value={formatNumber(totalInferences)} />
      </div>

      {/* Revenue Section — chart + breakdown side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-3 rounded-2xl border border-helix-border bg-helix-surface/50 p-6">
          <h2 className="text-sm font-medium text-helix-muted mb-4">Revenue (30d)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={revenueData}>
              <defs>
                <linearGradient id="revGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#818cf8" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#6b7280' }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fontSize: 11, fill: '#6b7280' }} tickLine={false} axisLine={false} width={40} />
              <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a4a', borderRadius: 12 }} />
              <Area type="monotone" dataKey="revenue" stroke="#818cf8" fill="url(#revGrad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="lg:col-span-2 rounded-2xl border border-helix-border bg-helix-surface/50 p-6">
          <h2 className="text-sm font-medium text-helix-muted mb-4">Top Earners</h2>
          {/* Horizontal bar chart or list showing commission per model */}
          {commissionBreakdown.map((model, i) => (
            <div key={model.tokenId} className="flex items-center gap-3 py-2">
              <div className="w-2 h-2 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />
              <span className="text-sm text-helix-text flex-1 truncate">{model.name}</span>
              <span className="text-sm font-mono text-helix-muted">{formatAdi(model.feesAccrued)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* My Models */}
      <div className="rounded-2xl border border-helix-border bg-helix-surface/50 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-medium text-helix-muted">My Models</h2>
          <div className="flex items-center gap-3">
            {/* Search input */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-helix-dim" />
              <input
                value={modelSearch}
                onChange={e => setModelSearch(e.target.value)}
                placeholder="Search..."
                className="pl-9 pr-3 py-1.5 text-xs rounded-lg bg-helix-bg/50 border border-helix-border text-helix-text placeholder:text-helix-dim focus:outline-none focus:border-helix-border2 w-40"
              />
            </div>
            {/* Sort pills */}
            <div className="flex gap-1">
              {(['revenue', 'accuracy', 'newest', 'name'] as const).map(s => (
                <button
                  key={s}
                  onClick={() => setModelSort(s)}
                  className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                    modelSort === s ? 'bg-white/10 text-helix-text' : 'text-helix-dim hover:text-helix-muted'
                  }`}
                >
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                </button>
              ))}
            </div>
            <button
              onClick={() => setCreateModalOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-indigo-500/20 text-indigo-300 hover:bg-indigo-500/30 transition-colors"
            >
              <Plus className="w-3.5 h-3.5" /> Create
            </button>
          </div>
        </div>
        {/* Model rows */}
        <div className="space-y-2">
          {sortedModels.map(model => (
            <motion.div
              key={model.tokenId}
              layoutId={`model-${model.tokenId}`}
              onClick={() => router.push(`/models/${model.tokenId}`)}
              className="flex items-center gap-4 p-3 rounded-xl hover:bg-white/[0.03] cursor-pointer transition-colors group"
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-helix-text truncate">{model.name}</span>
                  <span className="text-xs text-helix-dim">v{latestVersion(model)}</span>
                  {!model.isPublic && <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-helix-dim">Private</span>}
                </div>
              </div>
              <div className="flex items-center gap-6 text-xs text-helix-muted">
                <span>{bestAccuracy(model)}% acc</span>
                <span className="font-mono">{formatAdi(model.feesAccrued)} ADI</span>
                <ChevronRight className="w-4 h-4 text-helix-dim opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Active Training */}
      {activeTraining.length > 0 && (
        <div className="rounded-2xl border border-helix-border bg-helix-surface/50 p-6">
          <h2 className="text-sm font-medium text-helix-muted mb-4">Active Training</h2>
          <div className="space-y-3">
            {activeTraining.map(session => (
              <motion.div key={session.id} className="flex items-center gap-4 p-3 rounded-xl bg-white/[0.02]">
                <div className="flex-1">
                  <span className="text-sm text-helix-text">{session.name}</span>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                      <motion.div className="h-full rounded-full bg-indigo-500" style={{ width: `${sessionProgress(session)}%` }} />
                    </div>
                    <span className="text-xs text-helix-dim">{sessionProgress(session)}%</span>
                  </div>
                </div>
                <button
                  onClick={() => router.push('/train')}
                  className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
                >
                  View <ArrowUpRight className="w-3 h-3 inline" />
                </button>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Activity */}
      <div className="rounded-2xl border border-helix-border bg-helix-surface/50 p-6">
        <h2 className="text-sm font-medium text-helix-muted mb-4">Recent Activity</h2>
        <div className="space-y-1">
          {recentActivity.slice(0, 20).map((event, i) => (
            <div key={i} className="flex items-center gap-3 py-2 text-xs">
              <ActivityIcon type={event.type} />
              <span className="text-helix-text flex-1">{event.message}</span>
              <span className="text-helix-dim">{formatRelativeTime(event.timestamp)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Create Model Modal — reuse from old my-models page */}
      {createModalOpen && <CreateModelModal onClose={() => setCreateModalOpen(false)} />}
    </div>
  );
}
```

**Helper components to include in the same file:**

```typescript
function StatCard({ icon: Icon, label, value, suffix }: { icon: any; label: string; value: string | number; suffix?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-helix-border bg-helix-surface/50 p-5"
    >
      <div className="flex items-center gap-2 mb-3">
        <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center">
          <Icon className="w-4 h-4 text-helix-muted" />
        </div>
      </div>
      <div className="text-2xl font-semibold text-helix-text tracking-tight">
        {value}{suffix && <span className="text-sm font-normal text-helix-dim ml-1">{suffix}</span>}
      </div>
      <div className="text-xs text-helix-dim mt-1">{label}</div>
    </motion.div>
  );
}
```

**Important:** The `CreateModelModal` component should be extracted from the current `helix/dashboard/src/app/my-models/page.tsx` (it currently lives inline there). Either import it as a shared component or copy the relevant code into the new dashboard.

**Step 3: Extract CreateModelModal into shared component**

Create `helix/dashboard/src/components/models/CreateModelModal.tsx` by extracting the modal from `my-models/page.tsx`. It needs:
- `useModelRegistry().createModel()`
- Slug, name, description inputs
- Slug validation regex
- onClose callback prop

**Step 4: Extract AddVersionModal into shared component**

Create `helix/dashboard/src/components/models/AddVersionModal.tsx` by extracting from `my-models/page.tsx`. This will be needed by the merged model detail page. It needs:
- `useModelRegistry().addVersion()`
- `useSignMessage()` for key derivation
- Version, weights file, accuracy inputs
- Encryption + 0G upload flow
- `modelTokenId` and `onClose` props

**Step 5: Build and verify**

Run: `cd helix/dashboard && npm run build`
Expected: Clean build, no errors.

**Step 6: Commit**

```bash
git add helix/dashboard/src/app/dashboard/page.tsx helix/dashboard/src/components/models/
git commit -m "feat(dashboard): new portfolio dashboard with revenue, models, activity feed"
```

---

## Task 3: Build the Network Page (Public Worker Directory)

**Files:**
- Modify: `helix/dashboard/src/app/network/page.tsx` (rewrite to be the worker directory)

**Step 1: Read the current network page**

Read `helix/dashboard/src/app/network/page.tsx` to understand what exists.

**Step 2: Rewrite as public worker directory**

The network page currently shows a topology graph and node list. Enhance it to be the primary worker directory with:
1. Summary stats bar (workers online, total compute, uptime)
2. Filter/sort controls (status, reputation, type)
3. Worker cards with full detail (address, reputation, GPU, utilization, jobs, uptime, latency, status)
4. Network-wide stats (total jobs, avg latency, cheaters found)

Use the existing `useNodes()` hook which already provides everything needed: `filteredNodes`, `networkStats`, `updateFilter`. Also use `useBackendApi()` for worker health data from the old dashboard.

Key structure:
```typescript
'use client';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Globe, Server, Shield, Cpu, Activity, ChevronDown,
  ChevronRight, Wifi, WifiOff, Star, BarChart3, Zap, AlertTriangle
} from 'lucide-react';
import { useNodes } from '@/hooks/useNodes';

export default function NetworkPage() {
  const { filteredNodes, networkStats, isLoading, updateFilter } = useNodes({ includeOffline: true });
  const [activeFilter, setActiveFilter] = useState<'all' | 'active' | 'idle' | 'offline'>('all');
  const [sortBy, setSortBy] = useState<'reputation' | 'utilization' | 'jobs' | 'uptime'>('reputation');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const filteredByStatus = filteredNodes.filter(n =>
    activeFilter === 'all' ? true : n.status === activeFilter
  );

  const sorted = [...filteredByStatus].sort((a, b) => {
    switch (sortBy) {
      case 'reputation': return b.reputation - a.reputation;
      case 'utilization': return (b.metrics?.gpu ?? b.metrics?.cpu ?? 0) - (a.metrics?.gpu ?? a.metrics?.cpu ?? 0);
      case 'jobs': return b.roundsCompleted - a.roundsCompleted;
      case 'uptime': return /* derive uptime */ 0;
    }
  });

  const onlineCount = filteredNodes.filter(n => n.status !== 'offline').length;

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-helix-text">Network</h1>
        <p className="text-sm text-helix-muted mt-1">Workers powering the Helix network</p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Server} label="Workers Online" value={onlineCount} total={filteredNodes.length} />
        <StatCard icon={Zap} label="Total Compute" value={`${networkStats?.totalStaked ?? 0} TFLOPS`} />
        <StatCard icon={Activity} label="Network Uptime" value={`${networkStats?.networkUptime ?? 99}%`} />
        <StatCard icon={Shield} label="Avg Reputation" value={`${networkStats?.averageReputation?.toFixed(1) ?? '—'}%`} />
      </div>

      {/* Filters + Sort */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1">
          {(['all', 'active', 'idle', 'offline'] as const).map(f => (
            <button
              key={f}
              onClick={() => setActiveFilter(f)}
              className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
                activeFilter === f ? 'bg-white/10 text-helix-text' : 'text-helix-dim hover:text-helix-muted'
              }`}
            >
              {f === 'all' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1)}
              {f === 'all' && ` (${filteredNodes.length})`}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 text-xs text-helix-dim">
          <span>Sort:</span>
          {(['reputation', 'utilization', 'jobs', 'uptime'] as const).map(s => (
            <button
              key={s}
              onClick={() => setSortBy(s)}
              className={`px-2 py-1 rounded-md transition-colors ${
                sortBy === s ? 'bg-white/10 text-helix-text' : 'hover:text-helix-muted'
              }`}
            >
              {s.charAt(0).toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Worker Cards */}
      <div className="space-y-2">
        {sorted.map(worker => (
          <WorkerCard
            key={worker.id}
            worker={worker}
            expanded={expandedId === worker.id}
            onToggle={() => setExpandedId(expandedId === worker.id ? null : worker.id)}
          />
        ))}
      </div>

      {/* Network Stats Footer */}
      <div className="grid grid-cols-3 gap-4">
        <div className="rounded-2xl border border-helix-border bg-helix-surface/50 p-5 text-center">
          <div className="text-2xl font-semibold text-helix-text">{networkStats?.totalProofs?.toLocaleString() ?? '—'}</div>
          <div className="text-xs text-helix-dim mt-1">Total Jobs Processed</div>
        </div>
        <div className="rounded-2xl border border-helix-border bg-helix-surface/50 p-5 text-center">
          <div className="text-2xl font-semibold text-helix-text">—</div>
          <div className="text-xs text-helix-dim mt-1">Avg Latency</div>
        </div>
        <div className="rounded-2xl border border-helix-border bg-helix-surface/50 p-5 text-center">
          <div className="text-2xl font-semibold text-helix-text">—</div>
          <div className="text-xs text-helix-dim mt-1">Cheaters Detected</div>
        </div>
      </div>
    </div>
  );
}
```

**WorkerCard component** (in same file):
```typescript
function WorkerCard({ worker, expanded, onToggle }: { worker: WorkerNode; expanded: boolean; onToggle: () => void }) {
  const isOnline = worker.status !== 'offline';
  const utilization = worker.metrics?.gpu ?? worker.metrics?.cpu ?? 0;
  const reputation = worker.reputation ?? 0;
  const stars = Math.round(reputation / 20); // 0-5 stars from 0-100%

  return (
    <motion.div
      layout
      className="rounded-xl border border-helix-border bg-helix-surface/50 overflow-hidden"
    >
      <button onClick={onToggle} className="w-full flex items-center gap-4 p-4 text-left hover:bg-white/[0.02] transition-colors">
        {/* Status dot */}
        <div className={`w-2.5 h-2.5 rounded-full ${isOnline ? 'bg-emerald-400' : 'bg-zinc-600'}`} />

        {/* Address */}
        <span className="text-sm font-mono text-helix-text w-32 truncate">{truncateAddress(worker.address)}</span>

        {/* Reputation stars */}
        <div className="flex items-center gap-1">
          {Array.from({ length: 5 }).map((_, i) => (
            <Star key={i} className={`w-3 h-3 ${i < stars ? 'text-amber-400 fill-amber-400' : 'text-zinc-700'}`} />
          ))}
          <span className="text-xs text-helix-dim ml-1">{reputation.toFixed(1)}%</span>
        </div>

        {/* GPU/Resource */}
        <span className="text-xs text-helix-muted flex-1 truncate">
          {worker.capabilities?.gpuModel ?? 'CPU'} · {utilization.toFixed(0)}% util
        </span>

        {/* Jobs */}
        <span className="text-xs text-helix-dim">{worker.roundsCompleted} jobs</span>

        {/* Status badge */}
        <span className={`text-[10px] px-2 py-0.5 rounded-full ${
          isOnline ? 'bg-emerald-500/10 text-emerald-400' : 'bg-zinc-500/10 text-zinc-500'
        }`}>
          {worker.status}
        </span>

        <ChevronDown className={`w-4 h-4 text-helix-dim transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-helix-border"
          >
            <div className="p-4 grid grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
              <div><span className="text-helix-dim">CPU</span><br/><span className="text-helix-text">{worker.metrics?.cpu?.toFixed(1)}%</span></div>
              <div><span className="text-helix-dim">Memory</span><br/><span className="text-helix-text">{worker.metrics?.memory?.toFixed(1)}%</span></div>
              <div><span className="text-helix-dim">GPU Memory</span><br/><span className="text-helix-text">{worker.metrics?.gpuMemory?.toFixed(1) ?? '—'}%</span></div>
              <div><span className="text-helix-dim">Rounds</span><br/><span className="text-helix-text">{worker.roundsParticipated} / {worker.roundsCompleted}</span></div>
              <div><span className="text-helix-dim">Proofs Submitted</span><br/><span className="text-helix-text">{worker.proofsSubmitted}</span></div>
              <div><span className="text-helix-dim">Success Rate</span><br/><span className="text-helix-text">{worker.proofsSubmitted > 0 ? ((1 - worker.proofsFailed / worker.proofsSubmitted) * 100).toFixed(1) : '—'}%</span></div>
              <div><span className="text-helix-dim">Staked</span><br/><span className="text-helix-text">{formatBigint(worker.stakedAmount)} ADI</span></div>
              <div><span className="text-helix-dim">Earnings</span><br/><span className="text-helix-text">{formatBigint(worker.earningsTotal)} ADI</span></div>
              {worker.capabilities && (
                <>
                  <div><span className="text-helix-dim">Can Train</span><br/><span className="text-helix-text">{worker.capabilities.canTrain ? 'Yes' : 'No'}</span></div>
                  <div><span className="text-helix-dim">Can Aggregate</span><br/><span className="text-helix-text">{worker.capabilities.canAggregate ? 'Yes' : 'No'}</span></div>
                  <div><span className="text-helix-dim">Max Batch</span><br/><span className="text-helix-text">{worker.capabilities.maxBatchSize}</span></div>
                  <div><span className="text-helix-dim">Region</span><br/><span className="text-helix-text">{worker.location?.region ?? '—'}</span></div>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
```

Keep the existing `NetworkGraph` component if it exists and is useful — it can go above the worker list as a visual complement. The key is that the worker directory is the primary content now, not a secondary element.

**Step 3: Build and verify**

Run: `cd helix/dashboard && npm run build`
Expected: Clean build.

**Step 4: Commit**

```bash
git add helix/dashboard/src/app/network/page.tsx
git commit -m "feat(dashboard): network page as public worker directory with reputation, resources, status"
```

---

## Task 4: Merge Model Detail Pages

**Files:**
- Modify: `helix/dashboard/src/app/models/[id]/page.tsx` (enhance with owner features)
- Delete: `helix/dashboard/src/app/my-models/[id]/page.tsx` (after merging)
- Delete: `helix/dashboard/src/app/my-models/page.tsx` (after extracting modals)

**Step 1: Read both model detail pages**

Read:
- `helix/dashboard/src/app/models/[id]/page.tsx` (public view)
- `helix/dashboard/src/app/my-models/[id]/page.tsx` (owner view)

Understand the full feature set of both.

**Step 2: Enhance the public model detail page with owner features**

The merged page at `/models/[id]` should:
1. Use `useAccount()` to check if connected wallet is the model owner
2. Show all current public view content for everyone
3. Conditionally show owner controls when `address === model.owner`

Add to the existing page:
- Import `useModelRegistry` for write operations
- Import `useSignMessage` for key derivation
- Import `AddVersionModal` (extracted in Task 2)
- Add `isOwner` computed boolean: `address?.toLowerCase() === model?.owner?.toLowerCase()`
- In Overview tab: add inline edit for name/description when `isOwner`
- Add Settings section (or tab) when `isOwner`: visibility toggle, fee config, sale toggle/price
- In Versions tab: add "Add Version" button when `isOwner`, with download/decrypt for owner
- Show revenue stats card when `isOwner`

The owner controls should feel integrated, not bolted on. Use subtle visual cues (edit icons that appear on hover, toggle switches, inline inputs) rather than a separate tab for settings.

**Approach for owner UI integration:**

```typescript
const { address } = useAccount();
const isOwner = address?.toLowerCase() === model?.owner?.toLowerCase();

// In Overview section, after the description card:
{isOwner && (
  <div className="rounded-2xl border border-helix-border bg-helix-surface/50 p-6 space-y-4">
    <h3 className="text-sm font-medium text-helix-muted">Model Settings</h3>

    {/* Visibility toggle */}
    <div className="flex items-center justify-between">
      <span className="text-sm text-helix-text">Public</span>
      <ToggleSwitch checked={model.isPublic} onChange={() => setPublic(model.tokenId, !model.isPublic)} />
    </div>

    {/* Inference fee */}
    <div className="flex items-center justify-between">
      <span className="text-sm text-helix-text">Inference Fee</span>
      <div className="flex items-center gap-2">
        <input type="number" value={fee} onChange={...} className="w-20 text-right ..." />
        <span className="text-xs text-helix-dim">bps</span>
      </div>
    </div>

    {/* For sale toggle + price */}
    <div className="flex items-center justify-between">
      <span className="text-sm text-helix-text">For Sale</span>
      <ToggleSwitch checked={model.forSale} onChange={() => setForSale(model.tokenId, !model.forSale)} />
    </div>
    {model.forSale && (
      <div className="flex items-center justify-between">
        <span className="text-sm text-helix-text">Sale Price</span>
        <div className="flex items-center gap-2">
          <input type="number" value={salePrice} onChange={...} className="w-24 text-right ..." />
          <span className="text-xs text-helix-dim">ADI</span>
        </div>
      </div>
    )}

    {/* Revenue */}
    <div className="flex items-center justify-between pt-2 border-t border-helix-border">
      <span className="text-sm text-helix-text">Fees Accrued</span>
      <span className="text-sm font-mono text-indigo-400">{formatAdi(model.feesAccrued)} ADI</span>
    </div>
  </div>
)}
```

In the Versions tab, add the "Add Version" button and the AddVersionModal:
```typescript
{isOwner && (
  <button onClick={() => setAddVersionOpen(true)} className="...">
    <Plus className="w-3.5 h-3.5" /> Add Version
  </button>
)}
```

And for each version row, if `isOwner`, show a download button that decrypts weights.

**Step 3: Delete the old my-models pages**

After confirming the merged page works:
- Delete `helix/dashboard/src/app/my-models/[id]/page.tsx`
- Delete `helix/dashboard/src/app/my-models/page.tsx`
- Delete the `helix/dashboard/src/app/my-models/` directory entirely

**Step 4: Update any internal links**

Search the codebase for any links to `/my-models`:
- In the old dashboard page (already rewritten in Task 2)
- In any other component that links to my-models
- Update them to link to `/models/[id]` or `/dashboard`

Run: `grep -r "my-models" helix/dashboard/src/ --include="*.tsx" --include="*.ts" -l`

Update all occurrences.

**Step 5: Build and verify**

Run: `cd helix/dashboard && npm run build`
Expected: Clean build, no broken imports or dead links.

**Step 6: Commit**

```bash
git add -A helix/dashboard/src/app/models/ helix/dashboard/src/app/my-models/ helix/dashboard/src/components/models/
git commit -m "feat(dashboard): merge model detail pages — owner controls on /models/[id], remove /my-models"
```

---

## Task 5: Update Root Page Labels

**Files:**
- Modify: `helix/dashboard/src/app/page.tsx` (minor label changes)

**Step 1: Read the root page header area**

Read lines 1-50 and 150-230 of `helix/dashboard/src/app/page.tsx` to find the page title and mode toggle labels.

**Step 2: Update labels**

- Page title: Change from "Models" (or whatever it says) to "Marketplace"
- The mode toggle already has "Inference" and "Marketplace" modes — this is fine
- Ensure the "Mine" filter still works but now links to `/models/[id]` not `/my-models/[id]`

Check if clicking a model card currently navigates to `/my-models/[id]` for owned models — if so, update to `/models/[id]`.

**Step 3: Build and verify**

Run: `cd helix/dashboard && npm run build`

**Step 4: Commit**

```bash
git add helix/dashboard/src/app/page.tsx
git commit -m "refactor(dashboard): update marketplace labels and links"
```

---

## Task 6: Polish and Visual Refinement

**Files:**
- All modified pages from Tasks 1-5

**Step 1: Review all pages for visual consistency**

Check that:
- All pages use the same card style: `rounded-2xl border border-helix-border bg-helix-surface/50`
- Stat cards have consistent sizing and spacing
- Text hierarchy is consistent: `text-2xl` for page titles, `text-sm font-medium text-helix-muted` for section headers, `text-xs text-helix-dim` for metadata
- Framer Motion animations are smooth and consistent (use `initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}`)
- Empty states have clear messaging and calls to action

**Step 2: Add page transitions**

Ensure smooth transitions between pages. Each page's main container should have a fade-in:
```typescript
<motion.div
  initial={{ opacity: 0 }}
  animate={{ opacity: 1 }}
  transition={{ duration: 0.3 }}
  className="space-y-8 max-w-7xl mx-auto"
>
```

**Step 3: Final build check**

Run: `cd helix/dashboard && npm run build && npm run lint`
Expected: Clean build, no lint errors.

**Step 4: Commit**

```bash
git add helix/dashboard/src/
git commit -m "style(dashboard): polish visual consistency and page transitions"
```

---

## Task 7: Verify Full Flow

**Step 1: Run the dev server**

Run: `cd helix/dashboard && npm run dev`

**Step 2: Manual verification checklist**

Test each flow:
- [ ] Sidebar shows correct items when disconnected (Marketplace, Network, Settings)
- [ ] Sidebar shows all items when connected (Dashboard, Marketplace, Train, Inference, Network, Settings)
- [ ] `/` shows marketplace with inference/buyer toggle
- [ ] `/dashboard` shows portfolio with stat cards, revenue, models, activity
- [ ] `/dashboard` "Create" button opens modal and creates a model
- [ ] `/models/[id]` shows public view for non-owner
- [ ] `/models/[id]` shows owner controls (settings, add version) for owner
- [ ] `/network` shows worker directory with filters and expandable cards
- [ ] `/train` and `/inference` work unchanged
- [ ] No dead links to `/my-models`
- [ ] `/my-models` returns 404 (removed)

**Step 3: Final commit if any fixes needed**

```bash
git add helix/dashboard/src/
git commit -m "fix(dashboard): address issues found during verification"
```
