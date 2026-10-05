# Dashboard Real Backend Wiring Implementation Plan

**Goal:** Remove useless marketplace stats, wire all frontend to real backend data, rewrite Activity page with clique subgraphs grouped by inference request on user-owned models, add worker panel with reputation scores on Dashboard page.

**Architecture:** Backend-first approach — create a `useBackendApi` hook as a unified REST client that wraps all Rust backend calls. Each page feature gets a dedicated hook calling real endpoints. The Activity page network graph uses actual inference request data grouped into force-directed cliques per request.

**Tech Stack:** Next.js 14, React, TypeScript, Recharts, D3-force, Wagmi, Tailwind CSS, Rust (helix-node HTTP API, helix-client dashboard server)

---

### Task 1: Remove Marketplace Stats

**Files:**
- Modify: `helix/dashboard/src/app/page.tsx` (lines 672-790)

**Step 1: Remove stats section from marketplace page**

Remove the entire stats section (lines 737-791) — both the inference mode stats (`StatCard` for Public Models, Total Versions, Best Accuracy, Creators) and marketplace mode stats (`StatCard` for Models for Sale, Avg Price, Price Range, Creators).

Also remove unused imports and computed values:
- Remove `StatCard` import (line 25)
- Remove `Tag`, `Trophy`, `User`, `DollarSign`, `Store`, `ArrowUpDown` from lucide imports (lines 11, 12, 17, 20, 21)
- Remove `totalVersions`, `bestAccuracy`, `uniqueCreators` computations (lines 673-678)
- Remove `forSaleModels` and `marketplaceStats` computations (lines 681-688)

**Step 2: Verify page renders correctly**

Run: `cd helix/dashboard && npm run build`
Expected: Build succeeds with no errors

**Step 3: Commit**

```bash
git add helix/dashboard/src/app/page.tsx
git commit -m "feat(dashboard): remove marketplace stats from models page"
```

---

### Task 2: Extend Rust Backend Worker/Events API

**Files:**
- Modify: `helix/crates/helix-node/src/api/http.rs`
- Modify: `helix/crates/helix-client/src/dashboard.rs`

**Step 1: Extend WorkerInfo in helix-node http.rs**

In `helix/crates/helix-node/src/api/http.rs`, change `WorkerInfo` from:
```rust
pub struct WorkerInfo {
    pub id: String,
    pub status: String,
    pub rounds_completed: u64,
}
```
to:
```rust
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct WorkerInfo {
    pub id: String,
    pub status: String,
    pub rounds_completed: u64,
    pub rounds_participated: u64,
    pub cpu_load: u8,
    pub memory_mb: u64,
    pub reputation_score: f64,
    pub success_rate: f64,
    pub earnings_wei: u64,
    pub last_heartbeat: u64,
    pub capabilities: WorkerCapabilities,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct WorkerCapabilities {
    pub can_train: bool,
    pub can_prove: bool,
    pub can_aggregate: bool,
    pub gpu_model: Option<String>,
    pub max_batch_size: u64,
}
```

Update the `/workers` handler to populate these fields from `OrchestratorSnapshot` and the orchestrator's `WorkerState` (which has `load`, `failures`, `rounds_participated`, `rounds_completed`).

**Step 2: Add owner filter to dashboard events endpoint**

In `helix/crates/helix-client/src/dashboard.rs`, modify the `GET /api/events` handler to accept an optional `?owner=<address>` query parameter. When present, filter events to only include those associated with models owned by that address. The events list already stored in `DashboardState.events` has model/session info — filter by matching the session's model owner.

Add a new route for inference events specifically:
```rust
// GET /api/events/inference?owner=<address>
async fn get_inference_events(
    Query(params): Query<HashMap<String, String>>,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    let events = state.events.read().await;
    let owner_filter = params.get("owner").map(|o| o.to_lowercase());

    let filtered: Vec<_> = events.iter()
        .filter(|e| {
            // Filter by event type (inference-related)
            let event_type = e.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if event_type != "inference_request" && event_type != "inference_complete" {
                return false;
            }
            // Filter by owner if specified
            if let Some(ref owner) = owner_filter {
                let event_owner = e.get("model_owner")
                    .and_then(|o| o.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                return event_owner == *owner;
            }
            true
        })
        .cloned()
        .collect();

    Json(filtered)
}
```

**Step 3: Add reputation breakdown endpoint to dashboard**

Add `GET /api/workers/:id/reputation` that returns the reputation breakdown. Since the dashboard.rs already has `registered_workers` and the helix-node has `ReputationManager`, expose a combined view:

```rust
#[derive(Serialize)]
struct WorkerReputationResponse {
    worker_id: String,
    overall_score: f64,       // 0-100
    success_rate: f64,        // 0.0-1.0
    responsiveness: f64,      // 0-100
    validity: f64,            // 0-100
    bandwidth: f64,           // 0-100
    uptime: f64,              // 0-100
    total_interactions: u64,
    rounds_participated: u64,
    rounds_succeeded: u64,
    is_banned: bool,
}
```

**Step 4: Build and test Rust changes**

Run: `cd helix && cargo check -p helix-node -p helix-client`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add helix/crates/helix-node/src/api/http.rs helix/crates/helix-client/src/dashboard.rs
git commit -m "feat(backend): extend worker info with reputation, resources, and owner-filtered events"
```

---

### Task 3: Create useBackendApi Hook

**Files:**
- Create: `helix/dashboard/src/hooks/useBackendApi.ts`

**Step 1: Create the unified backend API hook**

```typescript
'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

// ============================================================================
// Types matching Rust backend responses
// ============================================================================

export interface BackendWorker {
  id: string;
  status: string;
  rounds_completed: number;
  rounds_participated: number;
  cpu_load: number;
  memory_mb: number;
  reputation_score: number;  // 0.0-1.0 from backend
  success_rate: number;
  earnings_wei: number;
  last_heartbeat: number;
  capabilities: {
    can_train: boolean;
    can_prove: boolean;
    can_aggregate: boolean;
    gpu_model: string | null;
    max_batch_size: number;
  };
}

export interface BackendHealth {
  status: string;
  role: string;
  workers: number;
  current_round: number | null;
  completed_rounds: number;
  connected_peers: number;
  uptime_secs: number;
  mpc: {
    enabled: boolean;
    active_session: boolean;
    session_id: string | null;
    num_parties: number;
    party_index: number | null;
  };
  memory_bytes: number;
  fault_tolerance: {
    healthy_workers: number;
    degraded_workers: number;
    failed_workers: number;
    system_healthy: boolean;
  };
}

export interface BackendPeer {
  id: string;
  address: string;
  last_seen: number;
  reputation: number;
}

export interface BackendRoundStatus {
  worker_count: number;
  available_workers: number;
  computing_workers: number;
  current_round: {
    round_id: number;
    phase: string;
    gradients_received: number;
    workers_assigned: number;
    commitment_hash: string | null;
  } | null;
  completed_rounds: number;
  workers: BackendWorker[];
}

export interface BackendEvent {
  type: string;
  timestamp?: number;
  session_id?: string;
  model_owner?: string;
  model_token_id?: number;
  requester?: string;
  prediction?: number;
  confidence?: number;
  fee?: number;
  latency?: number;
  workers?: number;
  [key: string]: unknown;
}

export interface WorkerReputation {
  worker_id: string;
  overall_score: number;
  success_rate: number;
  responsiveness: number;
  validity: number;
  bandwidth: number;
  uptime: number;
  total_interactions: number;
  rounds_participated: number;
  rounds_succeeded: number;
  is_banned: boolean;
}

// ============================================================================
// Hook
// ============================================================================

interface UseBackendApiOptions {
  refreshInterval?: number;
  enabled?: boolean;
}

export function useBackendApi(options: UseBackendApiOptions = {}) {
  const { refreshInterval = 5000, enabled = true } = options;

  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [workers, setWorkers] = useState<BackendWorker[]>([]);
  const [peers, setPeers] = useState<BackendPeer[]>([]);
  const [roundStatus, setRoundStatus] = useState<BackendRoundStatus | null>(null);
  const [events, setEvents] = useState<BackendEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  const fetchJson = useCallback(async <T>(path: string): Promise<T | null> => {
    try {
      const res = await fetch(`${API_BASE}${path}`, { signal: AbortSignal.timeout(10000) });
      if (!res.ok) return null;
      return await res.json() as T;
    } catch {
      return null;
    }
  }, []);

  const refresh = useCallback(async () => {
    if (!enabled) return;

    const [h, w, rs, ev] = await Promise.all([
      fetchJson<BackendHealth>('/health'),
      fetchJson<BackendWorker[]>('/api/workers'),
      fetchJson<BackendRoundStatus>('/round/status'),
      fetchJson<BackendEvent[]>('/api/events'),
    ]);

    if (!mountedRef.current) return;

    if (h) { setHealth(h); setIsConnected(true); }
    else { setIsConnected(false); }

    if (w) setWorkers(w);
    if (rs) setRoundStatus(rs);
    if (ev) setEvents(Array.isArray(ev) ? ev : []);

    setIsLoading(false);
    setError(null);
  }, [enabled, fetchJson]);

  const fetchInferenceEvents = useCallback(async (ownerAddress: string): Promise<BackendEvent[]> => {
    const data = await fetchJson<BackendEvent[]>(`/api/events/inference?owner=${ownerAddress}`);
    return data ?? [];
  }, [fetchJson]);

  const fetchWorkerReputation = useCallback(async (workerId: string): Promise<WorkerReputation | null> => {
    return fetchJson<WorkerReputation>(`/api/workers/${workerId}/reputation`);
  }, [fetchJson]);

  const fetchLosses = useCallback(async (sessionId: string): Promise<{ losses: number[]; current_step: number; total_steps: number } | null> => {
    return fetchJson(`/api/training/sessions/${sessionId}/losses`);
  }, [fetchJson]);

  useEffect(() => {
    mountedRef.current = true;
    refresh();
    const interval = setInterval(refresh, refreshInterval);
    return () => {
      mountedRef.current = false;
      clearInterval(interval);
    };
  }, [refresh, refreshInterval]);

  return {
    health,
    workers,
    peers,
    roundStatus,
    events,
    isLoading,
    isConnected,
    error,
    refresh,
    fetchInferenceEvents,
    fetchWorkerReputation,
    fetchLosses,
  };
}
```

**Step 2: Build check**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: No type errors

**Step 3: Commit**

```bash
git add helix/dashboard/src/hooks/useBackendApi.ts
git commit -m "feat(dashboard): add useBackendApi hook for unified backend REST client"
```

---

### Task 4: Add Worker Panel with Reputation to Dashboard Page

**Files:**
- Modify: `helix/dashboard/src/app/dashboard/page.tsx`

**Step 1: Import the new hook and add worker reputation panel**

Add import at top of file:
```typescript
import { useBackendApi, type BackendWorker } from '@/hooks/useBackendApi';
```

Add new icons to lucide import:
```typescript
import { Shield, Users, ChevronRight } from 'lucide-react';
```

**Step 2: Add reputation score display helper**

Add after the existing helper functions (around line 58):

```typescript
function ReputationBadge({ score }: { score: number }) {
  const display = Math.round(score * 100);
  const color = display >= 80 ? 'text-green-400 bg-green-400/10'
    : display >= 50 ? 'text-yellow-400 bg-yellow-400/10'
    : 'text-red-400 bg-red-400/10';
  return (
    <span className={cn('text-xs font-mono font-medium px-2 py-0.5 rounded-md tabular-nums', color)}>
      {display}
    </span>
  );
}
```

**Step 3: Wire useBackendApi into the page component**

Inside `DashboardPage()`, after the existing hook calls, add:

```typescript
const { workers: backendWorkers, health: backendHealth, roundStatus } = useBackendApi({ refreshInterval: 5000 });
const [workerTab, setWorkerTab] = useState<'active' | 'inactive'>('active');
const [expandedWorker, setExpandedWorker] = useState<string | null>(null);
```

Add computed worker lists:
```typescript
const activeWorkers = useMemo(
  () => backendWorkers.filter((w) => w.status !== 'offline' && w.status !== 'Excluded'),
  [backendWorkers],
);
const inactiveWorkers = useMemo(
  () => backendWorkers.filter((w) => w.status === 'offline' || w.status === 'Excluded'),
  [backendWorkers],
);
const displayWorkers = workerTab === 'active' ? activeWorkers : inactiveWorkers;
```

**Step 4: Add the worker panel after the Activity Feed section (around line 870)**

Insert before the closing `</motion.div>` of the training tab (before line 872):

```tsx
{/* Worker Panel */}
<div className="bg-helix-surface border border-helix-border rounded-2xl overflow-hidden">
  <div className="flex items-center justify-between px-6 py-4 border-b border-helix-border">
    <div className="flex items-center gap-3">
      <Users size={16} className="text-helix-muted" />
      <span className="text-sm font-medium text-white">Workers</span>
      {backendHealth && (
        <span className="text-xs text-helix-muted">
          {backendHealth.fault_tolerance.healthy_workers} healthy /
          {backendHealth.fault_tolerance.degraded_workers} degraded /
          {backendHealth.fault_tolerance.failed_workers} failed
        </span>
      )}
    </div>
    <div className="flex bg-helix-bg rounded-lg p-0.5 border border-helix-border">
      {(['active', 'inactive'] as const).map((t) => (
        <button
          key={t}
          onClick={() => setWorkerTab(t)}
          className={cn(
            'px-3 py-1 rounded-md text-xs font-medium transition-all capitalize',
            workerTab === t ? 'bg-white text-black' : 'text-helix-muted hover:text-white',
          )}
        >
          {t} ({t === 'active' ? activeWorkers.length : inactiveWorkers.length})
        </button>
      ))}
    </div>
  </div>

  {/* Table Header */}
  <div className="grid grid-cols-[1fr_70px_60px_60px_60px_70px_60px_70px_60px] gap-2 px-6 py-2.5 border-b border-helix-border bg-helix-bg/30 text-xs text-helix-muted">
    <span>Worker ID</span>
    <span>Status</span>
    <span>Rep</span>
    <span>CPU</span>
    <span>Mem</span>
    <span>Rounds</span>
    <span>Success</span>
    <span>Earnings</span>
    <span></span>
  </div>

  {/* Worker Rows */}
  <div className="max-h-[320px] overflow-y-auto">
    {displayWorkers.map((w) => (
      <div key={w.id}>
        <button
          type="button"
          onClick={() => setExpandedWorker(expandedWorker === w.id ? null : w.id)}
          className="w-full grid grid-cols-[1fr_70px_60px_60px_60px_70px_60px_70px_60px] gap-2 px-6 py-3 border-b border-helix-border/50 hover:bg-white/[0.015] text-xs transition-colors items-center"
        >
          <span className="text-white font-mono truncate">
            {w.id.slice(0, 8)}...{w.id.slice(-4)}
          </span>
          <span className={cn(
            'px-1.5 py-0.5 rounded text-center capitalize',
            w.status === 'Available' ? 'bg-green-400/10 text-green-400'
              : w.status === 'Computing' ? 'bg-blue-400/10 text-blue-400'
              : w.status === 'Assigned' ? 'bg-yellow-400/10 text-yellow-400'
              : 'bg-helix-border text-helix-muted',
          )}>
            {w.status.toLowerCase()}
          </span>
          <ReputationBadge score={w.reputation_score} />
          <span className="text-white font-mono tabular-nums">{w.cpu_load}%</span>
          <span className="text-white font-mono tabular-nums">{Math.round(w.memory_mb)}MB</span>
          <span className="text-helix-text2 font-mono tabular-nums">
            {w.rounds_completed}/{w.rounds_participated}
          </span>
          <span className="text-white font-mono tabular-nums">
            {(w.success_rate * 100).toFixed(0)}%
          </span>
          <span className="text-white font-mono tabular-nums">
            {(w.earnings_wei / 1e18).toFixed(3)}
          </span>
          <ChevronRight size={12} className={cn(
            'text-helix-dim transition-transform',
            expandedWorker === w.id && 'rotate-90',
          )} />
        </button>

        {/* Expanded Detail */}
        <AnimatePresence>
          {expandedWorker === w.id && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="px-6 py-4 bg-helix-bg/50 border-b border-helix-border/50 space-y-3">
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <div className="text-xs text-helix-muted mb-2">Capabilities</div>
                    <div className="flex flex-wrap gap-1.5">
                      {w.capabilities.can_train && <span className="text-xs px-2 py-0.5 rounded bg-white/5 text-helix-text2">Train</span>}
                      {w.capabilities.can_prove && <span className="text-xs px-2 py-0.5 rounded bg-white/5 text-helix-text2">Prove</span>}
                      {w.capabilities.can_aggregate && <span className="text-xs px-2 py-0.5 rounded bg-white/5 text-helix-text2">Aggregate</span>}
                      {w.capabilities.gpu_model && <span className="text-xs px-2 py-0.5 rounded bg-white/5 text-helix-text2">{w.capabilities.gpu_model}</span>}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-helix-muted mb-2">Resources</div>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-helix-text2 w-10">CPU</span>
                        <div className="flex-1 h-1.5 bg-helix-border rounded-full overflow-hidden">
                          <div className="h-full rounded-full bg-white/70" style={{ width: `${Math.min(100, w.cpu_load)}%` }} />
                        </div>
                        <span className="text-xs text-white font-mono w-8 text-right">{w.cpu_load}%</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-helix-muted mb-2">Reputation (0-100)</div>
                    <div className="text-2xl font-mono font-semibold text-white tabular-nums">
                      {Math.round(w.reputation_score * 100)}
                    </div>
                    <div className="text-xs text-helix-muted mt-1">
                      {w.rounds_participated > 0
                        ? `${w.rounds_succeeded ?? w.rounds_completed} of ${w.rounds_participated} rounds successful`
                        : 'No rounds yet'}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    ))}
    {displayWorkers.length === 0 && (
      <div className="flex items-center justify-center py-8 text-sm text-helix-muted">
        No {workerTab} workers
      </div>
    )}
  </div>
</div>
```

**Step 5: Build check**

Run: `cd helix/dashboard && npm run build`
Expected: Build succeeds

**Step 6: Commit**

```bash
git add helix/dashboard/src/app/dashboard/page.tsx
git commit -m "feat(dashboard): add worker panel with reputation scores to dashboard page"
```

---

### Task 5: Rewrite Activity Page Network Graph with Clique Subgraphs

**Files:**
- Modify: `helix/dashboard/src/app/activity/page.tsx`

**Step 1: Replace data source and clustering logic**

Replace the random `i % clusterCount` clustering in `NetworkGraph` (lines 112-125) with inference-request-based clustering.

The `NetworkGraph` component should accept a new prop `clusters`:
```typescript
interface InferenceCluster {
  requestId: string;
  modelName: string;
  modelTokenId: number;
  workers: string[];  // worker IDs in this cluster
  status: 'active' | 'completed';
}

function NetworkGraph({
  nodes,
  connections,
  clusters,
}: {
  nodes: WorkerNode[];
  connections: NetworkConnection[];
  clusters: InferenceCluster[];
}) {
```

In the simulation setup, assign cluster IDs from the `clusters` prop:
```typescript
// Build cluster map: workerId -> cluster index
const workerClusterMap = new Map<string, number>();
clusters.forEach((c, idx) => {
  c.workers.forEach((wid) => workerClusterMap.set(wid, idx));
});

// Add model center nodes (one per cluster)
const modelNodes: GNode[] = clusters.map((c, idx) => ({
  id: `model-${c.modelTokenId}`,
  type: 'aggregator' as const,  // larger node
  radius: 14,
  cluster: idx,
  isModelNode: true,
  label: c.modelName,
  x: cx + (Math.random() - 0.5) * w * 0.2,
  y: cy + (Math.random() - 0.5) * h * 0.2,
}));

const workerGNodes: GNode[] = nodes.map((n) => {
  const cluster = workerClusterMap.get(n.id) ?? 0;
  return {
    id: n.id,
    type: n.type,
    radius: 7,
    cluster,
    isModelNode: false,
    x: cx + (Math.random() - 0.5) * w * 0.3,
    y: cy + (Math.random() - 0.5) * h * 0.3,
  };
});

const gNodes = [...modelNodes, ...workerGNodes];

// Create intra-cluster links (each worker connects to its model center)
const clusterLinks: GLink[] = clusters.flatMap((c, idx) =>
  c.workers
    .filter((wid) => nodes.some((n) => n.id === wid))
    .map((wid) => ({
      source: `model-${c.modelTokenId}`,
      target: wid,
      latency: 50,
      connectionStatus: 'active' as const,
    })),
);
```

Update render loop to draw model center nodes differently (larger, colored per cluster):
```typescript
// Cluster colors (soft pastels)
const clusterColors = ['rgba(96,165,250,', 'rgba(167,139,250,', 'rgba(251,146,60,', 'rgba(52,211,153,', 'rgba(251,113,133,'];

// For model center nodes: draw larger with cluster color glow
if (node.isModelNode) {
  const clr = clusterColors[node.cluster % clusterColors.length];
  // Outer glow
  const glow = ctx.createRadialGradient(node.x, node.y, node.radius * 0.5, node.x, node.y, node.radius * 4);
  glow.addColorStop(0, `${clr}0.15)`);
  glow.addColorStop(1, `${clr}0)`);
  ctx.beginPath();
  ctx.arc(node.x, node.y, node.radius * 4, 0, Math.PI * 2);
  ctx.fillStyle = glow;
  ctx.fill();
  // Core
  ctx.beginPath();
  ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
  ctx.fillStyle = `${clr}0.9)`;
  ctx.fill();
  // Label
  ctx.fillStyle = 'rgba(255,255,255,0.7)';
  ctx.font = '9px var(--font-geist-mono)';
  ctx.textAlign = 'center';
  ctx.fillText(node.label || '', node.x, node.y + node.radius + 12);
}
```

**Step 2: Wire data source to backend events filtered by owner**

In `ActivityPage`, replace the backend events fetching (lines 354-369) with the `useBackendApi` hook:

```typescript
import { useBackendApi, type BackendEvent } from '@/hooks/useBackendApi';

// Inside ActivityPage:
const { fetchInferenceEvents } = useBackendApi({ enabled: false }); // don't auto-refresh, we do manual

const [inferenceEvents, setInferenceEvents] = useState<BackendEvent[]>([]);

useEffect(() => {
  if (!address) return;
  const load = async () => {
    const events = await fetchInferenceEvents(address);
    setInferenceEvents(events);
  };
  load();
  const interval = setInterval(load, 10000);
  return () => clearInterval(interval);
}, [address, fetchInferenceEvents]);
```

Build clusters from inference events:
```typescript
const clusters = useMemo(() => {
  const clusterMap = new Map<number, InferenceCluster>();
  inferenceEvents.forEach((evt) => {
    const tokenId = evt.model_token_id ?? 0;
    if (!clusterMap.has(tokenId)) {
      clusterMap.set(tokenId, {
        requestId: evt.session_id ?? `req-${tokenId}`,
        modelName: modelNames.get(tokenId) ?? `Model #${tokenId}`,
        modelTokenId: tokenId,
        workers: [],
        status: 'active',
      });
    }
    // Add worker IDs from the event
    if (evt.worker_ids && Array.isArray(evt.worker_ids)) {
      const cluster = clusterMap.get(tokenId)!;
      (evt.worker_ids as string[]).forEach((wid) => {
        if (!cluster.workers.includes(wid)) cluster.workers.push(wid);
      });
    }
  });
  return Array.from(clusterMap.values());
}, [inferenceEvents, modelNames]);
```

**Step 3: Wire stats to real computed values**

Replace the hardcoded stats computation (lines 470-472):
```typescript
const totalRevenue = useMemo(
  () => inferenceEvents.reduce((s, r) => s + (r.fee ?? 0), 0),
  [inferenceEvents],
);
const totalRequests = inferenceEvents.length;
const avgLatency = useMemo(
  () => {
    const latencies = inferenceEvents.filter((r) => r.latency).map((r) => r.latency!);
    return latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : 0;
  },
  [inferenceEvents],
);
```

**Step 4: Wire request table to real backend events**

Replace the `requests` useMemo (lines 372-431) to build from `inferenceEvents` directly:
```typescript
const requests = useMemo(() => {
  return inferenceEvents
    .map((evt, i) => ({
      id: evt.session_id ?? `evt-${i}`,
      timestamp: evt.timestamp ?? Date.now() - i * 1000,
      requester: (evt.requester as string) ?? '0x000000',
      modelTokenId: evt.model_token_id ?? 0,
      modelName: modelNames.get(evt.model_token_id ?? 0) ?? 'Unknown',
      prediction: evt.prediction ?? 0,
      confidence: evt.confidence ?? 0,
      fee: evt.fee ?? 0,
      ownerRevenue: evt.fee ?? 0,
      latency: evt.latency ?? 0,
      workers: evt.workers ?? 3,
      status: 'completed' as const,
    }))
    .sort((a, b) => b.timestamp - a.timestamp);
}, [inferenceEvents, modelNames]);
```

**Step 5: Add empty state for no-models user**

After the header section:
```tsx
{myModels.length === 0 && (
  <div className="bg-helix-surface border border-helix-border rounded-2xl p-16 text-center">
    <div className="text-sm text-helix-muted">Register a model to see activity</div>
    <div className="text-xs text-helix-dim mt-2">
      Inference requests on your models will appear here
    </div>
  </div>
)}
```

**Step 6: Build and verify**

Run: `cd helix/dashboard && npm run build`
Expected: Build succeeds

**Step 7: Commit**

```bash
git add helix/dashboard/src/app/activity/page.tsx
git commit -m "feat(activity): rewrite network graph with clique subgraphs grouped by inference request"
```

---

### Task 6: Wire Dashboard Graphs to Real Backend Data

**Files:**
- Modify: `helix/dashboard/src/app/dashboard/page.tsx`

**Step 1: Wire loss curve to real backend losses endpoint**

The dashboard page already uses `useDashboardSessions()` which fetches from `/api/training/sessions/:id/losses`. Verify this is real:
- The `losses` array comes from `useDashboardSessions` which polls `/api/training/sessions/{sessionId}/losses`
- The backend route in `dashboard.rs` reads from `sessions.read().await` which contains live `TrainingSessionState.losses`
- This is REAL data — no changes needed for the loss curve.

**Step 2: Wire training metrics to real round status**

Add real-time round status display. After the progress bar section (~line 743), add backend round info:

```tsx
{roundStatus?.current_round && (
  <div className="grid grid-cols-4 gap-3">
    <div className="bg-helix-bg rounded-xl px-4 py-3">
      <div className="text-xs text-helix-muted mb-1">Phase</div>
      <div className="text-sm font-mono text-white">{roundStatus.current_round.phase}</div>
    </div>
    <div className="bg-helix-bg rounded-xl px-4 py-3">
      <div className="text-xs text-helix-muted mb-1">Gradients</div>
      <div className="text-sm font-mono text-white tabular-nums">
        {roundStatus.current_round.gradients_received}/{roundStatus.current_round.workers_assigned}
      </div>
    </div>
    <div className="bg-helix-bg rounded-xl px-4 py-3">
      <div className="text-xs text-helix-muted mb-1">Completed Rounds</div>
      <div className="text-sm font-mono text-white tabular-nums">{roundStatus.completed_rounds}</div>
    </div>
    <div className="bg-helix-bg rounded-xl px-4 py-3">
      <div className="text-xs text-helix-muted mb-1">MPC Status</div>
      <div className="text-sm font-mono text-white">
        {backendHealth?.mpc.active_session ? 'Active' : 'Idle'}
      </div>
    </div>
  </div>
)}
```

**Step 3: Build and verify**

Run: `cd helix/dashboard && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add helix/dashboard/src/app/dashboard/page.tsx
git commit -m "feat(dashboard): wire training metrics to real backend round status"
```

---

### Task 7: Verify Network, Train, and Inference Pages

**Files:**
- Verify: `helix/dashboard/src/app/network/page.tsx`
- Verify: `helix/dashboard/src/app/train/page.tsx`
- Verify: `helix/dashboard/src/app/inference/page.tsx`
- Verify: `helix/dashboard/src/hooks/useNodes.ts`
- Verify: `helix/dashboard/src/hooks/useWorkerHealth.ts`

**Step 1: Verify Network page uses real data**

Read `network/page.tsx`, `useNodes.ts`, and verify:
- `useNodes()` already has backend fallback to `GET /api/workers` (confirmed in exploration)
- `NodeGrid`, `NetworkHealthBar`, `NodeDetailSheet` all consume `useNodes()` return values
- If any mock data generation exists, replace with backend calls

**Step 2: Verify Train page uses real data**

Read `train/page.tsx` and verify:
- `useMpcTraining()` already calls `POST /api/training/start` and polls session state
- All 13 phases are reported from backend `ProgressEvent` enums
- Loss charts during training come from session polling

**Step 3: Verify Inference page uses real data**

Read `inference/page.tsx` and verify:
- Already proxies to `POST /api/inference` on the backend
- Result (prediction, confidence) comes from real backend response
- Inference history is written to localStorage — this is fine, it's client-side caching

**Step 4: Fix any issues found**

If any page has mock/generated data that should be real, replace with `useBackendApi` calls.

**Step 5: Build full project**

Run: `cd helix/dashboard && npm run build`
Expected: Build succeeds with no errors

**Step 6: Commit if changes were made**

```bash
git add -A helix/dashboard/src/
git commit -m "fix(dashboard): verify and wire remaining pages to real backend data"
```

---

### Task 8: Full Integration Build & Lint Check

**Files:**
- All modified dashboard files

**Step 1: Run lint**

Run: `cd helix/dashboard && npm run lint`
Expected: No errors (warnings OK)

**Step 2: Run full build**

Run: `cd helix/dashboard && npm run build`
Expected: Build succeeds

**Step 3: Run Rust workspace check**

Run: `cd helix && cargo check -p helix-node -p helix-client`
Expected: Compiles successfully

**Step 4: Fix any issues and commit**

If lint/build errors, fix them and commit:
```bash
git add -A
git commit -m "fix: resolve lint and build issues in dashboard wiring"
```
