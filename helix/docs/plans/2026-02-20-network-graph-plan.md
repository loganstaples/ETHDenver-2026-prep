# Network Graph Redesign Implementation Plan

**Goal:** Replace the Activity page with a real-time ring-topology network graph on the Network page, wired into backend WebSocket events for live worker-model assignments.

**Architecture:** Pure-geometry canvas renderer (no D3 physics). Backend emits `worker_assigned`, `worker_removed`, `model_activity_changed` events via WebSocket. Frontend hook (`useActiveTopology`) manages live state. Network page = graph hero + existing worker table.

**Tech Stack:** Next.js 16 App Router, Canvas 2D API, requestAnimationFrame, existing HelixWebSocketClient, Rust axum backend with broadcast channels.

---

### Task 1: Add active-tasks endpoint and WebSocket events to Rust backend

**Files:**
- Modify: `crates/helix-node/src/api/http.rs`

**Step 1: Add ActiveTask types and the /api/active-tasks endpoint**

Add these types after the `FaultToleranceStatus` struct (around line 288):

```rust
/// An active model-worker assignment (for the network graph).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTask {
    pub model_id: String,
    pub model_name: String,
    pub task_type: String, // "training" or "inference"
    pub worker_ids: Vec<String>,
    pub started_at: u64,
}

/// Response for GET /api/active-tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTasksResponse {
    pub active_models: Vec<ActiveTask>,
}
```

Add a new field to `ApiState` (line ~108):

```rust
    /// Active model-worker task assignments (for network graph).
    pub active_tasks: Arc<RwLock<Vec<ActiveTask>>>,
```

Add the handler function:

```rust
async fn active_tasks_handler(State(state): State<Arc<ApiState>>) -> Json<ActiveTasksResponse> {
    let tasks = state.active_tasks.read().clone();
    Json(ActiveTasksResponse {
        active_models: tasks,
    })
}
```

Register the route in `start_api_server` (add to `public_routes` at line ~377):

```rust
    let public_routes = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/api/active-tasks", get(active_tasks_handler));
```

**Step 2: Verify it compiles**

Run: `cargo check -p helix-node`
Expected: Compiles (may have warnings about unused field until wired)

**Step 3: Commit**

```bash
git add crates/helix-node/src/api/http.rs
git commit -m "feat(backend): add /api/active-tasks endpoint and ActiveTask types"
```

---

### Task 2: Wire active_tasks into backend runtime

**Files:**
- Modify: `crates/helix-node/src/runtime.rs` (or wherever `ApiState` is constructed)

**Step 1: Find where ApiState is constructed and add the active_tasks field**

Search for `ApiState {` in the codebase to find the construction site. Add:

```rust
active_tasks: Arc::new(RwLock::new(Vec::new())),
```

**Step 2: Find training round start/stop points and update active_tasks**

In the training orchestrator or distributed coordinator, when a round starts with workers:
- Push an `ActiveTask` to the `active_tasks` vec
- When round ends or worker is removed, update the vec

This requires passing `active_tasks: Arc<RwLock<Vec<ActiveTask>>>` to the training coordinator.

**Step 3: Verify it compiles**

Run: `cargo check -p helix-node`
Expected: Compiles without errors

**Step 4: Commit**

```bash
git add -A crates/helix-node/
git commit -m "feat(backend): wire active_tasks into runtime and training coordinator"
```

---

### Task 3: Add WebSocket event broadcasting for worker assignments

**Files:**
- Modify: `crates/helix-node/src/api/http.rs` (add broadcast channel to ApiState)
- Modify: Training coordinator files that assign/remove workers

**Step 1: Add a broadcast channel for dashboard events to ApiState**

```rust
use serde_json::Value as JsonValue;

// In ApiState:
    /// Broadcast channel for real-time dashboard events (worker_assigned, etc.)
    pub dashboard_event_tx: broadcast::Sender<String>,
```

Initialize in construction: `let (dashboard_event_tx, _) = broadcast::channel(256);`

**Step 2: Create helper function to broadcast events**

```rust
pub fn broadcast_dashboard_event(tx: &broadcast::Sender<String>, event_type: &str, data: serde_json::Value) {
    let msg = serde_json::json!({
        "type": event_type,
        "channel": "nodes",
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        "data": data,
    });
    let _ = tx.send(msg.to_string());
}
```

**Step 3: At training round assignment, broadcast worker_assigned events**

When workers are assigned to a training round, call:

```rust
for worker_id in &assigned_workers {
    broadcast_dashboard_event(&state.dashboard_event_tx, "worker_assigned", serde_json::json!({
        "worker_id": worker_id,
        "model_id": model_id,
        "task_type": "training",
    }));
}
broadcast_dashboard_event(&state.dashboard_event_tx, "model_activity_changed", serde_json::json!({
    "model_id": model_id,
    "model_name": model_name,
    "active": true,
    "task_type": "training",
    "worker_ids": assigned_workers,
    "started_at": now_epoch_secs,
}));
```

**Step 4: At worker removal/round completion, broadcast worker_removed events**

```rust
broadcast_dashboard_event(&state.dashboard_event_tx, "worker_removed", serde_json::json!({
    "worker_id": worker_id,
    "model_id": model_id,
    "reason": reason, // "completed", "kicked", "quit"
}));
```

**Step 5: Verify compilation**

Run: `cargo check -p helix-node`

**Step 6: Commit**

```bash
git add -A crates/helix-node/
git commit -m "feat(backend): broadcast worker_assigned/removed events via dashboard channel"
```

---

### Task 4: Add WebSocket endpoint to Rust backend

**Files:**
- Modify: `crates/helix-node/src/api/http.rs`

The current backend has HTTP and JSON-RPC but no WebSocket endpoint. Add one using axum's WebSocket support.

**Step 1: Add axum WebSocket dependency and handler**

In `crates/helix-node/Cargo.toml`, ensure `axum` has the `ws` feature enabled:
```toml
axum = { version = "...", features = ["ws"] }
```

Add WebSocket handler in http.rs:

```rust
use axum::extract::ws::{WebSocket, WebSocketUpgrade, Message};
use futures_util::{SinkExt, StreamExt};

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ApiState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_connection(socket, state))
}

async fn handle_ws_connection(socket: WebSocket, state: Arc<ApiState>) {
    let (mut sender, mut receiver) = socket.split();
    let mut event_rx = state.dashboard_event_tx.subscribe();

    // Forward broadcast events to WebSocket client
    let send_task = tokio::spawn(async move {
        while let Ok(msg) = event_rx.recv().await {
            if sender.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages (ping/pong, subscribe requests)
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    // Handle subscribe/unsubscribe/ping
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                        if parsed.get("type").and_then(|t| t.as_str()) == Some("ping") {
                            // Pong handled by send_task via broadcast
                        }
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    });

    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }
}
```

**Step 2: Register the /ws route**

In `start_api_server`, add to public routes:

```rust
    let public_routes = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/api/active-tasks", get(active_tasks_handler))
        .route("/ws", get(ws_handler));
```

**Step 3: Verify it compiles**

Run: `cargo check -p helix-node`

**Step 4: Commit**

```bash
git add crates/helix-node/
git commit -m "feat(backend): add WebSocket endpoint for real-time dashboard events"
```

---

### Task 5: Create useActiveTopology hook

**Files:**
- Create: `dashboard/src/hooks/useActiveTopology.ts`

**Step 1: Write the hook**

```typescript
'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { getApiClient } from '@/lib/api';

export interface ActiveModel {
  modelId: string;
  modelName: string;
  taskType: 'training' | 'inference';
  workerIds: string[];
  startedAt: number;
}

export interface UseActiveTopologyReturn {
  activeModels: ActiveModel[];
  isConnected: boolean;
}

const INFERENCE_MIN_DURATION_MS = 5000;
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

export function useActiveTopology(): UseActiveTopologyReturn {
  const [activeModels, setActiveModels] = useState<ActiveModel[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const apiClient = useRef(getApiClient());

  // Fetch initial snapshot
  const fetchSnapshot = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/active-tasks`);
      if (res.ok) {
        const data = await res.json();
        const models: ActiveModel[] = (data.active_models || []).map(
          (m: { model_id: string; model_name: string; task_type: string; worker_ids: string[]; started_at: number }) => ({
            modelId: m.model_id,
            modelName: m.model_name,
            taskType: m.task_type as 'training' | 'inference',
            workerIds: m.worker_ids,
            startedAt: m.started_at,
          }),
        );
        setActiveModels(models);
      }
    } catch {
      // Backend unavailable — empty state
    }
  }, []);

  // Subscribe to WebSocket events
  useEffect(() => {
    fetchSnapshot();

    const client = apiClient.current;
    client.connectWebSocket().then(() => {
      setIsConnected(true);
      client.subscribeToChannel('nodes');
    }).catch(() => setIsConnected(false));

    const unsubAssigned = client.onWebSocketMessage<{
      worker_id: string;
      model_id: string;
      task_type: string;
    }>('worker_assigned', (msg) => {
      const { worker_id, model_id, task_type } = msg.data;
      setActiveModels((prev) => {
        const existing = prev.find((m) => m.modelId === model_id);
        if (existing) {
          if (existing.workerIds.includes(worker_id)) return prev;
          return prev.map((m) =>
            m.modelId === model_id
              ? { ...m, workerIds: [...m.workerIds, worker_id] }
              : m,
          );
        }
        // New model — will be fully populated by model_activity_changed
        return [
          ...prev,
          {
            modelId: model_id,
            modelName: model_id,
            taskType: task_type as 'training' | 'inference',
            workerIds: [worker_id],
            startedAt: Date.now(),
          },
        ];
      });
    });

    const unsubRemoved = client.onWebSocketMessage<{
      worker_id: string;
      model_id: string;
    }>('worker_removed', (msg) => {
      const { worker_id, model_id } = msg.data;
      setActiveModels((prev) => {
        const updated = prev.map((m) =>
          m.modelId === model_id
            ? { ...m, workerIds: m.workerIds.filter((id) => id !== worker_id) }
            : m,
        );
        // Remove model if no workers left
        return updated.filter((m) => m.workerIds.length > 0);
      });
    });

    const unsubModelChanged = client.onWebSocketMessage<{
      model_id: string;
      model_name: string;
      active: boolean;
      task_type: string;
      worker_ids: string[];
      started_at: number;
    }>('model_activity_changed', (msg) => {
      const d = msg.data;
      if (!d.active) {
        setActiveModels((prev) => prev.filter((m) => m.modelId !== d.model_id));
        return;
      }
      setActiveModels((prev) => {
        const existing = prev.find((m) => m.modelId === d.model_id);
        if (existing) {
          return prev.map((m) =>
            m.modelId === d.model_id
              ? { ...m, modelName: d.model_name, workerIds: d.worker_ids, taskType: d.task_type as 'training' | 'inference' }
              : m,
          );
        }
        return [
          ...prev,
          {
            modelId: d.model_id,
            modelName: d.model_name,
            taskType: d.task_type as 'training' | 'inference',
            workerIds: d.worker_ids,
            startedAt: d.started_at * 1000,
          },
        ];
      });
    });

    return () => {
      unsubAssigned();
      unsubRemoved();
      unsubModelChanged();
    };
  }, [fetchSnapshot]);

  // Filter: only show inference tasks active > 5s
  const now = Date.now();
  const visibleModels = activeModels.filter((m) => {
    if (m.taskType === 'training') return true;
    return now - m.startedAt >= INFERENCE_MIN_DURATION_MS;
  });

  return { activeModels: visibleModels, isConnected };
}
```

**Step 2: Verify it compiles**

Run: `cd dashboard && npx tsc --noEmit`

**Step 3: Commit**

```bash
git add dashboard/src/hooks/useActiveTopology.ts
git commit -m "feat(dashboard): add useActiveTopology hook for live model-worker tracking"
```

---

### Task 6: Create NetworkGraph canvas component

**Files:**
- Create: `dashboard/src/components/network/NetworkGraph.tsx`

**Step 1: Write the ring-topology graph renderer**

```typescript
'use client';

import { useRef, useEffect, useState } from 'react';
import type { ActiveModel } from '@/hooks/useActiveTopology';

interface Props {
  activeModels: ActiveModel[];
}

interface NodePosition {
  id: string;
  x: number;
  y: number;
  targetX: number;
  targetY: number;
  radius: number;
  type: 'model' | 'worker';
  label: string;
  modelId?: string;
  opacity: number;
  targetOpacity: number;
}

export default function NetworkGraph({ activeModels }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const nodesRef = useRef<Map<string, NodePosition>>(new Map());
  const animRef = useRef<number>(0);
  const pulseRef = useRef(0);
  const [size, setSize] = useState({ width: 800, height: 400 });

  // Resize observer
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setSize({ width: Math.floor(width), height: Math.floor(height) });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Compute layout and render
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Compute target positions for all active models + workers
    const computeLayout = () => {
      const { width: W, height: H } = size;
      const cx = W / 2;
      const cy = H / 2;
      const modelCount = activeModels.length;
      const newIds = new Set<string>();

      // Position model centers
      const modelPositions: { id: string; x: number; y: number }[] = [];
      if (modelCount === 1) {
        modelPositions.push({ id: activeModels[0].modelId, x: cx, y: cy });
      } else {
        // Spread models in a circle
        const spreadR = Math.min(W, H) * 0.25;
        activeModels.forEach((m, i) => {
          const angle = (i / modelCount) * Math.PI * 2 - Math.PI / 2;
          modelPositions.push({
            id: m.modelId,
            x: cx + Math.cos(angle) * spreadR,
            y: cy + Math.sin(angle) * spreadR,
          });
        });
      }

      activeModels.forEach((model, mi) => {
        const mp = modelPositions[mi];
        const modelNodeId = `model-${model.modelId}`;
        newIds.add(modelNodeId);

        // Update or create model node
        const existing = nodesRef.current.get(modelNodeId);
        if (existing) {
          existing.targetX = mp.x;
          existing.targetY = mp.y;
          existing.targetOpacity = 1;
          existing.label = model.modelName;
        } else {
          nodesRef.current.set(modelNodeId, {
            id: modelNodeId,
            x: mp.x,
            y: mp.y,
            targetX: mp.x,
            targetY: mp.y,
            radius: 20,
            type: 'model',
            label: model.modelName,
            modelId: model.modelId,
            opacity: 0,
            targetOpacity: 1,
          });
        }

        // Position workers in ring around model
        const workerCount = model.workerIds.length;
        const ringR = Math.max(60, Math.min(120, workerCount * 25));
        model.workerIds.forEach((wid, wi) => {
          const angle = (wi / workerCount) * Math.PI * 2 - Math.PI / 2;
          const wx = mp.x + Math.cos(angle) * ringR;
          const wy = mp.y + Math.sin(angle) * ringR;
          newIds.add(wid);

          const existingW = nodesRef.current.get(wid);
          if (existingW) {
            existingW.targetX = wx;
            existingW.targetY = wy;
            existingW.targetOpacity = 1;
            existingW.modelId = model.modelId;
          } else {
            nodesRef.current.set(wid, {
              id: wid,
              x: wx,
              y: wy,
              targetX: wx,
              targetY: wy,
              radius: 10,
              type: 'worker',
              label: wid.length > 12 ? `${wid.slice(0, 6)}...${wid.slice(-4)}` : wid,
              modelId: model.modelId,
              opacity: 0,
              targetOpacity: 1,
            });
          }
        });
      });

      // Fade out removed nodes
      nodesRef.current.forEach((node, id) => {
        if (!newIds.has(id)) {
          node.targetOpacity = 0;
        }
      });
    };

    computeLayout();

    const render = () => {
      pulseRef.current += 0.015;
      const pulse = pulseRef.current;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = size.width * dpr;
      canvas.height = size.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Background
      ctx.fillStyle = '#09090b';
      ctx.fillRect(0, 0, size.width, size.height);

      // Interpolate positions and opacity
      const lerp = 0.08;
      const toRemove: string[] = [];
      nodesRef.current.forEach((node) => {
        node.x += (node.targetX - node.x) * lerp;
        node.y += (node.targetY - node.y) * lerp;
        node.opacity += (node.targetOpacity - node.opacity) * lerp;
        if (node.targetOpacity === 0 && node.opacity < 0.01) {
          toRemove.push(node.id);
        }
      });
      toRemove.forEach((id) => nodesRef.current.delete(id));

      // Build lookup for edges
      const nodeMap = nodesRef.current;

      // Draw edges per model
      activeModels.forEach((model) => {
        const modelNode = nodeMap.get(`model-${model.modelId}`);
        if (!modelNode || modelNode.opacity < 0.01) return;

        const workerNodes = model.workerIds
          .map((wid) => nodeMap.get(wid))
          .filter((n): n is NodePosition => !!n && n.opacity > 0.01);

        // Worker-worker ring edges (dotted, static)
        for (let i = 0; i < workerNodes.length; i++) {
          const a = workerNodes[i];
          const b = workerNodes[(i + 1) % workerNodes.length];
          if (workerNodes.length < 2) break;

          const edgeOpacity = Math.min(a.opacity, b.opacity) * 0.25;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(255, 255, 255, ${edgeOpacity})`;
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 5]);
          ctx.lineDashOffset = 0; // static
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Model-worker edges (solid, animated dashes flowing worker→model)
        workerNodes.forEach((wn) => {
          const edgeOpacity = Math.min(modelNode.opacity, wn.opacity) * 0.6;
          ctx.beginPath();
          ctx.moveTo(wn.x, wn.y);
          ctx.lineTo(modelNode.x, modelNode.y);
          ctx.strokeStyle = `rgba(255, 255, 255, ${edgeOpacity})`;
          ctx.lineWidth = 2;
          ctx.setLineDash([8, 6]);
          // Dash offset: positive = dashes move from worker to model
          // We compute direction and use pulse to animate
          ctx.lineDashOffset = pulse * 30;
          ctx.stroke();
          ctx.setLineDash([]);
        });
      });

      // Draw nodes
      nodeMap.forEach((node) => {
        if (node.opacity < 0.01) return;

        if (node.type === 'model') {
          // Outer glow
          const glowR = node.radius * 4;
          const glow = ctx.createRadialGradient(node.x, node.y, node.radius * 0.5, node.x, node.y, glowR);
          const glowAlpha = (0.12 + Math.sin(pulse * 1.5) * 0.04) * node.opacity;
          glow.addColorStop(0, `rgba(255,255,255,${glowAlpha})`);
          glow.addColorStop(1, 'rgba(255,255,255,0)');
          ctx.beginPath();
          ctx.arc(node.x, node.y, glowR, 0, Math.PI * 2);
          ctx.fillStyle = glow;
          ctx.fill();

          // Core
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255,255,255,${0.95 * node.opacity})`;
          ctx.fill();
          ctx.strokeStyle = `rgba(255,255,255,${0.3 * node.opacity})`;
          ctx.lineWidth = 1;
          ctx.stroke();

          // Label below
          ctx.fillStyle = `rgba(255,255,255,${0.6 * node.opacity})`;
          ctx.font = '10px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(node.label.slice(0, 20), node.x, node.y + node.radius + 14);
        } else {
          // Worker node — green-ish
          // Glow
          const glowR = node.radius * 3;
          const glow = ctx.createRadialGradient(node.x, node.y, node.radius * 0.5, node.x, node.y, glowR);
          const pa = (0.1 + Math.sin(pulse * 1.5 + node.x * 0.01) * 0.04) * node.opacity;
          glow.addColorStop(0, `rgba(16,185,129,${pa})`);
          glow.addColorStop(1, 'rgba(16,185,129,0)');
          ctx.beginPath();
          ctx.arc(node.x, node.y, glowR, 0, Math.PI * 2);
          ctx.fillStyle = glow;
          ctx.fill();

          // Core
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(16,185,129,${0.85 * node.opacity})`;
          ctx.fill();
          ctx.strokeStyle = `rgba(16,185,129,${0.2 * node.opacity})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      });

      animRef.current = requestAnimationFrame(render);
    };

    animRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animRef.current);
  }, [size, activeModels]);

  // Empty state
  if (activeModels.length === 0) {
    return (
      <div ref={containerRef} className="relative w-full h-full min-h-[340px] rounded-2xl bg-helix-bg overflow-hidden flex items-center justify-center">
        <div className="text-center">
          <div className="text-sm text-helix-muted">No active tasks</div>
          <div className="text-xs text-helix-dim mt-1">
            Start training or inference to see the network topology
          </div>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[340px] rounded-2xl bg-helix-bg overflow-hidden">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ width: size.width, height: size.height }}
      />
      <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-helix-surface/80 backdrop-blur-sm border border-helix-border/50">
        <div className="w-1.5 h-1.5 rounded-full bg-white/60 animate-pulse" />
        <span className="text-xs text-helix-muted">Live</span>
      </div>
    </div>
  );
}
```

**Step 2: Verify it compiles**

Run: `cd dashboard && npx tsc --noEmit`

**Step 3: Commit**

```bash
git add dashboard/src/components/network/NetworkGraph.tsx
git commit -m "feat(dashboard): add ring-topology NetworkGraph canvas component"
```

---

### Task 7: Rewrite Network page with graph + table

**Files:**
- Modify: `dashboard/src/app/network/page.tsx`

**Step 1: Add graph import and integration at the top of the page**

Add imports:

```typescript
import NetworkGraph from '@/components/network/NetworkGraph';
import { useActiveTopology } from '@/hooks/useActiveTopology';
```

Add the hook call inside `NetworkPage()`:

```typescript
const { activeModels } = useActiveTopology();
```

Add the graph section between the stats row and the filter tabs (after the `</div>` closing the stats grid, before the filter grid):

```tsx
{/* ── Network Graph ─────────────────────────────────── */}
<div className="bg-helix-surface border border-helix-border rounded-2xl overflow-hidden" style={{ height: 400 }}>
  <NetworkGraph activeModels={activeModels} />
</div>
```

**Step 2: Verify it renders**

Run: `cd dashboard && npm run build`

**Step 3: Commit**

```bash
git add dashboard/src/app/network/page.tsx
git commit -m "feat(dashboard): add NetworkGraph to Network page"
```

---

### Task 8: Delete Activity page and remove from sidebar

**Files:**
- Delete: `dashboard/src/app/activity/page.tsx`
- Modify: `dashboard/src/components/layout/Sidebar.tsx`

**Step 1: Delete the Activity page file**

```bash
rm dashboard/src/app/activity/page.tsx
```

If the directory is now empty, remove it too:
```bash
rmdir dashboard/src/app/activity/
```

**Step 2: Remove Activity from sidebar navigation**

In `Sidebar.tsx`, remove the Activity nav item (line 16):

Change the navItems array from:
```typescript
const navItems = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Models', href: '/', icon: Boxes },
  { name: 'My Models', href: '/my-models', icon: Layers },
  { name: 'Train', href: '/train', icon: Cpu },
  { name: 'Inference', href: '/inference', icon: Sparkles },
  { name: 'Activity', href: '/activity', icon: Receipt },
  { name: 'Network', href: '/network', icon: Globe },
  { name: 'Settings', href: '/settings', icon: Settings },
] as const;
```

To:
```typescript
const navItems = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Models', href: '/', icon: Boxes },
  { name: 'My Models', href: '/my-models', icon: Layers },
  { name: 'Train', href: '/train', icon: Cpu },
  { name: 'Inference', href: '/inference', icon: Sparkles },
  { name: 'Network', href: '/network', icon: Globe },
  { name: 'Settings', href: '/settings', icon: Settings },
] as const;
```

Also remove the `Receipt` import from lucide-react (line 7).

**Step 3: Remove any other references to /activity**

Search for `/activity` or `activity` in dashboard/src and clean up any leftover imports or links.

**Step 4: Verify the build**

Run: `cd dashboard && npm run build`

**Step 5: Commit**

```bash
git add -A dashboard/
git commit -m "feat(dashboard): delete Activity page and remove from sidebar navigation"
```

---

### Task 9: End-to-end verification

**Step 1: Build the Rust backend**

Run: `cargo build -p helix-node`

**Step 2: Build the dashboard**

Run: `cd dashboard && npm run build`

**Step 3: Verify no lint errors**

Run: `cd dashboard && npm run lint`

**Step 4: Run Rust tests**

Run: `cargo test -p helix-node -- --test-threads=4` (expect existing tests to pass)

**Step 5: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address build issues from network graph integration"
```
