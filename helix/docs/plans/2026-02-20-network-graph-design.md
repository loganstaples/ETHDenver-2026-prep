# Network Graph Redesign

**Date**: 2026-02-20
**Status**: Approved

## Summary

Replace the Activity page's force-directed network graph with a deterministic ring-topology graph on the Network page. Only active models and their assigned workers are visible. Delete the Activity page entirely.

## Requirements

1. **Visibility**: Nodes only appear when relevant. Idle workers and inactive models are hidden.
2. **Ring topology**: Workers form a cycling ring (each connected to at most 2 neighbors) around the model they compute for.
3. **Edge styles**: Model-worker edges are solid white with animated dashes flowing worker→model. Worker-worker ring edges are static dotted lines.
4. **Real-time**: Workers appear/disappear as they join/leave tasks. Backend pushes events via WebSocket.
5. **Multi-model**: Multiple active models share one canvas, each with its own worker ring.
6. **Inference filter**: Inference tasks only appear if active for >5 seconds (skip quick requests).
7. **Page restructure**: Delete Activity page. Network page gets graph on top + existing worker table below.

## Backend Changes

### New WebSocket Events (helix-node)

Emitted on the `nodes` channel:

- **`worker_assigned`**: `{ worker_id, model_id, task_type, timestamp }`
- **`worker_removed`**: `{ worker_id, model_id, reason, timestamp }`
- **`model_activity_changed`**: `{ model_id, model_name, active, task_type, worker_ids, started_at }`

Emission points:
- Training round assigns workers → `worker_assigned` per worker + `model_activity_changed`
- Worker removed (cheater/disconnect) → `worker_removed`
- Training/inference completes → `model_activity_changed(active=false)` + `worker_removed` each
- Inference: only emit if expected duration >5s

### New REST Endpoint

`GET /api/active-tasks` → snapshot of all active model-worker assignments for initial page load.

```json
{
  "active_models": [{
    "model_id": "model-7",
    "model_name": "MNIST-784x32x10",
    "task_type": "training",
    "worker_ids": ["node-abc", "node-def", "node-ghi"],
    "started_at": 1708000000
  }]
}
```

## Frontend Changes

### Rendering: Pure Geometry (no D3 physics)

Canvas-based renderer with `requestAnimationFrame` at 60fps.

**Layout**:
- Single model: center of canvas, workers at `(cx + R*cos(i*2pi/n), cy + R*sin(i*2pi/n))`
- Multiple models: circle-packing to space model clusters across canvas

**Edges**:
| Edge | Style | Animation |
|------|-------|-----------|
| Model-Worker | Solid white, 2px | `setLineDash([8,6])`, `lineDashOffset` decreasing each frame (dashes flow worker→model) |
| Worker-Worker (ring) | Dotted gray, 1px | Static, no animation |

**Nodes**:
| Node | Radius | Style |
|------|--------|-------|
| Model | 20px | Bright white fill, glow, label below |
| Worker | 10px | Reputation-colored (red→green), label on hover |

**Transitions**: 500ms opacity fade on join/leave. Ring positions recalculate smoothly.

**Empty state**: "No active tasks" message when nothing is happening.

### Page Structure

- **Delete**: `src/app/activity/page.tsx` and remove from sidebar
- **Rewrite**: `src/app/network/page.tsx` — graph hero section + existing worker table below
- **New**: `src/components/network/NetworkGraph.tsx` — canvas component
- **New**: `src/hooks/useActiveTopology.ts` — WebSocket + REST state management

### Hook: useActiveTopology

```typescript
interface ActiveModel {
  modelId: string;
  modelName: string;
  taskType: 'training' | 'inference';
  workerIds: string[];
  startedAt: number;
}

// On mount: GET /api/active-tasks
// Live: subscribe to worker_assigned, worker_removed, model_activity_changed
// Filter: inference tasks only included if active >5s
```

## Approach

Pure geometry (trig-based positioning) instead of D3-force physics. Deterministic ring layout with no jitter.
