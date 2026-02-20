# Live Dashboard Updates Design

**Date:** 2026-02-20
**Status:** Approved

## Problem

The dashboard's loss/accuracy curves, step counter, and status indicator only update when training completes. They should update in real-time as each MPC training step executes.

The wiring exists end-to-end (backend fires `TrainingStep` events, broadcasts over WebSocket, frontend handles them), but several implementation bugs prevent it from working:

1. **`try_write()` contention** — the progress callback runs synchronously inside `spawn_blocking` and uses `tokio::sync::RwLock::try_write()` with 3 retries at 1ms. When polling handlers or the WebSocket snapshot handler hold a read lock, writes silently fail. The in-memory session state falls behind, making polling and reconnection snapshots stale.

2. **Read lock held during async WebSocket send** — at subscription time, `sessions.read().await` is held across `sender.send(Message::Text(...)).await`, blocking all writers during TCP writes.

3. **Status descriptions are fake** — `SubStatusTicker` cycles through hardcoded `MPC_STEP_OPERATIONS` strings keyed to `step % 6`. No real sub-step events exist from the backend.

4. **No sub-step granularity** — the backend fires one event per completed training step. Individual operations within each step (forward pass, ReLU, backward pass, MAC check) are not reported.

## Approach

Fix the existing architecture (it's correctly designed) and add real sub-step events.

## Part 1: Fix State Update Contention

Replace `try_write()` with an mpsc channel pattern in `dashboard.rs`:

- Add `state_update_tx: tokio::sync::mpsc::UnboundedSender<(String, ProgressEvent)>` to `DashboardState`
- Spawn a dedicated `state_updater` task at startup that drains the channel and applies updates using `.write().await`
- The progress callback sends to the channel (sync, never fails) instead of calling `try_write()`
- The callback still does `ws_broadcast.send()` for WebSocket delivery

**Files:** `helix/crates/helix-client/src/dashboard.rs`

## Part 2: Add SubStep Events from MPC Trainer

- Add `SubStep { step: usize, total: usize, operation: String }` variant to `ProgressEvent`
- Add `on_sub_step` callback field to `MPCTrainer`
- Fire at natural communication boundaries in `training_step_unproved` and `training_step_with_mac`:
  1. "Forward pass — layer 1 matmul"
  2. "Garbled-circuit ReLU activation"
  3. "Forward pass — layer 2 matmul"
  4. "Computing loss and gradients"
  5. "Backward pass — gradient computation"
  6. "Applying weight update"
  7. "SPDZ MAC verification" (MAC path only)
- Wire through `MPCIntegrationConfig` -> `run_party_training` -> `MPCTrainer`
- Wire through `FullOrchestrator` adapter (same pattern as `on_step`)

**Files:** `helix/crates/helix-mpc/src/mpc_trainer.rs`, `helix/crates/helix-mpc/src/e2e_integration.rs`, `helix/crates/helix-client/src/full_orchestration.rs`, `helix/crates/helix-client/src/dashboard.rs`

## Part 3: Wire Sub-Step Events to Frontend

- Add `sub_step` field to `TrainingSessionState` (both Rust and TS types)
- Handle `sub_step` WebSocket events in `useMpcTraining.ts`
- Update `SubStatusTicker` to display `session.sub_step` when available
- Remove hardcoded `MPC_STEP_OPERATIONS` cycling

**Files:** `helix/dashboard/src/hooks/useMpcTraining.ts`, `helix/dashboard/src/app/train/page.tsx`, `helix/crates/helix-client/src/dashboard.rs`

## Part 4: Fix WebSocket Read Lock During Send

Clone session data before sending instead of holding the read lock across the async send:

```rust
let snapshot = {
    let sessions = state.sessions.read().await;
    sessions.get(session_id).cloned()
};
if let Some(session) = snapshot {
    // send snapshot over WebSocket (lock already released)
}
```

**Files:** `helix/crates/helix-client/src/dashboard.rs`
