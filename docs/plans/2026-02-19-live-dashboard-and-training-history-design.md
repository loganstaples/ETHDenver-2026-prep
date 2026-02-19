# Live Dashboard Updates & Training History

**Date:** 2026-02-19
**Status:** Approved

## Problem

During MPC training sessions, the dashboard stats/graphs/data don't update in real-time (except the phase tracker). Most metrics only appear in a burst at the end of training. Additionally, there is no persistent training history — past sessions are stored only in browser localStorage and server memory, both lost on restart.

## Root Cause

The MPC training function (`run_party_training` in `e2e_integration.rs`) runs all steps to completion and returns losses as a `Vec<f64>` in `PartyResult`. The orchestrator (`full_orchestration.rs:664-672`) then loops through results and emits all `TrainingStep` events in a burst post-training. The WebSocket infrastructure works correctly — the problem is upstream: no progress reporting during training execution.

## Design

### 1. Live Progress Fix — Progress Callback in MPC Config

**Approach:** Add an optional callback to `MPCIntegrationConfig` that gets called after each training step.

**Changes:**
- `MPCIntegrationConfig`: Add `on_step: Option<Arc<dyn Fn(usize, usize, f64, bool) + Send + Sync>>` (step, total, loss, mac_ok)
- `run_party_training`: Party 0 calls `on_step` after each training step (only party 0 to avoid duplicate events)
- `run_with_local_transport` / `run_with_tcp_transport` / `run_with_node_transport`: Thread `on_step` through to the party 0 spawn
- `full_orchestration.rs`: Pass `emit(TrainingStep{...})` as the `on_step` callback. Remove the post-hoc loop at lines 664-672.
- Also emit `checkpoint_submitted` events from within the training loop when checkpoints are created
- Client-side: add `setInterval` ticking every second during training to update elapsed time display

### 2. SQLite Persistence Layer

**Dependency:** `rusqlite` with `bundled` feature (statically links SQLite, zero external deps).

**Schema:**
```sql
CREATE TABLE training_sessions (
  session_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  model_name TEXT,
  model_slug TEXT,
  current_step INTEGER DEFAULT 0,
  total_steps INTEGER DEFAULT 0,
  current_loss REAL DEFAULT 0.0,
  losses TEXT DEFAULT '[]',
  accuracy REAL,
  checkpoints_submitted INTEGER DEFAULT 0,
  mac_checks_passed INTEGER DEFAULT 0,
  zk_proofs_generated INTEGER DEFAULT 0,
  cheater_detected TEXT,
  phase INTEGER DEFAULT 0,
  phase_description TEXT DEFAULT '',
  coordinator_address TEXT DEFAULT '',
  job_id INTEGER DEFAULT 0,
  elapsed_secs REAL DEFAULT 0.0,
  started_at INTEGER DEFAULT 0,
  completed_at INTEGER,
  error_message TEXT
);
```

**Behavior:**
- `TrainingDb` struct wraps `rusqlite::Connection` behind `Mutex` for thread safety
- Auto-save: upsert session row on every status change (starting → running → complete/failed)
- On server startup: load all sessions from DB into in-memory `HashMap<session_id, TrainingSessionState>`
- DB file at `./helix-data/training.db` (created automatically)

**Endpoint changes:**
- `GET /api/training/sessions` — returns both in-memory and DB sessions (deduplicated)
- `GET /api/training/sessions/:id` — checks in-memory first, falls back to DB

### 3. Training History UI

**Toggle (top of train page):**
- Two-state toggle: "Live" | "History"
- Positioned above the config form / live progress area
- "Live" = current behavior (config form or active training session)
- "History" = session card grid

**Card grid (history view):**
- Fetched from `GET /api/training/sessions`
- Reverse chronological order (newest first)
- Each card shows: model name + version, status pill, final accuracy, steps completed/total, duration, date, mini loss sparkline

**Detail modal (click card):**
- Full-screen overlay (~90% viewport, dark backdrop)
- Reuses `LiveProgress` layout with historical data:
  - Loss chart (full resolution)
  - Stats grid (MAC checks, checkpoints, ZK proofs, elapsed time)
  - Phase breakdown (all phases shown as completed)
  - Cheater alert (if applicable)
  - Session metadata (ID, coordinator address, job ID)
- Close: X button, backdrop click, or Escape key
- No download/0G buttons (historical view only)

### 4. Client-side Live Timer

- `useEffect` with `setInterval(1000)` during training
- Computes elapsed as `Date.now()/1000 - session.started_at`
- Updates the "Elapsed Time" stat card every second
- Cleared when training reaches terminal state (complete/failed)

## Files Modified

**Rust (helix-mpc):**
- `helix/crates/helix-mpc/src/e2e_integration.rs` — add `on_step` callback to config, call from party 0 training loop

**Rust (helix-client):**
- `helix/crates/helix-client/Cargo.toml` — add `rusqlite` dependency
- `helix/crates/helix-client/src/training_db.rs` — new file: `TrainingDb` struct with SQLite operations
- `helix/crates/helix-client/src/dashboard.rs` — integrate `TrainingDb`, modify session endpoints, remove post-hoc emit loop
- `helix/crates/helix-client/src/full_orchestration.rs` — pass progress callback to MPC config, remove post-hoc TrainingStep emit loop
- `helix/crates/helix-client/src/lib.rs` — add `mod training_db`

**Frontend (dashboard):**
- `helix/dashboard/src/hooks/useMpcTraining.ts` — add live timer, remove localStorage history
- `helix/dashboard/src/app/train/page.tsx` — add Live/History toggle, history card grid, detail modal
