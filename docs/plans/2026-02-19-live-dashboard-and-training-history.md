# Live Dashboard Updates & Training History — Implementation Plan

**Goal:** Make all dashboard stats, graphs, and metrics update in real-time during MPC training, persist training sessions to SQLite, and add a training history UI with detail modals.

**Architecture:** Thread a progress callback from the orchestrator into the MPC training loop so per-step events emit live (not batched at end). Add a SQLite persistence layer for sessions. Replace localStorage-based history with server-fetched history and add a full-screen detail modal.

**Tech Stack:** Rust (helix-mpc, helix-client), rusqlite, TypeScript/React (Next.js dashboard), Recharts, Framer Motion, Lucide icons.

---

## Task 1: Add `on_step` progress callback to `MPCIntegrationConfig`

**Files:**
- Modify: `helix/crates/helix-mpc/src/e2e_integration.rs`

**Step 1: Add `on_step` field to `MPCIntegrationConfig`**

At `e2e_integration.rs:85` (after the `batch_size` field), add:

```rust
    /// Optional callback invoked after each training step on party 0.
    /// Arguments: (step_1indexed, total_steps, loss, mac_ok).
    #[serde(skip)]
    pub on_step: Option<std::sync::Arc<dyn Fn(usize, usize, f64, bool) + Send + Sync>>,
```

**Step 2: Add `on_step: None` to the Default impl**

At `e2e_integration.rs:119` (after `batch_size: 1,`), add:

```rust
            on_step: None,
```

**Step 3: Add `on_step: None` to all test configs**

Every test that constructs `MPCIntegrationConfig { ... }` explicitly (around lines 1488, 1525, 1566, 1598, 1659) needs `on_step: None` appended.

**Step 4: Thread `on_step` through `run_with_local_transport`**

At `e2e_integration.rs:439`, add `on_step` parameter to the function signature:

```rust
async fn run_with_local_transport(
    parties: Vec<PartyId>,
    trainer_config: MPCTrainerConfig,
    initial_weights: Option<ModelWeights>,
    training_data: Vec<(Vec<f64>, Vec<f64>)>,
    num_steps: usize,
    checkpoint_interval: usize,
    seed: u64,
    start: Instant,
    on_step: Option<std::sync::Arc<dyn Fn(usize, usize, f64, bool) + Send + Sync>>,
) -> Result<MPCIntegrationResult, anyhow::Error> {
```

In the `for` loop (line 457), clone `on_step` for party 0 only and pass to `run_party_training`:

```rust
    for (i, (transport, bundle)) in transports.into_iter().zip(bundles.into_iter()).enumerate() {
        let cfg = trainer_config.clone();
        let data = training_data.clone();
        let step_cb = if i == 0 { on_step.clone() } else { None };

        let handle = tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build per-worker tokio runtime");
            rt.block_on(run_party_training(
                cfg, transport, i, None, data,
                num_steps, checkpoint_interval, seed,
                None, 0,
                Some(bundle),
                step_cb,
            ))
        });
        handles.push(handle);
    }
```

**Step 5: Thread `on_step` through `run_with_node_transport`**

Same pattern as step 4. Add `on_step` parameter, clone for party 0, pass to `run_party_training`.

**Step 6: Thread `on_step` through `run_with_tcp_transport`**

Same pattern. Add `on_step` parameter, clone for party 0, pass to `run_party_training`.

**Step 7: Thread `on_step` through `run_with_cheater`**

Same pattern. Add `on_step` parameter, clone for party 0, pass to `run_party_training`.

**Step 8: Thread `on_step` through `run_with_tcp_cheater`**

Same pattern. Add `on_step` parameter, clone for party 0, pass to `run_party_training`.

**Step 9: Add `on_step` parameter to `run_party_training`**

At `e2e_integration.rs:1086`, add parameter:

```rust
pub async fn run_party_training<T: crate::session::transport::MPCTransport + 'static>(
    config: MPCTrainerConfig,
    transport: T,
    party_index: usize,
    initial_weights: Option<ModelWeights>,
    training_data: Vec<(Vec<f64>, Vec<f64>)>,
    num_steps: usize,
    checkpoint_interval: usize,
    seed: u64,
    cheater_info: Option<usize>,
    corrupt_at_step: u64,
    encrypted_bundle: Option<EncryptedShareBundle>,
    on_step: Option<std::sync::Arc<dyn Fn(usize, usize, f64, bool) + Send + Sync>>,
) -> Result<PartyResult, anyhow::Error> {
```

**Step 10: Call `on_step` in the training loop**

At `e2e_integration.rs:1223` (after `losses.push(step_result.loss);`), add:

```rust
        // Emit real-time progress for party 0.
        if let Some(ref cb) = on_step {
            cb(step + 1, num_steps, step_result.loss, true);
        }
```

For the MAC failure path (after `cheater_detected = Some(...)` around line 1203, before `break;`), add:

```rust
                    if let Some(ref cb) = on_step {
                        cb(step + 1, num_steps, 0.0, false);
                    }
```

**Step 11: Update callers of `run_mpc_training` and `run_mpc_training_with_cheater`**

In `run_mpc_training` (line 200): extract `on_step` from config, pass through:

```rust
pub async fn run_mpc_training(
    config: MPCIntegrationConfig,
) -> Result<MPCIntegrationResult, anyhow::Error> {
    let start = Instant::now();
    let num_workers = config.num_workers;
    let num_steps = config.num_steps;
    let checkpoint_interval = config.checkpoint_interval;
    let on_step = config.on_step.clone();
    // ... existing code ...
```

Then at each dispatch site (lines 255, 268, 273), pass `on_step.clone()` as the last arg:

- `run_with_tcp_transport(..., on_step.clone()).await`
- `run_with_node_transport(..., on_step.clone()).await`
- `run_with_local_transport(..., on_step.clone()).await`

Same for `run_mpc_training_with_cheater` (line 284): extract `on_step`, pass to `run_with_tcp_cheater` and `run_with_cheater`.

**Step 12: Build and run tests**

Run: `cargo test -p helix-mpc --lib e2e_integration -- --test-threads=1`
Expected: All existing tests pass (they all pass `on_step: None` now).

**Step 13: Commit**

```bash
git add helix/crates/helix-mpc/src/e2e_integration.rs
git commit -m "feat: add on_step progress callback to MPCIntegrationConfig for live updates"
```

---

## Task 2: Wire orchestrator to emit live per-step events

**Files:**
- Modify: `helix/crates/helix-client/src/full_orchestration.rs`

**Step 1: Pass `on_step` callback in the local-mode MPC config**

At `full_orchestration.rs:620-637` (where `MPCIntegrationConfig` is constructed for local mode), add the `on_step` field. The callback should call `self.emit(ProgressEvent::TrainingStep{...})`. Since we're inside `run()` and the emit closure needs `self.progress`, we clone the progress callback:

```rust
    let progress_for_step = self.progress.clone();
    let total_steps = self.config.num_steps;
```

Then in the `MPCIntegrationConfig` construction:

```rust
        on_step: progress_for_step.map(|cb| -> std::sync::Arc<dyn Fn(usize, usize, f64, bool) + Send + Sync> {
            std::sync::Arc::new(move |step, total, loss, mac_ok| {
                cb(ProgressEvent::TrainingStep { step, total, loss, mac_ok });
            })
        }),
```

**Step 2: Remove post-hoc TrainingStep emit loop for local mode**

At `full_orchestration.rs:661-673`, remove or comment out:

```rust
        // REMOVE THIS BLOCK — progress events are now emitted live via on_step callback
        // if !self.config.distributed {
        //     for (i, loss) in mpc_result.losses.iter().enumerate() {
        //         self.emit(ProgressEvent::TrainingStep {
        //             step: i + 1,
        //             total: self.config.num_steps,
        //             loss: *loss,
        //             mac_ok: true,
        //         });
        //     }
        // }
```

**Step 3: For distributed mode, keep the existing behavior**

The distributed mode (lines 1075-1085) still receives all results at the end from worker 0. This is inherent to the distributed architecture — the owner can't see intermediate steps. Leave this loop as-is.

**Step 4: Build check**

Run: `cargo check -p helix-client`
Expected: Compiles successfully.

**Step 5: Commit**

```bash
git add helix/crates/helix-client/src/full_orchestration.rs
git commit -m "feat: wire live on_step callback into orchestrator MPC config"
```

---

## Task 3: Add SQLite persistence layer

**Files:**
- Modify: `helix/crates/helix-client/Cargo.toml`
- Create: `helix/crates/helix-client/src/training_db.rs`
- Modify: `helix/crates/helix-client/src/lib.rs`

**Step 1: Add rusqlite dependency**

In `helix/crates/helix-client/Cargo.toml`, add after `base64 = "0.21"` (line 94):

```toml
# SQLite for training session persistence
rusqlite = { version = "0.31", features = ["bundled"] }
```

**Step 2: Create `training_db.rs`**

Create `helix/crates/helix-client/src/training_db.rs`:

```rust
//! SQLite persistence for training session history.

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection};
use tracing::{info, warn};

use crate::dashboard::TrainingSessionState;

/// SQLite-backed training session store.
pub struct TrainingDb {
    conn: Mutex<Connection>,
}

impl TrainingDb {
    /// Open (or create) the database at the given path.
    pub fn open(path: &Path) -> Result<Self, rusqlite::Error> {
        // Ensure parent directory exists.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        let db = Self { conn: Mutex::new(conn) };
        db.create_tables()?;
        info!(path = %path.display(), "Training database opened");
        Ok(db)
    }

    fn create_tables(&self) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS training_sessions (
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
                started_at REAL DEFAULT 0.0,
                completed_at REAL,
                error_message TEXT
            );"
        )?;
        Ok(())
    }

    /// Upsert a training session into the database.
    pub fn save_session(&self, session: &TrainingSessionState) {
        let conn = self.conn.lock().unwrap();
        let losses_json = serde_json::to_string(&session.losses).unwrap_or_else(|_| "[]".to_string());
        let cheater_json = session.cheater_detected.as_ref().map(|v| v.to_string());
        let completed_at: Option<f64> = if session.status == "complete" || session.status == "failed" {
            Some(session.started_at + session.elapsed_secs)
        } else {
            None
        };
        let error_message: Option<String> = if session.status == "failed" {
            Some(session.phase_description.clone())
        } else {
            None
        };

        let result = conn.execute(
            "INSERT INTO training_sessions (
                session_id, status, model_name, model_slug,
                current_step, total_steps, current_loss, losses,
                accuracy, checkpoints_submitted, mac_checks_passed,
                zk_proofs_generated, cheater_detected, phase,
                phase_description, coordinator_address, job_id,
                elapsed_secs, started_at, completed_at, error_message
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21)
            ON CONFLICT(session_id) DO UPDATE SET
                status=excluded.status,
                current_step=excluded.current_step,
                total_steps=excluded.total_steps,
                current_loss=excluded.current_loss,
                losses=excluded.losses,
                accuracy=excluded.accuracy,
                checkpoints_submitted=excluded.checkpoints_submitted,
                mac_checks_passed=excluded.mac_checks_passed,
                zk_proofs_generated=excluded.zk_proofs_generated,
                cheater_detected=excluded.cheater_detected,
                phase=excluded.phase,
                phase_description=excluded.phase_description,
                coordinator_address=excluded.coordinator_address,
                job_id=excluded.job_id,
                elapsed_secs=excluded.elapsed_secs,
                completed_at=excluded.completed_at,
                error_message=excluded.error_message",
            params![
                session.session_id,
                session.status,
                session.model_name,
                session.model_slug,
                session.current_step as i64,
                session.total_steps as i64,
                session.current_loss,
                losses_json,
                session.accuracy,
                session.checkpoints_submitted as i64,
                session.mac_checks_passed as i64,
                session.zk_proofs_generated as i64,
                cheater_json,
                session.phase as i64,
                session.phase_description,
                session.coordinator_address,
                session.job_id as i64,
                session.elapsed_secs,
                session.started_at,
                completed_at,
                error_message,
            ],
        );

        if let Err(e) = result {
            warn!(session_id = %session.session_id, error = %e, "Failed to save session to DB");
        }
    }

    /// Load all sessions from the database.
    pub fn load_all_sessions(&self) -> Vec<TrainingSessionState> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT session_id, status, model_name, model_slug,
                    current_step, total_steps, current_loss, losses,
                    accuracy, checkpoints_submitted, mac_checks_passed,
                    zk_proofs_generated, cheater_detected, phase,
                    phase_description, coordinator_address, job_id,
                    elapsed_secs, started_at
             FROM training_sessions
             ORDER BY started_at DESC"
        ) {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "Failed to prepare load query");
                return Vec::new();
            }
        };

        let rows = stmt.query_map([], |row| {
            let losses_json: String = row.get(7)?;
            let losses: Vec<f64> = serde_json::from_str(&losses_json).unwrap_or_default();
            let cheater_json: Option<String> = row.get(12)?;
            let cheater_detected: Option<serde_json::Value> = cheater_json
                .and_then(|s| serde_json::from_str(&s).ok());

            Ok(TrainingSessionState {
                session_id: row.get(0)?,
                status: row.get(1)?,
                model_name: row.get(2)?,
                model_slug: row.get(3)?,
                current_step: row.get::<_, i64>(4)? as usize,
                total_steps: row.get::<_, i64>(5)? as usize,
                current_loss: row.get(6)?,
                losses,
                accuracy: row.get(8)?,
                checkpoints_submitted: row.get::<_, i64>(9)? as usize,
                mac_checks_passed: row.get::<_, i64>(10)? as usize,
                zk_proofs_generated: row.get::<_, i64>(11)? as usize,
                cheater_detected,
                phase: row.get::<_, i64>(13)? as u32,
                phase_description: row.get(14)?,
                coordinator_address: row.get(15)?,
                job_id: row.get::<_, i64>(16)? as u64,
                elapsed_secs: row.get(17)?,
                started_at: row.get(18)?,
                zk_activated_by_risk: false,
                final_weights: None,
                model_token_id: None,
                model_version_index: None,
            })
        });

        match rows {
            Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
            Err(e) => {
                warn!(error = %e, "Failed to load sessions from DB");
                Vec::new()
            }
        }
    }
}
```

**Step 3: Add `mod training_db` to `lib.rs`**

At `helix/crates/helix-client/src/lib.rs`, add after the existing mod declarations (around line 26):

```rust
pub mod training_db;
```

**Step 4: Build check**

Run: `cargo check -p helix-client`
Expected: Compiles successfully.

**Step 5: Commit**

```bash
git add helix/crates/helix-client/Cargo.toml helix/crates/helix-client/src/training_db.rs helix/crates/helix-client/src/lib.rs
git commit -m "feat: add SQLite training session persistence layer"
```

---

## Task 4: Integrate `TrainingDb` into `DashboardState`

**Files:**
- Modify: `helix/crates/helix-client/src/dashboard.rs`

**Step 1: Add `training_db` field to `DashboardState`**

At `dashboard.rs:454` (after `model_weights_cache`), add:

```rust
    /// SQLite database for persisting training sessions across restarts.
    pub training_db: Option<std::sync::Arc<crate::training_db::TrainingDb>>,
```

**Step 2: Initialize `training_db` in `DashboardState::new`**

At `dashboard.rs:477` (after `model_weights_cache` init in `new()`), add:

```rust
            training_db: {
                let db_path = std::path::PathBuf::from("./helix-data/training.db");
                match crate::training_db::TrainingDb::open(&db_path) {
                    Ok(db) => Some(std::sync::Arc::new(db)),
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to open training DB, history will not persist");
                        None
                    }
                }
            },
```

**Step 3: Load persisted sessions on startup**

In `DashboardState::new()`, after creating the state, load sessions from DB. Since `new` returns `Self` (not `Arc<Self>`), we load after construction. Add after the training_db field init:

Actually, better approach: do the loading in `with_defaults()` (line 482) which returns `Arc<Self>`. But `new()` is also called from `create_dashboard_router`. The simplest approach: load sessions right after DB init.

Add a method to DashboardState:

```rust
    /// Load persisted sessions from the SQLite database into memory.
    pub async fn load_persisted_sessions(&self) {
        if let Some(ref db) = self.training_db {
            let sessions = db.load_all_sessions();
            let count = sessions.len();
            if count > 0 {
                let mut store = self.sessions.write().await;
                for s in sessions {
                    // Only load completed/failed sessions (don't restore in-progress ones)
                    if s.status == "complete" || s.status == "failed" {
                        store.entry(s.session_id.clone()).or_insert(s);
                    }
                }
                tracing::info!(count, "Loaded persisted training sessions from database");
            }
        }
    }
```

Then in `create_dashboard_router_with_state()` or wherever the server starts, call `state.load_persisted_sessions().await`. Find the right callsite (likely in the binary that starts the dashboard, or in `create_dashboard_router`). Since `create_dashboard_router` is sync, we need to call it from the async context. Add it to the start of `start_dashboard` or similar entry point.

Alternative: load synchronously in `new()` using the `sessions` `RwLock::new(...)` constructor:

```rust
            sessions: {
                let mut map = HashMap::new();
                if let Some(ref db_arc) = training_db_instance {
                    for s in db_arc.load_all_sessions() {
                        if s.status == "complete" || s.status == "failed" {
                            map.insert(s.session_id.clone(), s);
                        }
                    }
                    if !map.is_empty() {
                        tracing::info!(count = map.len(), "Loaded persisted sessions from DB");
                    }
                }
                RwLock::new(map)
            },
```

This requires restructuring `new()` slightly so `training_db` is computed first, then `sessions` uses it. This is the cleanest approach.

**Step 4: Save session to DB on status changes**

In `run_training_session()` (dashboard.rs), save to DB at two points:

a) After session creation (line 1208):
```rust
    state.sessions.write().await.insert(session_id.clone(), session.clone());
    // Persist to SQLite
    if let Some(ref db) = state.training_db {
        db.save_session(&session);
    }
```

b) After training completes (line 1443, after final state update):
```rust
            // Persist completed session to SQLite
            if let Some(ref db) = state.training_db {
                if let Some(session) = sessions.get(&session_id) {
                    db.save_session(session);
                }
            }
```

c) After training fails (line 1488, after error state update):
```rust
            // Persist failed session to SQLite
            if let Some(ref db) = state.training_db {
                if let Some(session) = sessions.get(&session_id) {
                    db.save_session(session);
                }
            }
```

**Step 5: Build check**

Run: `cargo check -p helix-client`
Expected: Compiles successfully.

**Step 6: Commit**

```bash
git add helix/crates/helix-client/src/dashboard.rs
git commit -m "feat: integrate SQLite persistence into dashboard state"
```

---

## Task 5: Add live elapsed time timer on the frontend

**Files:**
- Modify: `helix/dashboard/src/hooks/useMpcTraining.ts`
- Modify: `helix/dashboard/src/app/train/page.tsx`

**Step 1: Add `elapsedTime` state and timer to useMpcTraining**

In `useMpcTraining.ts`, add a new state variable after the existing state declarations (around line 128):

```typescript
const [elapsedTime, setElapsedTime] = useState(0);
```

Add a `useEffect` that runs a 1-second interval when session is active:

```typescript
// Live elapsed time counter
useEffect(() => {
  if (!session || session.status === 'complete' || session.status === 'failed') {
    return;
  }
  const interval = setInterval(() => {
    if (session.started_at > 0) {
      setElapsedTime(Math.floor(Date.now() / 1000 - session.started_at));
    }
  }, 1000);
  return () => clearInterval(interval);
}, [session?.status, session?.started_at]);
```

Export `elapsedTime` in the returned object.

**Step 2: Use `elapsedTime` in the Elapsed Time stat card**

In `page.tsx`, the LiveProgress component currently shows elapsed time from `session.elapsed_secs`. Update the StatCard for elapsed time (around line 1073) to use the new `elapsedTime` value:

```typescript
const displayElapsed = isTerminal ? session.elapsed_secs : elapsedTime;
```

Format it nicely:
```typescript
function formatDuration(secs: number): string {
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}
```

**Step 3: Build check**

Run: `cd helix/dashboard && npm run build`
Expected: Builds successfully.

**Step 4: Commit**

```bash
git add helix/dashboard/src/hooks/useMpcTraining.ts helix/dashboard/src/app/train/page.tsx
git commit -m "feat: add live elapsed time counter during training"
```

---

## Task 6: Fetch training history from server instead of localStorage

**Files:**
- Modify: `helix/dashboard/src/hooks/useMpcTraining.ts`
- Modify: `helix/dashboard/src/app/train/page.tsx`

**Step 1: Add `fetchHistory` function to useMpcTraining hook**

Add a function that fetches from `GET /api/training/sessions`:

```typescript
const [history, setHistory] = useState<TrainingSessionState[]>([]);
const [isLoadingHistory, setIsLoadingHistory] = useState(false);

const fetchHistory = useCallback(async () => {
  setIsLoadingHistory(true);
  try {
    const res = await fetch(`${API_BASE}/api/training/sessions`);
    if (res.ok) {
      const sessions: TrainingSessionState[] = await res.json();
      // Only include completed/failed sessions, sorted newest first
      setHistory(sessions.filter(s => s.status === 'complete' || s.status === 'failed'));
    }
  } catch {
    // Silently fail — history is non-critical
  } finally {
    setIsLoadingHistory(false);
  }
}, []);
```

Export `history`, `isLoadingHistory`, `fetchHistory` from the hook.

**Step 2: Fetch history on mount and after training completes**

In the hook, add:

```typescript
// Fetch history on mount
useEffect(() => {
  fetchHistory();
}, [fetchHistory]);
```

Also call `fetchHistory()` after a session reaches terminal state (inside the `session_complete` or `session_failed` handler).

**Step 3: Remove localStorage-based history from page.tsx**

In `page.tsx`:
- Remove the `TrainingHistoryEntry` interface (lines 100-110)
- Remove `HISTORY_KEY`, `getTrainingHistory()`, `saveTrainingHistory()` functions (lines 112-123)
- Remove `getNextVersion()` function (lines 125-133)
- Remove the localStorage-based `history` state, `historyRecordedRef`, and the useEffects that save to localStorage (lines 1479-1611)
- Update `handleClearHistory` to just clear the state (no more localStorage)
- Use the `history` from the hook instead

**Step 4: Update TrainPageInner to use hook-provided history**

Destructure `history` and `fetchHistory` from `useMpcTraining()`. The `getNextVersion` function should compute from the server-returned history (use `model_name` or just count sessions).

**Step 5: Build check**

Run: `cd helix/dashboard && npm run build`
Expected: Builds successfully.

**Step 6: Commit**

```bash
git add helix/dashboard/src/hooks/useMpcTraining.ts helix/dashboard/src/app/train/page.tsx
git commit -m "feat: replace localStorage history with server-fetched training sessions"
```

---

## Task 7: Build the Training History UI with detail modal

**Files:**
- Modify: `helix/dashboard/src/app/train/page.tsx`

**Step 1: Add `viewMode` toggle state**

In `TrainPageInner`, add:

```typescript
const [viewMode, setViewMode] = useState<'live' | 'history'>('live');
const [selectedHistorySession, setSelectedHistorySession] = useState<TrainingSessionState | null>(null);
```

**Step 2: Add the toggle UI**

At the top of the page (before the conditional `!hasSession` block, around line 1750), add a toggle bar:

```tsx
{/* View mode toggle */}
<div className="flex items-center gap-1 p-1 rounded-xl bg-helix-surface border border-helix-border w-fit">
  <button
    type="button"
    onClick={() => setViewMode('live')}
    className={cn(
      'px-4 py-1.5 rounded-lg text-sm font-medium transition-all',
      viewMode === 'live'
        ? 'bg-white text-black'
        : 'text-helix-dim hover:text-helix-text2'
    )}
  >
    Live
  </button>
  <button
    type="button"
    onClick={() => { setViewMode('history'); fetchHistory(); }}
    className={cn(
      'px-4 py-1.5 rounded-lg text-sm font-medium transition-all',
      viewMode === 'history'
        ? 'bg-white text-black'
        : 'text-helix-dim hover:text-helix-text2'
    )}
  >
    History
  </button>
</div>
```

**Step 3: Build the history card grid**

When `viewMode === 'history'`, render a grid of session cards:

```tsx
function HistoryCardGrid({ sessions, onSelect }: {
  sessions: TrainingSessionState[];
  onSelect: (s: TrainingSessionState) => void;
}) {
  if (sessions.length === 0) {
    return (
      <div className="text-center py-16 text-helix-dim">
        <History size={32} className="mx-auto mb-3 opacity-40" />
        <p>No training sessions yet</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {sessions.map((s) => (
        <button
          key={s.session_id}
          type="button"
          onClick={() => onSelect(s)}
          className="text-left p-5 rounded-2xl bg-helix-surface border border-helix-border hover:border-helix-border-hover transition-all group"
        >
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-white truncate max-w-[60%]">
              {s.model_name || 'Unnamed Model'}
            </span>
            <Badge variant={s.status === 'complete' ? 'green' : 'red'}>
              {s.status}
            </Badge>
          </div>

          {/* Mini sparkline */}
          {s.losses.length > 1 && (
            <div className="h-12 mb-3 opacity-60 group-hover:opacity-100 transition-opacity">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={s.losses.slice(-50).map((l, i) => ({ i, l }))}>
                  <Area type="monotone" dataKey="l" stroke="#ffffff40" fill="#ffffff10" strokeWidth={1} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="flex items-center gap-3 text-sm text-helix-dim">
            {s.accuracy !== null && s.accuracy !== undefined && (
              <span className="text-white font-medium">{(s.accuracy * 100).toFixed(1)}%</span>
            )}
            <span>{s.current_step}/{s.total_steps} steps</span>
            <span className="ml-auto">{formatDuration(Math.round(s.elapsed_secs))}</span>
          </div>

          <div className="text-xs text-helix-dim mt-2">
            {new Date(s.started_at * 1000).toLocaleDateString()} {new Date(s.started_at * 1000).toLocaleTimeString()}
          </div>
        </button>
      ))}
    </div>
  );
}
```

**Step 4: Build the detail modal**

Create a `HistoryDetailModal` component:

```tsx
function HistoryDetailModal({ session, onClose }: {
  session: TrainingSessionState;
  onClose: () => void;
}) {
  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const lossData = session.losses.map((l, i) => ({ step: i + 1, loss: l }));

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        {/* Backdrop */}
        <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />

        {/* Modal */}
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="relative w-full max-w-5xl max-h-[90vh] overflow-y-auto rounded-3xl bg-helix-bg border border-helix-border p-8"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Close button */}
          <button
            type="button"
            onClick={onClose}
            className="absolute top-4 right-4 p-2 rounded-xl text-helix-dim hover:text-white hover:bg-helix-surface transition-colors"
          >
            <XCircle size={20} />
          </button>

          {/* Header */}
          <div className="flex items-center gap-3 mb-6">
            <h2 className="text-xl font-semibold text-white">
              {session.model_name || 'Training Session'}
            </h2>
            <Badge variant={session.status === 'complete' ? 'green' : 'red'}>
              {session.status}
            </Badge>
            <span className="text-sm text-helix-dim ml-auto font-mono">
              {session.session_id.slice(0, 8)}
            </span>
          </div>

          {/* Two-column layout mirroring LiveProgress */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* LEFT: Hero metric + loss curve */}
            <div className="space-y-4">
              <div className="p-6 rounded-2xl bg-helix-surface border border-helix-border text-center">
                {session.status === 'complete' && session.accuracy !== null ? (
                  <>
                    <p className="text-5xl font-bold text-white tabular-nums">
                      {(session.accuracy! * 100).toFixed(1)}
                      <span className="text-2xl text-helix-dim">%</span>
                    </p>
                    <p className="text-sm text-helix-dim mt-1">Final Accuracy</p>
                  </>
                ) : (
                  <>
                    <p className="text-5xl font-bold text-white tabular-nums">
                      {session.current_loss.toFixed(4)}
                    </p>
                    <p className="text-sm text-helix-dim mt-1">Final Loss</p>
                  </>
                )}
              </div>

              {lossData.length > 1 && (
                <div className="h-64">
                  <LossCurve data={lossData} />
                </div>
              )}
            </div>

            {/* RIGHT: Stats + metadata */}
            <div className="space-y-4">
              {/* Progress bar (static, showing final) */}
              <div>
                <div className="h-1.5 rounded-full bg-helix-border overflow-hidden">
                  <div
                    className="h-full rounded-full bg-white"
                    style={{ width: `${(session.current_step / Math.max(session.total_steps, 1)) * 100}%` }}
                  />
                </div>
                <p className="text-sm text-helix-dim mt-1 tabular-nums">
                  Step {session.current_step} / {session.total_steps}
                </p>
              </div>

              {/* Stats grid */}
              <div className="grid grid-cols-2 gap-3">
                <StatCard label="MAC Checks" value={session.mac_checks_passed} icon={<Shield size={14} />} />
                <StatCard label="Checkpoints" value={session.checkpoints_submitted} icon={<Zap size={14} />} />
                <StatCard label="ZK Proofs" value={session.zk_proofs_generated} icon={<Activity size={14} />} />
                <StatCard label="Elapsed" value={formatDuration(Math.round(session.elapsed_secs))} icon={null} />
              </div>

              {/* Cheater alert */}
              {session.cheater_detected && (
                <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
                  <strong>Cheater Detected</strong>
                  <span className="ml-2">Worker {(session.cheater_detected as any).party_index}</span>
                </div>
              )}

              {/* Session metadata */}
              <div className="p-4 rounded-xl bg-helix-surface border border-helix-border space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-helix-dim">Session ID</span>
                  <span className="text-helix-text2 font-mono">{session.session_id.slice(0, 16)}...</span>
                </div>
                {session.coordinator_address && (
                  <div className="flex justify-between">
                    <span className="text-helix-dim">Coordinator</span>
                    <span className="text-helix-text2 font-mono">{session.coordinator_address.slice(0, 10)}...</span>
                  </div>
                )}
                {session.job_id > 0 && (
                  <div className="flex justify-between">
                    <span className="text-helix-dim">Job ID</span>
                    <span className="text-helix-text2 font-mono">{session.job_id}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-helix-dim">Started</span>
                  <span className="text-helix-text2">
                    {new Date(session.started_at * 1000).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
```

**Step 5: Wire the toggle and modal into the page layout**

In `TrainPageInner`, update the main render:

```tsx
{/* View mode toggle */}
{/* ... toggle code from Step 2 ... */}

{viewMode === 'live' ? (
  <>
    {!hasSession ? (
      <ConfigForm ... />
    ) : (
      <LiveProgress ... />
    )}
  </>
) : (
  <HistoryCardGrid
    sessions={history}
    onSelect={setSelectedHistorySession}
  />
)}

{/* Detail modal (always rendered when selected, regardless of viewMode) */}
{selectedHistorySession && (
  <HistoryDetailModal
    session={selectedHistorySession}
    onClose={() => setSelectedHistorySession(null)}
  />
)}
```

**Step 6: Remove the old `TrainingHistoryList` component**

Delete the `TrainingHistoryList` function (lines 1324-1382) and the `<TrainingHistoryList ... />` usage (line 1786). The new `HistoryCardGrid` replaces it.

**Step 7: Build check**

Run: `cd helix/dashboard && npm run build`
Expected: Builds successfully.

**Step 8: Commit**

```bash
git add helix/dashboard/src/app/train/page.tsx helix/dashboard/src/hooks/useMpcTraining.ts
git commit -m "feat: add training history UI with card grid and detail modal"
```

---

## Task 8: End-to-end verification

**Step 1: Start the dashboard backend**

Run: `cd helix && cargo run -- dashboard --port 3001`

**Step 2: Start the dashboard frontend**

Run: `cd helix/dashboard && npm run dev`

**Step 3: Verify live updates**

1. Open http://localhost:3000/train
2. Configure and start a training session (50+ steps recommended)
3. Verify:
   - Loss chart updates in real-time (new points appear during training, not at end)
   - Current loss number updates live
   - Step counter increments live
   - Phase indicator updates when phases change
   - Elapsed time ticks every second
   - MAC checks counter increments
   - Checkpoints counter increments at checkpoint intervals

**Step 4: Verify training history persistence**

1. Wait for training to complete
2. Click "History" toggle
3. Verify the completed session appears as a card
4. Click the card — verify the detail modal shows:
   - Loss chart with full data
   - Final accuracy
   - Stats (MAC checks, checkpoints, ZK proofs, elapsed time)
   - Session metadata
5. Close and restart the backend server
6. Open the dashboard again, click History
7. Verify the session is still there (loaded from SQLite)

**Step 5: Verify the SQLite database file**

Run: `ls -la helix-data/training.db`
Expected: File exists with non-zero size.

**Step 6: Final commit**

If any fixes were needed during verification, commit them:

```bash
git add -A
git commit -m "fix: end-to-end verification fixes for live dashboard and history"
```
