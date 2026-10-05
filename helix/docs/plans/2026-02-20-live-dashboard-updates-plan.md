# Live Dashboard Updates Implementation Plan

**Goal:** Make the dashboard's loss/accuracy curves, step counter, and status indicator update in real-time during MPC training, with real sub-operation status from the backend.

**Architecture:** Fix the `try_write()` contention bug by replacing it with an mpsc channel + dedicated writer task. Add `SubStep` events from within `MPCTrainer` at natural communication boundaries. Wire them through the existing ProgressEvent → WebSocket → frontend pipeline.

**Tech Stack:** Rust (tokio, axum, serde), TypeScript/React (Next.js, Recharts), WebSocket

---

### Task 1: Add mpsc Channel to DashboardState

Replace the `try_write()` pattern with an unbounded mpsc channel for state updates. This is the primary bug fix — `try_write()` silently drops updates when polling handlers hold read locks.

**Files:**
- Modify: `helix/crates/helix-client/src/dashboard.rs:29` (import)
- Modify: `helix/crates/helix-client/src/dashboard.rs:430-467` (DashboardState struct)
- Modify: `helix/crates/helix-client/src/dashboard.rs:471-525` (DashboardState::new)

**Step 1: Add mpsc import and channel field**

At line 29, change:
```rust
use tokio::sync::{broadcast, RwLock};
```
to:
```rust
use tokio::sync::{broadcast, mpsc, RwLock};
```

Add a new field to `DashboardState` (after `ws_broadcast` at ~line 440):
```rust
pub state_update_tx: mpsc::UnboundedSender<StateUpdate>,
```

Add the `StateUpdate` enum above the struct (before line 430):
```rust
/// A state update message sent from the sync progress callback to the async state updater task.
pub enum StateUpdate {
    Progress { session_id: String, event: crate::full_orchestration::ProgressEvent },
}
```

**Step 2: Create the channel in `DashboardState::new()`**

In `new()` (around line 472, after the broadcast channel creation), add:
```rust
let (state_update_tx, state_update_rx) = mpsc::unbounded_channel::<StateUpdate>();
```

Add `state_update_tx` to the `Self { ... }` block (after `ws_broadcast`). Store `state_update_rx` — it will be consumed by the spawned task in the next step.

Return both from `new()`: change the return type to `(Self, mpsc::UnboundedReceiver<StateUpdate>)` or add a separate `spawn_state_updater` method. The cleanest approach: change `new()` to return `Self` and pass `state_update_rx` out via a separate method, OR have the caller (which creates the Axum router) spawn the task.

Simplest approach — add a public method:
```rust
/// Spawn the background state updater task. Must be called once after creating the state.
/// Consumes the receiver half of the state update channel.
pub fn spawn_state_updater(self: &Arc<Self>, mut rx: mpsc::UnboundedReceiver<StateUpdate>) {
    let state = Arc::clone(self);
    tokio::spawn(async move {
        while let Some(update) = rx.recv().await {
            match update {
                StateUpdate::Progress { session_id, event } => {
                    apply_progress_event(&state, &session_id, &event).await;
                }
            }
        }
    });
}
```

Change `new()` to also return the receiver:
```rust
pub fn new(config: DashboardConfig) -> (Self, mpsc::UnboundedReceiver<StateUpdate>) {
    let (ws_broadcast, _) = broadcast::channel(1024);
    let (state_update_tx, state_update_rx) = mpsc::unbounded_channel::<StateUpdate>();
    // ... rest of init ...
    (Self { ..., state_update_tx, ... }, state_update_rx)
}
```

**Step 3: Extract the state mutation logic into `apply_progress_event`**

Move the match arms from the `try_write` block (lines 1541-1588) into a standalone async function:
```rust
async fn apply_progress_event(
    state: &DashboardState,
    session_id: &str,
    event: &crate::full_orchestration::ProgressEvent,
) {
    let mut sessions = state.sessions.write().await;
    if let Some(session) = sessions.get_mut(session_id) {
        match event {
            ProgressEvent::PhaseStarted { phase, description, .. } => {
                session.phase = *phase;
                session.phase_description = description.clone();
                session.status = "running".to_string();
            }
            ProgressEvent::TrainingStep { step, loss, accuracy, .. } => {
                session.current_step = *step;
                session.current_loss = *loss;
                session.losses.push(*loss);
                session.accuracy = Some(*accuracy);
                session.mac_checks_passed += 1;
                session.phase_description = format!(
                    "Training step {}/{} — loss: {:.4}",
                    step, session.total_steps, loss
                );
            }
            ProgressEvent::CheckpointSubmitted { .. } => {
                session.checkpoints_submitted += 1;
            }
            ProgressEvent::CheaterDetected { party_index, step } => {
                session.cheater_detected = Some(serde_json::json!({
                    "party_index": party_index,
                    "step": step,
                }));
            }
            ProgressEvent::TrainingComplete { accuracy, .. } => {
                session.accuracy = Some(*accuracy);
                session.status = "complete".to_string();
            }
            ProgressEvent::ZkProofGenerated { .. } => {
                session.zk_proofs_generated += 1;
            }
            // SubStep will be handled in Task 8
            _ => {}
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        session.elapsed_secs = now - session.started_at;
    }
}
```

**Step 4: Update callers of `DashboardState::new()`**

In `cmd_dashboard` (around line 2130), update the call:
```rust
let (state, state_update_rx) = DashboardState::new(config);
let state = Arc::new(state);
state.spawn_state_updater(state_update_rx);
```

**Step 5: Compile check**

Run: `cargo check -p helix-client`
Expected: should compile (the progress callback still has the old try_write code — we'll change it in Task 2).

**Step 6: Commit**

```bash
git add helix/crates/helix-client/src/dashboard.rs
git commit -m "feat(dashboard): add mpsc state update channel to DashboardState"
```

---

### Task 2: Replace try_write with Channel Send in Progress Callback

Replace the `try_write()` retry block in the progress callback with a send to the mpsc channel.

**Files:**
- Modify: `helix/crates/helix-client/src/dashboard.rs:1442-1589` (progress callback in run_training_session)

**Step 1: Simplify the progress callback**

Replace the entire block from line 1442 to line 1589. The new callback:
```rust
let progress_cb: ProgressCallback = Arc::new(move |event: ProgressEvent| {
    let event_json = progress_event_to_json(&event);

    // Log key training events (keep existing logging — lines 1446-1521 unchanged)
    match &event {
        ProgressEvent::PhaseStarted { phase, description, total } => {
            tracing::info!(
                session_id = %sid_for_cb,
                phase = phase,
                total_phases = total,
                "[Dashboard] Phase {}/{}: {}",
                phase, total, description
            );
        }
        ProgressEvent::TrainingStep { step, total, loss, accuracy, mac_ok } => {
            tracing::info!(
                session_id = %sid_for_cb,
                step = step,
                total = total,
                loss = loss,
                accuracy = accuracy,
                mac_ok = mac_ok,
                "[Dashboard] Step {}/{} | loss={:.4} | accuracy={:.1}% | MAC: {}",
                step, total, loss, accuracy * 100.0,
                if *mac_ok { "PASS" } else { "FAIL" }
            );
        }
        // ... keep other logging arms as-is ...
        _ => {}
    }

    // Broadcast to WebSocket subscribers (sync-safe, unchanged)
    let _ = state_for_cb.ws_broadcast.send((sid_for_cb.clone(), event_json));

    // Send to state updater task (replaces try_write block)
    let _ = state_for_cb.state_update_tx.send(StateUpdate::Progress {
        session_id: sid_for_cb.clone(),
        event,
    });
});
```

Delete the entire `try_write` retry block (lines 1530-1588). Replace with the single `state_update_tx.send()` call above.

**Step 2: Compile check**

Run: `cargo check -p helix-client`
Expected: compiles. The `ProgressEvent` needs to be `Clone` for the channel send (it takes ownership). Check if it derives Clone — if not, add `#[derive(Clone)]` to the enum in `full_orchestration.rs:107`.

**Step 3: Commit**

```bash
git add helix/crates/helix-client/src/dashboard.rs helix/crates/helix-client/src/full_orchestration.rs
git commit -m "fix(dashboard): replace try_write with mpsc channel for reliable state updates"
```

---

### Task 3: Fix WebSocket Read Lock During Send

Fix the read lock being held across async WebSocket send at subscription time.

**Files:**
- Modify: `helix/crates/helix-client/src/dashboard.rs:1834-1845` (handle_websocket subscription)

**Step 1: Clone session data before sending**

Replace lines 1834-1845:
```rust
if let Some(session_id) = sub.strip_prefix("training:") {
    let snapshot = {
        let sessions = state.sessions.read().await;
        sessions.get(session_id).cloned()
    };
    if let Some(session) = snapshot {
        let state_msg = serde_json::json!({
            "type": "session_state",
            "session_id": session_id,
            "state": session,
        });
        if sender.send(Message::Text(state_msg.to_string().into())).await.is_err() {
            break;
        }
    }
}
```

The key change: the `sessions.read().await` lock is now scoped to the inner block and released before `sender.send()`.

**Step 2: Compile check**

Run: `cargo check -p helix-client`
Expected: compiles. `TrainingSessionState` already derives `Clone` (it's `#[derive(Debug, Clone, Serialize, Deserialize)]`).

**Step 3: Commit**

```bash
git add helix/crates/helix-client/src/dashboard.rs
git commit -m "fix(dashboard): release sessions read lock before async WebSocket send"
```

---

### Task 4: Add SubStep Variant to ProgressEvent

Add the new event type and its JSON conversion.

**Files:**
- Modify: `helix/crates/helix-client/src/full_orchestration.rs:107-136` (ProgressEvent enum)
- Modify: `helix/crates/helix-client/src/dashboard.rs:831-944` (progress_event_to_json)

**Step 1: Add SubStep variant**

In `ProgressEvent` enum (after `TrainingStep` at ~line 109):
```rust
SubStep { step: usize, total: usize, operation: String },
```

**Step 2: Add JSON conversion**

In `progress_event_to_json`, add a new arm (after the `TrainingStep` arm):
```rust
ProgressEvent::SubStep { step, total, operation } => {
    serde_json::json!({
        "type": "sub_step",
        "step": step,
        "total": total,
        "operation": operation,
    })
}
```

**Step 3: Handle SubStep in apply_progress_event**

In the `apply_progress_event` function (created in Task 1), add a match arm:
```rust
ProgressEvent::SubStep { operation, .. } => {
    session.sub_step = Some(operation.clone());
}
```

This requires adding `sub_step` to `TrainingSessionState` — do that now:
```rust
pub sub_step: Option<String>,
```
Add `#[serde(skip_serializing_if = "Option::is_none")]` to keep JSON clean.

Also initialize it as `None` in the default/construction site.

**Step 4: Compile check**

Run: `cargo check -p helix-client`
Expected: compiles.

**Step 5: Commit**

```bash
git add helix/crates/helix-client/src/full_orchestration.rs helix/crates/helix-client/src/dashboard.rs
git commit -m "feat: add SubStep variant to ProgressEvent with JSON conversion"
```

---

### Task 5: Add Sub-Step Callback to MPCTrainer

Add a callback field to `MPCTrainer` and fire it at natural boundaries within `training_step_unproved`.

**Files:**
- Modify: `helix/crates/helix-mpc/src/mpc_trainer.rs:279-321` (struct)
- Modify: `helix/crates/helix-mpc/src/mpc_trainer.rs:1274-1575` (training_step_unproved)
- Modify: `helix/crates/helix-mpc/src/mpc_trainer.rs:2918+` (training_step_with_mac)

**Step 1: Define the callback type and add field**

At the top of `mpc_trainer.rs` (near existing type aliases), add:
```rust
/// Callback for sub-step progress within a single training step.
/// Arguments: (step_1indexed, total_steps, operation_description)
pub type SubStepCallback = std::sync::Arc<dyn Fn(usize, usize, &str) + Send + Sync>;
```

Add to `MPCTrainer` struct (after `last_mac_failure_report` at line ~321):
```rust
sub_step_callback: Option<SubStepCallback>,
```

Add a setter method:
```rust
pub fn set_sub_step_callback(&mut self, cb: SubStepCallback) {
    self.sub_step_callback = Some(cb);
}
```

Initialize as `None` in the constructor (`new()` method).

Add a helper:
```rust
fn emit_sub_step(&self, operation: &str) {
    if let Some(ref cb) = self.sub_step_callback {
        cb(self.current_step as usize + 1, self.config.num_steps, operation);
    }
}
```

Note: `self.config.num_steps` — check that `MPCTrainerConfig` has a `num_steps` or similar field. If not, store `total_steps` on the struct. You may need to pass it through.

**Step 2: Fire sub-step events in `training_step_unproved`**

Insert calls at natural boundaries. The method starts at line 1274:

After layer 1 linear computation (before first `send_all`/`recv_all` at ~line 1325):
```rust
self.emit_sub_step("Forward pass — layer 1 matmul");
```

After first `recv_all` completes (reconstructed h_pre, ~line 1335):
```rust
self.emit_sub_step("Garbled-circuit ReLU activation");
```

After ReLU, before layer 2 linear (before second `send_all` at ~line 1383):
```rust
self.emit_sub_step("Forward pass — layer 2 matmul");
```

After second `recv_all` completes (reconstructed y, ~line 1395):
```rust
self.emit_sub_step("Computing loss and gradients");
```

After loss/dy computation, before backward pass `send_all` (~line 1465):
```rust
self.emit_sub_step("Backward pass — gradient computation");
```

After third `recv_all` completes (reconstructed dh, ~line 1479):
```rust
self.emit_sub_step("Applying weight update");
```

**Step 3: Fire sub-step events in `training_step_with_mac`**

Same pattern as above, plus additional MAC-specific events. After MAC check (around line 3224-3267):
```rust
self.emit_sub_step("SPDZ MAC verification");
```

After checkpoint commit exchange:
```rust
self.emit_sub_step("Checkpoint commitment exchange");
```

**Step 4: Compile check**

Run: `cargo check -p helix-mpc`
Expected: compiles. May need to adjust `num_steps` availability — check if `MPCTrainerConfig` has it or if you need to store the total on the struct.

**Step 5: Commit**

```bash
git add helix/crates/helix-mpc/src/mpc_trainer.rs
git commit -m "feat(mpc): add sub-step callback to MPCTrainer with operation-level granularity"
```

---

### Task 6: Wire Sub-Step Callback Through e2e_integration

Add `on_sub_step` to `MPCIntegrationConfig` and pass it through to `MPCTrainer`.

**Files:**
- Modify: `helix/crates/helix-mpc/src/e2e_integration.rs:46-90` (MPCIntegrationConfig)
- Modify: `helix/crates/helix-mpc/src/e2e_integration.rs:1167+` (run_party_training and variants)

**Step 1: Add on_sub_step to MPCIntegrationConfig**

After `on_step` (line 89):
```rust
/// Optional callback for sub-step progress within each training step (party 0 only).
/// Arguments: (step_1indexed, total_steps, operation_description).
#[serde(skip)]
pub on_sub_step: Option<crate::mpc_trainer::SubStepCallback>,
```

Initialize as `None` in the `Default` impl and in all construction sites within e2e_integration.rs.

**Step 2: Pass on_sub_step to run_party_training**

In `run_party_training` / `run_distributed_party` (the function that creates the MPCTrainer and runs the loop), after creating the trainer, set the callback:

```rust
let sub_step_cb = if party_index == 0 { on_sub_step.clone() } else { None };
// ... after trainer is created:
if let Some(cb) = sub_step_cb {
    trainer.set_sub_step_callback(cb);
}
```

This mirrors the existing `on_step` pattern where only party 0 gets the callback (line 509: `let step_cb = if i == 0 { on_step.clone() } else { None };`).

Do this for all transport variants: `run_with_local_transport` (line ~490), `run_with_node_transport` (line ~532), `run_with_tcp_transport` (line ~580), and any other `run_*` functions that create parties.

**Step 3: Add on_sub_step parameter to the inner training functions**

The `on_sub_step` needs to flow from `MPCIntegrationConfig` → the `spawn_blocking` closure → `run_party_training` → `trainer.set_sub_step_callback()`. Follow the exact same pattern as `on_step`.

In each `run_with_*` function, extract it alongside `on_step`:
```rust
let sub_step_cb = if i == 0 { config.on_sub_step.clone() } else { None };
```

Pass it into the spawned closure and call `trainer.set_sub_step_callback(cb)` inside.

**Step 4: Compile check**

Run: `cargo check -p helix-mpc`
Expected: compiles. May get unused variable warnings for `on_sub_step` in test configs — that's fine.

**Step 5: Commit**

```bash
git add helix/crates/helix-mpc/src/e2e_integration.rs
git commit -m "feat(mpc): wire sub-step callback through MPCIntegrationConfig to trainer"
```

---

### Task 7: Wire Sub-Step Through FullOrchestrator

Add the adapter closure in `full_orchestration.rs` that converts sub-step callbacks into `ProgressEvent::SubStep`.

**Files:**
- Modify: `helix/crates/helix-client/src/full_orchestration.rs:666-691` (MPCIntegrationConfig construction)

**Step 1: Add on_sub_step adapter**

In the local-mode config construction (around line 669-691), after the `on_step` adapter, add:
```rust
on_sub_step: progress_for_sub_step.map(|cb| -> std::sync::Arc<dyn Fn(usize, usize, &str) + Send + Sync> {
    std::sync::Arc::new(move |step, total, operation| {
        cb(ProgressEvent::SubStep {
            step,
            total,
            operation: operation.to_string(),
        });
    })
}),
```

Where `progress_for_sub_step` is another clone of `self.progress`:
```rust
let progress_for_sub_step = self.progress.clone();
```

(Add this next to `let progress_for_step = self.progress.clone();` at line 668.)

**Step 2: Compile check**

Run: `cargo check -p helix-client`
Expected: compiles.

**Step 3: Commit**

```bash
git add helix/crates/helix-client/src/full_orchestration.rs
git commit -m "feat: wire sub-step callback through FullOrchestrator to MPC config"
```

---

### Task 8: Frontend — Handle sub_step WebSocket Events

Add `sub_step` field to the frontend type and handle the new WebSocket event.

**Files:**
- Modify: `helix/dashboard/src/hooks/useMpcTraining.ts:43-65` (TrainingSessionState interface)
- Modify: `helix/dashboard/src/hooks/useMpcTraining.ts:253-272` (ws.onmessage handler)

**Step 1: Add sub_step to TrainingSessionState interface**

After `phase_description` (around line 57):
```typescript
sub_step: string | null;
```

**Step 2: Initialize sub_step in session creation**

Wherever a new session object is created from the POST response (in `startTraining`), ensure `sub_step: null` is included.

**Step 3: Handle sub_step WebSocket event**

In `ws.onmessage`, after the `training_step` handler (after line 272), add:
```typescript
// Handle sub-step events (real-time operation status)
if (evt.type === 'sub_step') {
    const operation = evt.operation as string;
    setSession((prev) => {
        if (!prev) return prev;
        return {
            ...prev,
            sub_step: operation,
        };
    });
}
```

**Step 4: Handle sub_step in polling fallback**

In the polling merge logic (around line 428-440), the session is replaced wholesale when `data.current_step > prev.current_step`. The `sub_step` field will come through automatically since we added it to the Rust `TrainingSessionState` struct (serialized as JSON).

**Step 5: Handle sub_step in session_state snapshot**

In the `session_state` handler (line 230-244), the full `TrainingSessionState` is applied via `setSession(s)`. The `sub_step` field will be included automatically.

**Step 6: Compile check**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: compiles. May need to add `sub_step` to the initial session object in `startTraining`.

**Step 7: Commit**

```bash
git add helix/dashboard/src/hooks/useMpcTraining.ts
git commit -m "feat(dashboard): handle sub_step WebSocket events for real-time operation status"
```

---

### Task 9: Frontend — Update SubStatusTicker to Use Real Data

Replace hardcoded cycling operations with real backend sub-step data.

**Files:**
- Modify: `helix/dashboard/src/app/train/page.tsx:100-107` (MPC_STEP_OPERATIONS — remove)
- Modify: `helix/dashboard/src/app/train/page.tsx:237-281` (SubStatusTicker)

**Step 1: Remove MPC_STEP_OPERATIONS constant**

Delete lines 100-107 (the `MPC_STEP_OPERATIONS` array). It is no longer needed.

**Step 2: Rewrite SubStatusTicker**

Replace the function (lines 237-281) with:
```typescript
function SubStatusTicker({ session }: { session: TrainingSessionState }) {
  const text = useMemo(() => {
    // Non-training phases: show phase description
    if (session.phase !== 8 || session.current_step === 0) {
      return session.phase_description || 'Processing...';
    }

    // Cheater detection takes priority
    if (session.cheater_detected) {
      return `Cheater detected — worker ${session.cheater_detected.party_index} at step ${session.cheater_detected.step}`;
    }

    // Use real sub-step data from backend when available
    if (session.sub_step) {
      const workers = session.workers_active ?? 3;
      return `${session.sub_step} — ${workers} workers`;
    }

    // Fallback to phase description
    return session.phase_description || 'MPC training in progress...';
  }, [session.phase, session.current_step, session.phase_description,
      session.cheater_detected, session.sub_step, session.workers_active]);

  return (
    <AnimatePresence mode="wait">
      <motion.span
        key={text}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -6 }}
        transition={{ duration: 0.2 }}
        className="text-sm text-helix-muted"
      >
        {text}
      </motion.span>
    </AnimatePresence>
  );
}
```

Key changes:
- Removed `MPC_STEP_OPERATIONS` cycling
- Removed `mac_checks_passed % 5` and `checkpoints_submitted % 50` fake triggers
- Uses `session.sub_step` from real backend data
- Cleaner dependency array

**Step 3: Check for other references to MPC_STEP_OPERATIONS**

Search for `MPC_STEP_OPERATIONS` elsewhere in the file. If nothing else references it, the deletion is safe. The `SubStatusTicker` was the only consumer.

**Step 4: Compile check**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: compiles.

**Step 5: Commit**

```bash
git add helix/dashboard/src/app/train/page.tsx
git commit -m "feat(dashboard): replace fake sub-step cycling with real backend operation status"
```

---

### Task 10: Build and Integration Test

Verify everything compiles and the existing tests still pass.

**Step 1: Full Rust workspace build**

Run: `cd helix && cargo build --workspace`
Expected: compiles with no errors. Warnings about unused variables in test configs are acceptable.

**Step 2: Run helix-mpc tests**

Run: `cd helix && cargo test -p helix-mpc --lib -- --test-threads=2`
Expected: all existing tests pass. The new callback fields default to `None` so they don't affect existing behavior.

**Step 3: Run helix-client tests**

Run: `cd helix && cargo test -p helix-client --lib`
Expected: all existing tests pass. The `DashboardState::new()` signature changed (now returns a tuple), so any test that calls it directly needs updating. Check for test failures and fix.

**Step 4: Dashboard type check**

Run: `cd helix/dashboard && npx tsc --noEmit`
Expected: no type errors.

**Step 5: Dashboard lint**

Run: `cd helix/dashboard && npm run lint`
Expected: no new lint errors.

**Step 6: Commit any test fixes**

```bash
git add -A
git commit -m "fix: update tests for new DashboardState::new() signature"
```

---

### Task 11: Manual Smoke Test

Verify end-to-end live updates work by running the dashboard and starting a training job.

**Step 1: Start the backend**

Run: `cd helix && cargo run -p helix-client -- dashboard`
Expected: server starts on port 3001.

**Step 2: Start the frontend**

Run: `cd helix/dashboard && npm run dev`
Expected: Next.js dev server starts on port 3000.

**Step 3: Start a training job**

Open `http://localhost:3000/train`, configure a small training job (e.g., 3 workers, 20 steps), and start it.

**Verify:**
- [ ] Loss/accuracy chart updates step-by-step during training (not just at the end)
- [ ] Step counter increments live
- [ ] Status indicator shows real sub-operations (e.g., "Forward pass — layer 1 matmul — 3 workers")
- [ ] Phase ring progresses through phases 1-13
- [ ] On WebSocket reconnect (close and reopen browser tab), the snapshot includes current progress
- [ ] Polling fallback (disable WebSocket by blocking port) shows incremental progress every 3s

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: live dashboard updates with real sub-step operation status

Fixes loss/accuracy curves, step counter, and status indicator to update
in real-time during MPC training. Root cause was try_write() contention
causing silent state update drops — replaced with mpsc channel pattern.
Added real sub-step events from MPCTrainer at communication boundaries."
```
