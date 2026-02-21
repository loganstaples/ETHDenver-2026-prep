# Pause/Stop Training, Weight Storage & Live Cost Tracker

**Date**: 2026-02-20
**Status**: Design approved, pending implementation

## Summary

Add the ability for model owners to pause (resumable) or stop (permanent) training jobs. Add live cost tracking during training. Add weight storage options (IndexedDB local or 0G permanent). Implement deposit refund mechanics for unused funds.

## Decisions Made

- Workers are **released** on pause (can join other jobs); new workers recruited on resume
- Workers get **proportional payment** for steps completed when paused/stopped
- Cost data is **backend-computed** (orchestrator tracks gas + fees, sends via WebSocket)
- Local weight storage uses **IndexedDB** (sufficient for MNIST-scale models)
- Pause/stop only applies to **training** (inference is atomic and fast)
- Live cost tracker only applies to **training** (not inference)

---

## 1. Contract State Machine (HelixCoordinatorV4)

### Job Status Enum

```solidity
enum JobStatus { Active, Paused, Stopped, Completed }
```

Replaces the current `active` + `completed` boolean pair with a single `status` field.

### New Functions

#### `pauseTraining(uint256 jobId)`
- Callable by: job owner or operator
- Requires: `status == Active`
- Actions:
  - Set `status = Paused`, record `pausedAtStep = currentStep`
  - Release all workers: clear active worker list, decrement `workerActiveJobs`
  - Do NOT distribute payment (funds held in escrow)
  - Do NOT return deposit to owner
  - Workers can withdraw stake immediately (no 7-day cooldown)
- Emits: `TrainingPaused(jobId, pausedAtStep, releasedWorkerCount)`

#### `stopTraining(uint256 jobId)`
- Callable by: job owner or operator
- Requires: `status == Active` or `status == Paused`
- Actions:
  - Set `status = Stopped`
  - Calculate proportional worker payment: `(currentStep / numRounds) * paymentAmount`
  - Distribute proportionally to workers by steps participated (same formula as `completeTraining`)
  - Refund remaining payment to job owner: `paymentAmount - totalWorkerPayouts`
  - Set `jobCompletionTime` for stake withdrawal cooldown
  - Record `latestWeightCommitment` from last checkpoint as final
- Emits: `TrainingStopped(jobId, stoppedAtStep, refundAmount)`

#### `resumeTraining(uint256 jobId)`
- Callable by: job owner or operator
- Requires: `status == Paused`
- Actions:
  - Set `status = Active`
  - Accept optional `msg.value` to top up payment pool
  - Workers must re-join via `stakeAndJoin` (new workers allowed)
  - Training continues from `currentStep`
- Emits: `TrainingResumed(jobId, resumeFromStep)`

### Deposit Refund Logic

- `paymentAmount` = deposit (existing `msg.value` on `registerTrainingJob`)
- **Stop**: `refund = paymentAmount - workerPayouts` → sent to owner
- **Complete**: same (any remainder returned)
- **Pause**: no refund, funds escrowed
- Worker payout formula: `(currentStep / numRounds) * paymentAmount`, distributed proportionally by steps each worker participated

### State Transition Diagram

```
Active → Paused  (pauseTraining)
Active → Stopped (stopTraining)
Active → Completed (completeTraining - existing)
Paused → Active  (resumeTraining)
Paused → Stopped (stopTraining)
```

No transitions from Stopped or Completed.

### Inference Refund

- If user IS the model owner: no commission charged (skip commission calculation)
- Unused funds from inference deposit returned after inference completes

---

## 2. Rust Training Loop Interruptibility

### Signal Channel

```rust
enum TrainingSignal { Continue, Pause, Stop }
```

Add `tokio::sync::watch::Receiver<TrainingSignal>` to `MPCIntegrationConfig`. The training loop checks the channel after each step:

- `Pause`: Save checkpoint to disk, return `TrainingOutcome::Paused { step, checkpoint }`
- `Stop`: Save checkpoint to disk, return `TrainingOutcome::Stopped { step, checkpoint }`

### Training Outcome Enum

```rust
enum TrainingOutcome {
    Completed { accuracy: f64 },
    Paused { step: usize, checkpoint: CheckpointRecord },
    Stopped { step: usize, checkpoint: CheckpointRecord },
    Failed { error: String },
}
```

Replace the current implicit "runs to completion or errors" pattern with explicit outcomes.

### Resume Flow

1. Load checkpoint from disk (`checkpoint.rs`)
2. Create fresh transport mesh with new workers
3. Distribute saved weights to new workers via secret sharing
4. Set `starting_step` in training config (new field)
5. Training loop starts from `starting_step`
6. On-chain: `currentStep` already reflects position

### New Progress Events

```rust
ProgressEvent::TrainingPaused { step, checkpoint_commitment }
ProgressEvent::TrainingStopped { step, checkpoint_commitment, refund_amount }
ProgressEvent::CostUpdate { gas_spent_wei: u128, worker_fees_accrued: f64, total_cost_adi: f64 }
```

### Full Orchestration Changes

- Accept `watch::Sender<TrainingSignal>` for external control
- Phase 8 (MPC training) checks signal, may exit early
- **Pause flow**: Phase 8 → Phase 9 (submit final checkpoint) → return `Paused`
- **Stop flow**: Phase 8 → Phase 9 → Phase 10 (MAC reports) → on-chain stop → return `Stopped`
- **Resume flow**: Phases 1-2 load from checkpoint, Phase 5 calls `resumeTraining`, Phase 6 re-recruits workers

---

## 3. Weight Storage

### Storage Decision Flow

After training stops, pauses, or completes:

1. Prompt: "Save weights to 0G Storage?" (user pays 0G fee) or store locally
2. **Yes (0G)**: Upload encrypted weights, store CID on-chain via `ModelRegistry`
3. **No (local)**: Store in IndexedDB

### IndexedDB Schema

```typescript
interface StoredWeights {
  jobId: number;
  step: number;
  commitment: string;       // Pedersen commitment (matches on-chain)
  weights: ArrayBuffer;     // Serialized weight tensors
  timestamp: number;
  status: 'stopped' | 'paused' | 'completed';
  modelName: string;
}
```

New `useWeightStorage` hook for IndexedDB CRUD.

### Availability Rules

- **Stopped model (with checkpoint)**: Weights downloadable. Model treated as finished — usable for inference, new training (from step 0), marketplace listing. Has "Stopped" badge.
- **Stopped model (no checkpoint)**: No weights available. Model exists on-chain but without usable weights. Message: "Training stopped before first checkpoint — no weights available."
- **Paused model**: Weights downloadable from last checkpoint. NOT usable for inference or marketplace. Resume or stop only.
- **Completed model**: Weights downloadable. Full model functionality.

---

## 4. Live Cost Tracker

### Backend Cost Tracking

```rust
struct CostTracker {
    gas_spent_wei: u128,          // Actual gas consumed
    worker_fees_adi: f64,         // steps * workers * PER_STEP_FEE
    deposit_amount_adi: f64,      // Original deposit
}
```

Emits `CostUpdate` progress event after each gas-consuming operation.

### Dashboard Display

New "Costs So Far" panel on training page during Phase 8+:

- Worker compute fees (running total)
- Gas spent (checkpoint submissions, staking txs)
- Total consumed
- Deposit remaining (deposit - consumed)
- Progress bar showing deposit consumption percentage
- Updates live via WebSocket

---

## 5. Dashboard UI Changes

### Training Page

- **Pause button** (yellow): Appears during Phase 8. Saves checkpoint, shows "Paused at step N" state.
- **Stop button** (red): Confirmation dialog. Permanently ends training.
- **Paused state**: Shows last checkpoint info, "Resume Training" button, "Stop Training" button, weight download.
- **Live cost panel**: Below phase ring, same style as `CostBreakdownPanel` but live-updating.

### My Models Page

- **Status badges** on model cards:
  - Green: Completed
  - Yellow: Paused (with "Resume" action)
  - Red/gray: Stopped (with indicator)
  - Blue: Training (active)
- Training history entries show terminal status.

### Model Detail Pages

- **Stopped models**: Full page with "Stopped at step N/M" indicator. All normal actions.
- **Paused models**: Limited — download weights, resume/stop only. No inference or marketplace.

### Weight Management

- **Download button**: Model detail and my-models pages.
- **"Save to 0G" modal**: After training stops/completes.
- **Storage indicator**: Icon showing IndexedDB (local) vs 0G (cloud).

### Inference Page

- **Owner check**: "No commission (you own this model)" when user is model owner.
- **Refund tooltip**: Explains unused deposit refund.

---

## Files to Modify

### Contracts
- `helix/contracts/src/core/HelixCoordinatorV4.sol` — Add JobStatus enum, pause/stop/resume functions, refund logic
- `helix/contracts/test/HelixCoordinatorV4.t.sol` — Tests for all new functions

### Rust Crates
- `helix/crates/helix-mpc/src/e2e_integration.rs` — Add signal channel, TrainingOutcome, starting_step
- `helix/crates/helix-client/src/full_orchestration.rs` — Accept signal sender, handle pause/stop/resume flows
- `helix/crates/helix-client/src/session.rs` — Expose pause/stop API to dashboard
- `helix/crates/helix-client/src/demo/checkpoint.rs` — Ensure checkpoint supports resume with new workers
- `helix/crates/helix-demo/src/runner.rs` — Wire signal channel for demo mode
- `helix/crates/helix-demo/src/display.rs` — Display paused/stopped states

### Dashboard
- `helix/dashboard/src/app/train/page.tsx` — Pause/stop buttons, live cost panel, paused state UI
- `helix/dashboard/src/app/my-models/page.tsx` — Status badges, stopped/paused indicators
- `helix/dashboard/src/app/my-models/[id]/page.tsx` — Weight download, 0G storage prompt
- `helix/dashboard/src/app/models/[id]/page.tsx` — Stopped model indicator
- `helix/dashboard/src/app/inference/page.tsx` — Owner commission skip, refund tooltip
- `helix/dashboard/src/components/ui/TrainingWidget.tsx` — Paused/stopped terminal states
- `helix/dashboard/src/hooks/useModelRegistry.ts` — Model status field handling
- New: `helix/dashboard/src/hooks/useWeightStorage.ts` — IndexedDB weight CRUD
