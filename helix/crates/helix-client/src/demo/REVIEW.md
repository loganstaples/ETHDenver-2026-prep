# demo/ Module Review

**Score: B (78%)**
**Verdict:** Well-designed demo framework with real ZK proof generation. Timing orchestration and error recovery are production-quality. Pre-warming is mostly placeholder.

---

## Architecture

```
demo/
  mod.rs            # Re-exports + phased demo scenarios
  orchestrator.rs   # 90-second timed demo execution
  prewarm.rs        # Resource pre-warming + GPU detection
  real_training.rs  # REAL ZK proof generation via MLTrainingProverV2
  recovery.rs       # Error recovery + circuit breaker + heartbeat (1,334 lines)
```

Total: ~3,500+ lines across 5 files.

---

## Per-File Analysis

### mod.rs — Demo Scenarios
Declares 5 phased demo types: Quick, FullTraining, Slashing, MultiModel, FaultTolerance. Re-exports all public types from submodules. Clean module organization.

### orchestrator.rs — Timing Orchestration

**DemoTimingConfig** allocates time budgets:
- Setup: 10% (9s of 90s)
- Training: 70% (63s)
- Finalization: 10% (9s)
- Buffer: 10% (9s)
- Hard deadline: 100 seconds

**DemoOrchestrator** features:
- 13-phase state machine (PreStart through Complete/Failed)
- Adaptive pacing: pace_factor = target_progress / actual_progress
- If pace_factor > 1.2, speeds up remaining phases
- Graceful degradation: skips rounds if time runs critically low
- Minimum visibility window (500ms) per phase for UI
- Event-driven subscriber model via MPSC channel
- Snapshot reporting for external monitoring

**Real vs Simulated:** Real timing logic, but RPC calls go through MockRpcClient. Training state advances are simulated.

### prewarm.rs — Pre-warming

**DemoPrewarmer** orchestrates 4 stages:
1. Proving keys: loads from disk or creates 1 MB placeholder
2. Model weights: creates 2 MB placeholder (real impl would load from IPFS)
3. Network: simulates peer connections (30ms delay)
4. GPU: real detection via `nvidia-smi` (Linux) / `system_profiler` (macOS)

**Cache:** In-memory LRU-like cache with hit tracking and eviction when memory limit exceeded.

**Real vs Simulated:** GPU detection is real. Everything else is placeholder.

### real_training.rs — The Crown Jewel

**RealTrainingExecutor** (synchronous):
- `initialize()`: Builds `MLTrainingProverV2` with `V2ProverConfig` — expensive (~2-5s)
- `train_step()`: Generates real witness, calls `prover.prove()` (500-3000ms), updates weights, tracks error bounds
- `generate_invalid_proof()`: Corrupts proof bits and inflates error bound for slashing demo
- `stats()`: Returns step count, proof count, avg proof time, loss history

**AsyncRealTrainingExecutor** (thread-safe wrapper):
- `Arc<RwLock<RealTrainingExecutor>>`
- `train_step()` uses `tokio::task::spawn_blocking()` — critical for preventing async runtime congestion

**Key Details:**
- Model dims: d_in=16, d_hid=32, d_out=4 (small but real)
- Circuit k=12 (4096 rows)
- Fixed-point encoding: f64 * 2^16 for Fr conversion
- Weight initialization: Xavier scaling
- Loss simulation: multiply by 0.92 per step (synthetic convergence)
- Training batch: random input + one-hot target

### recovery.rs — Error Recovery (1,334 lines)

**RecoveryManager:**
- 9 error categories (NetworkError, NodeFailure, ProofError, TransactionError, ResourceError, Timeout, DataError, ConfigError, Unknown)
- 8 recovery actions (Retry, Skip, Restart, Fallback, Degrade, Abort, WaitAndRetry, ManualIntervention)
- Escalation: repeated errors for same component trigger increasingly aggressive recovery
- `can_continue()`: gating decision based on worker count and recovery rate

**CircuitBreaker:**
- State machine: Closed -> Open (after failure_threshold) -> HalfOpen (after reset_timeout) -> Closed (after success_threshold)
- Demo mode: 5 failure threshold, 2 success threshold, 10s timeout

**HeartbeatMonitor:**
- Tracks component liveness via last-heartbeat timestamps
- Active checking with optional TCP probe
- Configurable timeout (default 15s)

**Pre-Demo Checks:**
- Memory: attempts 64 MB allocation
- Network: DNS resolution of 1.1.1.1
- RPC endpoint: TCP connection to configured endpoint
- Proving system: checks component registry health
- Error state: verifies no unresolved errors
- Reliability score (0-100): weighted deductions per failed check

---

## Strengths

1. **Real ZK proof generation** — `real_training.rs` calls `MLTrainingProverV2::prove()`. Not a simulation. This is the most valuable module in the entire crate for demo credibility.

2. **`spawn_blocking()` for proof generation** — Correctly avoids blocking the tokio runtime during expensive proof computation. This is a common mistake; getting it right demonstrates understanding of async Rust.

3. **Adaptive demo pacing** — `DemoOrchestrator` dynamically adjusts phase timing based on actual vs expected progress. If proofs take longer than expected, it reduces the number of training rounds to stay within the 90-second budget.

4. **Comprehensive error recovery** — The recovery framework handles 9 error categories with escalating recovery strategies. Pre-demo checks catch common issues before the demo starts. Circuit breaker prevents repeated failures.

5. **`generate_invalid_proof()`** — Enables the adversarial/slashing demo scenario by corrupting proof bytes and inflating error bounds. This is key for demonstrating the protocol's security properties.

---

## Weaknesses

### HIGH

**`fr_to_f64()` conversion is approximate** (`real_training.rs`)
- Takes low 8 bytes of field element representation
- Loses precision for values that use the full 254-bit field
- **Impact:** Weight updates may drift from expected values over many training steps
- **Fix:** Use the same 2^16 fixed-point scale in reverse: extract the low bytes, divide by 2^16

**Loss simulation (0.92x per step) is disconnected from actual weights** (`real_training.rs`)
- Real weights are updated from witness, but displayed loss is synthetic
- **Impact:** Dashboard shows fake convergence curve even if weights diverge
- **Fix:** Compute actual loss from witness data (the loss field is in the proof's public inputs)

**Pre-warm proving keys are placeholder** (`prewarm.rs`)
- Creates 1 MB zero-filled allocation instead of actual `ParamsKZG`
- **Impact:** Pre-warming doesn't actually reduce first proof latency
- **Fix:** Generate or load real SRS parameters during pre-warm. `helix-demo/srs.rs` already has SRS caching logic.

### MEDIUM

**Orchestrator uses MockRpcClient** (`orchestrator.rs`)
- Phase transitions don't reflect real system state
- **Impact:** Demo phases may not match actual training progress
- **Fix:** Wire orchestrator to `UnifiedRpcClient` connected to real training execution

**`check_proving_system()` defaults to healthy if not registered** (`recovery.rs`)
- `registry.is_healthy("proof_generator").unwrap_or(true)` — absence is treated as healthy
- **Impact:** Pre-demo check won't catch a missing proving system
- **Fix:** Return false (unhealthy) when component not registered

### NICE-TO-HAVE

- No Trezor model detection in prewarm GPU check
- Network prewarm is just a 30ms sleep, not real peer discovery
- Recovery event channel has no backpressure
- `RealTrainingConfig` has `d_in=16, d_hid=32, d_out=4` hardcoded — should be configurable

---

## Demo Readiness

| Aspect | Status | Notes |
|--------|--------|-------|
| 90-second timing | PASS | Hard deadline + adaptive pacing |
| Real ZK proofs | PASS | MLTrainingProverV2 via spawn_blocking |
| Slashing demo | PASS | generate_invalid_proof() corrupts proofs |
| Error recovery | PASS | Circuit breaker + heartbeat + escalation |
| Pre-demo checks | PARTIAL | Memory + network real; proving system always "healthy" |
| GPU detection | PASS | nvidia-smi / system_profiler |
| Pre-warming | FAIL | Proving keys are 1 MB placeholder, not real SRS |

## Test Coverage

- `RecoveryConfig::default()` — basic config test
- `ErrorCategory::is_recoverable()` — classification test
- `RecoveryManager::record_error()` — error recording
- `RecoveryManager::can_continue()` — gating decision
- `RecoveryManager::health_check()` — component health
- `RecoveryManager::reset()` — state reset

**Gap:** No tests for `DemoOrchestrator` phase timing, `RealTrainingExecutor` proof generation, or `DemoPrewarmer` cache eviction.
