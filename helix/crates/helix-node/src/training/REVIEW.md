# helix-node/src/training/ — Technical Review

## Overview

The `training/` module implements distributed training coordination for the HELIX network. It is the most complex module in `helix-node` at ~9,000+ lines across 16 files, handling orchestration, Byzantine-tolerant aggregation, BFT consensus, proof verification, MPC integration, fault tolerance, checkpointing, and distributed state machines. This module bridges the gap between local ML training (in `trainer.rs`) and on-chain verification (in `sc_client.rs`).

## Architecture

### Module Tree

```
training/
├── mod.rs                    (~50 lines)   Re-exports
├── orchestrator.rs           (1818 lines)  Multi-node training orchestrator
├── coordinator.rs            (~800 lines)  Training coordination
├── distributed_coordinator.rs (~500 lines) High-level distributed coordination
├── consensus.rs              (1094 lines)  BFT 2-phase commit
├── verification.rs           (1210 lines)  Halo2 KZG proof verification
├── aggregation.rs            (717 lines)   Byzantine-tolerant gradient aggregation
├── round.rs                  (~300 lines)  Round state machine
├── session.rs                (~250 lines)  Training session
├── session_manager.rs        (~350 lines)  Session lifecycle
├── model.rs                  (~200 lines)  Model weight management
├── metrics.rs                (~150 lines)  Training metrics
├── mpc.rs                    (~400 lines)  MPC integration
├── data_loader.rs            (~250 lines)  Training data loading
├── checkpoint.rs             (~300 lines)  Local checkpointing
├── distributed_checkpoint.rs (~400 lines)  Coordinated checkpoints
├── state_machine.rs          (~500 lines)  Distributed state machine
├── synchronization.rs        (~350 lines)  Barrier synchronization
└── fault_tolerance.rs        (~400 lines)  Failure detection/recovery
```

### Key Types

| Type | File | Purpose |
|------|------|---------|
| `TrainingOrchestrator` | orchestrator.rs | Top-level coordinator with round lifecycle |
| `BftConsensus` | consensus.rs | Byzantine-fault-tolerant 2-phase commit |
| `ProofVerifier` | verification.rs | Real Halo2 KZG proof verification with policies |
| `GradientAggregator` | aggregation.rs | 6 Byzantine-tolerant aggregation strategies |
| `TrainingRound` | round.rs | Round state machine (Initializing→Completed) |
| `TrainingSession` | session.rs | Per-model training session |
| `SessionManager` | session_manager.rs | Session lifecycle management |
| `MPCTrainingBridge` | mpc.rs | Bridge to helix-mpc for private training |
| `DistributedStateMachine` | state_machine.rs | Replicated state machine |
| `BarrierSync` | synchronization.rs | Distributed barrier synchronization |
| `FaultDetector` | fault_tolerance.rs | Failure detection and recovery |

### Data Flow

```
1. Orchestrator creates round → broadcasts RoundStart to workers
2. Workers run Trainer.train_step() → produce ProvedStep
3. Workers send gradient shares via gossip to aggregator
4. Aggregator collects gradients, runs BFT consensus:
   a. Phase 1: Collect binding commitments from all parties
   b. Phase 2: Reveal gradients, verify against commitments
   c. Aggregate using selected strategy (Krum, TrimmedMean, etc.)
5. ProofVerifier validates ZK proofs (VerifyAll/Sample/StructuralOnly)
6. RoundCommitManager submits aggregated proof on-chain
7. Orchestrator advances to next round
```

### Dependencies

- Internal: `helix-core` (tensors, types), `helix-avm` (quantization), `helix-prover` (MLTrainingProverV2, proof verification), `helix-mpc` (MPCTrainer, secret sharing)
- External: `tokio` (async), `rand` (sampling), `sha2` (commitments), `halo2_proofs` (verification keys/proofs)

## Per-Module Analysis

### `orchestrator.rs` — Training Orchestrator (1818 lines)

**What it does**: The central coordinator for multi-node training. Manages the full round lifecycle: heartbeat monitoring, gradient collection, BFT consensus integration, proof verification, and on-chain submission via RoundCommitManager.

**Strengths**:
- Complete round lifecycle management (`orchestrator.rs:200-400`): Initializing → Collecting → Aggregating → Committing → Completed/Failed
- Heartbeat-based worker liveness (`orchestrator.rs:500-580`): Detects and removes dead workers
- Integration with RoundCommitManager for on-chain submission (`orchestrator.rs:700-800`)
- Configurable minimum participants and timeout (`orchestrator.rs:50-90`)

**Weaknesses**:
- **Massive file** (1818 lines): Too much responsibility in one type. Contains round management, heartbeat monitoring, gradient collection, consensus coordination, proof verification, and on-chain submission.
  - **Impact**: Hard to test, modify, or reason about any single concern
  - **Fix**: Extract into `OrchestratorHeartbeat`, `OrchestratorRound`, `OrchestratorConsensus`, `OrchestratorSubmission` sub-structs with clear interfaces
- **No leader re-election** (`orchestrator.rs:100-150`): Orchestrator is the single aggregator. If it fails, training halts.
  - **Impact**: Single point of failure
  - **Fix**: Implement Raft-style leader election among aggregator candidates, with automatic failover
- **Synchronous gradient collection** (`orchestrator.rs:600-700`): Waits for all workers to submit before aggregation. Slow workers block the entire round.
  - **Impact**: Round duration determined by slowest worker
  - **Fix**: Add timeout-based finalization — aggregate with available gradients after deadline, penalize missing workers

**Tests**: 2 tests. Vastly under-tested for 1818 lines of critical coordination logic.

### `consensus.rs` — BFT 2-Phase Commit (1094 lines)

**What it does**: Byzantine-fault-tolerant consensus for gradient aggregation. Phase 1: parties submit binding commitments (SHA-256 hash of gradient). Phase 2: parties reveal gradients, verified against commitments. Supports `f < n/3` Byzantine faults.

**Strengths**:
- Correct BFT 2-phase structure (`consensus.rs:100-250`): Binding commitment prevents equivocation
- Equivocation detection (`consensus.rs:300-380`): Detects when a party sends different values to different peers
- Quorum tracking with `f < n/3` threshold (`consensus.rs:400-450`)
- Clean state machine: Idle → Committing → Revealing → Decided/Failed (`consensus.rs:50-90`)

**Weaknesses**:
- **SHA-256 commitment is not hiding** (`consensus.rs:150-180`): `commit = SHA256(gradient)` means the same gradient always produces the same commitment. Attacker can pre-compute commitments for likely gradients and learn others' values before reveal.
  - **Impact**: Information leak in Phase 1 — defeats purpose of commitment scheme
  - **Fix**: Use `commit = SHA256(gradient || nonce)` with random nonce revealed in Phase 2. Or use Pedersen commitments from helix-mpc for information-theoretic hiding.
- **No view change** (`consensus.rs:50-90`): If the consensus leader stalls (doesn't collect enough commitments), the protocol halts.
  - **Impact**: Malicious or crashed leader blocks consensus indefinitely
  - **Fix**: Add timeout-based view change: if no progress in T seconds, elect new leader and restart phase
- **Gradient serialization for commitment** (`consensus.rs:160-175`): Gradients are serialized to bytes for hashing, but no canonical serialization format is enforced.
  - **Impact**: Different serialization → different hashes → false equivocation detection
  - **Fix**: Define canonical gradient encoding (sorted keys, fixed-point representation, deterministic byte order)

**Tests**: 12 tests covering commitment/reveal, equivocation, quorum, state transitions. Good coverage.

### `verification.rs` — Proof Verification (1210 lines)

**What it does**: Verifies ZK proofs using real Halo2 KZG verification or structural checks. Supports three policies: `VerifyAll` (every proof through Halo2), `SampleVerify` (random sample), `StructuralOnly` (size/format checks only). Includes proof caching and Byzantine gradient filtering.

**Strengths**:
- Real Halo2 KZG verification integrated (`verification.rs:200-320`): Uses `verify_proof_multi` with ParamsKZG and VerifyingKey
- Three verification policies with clear tradeoffs (`verification.rs:40-80`): Full, sampled, structural
- Proof cache to avoid re-verifying identical proofs (`verification.rs:400-450`)
- Byzantine gradient filtering based on statistical outlier detection (`verification.rs:500-600`)

**Weaknesses**:
- **Default policy is StructuralOnly** (`verification.rs:45`): The default verification policy only checks proof format (size >= 384 bytes, non-zero fields), not cryptographic validity.
  - **Impact**: Invalid proofs pass verification by default. This is the **#5 critical issue** from the health assessment.
  - **Fix**: Change default to `VerifyAll` or at minimum `SampleVerify(0.5)`. Make `StructuralOnly` require explicit opt-in.
- **SampleVerify randomness source** (`verification.rs:280-300`): Uses `rand::thread_rng()` for sampling decision.
  - **Impact**: Not deterministic — can't reproduce which proofs were/weren't verified for auditing
  - **Fix**: Use seeded RNG derived from round number + model ID for reproducible sampling
- **Cache key is proof bytes hash** (`verification.rs:410-430`): Caching verified proofs by their hash. But cache doesn't bind to public inputs.
  - **Impact**: Proof verified for one set of public inputs could be served from cache for different inputs
  - **Fix**: Cache key should be `hash(proof_bytes || public_inputs_bytes)`

**Tests**: 12 tests covering all three policies, cache behavior, Byzantine filtering. Good coverage.

### `aggregation.rs` — Byzantine-Tolerant Aggregation (717 lines)

**What it does**: Implements 6 aggregation strategies: FedAvg (simple average), Krum (most representative), MultiKrum (top-k Krum), Median (coordinate-wise), TrimmedMean (remove outliers, average rest), GeometricMedian (Weiszfeld algorithm).

**Strengths**:
- 6 strategies covering the Byzantine aggregation literature (`aggregation.rs:40-80`)
- Krum correctly computes pairwise distances and selects most representative (`aggregation.rs:120-200`)
- TrimmedMean with configurable trim ratio (`aggregation.rs:300-380`)
- GeometricMedian using iterative Weiszfeld algorithm (`aggregation.rs:400-500`)

**Weaknesses**:
- **FedAvg is default but not Byzantine-tolerant** (`aggregation.rs:50`): Simple average is trivially manipulated by a single malicious party.
  - **Impact**: One Byzantine worker can shift the aggregated gradient arbitrarily
  - **Fix**: Default to Krum or TrimmedMean for Byzantine tolerance. FedAvg only when all workers are trusted.
- **Krum assumes f < n/3** (`aggregation.rs:130`): The `n - f - 2` nearest neighbors calculation requires knowing f.
  - **Impact**: Wrong f parameter → Krum selects manipulated gradient
  - **Fix**: Make f a required config parameter with validation `f < n/3`, warn when `n < 4` (need at least 4 workers for f=1)
- **GeometricMedian convergence** (`aggregation.rs:450-490`): Weiszfeld algorithm with fixed iteration count (50) and convergence threshold.
  - **Impact**: May not converge for certain gradient distributions, or waste iterations when already converged
  - **Fix**: This is already reasonable for demo. For production, add early termination AND max iteration limit.

**Tests**: 2 tests. **Severely under-tested** for 6 aggregation strategies with complex math. Each strategy needs at least 3 tests (normal case, Byzantine case, edge case).

### `round.rs` — Round State Machine

**What it does**: Tracks training round state: Initializing → Collecting → Aggregating → Committing → Completed/Failed. Each state has associated data (participants, gradients collected, aggregated result).

**Strengths**:
- Clean state machine with explicit transitions (`round.rs:40-100`)
- State-specific data prevents accessing data that doesn't exist yet

**Weaknesses**:
- **No timeout on states** (`round.rs:40-100`): A round stuck in "Collecting" waits forever.
  - **Impact**: Stuck rounds block future rounds
  - **Fix**: Add per-state timeout, auto-transition to Failed if exceeded

**Tests**: Part of orchestrator tests. No dedicated round.rs tests.

### `mpc.rs` — MPC Integration

**What it does**: Bridges to helix-mpc for private training. `MPCTrainingBridge` coordinates secret sharing of gradients, secure aggregation, and result reconstruction.

**Strengths**:
- Clean bridge pattern — wraps helix-mpc types into training module interface (`mpc.rs:40-100`)
- Supports both standalone MPC training and hybrid (MPC + ZK proof) modes

**Weaknesses**:
- **MPC + ZK proof composition unclear** (`mpc.rs:200-300`): How does a secret-shared gradient produce a ZK proof? The bridge doesn't clearly define the witness extraction process.
  - **Impact**: MPC and ZK may produce inconsistent state
  - **Fix**: Document the composition: MPC aggregation produces cleartext aggregated gradient → quantize → generate ZK proof of aggregation correctness
- **No error recovery** (`mpc.rs:250-300`): MPC protocol failures (network, share corruption) propagate as errors but aren't retried.
  - **Impact**: One failed MPC round kills the training session
  - **Fix**: Add retry logic with fresh shares for transient failures, session abort for persistent failures

**Tests**: No dedicated tests. Tested only through integration tests.

### `state_machine.rs` — Distributed State Machine

**What it does**: Replicated state machine for coordinating distributed training state across nodes. Handles state transitions, conflict resolution, and state replication.

**Strengths**:
- Explicit state transition validation (`state_machine.rs:100-200`)
- Version numbering for state updates

**Weaknesses**:
- **No formal replication protocol** (`state_machine.rs:200-400`): State is broadcast but there's no consensus on state transitions. Two nodes can apply different transitions and diverge.
  - **Impact**: State divergence between nodes → inconsistent training
  - **Fix**: Route state transitions through BftConsensus before applying locally

### `fault_tolerance.rs` — Failure Detection & Recovery

**What it does**: Detects worker/aggregator failures via heartbeat timeouts and provides recovery strategies (replace worker, restart round, abort session).

**Strengths**:
- Multiple recovery strategies with severity levels (`fault_tolerance.rs:100-200`)
- Escalating recovery: retry → replace → restart → abort

**Weaknesses**:
- **Recovery is advisory only** (`fault_tolerance.rs:200-300`): Recommends actions but doesn't execute them. The orchestrator must implement recovery.
  - **Impact**: Recovery strategies exist but may not be properly invoked
  - **Fix**: Tighter integration with orchestrator — register recovery handlers that execute automatically

### `checkpoint.rs` & `distributed_checkpoint.rs`

**What it does**: Local checkpointing saves model weights/optimizer state to disk. Distributed checkpointing coordinates checkpoint creation across all participants.

**Strengths**:
- Atomic checkpoint writes (write to temp, rename) (`checkpoint.rs:80-120`)
- Checkpoint verification with SHA-256 integrity (`checkpoint.rs:150-200`)
- Distributed checkpoint coordination with barrier sync (`distributed_checkpoint.rs:100-200`)

**Weaknesses**:
- **No checkpoint pruning** (`checkpoint.rs:80-120`): Checkpoints accumulate on disk without limit.
  - **Impact**: Disk exhaustion on long training runs
  - **Fix**: Keep last N checkpoints + every Kth checkpoint, delete others. `LocalStorage` in `storage/local.rs` has GC — wire it to checkpoint manager.

### `synchronization.rs` — Barrier Synchronization

**What it does**: Distributed barrier for coordinating round transitions. All participants must reach barrier before round advances.

**Strengths**:
- Timeout-based barrier with configurable deadline (`synchronization.rs:60-120`)
- Partial arrival tracking

**Weaknesses**:
- **Blocking barrier** (`synchronization.rs:100-150`): Workers that arrive early wait idly.
  - **Impact**: Wasted compute time
  - **Fix**: Allow early-arriving workers to prefetch next round's data or begin speculative training

## Strengths Summary

1. **Real Halo2 verification**: Not just structural checks — actual KZG pairing-based verification is implemented and tested (`verification.rs:200-320`)
2. **Comprehensive aggregation strategies**: 6 Byzantine-tolerant methods from the distributed ML literature — Krum, MultiKrum, Median, TrimmedMean, GeometricMedian, plus FedAvg (`aggregation.rs`)
3. **BFT consensus**: Proper 2-phase commit with binding commitments and equivocation detection (`consensus.rs`)
4. **Complete round lifecycle**: From orchestration through collection, aggregation, verification, to on-chain submission — the full pipeline exists
5. **MPC bridge**: Secret-shared gradient aggregation path exists for weight privacy

## Weaknesses Summary (Prioritized)

### Critical

1. **Default verification is StructuralOnly** (`verification.rs:45`): Proofs are not cryptographically verified by default. Any correctly formatted 384+ byte payload passes.
   - **Fix**: Default to `VerifyAll`. Require explicit opt-in for weaker policies.

2. **SHA-256 commitments aren't hiding** (`consensus.rs:150-180`): Gradient commitments leak information pre-reveal.
   - **Fix**: Add random nonce to commitment: `SHA256(gradient || nonce)`.

3. **Proof verification cache doesn't bind public inputs** (`verification.rs:410-430`): Can serve cached verification result for wrong inputs.
   - **Fix**: Include public inputs in cache key.

### High Priority

4. **Orchestrator has no fault tolerance** (`orchestrator.rs:100-150`): Single aggregator, no failover.
   - **Fix**: Leader election among aggregator candidates.

5. **FedAvg default is not Byzantine-tolerant** (`aggregation.rs:50`): Trivially manipulated.
   - **Fix**: Default to Krum or TrimmedMean.

6. **Orchestrator is 1818 lines** (`orchestrator.rs`): Unmaintainable monolith.
   - **Fix**: Extract sub-concerns into focused structs.

### Nice to Have

7. **Aggregation has only 2 tests** (`aggregation.rs`): Complex math with no Byzantine case tests.
8. **No checkpoint pruning** (`checkpoint.rs`): Disk exhaustion risk.
9. **State machine has no consensus backing** (`state_machine.rs`): State can diverge.

## Testing Assessment

| Module | Tests | Coverage | Assessment |
|--------|-------|----------|------------|
| orchestrator.rs | 2 | Very Low | 1818 lines, 2 tests. Critical gap. |
| consensus.rs | 12 | High | Commitment, reveal, equivocation, quorum |
| verification.rs | 12 | High | All policies, cache, Byzantine filtering |
| aggregation.rs | 2 | Very Low | 6 strategies, 2 tests. Major gap. |
| round.rs | 0 | None | Tested indirectly via orchestrator |
| session.rs | 0 | None | Tested indirectly |
| session_manager.rs | 0 | None | No tests |
| mpc.rs | 0 | None | Integration tests only |
| state_machine.rs | 0 | None | No tests |
| synchronization.rs | 0 | None | No tests |
| fault_tolerance.rs | 0 | None | No tests |
| checkpoint.rs | ~3 | Low | Basic save/load |
| distributed_checkpoint.rs | ~2 | Low | Coordination only |
| data_loader.rs | ~2 | Low | Basic loading |

**Integration tests** (in `tests/`):
- `distributed_training_integration.rs`: 11 tests covering state machine, fault tolerance, checkpointing, E2E flow
- `multi_worker.rs`: 10 tests with mock contract, session manager, proof aggregation
- `halo2_verification_integration.rs`: 4 tests for real Halo2 KZG verification

**Total**: ~60 tests. Consensus and verification are well-tested. Orchestrator, aggregation, and all auxiliary modules are severely under-tested.

## Demo Readiness

| Feature | Status | Notes |
|---------|--------|-------|
| Round orchestration | Ready | Happy path works |
| Gradient collection | Ready | Collects from workers |
| BFT consensus | Ready | 2-phase commit works |
| Proof verification | Ready | Real Halo2 with caveats (StructuralOnly default) |
| Aggregation (FedAvg) | Ready | Simple average works |
| Aggregation (Byzantine) | Ready | Krum tested |
| MPC training | Partial | Bridge exists, integration untested |
| Fault tolerance | Partial | Detection works, recovery advisory |
| On-chain submission | Ready | Via RoundCommitManager |
| Checkpointing | Ready | Local save/load works |

**Demo target**: Multi-worker training rounds with ZK proof aggregation and on-chain submission. **Achievable — the happy path works end-to-end.**

## Summary

### Health Score: **C+** (58/100)

The training module has impressive breadth — BFT consensus, 6 aggregation strategies, real Halo2 verification, MPC bridge, distributed state machine, fault tolerance — but suffers from integration gaps and critical defaults. The **default StructuralOnly verification** means proofs aren't actually verified cryptographically unless explicitly configured. The **SHA-256 commitments leak information** in the consensus protocol. The **orchestrator is a 1818-line monolith** with only 2 tests. The **aggregation module has 6 strategies and 2 tests**. For demo purposes, the happy path works: rounds orchestrate, gradients collect, proofs aggregate, on-chain submission succeeds. For production, the defaults are dangerous and the testing is inadequate for the complexity of the distributed protocols.
