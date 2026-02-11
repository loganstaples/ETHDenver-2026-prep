# helix-node/src/roles/ — Technical Review

## Overview

The `roles/` module defines the three node roles in the HELIX network: **Aggregator** (collects gradients, coordinates rounds), **Compute** (trains models, generates proofs), and **Verifier** (validates proofs with replay detection). Each role is implemented as a state machine with well-defined transitions. Total: ~1,600 lines across 4 files.

## Architecture

### Module Tree

```
roles/
├── mod.rs           (~30 lines)  Re-exports
├── aggregator.rs    (620 lines)  Gradient collection & aggregation state machine
├── compute.rs       (368 lines)  Training & proof generation state machine
└── verifier.rs      (590 lines)  Proof verification with policies & replay detection
```

### Key Types

| Type | File | Purpose |
|------|------|---------|
| `AggregatorNode` | aggregator.rs | Gradient collection, aggregation, round management |
| `AggregatorState` | aggregator.rs | Idle → Recruiting → Collecting → Aggregating → Complete |
| `ComputeNode` | compute.rs | Local training, proof generation, gradient sharing |
| `ComputeState` | compute.rs | Idle → Registered → Training → Proving → Ready |
| `VerifierNode` | verifier.rs | Proof verification with VerifyAll/Structural/Permissive |
| `VerificationPolicy` | verifier.rs | Determines verification depth |

### Data Flow

```
Round lifecycle through roles:

Aggregator (coordinator):
  Idle → start_round() → Recruiting
       → register_participant() × N → Collecting
       → receive_gradient() → Aggregating
       → aggregate() → Complete → submit_on_chain()

Compute (worker):
  Idle → register(aggregator) → Registered
       → receive_round_start() → Training
       → train_step() → Proving
       → generate_proof() → Ready → send_gradient()

Verifier (validator):
  receive_proof() → check_replay() → verify_proof(policy) → accept/reject
```

## Per-Module Analysis

### `aggregator.rs` — Aggregator Node (620 lines)

**What it does**: State machine for the aggregator role. Manages round lifecycle: recruits participants, collects gradient shares, aggregates using hash-based or Pedersen commitment schemes, tracks completion.

**Strengths**:
- Clean state machine with explicit transitions and guards (`aggregator.rs:50-120`): Each transition validates preconditions
- Two aggregation modes: hash-based (SHA-256) and Pedersen commitment (`aggregator.rs:250-350`)
- Shard assignment for participants (`aggregator.rs:150-200`): Supports data parallelism
- Round metadata tracking: participants, completion status, error bounds (`aggregator.rs:80-140`)

**Weaknesses**:
- **Pedersen aggregation is homomorphic addition only** (`aggregator.rs:300-350`): Adds commitment points, which only works for sum aggregation (FedAvg). Not compatible with Krum/Median/TrimmedMean.
  - **Impact**: Byzantine-tolerant aggregation requires opening commitments, defeating privacy
  - **Fix**: For Byzantine aggregation with privacy, use MPC-based aggregation (helix-mpc). Pedersen is correct for trusted FedAvg.
- **No timeout on Collecting state** (`aggregator.rs:180-230`): Waits for all registered participants. One slow/dead worker blocks the round.
  - **Impact**: Round hangs indefinitely
  - **Fix**: Add configurable collection deadline. After timeout, aggregate with available gradients and penalize missing workers via reputation.
- **XOR aggregation still present as fallback** (`aggregator.rs:280`): Hash-based aggregation uses XOR for combining hashes, which is not cryptographically sound.
  - **Impact**: Collision-prone, provides no binding guarantee
  - **Fix**: Use Merkle tree (as done in `round_commit.rs`) instead of XOR for hash-based aggregation. The better implementation already exists — use it.

**Tests**: 12 tests covering state transitions, participant registration, aggregation modes. **Well tested.**

### `compute.rs` — Compute Node (368 lines)

**What it does**: State machine for compute workers. Registers with aggregator, waits for round start, runs local training via `Trainer`, generates ZK proof, creates gradient share message.

**Strengths**:
- Clean state machine with proper transition guards (`compute.rs:40-100`)
- Integration with `Trainer` for real ML training + ZK proof generation (`compute.rs:150-200`)
- Creates properly formatted `GradientShare` messages for network transmission (`compute.rs:220-280`)

**Weaknesses**:
- **No retry on training failure** (`compute.rs:160-190`): If `Trainer.train_step()` fails (e.g. NaN loss, proof generation error), the compute node enters a failed state with no recovery.
  - **Impact**: Transient failures permanently disable the worker for the round
  - **Fix**: Add retry logic (up to N attempts) with backoff. On persistent failure, report to aggregator and skip round.
- **No resource monitoring** (`compute.rs:40-100`): No check on available memory/compute before accepting training work.
  - **Impact**: OOM during proof generation (which is memory-intensive) crashes the node
  - **Fix**: Check available memory before starting proof generation. Decline round participation if resources insufficient.
- **Gradient share includes raw gradient values** (`compute.rs:220-280`): When not using MPC, gradient values are sent in plaintext.
  - **Impact**: Weight privacy violated — aggregator and network observers can see exact gradients
  - **Fix**: Default to MPC-protected gradient sharing, or at minimum encrypt gradient shares to aggregator's public key

**Tests**: 3 tests covering state transitions, registration, basic training flow. **Adequate for the module size.**

### `verifier.rs` — Verifier Node (590 lines)

**What it does**: Validates proofs with three policies: `VerifyAll` (real Halo2 KZG verification), `Structural` (format/size checks), and `Permissive` (always accept). Includes replay detection via bounded hash set and concurrency control via tokio semaphore.

**Strengths**:
- Real Halo2 KZG verification in `VerifyAll` mode (`verifier.rs:150-250`): Not a stub — actually runs pairing checks
- Replay detection with bounded set (`verifier.rs:300-370`): Prevents the same proof from being submitted twice, with configurable max tracked proofs
- Concurrency control (`verifier.rs:380-420`): Semaphore limits concurrent verifications to prevent CPU exhaustion
- Structural validation checks (`verifier.rs:100-140`): Proof size >= 384 bytes, non-zero hash fields, valid step number range

**Weaknesses**:
- **Replay set is bounded but eviction is FIFO** (`verifier.rs:310-340`): When the set reaches capacity, oldest entries are evicted. An attacker can flush the replay cache by submitting many unique proofs, then replay an old one.
  - **Impact**: Replay protection defeated by cache flooding
  - **Fix**: Use a Bloom filter (space-efficient, no eviction needed for reasonable time windows) or time-partitioned sets (keep proofs for last N rounds, guaranteed no eviction within window)
- **Structural policy is too lenient** (`verifier.rs:100-140`): Only checks `proof.len() >= 384`, non-zero hashes, and step number range. Any random 384+ bytes pass.
  - **Impact**: Trivial to forge proofs that pass Structural verification
  - **Fix**: Add more structural checks: validate proof points are on BN254 curve (deserialize and check), validate public inputs are in field range (< BN254 p). These are cheap and catch random data.
- **No batch verification** (`verifier.rs:150-250`): Each proof verified independently. Halo2 KZG supports batch verification (amortize pairing across multiple proofs).
  - **Impact**: Verifying N proofs takes N × single_verify_time instead of ~1.5 × single_verify_time
  - **Fix**: Collect proofs for batch, use `verify_proof_multi` with multiple proof/instance pairs. The function already supports this.

**Tests**: 15 tests covering all policies, replay detection, concurrency, structural checks. **Well tested.**

## Strengths Summary

1. **Clean state machines**: Both Aggregator and Compute use explicit state enums with guarded transitions — impossible to call `aggregate()` before collection is complete.
2. **Real verification**: VerifierNode actually runs Halo2 KZG pairing checks, not just format validation.
3. **Security features**: Replay detection and concurrency control in Verifier show security awareness.
4. **Good test coverage**: 30 tests across the module (12 + 3 + 15), above average for helix-node.

## Weaknesses Summary (Prioritized)

### Critical

1. **Replay cache floodable** (`verifier.rs:310-340`): FIFO eviction allows cache flushing attack.
   - **Fix**: Time-partitioned sets or Bloom filter.

### High Priority

2. **No collection timeout in Aggregator** (`aggregator.rs:180-230`): Dead worker blocks round.
   - **Fix**: Deadline-based finalization.

3. **XOR aggregation in hash mode** (`aggregator.rs:280`): Not cryptographically sound.
   - **Fix**: Use Merkle tree from `round_commit.rs`.

### Nice to Have

4. **No batch verification** (`verifier.rs:150-250`): Performance opportunity.
5. **No retry in Compute** (`compute.rs:160-190`): Transient failures are permanent.
6. **Plaintext gradients without MPC** (`compute.rs:220-280`): Privacy gap.

## Testing Assessment

| Module | Tests | Coverage | Assessment |
|--------|-------|----------|------------|
| aggregator.rs | 12 | High | State transitions, aggregation modes |
| compute.rs | 3 | Medium | Basic state flow, needs failure cases |
| verifier.rs | 15 | High | All policies, replay, concurrency |

**Total**: 30 tests. Well tested overall.

**Missing tests**:
- Aggregator timeout/deadline behavior
- Compute training failure and recovery
- Verifier with real malformed proofs (not just random bytes)
- Verifier batch verification
- Role interaction (Aggregator + Compute + Verifier in same test)

## Demo Readiness

| Feature | Status | Notes |
|---------|--------|-------|
| Aggregator state machine | Ready | Happy path works |
| Pedersen aggregation | Ready | Correct for FedAvg |
| Compute training | Ready | Real ML + ZK proofs |
| Verifier (Structural) | Ready | Fast, good for demo |
| Verifier (VerifyAll) | Ready | Real Halo2 KZG |
| Replay detection | Ready | Works within cache capacity |
| Cross-role coordination | Ready | Via network messages |

**Demo target**: Three roles working together across network. **Ready for demo.**

## Summary

### Health Score: **B** (72/100)

The roles module is one of the better-tested parts of helix-node with clean state machine designs and real cryptographic verification. The Aggregator and Compute roles correctly implement their lifecycle with proper guards. The Verifier is particularly strong with three policy tiers, replay detection, and concurrency control. The main concerns are the replay cache's vulnerability to flooding, the Aggregator's lack of collection timeout (dead worker blocks everything), and the hash-mode XOR aggregation that should use Merkle trees instead. For demo, all three roles work correctly on the happy path.
