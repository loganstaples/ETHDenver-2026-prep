# ZK Proof Integration Design

## Problem

The optional ZK proof path exists in pieces but is not wired end-to-end:
- `StateTransitionCircuit` (6 public inputs) and `CheckpointProver` are real Halo2 circuits
- `Halo2VerifierCore` is a real BN254 pairing verifier
- `HelixCoordinatorV4` has `zkEnabled`, `riskZkEnabled`, `zkActivatedByRisk` fields
- Dashboard has a three-mode ZK selector UI and handles `zk_proof_generated` WebSocket events

But:
1. `LazyZkProver` is never called from the demo runner
2. `LazyZkProver::generate_proof()` requires live `&[MPCTrainer]` handles that are consumed inside `run_mpc_training()`
3. No risk assessment logic exists in the Rust demo
4. `submitCheckpointWithProof` is missing from the dashboard's V4 ABI
5. Rust backend never emits `zk_proof_started` or `zk_proof_generated` WebSocket events

## Three ZK Modes

1. **Off** (`zk_mode: 'off'`): No ZK proofs. Pure MPC attestation with SPDZ MACs.
2. **Always** (`zk_mode: 'always'`): ZK proof at every checkpoint. Same frequency as `checkpoint_freq`.
3. **Risk-Based** (`zk_mode: 'risk'`): ZK proofs auto-activate when risk is detected, then remain active for all subsequent checkpoints.

## Weight Privacy Constraint

Reconstructing weights from MPC shares for ZK witness generation exposes full weights. Only the trusted operator/aggregator node performs reconstruction and proving. The ZK proof's public inputs contain only hashes (old_hash_lo, old_hash_hi, new_hash_lo, new_hash_hi, delta_hash, error_bound) — never raw weights.

## Architecture

### Layer 1: MPC Integration (`helix-mpc/src/e2e_integration.rs`)

Add to `MPCIntegrationConfig`:
```rust
pub capture_checkpoint_weights: bool,  // gate weight capture for ZK
```

Add to `CheckpointRecord`:
```rust
pub weight_snapshot: Option<Vec<Fr>>,  // reconstructed weights at checkpoint time
```

When `capture_checkpoint_weights` is true and the node is trusted, reconstruct weights at each checkpoint while trainers are alive and store in the snapshot. ~800KB per checkpoint for MNIST (25K params * 32 bytes).

### Layer 2: Risk Assessment (`helix-demo/src/risk.rs`)

New module evaluating risk conditions:
- **Cheater detected**: Any MAC verification failure triggers risk
- **Worker dropout**: Active workers < `min_workers_for_mpc`
- **Recovery event**: Cheater removal + recovery = elevated risk

`RiskAssessor` tracks state and returns `is_zk_required()`. Once risk is triggered, it stays active (matches on-chain `zkActivatedByRisk` which is irreversible).

Risk assessment runs post-training on the `MPCIntegrationResult`:
- If `cheater_detected.is_some()` → risk triggered
- If recovery reduced worker count below threshold → risk triggered

### Layer 3: Demo Runner (`helix-demo/src/runner.rs`)

After `run_mpc_training()` returns, iterate checkpoints:

```
for each checkpoint in result.checkpoints:
    match zk_mode:
        Off → skip
        Always → generate proof
        Risk → generate proof if risk_assessor.is_zk_required()
```

Refactor `LazyZkProver::generate_proof()` to accept `(prev_weights: &[Fr], current_weights: &[Fr], error_bound: Fr)` instead of live trainers. The weight data comes from `CheckpointRecord::weight_snapshot`.

Emit events via the `on_step` callback for WebSocket forwarding:
- `zk_proof_started` when risk triggers ZK activation
- `zk_proof_generated` after each proof is created

### Layer 4: On-Chain Settlement (`helix-demo/src/chain.rs`)

When a checkpoint has a ZK proof:
- Call `submitCheckpointWithProof(jobId, step, commitment, loss, proof, publicInputs)` instead of `submitCheckpoint`
- The 6 public inputs from `StateTransitionWitness::public_inputs()` go directly to the contract
- `Halo2VerifierCore` validates the BN254 pairing; invalid proofs revert

### Layer 5: Dashboard

**contracts.ts**: Add `submitCheckpointWithProof` and `CheckpointWithProofSubmitted` event to `HELIX_COORDINATOR_V4_ABI`.

**train/page.tsx**:
- Remove separate `zkCheckpointFreq` input — ZK frequency = checkpoint frequency
- Show "ZK Activated by Risk" indicator when `session.zk_activated_by_risk` flips true mid-training

**useMpcTraining.ts**: Already handles `zk_proof_started` (sets `zk_activated_by_risk: true`) and `zk_proof_generated` (increments counter). No changes needed — just needs the backend to emit these events.

## Simplifications

- Remove `zk_checkpoint_freq` from `registerTrainingJob` args (pass 1, meaning every checkpoint)
- `RLCAggregationVerifier` is out of scope — it's for a future aggregation circuit, not the checkpoint flow
- V2/V3 coordinator paths are legacy; only V4 is active

## Data Flow (ZK Always mode)

```
run_mpc_training(capture_checkpoint_weights: true)
  → at each checkpoint, reconstruct weights from shares (trusted node only)
  → store Vec<Fr> in CheckpointRecord::weight_snapshot

runner.rs post-training loop:
  → for each checkpoint with weight_snapshot:
      build StateTransitionWitness(prev_weights, current_weights, error_bound)
      CheckpointProver::prove(witness) → proof_bytes + public_inputs
      emit zk_proof_generated event

chain.rs on-chain settlement:
  → submitCheckpointWithProof(jobId, step, commitment, loss, proof_bytes, public_inputs)
  → Halo2VerifierCore validates BN254 pairing
  → contract stores zkWeightHash for chain continuity
```

## Data Flow (Risk mode)

Same as above, but proof generation only starts after `RiskAssessor::is_zk_required()` returns true. All checkpoints before the risk trigger use plain `submitCheckpoint`. All checkpoints after use `submitCheckpointWithProof`.
