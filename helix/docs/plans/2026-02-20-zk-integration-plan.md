# ZK Proof End-to-End Integration Plan

**Goal:** Wire real Halo2 ZK proofs end-to-end through MPC training, on-chain settlement, and dashboard for all three modes (off, always, risk-based).

**Architecture:** Capture weight shares at checkpoint time in the MPC integration layer, reconstruct in the trusted operator context after training, generate StateTransitionCircuit proofs, and submit via `submitCheckpointWithProof` on-chain. Risk assessment evaluates post-training results to decide which checkpoints need proofs. ZK checkpoint frequency equals the regular checkpoint frequency (one knob).

**Tech Stack:** Rust (halo2 PSE, BN254), Solidity (Halo2VerifierCore), TypeScript/Next.js (dashboard)

---

### Task 1: Add checkpoint weight capture to MPC integration

**Files:**
- Modify: `helix/crates/helix-mpc/src/e2e_integration.rs`

**Step 1: Add `capture_checkpoint_weights` to `MPCIntegrationConfig`**

At line 82 (after `pub batch_size: usize,`), add:

```rust
    /// Whether to capture weight snapshots at checkpoints for ZK proof generation.
    /// Only the trusted operator should enable this (weights are reconstructed from shares).
    pub capture_checkpoint_weights: bool,
```

In `Default for MPCIntegrationConfig` (line 152, after `batch_size: 1,`), add:

```rust
            capture_checkpoint_weights: false,
```

In the `Debug` impl (line 113, after `.field("batch_size", ...)`), add:

```rust
            .field("capture_checkpoint_weights", &self.capture_checkpoint_weights)
```

**Step 2: Add `checkpoint_weight_snapshots` to `PartyResult`**

At line 1095 (after `pub checkpoint_step: u64,`), add:

```rust
    /// Per-checkpoint weight share snapshots (step, flat_shares) for ZK proof generation.
    /// Each entry contains THIS party's additive shares — must be summed across parties
    /// to reconstruct full weights. Only populated when `capture_checkpoint_weights` is true.
    pub checkpoint_weight_snapshots: Vec<(usize, Vec<Fr>)>,
```

**Step 3: Capture weight shares at checkpoint time in `run_party_training`**

The `run_party_training` function already computes `all_weights` (per-party shares) at line 1302-1308 during checkpoint creation. We need to save these.

Before the checkpoint block (around line 1280, near `steps_completed += 1;`), add a local `Vec<(usize, Vec<Fr>)>`:

At the start of `run_party_training` (near the `let mut checkpoints = Vec::new();` line), add:

```rust
    let mut weight_snapshots: Vec<(usize, Vec<Fr>)> = Vec::new();
```

Inside the checkpoint block, after `let all_weights: Vec<Fr> = ...` (line 1308) and before `let blindings = ...` (line 1310), add:

```rust
            if capture_checkpoint_weights {
                weight_snapshots.push((step + 1, all_weights.clone()));
            }
```

The `capture_checkpoint_weights` bool needs to be plumbed from the config. It's already available via the config passed to `run_mpc_training`.

In the `PartyResult` construction at the end of `run_party_training`, add:

```rust
        checkpoint_weight_snapshots: weight_snapshots,
```

**Step 4: Add `weight_snapshot` to `CheckpointRecord` and reconstruct in `collect_results`**

Add to `CheckpointRecord` struct (after `pub loss: f64,` at line 211):

```rust
    /// Reconstructed weights at this checkpoint (trusted operator only).
    /// Sum of all parties' additive shares. Used for ZK proof witness generation.
    pub weight_snapshot: Option<Vec<Fr>>,
```

In `collect_results` (lines 1726-1734), replace the checkpoint mapping with reconstruction:

```rust
    // Build checkpoint weight snapshots by summing per-party shares.
    let has_weight_snapshots = party_results[0]
        .checkpoint_weight_snapshots
        .len() == party_results[0].checkpoints.len();

    let checkpoints: Vec<CheckpointRecord> = party_results[0]
        .checkpoints
        .iter()
        .enumerate()
        .map(|(i, cp)| {
            let weight_snapshot = if has_weight_snapshots {
                // Sum additive shares across all parties at this checkpoint.
                let num_weights = party_results[0].checkpoint_weight_snapshots[i].1.len();
                let mut summed = vec![Fr::ZERO; num_weights];
                for pr in &party_results {
                    if i < pr.checkpoint_weight_snapshots.len() {
                        for (j, share) in pr.checkpoint_weight_snapshots[i].1.iter().enumerate() {
                            summed[j] = Fr::add(&summed[j], share);
                        }
                    }
                }
                Some(summed)
            } else {
                None
            };
            CheckpointRecord {
                step: cp.step,
                commitment_bytes32: cp.commitment_bytes32,
                loss: cp.loss,
                weight_snapshot,
            }
        })
        .collect();
```

**Step 5: Update the cheater path's `CheckpointRecord` construction**

In `run_with_cheater` (around line 1042), update the fallback checkpoint construction to add `weight_snapshot: None`. Same for the `checkpoints: Vec::new()` path at line 997.

**Step 6: Run tests to verify**

Run: `cargo test -p helix-mpc -- --test-threads=1 test_basic_integration`
Expected: PASS (existing tests shouldn't break since `capture_checkpoint_weights` defaults to false)

**Step 7: Commit**

```bash
git add helix/crates/helix-mpc/src/e2e_integration.rs
git commit -m "feat(mpc): add checkpoint weight capture for ZK proof generation"
```

---

### Task 2: Refactor LazyZkProver to accept weight vectors

**Files:**
- Modify: `helix/crates/helix-demo/src/zk_prover.rs`

**Step 1: Change `generate_proof` to accept `Vec<Fr>` instead of live trainers**

Replace the `generate_proof` method signature and body. The new version accepts pre-reconstructed Halo2Fr weights instead of live MPCTrainer handles:

```rust
    /// Generate a ZK proof for a checkpoint transition.
    ///
    /// `current_weights` must be pre-reconstructed Halo2Fr field elements
    /// (sum of all parties' additive shares). Only the trusted operator
    /// should call this — weight privacy is maintained because the proof's
    /// public inputs contain only hashes, never raw weights.
    pub fn generate_proof(
        &mut self,
        step: u64,
        current_weights: Vec<Halo2Fr>,
        error_bound_f64: f64,
    ) -> Result<Option<ZkCheckpointResult>> {
        let total_start = Instant::now();

        // First checkpoint: store initial weights, no proof to generate yet
        if self.prev_weights.is_none() {
            self.prev_weights = Some(current_weights);
            return Ok(None);
        }

        // Lazy-init the prover (SRS + keygen)
        self.ensure_initialized()?;

        let prev = self.prev_weights.as_ref().unwrap();

        let error_bound = {
            let scaled = (error_bound_f64.abs() * 1e9) as u64;
            Halo2Fr::from(scaled)
        };

        let witness = StateTransitionWitness::new(
            prev.clone(),
            current_weights.clone(),
            error_bound,
        );

        display::zk_proof_start(step);

        let prover = self.prover.as_ref().unwrap();
        match prover.prove(&witness) {
            Ok(proof_result) => {
                let total_time_ms = total_start.elapsed().as_millis() as u64;
                display::zk_proof_success(
                    step,
                    proof_result.proof_size(),
                    proof_result.generation_time.as_millis() as u64,
                    proof_result.verified,
                );
                let result = ZkCheckpointResult {
                    proof: proof_result,
                    step,
                    total_time_ms,
                };
                self.prev_weights = Some(current_weights);
                self.results.push(result);
                Ok(self.results.last().cloned())
            }
            Err(e) => {
                display::zk_proof_failed(step, &e.to_string());
                self.prev_weights = Some(current_weights);
                Err(anyhow::anyhow!("ZK proof generation failed: {}", e))
            }
        }
    }
```

**Step 2: Remove `record_initial_weights` and `reconstruct_weights_halo2`**

Delete the `record_initial_weights` method (lines 174-180) and the standalone `reconstruct_weights_halo2` function (lines 208-246). Also remove the unused imports for `MPCTrainer`, `LocalTransport`, and `MpcFr`.

**Step 3: Derive `Clone` on `ZkCheckpointResult`**

The struct needs `Clone` since we return `self.results.last().cloned()`. Add `Clone` to the derive and ensure `CheckpointProofResult` supports `Clone` (it already does per the prover code).

```rust
#[derive(Clone)]
pub struct ZkCheckpointResult {
```

**Step 4: Verify compilation**

Run: `cargo check -p helix-demo`
Expected: PASS (with warnings about unused `zk_prover` module since runner doesn't call it yet)

**Step 5: Commit**

```bash
git add helix/crates/helix-demo/src/zk_prover.rs
git commit -m "refactor(demo): LazyZkProver accepts weight vectors instead of live trainers"
```

---

### Task 3: Add ZK mode CLI args and risk assessment

**Files:**
- Modify: `helix/crates/helix-demo/src/main.rs`
- Create: `helix/crates/helix-demo/src/risk.rs`
- Modify: `helix/crates/helix-demo/src/main.rs` (add `mod risk;`)

**Step 1: Replace `--zk-proofs` and `--zk-checkpoint-freq` with `--zk-mode`**

In `main.rs`, replace lines 55-61 (the `zk_proofs` and `zk_checkpoint_freq` fields) with:

```rust
    /// ZK proof mode: off (no ZK), always (every checkpoint), risk (auto-activate on threat).
    #[arg(long, default_value = "off", value_parser = parse_zk_mode)]
    pub zk_mode: ZkMode,

    /// Minimum worker count before risk-based ZK activates (only for --zk-mode risk).
    #[arg(long, default_value = "2")]
    pub min_workers_for_mpc: usize,
```

Add the `ZkMode` enum and parser before the `Args` struct:

```rust
/// ZK proof generation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZkMode {
    /// No ZK proofs.
    Off,
    /// ZK proof at every checkpoint.
    Always,
    /// ZK proofs activate automatically when risk is detected.
    Risk,
}

fn parse_zk_mode(s: &str) -> Result<ZkMode, String> {
    match s.to_lowercase().as_str() {
        "off" => Ok(ZkMode::Off),
        "always" => Ok(ZkMode::Always),
        "risk" => Ok(ZkMode::Risk),
        _ => Err(format!("invalid ZK mode '{}': expected off, always, or risk", s)),
    }
}
```

Update the `main()` ZK banner (lines 114-119) to use the new enum:

```rust
    match args.zk_mode {
        ZkMode::Off => {}
        ZkMode::Always => {
            display::info("ZK proofs ENABLED: StateTransitionCircuit proof at every checkpoint");
            println!();
        }
        ZkMode::Risk => {
            display::info(&format!(
                "ZK proofs in RISK mode: auto-activate when workers < {} or cheater detected",
                args.min_workers_for_mpc,
            ));
            println!();
        }
    }
```

**Step 2: Create `risk.rs` module**

Create `helix/crates/helix-demo/src/risk.rs`:

```rust
//! Risk assessment for automatic ZK proof activation.
//!
//! Evaluates conditions from an MPC training result to determine whether
//! ZK proofs should be generated. Once risk is triggered, it stays active
//! (matching on-chain `zkActivatedByRisk` which is irreversible).

use helix_mpc::e2e_integration::MPCIntegrationResult;

/// Evaluates risk conditions and determines which checkpoints need ZK proofs.
pub struct RiskAssessor {
    min_workers: usize,
    /// Step at which risk was first detected (None = no risk).
    risk_triggered_at: Option<usize>,
}

impl RiskAssessor {
    pub fn new(min_workers_for_mpc: usize) -> Self {
        Self {
            min_workers: min_workers_for_mpc,
            risk_triggered_at: None,
        }
    }

    /// Evaluate the training result and determine the risk trigger point.
    ///
    /// Returns the step number at which risk was first detected, or None if
    /// training completed without risk conditions.
    pub fn evaluate(&mut self, result: &MPCIntegrationResult) -> Option<usize> {
        // Condition 1: Cheater detected during training
        if let Some(ref cheater) = result.cheater_detected {
            let step = cheater.detected_at_step as usize;
            self.risk_triggered_at = Some(step);
            return Some(step);
        }

        // Condition 2: Recovery occurred (implies worker loss)
        if result.recovery_completed {
            // Risk from step 0 — recovery means the network was compromised
            self.risk_triggered_at = Some(0);
            return Some(0);
        }

        // Condition 3: Worker count dropped below threshold
        // In the demo, worker count is static (no dynamic joins/leaves),
        // so this check is based on the initial config.
        // In production, this would check on-chain active worker counts.

        None
    }

    /// Whether a given checkpoint step requires a ZK proof.
    pub fn needs_proof(&self, checkpoint_step: usize) -> bool {
        match self.risk_triggered_at {
            Some(trigger_step) => checkpoint_step >= trigger_step,
            None => false,
        }
    }

    /// Whether risk was triggered at all.
    pub fn is_triggered(&self) -> bool {
        self.risk_triggered_at.is_some()
    }

    /// The step at which risk was triggered.
    pub fn trigger_step(&self) -> Option<usize> {
        self.risk_triggered_at
    }
}
```

**Step 3: Register the module**

In `main.rs`, add `mod risk;` alongside the other module declarations (near `mod zk_prover;` at line 18).

**Step 4: Verify compilation**

Run: `cargo check -p helix-demo`
Expected: PASS

**Step 5: Commit**

```bash
git add helix/crates/helix-demo/src/main.rs helix/crates/helix-demo/src/risk.rs
git commit -m "feat(demo): add ZkMode enum and RiskAssessor for ZK proof activation"
```

---

### Task 4: Wire ZK proof generation into runner.rs

**Files:**
- Modify: `helix/crates/helix-demo/src/runner.rs`

This is the critical wiring task. After `run_mpc_training()` returns, iterate checkpoints and generate proofs based on ZK mode.

**Step 1: Add imports**

At the top of `runner.rs`, add:

```rust
use crate::risk::RiskAssessor;
use crate::zk_prover::LazyZkProver;
use crate::ZkMode;
```

Also add (for converting MPC Fr to Halo2Fr):

```rust
use helix_prover::halo2curves::bn256::Fr as Halo2Fr;
```

**Step 2: Enable weight capture when ZK is enabled**

In the `run()` method, where `MPCIntegrationConfig` is built (find the config construction), add:

```rust
        capture_checkpoint_weights: self.args.zk_mode != ZkMode::Off,
```

**Step 3: Add ZK proof generation phase after training**

After the checkpoint display loop (line ~212) and before Phase 5 (on-chain settlement), add a new phase:

```rust
    // ================================================================
    // Phase 4b: ZK Proof Generation (optional)
    // ================================================================
    let mut zk_proofs: Vec<crate::zk_prover::ZkCheckpointResult> = Vec::new();

    if self.args.zk_mode != ZkMode::Off && !result.checkpoints.is_empty() {
        let has_snapshots = result.checkpoints.iter().all(|cp| cp.weight_snapshot.is_some());

        if has_snapshots {
            display::phase("Phase 4b", "ZK Proof Generation (StateTransitionCircuit)");

            // Determine which checkpoints need proofs
            let prove_checkpoint: Vec<bool> = match self.args.zk_mode {
                ZkMode::Off => vec![false; result.checkpoints.len()],
                ZkMode::Always => vec![true; result.checkpoints.len()],
                ZkMode::Risk => {
                    let mut assessor = RiskAssessor::new(self.args.min_workers_for_mpc);
                    assessor.evaluate(&result);
                    if assessor.is_triggered() {
                        display::info(&format!(
                            "Risk detected at step {} — activating ZK proofs for subsequent checkpoints",
                            assessor.trigger_step().unwrap_or(0),
                        ));
                    } else {
                        display::info("No risk conditions detected — skipping ZK proofs");
                    }
                    result.checkpoints.iter()
                        .map(|cp| assessor.needs_proof(cp.step))
                        .collect()
                }
            };

            let mut prover = LazyZkProver::new(
                self.args.d_in(),
                self.args.d_hid(),
                self.args.d_out(),
            );

            for (i, cp) in result.checkpoints.iter().enumerate() {
                if !prove_checkpoint[i] {
                    continue;
                }

                let weights_halo2: Vec<Halo2Fr> = cp.weight_snapshot.as_ref().unwrap()
                    .iter()
                    .map(|mpc_fr| *mpc_fr.inner())
                    .collect();

                let error_bound = 0.001; // Conservative error bound for state transition

                match prover.generate_proof(cp.step as u64, weights_halo2, error_bound) {
                    Ok(Some(proof_result)) => {
                        zk_proofs.push(proof_result);
                    }
                    Ok(None) => {
                        // First checkpoint — stored as initial state, no proof
                        display::info(&format!(
                            "  Step {}: stored as initial weight state (no proof needed)",
                            cp.step,
                        ));
                    }
                    Err(e) => {
                        display::warning(&format!(
                            "  Step {}: ZK proof failed: {}",
                            cp.step, e,
                        ));
                    }
                }
            }

            display::info(&format!(
                "ZK proof generation complete: {} proofs generated, {} verified",
                prover.proofs_generated(),
                prover.proofs_verified(),
            ));
        }
    }
```

**Step 4: Add helper methods to Args for model dimensions**

In `main.rs`, add these helper methods to the `Args` impl (or add an `impl Args` block):

```rust
impl Args {
    /// Input dimension (MNIST = 784).
    pub fn d_in(&self) -> usize { 784 }
    /// Hidden dimension.
    pub fn d_hid(&self) -> usize { 32 }
    /// Output dimension (MNIST = 10 classes).
    pub fn d_out(&self) -> usize { 10 }
}
```

These match the MNIST 784→32→10 architecture used in the demo.

**Step 5: Update the summary with real ZK stats**

Replace the hardcoded zeros in the `display::summary` call (lines 341-343):

```rust
    zk_proofs_generated: zk_proofs.len(),
    zk_proofs_verified: zk_proofs.iter().filter(|p| p.proof.verified).count(),
    zk_total_proving_time_ms: zk_proofs.iter().map(|p| p.total_time_ms).sum(),
```

**Step 6: Verify compilation**

Run: `cargo check -p helix-demo`
Expected: PASS (may have warnings about unused `zk_proofs` in the chain settlement path — addressed in Task 5)

**Step 7: Commit**

```bash
git add helix/crates/helix-demo/src/runner.rs helix/crates/helix-demo/src/main.rs
git commit -m "feat(demo): wire ZK proof generation into runner for all three modes"
```

---

### Task 5: Wire ZK proofs into on-chain settlement

**Files:**
- Modify: `helix/crates/helix-demo/src/runner.rs` (the `run_onchain_settlement` method)
- Modify: `helix/crates/helix-client/src/checkpoint_submitter.rs`

**Step 1: Add proof data to `CheckpointData`**

In `helix/crates/helix-client/src/checkpoint_submitter.rs`, add optional proof fields to `CheckpointData`:

```rust
#[derive(Debug, Clone)]
pub struct CheckpointData {
    pub step: u64,
    pub commitment_bytes32: [u8; 32],
    pub loss: f64,
    /// Optional ZK proof bytes (serialized Halo2 SHPLONK proof).
    pub proof: Option<Vec<u8>>,
    /// Optional public inputs for the ZK proof (6 elements for StateTransitionCircuit).
    pub public_inputs: Option<Vec<ethers::types::U256>>,
}
```

**Step 2: Update `sign_and_submit_checkpoint` to use proof when available**

In `sign_and_submit_checkpoint`, after the existing `chain_client.submit_checkpoint(...)` call, add a branch that calls `submit_checkpoint_with_proof` instead when proof data is present:

```rust
    let receipt = if let (Some(proof), Some(pub_inputs)) = (&checkpoint.proof, &checkpoint.public_inputs) {
        // Submit with ZK proof — no signatures needed, proof is the attestation
        chain_client
            .submit_checkpoint_with_proof(
                job_id,
                checkpoint.step,
                checkpoint.commitment_bytes32,
                loss_u256,
                proof.clone(),
                pub_inputs.clone(),
            )
            .await?
    } else {
        // Submit with multi-party signatures
        let signatures = sign_checkpoint(checkpoint, job_id, worker_wallets)?;
        chain_client
            .submit_checkpoint(job_id, checkpoint.step, checkpoint.commitment_bytes32, loss_u256, signatures)
            .await?
    };
```

**Step 3: Pass ZK proofs into `CheckpointData` in runner's `run_onchain_settlement`**

In `runner.rs`, where `checkpoint_data` is built from `result.checkpoints` (around line 456-464), incorporate the generated proofs:

```rust
    let checkpoint_data: Vec<CheckpointData> = result
        .checkpoints
        .iter()
        .map(|cp| {
            // Find matching ZK proof for this checkpoint
            let zk_proof = zk_proofs.iter().find(|p| p.step == cp.step as u64);
            let (proof_bytes, pub_inputs) = if let Some(zk) = zk_proof {
                let pub_inputs_u256: Vec<U256> = zk.proof.public_inputs.iter()
                    .map(|fr| {
                        let bytes = fr.to_repr();
                        U256::from_little_endian(&bytes)
                    })
                    .collect();
                (Some(zk.proof.proof_bytes.clone()), Some(pub_inputs_u256))
            } else {
                (None, None)
            };
            CheckpointData {
                step: cp.step as u64,
                commitment_bytes32: cp.commitment_bytes32,
                loss: cp.loss,
                proof: proof_bytes,
                public_inputs: pub_inputs,
            }
        })
        .collect();
```

This requires `zk_proofs` to be accessible in `run_onchain_settlement`. Change the method signature to accept it:

```rust
async fn run_onchain_settlement(
    &self,
    result: &MPCIntegrationResult,
    zk_proofs: &[crate::zk_prover::ZkCheckpointResult],
) -> Result<ChainStats> {
```

And update the call site (Phase 5) to pass `&zk_proofs`.

**Step 4: Register job with ZK params when ZK mode is enabled**

In `run_onchain_settlement`, replace the `register_training_job` call with `register_training_job_with_zk` when ZK is enabled:

```rust
    let (reg_receipt, job_id) = match self.args.zk_mode {
        ZkMode::Off => {
            owner_client.register_training_job(
                architecture_hash, self.args.checkpoint_freq,
                self.args.steps as u64, payment,
            ).await?
        }
        ZkMode::Always => {
            owner_client.register_training_job_with_zk(
                architecture_hash, self.args.checkpoint_freq,
                self.args.steps as u64, payment,
                true,  // zkEnabled
                1,     // zkCheckpointFreq = every checkpoint
                false, // riskZkEnabled
                0,     // minWorkersForMpc (not used in always mode)
                Address::zero(),
            ).await?
        }
        ZkMode::Risk => {
            owner_client.register_training_job_with_zk(
                architecture_hash, self.args.checkpoint_freq,
                self.args.steps as u64, payment,
                false, // zkEnabled
                1,     // zkCheckpointFreq = every checkpoint
                true,  // riskZkEnabled
                self.args.min_workers_for_mpc as u64,
                Address::zero(),
            ).await?
        }
    };
```

**Step 5: Deploy real Halo2Verifier when ZK is enabled**

In `run_onchain_settlement`, before the `ChainClientV4::deploy` call, deploy the Halo2Verifier and pass its address to the V4 coordinator when ZK is on:

```rust
    let verifier_address = if self.args.zk_mode != ZkMode::Off {
        display::subphase("Deploying Halo2Verifier (real BN254 pairing verifier)...");
        let addr = owner_client.deploy_halo2_verifier().await?;
        owner_client.set_verifier(addr).await?;
        display::info(&format!("  Halo2Verifier deployed at {:#x}", addr));
        Some(addr)
    } else {
        None
    };
```

Note: `deploy_halo2_verifier` and `set_verifier` already exist on `ChainClientV4`. This deploys AFTER the coordinator, so the coordinator is already up. Then `set_verifier` points the coordinator at the real verifier.

**Step 6: Verify compilation**

Run: `cargo check -p helix-demo --features chain`
Expected: PASS

**Step 7: Commit**

```bash
git add helix/crates/helix-demo/src/runner.rs helix/crates/helix-client/src/checkpoint_submitter.rs
git commit -m "feat(demo): submit ZK proofs on-chain via submitCheckpointWithProof"
```

---

### Task 6: Dashboard — add submitCheckpointWithProof to V4 ABI

**Files:**
- Modify: `helix/dashboard/src/lib/contracts.ts`

**Step 1: Add `submitCheckpointWithProof` and related items to `HELIX_COORDINATOR_V4_ABI`**

After the existing `JobRegistered` event entry (line ~867), add:

```typescript
    {
        name: 'submitCheckpointWithProof',
        type: 'function',
        stateMutability: 'nonpayable',
        inputs: [
            { name: 'jobId', type: 'uint256' },
            { name: 'stepNumber', type: 'uint256' },
            { name: 'weightCommitment', type: 'bytes32' },
            { name: 'loss', type: 'uint256' },
            { name: 'proof', type: 'bytes' },
            { name: 'publicInputs', type: 'uint256[]' },
        ],
        outputs: [],
    },
    {
        name: 'isZkRequired',
        type: 'function',
        stateMutability: 'view',
        inputs: [{ name: 'jobId', type: 'uint256' }],
        outputs: [{ name: '', type: 'bool' }],
    },
    {
        name: 'ZkActivatedByRisk',
        type: 'event',
        inputs: [
            { name: 'jobId', type: 'uint256', indexed: true },
            { name: 'activeWorkerCount', type: 'uint256', indexed: false },
        ],
    },
    {
        name: 'CheckpointWithProofSubmitted',
        type: 'event',
        inputs: [
            { name: 'jobId', type: 'uint256', indexed: true },
            { name: 'stepNumber', type: 'uint256', indexed: false },
            { name: 'weightCommitment', type: 'bytes32', indexed: false },
        ],
    },
```

**Step 2: Verify build**

Run: `cd helix/dashboard && npm run build`
Expected: PASS (ABI additions are additive, no breaking changes)

**Step 3: Commit**

```bash
git add helix/dashboard/src/lib/contracts.ts
git commit -m "feat(dashboard): add submitCheckpointWithProof to V4 ABI"
```

---

### Task 7: Dashboard — simplify ZK UI (remove separate ZK checkpoint freq)

**Files:**
- Modify: `helix/dashboard/src/app/train/page.tsx`

**Step 1: Remove `zkCheckpointFreq` state variable**

Delete line 723: `const [zkCheckpointFreq, setZkCheckpointFreq] = useState(5);`

**Step 2: Remove the ZK Checkpoint Freq NumberInput**

In the ZK Mode UI block (around line 1217), remove the `AnimatePresence` child for `zkMode === 'always'` that shows the "ZK Checkpoint Freq" NumberInput. Keep the `zkMode === 'risk'` child that shows "Min Workers for MPC".

Replace with a simple info text:

```tsx
    {zkMode === 'always' && (
      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}>
        <p className="text-sm text-helix-muted px-1">
          ZK proof generated at every checkpoint (frequency = checkpoint interval)
        </p>
      </motion.div>
    )}
```

**Step 3: Update `registerTrainingJob` call**

In the `registerTrainingJob` args (around line 2905), change:

```typescript
BigInt(config.zk_checkpoint_freq),
```

to:

```typescript
BigInt(1), // ZK at every checkpoint (frequency = checkpoint interval)
```

**Step 4: Remove `zk_checkpoint_freq` from the config object passed to `onStart`**

In `handleSubmit` (around line 798), remove the `zk_checkpoint_freq` line and replace with hardcoded 1:

```typescript
zk_checkpoint_freq: 1,
```

**Step 5: Add "ZK Activated" indicator for risk mode**

In the stats grid (around line 2454), add a conditional indicator when risk-based ZK activates:

```tsx
{session.zk_activated_by_risk && (
  <div className="col-span-2 flex items-center gap-2 px-3 py-2 rounded-xl bg-amber-500/10 border border-amber-500/20">
    <Shield size={14} className="text-amber-400" />
    <span className="text-sm text-amber-300 font-medium">ZK Activated by Risk</span>
  </div>
)}
```

**Step 6: Verify build**

Run: `cd helix/dashboard && npm run build`
Expected: PASS

**Step 7: Commit**

```bash
git add helix/dashboard/src/app/train/page.tsx
git commit -m "feat(dashboard): simplify ZK UI — single checkpoint frequency, risk indicator"
```

---

### Task 8: End-to-end verification

**Step 1: Run the demo in "always" mode**

```bash
cd helix && cargo run -p helix-demo -- --steps 20 --checkpoint-freq 10 --zk-mode always --skip-chain
```

Expected output should show:
- Phase 4b: ZK Proof Generation
- "Initializing ZK prover (StateTransitionCircuit, k=12)..."
- Proof generation for each checkpoint (with proof size and generation time)
- Summary showing non-zero `zk_proofs_generated` and `zk_proofs_verified`

**Step 2: Run the demo in "risk" mode with cheater**

```bash
cd helix && cargo run -p helix-demo -- --steps 20 --checkpoint-freq 5 --zk-mode risk --simulate-cheater --skip-chain
```

Expected: ZK proofs generated only for checkpoints AFTER the cheater was detected.

**Step 3: Run with on-chain settlement (real verifier)**

```bash
cd helix && cargo run -p helix-demo -- --steps 20 --checkpoint-freq 10 --zk-mode always
```

Expected:
- Halo2Verifier deployed
- `submitCheckpointWithProof` transactions succeed (real BN254 pairing verification)
- No `InvalidProof` reverts

**Step 4: Run existing tests**

```bash
cargo test -p helix-mpc -- --test-threads=1
cargo test -p helix-demo
cargo test -p helix-client
```

Expected: All existing tests pass (no regressions).

**Step 5: Final commit**

If any fixes were needed during verification, commit them.

---

### Summary of changes by file

| File | Change |
|------|--------|
| `helix-mpc/src/e2e_integration.rs` | Add `capture_checkpoint_weights` config, `checkpoint_weight_snapshots` to `PartyResult`, `weight_snapshot` to `CheckpointRecord`, reconstruct in `collect_results` |
| `helix-demo/src/zk_prover.rs` | Refactor `generate_proof` to accept `Vec<Halo2Fr>` instead of `&[MPCTrainer]` |
| `helix-demo/src/main.rs` | Replace `--zk-proofs`/`--zk-checkpoint-freq` with `--zk-mode` enum, add `mod risk`, add `Args` dimension helpers |
| `helix-demo/src/risk.rs` | New: `RiskAssessor` evaluating cheater detection and worker dropout |
| `helix-demo/src/runner.rs` | Wire proof generation after training, pass proofs to on-chain settlement, register jobs with ZK params, deploy real Halo2Verifier |
| `helix-client/src/checkpoint_submitter.rs` | Add optional `proof`/`public_inputs` to `CheckpointData`, branch to `submit_checkpoint_with_proof` |
| `dashboard/src/lib/contracts.ts` | Add `submitCheckpointWithProof`, `isZkRequired`, events to V4 ABI |
| `dashboard/src/app/train/page.tsx` | Remove ZK checkpoint freq input, hardcode freq=1, add risk activation indicator |
