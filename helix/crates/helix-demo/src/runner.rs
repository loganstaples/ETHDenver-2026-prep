//! Core demo orchestration for HELIX MPC training.
//!
//! The `DemoRunner` coordinates the full demo lifecycle:
//! 1. MNIST dataset generation
//! 2. MPC training via `run_mpc_training` (SPDZ MACs, Beaver triples, Pedersen checkpoints)
//! 3. Weight reconstruction and accuracy evaluation
//! 4. Native baseline comparison
//! 5. On-chain checkpoint settlement (optional, via `--on-chain`)
//! 6. Summary display

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use helix_mpc::e2e_integration::{
    MPCIntegrationConfig, MPCIntegrationResult, TrainingOutcome, TrainingSignal,
    run_mpc_training, run_mpc_training_with_cheater,
};
use helix_mpc::mnist::{MnistDataset, NativeTrainer};

#[cfg(feature = "chain")]
use ethers::signers::{LocalWallet, Signer};
#[cfg(feature = "chain")]
use ethers::types::{Address, U256};
#[cfg(feature = "chain")]
use ethers::utils::Anvil;
#[cfg(feature = "chain")]
use helix_client::checkpoint_submitter::{submit_all_checkpoints, CheckpointData};
#[cfg(feature = "chain")]
use helix_client::rpc::chain_v4::{sign_completion, ChainClientV4};

use crate::display::{self, ChainStats};
use crate::evaluator;
use crate::risk::RiskAssessor;
use crate::zk_prover::LazyZkProver;
use crate::Args;
use crate::ZkMode;

use helix_prover::halo2curves::bn256::Fr as Halo2Fr;

/// MNIST network dimensions.
const D_IN: usize = 784;
const D_HID: usize = 32;
const D_OUT: usize = 10;

/// Core demo runner.
pub struct DemoRunner {
    args: Args,
    /// Sender for training control signals (pause/stop).
    /// Kept here so a future CLI or API layer can send signals.
    #[allow(dead_code)]
    signal_tx: tokio::sync::watch::Sender<TrainingSignal>,
}

impl DemoRunner {
    /// Creates a new demo runner from CLI arguments.
    pub fn new(args: Args) -> Self {
        let (signal_tx, _signal_rx) = tokio::sync::watch::channel(TrainingSignal::Continue);
        Self { args, signal_tx }
    }

    /// Returns a reference to the signal sender.
    ///
    /// External code (e.g. a CLI command handler) can call
    /// `runner.signal_sender().send(TrainingSignal::Pause)` to pause training.
    #[allow(dead_code)]
    pub fn signal_sender(&self) -> &tokio::sync::watch::Sender<TrainingSignal> {
        &self.signal_tx
    }

    /// Runs the complete HELIX MPC training demo.
    pub async fn run(&self) -> Result<()> {
        let demo_start = Instant::now();

        // ================================================================
        // Banner
        // ================================================================
        display::banner();
        display::info(&format!(
            "Configuration: {} workers, {} steps, lr={}, checkpoint every {} steps",
            self.args.workers, self.args.steps, self.args.lr, self.args.checkpoint_freq,
        ));
        if self.args.simulate_cheater {
            display::warn("Cheater simulation ENABLED: Party 2 will be corrupted mid-training");
        }
        #[cfg(feature = "chain")]
        if self.args.skip_chain {
            display::info("On-chain settlement SKIPPED (--skip-chain)");
        }
        println!();

        // ================================================================
        // Phase 1: Load/Generate MNIST Dataset
        // ================================================================
        let phase1_start = Instant::now();

        let dataset = if self.args.real_mnist {
            display::phase("Phase 1: Real MNIST Dataset Loading");
            let cache_dir = self.args.mnist_cache_dir.as_ref().map(std::path::Path::new);
            let full = MnistDataset::load_real_shuffled(
                self.args.train_size,
                self.args.test_size,
                self.args.seed,
                cache_dir,
            )
            .context("Failed to load real MNIST data")?;
            display::success(&format!(
                "Loaded {} training + {} test real MNIST samples ({:.1}ms)",
                full.train.len(),
                full.test.len(),
                phase1_start.elapsed().as_secs_f64() * 1000.0,
            ));
            full
        } else {
            display::phase("Phase 1: Synthetic MNIST Dataset Generation");
            let ds = MnistDataset::generate(
                self.args.train_size,
                self.args.test_size,
                self.args.seed,
            );
            display::success(&format!(
                "Generated {} training + {} test synthetic samples ({:.1}ms)",
                ds.train.len(),
                ds.test.len(),
                phase1_start.elapsed().as_secs_f64() * 1000.0,
            ));
            ds
        };
        display::info(&format!(
            "Architecture: {} -> {} (ReLU) -> {} (argmax)",
            D_IN, D_HID, D_OUT,
        ));
        let total_params = D_HID * D_IN + D_HID + D_OUT * D_HID + D_OUT;
        display::info(&format!(
            "Total parameters: {} (W1: {}, b1: {}, W2: {}, b2: {})",
            total_params,
            D_HID * D_IN,
            D_HID,
            D_OUT * D_HID,
            D_OUT,
        ));

        let train_pairs = MnistDataset::as_training_pairs(&dataset.train);
        let test_pairs = MnistDataset::as_training_pairs(&dataset.test);

        // ================================================================
        // Phase 2: MPC Training with SPDZ MAC Verification
        // ================================================================
        display::phase("Phase 2: MPC Training with SPDZ MAC Verification");
        display::info(&format!(
            "Setting up {} MPC workers, distributing weights, generating Beaver triples...",
            self.args.workers,
        ));

        let phase2_start = Instant::now();
        let num_workers = self.args.workers;
        let checkpoint_freq = self.args.checkpoint_freq;

        // on_step callback for live display during training.
        // Fires on party 0 after each step: (step_1indexed, total, loss, acc_est, mac_ok).
        let on_step: Arc<dyn Fn(usize, usize, f64, f64, bool) + Send + Sync> = Arc::new(
            move |step: usize, total: usize, loss: f64, _acc_est: f64, mac_ok: bool| {
                display::step_update(step, total, loss, num_workers, mac_ok);
                if checkpoint_freq > 0 && step as u64 % checkpoint_freq == 0 {
                    display::step_update_finish();
                    display::mac_verified(step as u64);
                }
            },
        );

        let signal_rx = self.signal_tx.subscribe();

        let config = MPCIntegrationConfig {
            d_in: D_IN,
            d_hid: D_HID,
            d_out: D_OUT,
            num_workers: self.args.workers,
            num_steps: self.args.steps,
            learning_rate: self.args.lr,
            checkpoint_interval: checkpoint_freq as usize,
            mac_check_interval: checkpoint_freq,
            beaver_batch_size: 10000,
            initial_weights: None,
            training_data: train_pairs.clone(),
            seed: self.args.seed,
            use_node_transport: false,
            use_tcp_transport: false,
            worker_endpoints: None,
            batch_size: 1,
            on_step: Some(on_step),
            on_sub_step: None,
            capture_checkpoint_weights: self.args.zk_mode != ZkMode::Off,
            signal_rx: Some(signal_rx),
            starting_step: 0,
        };

        let result = if self.args.simulate_cheater && self.args.workers > 2 {
            let cheater_step = (self.args.steps / 2) as u64;
            display::info(&format!(
                "Party 2 will be corrupted at step {}",
                cheater_step,
            ));
            run_mpc_training_with_cheater(config, 2, cheater_step)
                .await
                .context("MPC training with cheater failed")?
        } else {
            run_mpc_training(config)
                .await
                .context("MPC training failed")?
        };

        display::step_update_finish();

        // Handle training outcome (pause/stop signals)
        match &result.outcome {
            TrainingOutcome::Paused { step, .. } => {
                display::training_paused(*step);
                let training_time = phase2_start.elapsed();
                display::success(&format!(
                    "Completed {} steps in {:.1}s before pause",
                    result.steps_completed,
                    training_time.as_secs_f64(),
                ));
                display::success(&format!(
                    "MAC checks passed: {} (information-theoretic integrity)",
                    result.mac_checks_passed,
                ));
                return Ok(());
            }
            TrainingOutcome::Stopped { step, .. } => {
                display::training_stopped(*step);
                // Continue to settlement/summary with partial results
            }
            TrainingOutcome::Completed => {
                // Normal flow continues below
            }
            TrainingOutcome::CheaterDetected { .. } => {
                // Handled by existing cheater_detected logic below
            }
        }

        let training_time = phase2_start.elapsed();
        display::success(&format!(
            "Training complete: {} steps in {:.1}s ({:.1} steps/sec)",
            result.steps_completed,
            training_time.as_secs_f64(),
            result.steps_completed as f64 / training_time.as_secs_f64(),
        ));
        display::success(&format!(
            "MAC checks passed: {} (information-theoretic integrity)",
            result.mac_checks_passed,
        ));

        // Display checkpoint commitments
        if !result.checkpoints.is_empty() {
            display::subphase(&format!(
                "{} Pedersen checkpoints computed (joint commitment, no weight reconstruction):",
                result.checkpoints.len(),
            ));
            for cp in &result.checkpoints {
                let commitment_hex = hex::encode(cp.commitment_bytes32);
                display::checkpoint(cp.step as u64, &commitment_hex);
                display::info(&format!("  Loss at checkpoint: {:.6}", cp.loss));
            }
        }

        if let Some(ref cheater) = result.cheater_detected {
            display::cheater_detected(cheater.party_index, cheater.detected_at_step);
            if result.recovery_completed {
                display::success(&format!(
                    "Recovery completed: {} additional steps after cheater removal",
                    result.post_recovery_steps,
                ));
            }
        }

        // ================================================================
        // Phase 4b: ZK Proof Generation (optional)
        // ================================================================
        let mut zk_proofs: Vec<crate::zk_prover::ZkCheckpointResult> = Vec::new();

        if self.args.zk_mode != ZkMode::Off && !result.checkpoints.is_empty() {
            let has_snapshots = result.checkpoints.iter().all(|cp| cp.weight_snapshot.is_some());

            if has_snapshots {
                display::phase("Phase 4b: ZK Proof Generation (StateTransitionCircuit)");

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

                    // Convert MPC Fr to Halo2 Fr. The MPC Fr wraps bn256::Fr, accessed via .inner()
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
                            // First checkpoint — stored as initial weight state
                            display::info(&format!(
                                "  Step {}: stored as initial weight state (no proof needed)",
                                cp.step,
                            ));
                        }
                        Err(e) => {
                            display::warn(&format!(
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

        // ================================================================
        // Phase 3: Weight Reconstruction and Accuracy Evaluation
        // ================================================================
        display::phase("Phase 3: Weight Reconstruction and Evaluation");

        let w = &result.final_weights;
        display::success("Weights reconstructed by summing party shares");

        let w1_norm: f64 = w.w1.iter().map(|v| v * v).sum::<f64>().sqrt();
        let w2_norm: f64 = w.w2.iter().map(|v| v * v).sum::<f64>().sqrt();
        display::metric("||W1|| (Frobenius)", &format!("{:.6}", w1_norm));
        display::metric("||W2|| (Frobenius)", &format!("{:.6}", w2_norm));

        let (overall_accuracy, per_class_acc) = evaluator::evaluate_per_class(
            &w.w1, &w.b1, &w.w2, &w.b2, D_IN, D_HID, D_OUT, &test_pairs,
        );

        display::success(&format!(
            "Test accuracy: {:.1}% ({}/{} correct)",
            overall_accuracy * 100.0,
            (overall_accuracy * test_pairs.len() as f64) as usize,
            test_pairs.len(),
        ));

        display::subphase("Per-class accuracy:");
        for (digit, acc) in per_class_acc.iter().enumerate() {
            let bar_len = (*acc * 20.0) as usize;
            let bar = format!(
                "[{}{}]",
                "#".repeat(bar_len),
                "-".repeat(20 - bar_len),
            );
            println!("    Digit {}: {} {:.1}%", digit, bar, acc * 100.0);
        }

        // ================================================================
        // Phase 4: Native Baseline Comparison
        // ================================================================
        display::phase("Phase 4: Native Baseline Comparison");

        let actual_steps = result.steps_completed;
        let mut native_trainer =
            NativeTrainer::new(D_IN, D_HID, D_OUT, self.args.lr, self.args.seed);
        let native_steps = actual_steps.min(self.args.steps);
        let mut native_losses = Vec::with_capacity(native_steps);

        for step in 0..native_steps {
            let idx = step % train_pairs.len();
            let (ref input, ref target) = train_pairs[idx];
            let r = native_trainer.training_step_mse(input, target, step as u64);
            native_losses.push(r.loss);
        }

        let native_accuracy = native_trainer.evaluate(&test_pairs);
        display::metric(
            "Native (no MPC) accuracy",
            &format!("{:.1}%", native_accuracy * 100.0),
        );
        display::metric(
            "MPC accuracy",
            &format!("{:.1}%", overall_accuracy * 100.0),
        );

        let accuracy_gap = (native_accuracy - overall_accuracy).abs() * 100.0;
        if accuracy_gap < 10.0 {
            display::success(&format!(
                "MPC-native accuracy gap: {:.1}pp (MPC overhead is minimal)",
                accuracy_gap,
            ));
        } else {
            display::info(&format!(
                "MPC-native accuracy gap: {:.1}pp (expected due to fixed-point quantization)",
                accuracy_gap,
            ));
        }

        // ================================================================
        // Phase 5: On-Chain Checkpoint Settlement (optional)
        // ================================================================
        #[cfg(feature = "chain")]
        let chain_stats = if !self.args.skip_chain {
            Some(self.run_onchain_settlement(&result, &zk_proofs).await?)
        } else {
            None
        };
        #[cfg(not(feature = "chain"))]
        let chain_stats: Option<ChainStats> = None;

        // ================================================================
        // Phase 6: Loss Curves and Summary
        // ================================================================
        display::phase("Phase 6: Training Results");

        display::subphase("MPC Training Loss:");
        display::loss_curve(&result.losses);

        display::subphase("Native Training Loss (comparison):");
        display::loss_curve(&native_losses);

        // Estimate triples generated (run_mpc_training doesn't expose this directly)
        let triples_per_step = helix_mpc::mnist::triples_per_step(D_HID, D_OUT);
        let triples_generated = triples_per_step * actual_steps;

        let loss_start = result.losses.first().copied().unwrap_or(0.0);
        let loss_end = result.losses.last().copied().unwrap_or(0.0);

        display::summary(&display::DemoSummary {
            total_time: demo_start.elapsed(),
            num_workers: self.args.workers,
            num_steps: actual_steps,
            loss_start,
            loss_end,
            final_accuracy: overall_accuracy,
            triples_generated,
            mac_checks_passed: result.mac_checks_passed,
            cheater_detected: result.cheater_detected.is_some(),
            cheater_party: result.cheater_detected.as_ref().map(|c| c.party_index),
            zk_proofs_generated: zk_proofs.len(),
            zk_proofs_verified: zk_proofs.iter().filter(|p| p.proof.verified).count(),
            zk_total_proving_time_ms: zk_proofs.iter().map(|p| p.total_time_ms).sum(),
            chain_stats,
        });

        Ok(())
    }

    /// Run on-chain settlement: deploy V4, register job, stake workers, submit checkpoints.
    #[cfg(feature = "chain")]
    async fn run_onchain_settlement(
        &self,
        result: &MPCIntegrationResult,
        zk_proofs: &[crate::zk_prover::ZkCheckpointResult],
    ) -> Result<ChainStats> {
        display::phase("Phase 5: On-Chain Checkpoint Settlement");
        display::info("Booting local Anvil node...");

        let anvil = Anvil::new().spawn();
        let rpc_url = anvil.endpoint();
        let chain_id = anvil.chain_id();

        display::success(&format!(
            "Anvil running at {} (chain_id={})",
            rpc_url, chain_id,
        ));

        // Account 0 = deployer/owner, Accounts 1..N = workers
        let owner_wallet: LocalWallet = anvil.keys()[0].clone().into();
        let owner_wallet = owner_wallet.with_chain_id(chain_id);
        let owner_pk = hex::encode(anvil.keys()[0].to_bytes());

        let mut worker_wallets: Vec<LocalWallet> = Vec::new();
        let mut worker_pks: Vec<String> = Vec::new();
        for i in 1..=self.args.workers {
            let wallet: LocalWallet = anvil.keys()[i].clone().into();
            let wallet = wallet.with_chain_id(chain_id);
            worker_pks.push(hex::encode(anvil.keys()[i].to_bytes()));
            worker_wallets.push(wallet);
        }

        display::info(&format!(
            "Owner: {:?}, {} workers configured",
            owner_wallet.address(),
            worker_wallets.len(),
        ));

        // Deploy V4 coordinator
        display::info("Deploying HelixCoordinatorV4...");
        let (owner_client, deploy_result) = ChainClientV4::deploy(
            &rpc_url,
            &owner_pk,
            owner_wallet.address(),
            Address::zero(), // no ZK verifier needed
            Some(chain_id),
        )
        .await
        .context("V4 contract deployment failed")?;

        let coordinator_addr = deploy_result.coordinator.clone();
        display::success(&format!(
            "HelixCoordinatorV4 deployed at {}",
            coordinator_addr,
        ));

        // Deploy Halo2Verifier when ZK mode is enabled
        if self.args.zk_mode != ZkMode::Off {
            display::subphase("Deploying Halo2Verifier (real BN254 pairing verifier)...");
            let verifier_addr = owner_client
                .deploy_halo2_verifier()
                .await
                .context("Halo2Verifier deployment failed")?;
            owner_client
                .set_verifier(verifier_addr)
                .await
                .context("set_verifier failed")?;
            display::info(&format!("  Halo2Verifier deployed at {:#x}", verifier_addr));
        }

        // Register training job
        let architecture_hash = [0x42u8; 32];
        let payment = U256::from(1_000_000_000_000_000_000u128); // 1 ETH
        let (_receipt, job_id) = match self.args.zk_mode {
            ZkMode::Off => {
                owner_client
                    .register_training_job(
                        architecture_hash,
                        self.args.checkpoint_freq,
                        self.args.steps as u64,
                        payment,
                    )
                    .await
                    .context("register_training_job failed")?
            }
            ZkMode::Always => {
                owner_client
                    .register_training_job_with_zk(
                        architecture_hash,
                        self.args.checkpoint_freq,
                        self.args.steps as u64,
                        payment,
                        true,  // zkEnabled
                        1,     // zkCheckpointFreq = every checkpoint
                        false, // riskZkEnabled
                        0,     // minWorkersForMpc (not used in always mode)
                        Address::zero(),
                    )
                    .await
                    .context("register_training_job_with_zk failed")?
            }
            ZkMode::Risk => {
                owner_client
                    .register_training_job_with_zk(
                        architecture_hash,
                        self.args.checkpoint_freq,
                        self.args.steps as u64,
                        payment,
                        false, // zkEnabled
                        1,     // zkCheckpointFreq = every checkpoint
                        true,  // riskZkEnabled
                        self.args.min_workers_for_mpc as u64,
                        Address::zero(),
                    )
                    .await
                    .context("register_training_job_with_zk failed")?
            }
        };

        display::success(&format!("Training job registered (job_id={})", job_id));

        // Workers stake and join
        let stake = U256::from(1_000_000_000_000_000u128); // 0.001 ETH (matches poolMinStake)
        for (i, pk) in worker_pks.iter().enumerate() {
            let worker_client = ChainClientV4::new(
                &rpc_url,
                pk,
                &coordinator_addr,
                Some(chain_id),
            )
            .await
            .context("worker client creation failed")?;

            worker_client
                .stake_and_join(job_id, stake)
                .await
                .context("stake_and_join failed")?;

            display::success(&format!(
                "Worker {} staked 0.1 ETH and joined (addr: {:?})",
                i + 1,
                worker_wallets[i].address(),
            ));
        }

        // Submit checkpoints on-chain
        if result.checkpoints.is_empty() {
            display::warn("No checkpoints to submit (training may have been too short)");
            return Ok(ChainStats {
                checkpoints_submitted: 0,
                total_gas: 0,
                contract_address: coordinator_addr,
                job_completed: false,
            });
        }

        let checkpoint_data: Vec<CheckpointData> = result
            .checkpoints
            .iter()
            .map(|cp| {
                // Find matching ZK proof for this checkpoint
                let zk_proof = zk_proofs.iter().find(|p| p.step == cp.step as u64);
                let (proof_bytes, pub_inputs) = if let Some(zk) = zk_proof {
                    use helix_prover::halo2curves::ff::PrimeField;
                    let pub_inputs_u256: Vec<U256> = zk
                        .proof
                        .public_inputs
                        .iter()
                        .map(|fr| {
                            let repr = fr.to_repr();
                            U256::from_little_endian(repr.as_ref())
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

        display::info(&format!(
            "Signing and submitting {} checkpoints on-chain ({} worker signatures each)...",
            checkpoint_data.len(),
            worker_wallets.len(),
        ));

        let submissions = submit_all_checkpoints(
            &owner_client,
            job_id,
            &checkpoint_data,
            &worker_wallets,
        )
        .await
        .context("checkpoint submission failed")?;

        let mut total_gas = 0u64;
        for sub in &submissions {
            let gas = sub.receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
            total_gas += gas;
            display::chain_checkpoint_submitted(
                sub.step,
                &format!("{:?}", sub.receipt.transaction_hash),
                gas,
                sub.signer_count,
            );
        }

        // Verify on-chain state
        let onchain_count = owner_client
            .get_checkpoint_count(job_id)
            .await
            .context("get_checkpoint_count failed")?;

        display::success(&format!(
            "{} checkpoints verified on-chain (total gas: {})",
            onchain_count, total_gas,
        ));

        // Complete training with final commitment signed by all workers
        let final_commitment = checkpoint_data
            .last()
            .map(|cp| cp.commitment_bytes32)
            .unwrap_or([0u8; 32]);

        let mut completion_sigs = Vec::new();
        for wallet in &worker_wallets {
            let sig = sign_completion(wallet, U256::from(job_id), final_commitment)
                .await
                .context("sign_completion failed")?;
            completion_sigs.push(sig);
        }

        owner_client
            .complete_training(job_id, final_commitment, completion_sigs)
            .await
            .context("complete_training failed")?;

        display::success("Training completed and finalized on-chain");

        // Keep anvil alive until function returns (it's owned by this scope)
        let _ = &anvil;

        Ok(ChainStats {
            checkpoints_submitted: submissions.len(),
            total_gas,
            contract_address: coordinator_addr,
            job_completed: true,
        })
    }
}
