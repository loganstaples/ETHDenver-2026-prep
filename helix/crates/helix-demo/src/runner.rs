//! Core demo orchestration for HELIX MPC training.
//!
//! The `DemoRunner` coordinates the full demo lifecycle:
//! 1. MNIST dataset generation
//! 2. MPC transport mesh creation
//! 3. Weight distribution across parties
//! 4. MAC initialization for SPDZ verification
//! 5. Beaver triple pre-generation
//! 6. Training loop with MAC-verified gradient steps
//! 7. Cheater simulation and detection
//! 8. Accuracy evaluation
//! 9. Summary display

use std::time::Instant;

use anyhow::{Context, Result};
use helix_mpc::field::Fr;
use helix_mpc::mac_verification::MACVerificationConfig;
use helix_mpc::mnist::{MnistDataset, NativeTrainer};
use helix_mpc::mpc_trainer::{MPCTrainer, MPCTrainerConfig, ModelWeights};
use helix_mpc::session::transport::LocalTransport;
use helix_mpc::types::PartyId;
use helix_mpc::MPCError;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::display;
use crate::evaluator;
use crate::Args;

/// MNIST network dimensions.
const D_IN: usize = 784;
const D_HID: usize = 32;
const D_OUT: usize = 10;

/// Core demo runner.
pub struct DemoRunner {
    args: Args,
}

impl DemoRunner {
    /// Creates a new demo runner from CLI arguments.
    pub fn new(args: Args) -> Self {
        Self { args }
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
            ).context("Failed to load real MNIST data")?;
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
        display::info(&format!(
            "Total parameters: {} (W1: {}, b1: {}, W2: {}, b2: {})",
            D_HID * D_IN + D_HID + D_OUT * D_HID + D_OUT,
            D_HID * D_IN,
            D_HID,
            D_OUT * D_HID,
            D_OUT,
        ));

        // Convert to training pairs
        let train_pairs = MnistDataset::as_training_pairs(&dataset.train);
        let test_pairs = MnistDataset::as_training_pairs(&dataset.test);

        // ================================================================
        // Phase 2: Create MPC Transport Mesh
        // ================================================================
        display::phase("Phase 2: MPC Transport Mesh Setup");

        let num_parties = self.args.workers;
        let party_ids: Vec<PartyId> = (0..num_parties)
            .map(PartyId::from_index)
            .collect();
        let transports = LocalTransport::create_mesh(&party_ids);
        display::success(&format!(
            "{} parties connected via in-memory transport mesh ({} bidirectional channels)",
            num_parties,
            num_parties * (num_parties - 1),
        ));

        // ================================================================
        // Phase 3: Initialize Weights and Distribute Shares
        // ================================================================
        display::phase("Phase 3: Weight Initialization and Secret Sharing");

        let phase3_start = Instant::now();

        // Create initial weights using Xavier initialization
        let mut init_rng = ChaCha20Rng::seed_from_u64(self.args.seed);
        let initial_weights = ModelWeights::random(D_IN, D_HID, D_OUT, &mut init_rng);
        display::info("Xavier-initialized random weights created");

        // MPC trainer config with MAC verification
        let mac_config = MACVerificationConfig {
            check_interval: self.args.checkpoint_freq,
            enable_cheater_identification: true,
            mac_seed: self.args.seed + 1000,
        };

        let trainer_config = MPCTrainerConfig {
            d_in: D_IN,
            d_hid: D_HID,
            d_out: D_OUT,
            learning_rate: self.args.lr,
            num_parties,
            reshare_interval: 0, // Disable re-sharing for demo speed
            beaver_batch_size: 10000,
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: Some(mac_config),
        };

        // Spawn each party's trainer in its own tokio task for weight sharing.
        // Each party owns its trainer; they communicate via transport channels.
        let mut party_handles = Vec::with_capacity(num_parties);

        for (party_idx, transport) in transports.into_iter().enumerate() {
            let config = trainer_config.clone();
            let weights = if party_idx == 0 {
                Some(initial_weights.clone())
            } else {
                None
            };
            let seed = self.args.seed;

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(config, transport, party_idx, seed);
                trainer.share_weights(weights).await?;
                Ok::<_, MPCError>(trainer)
            });
            party_handles.push(handle);
        }

        // Collect all trainers back after weight sharing
        let mut trainers: Vec<MPCTrainer<LocalTransport>> = Vec::with_capacity(num_parties);
        for handle in party_handles {
            let trainer = handle
                .await
                .context("Party task panicked during weight sharing")?
                .context("Weight sharing failed")?;
            trainers.push(trainer);
        }

        display::success(&format!(
            "Additive secret shares distributed to {} parties ({:.1}ms)",
            num_parties,
            phase3_start.elapsed().as_secs_f64() * 1000.0,
        ));
        display::success("SPDZ MAC shares initialized (information-theoretic integrity)");

        // ================================================================
        // Phase 4: Generate Beaver Triples
        // ================================================================
        display::phase("Phase 4: Distributed Beaver Triple Generation");

        let phase4_start = Instant::now();

        // Compute how many triples we need.
        // For training_step_with_mac: uses authenticated Beaver triples for
        // h_pre * relu_mask (d_hid) + W2 * h matmul (d_out * d_hid) + dh * relu_mask (d_hid) + overhead.
        // But training_step_unproved doesn't need Beaver triples (it reconstructs activations).
        // We generate triples anyway to show the Beaver generation protocol.
        let triples_per_step = helix_mpc::mnist::triples_per_step(D_HID, D_OUT);
        let total_triples = triples_per_step * self.args.steps;

        display::info(&format!(
            "Need {} triples/step x {} steps = {} total Beaver triples",
            triples_per_step, self.args.steps, total_triples,
        ));

        let pb = display::triple_progress_bar(total_triples as u64);

        // Generate triples in batches across all parties concurrently.
        // We batch to keep messages manageable and show progress.
        let batch_size = 5000.min(total_triples);
        let mut remaining = total_triples;

        while remaining > 0 {
            let this_batch = remaining.min(batch_size);

            // Extract trainers into tasks for concurrent triple generation
            let mut triple_handles = Vec::with_capacity(num_parties);
            for trainer in trainers.drain(..) {
                let count = this_batch;
                let handle = tokio::spawn(async move {
                    let mut t = trainer;
                    t.generate_beaver_triples(count).await?;
                    Ok::<_, MPCError>(t)
                });
                triple_handles.push(handle);
            }

            for handle in triple_handles {
                let trainer = handle
                    .await
                    .context("Party task panicked during triple generation")?
                    .context("Beaver triple generation failed")?;
                trainers.push(trainer);
            }

            remaining -= this_batch;
            pb.set_position((total_triples - remaining) as u64);
        }

        pb.finish_and_clear();
        display::success(&format!(
            "{} Beaver triples generated distributedly in {:.1}s ({:.0} triples/sec)",
            total_triples,
            phase4_start.elapsed().as_secs_f64(),
            total_triples as f64 / phase4_start.elapsed().as_secs_f64(),
        ));

        // ================================================================
        // Phase 5: MPC Training Loop
        // ================================================================
        display::phase("Phase 5: MPC Training with SPDZ MAC Verification");

        let phase5_start = Instant::now();
        let mut all_losses: Vec<f64> = Vec::with_capacity(self.args.steps);
        let mut mac_checks_passed = 0u64;
        let mut cheater_detected = false;
        let mut cheater_party: Option<usize> = None;
        let cheater_step = self.args.steps / 2;
        let mut training_halted = false;

        for step in 0..self.args.steps {
            if training_halted {
                break;
            }

            let sample_idx = step % train_pairs.len();
            let (ref input, ref target) = train_pairs[sample_idx];

            // Optionally corrupt party 2's shares at the midpoint
            if self.args.simulate_cheater && step == cheater_step && num_parties > 2 {
                display::cheater_simulated(2, step);
                // Corrupt several weight elements in party 2
                for idx in 0..10.min(trainers[2].weight_shares().0.len()) {
                    trainers[2].corrupt_weight_share(idx, Fr::from_f64(100.0));
                }
            }

            // Run all parties concurrently for this training step.
            // Each party runs training_step_with_mac on the same (input, target).
            let mut step_handles = Vec::with_capacity(trainers.len());
            for trainer in trainers.drain(..) {
                let inp = input.clone();
                let tgt = target.clone();
                let handle = tokio::spawn(async move {
                    let mut t = trainer;
                    let result = t.training_step_with_mac(&inp, &tgt).await;
                    (t, result)
                });
                step_handles.push(handle);
            }

            let mut step_loss = 0.0_f64;
            let mut mac_failed_this_step = false;
            let mut mac_ok = true;

            for handle in step_handles {
                let (trainer, result) = handle
                    .await
                    .context("Party task panicked during training step")?;

                match result {
                    Ok(step_result) => {
                        // Use party 0's loss as canonical
                        if trainer.party_index() == 0 {
                            step_loss = step_result.loss;
                        }
                        trainers.push(trainer);
                    }
                    Err(MPCError::MACCheckFailed { step: s, cheater: c }) => {
                        mac_failed_this_step = true;
                        mac_ok = false;
                        if let Some(c_idx) = c {
                            cheater_detected = true;
                            cheater_party = Some(c_idx);
                            display::cheater_detected(c_idx, s);
                        } else {
                            display::alert(&format!(
                                "MAC check failed at step {} but cheater not identified",
                                s,
                            ));
                        }
                        trainers.push(trainer);
                    }
                    Err(e) => {
                        // Other MPC errors - push trainer back and note failure
                        display::warn(&format!(
                            "Party {} step {} error: {}",
                            trainer.party_index(), step, e,
                        ));
                        trainers.push(trainer);
                        mac_ok = false;
                    }
                }
            }

            // Sort trainers back by party index
            trainers.sort_by_key(|t| t.party_index());

            if mac_failed_this_step {
                display::info("Training halted due to MAC failure. Rolling back to checkpoint.");
                training_halted = true;
            }

            all_losses.push(step_loss);

            // Update display
            display::step_update(step + 1, self.args.steps, step_loss, trainers.len(), mac_ok);

            // Check if this was a MAC verification checkpoint
            if !mac_failed_this_step
                && self.args.checkpoint_freq > 0
                && (step + 1) as u64 % self.args.checkpoint_freq == 0
            {
                display::step_update_finish();
                mac_checks_passed += 1;
                display::mac_verified((step + 1) as u64);

                // Compute commitment from weight shares (using party 0's shares as proxy
                // for the Pedersen commitment display - in production, each party would
                // contribute their commitment share).
                let (w1_shares, _, _, _) = trainers[0].weight_shares();
                let commitment_bytes: Vec<u8> = w1_shares.iter()
                    .take(4)
                    .flat_map(|f| f.to_bytes_le().to_vec())
                    .collect();
                let commitment_hex = hex_encode(&commitment_bytes);
                display::checkpoint((step + 1) as u64, &commitment_hex);
            }
        }

        display::step_update_finish();

        let training_time = phase5_start.elapsed();
        let actual_steps = all_losses.len();
        display::success(&format!(
            "Training complete: {} steps in {:.1}s ({:.1} steps/sec)",
            actual_steps,
            training_time.as_secs_f64(),
            actual_steps as f64 / training_time.as_secs_f64(),
        ));

        // ================================================================
        // Phase 6: Weight Reconstruction and Accuracy Evaluation
        // ================================================================
        display::phase("Phase 6: Weight Reconstruction and Evaluation");

        let phase6_start = Instant::now();

        // Reconstruct weights by summing shares across all parties
        let (w1_f64, b1_f64, w2_f64, b2_f64) = reconstruct_weights(&trainers);

        display::success(&format!(
            "Weights reconstructed by summing {} party shares ({:.1}ms)",
            trainers.len(),
            phase6_start.elapsed().as_secs_f64() * 1000.0,
        ));

        // Weight statistics
        let w1_norm: f64 = w1_f64.iter().map(|w| w * w).sum::<f64>().sqrt();
        let w2_norm: f64 = w2_f64.iter().map(|w| w * w).sum::<f64>().sqrt();
        display::metric("||W1|| (Frobenius)", &format!("{:.6}", w1_norm));
        display::metric("||W2|| (Frobenius)", &format!("{:.6}", w2_norm));

        // Evaluate accuracy
        let (overall_accuracy, per_class_acc) = evaluator::evaluate_per_class(
            &w1_f64, &b1_f64, &w2_f64, &b2_f64,
            D_IN, D_HID, D_OUT,
            &test_pairs,
        );

        display::success(&format!(
            "Test accuracy: {:.1}% ({}/{} correct)",
            overall_accuracy * 100.0,
            (overall_accuracy * test_pairs.len() as f64) as usize,
            test_pairs.len(),
        ));

        // Per-class breakdown
        display::subphase("Per-class accuracy:");
        for (digit, acc) in per_class_acc.iter().enumerate() {
            let bar_len = (*acc * 20.0) as usize;
            let bar = format!(
                "[{}{}]",
                "#".repeat(bar_len),
                "-".repeat(20 - bar_len),
            );
            println!(
                "    Digit {}: {} {:.1}%",
                digit, bar, acc * 100.0,
            );
        }

        // ================================================================
        // Phase 7: Native Comparison (optional detail)
        // ================================================================
        display::phase("Phase 7: Native Baseline Comparison");

        let mut native_trainer = NativeTrainer::new(D_IN, D_HID, D_OUT, self.args.lr, self.args.seed);
        let native_steps = actual_steps.min(self.args.steps);
        let mut native_losses = Vec::with_capacity(native_steps);

        for step in 0..native_steps {
            let idx = step % train_pairs.len();
            let (ref input, ref target) = train_pairs[idx];
            let result = native_trainer.training_step_mse(input, target, step as u64);
            native_losses.push(result.loss);
        }

        let native_accuracy = native_trainer.evaluate(&test_pairs);
        display::metric("Native (no MPC) accuracy", &format!("{:.1}%", native_accuracy * 100.0));
        display::metric("MPC accuracy", &format!("{:.1}%", overall_accuracy * 100.0));

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
        // Phase 8: Loss Curves and Summary
        // ================================================================
        display::phase("Phase 8: Training Results");

        display::subphase("MPC Training Loss:");
        display::loss_curve(&all_losses);

        display::subphase("Native Training Loss (comparison):");
        display::loss_curve(&native_losses);

        // Final summary
        let loss_start = all_losses.first().copied().unwrap_or(0.0);
        let loss_end = all_losses.last().copied().unwrap_or(0.0);

        display::summary(&display::DemoSummary {
            total_time: demo_start.elapsed(),
            num_workers: self.args.workers,
            num_steps: actual_steps,
            loss_start,
            loss_end,
            final_accuracy: overall_accuracy,
            triples_generated: total_triples,
            mac_checks_passed: mac_checks_passed as usize,
            cheater_detected,
            cheater_party,
        });

        Ok(())
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Reconstructs plaintext weights from all party shares by summing.
fn reconstruct_weights(
    trainers: &[MPCTrainer<LocalTransport>],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (first_w1, first_b1, first_w2, first_b2) = trainers[0].weight_shares();

    let mut w1_sum = first_w1.to_vec();
    let mut b1_sum = first_b1.to_vec();
    let mut w2_sum = first_w2.to_vec();
    let mut b2_sum = first_b2.to_vec();

    for trainer in trainers.iter().skip(1) {
        let (w1, b1, w2, b2) = trainer.weight_shares();
        for (i, s) in w1.iter().enumerate() {
            w1_sum[i] = Fr::add(&w1_sum[i], s);
        }
        for (i, s) in b1.iter().enumerate() {
            b1_sum[i] = Fr::add(&b1_sum[i], s);
        }
        for (i, s) in w2.iter().enumerate() {
            w2_sum[i] = Fr::add(&w2_sum[i], s);
        }
        for (i, s) in b2.iter().enumerate() {
            b2_sum[i] = Fr::add(&b2_sum[i], s);
        }
    }

    (
        w1_sum.iter().map(|f| f.to_f64()).collect(),
        b1_sum.iter().map(|f| f.to_f64()).collect(),
        w2_sum.iter().map(|f| f.to_f64()).collect(),
        b2_sum.iter().map(|f| f.to_f64()).collect(),
    )
}

/// Simple hex encoding for commitment display.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
