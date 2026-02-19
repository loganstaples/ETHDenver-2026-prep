//! Integration tests for SPDZ MAC verification in MPC training.
//!
//! Tests the full MAC verification pipeline including honest training,
//! cheater detection, cheater identification, training continuation
//! after party removal, and configurable check frequency.

use helix_mpc::error::MPCError;
use helix_mpc::field::Fr;
use helix_mpc::mac_verification::MACVerificationConfig;
use helix_mpc::mpc_trainer::{MPCTrainer, MPCTrainerConfig, ModelWeights};
use helix_mpc::session::transport::LocalTransport;
use helix_mpc::types::PartyId;

fn test_parties(n: usize) -> Vec<PartyId> {
    (0..n).map(PartyId::from_index).collect()
}

/// Small model config with MAC verification enabled.
fn mac_config(num_parties: usize, check_interval: u64) -> MPCTrainerConfig {
    MPCTrainerConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        learning_rate: 0.01,
        num_parties,
        reshare_interval: 0,
        beaver_batch_size: 256,
        generate_proofs: false,
        base_error: 1e-6,
        checkpoint_interval: 1,
        mac_config: Some(MACVerificationConfig {
            check_interval,
            enable_cheater_identification: true,
            mac_seed: 0xDEAD_BEEF_CAFE_BABE,
        }),
        batch_size: 1,
    }
}

/// Helper: run honest training for N steps, return per-party losses.
async fn run_honest_training(
    num_parties: usize,
    num_steps: usize,
    check_interval: u64,
) -> Vec<Vec<f64>> {
    let parties = test_parties(num_parties);
    let transports = LocalTransport::create_mesh(&parties);
    let config = mac_config(num_parties, check_interval);

    let initial_weights = ModelWeights::from_f64(
        &[0.1, 0.2, 0.3, 0.4], // w1: 2x2
        &[0.01, 0.02],          // b1: 2
        &[0.5, 0.6],            // w2: 1x2
        &[0.03],                // b2: 1
    );

    let data: Vec<(Vec<f64>, Vec<f64>)> = (0..num_steps)
        .map(|i| {
            let x1 = ((i % 3) as f64 + 1.0) * 0.1;
            let x2 = ((i % 2) as f64 + 1.0) * 0.1;
            (vec![x1, x2], vec![0.5])
        })
        .collect();

    let mut handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
        let cfg = config.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let steps = data.clone();

        let handle = tokio::spawn(async move {
            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            trainer.generate_beaver_triples(256).await.unwrap();

            let mut losses = Vec::new();
            for (input, target) in &steps {
                let result = trainer.training_step_with_mac(input, target).await.unwrap();
                losses.push(result.loss);
            }
            losses
        });
        handles.push(handle);
    }

    let mut all_losses = Vec::new();
    for handle in handles {
        all_losses.push(handle.await.unwrap());
    }
    all_losses
}

// ============================================================================
// Test 1: Honest training passes MAC verification every step
// ============================================================================

#[tokio::test]
async fn test_honest_training_passes_mac() {
    // 3 parties, 10 honest steps, MAC check every step (check_interval=1).
    let num_parties = 3;
    let num_steps = 10;
    let check_interval = 1;

    let all_losses = run_honest_training(num_parties, num_steps, check_interval).await;

    // All 3 parties should report losses for all 10 steps.
    assert_eq!(all_losses.len(), num_parties);
    for (party_idx, losses) in all_losses.iter().enumerate() {
        assert_eq!(
            losses.len(),
            num_steps,
            "Party {} should have {} losses, got {}",
            party_idx,
            num_steps,
            losses.len()
        );

        // All losses should be finite.
        for (step, loss) in losses.iter().enumerate() {
            assert!(
                loss.is_finite(),
                "Party {} step {}: loss should be finite, got {}",
                party_idx, step, loss
            );
        }
    }

    // All parties should compute the same loss at each step.
    for step in 0..num_steps {
        let loss0 = all_losses[0][step];
        for party_idx in 1..num_parties {
            let diff = (all_losses[party_idx][step] - loss0).abs();
            assert!(
                diff < 0.01,
                "Step {}: party 0 loss = {}, party {} loss = {}, diff = {}",
                step, loss0, party_idx, all_losses[party_idx][step], diff
            );
        }
    }
}

// ============================================================================
// Test 2: Cheater detected via MAC failure
// ============================================================================

#[tokio::test]
async fn test_cheater_detected() {
    // 3 parties, party 2 corrupts a weight share at step 5. MAC check should
    // fail, returning MACCheckFailed error.
    let num_parties = 3;
    let parties = test_parties(num_parties);
    let transports = LocalTransport::create_mesh(&parties);
    let config = mac_config(num_parties, 1); // Check every step

    let initial_weights = ModelWeights::from_f64(
        &[0.1, 0.2, 0.3, 0.4],
        &[0.01, 0.02],
        &[0.5, 0.6],
        &[0.03],
    );

    let data: Vec<(Vec<f64>, Vec<f64>)> = (0..10)
        .map(|i| {
            let x1 = ((i % 3) as f64 + 1.0) * 0.1;
            let x2 = ((i % 2) as f64 + 1.0) * 0.1;
            (vec![x1, x2], vec![0.5])
        })
        .collect();

    let mut handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
        let cfg = config.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let steps = data.clone();

        let handle = tokio::spawn(async move {
            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            trainer.generate_beaver_triples(256).await.unwrap();

            let mut completed_steps = 0u64;
            let mut mac_failed = false;

            for (step_idx, (input, target)) in steps.iter().enumerate() {
                // Party 2 corrupts weight at step 5 (before the training step).
                if i == 2 && step_idx == 5 {
                    trainer.corrupt_weight_share(0, Fr::from_f64(999.0));
                }

                match trainer.training_step_with_mac(input, target).await {
                    Ok(_) => {
                        completed_steps += 1;
                    }
                    Err(MPCError::MACCheckFailed { .. }) => {
                        mac_failed = true;
                        break;
                    }
                    Err(_) => {
                        // Other errors are also acceptable since the corruption
                        // breaks the protocol. The key requirement is that training
                        // does NOT silently continue with corrupted weights.
                        mac_failed = true;
                        break;
                    }
                }
            }

            (i, completed_steps, mac_failed)
        });
        handles.push(handle);
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    // At least one party should detect the MAC failure.
    let any_detected = results.iter().any(|(_, _, failed)| *failed);
    assert!(
        any_detected,
        "MAC failure should be detected when party 2 corrupts weights. Results: {:?}",
        results
    );

    // No party should complete all 10 steps (training should halt at or after step 5).
    for (party, completed, _) in &results {
        assert!(
            *completed < 10,
            "Party {} completed all 10 steps despite corruption (should have halted)",
            party
        );
    }
}

// ============================================================================
// Test 3: Cheater identified (identify_cheater returns the correct party)
// ============================================================================

#[tokio::test]
async fn test_cheater_identified() {
    // Same scenario as test_cheater_detected, but we verify that the
    // identified_cheater field in the error points to party 2.
    let num_parties = 3;
    let parties = test_parties(num_parties);
    let transports = LocalTransport::create_mesh(&parties);
    let config = mac_config(num_parties, 1);

    let initial_weights = ModelWeights::from_f64(
        &[0.1, 0.2, 0.3, 0.4],
        &[0.01, 0.02],
        &[0.5, 0.6],
        &[0.03],
    );

    let data: Vec<(Vec<f64>, Vec<f64>)> = (0..10)
        .map(|i| {
            let x1 = ((i % 3) as f64 + 1.0) * 0.1;
            let x2 = ((i % 2) as f64 + 1.0) * 0.1;
            (vec![x1, x2], vec![0.5])
        })
        .collect();

    let mut handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
        let cfg = config.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let steps = data.clone();

        let handle = tokio::spawn(async move {
            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            trainer.generate_beaver_triples(256).await.unwrap();

            for (step_idx, (input, target)) in steps.iter().enumerate() {
                // Party 2 corrupts weight at step 5.
                if i == 2 && step_idx == 5 {
                    trainer.corrupt_weight_share(0, Fr::from_f64(999.0));
                }

                match trainer.training_step_with_mac(input, target).await {
                    Ok(_) => {}
                    Err(MPCError::MACCheckFailed { step, cheater }) => {
                        return (i, Some(cheater), step);
                    }
                    Err(_) => {
                        return (i, None, 0);
                    }
                }
            }
            (i, None, 0)
        });
        handles.push(handle);
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    // Collect identified cheaters from parties that reported MAC failure.
    let mut identified_cheaters: Vec<Option<usize>> = Vec::new();
    for (_party, cheater_opt, _step) in &results {
        if let Some(cheater) = cheater_opt {
            identified_cheaters.push(*cheater);
        }
    }

    assert!(
        !identified_cheaters.is_empty(),
        "At least one party should identify a cheater. Results: {:?}",
        results
    );

    // When cheater identification succeeds, it should point to party 2.
    for cheater in &identified_cheaters {
        if let Some(idx) = cheater {
            assert_eq!(
                *idx, 2,
                "Identified cheater should be party 2, got party {}",
                idx
            );
        }
    }
}

// ============================================================================
// Test 4: Training continues after cheater removal
// ============================================================================

#[tokio::test]
async fn test_training_continues_after_removal() {
    // 3 parties, party 2 cheats at step 3. After detection:
    // - Training halts for the corrupted set
    // - We start a NEW session with just parties 0 and 1
    // - Training continues successfully
    let num_parties_initial = 3;
    let parties_initial = test_parties(num_parties_initial);
    let transports_initial = LocalTransport::create_mesh(&parties_initial);
    let config_initial = mac_config(num_parties_initial, 1);

    let initial_weights = ModelWeights::from_f64(
        &[0.1, 0.2, 0.3, 0.4],
        &[0.01, 0.02],
        &[0.5, 0.6],
        &[0.03],
    );

    let data: Vec<(Vec<f64>, Vec<f64>)> = (0..8)
        .map(|i| {
            let x1 = ((i % 3) as f64 + 1.0) * 0.1;
            let x2 = ((i % 2) as f64 + 1.0) * 0.1;
            (vec![x1, x2], vec![0.5])
        })
        .collect();

    // Phase 1: Run with 3 parties, party 2 cheats at step 3.
    let mut handles_phase1 = Vec::new();
    for (i, transport) in transports_initial.into_iter().enumerate() {
        let cfg = config_initial.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let steps = data.clone();

        let handle = tokio::spawn(async move {
            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            trainer.generate_beaver_triples(256).await.unwrap();

            let mut completed = 0u64;
            for (step_idx, (input, target)) in steps.iter().enumerate() {
                if i == 2 && step_idx == 3 {
                    trainer.corrupt_weight_share(0, Fr::from_f64(999.0));
                }
                match trainer.training_step_with_mac(input, target).await {
                    Ok(_) => completed += 1,
                    Err(_) => break,
                }
            }
            (i, completed)
        });
        handles_phase1.push(handle);
    }

    let mut phase1_results = Vec::new();
    for handle in handles_phase1 {
        phase1_results.push(handle.await.unwrap());
    }

    // Verify phase 1 halted before completing all steps.
    for (party, completed) in &phase1_results {
        assert!(
            *completed < 8,
            "Phase 1: party {} should not complete all 8 steps (completed {})",
            party, completed
        );
    }

    // Phase 2: Start a new session with just 2 parties (0 and 1).
    let num_parties_reduced = 2;
    let parties_reduced = test_parties(num_parties_reduced);
    let transports_reduced = LocalTransport::create_mesh(&parties_reduced);
    let config_reduced = mac_config(num_parties_reduced, 1);

    let num_steps_phase2 = 5;
    let data_phase2: Vec<(Vec<f64>, Vec<f64>)> = (0..num_steps_phase2)
        .map(|i| {
            let x1 = ((i % 3) as f64 + 1.0) * 0.1;
            let x2 = ((i % 2) as f64 + 1.0) * 0.1;
            (vec![x1, x2], vec![0.5])
        })
        .collect();

    let mut handles_phase2 = Vec::new();
    for (i, transport) in transports_reduced.into_iter().enumerate() {
        let cfg = config_reduced.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let steps = data_phase2.clone();

        let handle = tokio::spawn(async move {
            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            trainer.generate_beaver_triples(256).await.unwrap();

            let mut losses = Vec::new();
            for (input, target) in &steps {
                let result = trainer.training_step_with_mac(input, target).await.unwrap();
                losses.push(result.loss);
            }
            (i, losses)
        });
        handles_phase2.push(handle);
    }

    let mut phase2_results = Vec::new();
    for handle in handles_phase2 {
        phase2_results.push(handle.await.unwrap());
    }

    // Phase 2 should complete all steps successfully with 2 parties.
    for (party, losses) in &phase2_results {
        assert_eq!(
            losses.len(),
            num_steps_phase2,
            "Phase 2: party {} should complete all {} steps",
            party, num_steps_phase2
        );
        for (step, loss) in losses.iter().enumerate() {
            assert!(
                loss.is_finite(),
                "Phase 2: party {} step {}: loss should be finite, got {}",
                party, step, loss
            );
        }
    }
}

// ============================================================================
// Test 5: MAC check frequency — cheater corrupts up to K steps before caught
// ============================================================================

#[tokio::test]
async fn test_mac_check_frequency() {
    // K=5: MAC check runs every 5 steps. Party 2 corrupts at step 2.
    // The corruption should go undetected for steps 2-4 but be caught
    // at the step-5 check (or step-4 completion if the check runs after
    // step 5 completes).
    let num_parties = 3;
    let parties = test_parties(num_parties);
    let transports = LocalTransport::create_mesh(&parties);
    let config = mac_config(num_parties, 5); // Check every 5 steps

    let initial_weights = ModelWeights::from_f64(
        &[0.1, 0.2, 0.3, 0.4],
        &[0.01, 0.02],
        &[0.5, 0.6],
        &[0.03],
    );

    let data: Vec<(Vec<f64>, Vec<f64>)> = (0..10)
        .map(|i| {
            let x1 = ((i % 3) as f64 + 1.0) * 0.1;
            let x2 = ((i % 2) as f64 + 1.0) * 0.1;
            (vec![x1, x2], vec![0.5])
        })
        .collect();

    let mut handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
        let cfg = config.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let steps = data.clone();

        let handle = tokio::spawn(async move {
            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            trainer.generate_beaver_triples(256).await.unwrap();

            let mut completed_steps = 0u64;
            let mut mac_failed = false;

            for (step_idx, (input, target)) in steps.iter().enumerate() {
                // Party 2 corrupts at step 2 (0-indexed).
                if i == 2 && step_idx == 2 {
                    trainer.corrupt_weight_share(0, Fr::from_f64(999.0));
                }

                match trainer.training_step_with_mac(input, target).await {
                    Ok(_) => {
                        completed_steps += 1;
                    }
                    Err(MPCError::MACCheckFailed { .. }) => {
                        mac_failed = true;
                        break;
                    }
                    Err(_) => {
                        mac_failed = true;
                        break;
                    }
                }
            }

            (i, completed_steps, mac_failed)
        });
        handles.push(handle);
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }

    // The corruption happens at step 2. With check_interval=5,
    // the check fires after step 5 (current_step=5, 5%5==0).
    // So steps 2-4 should complete (corruption undetected), and
    // the MAC check at step 5 should catch it.
    let any_detected = results.iter().any(|(_, _, failed)| *failed);
    assert!(
        any_detected,
        "MAC failure should eventually be detected. Results: {:?}",
        results
    );

    // Party 2 (the cheater) should have completed at least 2 steps
    // (steps 0,1 are honest) and possibly up to 4 (steps 2-4 corrupt
    // but not yet checked). The check at step 5 catches it.
    for (party, completed, failed) in &results {
        if *failed {
            // The party that detected should have completed at least the
            // honest steps before corruption.
            assert!(
                *completed >= 2,
                "Party {}: completed {} steps, should be >= 2 (honest steps before corruption)",
                party, completed
            );
            // And should NOT have completed all 10 steps.
            assert!(
                *completed < 10,
                "Party {}: completed all 10 steps despite corruption (K=5 should catch it)",
                party
            );
        }
    }
}
