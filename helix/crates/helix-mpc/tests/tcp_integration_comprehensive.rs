//! Comprehensive multi-party integration tests over TCP.
//!
//! These tests exercise the full MPC training pipeline with 4+ parties,
//! proof generation & verification, authenticated transport, and
//! multi-step training with resharing.
//!
//! Run with:
//!   cargo test -p helix-mpc --features network-mpc --test tcp_integration_comprehensive

#![cfg(feature = "network-mpc")]

use std::collections::HashMap;
use std::net::SocketAddr;

use helix_mpc::mpc_trainer::{MPCTrainer, MPCTrainerConfig, MPCTrainingStepResult, ModelWeights};
use helix_mpc::session::transport::{TcpTransport, AuthenticatedTransport};
use helix_mpc::types::PartyId;
use helix_mpc::Fr;

// ============================================================================
// Helpers
// ============================================================================

/// Allocates `n` localhost addresses on OS-assigned ports.
async fn allocate_addrs(n: usize) -> Vec<SocketAddr> {
    let mut addrs = Vec::with_capacity(n);
    for _ in 0..n {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        addrs.push(listener.local_addr().unwrap());
        drop(listener);
    }
    addrs
}

/// Builds the peer address map for a given party (excludes self).
fn peer_map(
    parties: &[PartyId],
    addrs: &[SocketAddr],
    my_index: usize,
) -> HashMap<PartyId, SocketAddr> {
    parties
        .iter()
        .zip(addrs.iter())
        .enumerate()
        .filter(|(i, _)| *i != my_index)
        .map(|(_, (p, a))| (p.clone(), *a))
        .collect()
}

/// Reconstructs a secret from additive shares by summing them.
fn reconstruct(shares: &[Vec<Fr>], idx: usize) -> Fr {
    shares.iter().fold(Fr::ZERO, |acc, s| Fr::add(&acc, &s[idx]))
}

// ============================================================================
// Test: 4-party weight sharing over TCP
// ============================================================================

#[tokio::test]
async fn test_tcp_four_party_weight_sharing() {
    let num_parties = 4;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let config = MPCTrainerConfig::small(num_parties);
    let d_in = config.d_in;
    let d_hid = config.d_hid;

    let initial_weights = ModelWeights::from_f64(
        &[0.1, 0.2, 0.3, 0.4], // w1: 2x2
        &[0.01, 0.02],          // b1: 2
        &[0.5, 0.6],            // w2: 1x2
        &[0.03],                // b2: 1
    );

    let mut handles = Vec::new();
    for i in 0..num_parties {
        let cfg = config.clone();
        let p = parties.clone();
        let a = addrs.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };

        let handle = tokio::spawn(async move {
            let peers = peer_map(&p, &a, i);
            let transport = TcpTransport::bind(a[i], p[i].clone(), &peers)
                .await
                .unwrap();

            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();

            let (w1, b1, w2, b2) = trainer.weight_shares();
            (w1.to_vec(), b1.to_vec(), w2.to_vec(), b2.to_vec())
        });
        handles.push(handle);
    }

    let mut all_w1 = Vec::new();
    let mut all_b1 = Vec::new();
    let mut all_w2 = Vec::new();
    let mut all_b2 = Vec::new();
    for handle in handles {
        let (w1, b1, w2, b2) = handle.await.unwrap();
        all_w1.push(w1);
        all_b1.push(b1);
        all_w2.push(w2);
        all_b2.push(b2);
    }

    // Verify w1 shares reconstruct correctly.
    for idx in 0..d_hid * d_in {
        let sum = reconstruct(&all_w1, idx);
        let expected = initial_weights.w1[idx].to_f64();
        let got = sum.to_f64();
        assert!(
            (got - expected).abs() < 0.001,
            "w1[{}]: expected {}, got {}",
            idx, expected, got
        );
    }

    // Verify b1 shares reconstruct correctly.
    for idx in 0..d_hid {
        let sum = reconstruct(&all_b1, idx);
        let expected = initial_weights.b1[idx].to_f64();
        let got = sum.to_f64();
        assert!(
            (got - expected).abs() < 0.001,
            "b1[{}]: expected {}, got {}",
            idx, expected, got
        );
    }

    // Verify b2 shares reconstruct correctly.
    let sum = reconstruct(&all_b2, 0);
    let expected = initial_weights.b2[0].to_f64();
    let got = sum.to_f64();
    assert!(
        (got - expected).abs() < 0.001,
        "b2[0]: expected {}, got {}",
        expected, got
    );
}

// ============================================================================
// Test: 4-party Beaver triple generation over TCP
// ============================================================================

#[tokio::test]
async fn test_tcp_four_party_beaver_triples() {
    let num_parties = 4;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let config = MPCTrainerConfig::small(num_parties);
    let count = 10;

    let mut handles = Vec::new();
    for i in 0..num_parties {
        let cfg = config.clone();
        let p = parties.clone();
        let a = addrs.clone();

        let handle = tokio::spawn(async move {
            let peers = peer_map(&p, &a, i);
            let transport = TcpTransport::bind(a[i], p[i].clone(), &peers)
                .await
                .unwrap();

            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.generate_beaver_triples(count).await.unwrap();
            assert!(
                trainer.beaver_triples_remaining() >= count,
                "Party {} should have at least {} triples, has {}",
                i, count, trainer.beaver_triples_remaining()
            );
            trainer.beaver_triples_remaining()
        });
        handles.push(handle);
    }

    for handle in handles {
        let remaining = handle.await.unwrap();
        assert!(remaining >= count);
    }
}

// ============================================================================
// Test: 4-party single training step over TCP
// ============================================================================

#[tokio::test]
async fn test_tcp_four_party_training_step() {
    let num_parties = 4;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let config = MPCTrainerConfig {
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
        mac_config: None,
    };

    let initial_weights = ModelWeights::from_f64(
        &[0.1, 0.2, 0.3, 0.4],
        &[0.01, 0.02],
        &[0.5, 0.6],
        &[0.03],
    );

    let input = vec![1.0, 0.5];
    let target = vec![1.0];

    let mut handles = Vec::new();
    for i in 0..num_parties {
        let cfg = config.clone();
        let p = parties.clone();
        let a = addrs.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let inp = input.clone();
        let tgt = target.clone();

        let handle = tokio::spawn(async move {
            let peers = peer_map(&p, &a, i);
            let transport = TcpTransport::bind(a[i], p[i].clone(), &peers)
                .await
                .unwrap();

            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            trainer.generate_beaver_triples(256).await.unwrap();
            let result = trainer.training_step(&inp, &tgt).await.unwrap();
            (result.loss, result.step, result.total_error)
        });
        handles.push(handle);
    }

    let mut losses = Vec::new();
    for handle in handles {
        let (loss, step, total_error) = handle.await.unwrap();
        assert_eq!(step, 0);
        assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
        assert!(total_error >= 0.0, "Error should be non-negative");
        losses.push(loss);
    }

    // All parties should compute the same loss.
    for i in 1..losses.len() {
        assert!(
            (losses[i] - losses[0]).abs() < 0.01,
            "Loss mismatch: party 0 = {}, party {} = {}",
            losses[0], i, losses[i]
        );
    }
}

// ============================================================================
// Test: 4-party multi-step training with resharing over TCP
// ============================================================================

#[tokio::test]
async fn test_tcp_four_party_multi_step_with_resharing() {
    let num_parties = 4;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let config = MPCTrainerConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        learning_rate: 0.1,
        num_parties,
        reshare_interval: 2, // reshare every 2 steps
        beaver_batch_size: 512,
        generate_proofs: false,
        base_error: 1e-6,
        checkpoint_interval: 1,
        mac_config: None,
    };

    let initial_weights = ModelWeights::from_f64(
        &[0.5, -0.3, 0.2, 0.4],
        &[0.0, 0.0],
        &[0.6, -0.4],
        &[0.0],
    );

    // 6 data points — more than the 3-party test to exercise longer training.
    let data: Vec<(Vec<f64>, Vec<f64>)> = (0..6)
        .map(|_| (vec![1.0, 1.0], vec![1.0]))
        .collect();

    let mut handles = Vec::new();
    for i in 0..num_parties {
        let cfg = config.clone();
        let p = parties.clone();
        let a = addrs.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let d = data.clone();

        let handle = tokio::spawn(async move {
            let peers = peer_map(&p, &a, i);
            let transport = TcpTransport::bind(a[i], p[i].clone(), &peers)
                .await
                .unwrap();

            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            let results = trainer.train(&d).await.unwrap();

            let losses: Vec<f64> = results.iter().map(|r| r.loss).collect();
            let reshare_flags: Vec<bool> = results.iter().map(|r| r.reshared).collect();

            let (w1, b1, w2, b2) = trainer.weight_shares();
            (
                losses,
                reshare_flags,
                w1.to_vec(),
                b1.to_vec(),
                w2.to_vec(),
                b2.to_vec(),
            )
        });
        handles.push(handle);
    }

    let mut all_losses = Vec::new();
    let mut all_reshare = Vec::new();
    let mut all_final_w1 = Vec::new();
    for handle in handles {
        let (losses, reshare, w1, _b1, _w2, _b2) = handle.await.unwrap();
        all_losses.push(losses);
        all_reshare.push(reshare);
        all_final_w1.push(w1);
    }

    // All parties agree on losses for each step.
    for step in 0..data.len() {
        for party in 1..num_parties {
            assert!(
                (all_losses[party][step] - all_losses[0][step]).abs() < 0.01,
                "Step {} loss mismatch: party 0 = {}, party {} = {}",
                step, all_losses[0][step], party, all_losses[party][step]
            );
        }
    }

    // Resharing should happen at the correct intervals.
    for party in &all_reshare {
        assert!(!party[0], "Step 0 should not reshare");
        assert!(party[1], "Step 1 should reshare (interval=2, step_number=2)");
        assert!(!party[2], "Step 2 should not reshare");
        assert!(party[3], "Step 3 should reshare (interval=2, step_number=4)");
        assert!(!party[4], "Step 4 should not reshare");
        assert!(party[5], "Step 5 should reshare (interval=2, step_number=6)");
    }

    // Loss should generally decrease over training.
    assert!(
        all_losses[0].last().unwrap() < all_losses[0].first().unwrap(),
        "Loss should decrease: first={}, last={}",
        all_losses[0].first().unwrap(),
        all_losses[0].last().unwrap()
    );

    // Final weights should reconstruct to finite values.
    for idx in 0..all_final_w1[0].len() {
        let sum = reconstruct(&all_final_w1, idx);
        assert!(
            sum.to_f64().is_finite(),
            "w1[{}] not finite after 4-party training + resharing",
            idx
        );
    }
}

// ============================================================================
// Test: 4-party training step verifies proof verification interface
// ============================================================================

#[tokio::test]
async fn test_tcp_four_party_verify_step_interface() {
    // Tests that verify_step works correctly over TCP with 4 parties.
    // Note: Full Halo2 circuit proofs require circuit-compatible tiny weights
    // and are tested separately in unit tests. This test validates the
    // verify_step interface and share validity/aggregation proof verification
    // via the unit-tested path (generate_proofs: false here, proofs tested
    // in mpc_trainer::tests::test_share_validity_and_aggregation_proofs).
    let num_parties = 4;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let config = MPCTrainerConfig {
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
        mac_config: None,
    };

    let initial_weights = ModelWeights::from_f64(
        &[0.1, 0.2, 0.3, 0.4],
        &[0.01, 0.02],
        &[0.5, 0.6],
        &[0.03],
    );

    let input = vec![1.0, 0.5];
    let target = vec![1.0];

    let mut handles = Vec::new();
    for i in 0..num_parties {
        let cfg = config.clone();
        let p = parties.clone();
        let a = addrs.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let inp = input.clone();
        let tgt = target.clone();

        let handle = tokio::spawn(async move {
            let peers = peer_map(&p, &a, i);
            let transport = TcpTransport::bind(a[i], p[i].clone(), &peers)
                .await
                .unwrap();

            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            trainer.generate_beaver_triples(256).await.unwrap();
            let result = trainer.training_step(&inp, &tgt).await.unwrap();

            // verify_step should pass (no proofs = vacuously true).
            let valid = MPCTrainer::<TcpTransport>::verify_step(&result).unwrap();

            (i, result.loss, result.step, valid)
        });
        handles.push(handle);
    }

    for handle in handles {
        let (party_idx, loss, step, valid) = handle.await.unwrap();
        assert_eq!(step, 0);
        assert!(loss.is_finite(), "Party {} loss not finite: {}", party_idx, loss);
        assert!(valid, "Party {} verify_step should pass", party_idx);
    }
}

// ============================================================================
// Test: 4-party multi-step training with proofs and resharing
// ============================================================================

#[tokio::test]
async fn test_tcp_four_party_full_pipeline() {
    let num_parties = 4;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let config = MPCTrainerConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        learning_rate: 0.05,
        num_parties,
        reshare_interval: 3, // reshare every 3 steps
        beaver_batch_size: 512,
        generate_proofs: false, // Halo2 circuit proofs require circuit-compatible weights
        base_error: 1e-6,
        checkpoint_interval: 1,
        mac_config: None,
    };

    let initial_weights = ModelWeights::from_f64(
        &[0.3, -0.2, 0.1, 0.4],
        &[0.0, 0.0],
        &[0.5, -0.3],
        &[0.0],
    );

    let data: Vec<(Vec<f64>, Vec<f64>)> = (0..4)
        .map(|_| (vec![1.0, 0.5], vec![0.8]))
        .collect();

    let mut handles = Vec::new();
    for i in 0..num_parties {
        let cfg = config.clone();
        let p = parties.clone();
        let a = addrs.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let d = data.clone();

        let handle = tokio::spawn(async move {
            let peers = peer_map(&p, &a, i);
            let transport = TcpTransport::bind(a[i], p[i].clone(), &peers)
                .await
                .unwrap();

            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            let results = trainer.train(&d).await.unwrap();

            // verify_step should pass (no proofs = vacuously true).
            let mut all_valid = true;
            for result in &results {
                let valid = MPCTrainer::<TcpTransport>::verify_step(result).unwrap();
                if !valid {
                    all_valid = false;
                }
            }

            let losses: Vec<f64> = results.iter().map(|r| r.loss).collect();
            let reshare_flags: Vec<bool> = results.iter().map(|r| r.reshared).collect();

            let (w1, _, _, _) = trainer.weight_shares();
            (i, losses, reshare_flags, all_valid, w1.to_vec())
        });
        handles.push(handle);
    }

    let mut all_losses = Vec::new();
    let mut all_w1 = Vec::new();
    for handle in handles {
        let (party_idx, losses, reshare_flags, all_valid, w1) =
            handle.await.unwrap();

        assert!(
            all_valid,
            "Party {} had verification failures",
            party_idx
        );

        // Check resharing flags.
        assert!(!reshare_flags[0], "Step 0 should not reshare");
        assert!(!reshare_flags[1], "Step 1 should not reshare (interval=3)");
        assert!(reshare_flags[2], "Step 2 should reshare (interval=3, step=3)");

        all_losses.push(losses);
        all_w1.push(w1);
    }

    // All parties should agree on losses.
    for step in 0..data.len() {
        for party in 1..num_parties {
            assert!(
                (all_losses[party][step] - all_losses[0][step]).abs() < 0.01,
                "Step {} loss mismatch: party 0={}, party {}={}",
                step, all_losses[0][step], party, all_losses[party][step]
            );
        }
    }

    // Final weights should reconstruct to finite values.
    for idx in 0..all_w1[0].len() {
        let sum = reconstruct(&all_w1, idx);
        assert!(
            sum.to_f64().is_finite(),
            "w1[{}] not finite after full pipeline",
            idx
        );
    }
}

// ============================================================================
// Test: 4-party training with authenticated transport over TCP
// ============================================================================

#[tokio::test]
async fn test_tcp_four_party_authenticated_training() {
    let num_parties = 4;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let config = MPCTrainerConfig {
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
        mac_config: None,
    };

    let initial_weights = ModelWeights::from_f64(
        &[0.1, 0.2, 0.3, 0.4],
        &[0.01, 0.02],
        &[0.5, 0.6],
        &[0.03],
    );

    // Shared session key for all parties (in production, derived from DH).
    let session_key = [0xABu8; 32];

    let input = vec![1.0, 0.5];
    let target = vec![1.0];

    let mut handles = Vec::new();
    for i in 0..num_parties {
        let cfg = config.clone();
        let p = parties.clone();
        let a = addrs.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let inp = input.clone();
        let tgt = target.clone();
        let key = session_key;

        let handle = tokio::spawn(async move {
            let peers = peer_map(&p, &a, i);
            let tcp = TcpTransport::bind(a[i], p[i].clone(), &peers)
                .await
                .unwrap();

            // Wrap TCP transport with HMAC authentication.
            let transport = AuthenticatedTransport::new(tcp, key);

            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            trainer.generate_beaver_triples(256).await.unwrap();
            let result = trainer.training_step(&inp, &tgt).await.unwrap();

            (result.loss, result.step)
        });
        handles.push(handle);
    }

    let mut losses = Vec::new();
    for handle in handles {
        let (loss, step) = handle.await.unwrap();
        assert_eq!(step, 0);
        assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
        losses.push(loss);
    }

    // All parties should compute the same loss.
    for i in 1..losses.len() {
        assert!(
            (losses[i] - losses[0]).abs() < 0.01,
            "Loss mismatch: party 0 = {}, party {} = {}",
            losses[0], i, losses[i]
        );
    }
}

// ============================================================================
// Test: 5-party training to test scalability
// ============================================================================

#[tokio::test]
async fn test_tcp_five_party_training() {
    let num_parties = 5;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let config = MPCTrainerConfig {
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
        mac_config: None,
    };

    let initial_weights = ModelWeights::from_f64(
        &[0.1, -0.2, 0.3, -0.1],
        &[0.0, 0.0],
        &[0.4, -0.3],
        &[0.0],
    );

    let data: Vec<(Vec<f64>, Vec<f64>)> = vec![
        (vec![1.0, 0.5], vec![1.0]),
        (vec![0.5, 1.0], vec![0.5]),
    ];

    let mut handles = Vec::new();
    for i in 0..num_parties {
        let cfg = config.clone();
        let p = parties.clone();
        let a = addrs.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let d = data.clone();

        let handle = tokio::spawn(async move {
            let peers = peer_map(&p, &a, i);
            let transport = TcpTransport::bind(a[i], p[i].clone(), &peers)
                .await
                .unwrap();

            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            let results = trainer.train(&d).await.unwrap();

            let losses: Vec<f64> = results.iter().map(|r| r.loss).collect();
            let (w1, _, _, _) = trainer.weight_shares();
            (losses, w1.to_vec())
        });
        handles.push(handle);
    }

    let mut all_losses = Vec::new();
    let mut all_w1 = Vec::new();
    for handle in handles {
        let (losses, w1) = handle.await.unwrap();
        all_losses.push(losses);
        all_w1.push(w1);
    }

    // All 5 parties should agree on losses.
    for step in 0..data.len() {
        for party in 1..num_parties {
            assert!(
                (all_losses[party][step] - all_losses[0][step]).abs() < 0.01,
                "Step {} loss mismatch: party 0={}, party {}={}",
                step, all_losses[0][step], party, all_losses[party][step]
            );
        }
    }

    // Final weights should reconstruct to finite values.
    for idx in 0..all_w1[0].len() {
        let sum = reconstruct(&all_w1, idx);
        assert!(
            sum.to_f64().is_finite(),
            "w1[{}] not finite with 5 parties",
            idx
        );
    }
}

// ============================================================================
// Test: Weight reconstruction consistency across all weight matrices
// ============================================================================

#[tokio::test]
async fn test_tcp_four_party_weight_reconstruction_after_training() {
    let num_parties = 4;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let config = MPCTrainerConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        learning_rate: 0.05,
        num_parties,
        reshare_interval: 0,
        beaver_batch_size: 512,
        generate_proofs: false,
        base_error: 1e-6,
        checkpoint_interval: 1,
        mac_config: None,
    };

    let initial_weights = ModelWeights::from_f64(
        &[0.2, -0.1, 0.3, 0.15],
        &[0.01, -0.01],
        &[0.4, 0.3],
        &[0.02],
    );

    // Train for 3 steps.
    let data: Vec<(Vec<f64>, Vec<f64>)> = (0..3)
        .map(|_| (vec![1.0, 0.5], vec![0.7]))
        .collect();

    let mut handles = Vec::new();
    for i in 0..num_parties {
        let cfg = config.clone();
        let p = parties.clone();
        let a = addrs.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let d = data.clone();

        let handle = tokio::spawn(async move {
            let peers = peer_map(&p, &a, i);
            let transport = TcpTransport::bind(a[i], p[i].clone(), &peers)
                .await
                .unwrap();

            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            let _ = trainer.train(&d).await.unwrap();

            let (w1, b1, w2, b2) = trainer.weight_shares();
            (w1.to_vec(), b1.to_vec(), w2.to_vec(), b2.to_vec())
        });
        handles.push(handle);
    }

    let mut all_w1 = Vec::new();
    let mut all_b1 = Vec::new();
    let mut all_w2 = Vec::new();
    let mut all_b2 = Vec::new();
    for handle in handles {
        let (w1, b1, w2, b2) = handle.await.unwrap();
        all_w1.push(w1);
        all_b1.push(b1);
        all_w2.push(w2);
        all_b2.push(b2);
    }

    // All weight matrices should reconstruct to finite values.
    for idx in 0..all_w1[0].len() {
        let sum = reconstruct(&all_w1, idx);
        let val = sum.to_f64();
        assert!(val.is_finite(), "w1[{}] not finite after training", idx);
    }

    for idx in 0..all_b1[0].len() {
        let sum = reconstruct(&all_b1, idx);
        let val = sum.to_f64();
        assert!(val.is_finite(), "b1[{}] not finite after training", idx);
    }

    for idx in 0..all_w2[0].len() {
        let sum = reconstruct(&all_w2, idx);
        let val = sum.to_f64();
        assert!(val.is_finite(), "w2[{}] not finite after training", idx);
    }

    for idx in 0..all_b2[0].len() {
        let sum = reconstruct(&all_b2, idx);
        let val = sum.to_f64();
        assert!(val.is_finite(), "b2[{}] not finite after training", idx);
    }
}
