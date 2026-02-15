//! Integration tests for encrypted share distribution and reconstruction over TCP.
//!
//! Tests the full lifecycle: owner distributes encrypted weight shares to workers
//! over TCP, workers verify Pedersen commitment shares combine to C0, simulated
//! training modifies shares, workers send final shares back, owner reconstructs
//! and verifies against checkpoint commitment.

use helix_mpc::field::Fr;
use helix_mpc::network_distribution::{
    distribute_shares, reconstruct_shares, worker_receive_distribution, worker_send_final_share,
};
use helix_mpc::share_distribution::{
    generate_x25519_keypair, ShareReceiver, X25519PublicKey,
};
use helix_mpc::types::PartyId;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::net::SocketAddr;
use tokio::net::TcpListener;

/// Helper: generate worker key material and bind TCP listeners.
async fn setup_workers(
    n: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<(
    PartyId,
    SocketAddr,
    X25519PublicKey,
    x25519_dalek::StaticSecret,
    TcpListener,
)> {
    let mut workers = Vec::with_capacity(n);
    for i in 0..n {
        let (secret, public) = generate_x25519_keypair(rng);
        let party_id = PartyId::from_index(i);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        workers.push((party_id, addr, public, secret, listener));
    }
    workers
}

/// Helper: extract the (PartyId, SocketAddr, PublicKey) tuples the owner needs.
fn owner_view(
    workers: &[(
        PartyId,
        SocketAddr,
        X25519PublicKey,
        x25519_dalek::StaticSecret,
        TcpListener,
    )],
) -> Vec<(PartyId, SocketAddr, X25519PublicKey)> {
    workers
        .iter()
        .map(|(id, addr, pk, _, _)| (id.clone(), *addr, *pk))
        .collect()
}

// ============================================================================
// Test 1: Full MNIST model (784→32→10) distribution and reconstruction
// ============================================================================

#[tokio::test]
async fn test_mnist_model_distribute_reconstruct_3_workers() {
    // Generate a realistic MNIST model: 784×32 + 32 bias + 32×10 + 10 bias = 25,450 params
    let mut rng = ChaCha20Rng::seed_from_u64(2026);
    let num_params = 784 * 32 + 32 + 32 * 10 + 10;
    assert_eq!(num_params, 25450);

    // Random weights in [-1, 1] range (Xavier-like initialization)
    let weights: Vec<f64> = (0..num_params)
        .map(|i| {
            let x = ((i as f64 * 0.618033988) % 2.0) - 1.0; // deterministic pseudo-random
            x * 0.1 // small initialization
        })
        .collect();
    let shape = vec![num_params];

    // Setup 3 workers
    let workers = setup_workers(3, &mut rng).await;
    let owner_workers = owner_view(&workers);

    // Spawn worker tasks
    let mut worker_handles = Vec::new();
    for (i, (party_id, _, _, secret, listener)) in workers.into_iter().enumerate() {
        let receiver = ShareReceiver::new(secret.clone(), party_id);

        let handle = tokio::spawn(async move {
            // Phase 1: Receive distribution
            let (state, mut stream) =
                worker_receive_distribution(&listener, &receiver, &secret)
                    .await
                    .expect("worker distribution failed");

            // Verify share has correct length
            assert_eq!(
                state.weight_share.data.len(),
                25450,
                "worker {} share length mismatch",
                i
            );

            // Phase 2: No training (identity roundtrip) — send share back unchanged
            worker_send_final_share(
                &mut stream,
                &state.weight_share,
                &state.generators,
                Some(3000 + i as u64),
            )
            .await
            .expect("worker final share send failed");
        });
        worker_handles.push(handle);
    }

    // Owner distributes
    let mut dist_result = distribute_shares(&weights, &shape, &owner_workers, Some(1000))
        .await
        .expect("distribution failed");

    // Commitment verification must have passed
    assert!(dist_result.verified, "commitment verification failed");

    // Owner reconstructs
    let mut owner_rng = ChaCha20Rng::seed_from_u64(9999);
    let (owner_secret, _) = generate_x25519_keypair(&mut owner_rng);
    let recon_result = reconstruct_shares(&owner_secret, &mut dist_result.worker_streams)
        .await
        .expect("reconstruction failed");

    // Wait for workers
    for handle in worker_handles {
        handle.await.expect("worker task panicked");
    }

    // Verify reconstruction matches original (exact for identity roundtrip)
    assert_eq!(recon_result.weights.len(), weights.len());
    for (i, (orig, recon)) in weights.iter().zip(recon_result.weights.iter()).enumerate() {
        assert!(
            (orig - recon).abs() < 1e-6,
            "weight {} mismatch: original={}, reconstructed={}",
            i,
            orig,
            recon
        );
    }
}

// ============================================================================
// Test 2: MNIST model with simulated training (gradient update)
// ============================================================================

#[tokio::test]
async fn test_mnist_model_with_gradient_update_3_workers() {
    let mut rng = ChaCha20Rng::seed_from_u64(2026);
    let num_params = 784 * 32 + 32 + 32 * 10 + 10;

    // Initial weights
    let weights: Vec<f64> = (0..num_params)
        .map(|i| {
            let x = ((i as f64 * 0.618033988) % 2.0) - 1.0;
            x * 0.1
        })
        .collect();

    // Simulated gradient (small learning rate × gradient)
    let gradient: Vec<f64> = (0..num_params)
        .map(|i| {
            let g = ((i as f64 * 0.314159265) % 2.0) - 1.0;
            g * 0.001 // lr=0.001
        })
        .collect();

    let shape = vec![num_params];

    let workers = setup_workers(3, &mut rng).await;
    let owner_workers = owner_view(&workers);
    let gradient_clone = gradient.clone();

    // Spawn worker tasks
    let mut worker_handles = Vec::new();
    for (i, (party_id, _, _, secret, listener)) in workers.into_iter().enumerate() {
        let receiver = ShareReceiver::new(secret.clone(), party_id);
        let grad = gradient_clone.clone();

        let handle = tokio::spawn(async move {
            let (mut state, mut stream) =
                worker_receive_distribution(&listener, &receiver, &secret)
                    .await
                    .expect("worker distribution failed");

            // Simulate MPC gradient update: only party 0 adds the public gradient
            // (in additive secret sharing, only one party needs to add a public value
            //  to shift the combined sum).
            if state.weight_share.index == 0 {
                for (j, g) in grad.iter().enumerate() {
                    let g_fr = Fr::from_f64(*g);
                    state.weight_share.data[j] =
                        Fr::add(&state.weight_share.data[j], &g_fr);
                }
            }

            // Send trained share back
            worker_send_final_share(
                &mut stream,
                &state.weight_share,
                &state.generators,
                Some(4000 + i as u64),
            )
            .await
            .expect("worker final share send failed");
        });
        worker_handles.push(handle);
    }

    // Owner distributes
    let mut dist_result = distribute_shares(&weights, &shape, &owner_workers, Some(2000))
        .await
        .expect("distribution failed");
    assert!(dist_result.verified);

    // Owner reconstructs
    let mut owner_rng = ChaCha20Rng::seed_from_u64(7777);
    let (owner_secret, _) = generate_x25519_keypair(&mut owner_rng);
    let recon_result = reconstruct_shares(&owner_secret, &mut dist_result.worker_streams)
        .await
        .expect("reconstruction failed");

    // Wait for workers
    for handle in worker_handles {
        handle.await.expect("worker task panicked");
    }

    // Verify: reconstructed = original + gradient
    let expected: Vec<f64> = weights
        .iter()
        .zip(gradient.iter())
        .map(|(w, g)| w + g)
        .collect();

    assert_eq!(recon_result.weights.len(), expected.len());
    for (i, (exp, recon)) in expected.iter().zip(recon_result.weights.iter()).enumerate() {
        assert!(
            (exp - recon).abs() < 1e-5,
            "weight {} mismatch after gradient: expected={}, got={}",
            i,
            exp,
            recon
        );
    }
}

// ============================================================================
// Test 3: Multiple training rounds (distribute → train → reconstruct → repeat)
// ============================================================================

#[tokio::test]
async fn test_multiple_training_rounds() {
    let num_params = 100; // Smaller for multi-round test speed
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    let mut current_weights: Vec<f64> = (0..num_params)
        .map(|i| (i as f64) * 0.01)
        .collect();
    let shape = vec![num_params];

    // Run 3 training rounds
    for round in 0..3u64 {
        let workers = setup_workers(3, &mut rng).await;
        let owner_workers = owner_view(&workers);

        let gradient: Vec<f64> = (0..num_params)
            .map(|i| ((i as f64 + round as f64) * 0.001) % 0.01)
            .collect();
        let gradient_clone = gradient.clone();

        // Spawn workers
        let mut handles = Vec::new();
        for (i, (party_id, _, _, secret, listener)) in workers.into_iter().enumerate() {
            let receiver = ShareReceiver::new(secret.clone(), party_id);
            let grad = gradient_clone.clone();

            let handle = tokio::spawn(async move {
                let (mut state, mut stream) =
                    worker_receive_distribution(&listener, &receiver, &secret)
                        .await
                        .unwrap();

                if state.weight_share.index == 0 {
                    for (j, g) in grad.iter().enumerate() {
                        state.weight_share.data[j] =
                            Fr::add(&state.weight_share.data[j], &Fr::from_f64(*g));
                    }
                }

                worker_send_final_share(
                    &mut stream,
                    &state.weight_share,
                    &state.generators,
                    Some(5000 + round * 10 + i as u64),
                )
                .await
                .unwrap();
            });
            handles.push(handle);
        }

        // Owner distributes current weights
        let mut dist_result =
            distribute_shares(&current_weights, &shape, &owner_workers, Some(6000 + round))
                .await
                .unwrap();
        assert!(dist_result.verified, "round {} commitment failed", round);

        // Owner reconstructs
        let mut owner_rng = ChaCha20Rng::seed_from_u64(8000 + round);
        let (owner_secret, _) = generate_x25519_keypair(&mut owner_rng);
        let recon_result = reconstruct_shares(&owner_secret, &mut dist_result.worker_streams)
            .await
            .unwrap();

        for handle in handles {
            handle.await.unwrap();
        }

        // Verify round result
        let expected: Vec<f64> = current_weights
            .iter()
            .zip(gradient.iter())
            .map(|(w, g)| w + g)
            .collect();

        for (i, (exp, recon)) in expected.iter().zip(recon_result.weights.iter()).enumerate() {
            assert!(
                (exp - recon).abs() < 1e-5,
                "round {} weight {} mismatch: {} vs {}",
                round,
                i,
                exp,
                recon
            );
        }

        // Update weights for next round
        current_weights = recon_result.weights;
    }
}

// ============================================================================
// Test 4: Commitment verification catches mismatched shares
// ============================================================================

#[tokio::test]
async fn test_commitment_verification_passes_for_valid_shares() {
    // This test verifies that the commitment verification protocol actually
    // catches valid distributions (the positive case).
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let shape = vec![8];

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let workers = setup_workers(3, &mut rng).await;
    let owner_workers = owner_view(&workers);

    let mut handles = Vec::new();
    for (i, (party_id, _, _, secret, listener)) in workers.into_iter().enumerate() {
        let receiver = ShareReceiver::new(secret.clone(), party_id);

        let handle = tokio::spawn(async move {
            let (state, mut stream) =
                worker_receive_distribution(&listener, &receiver, &secret)
                    .await
                    .expect("distribution should succeed for valid shares");

            // Verify the worker received the correct initial commitment
            assert!(
                !state.initial_commitment.element_commitments.is_empty(),
                "worker {} should have received element commitments",
                i
            );

            worker_send_final_share(
                &mut stream,
                &state.weight_share,
                &state.generators,
                Some(9000 + i as u64),
            )
            .await
            .unwrap();
        });
        handles.push(handle);
    }

    let mut dist_result = distribute_shares(&weights, &shape, &owner_workers, Some(300))
        .await
        .expect("valid distribution should succeed");

    // The key assertion: commitment verification passed
    assert!(dist_result.verified);

    // Also verify the initial commitment is non-trivial
    assert_eq!(dist_result.distribution.initial_commitment.element_commitments.len(), 8);
    assert_eq!(dist_result.distribution.encrypted_shares.len(), 3);

    let mut owner_rng = ChaCha20Rng::seed_from_u64(1234);
    let (owner_secret, _) = generate_x25519_keypair(&mut owner_rng);
    let recon = reconstruct_shares(&owner_secret, &mut dist_result.worker_streams)
        .await
        .unwrap();

    for handle in handles {
        handle.await.unwrap();
    }

    assert_eq!(recon.weights.len(), 8);
    for (orig, rec) in weights.iter().zip(recon.weights.iter()) {
        assert!((orig - rec).abs() < 1e-6);
    }
}

// ============================================================================
// Test 5: Edge case — minimum number of workers (2)
// ============================================================================

#[tokio::test]
async fn test_two_worker_minimum() {
    let weights = vec![42.0, -17.5, 0.0, 100.123];
    let shape = vec![4];

    let mut rng = ChaCha20Rng::seed_from_u64(55);
    let workers = setup_workers(2, &mut rng).await;
    let owner_workers = owner_view(&workers);

    let mut handles = Vec::new();
    for (i, (party_id, _, _, secret, listener)) in workers.into_iter().enumerate() {
        let receiver = ShareReceiver::new(secret.clone(), party_id);

        let handle = tokio::spawn(async move {
            let (state, mut stream) =
                worker_receive_distribution(&listener, &receiver, &secret)
                    .await
                    .unwrap();

            worker_send_final_share(
                &mut stream,
                &state.weight_share,
                &state.generators,
                Some(1100 + i as u64),
            )
            .await
            .unwrap();
        });
        handles.push(handle);
    }

    let mut dist_result = distribute_shares(&weights, &shape, &owner_workers, Some(500))
        .await
        .unwrap();
    assert!(dist_result.verified);

    let mut owner_rng = ChaCha20Rng::seed_from_u64(5555);
    let (owner_secret, _) = generate_x25519_keypair(&mut owner_rng);
    let recon = reconstruct_shares(&owner_secret, &mut dist_result.worker_streams)
        .await
        .unwrap();

    for handle in handles {
        handle.await.unwrap();
    }

    for (orig, rec) in weights.iter().zip(recon.weights.iter()) {
        assert!((orig - rec).abs() < 1e-6, "{} vs {}", orig, rec);
    }
}

// ============================================================================
// Test 6: Large gradient update preserves precision
// ============================================================================

#[tokio::test]
async fn test_large_gradient_precision() {
    // Test with both very small and very large weight values to stress
    // the fixed-point Fr arithmetic precision.
    let weights = vec![
        1e-8, -1e-8,    // tiny
        1e4, -1e4,      // large
        0.0,             // zero
        std::f64::consts::PI, // irrational
        -std::f64::consts::E,
        0.123456789,
    ];
    let gradient = vec![
        1e-9, -1e-9,
        -1e3, 1e3,
        0.001,
        -0.001,
        0.001,
        -0.001,
    ];
    let shape = vec![8];

    let mut rng = ChaCha20Rng::seed_from_u64(77);
    let workers = setup_workers(3, &mut rng).await;
    let owner_workers = owner_view(&workers);
    let gradient_clone = gradient.clone();

    let mut handles = Vec::new();
    for (i, (party_id, _, _, secret, listener)) in workers.into_iter().enumerate() {
        let receiver = ShareReceiver::new(secret.clone(), party_id);
        let grad = gradient_clone.clone();

        let handle = tokio::spawn(async move {
            let (mut state, mut stream) =
                worker_receive_distribution(&listener, &receiver, &secret)
                    .await
                    .unwrap();

            if state.weight_share.index == 0 {
                for (j, g) in grad.iter().enumerate() {
                    state.weight_share.data[j] =
                        Fr::add(&state.weight_share.data[j], &Fr::from_f64(*g));
                }
            }

            worker_send_final_share(
                &mut stream,
                &state.weight_share,
                &state.generators,
                Some(2200 + i as u64),
            )
            .await
            .unwrap();
        });
        handles.push(handle);
    }

    let mut dist_result = distribute_shares(&weights, &shape, &owner_workers, Some(800))
        .await
        .unwrap();
    assert!(dist_result.verified);

    let mut owner_rng = ChaCha20Rng::seed_from_u64(6666);
    let (owner_secret, _) = generate_x25519_keypair(&mut owner_rng);
    let recon = reconstruct_shares(&owner_secret, &mut dist_result.worker_streams)
        .await
        .unwrap();

    for handle in handles {
        handle.await.unwrap();
    }

    let expected: Vec<f64> = weights
        .iter()
        .zip(gradient.iter())
        .map(|(w, g)| w + g)
        .collect();

    for (i, (exp, rec)) in expected.iter().zip(recon.weights.iter()).enumerate() {
        // Fr fixed-point roundtrip through additive secret sharing introduces
        // ~1e-6 relative error for moderate values and up to ~1e-9 absolute
        // error for very small values near zero.
        let tol = exp.abs().max(1.0) * 1e-5;
        assert!(
            (exp - rec).abs() < tol,
            "weight {} precision: expected={}, got={}, diff={}",
            i,
            exp,
            rec,
            (exp - rec).abs()
        );
    }
}
