//! Integration test: 3-party MPC training over TCP transport.
//!
//! Tests the full MPCTrainer flow using TcpTransport (localhost).
//! Feature-gated behind `network-mpc`.
//!
//! Run with:
//!   cargo test -p helix-mpc --features network-mpc --test tcp_mpc_training

#![cfg(feature = "network-mpc")]

use std::collections::HashMap;
use std::net::SocketAddr;

use helix_mpc::mpc_trainer::{MPCTrainer, MPCTrainerConfig, ModelWeights};
use helix_mpc::session::transport::TcpTransport;
use helix_mpc::types::PartyId;

/// Allocates `n` localhost addresses on OS-assigned ports by binding and
/// immediately closing TCP listeners.
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

#[tokio::test]
async fn test_tcp_three_party_weight_sharing() {
    let num_parties = 3;
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

    let mut all_shares: Vec<(Vec<_>, Vec<_>, Vec<_>, Vec<_>)> = Vec::new();
    for handle in handles {
        all_shares.push(handle.await.unwrap());
    }

    // Verify w1 shares reconstruct to original weights.
    for idx in 0..d_hid * d_in {
        let sum = all_shares.iter().fold(helix_mpc::Fr::ZERO, |acc, s| {
            helix_mpc::Fr::add(&acc, &s.0[idx])
        });
        let expected = initial_weights.w1[idx].to_f64();
        let got = sum.to_f64();
        assert!(
            (got - expected).abs() < 0.001,
            "w1[{}]: expected {}, got {}",
            idx,
            expected,
            got
        );
    }
}

#[tokio::test]
async fn test_tcp_three_party_beaver_triples() {
    let num_parties = 3;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let config = MPCTrainerConfig::small(num_parties);
    let count = 5;

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
            assert_eq!(trainer.beaver_triples_remaining(), count);
            count
        });
        handles.push(handle);
    }

    for handle in handles {
        let triple_count = handle.await.unwrap();
        assert_eq!(triple_count, count);
    }
}

#[tokio::test]
async fn test_tcp_three_party_training_step() {
    let num_parties = 3;
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

    // All parties should compute the same loss (since intermediate values are reconstructed).
    for i in 1..losses.len() {
        assert!(
            (losses[i] - losses[0]).abs() < 0.01,
            "Loss mismatch: party 0 = {}, party {} = {}",
            losses[0],
            i,
            losses[i]
        );
    }
}

#[tokio::test]
async fn test_tcp_three_party_multi_step_with_resharing() {
    let num_parties = 3;
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
    };

    let initial_weights = ModelWeights::from_f64(
        &[0.5, -0.3, 0.2, 0.4],
        &[0.0, 0.0],
        &[0.6, -0.4],
        &[0.0],
    );

    // Same data point repeated — loss should decrease.
    let data: Vec<(Vec<f64>, Vec<f64>)> = (0..4)
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

    // All parties agree on losses.
    for step in 0..data.len() {
        for party in 1..num_parties {
            assert!(
                (all_losses[party][step] - all_losses[0][step]).abs() < 0.01,
                "Step {} loss mismatch: party 0 = {}, party {} = {}",
                step,
                all_losses[0][step],
                party,
                all_losses[party][step]
            );
        }
    }

    // Resharing should happen on step 1 (interval=2, step_number=2).
    for party in &all_reshare {
        assert!(!party[0], "Step 0 should not reshare");
        assert!(party[1], "Step 1 should reshare (interval=2)");
        assert!(!party[2], "Step 2 should not reshare");
        assert!(party[3], "Step 3 should reshare (interval=2)");
    }

    // Loss should decrease over training.
    assert!(
        all_losses[0].last().unwrap() < all_losses[0].first().unwrap(),
        "Loss should decrease: first={}, last={}",
        all_losses[0].first().unwrap(),
        all_losses[0].last().unwrap()
    );

    // Final weights should reconstruct to finite values.
    for idx in 0..all_final_w1[0].len() {
        let sum = all_final_w1.iter().fold(helix_mpc::Fr::ZERO, |acc, w| {
            helix_mpc::Fr::add(&acc, &w[idx])
        });
        assert!(
            sum.to_f64().is_finite(),
            "w1[{}] not finite after training + resharing",
            idx
        );
    }
}
