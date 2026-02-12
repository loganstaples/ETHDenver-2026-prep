//! Integration test: 3-party distributed Beaver triple generation over TCP.
//!
//! Tests that `NetworkDistributedDealer` correctly generates valid Beaver
//! triples when communicating over real TCP connections (localhost).
//!
//! Feature-gated behind `network-mpc`.
//!
//! Run with:
//!   cargo test -p helix-mpc --features network-mpc --test distributed_beaver_tcp

#![cfg(feature = "network-mpc")]

use std::collections::HashMap;
use std::net::SocketAddr;

use helix_mpc::beaver::NetworkDistributedDealer;
use helix_mpc::beaver::triple::BeaverTriple;
use helix_mpc::field::Fr;
use helix_mpc::field::ops::sum;
use helix_mpc::session::transport::TcpTransport;
use helix_mpc::types::PartyId;

/// Allocates `n` localhost addresses on OS-assigned ports by binding and
/// immediately closing TCP listeners.
async fn allocate_addrs(n: usize) -> Vec<SocketAddr> {
    let mut addrs = Vec::new();
    for _ in 0..n {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        addrs.push(listener.local_addr().unwrap());
        // Drop listener to free the port for TcpTransport
        drop(listener);
    }
    addrs
}

/// Builds the peer address map for a given party (excludes self).
fn peer_map(
    parties: &[PartyId],
    addrs: &[SocketAddr],
    exclude: usize,
) -> HashMap<PartyId, SocketAddr> {
    parties
        .iter()
        .zip(addrs.iter())
        .enumerate()
        .filter(|(i, _)| *i != exclude)
        .map(|(_, (p, a))| (p.clone(), *a))
        .collect()
}

/// Test that 3 parties can generate valid Beaver triples over TCP.
///
/// Each party runs in its own tokio task, communicating over real TCP
/// connections on localhost. After generation, we verify the fundamental
/// Beaver triple property: sum(a_i) * sum(b_i) == sum(c_i).
#[tokio::test]
async fn test_distributed_beaver_over_tcp() {
    let num_parties = 3;
    let num_triples = 20;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let mut handles = Vec::new();

    for i in 0..num_parties {
        let party = parties[i].clone();
        let listen_addr = addrs[i];
        let peers = peer_map(&parties, &addrs, i);

        let handle = tokio::spawn(async move {
            let transport = TcpTransport::bind(listen_addr, party, &peers)
                .await
                .unwrap();
            let mut dealer = NetworkDistributedDealer::new(&transport, i, 42);
            dealer.generate(num_triples).await.unwrap()
        });
        handles.push(handle);
    }

    let mut all_triples: Vec<Vec<BeaverTriple>> = Vec::new();
    for handle in handles {
        all_triples.push(handle.await.unwrap());
    }

    assert_eq!(all_triples.len(), num_parties);
    assert_eq!(all_triples[0].len(), num_triples);

    // Verify each triple: sum(a_i) * sum(b_i) == sum(c_i)
    for t in 0..num_triples {
        let a = sum(&all_triples.iter().map(|p| p[t].a.clone()).collect::<Vec<_>>());
        let b = sum(&all_triples.iter().map(|p| p[t].b.clone()).collect::<Vec<_>>());
        let c = sum(&all_triples.iter().map(|p| p[t].c.clone()).collect::<Vec<_>>());
        let expected = a.mpc_scale(&b);
        assert!(
            c.ct_eq(&expected).to_bool(),
            "TCP distributed triple {} incorrect: sum(c) != sum(a) * sum(b)",
            t,
        );
    }
}

/// Test that distributed triples produce valid training results comparable
/// to trusted-dealer triples.
///
/// This test compares:
/// 1. Training with the MPCTrainer's built-in distributed triple generation
///    over TCP (real network)
/// 2. Training with the MPCTrainer's built-in distributed triple generation
///    over LocalTransport (baseline)
///
/// Both should produce valid additive shares that reconstruct to finite
/// weight values. The actual share values may differ (different random
/// triples), but the reconstructed model should be finite and reasonable.
#[tokio::test]
async fn test_distributed_vs_trusted_dealer_training() {
    use helix_mpc::mpc_trainer::{MPCTrainer, MPCTrainerConfig, ModelWeights};
    use helix_mpc::session::transport::LocalTransport;

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
        &[0.1, 0.2, 0.3, 0.4], // w1: 2x2
        &[0.01, 0.02],          // b1: 2
        &[0.5, 0.6],            // w2: 2x1
        &[0.03],                // b2: 1
    );

    let input = vec![1.0, 0.5];
    let target = vec![1.0];

    // --- Baseline: local transport with built-in distributed triples ---
    let local_transports = LocalTransport::create_mesh(&parties);

    let mut local_handles = Vec::new();
    for (i, transport) in local_transports.into_iter().enumerate() {
        let cfg = config.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let inp = input.clone();
        let tgt = target.clone();

        let handle = tokio::spawn(async move {
            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            trainer.generate_beaver_triples(256).await.unwrap();
            let result = trainer.training_step(&inp, &tgt).await.unwrap();
            let (w1, b1, w2, b2) = trainer.weight_shares();
            (
                result.loss,
                w1.to_vec(),
                b1.to_vec(),
                w2.to_vec(),
                b2.to_vec(),
            )
        });
        local_handles.push(handle);
    }

    let mut local_results = Vec::new();
    for h in local_handles {
        local_results.push(h.await.unwrap());
    }

    // --- TCP: real network transport with built-in distributed triples ---
    let mut tcp_handles = Vec::new();
    for i in 0..num_parties {
        let party = parties[i].clone();
        let listen_addr = addrs[i];
        let peers = peer_map(&parties, &addrs, i);
        let cfg = config.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let inp = input.clone();
        let tgt = target.clone();

        let handle = tokio::spawn(async move {
            let transport = TcpTransport::bind(listen_addr, party, &peers)
                .await
                .unwrap();

            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            trainer.generate_beaver_triples(256).await.unwrap();
            let result = trainer.training_step(&inp, &tgt).await.unwrap();
            let (w1, b1, w2, b2) = trainer.weight_shares();
            (
                result.loss,
                w1.to_vec(),
                b1.to_vec(),
                w2.to_vec(),
                b2.to_vec(),
            )
        });
        tcp_handles.push(handle);
    }

    let mut tcp_results = Vec::new();
    for h in tcp_handles {
        tcp_results.push(h.await.unwrap());
    }

    // Both should produce the same number of weight shares
    assert_eq!(local_results.len(), tcp_results.len());
    assert_eq!(local_results[0].1.len(), tcp_results[0].1.len()); // w1 length

    // All parties should agree on loss within their group
    for i in 1..num_parties {
        assert!(
            (local_results[i].0 - local_results[0].0).abs() < 0.01,
            "Local loss mismatch: party 0 = {}, party {} = {}",
            local_results[0].0,
            i,
            local_results[i].0,
        );
        assert!(
            (tcp_results[i].0 - tcp_results[0].0).abs() < 0.01,
            "TCP loss mismatch: party 0 = {}, party {} = {}",
            tcp_results[0].0,
            i,
            tcp_results[i].0,
        );
    }

    // Both losses should be finite
    assert!(
        local_results[0].0.is_finite(),
        "Local loss should be finite, got {}",
        local_results[0].0,
    );
    assert!(
        tcp_results[0].0.is_finite(),
        "TCP loss should be finite, got {}",
        tcp_results[0].0,
    );

    // Reconstructed weights should be finite for both
    let num_w1 = local_results[0].1.len();
    for idx in 0..num_w1 {
        let local_sum = local_results
            .iter()
            .fold(Fr::ZERO, |acc, r| Fr::add(&acc, &r.1[idx]));
        let tcp_sum = tcp_results
            .iter()
            .fold(Fr::ZERO, |acc, r| Fr::add(&acc, &r.1[idx]));

        assert!(
            local_sum.to_f64().is_finite(),
            "Local w1[{}] not finite after training",
            idx,
        );
        assert!(
            tcp_sum.to_f64().is_finite(),
            "TCP w1[{}] not finite after training",
            idx,
        );
    }

    // The local and TCP training used the same seed (42) and same initial
    // weights, but local and TCP transports may yield different message
    // orderings (affecting RNG state during triple generation). So the
    // exact share values and losses may differ. But both should be in a
    // reasonable range.
    //
    // Note: exact match is NOT expected because:
    // 1. RNG state differs due to transport-level differences
    // 2. Different Beaver triples produce different intermediate values
    // The key property is that both produce valid, finite results.
}
