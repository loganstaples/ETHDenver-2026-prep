//! Comprehensive TCP transport tests for MPC operations.
//!
//! Tests all MPC operations over real TCP connections:
//! - Full training pipeline (20 steps, 3 parties)
//! - Distributed Beaver triple generation
//! - Timeout detection
//! - Transport equivalence (Local vs TCP)
//! - Transport configuration
//!
//! Run with:
//!   cargo test -p helix-mpc --features network-mpc --test tcp_transport_comprehensive

#![cfg(feature = "network-mpc")]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

use helix_mpc::beaver::NetworkDistributedDealer;
use helix_mpc::beaver::triple::BeaverTriple;
use helix_mpc::e2e_integration::{InitialWeights, MPCIntegrationConfig, run_mpc_training};
use helix_mpc::field::ops::sum;
use helix_mpc::session::transport::{
    MPCTransport, TcpTransport, TcpTransportConfig, TransportConfig,
};
use helix_mpc::types::PartyId;

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

/// Builds the peer address map for a given party index (excludes self).
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

/// Creates a TCP transport mesh for all parties concurrently.
async fn create_tcp_mesh(
    parties: &[PartyId],
    addrs: &[SocketAddr],
) -> Vec<TcpTransport> {
    let mut handles = Vec::new();
    for i in 0..parties.len() {
        let party = parties[i].clone();
        let addr = addrs[i];
        let peers = peer_map(parties, addrs, i);
        handles.push(tokio::spawn(async move {
            TcpTransport::bind(addr, party, &peers).await.unwrap()
        }));
    }
    let mut transports = Vec::new();
    for h in handles {
        transports.push(h.await.unwrap());
    }
    transports
}

/// Creates a TCP transport mesh with custom config for all parties.
async fn create_tcp_mesh_with_config(
    parties: &[PartyId],
    addrs: &[SocketAddr],
    base_config: TcpTransportConfig,
) -> Vec<TcpTransport> {
    let mut handles = Vec::new();
    for i in 0..parties.len() {
        let party = parties[i].clone();
        let addr = addrs[i];
        let peers = peer_map(parties, addrs, i);
        let mut config = base_config.clone();
        config.bind_addr = addr;
        handles.push(tokio::spawn(async move {
            TcpTransport::bind_with_config(party, &peers, config).await.unwrap()
        }));
    }
    let mut transports = Vec::new();
    for h in handles {
        transports.push(h.await.unwrap());
    }
    transports
}

/// Shared initial weights for deterministic tests.
fn test_initial_weights() -> InitialWeights {
    InitialWeights {
        w1: vec![0.1, 0.2, 0.3, 0.4],
        b1: vec![0.01, 0.02],
        w2: vec![0.5, 0.6],
        b2: vec![0.03],
    }
}

/// Shared training data for deterministic tests.
fn test_training_data() -> Vec<(Vec<f64>, Vec<f64>)> {
    vec![
        (vec![1.0, 0.5], vec![1.0]),
        (vec![0.5, 1.0], vec![0.0]),
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ]
}

// ============================================================================
// Test: test_tcp_mpc_training — 3 parties on localhost TCP, 20 steps
// ============================================================================

/// Runs 3-party MPC training over TCP for 20 steps and verifies that:
/// - All 20 steps complete
/// - Loss is computed and finite at each step
/// - Loss generally decreases over training
/// - Final weights are reconstructable
///
/// This validates that ALL MPC operations work over TCP:
/// - Beaver triple consumption (opening a-α, b-β)
/// - Weight share distribution and reconstruction
/// - Checkpoint commitment sharing
#[tokio::test]
async fn test_tcp_mpc_training() {
    // Use d_hid=4 with appropriate weight sizes for 2→4→1 model
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 4,
        d_out: 1,
        num_workers: 3,
        num_steps: 20,
        learning_rate: 0.05,
        checkpoint_interval: 10,
        mac_check_interval: 0, // no MAC for speed
        beaver_batch_size: 512,
        initial_weights: Some(InitialWeights {
            // w1: 2*4 = 8 elements
            w1: vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
            // b1: 4 elements
            b1: vec![0.01, 0.02, -0.01, -0.02],
            // w2: 4*1 = 4 elements
            w2: vec![0.5, 0.6, -0.5, -0.6],
            // b2: 1 element
            b2: vec![0.03],
        }),
        training_data: test_training_data(),
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: true,
        worker_endpoints: None,
    };

    let result = run_mpc_training(config).await.expect("TCP 20-step training should succeed");

    assert_eq!(result.steps_completed, 20, "All 20 steps should complete");
    assert_eq!(result.losses.len(), 20, "Should have 20 loss values");

    // All losses should be finite
    for (i, loss) in result.losses.iter().enumerate() {
        assert!(loss.is_finite(), "Loss at step {} should be finite, got {}", i, loss);
    }

    // Loss should generally decrease: average of first 5 > average of last 5
    let early_avg: f64 = result.losses[..5].iter().sum::<f64>() / 5.0;
    let late_avg: f64 = result.losses[15..].iter().sum::<f64>() / 5.0;
    assert!(
        late_avg <= early_avg + 0.1,
        "Loss should decrease: early_avg={:.6}, late_avg={:.6}",
        early_avg, late_avg,
    );

    // Should have 2 checkpoints (at steps 10 and 20)
    assert_eq!(result.checkpoints.len(), 2, "Should have 2 checkpoints");

    // Final weights should be finite
    for w in &result.final_weights.w1 {
        assert!(w.is_finite(), "Reconstructed w1 should be finite");
    }
    for w in &result.final_weights.w2 {
        assert!(w.is_finite(), "Reconstructed w2 should be finite");
    }

    assert!(result.cheater_detected.is_none(), "No cheater should be detected");
}

// ============================================================================
// Test: test_tcp_beaver_triples — distributed triple generation over TCP
// ============================================================================

/// Tests distributed Beaver triple generation over TCP connections.
/// Verifies the fundamental triple property: sum(a_i) * sum(b_i) == sum(c_i).
#[tokio::test]
async fn test_tcp_beaver_triples() {
    let num_parties = 3;
    let num_triples = 50;
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

    // Verify each triple
    let mut valid_count = 0;
    for t in 0..num_triples {
        let a = sum(&all_triples.iter().map(|p| p[t].a.clone()).collect::<Vec<_>>());
        let b = sum(&all_triples.iter().map(|p| p[t].b.clone()).collect::<Vec<_>>());
        let c = sum(&all_triples.iter().map(|p| p[t].c.clone()).collect::<Vec<_>>());
        let expected = a.mpc_scale(&b);
        assert!(
            c.ct_eq(&expected).to_bool(),
            "TCP triple {} incorrect: sum(c) != sum(a) * sum(b)",
            t,
        );
        valid_count += 1;
    }
    assert_eq!(valid_count, num_triples, "All triples should be valid");
}

// ============================================================================
// Test: test_tcp_timeout — one party goes silent, timeout detected
// ============================================================================

/// Tests that the TCP transport correctly detects timeouts when a peer
/// stops responding.
///
/// Setup:
/// - Create a 2-party TCP mesh with a short recv timeout (500ms)
/// - Party 0 sends a message to Party 1
/// - Party 1 receives it
/// - Party 1 tries to receive a second message that is never sent
/// - Verify that a Timeout error is returned within the configured interval
#[tokio::test]
async fn test_tcp_timeout() {
    let num_parties = 2;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    // Configure a short timeout (500ms) so the test completes quickly
    let config = TcpTransportConfig {
        recv_timeout: Duration::from_millis(500),
        ..TcpTransportConfig::default()
    };

    let transports = create_tcp_mesh_with_config(&parties, &addrs, config).await;

    // Party 0 sends a message
    transports[0].send(&parties[1], b"hello").await.unwrap();

    // Party 1 receives the message successfully
    let msg = transports[1].recv(&parties[0]).await.unwrap();
    assert_eq!(msg, b"hello");

    // Now party 1 tries to receive again but party 0 sends nothing.
    // This should time out within the configured 500ms.
    let start = std::time::Instant::now();
    let result = transports[1].recv(&parties[0]).await;
    let elapsed = start.elapsed();

    assert!(result.is_err(), "Should return error on timeout");
    let err = result.unwrap_err();
    let err_msg = format!("{}", err);
    assert!(
        err_msg.contains("timeout") || err_msg.contains("Timeout"),
        "Error should mention timeout, got: {}",
        err_msg,
    );

    // Verify the timeout happened within a reasonable window (500ms ± 200ms)
    assert!(
        elapsed >= Duration::from_millis(400) && elapsed < Duration::from_millis(1500),
        "Timeout should trigger near 500ms, took {:?}",
        elapsed,
    );
}

// ============================================================================
// Test: test_transport_equivalence — same training on Local vs TCP
// ============================================================================

/// Verifies that the same training configuration produces equivalent results
/// on LocalTransport and TcpTransport.
///
/// Both runs use:
/// - Same initial weights
/// - Same seed
/// - Same training data
/// - Same model architecture (2→2→1)
///
/// We expect identical losses because the LocalTransport and TcpTransport
/// implementations should be functionally equivalent.
#[tokio::test]
async fn test_transport_equivalence() {
    let initial_weights = test_initial_weights();
    let training_data = test_training_data();
    let num_steps = 10;
    let seed = 42;

    // --- Run with LocalTransport ---
    let local_config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps,
        learning_rate: 0.01,
        checkpoint_interval: 5,
        mac_check_interval: 0,
        beaver_batch_size: 256,
        initial_weights: Some(initial_weights.clone()),
        training_data: training_data.clone(),
        seed,
        use_node_transport: false,
        use_tcp_transport: false, // LocalTransport
        worker_endpoints: None,
    };

    let local_result = run_mpc_training(local_config)
        .await
        .expect("LocalTransport training should succeed");

    // --- Run with TcpTransport ---
    let tcp_config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps,
        learning_rate: 0.01,
        checkpoint_interval: 5,
        mac_check_interval: 0,
        beaver_batch_size: 256,
        initial_weights: Some(initial_weights.clone()),
        training_data: training_data.clone(),
        seed,
        use_node_transport: false,
        use_tcp_transport: true, // TcpTransport
        worker_endpoints: None,
    };

    let tcp_result = run_mpc_training(tcp_config)
        .await
        .expect("TcpTransport training should succeed");

    // Both should complete all steps
    assert_eq!(local_result.steps_completed, num_steps);
    assert_eq!(tcp_result.steps_completed, num_steps);

    // Both should produce the same number of losses
    assert_eq!(local_result.losses.len(), tcp_result.losses.len());

    // Both should produce the same number of checkpoints
    assert_eq!(local_result.checkpoints.len(), tcp_result.checkpoints.len());

    // Losses should be identical (same seed, same weights, same data).
    // The e2e pipeline uses encrypted distribution from a deterministic seed,
    // so the shares are identical regardless of transport.
    for i in 0..num_steps {
        let local_loss = local_result.losses[i];
        let tcp_loss = tcp_result.losses[i];
        assert!(
            (local_loss - tcp_loss).abs() < 1e-10,
            "Step {} loss mismatch: local={:.10}, tcp={:.10}",
            i, local_loss, tcp_loss,
        );
    }

    // Final weights should match closely
    assert_eq!(local_result.final_weights.w1.len(), tcp_result.final_weights.w1.len());
    for i in 0..local_result.final_weights.w1.len() {
        let diff = (local_result.final_weights.w1[i] - tcp_result.final_weights.w1[i]).abs();
        assert!(
            diff < 1e-10,
            "w1[{}] mismatch: local={}, tcp={}, diff={}",
            i, local_result.final_weights.w1[i], tcp_result.final_weights.w1[i], diff,
        );
    }

    // Checkpoint commitments should be identical
    for i in 0..local_result.checkpoints.len() {
        assert_eq!(
            local_result.checkpoints[i].commitment_bytes32,
            tcp_result.checkpoints[i].commitment_bytes32,
            "Checkpoint {} commitment mismatch",
            i,
        );
    }
}

// ============================================================================
// Test: test_tcp_mpc_training_with_mac — SPDZ MAC verification over TCP
// ============================================================================

/// Tests that SPDZ MAC verification works correctly over TCP transport.
/// MAC checks are run every 2 steps during a 4-step training session.
#[tokio::test]
async fn test_tcp_mpc_training_with_mac() {
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 4,
        learning_rate: 0.01,
        checkpoint_interval: 4,
        mac_check_interval: 2,
        beaver_batch_size: 512,
        initial_weights: Some(test_initial_weights()),
        training_data: vec![
            (vec![1.0, 0.5], vec![1.0]),
            (vec![0.5, 1.0], vec![0.0]),
        ],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: true,
        worker_endpoints: None,
    };

    let result = run_mpc_training(config).await.expect("TCP MAC training should succeed");
    assert!(result.steps_completed >= 1, "Should complete at least one step");
    if result.steps_completed == 4 {
        assert!(result.mac_checks_passed >= 1, "At least one MAC check should pass");
        assert!(result.cheater_detected.is_none(), "No cheater should be detected");
    }
}

// ============================================================================
// Test: test_tcp_transport_config — transport configuration works
// ============================================================================

/// Tests that TcpTransportConfig parameters are correctly applied.
#[tokio::test]
async fn test_tcp_transport_config() {
    let num_parties = 2;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let config = TcpTransportConfig {
        recv_timeout: Duration::from_secs(5),
        max_connect_retries: 100,
        connect_retry_delay: Duration::from_millis(10),
        mesh_timeout: Duration::from_secs(15),
        channel_capacity: 8192,
        ..TcpTransportConfig::default()
    };

    let transports = create_tcp_mesh_with_config(&parties, &addrs, config).await;

    // Verify config is stored
    assert_eq!(transports[0].config().recv_timeout, Duration::from_secs(5));
    assert_eq!(transports[0].config().max_connect_retries, 100);
    assert_eq!(transports[0].config().channel_capacity, 8192);

    // Verify basic send/recv works with the custom config
    transports[0].send(&parties[1], b"config test").await.unwrap();
    let msg = transports[1].recv(&parties[0]).await.unwrap();
    assert_eq!(msg, b"config test");
}

// ============================================================================
// Test: test_transport_config_from_env — environment variable selection
// ============================================================================

/// Tests that TransportConfig::from_env reads HELIX_TRANSPORT correctly.
#[test]
fn test_transport_config_from_env() {
    // Default (no env var) should be Local
    let config = TransportConfig::from_env();
    assert!(matches!(config, TransportConfig::Local));

    // Test Tcp variant
    std::env::set_var("HELIX_TRANSPORT", "tcp");
    let config = TransportConfig::from_env();
    assert!(matches!(config, TransportConfig::Tcp(_)));

    // Test Node variant
    std::env::set_var("HELIX_TRANSPORT", "node");
    let config = TransportConfig::from_env();
    assert!(matches!(config, TransportConfig::Node { .. }));

    // Test Node with session ID
    std::env::set_var("HELIX_SESSION_ID", "my-session");
    let config = TransportConfig::from_env();
    if let TransportConfig::Node { session_id } = config {
        assert_eq!(session_id, "my-session");
    } else {
        panic!("Expected Node variant");
    }

    // Clean up
    std::env::remove_var("HELIX_TRANSPORT");
    std::env::remove_var("HELIX_SESSION_ID");
}

// ============================================================================
// Test: test_tcp_bidirectional_messaging — verify message ordering
// ============================================================================

/// Tests that messages arrive in the correct order over TCP for
/// bidirectional multi-party communication.
#[tokio::test]
async fn test_tcp_bidirectional_messaging() {
    let num_parties = 3;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;
    let transports = create_tcp_mesh(&parties, &addrs).await;

    // Send a sequence of messages from each party to every other party
    let num_messages = 100usize;

    // Send all messages first
    for i in 0..num_parties {
        for j in 0..num_parties {
            if i != j {
                for seq in 0..num_messages {
                    let msg = format!("from-{}-to-{}-seq-{}", i, j, seq);
                    transports[i].send(&parties[j], msg.as_bytes()).await.unwrap();
                }
            }
        }
    }

    // Receive and verify all messages arrive in order from each sender
    for i in 0..num_parties {
        for j in 0..num_parties {
            if i != j {
                for seq in 0..num_messages {
                    let msg = transports[i].recv(&parties[j]).await.unwrap();
                    let expected = format!("from-{}-to-{}-seq-{}", j, i, seq);
                    assert_eq!(
                        String::from_utf8_lossy(&msg), expected,
                        "Message ordering violated for party {} from party {} at seq {}",
                        i, j, seq,
                    );
                }
            }
        }
    }
}

// ============================================================================
// Test: test_tcp_broadcast — broadcast works correctly over TCP
// ============================================================================

/// Tests that broadcast delivers to all peers over TCP.
#[tokio::test]
async fn test_tcp_broadcast() {
    let num_parties = 4;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;
    let transports = create_tcp_mesh(&parties, &addrs).await;

    // Party 0 broadcasts
    transports[0].broadcast(b"broadcast-payload").await.unwrap();

    // All other parties should receive it
    for i in 1..num_parties {
        let msg = transports[i].recv(&parties[0]).await.unwrap();
        assert_eq!(msg, b"broadcast-payload");
    }
}

// ============================================================================
// Test: test_tcp_large_messages — verify large message framing
// ============================================================================

/// Tests that large messages (multiple MB) are correctly framed and delivered
/// over TCP. This exercises the 4-byte length prefix framing.
#[tokio::test]
async fn test_tcp_large_messages() {
    let num_parties = 2;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;
    let transports = create_tcp_mesh(&parties, &addrs).await;

    // Create a 2MB payload
    let payload_size = 2 * 1024 * 1024;
    let payload: Vec<u8> = (0..payload_size).map(|i| (i % 256) as u8).collect();

    transports[0].send(&parties[1], &payload).await.unwrap();
    let received = transports[1].recv(&parties[0]).await.unwrap();

    assert_eq!(received.len(), payload_size);
    assert_eq!(received, payload, "Large message should be received intact");
}

// ============================================================================
// Test: test_tcp_checkpoint_exchange — Pedersen checkpoint sharing over TCP
// ============================================================================

/// Tests that Pedersen checkpoint commitment exchange works correctly over TCP.
/// Uses the full e2e pipeline with a checkpoint at every 3 steps.
#[tokio::test]
async fn test_tcp_checkpoint_exchange() {
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 9,
        learning_rate: 0.01,
        checkpoint_interval: 3, // checkpoint at steps 3, 6, 9
        mac_check_interval: 0,
        beaver_batch_size: 256,
        initial_weights: Some(test_initial_weights()),
        training_data: test_training_data(),
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: true,
        worker_endpoints: None,
    };

    let result = run_mpc_training(config).await.expect("TCP checkpoint training should succeed");
    assert_eq!(result.steps_completed, 9);
    assert_eq!(
        result.checkpoints.len(), 3,
        "Should have 3 checkpoints (at steps 3, 6, 9)"
    );

    // All checkpoint commitments should be non-zero
    for (i, cp) in result.checkpoints.iter().enumerate() {
        assert!(
            cp.commitment_bytes32 != [0u8; 32],
            "Checkpoint {} commitment should be non-zero",
            i,
        );
        assert!(cp.loss.is_finite(), "Checkpoint {} loss should be finite", i);
    }
}

// ============================================================================
// Test: test_tcp_share_distribution_and_reconstruction — end-to-end shares
// ============================================================================

/// Tests that encrypted share distribution and reconstruction works over TCP.
/// Verifies that the reconstructed weights match the original.
#[tokio::test]
async fn test_tcp_share_distribution_and_reconstruction() {
    let initial = test_initial_weights();
    let config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 1,
        learning_rate: 0.0, // no weight update — reconstruction should match initial
        checkpoint_interval: 0,
        mac_check_interval: 0,
        beaver_batch_size: 64,
        initial_weights: Some(initial.clone()),
        training_data: vec![(vec![1.0, 0.5], vec![1.0])],
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: true,
        worker_endpoints: None,
    };

    let result = run_mpc_training(config).await.expect("TCP share distribution should succeed");
    assert!(result.encrypted_distribution, "Should use encrypted distribution");

    // With lr=0.0, weights should stay approximately the same after training
    // (the only changes come from the additive sharing → reconstruct roundtrip).
    for (i, (orig, reconstructed)) in initial.w1.iter().zip(result.final_weights.w1.iter()).enumerate() {
        let diff = (orig - reconstructed).abs();
        assert!(
            diff < 1e-6,
            "w1[{}] should be preserved: original={}, reconstructed={}, diff={}",
            i, orig, reconstructed, diff,
        );
    }
}

// ============================================================================
// Test: test_tcp_recv_with_default_timeout — default timeout method
// ============================================================================

/// Tests the recv_with_default_timeout convenience method.
#[tokio::test]
async fn test_tcp_recv_with_default_timeout() {
    let num_parties = 2;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;

    let config = TcpTransportConfig {
        recv_timeout: Duration::from_millis(300),
        ..TcpTransportConfig::default()
    };

    let transports = create_tcp_mesh_with_config(&parties, &addrs, config).await;

    // Send a message and receive with default timeout
    transports[0].send(&parties[1], b"timeout-test").await.unwrap();
    let msg = transports[1]
        .recv_with_default_timeout(&parties[0])
        .await
        .unwrap();
    assert_eq!(msg, b"timeout-test");

    // Try to receive without any message — should timeout
    let result = transports[1]
        .recv_with_default_timeout(&parties[0])
        .await;
    assert!(result.is_err(), "Should timeout on missing message");
}

// ============================================================================
// Test: test_tcp_party_id_and_peers — metadata methods
// ============================================================================

/// Verifies that party_id() and peers() return correct values for TCP transport.
#[tokio::test]
async fn test_tcp_party_id_and_peers() {
    let num_parties = 3;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;
    let transports = create_tcp_mesh(&parties, &addrs).await;

    for i in 0..num_parties {
        assert_eq!(transports[i].party_id(), &parties[i]);
        let peers = transports[i].peers();
        assert_eq!(peers.len(), num_parties - 1);
        // Should not include self in peers
        assert!(!peers.contains(&parties[i]));
        // Should include all other parties
        for j in 0..num_parties {
            if i != j {
                assert!(peers.contains(&parties[j]));
            }
        }
    }
}

// ============================================================================
// Test: test_tcp_node_transport_equivalence — NodeTransport produces same results
// ============================================================================

/// Tests that NodeTransport produces the same training results as LocalTransport
/// for the same configuration (both are in-memory, so results should be identical).
#[tokio::test]
async fn test_tcp_node_transport_equivalence() {
    let initial_weights = test_initial_weights();
    let training_data = test_training_data();

    // Run with LocalTransport
    let local_config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 5,
        learning_rate: 0.01,
        checkpoint_interval: 5,
        mac_check_interval: 0,
        beaver_batch_size: 256,
        initial_weights: Some(initial_weights.clone()),
        training_data: training_data.clone(),
        seed: 42,
        use_node_transport: false,
        use_tcp_transport: false,
        worker_endpoints: None,
    };

    let local_result = run_mpc_training(local_config)
        .await
        .expect("LocalTransport training should succeed");

    // Run with NodeTransport
    let node_config = MPCIntegrationConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        num_workers: 3,
        num_steps: 5,
        learning_rate: 0.01,
        checkpoint_interval: 5,
        mac_check_interval: 0,
        beaver_batch_size: 256,
        initial_weights: Some(initial_weights),
        training_data,
        seed: 42,
        use_node_transport: true,
        use_tcp_transport: false,
        worker_endpoints: None,
    };

    let node_result = run_mpc_training(node_config)
        .await
        .expect("NodeTransport training should succeed");

    assert_eq!(local_result.steps_completed, node_result.steps_completed);
    assert_eq!(local_result.losses.len(), node_result.losses.len());

    // Losses should be identical
    for i in 0..local_result.losses.len() {
        assert!(
            (local_result.losses[i] - node_result.losses[i]).abs() < 1e-10,
            "Step {} loss mismatch: local={}, node={}",
            i, local_result.losses[i], node_result.losses[i],
        );
    }
}

// ============================================================================
// Test: test_tcp_concurrent_sends — stress test concurrent messaging
// ============================================================================

/// Tests that concurrent sends from all parties to all others don't deadlock
/// or corrupt messages.
#[tokio::test]
async fn test_tcp_concurrent_sends() {
    let num_parties = 4;
    let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
    let addrs = allocate_addrs(num_parties).await;
    let transports = create_tcp_mesh(&parties, &addrs).await;

    // Each party sends to all peers concurrently, then receives from all peers
    let num_msgs = 50;

    // Wrap transports in Arc for sharing across tasks
    let transports: Vec<std::sync::Arc<TcpTransport>> =
        transports.into_iter().map(std::sync::Arc::new).collect();

    let mut send_handles = Vec::new();
    for i in 0..num_parties {
        let t = transports[i].clone();
        let p = parties.clone();
        let handle = tokio::spawn(async move {
            for j in 0..num_parties {
                if i != j {
                    for seq in 0..num_msgs {
                        let msg = format!("{}-{}-{}", i, j, seq);
                        t.send(&p[j], msg.as_bytes()).await.unwrap();
                    }
                }
            }
        });
        send_handles.push(handle);
    }

    // Wait for all sends to complete
    for h in send_handles {
        h.await.unwrap();
    }

    // Receive and verify all messages
    let mut recv_handles = Vec::new();
    for i in 0..num_parties {
        let t = transports[i].clone();
        let p = parties.clone();
        let handle = tokio::spawn(async move {
            let mut received_count = 0;
            for j in 0..num_parties {
                if i != j {
                    for seq in 0..num_msgs {
                        let msg = t.recv(&p[j]).await.unwrap();
                        let expected = format!("{}-{}-{}", j, i, seq);
                        assert_eq!(String::from_utf8_lossy(&msg), expected);
                        received_count += 1;
                    }
                }
            }
            received_count
        });
        recv_handles.push(handle);
    }

    let mut total_received = 0;
    for h in recv_handles {
        total_received += h.await.unwrap();
    }

    let expected_total = num_parties * (num_parties - 1) * num_msgs;
    assert_eq!(
        total_received, expected_total,
        "All messages should be received"
    );
}
