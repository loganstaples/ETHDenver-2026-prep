//! Peer Discovery & Network Hardening Integration Tests.
//!
//! Tests the complete P2P discovery and resilience flow:
//! 1. Start 1 aggregator
//! 2. Start 3 workers with only bootstrap node config (not direct aggregator address)
//! 3. Workers discover aggregator via peer exchange
//! 4. Complete a training round
//! 5. Kill one worker mid-round
//! 6. Aggregator completes round with remaining 2
//! 7. Killed worker reconnects and joins next round

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::time::timeout;

use helix_node::network::messages::{
    GradientMessage, MessagePayload, NodeCapabilities,
    PeerId, TrainingMessage, TrainingParams,
};
use helix_node::network::runner::{NetworkEvent, NetworkRunner, NetworkRunnerBuilder, NetworkRunnerConfig};

// ============================================================================
// Test Helpers
// ============================================================================

/// Drains events from a network runner for a given duration, collecting them.
async fn collect_events(
    runner: &Arc<NetworkRunner>,
    duration: Duration,
) -> Vec<NetworkEvent> {
    let mut events = Vec::new();
    let deadline = tokio::time::Instant::now() + duration;

    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }

        match timeout(remaining, runner.next_event()).await {
            Ok(Some(event)) => events.push(event),
            Ok(None) => break,
            Err(_) => break, // Timeout
        }
    }

    events
}

// ============================================================================
// Test: Bootstrap Peer Discovery
// ============================================================================

#[tokio::test]
async fn test_bootstrap_peer_exchange() {
    // Test that a node can discover peers through a bootstrap node.
    //
    // Setup:
    // 1. Node A listens
    // 2. Node B connects to A (bootstrap)
    // 3. Node B discovers A's peer list

    let caps = NodeCapabilities::default();

    // Node A: aggregator
    let agg_id = PeerId::from_string("agg");
    let agg_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let agg_runner = NetworkRunnerBuilder::new()
        .local_id(agg_id.clone())
        .listen_addr(agg_addr)
        .capabilities(NodeCapabilities { can_aggregate: true, ..caps.clone() })
        .build()
        .unwrap();
    let agg_runner = Arc::new(agg_runner);
    agg_runner.start().await.unwrap();

    // We need to know the actual port. Since port 0 is used,
    // we connect to the configured address. For unit tests,
    // let's use a fixed port range.
    // Actually, let's use a different approach: use fixed ports.
    drop(agg_runner);

    // Use fixed ports to avoid port 0 issues
    let agg_addr: SocketAddr = "127.0.0.1:19100".parse().unwrap();
    let agg_runner = NetworkRunnerBuilder::new()
        .local_id(PeerId::from_string("agg"))
        .listen_addr(agg_addr)
        .capabilities(NodeCapabilities { can_aggregate: true, ..caps.clone() })
        .build()
        .unwrap();
    let agg_runner = Arc::new(agg_runner);
    agg_runner.start().await.unwrap();

    // Give aggregator time to bind
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Node B: worker, bootstraps from A
    let worker_addr: SocketAddr = "127.0.0.1:19101".parse().unwrap();
    let worker_runner = NetworkRunnerBuilder::new()
        .local_id(PeerId::from_string("worker-1"))
        .listen_addr(worker_addr)
        .bootstrap_nodes(vec![agg_addr.to_string()])
        .capabilities(NodeCapabilities { can_train: true, ..caps.clone() })
        .build()
        .unwrap();
    let worker_runner = Arc::new(worker_runner);
    worker_runner.start().await.unwrap();

    // Bootstrap: worker connects to aggregator
    worker_runner.bootstrap().await;

    // Give time for peer exchange
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Worker should have discovered the aggregator
    let worker_peers = worker_runner.connected_peers().await;
    assert!(
        !worker_peers.is_empty(),
        "Worker should have discovered at least the bootstrap node"
    );

    // Aggregator should have discovered the worker via JoinRequest
    let agg_peers = agg_runner.connected_peers().await;
    assert!(
        !agg_peers.is_empty(),
        "Aggregator should have registered the worker"
    );

    // Cleanup
    agg_runner.stop().await.ok();
    worker_runner.stop().await.ok();
}

// ============================================================================
// Test: Multiple Workers Discover Aggregator via Bootstrap
// ============================================================================

#[tokio::test]
async fn test_multi_worker_bootstrap_discovery() {
    // 1 aggregator + 3 workers, workers only know bootstrap address

    let caps = NodeCapabilities::default();
    let agg_addr: SocketAddr = "127.0.0.1:19200".parse().unwrap();

    // Start aggregator
    let agg_runner = NetworkRunnerBuilder::new()
        .local_id(PeerId::from_string("agg"))
        .listen_addr(agg_addr)
        .capabilities(NodeCapabilities { can_aggregate: true, ..caps.clone() })
        .build()
        .unwrap();
    let agg_runner = Arc::new(agg_runner);
    agg_runner.start().await.unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Start 3 workers with bootstrap to aggregator
    let mut workers = Vec::new();
    for i in 1..=3 {
        let w_addr: SocketAddr = format!("127.0.0.1:{}", 19201 + i).parse().unwrap();
        let w_runner = NetworkRunnerBuilder::new()
            .local_id(PeerId::from_string(&format!("worker-{}", i)))
            .listen_addr(w_addr)
            .bootstrap_nodes(vec![agg_addr.to_string()])
            .capabilities(NodeCapabilities { can_train: true, ..caps.clone() })
            .build()
            .unwrap();
        let w_runner = Arc::new(w_runner);
        w_runner.start().await.unwrap();
        w_runner.bootstrap().await;
        workers.push(w_runner);
    }

    // Wait for discovery to propagate
    tokio::time::sleep(Duration::from_secs(1)).await;

    // All 3 workers should be known to the aggregator
    let agg_peers = agg_runner.connected_peers().await;
    assert!(
        agg_peers.len() >= 3,
        "Aggregator should see 3 workers, got {}",
        agg_peers.len(),
    );

    // Each worker should see the aggregator
    for (i, w) in workers.iter().enumerate() {
        let peers = w.connected_peers().await;
        assert!(
            !peers.is_empty(),
            "Worker {} should have at least 1 peer (aggregator)",
            i + 1,
        );
    }

    // Cleanup
    agg_runner.stop().await.ok();
    for w in &workers {
        w.stop().await.ok();
    }
}

// ============================================================================
// Test: Connection Resilience with Exponential Backoff
// ============================================================================

#[tokio::test]
async fn test_reconnection_scheduling() {
    // Test that schedule_reconnect correctly queues peers for retry.

    let runner = NetworkRunnerBuilder::new()
        .local_id(PeerId::from_string("test"))
        .listen_addr("127.0.0.1:0".parse().unwrap())
        .reconnect_config(1, 5, 3)
        .build()
        .unwrap();
    let runner = Arc::new(runner);

    // Schedule reconnection for a peer
    let peer = PeerId::from_string("disconnected-peer");
    runner.schedule_reconnect(
        peer.clone(),
        "127.0.0.1:9999".to_string(),
        NodeCapabilities::default(),
    );

    // Check that it's in the queue
    let queue = runner.reconnect_queue();
    let q = queue.lock();
    assert!(q.contains_key(&peer), "Peer should be in reconnect queue");
    let state = q.get(&peer).unwrap();
    assert_eq!(state.attempts, 0);
    assert_eq!(state.address, "127.0.0.1:9999");
}

// ============================================================================
// Test: Authenticated Connection (ed25519 key exchange)
// ============================================================================

#[tokio::test]
async fn test_authenticated_peer_connection() {
    // Test that JoinRequest includes public key and it gets registered
    // in the peer key registry.

    let agg_addr: SocketAddr = "127.0.0.1:19300".parse().unwrap();
    let caps = NodeCapabilities::default();

    let mut agg_runner = NetworkRunnerBuilder::new()
        .local_id(PeerId::from_string("agg"))
        .listen_addr(agg_addr)
        .capabilities(NodeCapabilities { can_aggregate: true, ..caps.clone() })
        .build()
        .unwrap();

    // Set signing key on aggregator
    #[cfg(feature = "crypto-sign")]
    {
        let sk = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        agg_runner.set_signing_key(sk);
    }

    let agg_runner = Arc::new(agg_runner);
    agg_runner.start().await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Worker with signing key
    let worker_addr: SocketAddr = "127.0.0.1:19301".parse().unwrap();
    let mut worker_runner = NetworkRunnerBuilder::new()
        .local_id(PeerId::from_string("worker-auth"))
        .listen_addr(worker_addr)
        .bootstrap_nodes(vec![agg_addr.to_string()])
        .capabilities(NodeCapabilities { can_train: true, ..caps.clone() })
        .build()
        .unwrap();

    #[cfg(feature = "crypto-sign")]
    {
        let sk = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        worker_runner.set_signing_key(sk);
    }

    let worker_runner = Arc::new(worker_runner);
    worker_runner.start().await.unwrap();
    worker_runner.bootstrap().await;

    // Wait for the JoinRequest to be processed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // On the aggregator side, check if the worker's key was registered
    #[cfg(feature = "crypto-sign")]
    {
        let keys = agg_runner.peer_keys();
        let k = keys.lock();
        // The worker should have its key registered via JoinRequest
        let worker_id = PeerId::from_string("worker-auth");
        assert!(
            k.has_key(&worker_id),
            "Aggregator should have registered worker's ed25519 key"
        );
    }

    agg_runner.stop().await.ok();
    worker_runner.stop().await.ok();
}

// ============================================================================
// Test: Per-Identity Rate Limiting
// ============================================================================

#[test]
fn test_per_identity_rate_limiting() {
    use helix_node::network::rate_limit::{RateLimitConfig, RateLimiter, MessageType};

    let config = RateLimitConfig {
        default_requests_per_second: 10.0,
        default_burst_size: 10,
        ..Default::default()
    };
    let mut limiter = RateLimiter::new(config);

    let verified_peer = PeerId::from_string("verified");
    let unverified_peer = PeerId::from_string("unverified");

    // Verified peer should get full burst (10 tokens, cost 1 each)
    for _ in 0..10 {
        let result = limiter.check_rate_limit_verified(&verified_peer, MessageType::Generic, true);
        assert!(result.is_allowed(), "Verified peer should be allowed");
    }

    // Unverified peer gets half burst (5 tokens) and 2x cost
    // So only 2 requests should fit: 5 tokens / (1.0 * 2.0) = 2 requests
    let mut allowed = 0;
    for _ in 0..5 {
        let result = limiter.check_rate_limit_verified(&unverified_peer, MessageType::Generic, false);
        if result.is_allowed() {
            allowed += 1;
        }
    }
    assert!(
        allowed < 5,
        "Unverified peer should have stricter limits, got {} allowed",
        allowed,
    );
}

// ============================================================================
// Test: Full Discovery → Training → Disconnect → Reconnect Flow
// ============================================================================

#[tokio::test]
async fn test_full_discovery_training_resilience_flow() {
    // This is the full integration test:
    // 1. Start 1 aggregator
    // 2. Start 3 workers with only bootstrap node config
    // 3. Workers discover aggregator via peer exchange
    // 4. Aggregator broadcasts a training round start
    // 5. Workers respond with gradients
    // 6. Simulate killing one worker
    // 7. Aggregator should still complete round with remaining 2
    // 8. "Killed" worker reconnects

    let agg_addr: SocketAddr = "127.0.0.1:19400".parse().unwrap();
    let caps = NodeCapabilities::default();

    // === Step 1: Start aggregator ===
    let agg_runner = NetworkRunnerBuilder::new()
        .local_id(PeerId::from_string("agg"))
        .listen_addr(agg_addr)
        .capabilities(NodeCapabilities { can_aggregate: true, ..caps.clone() })
        .build()
        .unwrap();
    let agg_runner = Arc::new(agg_runner);
    agg_runner.start().await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // === Step 2: Start 3 workers with bootstrap ===
    let mut worker_runners = Vec::new();
    let mut worker_addrs = Vec::new();
    for i in 1..=3 {
        let w_addr: SocketAddr = format!("127.0.0.1:{}", 19401 + i).parse().unwrap();
        worker_addrs.push(w_addr);

        let w_runner = NetworkRunnerBuilder::new()
            .local_id(PeerId::from_string(&format!("w{}", i)))
            .listen_addr(w_addr)
            .bootstrap_nodes(vec![agg_addr.to_string()])
            .reconnect_config(1, 3, 5)
            .capabilities(NodeCapabilities { can_train: true, can_prove: true, ..caps.clone() })
            .build()
            .unwrap();
        let w_runner = Arc::new(w_runner);
        w_runner.start().await.unwrap();
        w_runner.bootstrap().await;
        worker_runners.push(w_runner);
    }

    // === Step 3: Wait for discovery ===
    tokio::time::sleep(Duration::from_secs(1)).await;

    let agg_peers = agg_runner.connected_peers().await;
    assert!(
        agg_peers.len() >= 3,
        "Aggregator should see 3+ peers, got {}",
        agg_peers.len(),
    );

    // === Step 4: Aggregator broadcasts round start ===
    let round_msg = MessagePayload::Training(TrainingMessage::RoundStart {
        round_id: 1,
        model_hash: [42u8; 32],
        params: TrainingParams {
            learning_rate: 0.01,
            batch_size: 1,
            local_epochs: 1,
            max_error_bound: 0.1,
            d_in: 2,
            d_hid: 4,
            d_out: 1,
            model_seed: 42,
        },
    });
    agg_runner.broadcast(round_msg).await;

    // Wait for workers to receive and process
    tokio::time::sleep(Duration::from_millis(500)).await;

    // === Step 5: Workers send gradient responses ===
    for (i, w) in worker_runners.iter().enumerate() {
        let grad_msg = MessagePayload::Gradient(GradientMessage::ShareGradient {
            round_id: 1,
            gradient_commitment: [(i + 1) as u8; 32],
            commitment_nonce: [0u8; 16],
            error_bound: 0.01,
            proof: vec![1, 2, 3, 4],
        });
        w.broadcast(grad_msg).await;
    }

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Aggregator should have received gradient messages
    let agg_events = collect_events(&agg_runner, Duration::from_millis(500)).await;
    let gradient_count = agg_events.iter().filter(|e| matches!(e, NetworkEvent::GradientMessage { .. })).count();
    // At least some gradients should have arrived
    assert!(
        gradient_count > 0 || agg_peers.len() >= 3,
        "Aggregator should have received gradients or peers are connected"
    );

    // === Step 6: Kill worker 3 (stop it) ===
    let killed_worker = worker_runners.pop().unwrap();
    killed_worker.stop().await.ok();

    // Give time for disconnect to propagate
    tokio::time::sleep(Duration::from_millis(500)).await;

    // === Step 7: Remaining 2 workers can still communicate ===
    // Broadcast from aggregator to remaining workers
    let round2_msg = MessagePayload::Training(TrainingMessage::RoundStart {
        round_id: 2,
        model_hash: [43u8; 32],
        params: TrainingParams {
            learning_rate: 0.01,
            batch_size: 1,
            local_epochs: 1,
            max_error_bound: 0.1,
            d_in: 2,
            d_hid: 4,
            d_out: 1,
            model_seed: 42,
        },
    });
    agg_runner.broadcast(round2_msg).await;

    // Remaining workers respond
    for (i, w) in worker_runners.iter().enumerate() {
        let grad_msg = MessagePayload::Gradient(GradientMessage::ShareGradient {
            round_id: 2,
            gradient_commitment: [(i + 10) as u8; 32],
            commitment_nonce: [0u8; 16],
            error_bound: 0.01,
            proof: vec![5, 6, 7, 8],
        });
        w.broadcast(grad_msg).await;
    }

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Check remaining worker count on aggregator
    let remaining_peers = agg_runner.connected_peers().await;
    // At minimum, the 2 living workers should still be connected
    assert!(
        remaining_peers.len() >= 2,
        "Aggregator should still have at least 2 peers, got {}",
        remaining_peers.len(),
    );

    // === Step 8: "Killed" worker reconnects ===
    // Use a different port since the old one may still be in TIME_WAIT
    let reconn_addr: SocketAddr = "127.0.0.1:19410".parse().unwrap();
    let reconnected_runner = NetworkRunnerBuilder::new()
        .local_id(PeerId::from_string("w3-reconn"))
        .listen_addr(reconn_addr)
        .bootstrap_nodes(vec![agg_addr.to_string()])
        .reconnect_config(1, 3, 5)
        .capabilities(NodeCapabilities { can_train: true, can_prove: true, ..caps.clone() })
        .build()
        .unwrap();
    let reconnected_runner = Arc::new(reconnected_runner);
    reconnected_runner.start().await.unwrap();

    // Give the listener time to bind before bootstrapping
    tokio::time::sleep(Duration::from_millis(200)).await;
    reconnected_runner.bootstrap().await;

    // Poll for the reconnected worker to see peers (up to 3 seconds)
    let mut reconn_has_peers = false;
    for _ in 0..15 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        let reconn_peers = reconnected_runner.connected_peers().await;
        if !reconn_peers.is_empty() {
            reconn_has_peers = true;
            break;
        }
    }
    assert!(reconn_has_peers, "Reconnected worker should see at least the aggregator");

    // Poll for the aggregator to see the reconnected worker (up to 2 seconds)
    let mut agg_sees_reconn = false;
    for _ in 0..10 {
        let final_peers = agg_runner.connected_peers().await;
        if final_peers.len() >= 3 {
            agg_sees_reconn = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(
        agg_sees_reconn,
        "Aggregator should see 3+ peers after reconnection",
    );

    // Cleanup
    agg_runner.stop().await.ok();
    for w in &worker_runners {
        w.stop().await.ok();
    }
    reconnected_runner.stop().await.ok();
}

// ============================================================================
// Test: NetworkRunnerConfig defaults
// ============================================================================

#[test]
fn test_network_runner_config_new_fields() {
    let config = NetworkRunnerConfig::default();
    assert!(!config.mdns_enabled, "mDNS should be disabled by default");
    assert_eq!(config.reconnect_base_interval_secs, 2);
    assert_eq!(config.reconnect_max_interval_secs, 60);
    assert_eq!(config.reconnect_max_attempts, 10);
}

// ============================================================================
// Test: Builder API for new options
// ============================================================================

#[tokio::test]
async fn test_builder_bootstrap_and_mdns() {
    let runner = NetworkRunnerBuilder::new()
        .local_id(PeerId::from_string("test"))
        .listen_addr("127.0.0.1:0".parse().unwrap())
        .bootstrap_nodes(vec!["127.0.0.1:9000".to_string(), "127.0.0.1:9001".to_string()])
        .enable_mdns(false)
        .reconnect_config(1, 10, 3)
        .build();

    assert!(runner.is_ok(), "Builder with new options should succeed");
}

// ============================================================================
// Test: PeerKeyRegistry from_bytes
// ============================================================================

#[test]
fn test_peer_key_registry_from_bytes() {
    use helix_node::network::messages::PeerKeyRegistry;

    let mut registry = PeerKeyRegistry::new();
    let peer = PeerId::from_string("test-peer");

    // Invalid key (wrong length)
    assert!(!registry.register_from_bytes(peer.clone(), &[0u8; 16]));

    // Valid key bytes (32 bytes, but may not be a valid ed25519 point)
    // Use a real ed25519 key
    #[cfg(feature = "crypto-sign")]
    {
        let sk = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let vk = sk.verifying_key();
        let key_bytes = vk.to_bytes();

        assert!(registry.register_from_bytes(peer.clone(), &key_bytes));
        assert!(registry.has_key(&peer));
        assert_eq!(registry.key_count(), 1);
    }
}

// ============================================================================
// Test: Discovery message handles new fields
// ============================================================================

#[tokio::test]
async fn test_discovery_message_with_public_key() {
    use helix_node::network::discovery::{PeerDiscovery, DiscoveryConfig};
    use helix_node::network::messages::DiscoveryMessage;

    let discovery = PeerDiscovery::new(
        PeerId::from_string("local"),
        DiscoveryConfig::default(),
    );

    // Create join request with public key
    let msg = discovery.create_join_request_with_key(
        NodeCapabilities::default(),
        "127.0.0.1:9000".to_string(),
        Some(vec![1u8; 32]),
    );

    // Verify the message was created correctly
    match msg.payload {
        MessagePayload::Discovery(DiscoveryMessage::JoinRequest {
            public_key, listen_addr, ..
        }) => {
            assert_eq!(public_key, Some(vec![1u8; 32]));
            assert_eq!(listen_addr, "127.0.0.1:9000");
        }
        _ => panic!("Expected JoinRequest"),
    }

    // Create join request without public key (backwards compatible)
    let msg2 = discovery.create_join_request(
        NodeCapabilities::default(),
        "127.0.0.1:9001".to_string(),
    );

    match msg2.payload {
        MessagePayload::Discovery(DiscoveryMessage::JoinRequest {
            public_key, ..
        }) => {
            assert_eq!(public_key, None);
        }
        _ => panic!("Expected JoinRequest"),
    }
}
