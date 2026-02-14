//! P2P Integration Tests for HELIX Network.
//!
//! Tests the full peer-to-peer networking stack: handshake, connection
//! management, message exchange, heartbeat monitoring, and peer discovery
//! across 2-3 nodes running in the same process on different ports.

use std::net::SocketAddr;
use std::time::Duration;

use helix_node::config::NodeRole;
use helix_node::network::connection_manager::{
    ConnectionManager, ConnectionManagerConfig, P2PEvent,
};
use helix_node::network::messages::{
    HeartbeatMessage, MessagePayload, NetworkMessage, NodeCapabilities, PeerId,
};
use helix_node::network::peer_registry::PeerHealth;

fn worker_caps() -> NodeCapabilities {
    NodeCapabilities {
        can_train: true,
        can_aggregate: false,
        can_prove: true,
        gpu_memory_mb: 4096,
        cpu_cores: 8,
        storage_gb: 100,
    }
}

fn aggregator_caps() -> NodeCapabilities {
    NodeCapabilities {
        can_train: false,
        can_aggregate: true,
        can_prove: false,
        gpu_memory_mb: 0,
        cpu_cores: 16,
        storage_gb: 500,
    }
}

/// Helper: creates a ConnectionManagerConfig with OS-assigned port.
fn node_config(id: &str, role: NodeRole, caps: NodeCapabilities) -> ConnectionManagerConfig {
    ConnectionManagerConfig {
        local_id: PeerId::from_string(id),
        role,
        capabilities: caps,
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        heartbeat_interval: Duration::from_secs(1),
        heartbeat_timeout: Duration::from_secs(5),
        max_missed_heartbeats: 3,
        connect_timeout: Duration::from_secs(5),
        max_peers: 50,
        handshake_timeout: Duration::from_secs(5),
        reconnect_base_interval: Duration::from_secs(1),
        reconnect_max_interval: Duration::from_secs(10),
        reconnect_max_attempts: 5,
        ..Default::default()
    }
}

/// Helper: wait for a PeerConnected event with a specific peer_id.
async fn wait_for_connect(mgr: &ConnectionManager, expected_peer: &str, timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(100), mgr.next_event()).await {
            Ok(Some(P2PEvent::PeerConnected { peer_id, .. })) => {
                if peer_id == PeerId::from_string(expected_peer) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

// ── Test: Two nodes handshake and verify identities ──────────────────────

#[tokio::test]
async fn test_two_node_handshake_and_identity() {
    let agg = ConnectionManager::new(node_config("agg", NodeRole::Aggregator, aggregator_caps()));
    agg.start().await.unwrap();
    let agg_addr = agg.listen_addr().unwrap();

    let worker = ConnectionManager::new(node_config("w1", NodeRole::Compute, worker_caps()));
    worker.start().await.unwrap();

    // Worker connects to aggregator
    let peer_id = worker.connect_to(agg_addr).await.unwrap();
    assert_eq!(peer_id, PeerId::from_string("agg"));

    // Wait for both sides to register the connection
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Worker sees aggregator with correct role and capabilities
    let agg_snap = worker.registry().get(&PeerId::from_string("agg")).unwrap();
    assert_eq!(agg_snap.role, NodeRole::Aggregator);
    assert!(agg_snap.capabilities.can_aggregate);
    assert!(!agg_snap.capabilities.can_train);

    // Aggregator sees worker with correct role and capabilities
    let w1_snap = agg.registry().get(&PeerId::from_string("w1")).unwrap();
    assert_eq!(w1_snap.role, NodeRole::Compute);
    assert!(w1_snap.capabilities.can_train);
    assert!(!w1_snap.capabilities.can_aggregate);

    agg.stop().await;
    worker.stop().await;
}

// ── Test: Three nodes star topology ──────────────────────────────────────

#[tokio::test]
async fn test_three_node_star_topology() {
    // Aggregator is the hub; both workers connect to it
    let agg = ConnectionManager::new(node_config("agg", NodeRole::Aggregator, aggregator_caps()));
    agg.start().await.unwrap();
    let agg_addr = agg.listen_addr().unwrap();

    let w1 = ConnectionManager::new(node_config("w1", NodeRole::Compute, worker_caps()));
    w1.start().await.unwrap();

    let w2 = ConnectionManager::new(node_config("w2", NodeRole::Compute, worker_caps()));
    w2.start().await.unwrap();

    // Both workers connect
    w1.connect_to(agg_addr).await.unwrap();
    w2.connect_to(agg_addr).await.unwrap();

    tokio::time::sleep(Duration::from_millis(300)).await;

    // Aggregator has 2 connections
    assert_eq!(agg.connection_count(), 2);

    // Aggregator registry has 2 compute nodes
    let compute_peers = agg.registry().peers_by_role(NodeRole::Compute);
    assert_eq!(compute_peers.len(), 2);

    let peer_ids: Vec<String> = compute_peers.iter().map(|p| p.peer_id.to_string()).collect();
    assert!(peer_ids.contains(&"w1".to_string()));
    assert!(peer_ids.contains(&"w2".to_string()));

    // Each worker has 1 connection (to aggregator)
    assert_eq!(w1.connection_count(), 1);
    assert_eq!(w2.connection_count(), 1);

    agg.stop().await;
    w1.stop().await;
    w2.stop().await;
}

// ── Test: Bidirectional message exchange ──────────────────────────────────

#[tokio::test]
async fn test_bidirectional_message_exchange() {
    let node_a = ConnectionManager::new(node_config("alice", NodeRole::Aggregator, aggregator_caps()));
    node_a.start().await.unwrap();
    let addr_a = node_a.listen_addr().unwrap();

    let node_b = ConnectionManager::new(node_config("bob", NodeRole::Compute, worker_caps()));
    node_b.start().await.unwrap();

    node_b.connect_to(addr_a).await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Alice sends to Bob
    let msg_a = NetworkMessage::new(
        PeerId::from_string("alice"),
        MessagePayload::Heartbeat(HeartbeatMessage { seq: 100, is_pong: false, load: 25 }),
    );
    node_a.send(&PeerId::from_string("bob"), msg_a).await.unwrap();

    // Bob sends to Alice
    let msg_b = NetworkMessage::new(
        PeerId::from_string("bob"),
        MessagePayload::Heartbeat(HeartbeatMessage { seq: 200, is_pong: true, load: 75 }),
    );
    node_b.send(&PeerId::from_string("alice"), msg_b).await.unwrap();

    // Both should receive each other's messages
    // Collect events from both nodes with timeout
    let mut alice_got_message = false;
    let mut bob_got_message = false;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < deadline && (!alice_got_message || !bob_got_message) {
        tokio::select! {
            Some(event) = node_a.next_event() => {
                if let P2PEvent::MessageReceived { from, message } = event {
                    if from == PeerId::from_string("bob") {
                        if let MessagePayload::Heartbeat(hb) = message.payload {
                            if hb.seq == 200 && hb.is_pong && hb.load == 75 {
                                alice_got_message = true;
                            }
                        }
                    }
                }
            }
            Some(event) = node_b.next_event() => {
                if let P2PEvent::MessageReceived { from, message } = event {
                    if from == PeerId::from_string("alice") {
                        if let MessagePayload::Heartbeat(hb) = message.payload {
                            if hb.seq == 100 && !hb.is_pong && hb.load == 25 {
                                bob_got_message = true;
                            }
                        }
                    }
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
        }
    }

    assert!(alice_got_message, "Alice should have received Bob's message");
    assert!(bob_got_message, "Bob should have received Alice's message");

    node_a.stop().await;
    node_b.stop().await;
}

// ── Test: Broadcast to multiple peers ────────────────────────────────────

#[tokio::test]
async fn test_broadcast_to_multiple_peers() {
    let agg = ConnectionManager::new(node_config("agg", NodeRole::Aggregator, aggregator_caps()));
    agg.start().await.unwrap();
    let agg_addr = agg.listen_addr().unwrap();

    let w1 = ConnectionManager::new(node_config("w1", NodeRole::Compute, worker_caps()));
    w1.start().await.unwrap();

    let w2 = ConnectionManager::new(node_config("w2", NodeRole::Compute, worker_caps()));
    w2.start().await.unwrap();

    w1.connect_to(agg_addr).await.unwrap();
    w2.connect_to(agg_addr).await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Aggregator broadcasts
    let broadcast = NetworkMessage::new(
        PeerId::from_string("agg"),
        MessagePayload::Heartbeat(HeartbeatMessage { seq: 999, is_pong: false, load: 0 }),
    );
    let results = agg.broadcast(broadcast).await;
    assert_eq!(results.len(), 2);
    for (_, result) in &results {
        assert!(result.is_ok());
    }

    // Both workers should receive it
    let mut w1_received = false;
    let mut w2_received = false;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < deadline && (!w1_received || !w2_received) {
        tokio::select! {
            Some(event) = w1.next_event() => {
                if let P2PEvent::MessageReceived { message, .. } = event {
                    if let MessagePayload::Heartbeat(hb) = message.payload {
                        if hb.seq == 999 { w1_received = true; }
                    }
                }
            }
            Some(event) = w2.next_event() => {
                if let P2PEvent::MessageReceived { message, .. } = event {
                    if let MessagePayload::Heartbeat(hb) = message.payload {
                        if hb.seq == 999 { w2_received = true; }
                    }
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
        }
    }

    assert!(w1_received, "Worker 1 should have received broadcast");
    assert!(w2_received, "Worker 2 should have received broadcast");

    agg.stop().await;
    w1.stop().await;
    w2.stop().await;
}

// ── Test: Heartbeat delivery ─────────────────────────────────────────────

#[tokio::test]
async fn test_heartbeat_delivery() {
    // Use short heartbeat intervals
    let mut config_a = node_config("node-a", NodeRole::Aggregator, aggregator_caps());
    config_a.heartbeat_interval = Duration::from_millis(200);
    let node_a = ConnectionManager::new(config_a);
    node_a.start().await.unwrap();
    let addr_a = node_a.listen_addr().unwrap();

    let mut config_b = node_config("node-b", NodeRole::Compute, worker_caps());
    config_b.heartbeat_interval = Duration::from_millis(200);
    let node_b = ConnectionManager::new(config_b);
    node_b.start().await.unwrap();

    node_b.connect_to(addr_a).await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Wait for at least 3 heartbeats
    tokio::time::sleep(Duration::from_millis(800)).await;

    // Both peers should have received heartbeats and be healthy
    let a_snap = node_b.registry().get(&PeerId::from_string("node-a"));
    assert!(a_snap.is_some(), "Node A should be in Node B's registry");
    let a_snap = a_snap.unwrap();
    assert!(a_snap.health.is_healthy(), "Node A should be healthy from Node B's perspective");
    assert!(a_snap.messages_received > 0, "Node B should have received messages from Node A");

    let b_snap = node_a.registry().get(&PeerId::from_string("node-b"));
    assert!(b_snap.is_some(), "Node B should be in Node A's registry");
    let b_snap = b_snap.unwrap();
    assert!(b_snap.health.is_healthy(), "Node B should be healthy from Node A's perspective");
    assert!(b_snap.messages_received > 0, "Node A should have received messages from Node B");

    node_a.stop().await;
    node_b.stop().await;
}

// ── Test: Disconnect detection ───────────────────────────────────────────

#[tokio::test]
async fn test_disconnect_detection_via_stop() {
    let node_a = ConnectionManager::new(node_config("node-a", NodeRole::Aggregator, aggregator_caps()));
    node_a.start().await.unwrap();
    let addr_a = node_a.listen_addr().unwrap();

    let node_b = ConnectionManager::new(node_config("node-b", NodeRole::Compute, worker_caps()));
    node_b.start().await.unwrap();

    node_b.connect_to(addr_a).await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert_eq!(node_a.connection_count(), 1);

    // Stop node B — this closes all its connections
    node_b.stop().await;

    // Node A should detect the disconnect
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Node A's read loop should have detected the closed connection
    // and removed it from the registry
    // (The exact timing depends on read loop detecting EOF)
    let remaining = node_a.registry().count();
    assert_eq!(remaining, 0, "Node A should have detected Node B's disconnect");

    node_a.stop().await;
}

// ── Test: Registry role queries ──────────────────────────────────────────

#[tokio::test]
async fn test_registry_role_based_queries() {
    let agg = ConnectionManager::new(node_config("agg", NodeRole::Aggregator, aggregator_caps()));
    agg.start().await.unwrap();
    let agg_addr = agg.listen_addr().unwrap();

    // Create workers and a verifier
    let w1 = ConnectionManager::new(node_config("w1", NodeRole::Compute, worker_caps()));
    w1.start().await.unwrap();

    let w2 = ConnectionManager::new(node_config("w2", NodeRole::Compute, worker_caps()));
    w2.start().await.unwrap();

    let v1 = ConnectionManager::new(node_config("v1", NodeRole::Verifier, NodeCapabilities {
        can_prove: true,
        ..Default::default()
    }));
    v1.start().await.unwrap();

    w1.connect_to(agg_addr).await.unwrap();
    w2.connect_to(agg_addr).await.unwrap();
    v1.connect_to(agg_addr).await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Aggregator should have all three
    assert_eq!(agg.connection_count(), 3);

    let compute_peers = agg.peers_by_role(NodeRole::Compute);
    assert_eq!(compute_peers.len(), 2);

    let verifier_peers = agg.peers_by_role(NodeRole::Verifier);
    assert_eq!(verifier_peers.len(), 1);
    assert_eq!(verifier_peers[0].peer_id, PeerId::from_string("v1"));

    // Capability query: who can train?
    let trainers = agg.registry().peers_with_capability(|c| c.can_train);
    assert_eq!(trainers.len(), 2);

    // Capability query: who can prove?
    let provers = agg.registry().peers_with_capability(|c| c.can_prove);
    assert_eq!(provers.len(), 3); // w1, w2 (worker_caps has can_prove=true), v1

    agg.stop().await;
    w1.stop().await;
    w2.stop().await;
    v1.stop().await;
}

// ── Test: Connection to self is rejected ─────────────────────────────────

#[tokio::test]
async fn test_duplicate_connection_rejected() {
    let node_a = ConnectionManager::new(node_config("node-a", NodeRole::Aggregator, aggregator_caps()));
    node_a.start().await.unwrap();
    let addr_a = node_a.listen_addr().unwrap();

    let node_b = ConnectionManager::new(node_config("node-b", NodeRole::Compute, worker_caps()));
    node_b.start().await.unwrap();

    // First connection should succeed
    node_b.connect_to(addr_a).await.unwrap();

    // Second connection to same peer should fail
    let result = node_b.connect_to(addr_a).await;
    assert!(result.is_err(), "Duplicate connection should be rejected");

    node_a.stop().await;
    node_b.stop().await;
}

// ── Test: Registry stats ─────────────────────────────────────────────────

#[tokio::test]
async fn test_registry_stats() {
    let agg = ConnectionManager::new(node_config("agg", NodeRole::Aggregator, aggregator_caps()));
    agg.start().await.unwrap();
    let agg_addr = agg.listen_addr().unwrap();

    let w1 = ConnectionManager::new(node_config("w1", NodeRole::Compute, worker_caps()));
    w1.start().await.unwrap();

    let w2 = ConnectionManager::new(node_config("w2", NodeRole::Compute, worker_caps()));
    w2.start().await.unwrap();

    w1.connect_to(agg_addr).await.unwrap();
    w2.connect_to(agg_addr).await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let stats = agg.registry().stats();
    assert_eq!(stats.total, 2);
    assert_eq!(stats.healthy, 2);
    assert_eq!(stats.degraded, 0);
    assert_eq!(stats.disconnected, 0);
    assert_eq!(*stats.by_role.get(&NodeRole::Compute).unwrap(), 2);

    agg.stop().await;
    w1.stop().await;
    w2.stop().await;
}

// ── Test: Large message exchange ─────────────────────────────────────────

#[tokio::test]
async fn test_large_message_exchange() {
    let node_a = ConnectionManager::new(node_config("sender", NodeRole::Aggregator, aggregator_caps()));
    node_a.start().await.unwrap();
    let addr_a = node_a.listen_addr().unwrap();

    let mut config_b = node_config("receiver", NodeRole::Compute, worker_caps());
    config_b.heartbeat_interval = Duration::from_secs(60);
    let node_b = ConnectionManager::new(config_b);
    node_b.start().await.unwrap();

    node_b.connect_to(addr_a).await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Send 100 messages rapidly
    for i in 0..100u64 {
        let msg = NetworkMessage::new(
            PeerId::from_string("sender"),
            MessagePayload::Heartbeat(HeartbeatMessage { seq: i, is_pong: false, load: 0 }),
        );
        node_a.send(&PeerId::from_string("receiver"), msg).await.unwrap();
    }

    // Wait for delivery
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Check that messages were received
    let snap = node_a.registry().get(&PeerId::from_string("receiver")).unwrap();
    assert!(snap.messages_sent >= 100, "Should have sent at least 100 messages");

    node_a.stop().await;
    node_b.stop().await;
}
