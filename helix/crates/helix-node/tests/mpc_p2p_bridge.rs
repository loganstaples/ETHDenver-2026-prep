//! Integration test: MPC training sessions over the node's P2P connections.
//!
//! Tests the full MPC ↔ Node bridge by creating 3 nodes with real TCP connections,
//! then running MPC training sessions over those P2P connections instead of
//! separate TCP connections.
//!
//! Run with:
//!   cargo test -p helix-node --test mpc_p2p_bridge -- --test-threads=1

use std::sync::Arc;
use std::time::Duration;

use helix_mpc::mpc_trainer::{MPCTrainer, MPCTrainerConfig, ModelWeights};
use helix_mpc::session::node_transport::TaggedMessage;
use helix_mpc::types::PartyId;

use helix_node::mpc_bridge::NodeMpcBridge;
use helix_node::network::{
    ConnectionPool, MpcDataMessage, MessagePayload, PeerId,
    TcpTransport, Transport, TransportConfig,
};

/// Creates N nodes with TCP transports, each listening on a unique port.
/// Returns (peer_id, transport, pool, listen_addr) for each node.
async fn create_nodes(
    n: usize,
) -> Vec<(PeerId, Arc<TcpTransport>, Arc<ConnectionPool>, std::net::SocketAddr)> {
    let mut nodes = Vec::with_capacity(n);

    for i in 0..n {
        let peer_id = PeerId::from_string(format!("node-{}", i));
        let config = TransportConfig {
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            use_tls: false,
            max_connections_per_peer: 1,
            connect_timeout: Duration::from_secs(5),
            tcp_nodelay: true,
            keepalive_secs: None,
            max_pool_size: 16,
            ..Default::default()
        };

        let transport = Arc::new(TcpTransport::new(peer_id.clone(), config).unwrap());
        // Start listening
        transport.start().await.unwrap();
        let addr = transport.local_addr();
        let pool = Arc::new(ConnectionPool::new(transport.clone()));

        nodes.push((peer_id, transport, pool, addr));
    }

    nodes
}

/// Connects all nodes to each other and registers peer addresses.
async fn connect_mesh(
    nodes: &[(PeerId, Arc<TcpTransport>, Arc<ConnectionPool>, std::net::SocketAddr)],
) {
    let n = nodes.len();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let (ref target_peer, _, _, target_addr) = nodes[j];
            let (_, ref transport, ref pool, _) = nodes[i];

            // Register the peer address
            pool.register_peer(target_peer.clone(), target_addr);

            // Establish TCP connection
            transport.connect(target_peer.clone(), target_addr).await.unwrap();
        }
    }

    // Give connections a moment to stabilize
    tokio::time::sleep(Duration::from_millis(100)).await;
}

/// Event pump: reads incoming messages from the node's transport and routes
/// MPC messages to the bridge. Non-MPC messages are ignored.
///
/// Runs until the shutdown signal is received.
async fn run_event_pump(
    transport: Arc<TcpTransport>,
    bridge: Arc<NodeMpcBridge>,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            result = transport.recv() => {
                match result {
                    Ok((_transport_peer, msg)) => {
                        // Use msg.sender (the real peer ID from the NetworkMessage)
                        // instead of _transport_peer (which is a placeholder like
                        // "unknown-127.0.0.1:XXXXX" for inbound connections).
                        if let MessagePayload::MpcData(ref mpc_msg) = msg.payload {
                            if let Err(e) = bridge.handle_incoming(&msg.sender, mpc_msg).await {
                                eprintln!("MPC route error from {}: {}", msg.sender, e);
                            }
                        }
                    }
                    Err(e) => {
                        // Transport closed or error — check if we should shut down
                        if *shutdown.borrow() {
                            break;
                        }
                        tracing::trace!("Transport recv error (may be normal): {}", e);
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    break;
                }
            }
        }
    }
}

#[tokio::test]
async fn test_mpc_weight_sharing_over_p2p() {
    let num_parties = 3;
    let nodes = create_nodes(num_parties).await;
    connect_mesh(&nodes).await;

    let session_id = "weight-share-test";
    let n = num_parties;
    let party_peer_map: Vec<(PartyId, PeerId)> = nodes
        .iter()
        .enumerate()
        .map(|(i, (peer_id, _, _, _))| (PartyId::from_index(i), peer_id.clone()))
        .collect();

    let mut bridges = Vec::with_capacity(n);
    let mut transports = Vec::with_capacity(n);

    for i in 0..n {
        let bridge = Arc::new(NodeMpcBridge::new(nodes[i].0.clone(), nodes[i].2.clone()));
        let our_party = PartyId::from_index(i);
        let transport = bridge
            .create_session(session_id, &our_party, &party_peer_map)
            .await;
        bridges.push(bridge);
        transports.push(transport);
    }

    // Start event pumps
    let (shutdown_tx, _) = tokio::sync::watch::channel(false);
    let mut pump_handles = Vec::new();
    for i in 0..n {
        let transport_arc = nodes[i].1.clone();
        let bridge = bridges[i].clone();
        let shutdown_rx = shutdown_tx.subscribe();
        pump_handles.push(tokio::spawn(run_event_pump(transport_arc, bridge, shutdown_rx)));
    }

    let config = MPCTrainerConfig::small(num_parties);
    let d_in = config.d_in;
    let d_hid = config.d_hid;

    let initial_weights = ModelWeights::from_f64(
        &[0.1, 0.2, 0.3, 0.4],
        &[0.01, 0.02],
        &[0.5, 0.6],
        &[0.03],
    );

    // Run weight sharing on all parties in parallel
    let mut handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
        let cfg = config.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };

        let handle = tokio::spawn(async move {
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

    // Verify w1 shares reconstruct to original weights
    for idx in 0..d_hid * d_in {
        let sum = all_shares.iter().fold(helix_mpc::Fr::ZERO, |acc, s| {
            helix_mpc::Fr::add(&acc, &s.0[idx])
        });
        let expected = initial_weights.w1[idx].to_f64();
        let got = sum.to_f64();
        assert!(
            (got - expected).abs() < 0.001,
            "w1[{}]: expected {}, got {}",
            idx, expected, got,
        );
    }

    // Shutdown pumps
    shutdown_tx.send(true).unwrap();
    for handle in pump_handles {
        handle.abort();
    }

    // Cleanup sessions
    for bridge in &bridges {
        bridge.remove_session(session_id).await;
    }

    // Stop transports
    for (_, transport, _, _) in &nodes {
        let _ = transport.stop().await;
    }
}

#[tokio::test]
async fn test_mpc_training_step_over_p2p() {
    let num_parties = 3;
    let nodes = create_nodes(num_parties).await;
    connect_mesh(&nodes).await;

    let session_id = "training-step-test";
    let n = num_parties;

    let party_peer_map: Vec<(PartyId, PeerId)> = nodes
        .iter()
        .enumerate()
        .map(|(i, (peer_id, _, _, _))| (PartyId::from_index(i), peer_id.clone()))
        .collect();

    let mut bridges = Vec::with_capacity(n);
    let mut transports = Vec::with_capacity(n);

    for i in 0..n {
        let bridge = Arc::new(NodeMpcBridge::new(nodes[i].0.clone(), nodes[i].2.clone()));
        let our_party = PartyId::from_index(i);
        let transport = bridge
            .create_session(session_id, &our_party, &party_peer_map)
            .await;
        bridges.push(bridge);
        transports.push(transport);
    }

    // Start event pumps
    let (shutdown_tx, _) = tokio::sync::watch::channel(false);
    let mut pump_handles = Vec::new();
    for i in 0..n {
        let transport_arc = nodes[i].1.clone();
        let bridge = bridges[i].clone();
        let shutdown_rx = shutdown_tx.subscribe();
        pump_handles.push(tokio::spawn(run_event_pump(transport_arc, bridge, shutdown_rx)));
    }

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

    // Run full training step on all parties
    let mut handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
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

    // All parties should compute the same loss
    for i in 1..losses.len() {
        assert!(
            (losses[i] - losses[0]).abs() < 0.01,
            "Loss mismatch: party 0 = {}, party {} = {}",
            losses[0], i, losses[i],
        );
    }

    // Shutdown
    shutdown_tx.send(true).unwrap();
    for handle in pump_handles {
        handle.abort();
    }
    for bridge in &bridges {
        bridge.remove_session(session_id).await;
    }
    for (_, transport, _, _) in &nodes {
        let _ = transport.stop().await;
    }
}

#[tokio::test]
async fn test_mpc_multi_step_training_over_p2p() {
    let num_parties = 3;
    let nodes = create_nodes(num_parties).await;
    connect_mesh(&nodes).await;

    let session_id = "multi-step-test";
    let n = num_parties;

    let party_peer_map: Vec<(PartyId, PeerId)> = nodes
        .iter()
        .enumerate()
        .map(|(i, (peer_id, _, _, _))| (PartyId::from_index(i), peer_id.clone()))
        .collect();

    let mut bridges = Vec::with_capacity(n);
    let mut transports = Vec::with_capacity(n);

    for i in 0..n {
        let bridge = Arc::new(NodeMpcBridge::new(nodes[i].0.clone(), nodes[i].2.clone()));
        let our_party = PartyId::from_index(i);
        let transport = bridge
            .create_session(session_id, &our_party, &party_peer_map)
            .await;
        bridges.push(bridge);
        transports.push(transport);
    }

    let (shutdown_tx, _) = tokio::sync::watch::channel(false);
    let mut pump_handles = Vec::new();
    for i in 0..n {
        let transport_arc = nodes[i].1.clone();
        let bridge = bridges[i].clone();
        let shutdown_rx = shutdown_tx.subscribe();
        pump_handles.push(tokio::spawn(run_event_pump(transport_arc, bridge, shutdown_rx)));
    }

    let config = MPCTrainerConfig {
        d_in: 2,
        d_hid: 2,
        d_out: 1,
        learning_rate: 0.1,
        num_parties,
        reshare_interval: 2,
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

    let data: Vec<(Vec<f64>, Vec<f64>)> = (0..4)
        .map(|_| (vec![1.0, 1.0], vec![1.0]))
        .collect();

    let mut handles = Vec::new();
    for (i, transport) in transports.into_iter().enumerate() {
        let cfg = config.clone();
        let weights = if i == 0 {
            Some(initial_weights.clone())
        } else {
            None
        };
        let d = data.clone();

        let handle = tokio::spawn(async move {
            let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
            trainer.share_weights(weights).await.unwrap();
            let results = trainer.train(&d).await.unwrap();

            let losses: Vec<f64> = results.iter().map(|r| r.loss).collect();
            let reshare_flags: Vec<bool> = results.iter().map(|r| r.reshared).collect();

            let (w1, b1, w2, b2) = trainer.weight_shares();
            (losses, reshare_flags, w1.to_vec(), b1.to_vec(), w2.to_vec(), b2.to_vec())
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

    // All parties agree on losses
    for step in 0..data.len() {
        for party in 1..num_parties {
            assert!(
                (all_losses[party][step] - all_losses[0][step]).abs() < 0.01,
                "Step {} loss mismatch: party 0 = {}, party {} = {}",
                step, all_losses[0][step], party, all_losses[party][step],
            );
        }
    }

    // Resharing happens at correct steps
    for party in &all_reshare {
        assert!(!party[0], "Step 0 should not reshare");
        assert!(party[1], "Step 1 should reshare (interval=2)");
        assert!(!party[2], "Step 2 should not reshare");
        assert!(party[3], "Step 3 should reshare (interval=2)");
    }

    // Loss should decrease
    assert!(
        all_losses[0].last().unwrap() < all_losses[0].first().unwrap(),
        "Loss should decrease: first={}, last={}",
        all_losses[0].first().unwrap(), all_losses[0].last().unwrap(),
    );

    // Final weights should reconstruct to finite values
    for idx in 0..all_final_w1[0].len() {
        let sum = all_final_w1.iter().fold(helix_mpc::Fr::ZERO, |acc, w| {
            helix_mpc::Fr::add(&acc, &w[idx])
        });
        assert!(
            sum.to_f64().is_finite(),
            "w1[{}] not finite after training + resharing",
            idx,
        );
    }

    // Shutdown
    shutdown_tx.send(true).unwrap();
    for handle in pump_handles {
        handle.abort();
    }
    for bridge in &bridges {
        bridge.remove_session(session_id).await;
    }
    for (_, transport, _, _) in &nodes {
        let _ = transport.stop().await;
    }
}

#[tokio::test]
async fn test_bridge_session_cleanup() {
    let num_parties = 3;
    let nodes = create_nodes(num_parties).await;
    connect_mesh(&nodes).await;

    let party_peer_map: Vec<(PartyId, PeerId)> = nodes
        .iter()
        .enumerate()
        .map(|(i, (peer_id, _, _, _))| (PartyId::from_index(i), peer_id.clone()))
        .collect();

    // Create two sessions
    let bridge = Arc::new(NodeMpcBridge::new(nodes[0].0.clone(), nodes[0].2.clone()));

    let _t1 = bridge
        .create_session("session-1", &PartyId::from_index(0), &party_peer_map)
        .await;
    let _t2 = bridge
        .create_session("session-2", &PartyId::from_index(0), &party_peer_map)
        .await;

    assert_eq!(bridge.active_session_count().await, 2);

    bridge.remove_session("session-1").await;
    assert_eq!(bridge.active_session_count().await, 1);

    bridge.remove_session("session-2").await;
    assert_eq!(bridge.active_session_count().await, 0);

    // Removing a non-existent session is a no-op
    bridge.remove_session("session-3").await;
    assert_eq!(bridge.active_session_count().await, 0);

    for (_, transport, _, _) in &nodes {
        let _ = transport.stop().await;
    }
}

#[tokio::test]
async fn test_bridge_handles_unknown_peer_message() {
    let nodes = create_nodes(2).await;

    let party_peer_map = vec![
        (PartyId::from_index(0), nodes[0].0.clone()),
        (PartyId::from_index(1), nodes[1].0.clone()),
    ];

    let bridge = NodeMpcBridge::new(nodes[0].0.clone(), nodes[0].2.clone());
    let _transport = bridge
        .create_session("test", &PartyId::from_index(0), &party_peer_map)
        .await;

    // Try to route a message from an unknown peer
    let tagged = TaggedMessage {
        session_id: "test".to_string(),
        sender: PartyId::from_index(1),
        payload: b"hello".to_vec(),
    };
    let data = bincode::serialize(&tagged).unwrap();

    let result = bridge
        .handle_incoming(
            &PeerId::from_string("unknown-peer"),
            &MpcDataMessage {
                session_id: "test".to_string(),
                data,
            },
        )
        .await;

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unknown peer"));

    for (_, transport, _, _) in &nodes {
        let _ = transport.stop().await;
    }
}
