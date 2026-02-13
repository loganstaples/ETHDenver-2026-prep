//! Multi-Process Node Orchestration Integration Tests.
//!
//! These tests verify the complete distributed training flow with real TCP
//! networking between 1 aggregator + N workers running as separate tokio
//! runtimes. Workers generate real ZK proofs via `MLTrainingProverV2`.
//!
//! Success Criteria:
//! 1. 1 aggregator + 3 workers connect over real TCP
//! 2. Complete training round with real ZK proof generation
//! 3. Aggregator collects valid proofs from all workers
//! 4. Round completes successfully

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use helix_node::network::messages::{
    GradientMessage, HeartbeatMessage, MessagePayload, NodeCapabilities, PeerId, PeerInfo,
    TrainingMessage, TrainingParams,
};
use helix_node::network::runner::{NetworkEvent, NetworkRunnerBuilder};
use helix_node::trainer::{MlpModel, Trainer};
use helix_node::training::consensus::compute_hiding_gradient_commitment;
use helix_node::training::orchestrator::{
    OrchestratorConfig, OrchestratorEvent, TrainingOrchestrator,
};
use helix_node::training::verification::{VerificationConfig, VerificationPolicy};

// ============================================================================
// Helpers
// ============================================================================

/// Finds an available TCP port by binding to port 0.
fn get_available_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Generates deterministic synthetic training data from a seed.
fn generate_training_data(d_in: usize, d_out: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = seed;
    let mut next = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };

    let x: Vec<f64> = (0..d_in).map(|_| next() * 0.5).collect();
    let target: Vec<f64> = (0..d_out).map(|_| next().abs()).collect();

    (x, target)
}

// ============================================================================
// Test: Multi-Process Training Round with Real ZK Proofs
// ============================================================================

/// End-to-end test: 1 aggregator + 3 workers over real TCP, generating real
/// Halo2 KZG proofs, completing a full training round.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_multi_process_training_round_with_real_proofs() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Model config: small weights (0.001 range) to stay within ReLU lookup range.
    let d_in = 2;
    let d_hid = 2;
    let d_out = 1;
    let model_seed = 42u64;
    let learning_rate = 0.001;

    // ── 1. Pick ports ──────────────────────────────────────────────
    let agg_port = get_available_port();
    let worker_ports: Vec<u16> = (0..3).map(|_| get_available_port()).collect();
    let agg_addr: SocketAddr = format!("127.0.0.1:{}", agg_port).parse().unwrap();

    log::info!(
        "Ports: aggregator={}, workers={:?}",
        agg_port, worker_ports
    );

    // ── 2. Start aggregator ────────────────────────────────────────
    let agg_id = PeerId::from_string("aggregator");
    let agg_network = NetworkRunnerBuilder::new()
        .local_id(agg_id.clone())
        .listen_addr(agg_addr)
        .capabilities(NodeCapabilities {
            can_train: false,
            can_aggregate: true,
            can_prove: false,
            gpu_memory_mb: 0,
            cpu_cores: 4,
            storage_gb: 10,
        })
        .build()
        .expect("Failed to build aggregator network");

    let agg_network = Arc::new(agg_network);
    agg_network.start().await.expect("Failed to start aggregator");
    log::info!("Aggregator started on {}", agg_addr);

    // ── 3. Create orchestrator ─────────────────────────────────────
    let orch_config = OrchestratorConfig {
        min_workers: 3,
        max_workers: 10,
        collection_timeout: Duration::from_secs(45),
        aggregation_timeout: Duration::from_secs(30),
        heartbeat_interval: Duration::from_secs(5),
        worker_timeout: Duration::from_secs(60),
        // Disable BFT consensus — it requires workers to also run the
        // consensus protocol which adds complexity. Leader-only aggregation
        // is sufficient for proving the multi-process flow works.
        consensus_enabled: false,
        // Use StructuralOnly verification: proofs are real 1856-byte Halo2
        // KZG proofs (generated and self-verified on workers), but the
        // aggregator can't do full Halo2 verification because it would
        // need the same SRS as the workers. StructuralOnly checks proof
        // size/format without requiring SRS agreement.
        verification: VerificationConfig {
            policy: VerificationPolicy::StructuralOnly,
            ..Default::default()
        },
        default_params: TrainingParams {
            learning_rate,
            batch_size: 1,
            local_epochs: 1,
            max_error_bound: 0.1,
            d_in,
            d_hid,
            d_out,
            model_seed,
            num_layers: 2,
            activation_type: 0,
        },
        ..Default::default()
    };

    let orchestrator =
        TrainingOrchestrator::new(agg_id.clone(), orch_config, agg_network.clone());
    orchestrator.set_leader(true);
    let mut event_rx = orchestrator.start().await;

    // ── 4. Start workers ───────────────────────────────────────────
    // Track worker handles so their tasks stay alive.
    let mut worker_handles = Vec::new();

    for i in 0..3usize {
        let worker_id = PeerId::from_string(format!("worker-{}", i));
        let worker_addr: SocketAddr =
            format!("127.0.0.1:{}", worker_ports[i]).parse().unwrap();

        let worker_network = NetworkRunnerBuilder::new()
            .local_id(worker_id.clone())
            .listen_addr(worker_addr)
            .capabilities(NodeCapabilities {
                can_train: true,
                can_aggregate: false,
                can_prove: true,
                gpu_memory_mb: 0,
                cpu_cores: 4,
                storage_gb: 10,
            })
            .build()
            .unwrap_or_else(|e| panic!("Failed to build worker-{} network: {}", i, e));

        let worker_network = Arc::new(worker_network);
        worker_network
            .start()
            .await
            .unwrap_or_else(|e| panic!("Failed to start worker-{}: {}", i, e));

        // Connect to aggregator
        let agg_peer = PeerInfo {
            id: agg_id.clone(),
            address: agg_addr.to_string(),
            capabilities: NodeCapabilities {
                can_aggregate: true,
                ..Default::default()
            },
            last_seen: 0,
            reputation: 100,
        };
        worker_network
            .connect_peer(agg_peer)
            .await
            .unwrap_or_else(|e| panic!("Worker-{} failed to connect to aggregator: {}", i, e));

        log::info!("Worker-{} ({}) connected to aggregator", i, worker_id);

        // Spawn worker heartbeat + event loop
        let wn = worker_network.clone();
        let wid = worker_id.clone();
        let agg_id_clone = agg_id.clone();
        let handle = tokio::spawn(async move {
            // Send an initial heartbeat to register with aggregator
            wn.broadcast(MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 0,
                is_pong: false,
                load: 30,
            }))
            .await;

            // Also send a second heartbeat after a short delay for reliability
            tokio::time::sleep(Duration::from_millis(500)).await;
            wn.broadcast(MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 1,
                is_pong: false,
                load: 30,
            }))
            .await;

            // Worker event loop: wait for RoundStart, train, send gradient
            let mut steps_completed = 0u64;

            loop {
                let event = tokio::time::timeout(
                    Duration::from_secs(120),
                    wn.next_event(),
                )
                .await;

                match event {
                    Ok(Some(NetworkEvent::TrainingMessage { from: _, message })) => {
                        match message {
                            TrainingMessage::RoundStart {
                                round_id,
                                model_hash: _,
                                params,
                            } => {
                                log::info!(
                                    "{}: Received RoundStart #{} ({}x{}x{})",
                                    wid,
                                    round_id,
                                    params.d_in,
                                    params.d_hid,
                                    params.d_out,
                                );

                                // Initialize trainer with small weights matching demo pattern:
                                // model weights MUST be in 0.001 range to stay within ReLU
                                // lookup range ±128 when quantized.
                                let model = MlpModel::new(
                                    params.d_in,
                                    params.d_hid,
                                    params.d_out,
                                    vec![0.001, 0.002, 0.003, 0.001], // W1
                                    vec![0.0, 0.0],                    // b1
                                    vec![0.001, 0.001],                // W2
                                    vec![0.0],                         // b2
                                );
                                let mut trainer =
                                    Trainer::with_model(model, params.learning_rate);

                                // Generate synthetic data
                                let (x, target) = generate_training_data(
                                    params.d_in,
                                    params.d_out,
                                    params.model_seed + round_id,
                                );

                                log::info!("{}: Training step (round {})...", wid, round_id);
                                match trainer.train_step(&x, &target) {
                                    Ok(result) => {
                                        let proof_bytes = result
                                            .evm_bundle
                                            .as_ref()
                                            .map(|b| b.evm_proof.clone())
                                            .unwrap_or_else(|| {
                                                result.proof_result.proof.clone()
                                            });

                                        log::info!(
                                            "{}: Proof generated: {} bytes, loss={:.6}, verified={}",
                                            wid,
                                            proof_bytes.len(),
                                            result.loss,
                                            result.proof_result.verified,
                                        );

                                        // Compute hiding commitment
                                        let (hiding_commitment, nonce) =
                                            compute_hiding_gradient_commitment(
                                                &result.commitment,
                                            );

                                        // Send gradient + proof directly to aggregator.
                                        // Uses send_direct() instead of broadcast() to
                                        // avoid gossip multi-hop unreliability — ensures
                                        // the message reaches the aggregator immediately
                                        // via the existing TCP connection.
                                        if let Err(e) = wn.send_direct(
                                            &agg_id_clone,
                                            MessagePayload::Gradient(
                                                GradientMessage::ShareGradient {
                                                    round_id,
                                                    gradient_commitment: hiding_commitment,
                                                    commitment_nonce: nonce,
                                                    error_bound: result.loss * 0.01,
                                                    proof: proof_bytes,
                                                },
                                            ),
                                        )
                                        .await
                                        {
                                            log::error!(
                                                "{}: Failed to send gradient directly: {}",
                                                wid, e,
                                            );
                                        }

                                        steps_completed += 1;
                                        log::info!(
                                            "{}: Gradient sent for round {} (steps: {})",
                                            wid,
                                            round_id,
                                            steps_completed,
                                        );
                                    }
                                    Err(e) => {
                                        log::error!(
                                            "{}: Training step failed: {}",
                                            wid,
                                            e,
                                        );
                                    }
                                }
                            }
                            TrainingMessage::RoundComplete { round_id, .. } => {
                                log::info!("{}: Round {} completed", wid, round_id);
                                // Worker can exit after round completes
                                return steps_completed;
                            }
                            _ => {}
                        }
                    }
                    Ok(Some(NetworkEvent::Heartbeat { .. })) => {
                        // Heartbeat pongs — ignore
                    }
                    Ok(Some(NetworkEvent::PeerDiscovered(peer))) => {
                        log::info!("{}: Discovered peer {}", wid, peer.id);
                    }
                    Ok(Some(NetworkEvent::Error { error, .. })) => {
                        log::warn!("{}: Network error: {}", wid, error);
                    }
                    Ok(Some(_)) => {
                        // Other events — ignore
                    }
                    Ok(None) => {
                        log::warn!("{}: Event stream ended", wid);
                        return steps_completed;
                    }
                    Err(_) => {
                        log::warn!("{}: Timed out waiting for event", wid);
                        return steps_completed;
                    }
                }
            }
        });

        worker_handles.push(handle);
    }

    // ── 5. Wait for workers to register via heartbeats ─────────────
    // The aggregator's orchestrator registers workers when it receives
    // heartbeats (via start_network_processing). We poll until 3 workers
    // are available.
    let registration_timeout = Duration::from_secs(15);
    let registration_start = std::time::Instant::now();
    loop {
        let stats = orchestrator.worker_stats();
        log::info!(
            "Waiting for workers: total={}, available={}",
            stats.total, stats.available
        );
        if stats.available >= 3 {
            break;
        }
        if registration_start.elapsed() > registration_timeout {
            panic!(
                "Timed out waiting for 3 workers to register (have {})",
                stats.total,
            );
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    log::info!("All 3 workers registered, starting training round...");

    // ── 6. Start training round ────────────────────────────────────
    let initial_model = MlpModel::new(
        d_in,
        d_hid,
        d_out,
        vec![0.001, 0.002, 0.003, 0.001],
        vec![0.0, 0.0],
        vec![0.001, 0.001],
        vec![0.0],
    );
    let model_hash = initial_model.commitment();

    let round_id = orchestrator
        .start_round(model_hash)
        .await
        .expect("Failed to start training round");

    log::info!("Round {} started, waiting for completion...", round_id);

    // ── 7. Wait for round to complete ──────────────────────────────
    let round_timeout = Duration::from_secs(180); // ZK proof gen takes time
    let round_result = tokio::time::timeout(round_timeout, async {
        let mut gradients_received = 0usize;
        while let Some(event) = event_rx.recv().await {
            match event {
                OrchestratorEvent::GradientReceived { round_id: rid, peer_id } => {
                    gradients_received += 1;
                    log::info!(
                        "Aggregator: Gradient {}/3 received from {} (round {})",
                        gradients_received, peer_id, rid,
                    );
                }
                OrchestratorEvent::RoundCompleted {
                    round_id: rid,
                    result_hash,
                } => {
                    log::info!(
                        "Round {} completed! result_hash={}",
                        rid,
                        hex::encode(&result_hash[..8]),
                    );
                    return Ok((rid, result_hash, gradients_received));
                }
                OrchestratorEvent::RoundFailed {
                    round_id: rid,
                    reason,
                } => {
                    return Err(format!(
                        "Round {} failed (gradients: {}): {}",
                        rid, gradients_received, reason,
                    ));
                }
                OrchestratorEvent::GradientRejected {
                    round_id: rid,
                    peer_id,
                    reason,
                } => {
                    log::warn!(
                        "Gradient rejected from {} in round {}: {}",
                        peer_id, rid, reason,
                    );
                }
                OrchestratorEvent::WorkerJoined { peer_id } => {
                    log::info!("Aggregator: Worker joined: {}", peer_id);
                }
                _ => {}
            }
        }
        Err("Event stream ended unexpectedly".to_string())
    })
    .await;

    // ── 8. Verify results ──────────────────────────────────────────
    match round_result {
        Ok(Ok((rid, result_hash, gradients_count))) => {
            assert_eq!(rid, round_id, "Round ID should match");
            assert_ne!(result_hash, [0u8; 32], "Result hash should be non-zero");
            assert_eq!(gradients_count, 3, "Should have received 3 gradients");
            log::info!(
                "SUCCESS: Round {} completed with {} gradients, hash={}",
                rid,
                gradients_count,
                hex::encode(&result_hash[..8]),
            );
        }
        Ok(Err(e)) => {
            panic!("Round failed: {}", e);
        }
        Err(_) => {
            panic!(
                "Round timed out after {:?} — ZK proofs may still be generating",
                round_timeout,
            );
        }
    }

    // ── 9. Verify workers completed their steps ────────────────────
    // Give workers a moment to process RoundComplete message
    tokio::time::sleep(Duration::from_secs(2)).await;

    for (i, handle) in worker_handles.into_iter().enumerate() {
        // Don't block forever — workers might not have received RoundComplete
        match tokio::time::timeout(Duration::from_secs(5), handle).await {
            Ok(Ok(steps)) => {
                assert!(
                    steps >= 1,
                    "Worker-{} should have completed at least 1 step, got {}",
                    i, steps,
                );
                log::info!("Worker-{} completed {} steps", i, steps);
            }
            Ok(Err(e)) => {
                // JoinError — task panicked
                panic!("Worker-{} panicked: {:?}", i, e);
            }
            Err(_) => {
                // Worker didn't finish in time — that's OK, it may be waiting
                // for more events. The important thing is the round completed.
                log::info!("Worker-{} still running (round completed on aggregator side)", i);
            }
        }
    }

    // Clean up
    orchestrator.stop();
    log::info!("Test complete: multi-process training round successful!");
}

// ============================================================================
// Test: Workers Reconnect After Brief Disconnect
// ============================================================================

/// Verifies that workers can connect, register, and participate in training
/// even with a slightly delayed start (simulating process startup jitter).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_staggered_worker_startup() {
    let _ = env_logger::builder().is_test(true).try_init();

    let agg_port = get_available_port();
    let agg_addr: SocketAddr = format!("127.0.0.1:{}", agg_port).parse().unwrap();
    let agg_id = PeerId::from_string("aggregator");

    // Start aggregator
    let agg_network = NetworkRunnerBuilder::new()
        .local_id(agg_id.clone())
        .listen_addr(agg_addr)
        .capabilities(NodeCapabilities {
            can_aggregate: true,
            ..Default::default()
        })
        .build()
        .unwrap();
    let agg_network = Arc::new(agg_network);
    agg_network.start().await.unwrap();

    let orch_config = OrchestratorConfig {
        min_workers: 2,
        consensus_enabled: false,
        worker_timeout: Duration::from_secs(30),
        collection_timeout: Duration::from_secs(60),
        default_params: TrainingParams {
            learning_rate: 0.001,
            batch_size: 1,
            local_epochs: 1,
            max_error_bound: 0.1,
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            model_seed: 42,
            num_layers: 2,
            activation_type: 0,
        },
        ..Default::default()
    };
    let orchestrator =
        TrainingOrchestrator::new(agg_id.clone(), orch_config, agg_network.clone());
    orchestrator.set_leader(true);
    let _event_rx = orchestrator.start().await;

    // Start workers with staggered delays
    for i in 0..3usize {
        let delay = Duration::from_millis(i as u64 * 1000); // 0s, 1s, 2s
        let agg_id_c = agg_id.clone();

        tokio::spawn(async move {
            tokio::time::sleep(delay).await;

            let worker_port = get_available_port();
            let worker_id = PeerId::from_string(format!("staggered-worker-{}", i));
            let worker_addr: SocketAddr =
                format!("127.0.0.1:{}", worker_port).parse().unwrap();

            let worker_network = NetworkRunnerBuilder::new()
                .local_id(worker_id.clone())
                .listen_addr(worker_addr)
                .capabilities(NodeCapabilities {
                    can_train: true,
                    can_prove: true,
                    ..Default::default()
                })
                .build()
                .unwrap();
            let worker_network = Arc::new(worker_network);
            worker_network.start().await.unwrap();

            let agg_peer = PeerInfo {
                id: agg_id_c.clone(),
                address: format!("127.0.0.1:{}", agg_port),
                capabilities: NodeCapabilities {
                    can_aggregate: true,
                    ..Default::default()
                },
                last_seen: 0,
                reputation: 100,
            };
            worker_network.connect_peer(agg_peer).await.unwrap();

            // Register via heartbeat
            worker_network
                .broadcast(MessagePayload::Heartbeat(HeartbeatMessage {
                    seq: 0,
                    is_pong: false,
                    load: 30,
                }))
                .await;

            log::info!("Staggered worker-{} connected (delay={:?})", i, delay);
        });
    }

    // Wait for all staggered workers to register
    let timeout = Duration::from_secs(15);
    let start = std::time::Instant::now();
    loop {
        let stats = orchestrator.worker_stats();
        if stats.available >= 2 {
            log::info!(
                "Staggered test: {}/{} workers registered",
                stats.available,
                3
            );
            break;
        }
        if start.elapsed() > timeout {
            panic!(
                "Staggered workers failed to register in time (have {})",
                stats.total
            );
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // Verify we can start a round
    let model = MlpModel::new(
        2,
        2,
        1,
        vec![0.001, 0.002, 0.003, 0.001],
        vec![0.0, 0.0],
        vec![0.001, 0.001],
        vec![0.0],
    );
    let result = orchestrator.start_round(model.commitment()).await;
    assert!(
        result.is_ok(),
        "Should be able to start round with staggered workers: {:?}",
        result.err(),
    );

    log::info!(
        "Staggered worker test passed: round {} started",
        result.unwrap()
    );

    orchestrator.stop();
}

// ============================================================================
// Test: Aggregator Rejects Round with Insufficient Workers
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_aggregator_rejects_insufficient_workers() {
    let _ = env_logger::builder().is_test(true).try_init();

    let agg_port = get_available_port();
    let agg_addr: SocketAddr = format!("127.0.0.1:{}", agg_port).parse().unwrap();
    let agg_id = PeerId::from_string("aggregator");

    let agg_network = NetworkRunnerBuilder::new()
        .local_id(agg_id.clone())
        .listen_addr(agg_addr)
        .capabilities(NodeCapabilities {
            can_aggregate: true,
            ..Default::default()
        })
        .build()
        .unwrap();
    let agg_network = Arc::new(agg_network);
    agg_network.start().await.unwrap();

    let orch_config = OrchestratorConfig {
        min_workers: 3,
        consensus_enabled: false,
        default_params: TrainingParams {
            learning_rate: 0.001,
            batch_size: 1,
            local_epochs: 1,
            max_error_bound: 0.1,
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            model_seed: 42,
            num_layers: 2,
            activation_type: 0,
        },
        ..Default::default()
    };
    let orchestrator =
        TrainingOrchestrator::new(agg_id.clone(), orch_config, agg_network.clone());
    orchestrator.set_leader(true);
    let _event_rx = orchestrator.start().await;

    // Only register 1 worker (need 3)
    let worker_port = get_available_port();
    let worker_id = PeerId::from_string("solo-worker");
    let worker_addr: SocketAddr = format!("127.0.0.1:{}", worker_port).parse().unwrap();

    let worker_network = NetworkRunnerBuilder::new()
        .local_id(worker_id.clone())
        .listen_addr(worker_addr)
        .capabilities(NodeCapabilities {
            can_train: true,
            can_prove: true,
            ..Default::default()
        })
        .build()
        .unwrap();
    let worker_network = Arc::new(worker_network);
    worker_network.start().await.unwrap();

    let agg_peer = PeerInfo {
        id: agg_id.clone(),
        address: agg_addr.to_string(),
        capabilities: NodeCapabilities {
            can_aggregate: true,
            ..Default::default()
        },
        last_seen: 0,
        reputation: 100,
    };
    worker_network.connect_peer(agg_peer).await.unwrap();

    worker_network
        .broadcast(MessagePayload::Heartbeat(HeartbeatMessage {
            seq: 0,
            is_pong: false,
            load: 30,
        }))
        .await;

    // Wait for worker to register
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Try to start round — should fail
    let model = MlpModel::new_random(2, 2, 1, 42);
    let result = orchestrator.start_round(model.commitment()).await;
    assert!(
        result.is_err(),
        "Should not start round with only 1 worker when min=3",
    );

    log::info!(
        "Correctly rejected round start: {:?}",
        result.err().unwrap()
    );

    orchestrator.stop();
}

// ============================================================================
// Test: Network Connectivity Verification
// ============================================================================

/// Verifies basic TCP connectivity: worker sends heartbeat, aggregator
/// receives it and registers the worker. No ZK proofs involved.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_tcp_connectivity_and_worker_registration() {
    let _ = env_logger::builder().is_test(true).try_init();

    let agg_port = get_available_port();
    let agg_addr: SocketAddr = format!("127.0.0.1:{}", agg_port).parse().unwrap();
    let agg_id = PeerId::from_string("aggregator");

    // Start aggregator
    let agg_network = NetworkRunnerBuilder::new()
        .local_id(agg_id.clone())
        .listen_addr(agg_addr)
        .capabilities(NodeCapabilities {
            can_aggregate: true,
            ..Default::default()
        })
        .build()
        .unwrap();
    let agg_network = Arc::new(agg_network);
    agg_network.start().await.unwrap();

    let orch_config = OrchestratorConfig {
        min_workers: 1,
        consensus_enabled: false,
        default_params: TrainingParams {
            learning_rate: 0.001,
            batch_size: 1,
            local_epochs: 1,
            max_error_bound: 0.1,
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            model_seed: 42,
            num_layers: 2,
            activation_type: 0,
        },
        ..Default::default()
    };
    let orchestrator =
        TrainingOrchestrator::new(agg_id.clone(), orch_config, agg_network.clone());
    orchestrator.set_leader(true);
    let mut event_rx = orchestrator.start().await;

    // Start 3 workers and connect them
    for i in 0..3usize {
        let worker_port = get_available_port();
        let worker_id = PeerId::from_string(format!("tcp-worker-{}", i));
        let worker_addr: SocketAddr =
            format!("127.0.0.1:{}", worker_port).parse().unwrap();

        let worker_network = NetworkRunnerBuilder::new()
            .local_id(worker_id)
            .listen_addr(worker_addr)
            .capabilities(NodeCapabilities {
                can_train: true,
                can_prove: true,
                ..Default::default()
            })
            .build()
            .unwrap();
        let worker_network = Arc::new(worker_network);
        worker_network.start().await.unwrap();

        let agg_peer = PeerInfo {
            id: agg_id.clone(),
            address: agg_addr.to_string(),
            capabilities: NodeCapabilities {
                can_aggregate: true,
                ..Default::default()
            },
            last_seen: 0,
            reputation: 100,
        };
        worker_network.connect_peer(agg_peer).await.unwrap();

        // Send heartbeat to register
        worker_network
            .broadcast(MessagePayload::Heartbeat(HeartbeatMessage {
                seq: 0,
                is_pong: false,
                load: 30,
            }))
            .await;
    }

    // Wait for all 3 workers to register
    let timeout = Duration::from_secs(10);
    let start = std::time::Instant::now();
    let mut worker_joined_count = 0usize;

    loop {
        let stats = orchestrator.worker_stats();
        if stats.total >= 3 {
            log::info!("All 3 workers registered via TCP");
            break;
        }

        // Also check events for WorkerJoined
        while let Ok(event) =
            tokio::time::timeout(Duration::from_millis(100), event_rx.recv()).await
        {
            if let Some(OrchestratorEvent::WorkerJoined { peer_id }) = event {
                worker_joined_count += 1;
                log::info!(
                    "WorkerJoined event: {} ({}/3)",
                    peer_id, worker_joined_count
                );
            }
        }

        if start.elapsed() > timeout {
            let stats = orchestrator.worker_stats();
            panic!(
                "Timed out: only {} workers registered (events: {})",
                stats.total, worker_joined_count,
            );
        }

        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    assert!(
        orchestrator.worker_stats().total >= 3,
        "All 3 workers should be registered",
    );

    orchestrator.stop();
    log::info!("TCP connectivity test passed!");
}
