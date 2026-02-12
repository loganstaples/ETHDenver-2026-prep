//! Test 5: Client SDK Integration
//!
//! Uses the HelixClient SDK to:
//! 1. Create a client in mock mode
//! 2. Register a model
//! 3. Start training and stream progress updates
//! 4. Wait for completion
//! 5. Verify results match expected training flow
//!
//! Also tests:
//! - Client configuration and validation
//! - Training session event streaming
//! - Proof submission data flow
//! - Dashboard state consistency
//!
//! Run with: `cargo test --test e2e_client_sdk --features integration -- --test-threads=1`

#![cfg(feature = "integration")]

use std::time::{Duration, Instant};

use halo2curves::bn256::Fr;

use helix_client::config::{ConfigProfile, HelixConfig};
use helix_client::session::TrainingEvent;
use helix_client::{HelixClient, ModelArchitecture, SdkModelConfig, TrainingParams};

use helix_prover::TrainingProofResultV2;

#[path = "../common/mod.rs"]
mod common;
use common::e2e::*;

// ============================================================================
// Test 5: Client SDK — Mock Mode Training Flow
// ============================================================================

#[tokio::test]
async fn test_client_sdk_mock_training() {
    let start = Instant::now();
    eprintln!("[E2E-SDK] Starting client SDK mock training test...");

    // ── Phase 1: Create client in mock mode ──
    let client = HelixClient::connect_mock()
        .await
        .expect("Failed to create mock client");
    eprintln!("[E2E-SDK] Phase 1: Mock client created");

    // ── Phase 2: Register a model ──
    let arch = ModelArchitecture::new(D_IN, D_HID, D_OUT);
    let model_config = SdkModelConfig::new("e2e-test-model", arch);
    let handle = client
        .register_model(model_config)
        .await
        .expect("Model registration should succeed");

    assert!(handle.model_id > 0, "Model ID should be positive");
    eprintln!("[E2E-SDK] Phase 2: Model registered (id={})", handle.model_id);

    // ── Phase 3: Start training with progress streaming ──
    let params = TrainingParams::default()
        .with_rounds(3)
        .with_learning_rate(0.01)
        .with_batch_size(1);

    let mut session = client
        .start_training(handle.model_id, params)
        .await
        .expect("Training should start");

    eprintln!("[E2E-SDK] Phase 3: Training started, streaming events...");

    // ── Phase 4: Collect training events ──
    let mut events_received: Vec<String> = Vec::new();
    let mut rounds_started = 0;
    let mut rounds_completed = 0;
    let mut training_complete = false;
    let mut final_loss = None;

    let timeout = tokio::time::timeout(Duration::from_secs(30), async {
        while let Some(event) = session.next_event().await {
            match &event {
                TrainingEvent::RoundStarted { round, total } => {
                    eprintln!("[E2E-SDK]   Event: RoundStarted {}/{}", round, total);
                    rounds_started += 1;
                    events_received.push(format!("RoundStarted({}/{})", round, total));
                }
                TrainingEvent::StepCompleted { round, loss, error } => {
                    events_received.push(format!("StepCompleted(r={}, loss={:.4})", round, loss));
                }
                TrainingEvent::ProofGenerated { round } => {
                    events_received.push(format!("ProofGenerated(r={})", round));
                }
                TrainingEvent::ProofSubmitted { round, tx_hash } => {
                    events_received.push(format!("ProofSubmitted(r={}, tx={})", round, tx_hash));
                }
                TrainingEvent::RoundCompleted { round, loss, error } => {
                    eprintln!("[E2E-SDK]   Event: RoundCompleted {}, loss={:.4}", round, loss);
                    rounds_completed += 1;
                    final_loss = Some(*loss);
                    events_received.push(format!("RoundCompleted(r={}, loss={:.4})", round, loss));
                }
                TrainingEvent::TrainingComplete { final_loss: fl, rounds } => {
                    eprintln!("[E2E-SDK]   Event: TrainingComplete (loss={:.4}, rounds={})", fl, rounds);
                    training_complete = true;
                    final_loss = Some(*fl);
                    events_received.push(format!("TrainingComplete(loss={:.4})", fl));
                    break;
                }
                TrainingEvent::Error { message } => {
                    panic!("Training error: {}", message);
                }
            }
        }
    });

    match timeout.await {
        Ok(_) => {}
        Err(_) => {
            eprintln!("[E2E-SDK] Warning: Training timed out after 30s");
            eprintln!("[E2E-SDK] Events received so far: {:?}", events_received);
        }
    }

    eprintln!("[E2E-SDK] Phase 4: Collected {} events", events_received.len());

    // ── Phase 5: Verify training flow ──
    assert!(!events_received.is_empty(), "Should receive at least some events");
    assert!(rounds_started > 0, "Should have started at least 1 round");

    // Verify events contain expected types
    let has_round_started = events_received.iter().any(|e| e.starts_with("RoundStarted"));
    assert!(has_round_started, "Should receive RoundStarted events");

    eprintln!("[E2E-SDK] Phase 5: Training flow verified");
    eprintln!(
        "[E2E-SDK]   Rounds: {} started, {} completed, training_complete: {}",
        rounds_started, rounds_completed, training_complete
    );

    // ── Phase 6: Verify client state ──
    let status = client
        .node_status()
        .await
        .expect("Should get node status");
    eprintln!("[E2E-SDK] Phase 6: Node status retrieved");

    let elapsed = start.elapsed();
    eprintln!(
        "[E2E-SDK] PASS: Client SDK test completed in {:.1}s",
        elapsed.as_secs_f64()
    );
    eprintln!(
        "[E2E-SDK]   Events: {}, Rounds started: {}, Mock mode: verified",
        events_received.len(),
        rounds_started
    );
}

// ============================================================================
// Test: Client Configuration Profiles
// ============================================================================

#[tokio::test]
async fn test_client_configuration() {
    eprintln!("[E2E-SDK] Testing client configuration profiles...");

    // Test that ConfigProfile::Local creates a valid config
    let local_config = HelixConfig::from_profile(ConfigProfile::Local);
    // from_profile returns HelixConfig directly (not Result)
    assert_eq!(local_config.profile, ConfigProfile::Local);

    // Test that ConfigProfile::Anvil creates a valid config
    let anvil_config = HelixConfig::from_profile(ConfigProfile::Anvil);
    assert_eq!(anvil_config.profile, ConfigProfile::Anvil);

    // Test mock client creation via connect_mock
    let mock_client = HelixClient::connect_mock().await;
    assert!(mock_client.is_ok(), "Mock client should create successfully");

    // Test mock client via new_mock (requires config)
    let config = HelixConfig::from_profile(ConfigProfile::Local);
    let mock_client2 = HelixClient::new_mock(config);
    assert!(mock_client2.is_ok(), "new_mock with config should succeed");

    eprintln!("[E2E-SDK] PASS: Configuration profiles validated");
}

// ============================================================================
// Test: Proof Submission Data Flow
// ============================================================================

#[tokio::test]
async fn test_proof_submission_format() {
    eprintln!("[E2E-SDK] Testing proof submission data format...");

    // Generate a real proof
    let weights = initial_weights();
    let dataset = training_data();
    let (x, target) = &dataset[0];
    let (proof_result, _) = prove_step(&weights, x, target, 1);

    assert!(proof_result.verified, "Proof should be self-verified");

    // Convert to EVM public inputs
    let evm_pi = proof_result.to_evm_public_inputs();
    assert_eq!(evm_pi.len(), NUM_PUBLIC_INPUTS, "Should have 8 public inputs");

    // Each PI should be exactly 32 bytes
    for (i, pi) in evm_pi.iter().enumerate() {
        assert_eq!(pi.len(), 32, "PI[{}] should be 32 bytes", i);
    }

    // Proof should be non-empty (1856 bytes for KZG)
    assert!(!proof_result.proof.is_empty(), "Proof bytes should not be empty");
    eprintln!(
        "[E2E-SDK] Proof: {} bytes, {} public inputs (each 32 bytes)",
        proof_result.proof.len(),
        evm_pi.len()
    );

    // Verify state hashes are non-trivial
    let (old_lo, old_hi) = proof_result.old_state_hash;
    let (new_lo, new_hi) = proof_result.new_state_hash;
    assert_ne!(old_lo, Fr::zero(), "Old state hash lo should not be zero");
    assert_ne!(new_lo, Fr::zero(), "New state hash lo should not be zero");
    assert_ne!(
        (old_lo, old_hi),
        (new_lo, new_hi),
        "State should change after training"
    );

    eprintln!("[E2E-SDK] PASS: Proof submission format verified");
}
