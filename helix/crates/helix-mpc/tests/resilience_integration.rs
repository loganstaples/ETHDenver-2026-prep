//! Integration tests for network resilience, error handling, and security hardening.
//!
//! Tests cover:
//! 1. Worker disconnect with grace period and recovery
//! 2. Chain failures with exponential backoff retry
//! 3. Beaver triple exhaustion and transparent replenishment
//! 4. Graceful shutdown with emergency checkpoint save
//! 5. Checkpoint resume from saved state
//! 6. Malformed message rejection
//! 7. Message authentication (HMAC-SHA256)
//! 8. Error message sanitization (no secret leakage)
//! 9. Worker disconnect with honest majority lost
//! 10. Full resilience scenario combining multiple failure modes

use std::path::Path;
use std::time::Duration;

use helix_mpc::error::MPCError;
use helix_mpc::field::Fr;
use helix_mpc::graceful_shutdown::{EmergencyCheckpointState, GracefulShutdown};
use helix_mpc::message_validation::{
    sanitize_error, sanitize_party_id, AuthenticatedMessage, MessageValidator, ValidationConfig,
    derive_session_key, validate_field_elements, validate_training_message,
};
use helix_mpc::recovery::DisconnectionHandler;
use helix_mpc::resilience::{ChainRetrier, PendingCheckpoint, ResilientTriplePool, RetryConfig};
use helix_mpc::resilient_training::{
    find_latest_checkpoint, load_emergency_checkpoint, ResilientTrainingConfig, TrainingState,
};
use helix_mpc::types::PartyId;

// ============================================================================
// 1. Worker Disconnect Recovery
// ============================================================================

#[test]
fn test_worker_disconnect_recovery() {
    // Scenario: 3 parties, 1 disconnects, 2 remain (honest majority).
    let mut handler = DisconnectionHandler::new(3, 2, Duration::from_millis(10));

    // All parties active initially.
    assert_eq!(handler.active_count(), 3);
    assert!(handler.has_honest_majority());

    // Let grace period expire for all parties.
    std::thread::sleep(Duration::from_millis(20));

    // Only party 0 and 1 send heartbeats.
    handler.record_activity(0);
    handler.record_activity(1);

    // Check disconnections — party 2 should be detected.
    let disconnected = handler.check_disconnections();
    assert!(disconnected.contains(&2), "Party 2 should be disconnected");
    assert!(!disconnected.contains(&0), "Party 0 should NOT be disconnected");
    assert!(!disconnected.contains(&1), "Party 1 should NOT be disconnected");

    // Honest majority still holds (2 out of 3, min_honest=2).
    assert!(handler.has_honest_majority());
    assert_eq!(handler.active_count(), 2);
    assert!(handler.is_disconnected(2));
}

#[test]
fn test_worker_disconnect_grace_period_respected() {
    // Scenario: Party stops responding, but reconnects before grace period expires.
    let mut handler = DisconnectionHandler::new(3, 2, Duration::from_millis(100));

    // All parties active.
    handler.record_activity(0);
    handler.record_activity(1);
    handler.record_activity(2);

    // Small sleep (well within grace period).
    std::thread::sleep(Duration::from_millis(20));

    // Check — nobody should be disconnected yet.
    let disconnected = handler.check_disconnections();
    assert!(disconnected.is_empty(), "No disconnections expected during grace period");
    assert_eq!(handler.active_count(), 3);

    // Party 2 sends a heartbeat (simulating reconnection).
    handler.record_activity(2);

    // Wait well past the original grace period.
    std::thread::sleep(Duration::from_millis(120));

    // Only parties 0 and 1 didn't send recent activity.
    handler.record_activity(2);
    let disconnected = handler.check_disconnections();

    // Parties 0 and 1 should be disconnected (no recent activity).
    assert!(disconnected.contains(&0));
    assert!(disconnected.contains(&1));
    // Party 2 should still be active (sent heartbeat recently).
    assert!(!handler.is_disconnected(2));
}

#[test]
fn test_worker_disconnect_explicit_removal() {
    // Scenario: A cheater is identified and explicitly removed.
    let mut handler = DisconnectionHandler::new(3, 2, Duration::from_secs(30));

    assert_eq!(handler.active_count(), 3);

    handler.mark_disconnected(1);
    assert!(handler.is_disconnected(1));
    assert_eq!(handler.active_count(), 2);
    assert!(handler.has_honest_majority());

    // Removing another drops below minimum.
    handler.mark_disconnected(2);
    assert_eq!(handler.active_count(), 1);
    assert!(!handler.has_honest_majority());
}

// ============================================================================
// 2. Chain Failures with Exponential Backoff
// ============================================================================

#[tokio::test]
async fn test_chain_retry_with_backoff() {
    // Scenario: Checkpoint submission fails, is queued, and eventually succeeds on retry.
    let mut retrier = ChainRetrier::new(RetryConfig {
        max_retries: 3,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(10),
        jitter_factor: 0.0,
    });

    // Record a failure.
    let cp = PendingCheckpoint::new(100, "session-1", vec![0xDE, 0xAD]);
    retrier.record_failure(cp);
    assert_eq!(retrier.pending_count(), 1);

    // First retry fails.
    let succeeded = retrier
        .retry_pending(|_| async { Err("chain congested".to_string()) })
        .await;
    assert_eq!(succeeded, 0);
    assert_eq!(retrier.pending_count(), 1);

    // Second retry succeeds.
    let succeeded = retrier.retry_pending(|_| async { Ok(()) }).await;
    assert_eq!(succeeded, 1);
    assert_eq!(retrier.pending_count(), 0);
}

#[tokio::test]
async fn test_chain_retry_max_retries_exceeded() {
    // Scenario: All retries fail — checkpoint is permanently dropped.
    let mut retrier = ChainRetrier::new(RetryConfig {
        max_retries: 2,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_factor: 0.0,
    });

    retrier.record_failure(PendingCheckpoint::new(50, "session-1", vec![]));

    // Fail twice (reaches max_retries).
    retrier
        .retry_pending(|_| async { Err("permanent failure".to_string()) })
        .await;
    retrier
        .retry_pending(|_| async { Err("still failing".to_string()) })
        .await;

    // Checkpoint should be dropped after exceeding max retries.
    assert_eq!(retrier.pending_count(), 0, "Checkpoint should be dropped after max retries");
}

#[tokio::test]
async fn test_chain_retry_multiple_pending() {
    // Scenario: Multiple checkpoints fail, some succeed on retry, some fail permanently.
    let mut retrier = ChainRetrier::new(RetryConfig {
        max_retries: 3,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_factor: 0.0,
    });

    retrier.record_failure(PendingCheckpoint::new(10, "session-1", vec![]));
    retrier.record_failure(PendingCheckpoint::new(20, "session-1", vec![]));
    retrier.record_failure(PendingCheckpoint::new(30, "session-1", vec![]));
    assert_eq!(retrier.pending_count(), 3);

    // Retry: only step 20 succeeds.
    let succeeded = retrier
        .retry_pending(|cp| {
            let step = cp.step;
            async move {
                if step == 20 {
                    Ok(())
                } else {
                    Err("fail".to_string())
                }
            }
        })
        .await;

    assert_eq!(succeeded, 1);
    assert_eq!(retrier.pending_count(), 2, "Steps 10 and 30 should still be pending");
}

#[test]
fn test_chain_retry_exponential_backoff_delays() {
    let retrier = ChainRetrier::new(RetryConfig {
        max_retries: 5,
        base_delay: Duration::from_secs(1),
        max_delay: Duration::from_secs(30),
        jitter_factor: 0.0,
    });

    assert_eq!(retrier.compute_delay(1), Duration::from_secs(1));
    assert_eq!(retrier.compute_delay(2), Duration::from_secs(2));
    assert_eq!(retrier.compute_delay(3), Duration::from_secs(4));
    assert_eq!(retrier.compute_delay(4), Duration::from_secs(8));
    assert_eq!(retrier.compute_delay(5), Duration::from_secs(16));
    // Capped at 30s.
    assert_eq!(retrier.compute_delay(6), Duration::from_secs(30));
}

// ============================================================================
// 3. Beaver Triple Exhaustion Recovery
// ============================================================================

#[test]
fn test_beaver_triple_exhaustion_recovery() {
    // Scenario: Pool is exhausted, replenished, and training continues.
    let mut pool = ResilientTriplePool::new(Vec::new(), 5, 50, 42);

    // Pool starts empty.
    assert!(pool.is_exhausted());
    assert_eq!(pool.remaining(), 0);

    // Attempting to take should fail.
    assert!(pool.take().is_err());

    // Replenish.
    pool.replenish_local();
    assert!(!pool.is_exhausted());
    assert_eq!(pool.remaining(), 50);

    // Consume triples successfully.
    for _ in 0..50 {
        assert!(pool.take().is_ok());
    }

    // Exhausted again.
    assert!(pool.is_exhausted());

    // Take-or-replenish should auto-generate.
    let triple = pool.take_or_replenish();
    assert!(triple.is_ok());
    assert!(pool.remaining() > 0);
}

#[test]
fn test_beaver_triple_low_watermark_signal() {
    // Scenario: Pool signals need for replenishment when below low watermark.
    let mut pool = ResilientTriplePool::new(Vec::new(), 10, 100, 42);

    // Start with a replenishment.
    pool.replenish_local();
    assert_eq!(pool.remaining(), 100);
    assert!(!pool.needs_replenishment());

    // Consume down to just above watermark.
    for _ in 0..90 {
        pool.take().unwrap();
    }
    assert_eq!(pool.remaining(), 10);
    assert!(!pool.needs_replenishment());

    // One more puts us below watermark.
    pool.take().unwrap();
    assert!(pool.needs_replenishment());
    assert_eq!(pool.remaining(), 9);
}

#[test]
fn test_beaver_triple_stats_tracking() {
    let mut pool = ResilientTriplePool::new(Vec::new(), 5, 20, 42);

    pool.replenish_local();
    let stats = pool.stats();
    assert_eq!(stats.remaining, 20);
    assert_eq!(stats.total_generated, 20);
    assert_eq!(stats.total_consumed, 0);

    for _ in 0..10 {
        pool.take().unwrap();
    }

    let stats = pool.stats();
    assert_eq!(stats.remaining, 10);
    assert_eq!(stats.total_consumed, 10);
    assert_eq!(stats.total_generated, 20);
}

// ============================================================================
// 4. Graceful Shutdown Saves State
// ============================================================================

#[tokio::test]
async fn test_graceful_shutdown_saves_state() {
    let shutdown = GracefulShutdown::new();
    let dir = tempfile::tempdir().unwrap();

    // Simulate training state.
    let w1 = vec![Fr::from_f64(1.5), Fr::from_f64(-0.3), Fr::from_f64(2.1), Fr::from_f64(0.7)];
    let b1 = vec![Fr::from_f64(0.01), Fr::from_f64(-0.02)];
    let w2 = vec![Fr::from_f64(0.5), Fr::from_f64(-0.6)];
    let b2 = vec![Fr::from_f64(0.001)];
    let w1_macs = vec![Fr::from_f64(10.0), Fr::from_f64(20.0), Fr::from_f64(30.0), Fr::from_f64(40.0)];
    let b1_macs = vec![Fr::from_f64(1.0), Fr::from_f64(2.0)];
    let w2_macs = vec![Fr::from_f64(5.0), Fr::from_f64(6.0)];
    let b2_macs = vec![Fr::from_f64(0.1)];

    let state = EmergencyCheckpointState::from_shares(
        "resilience-test-session",
        0,
        42,
        &w1, &b1, &w2, &b2,
        &w1_macs, &b1_macs, &w2_macs, &b2_macs,
        1000, 500,
        "graceful_shutdown",
    );

    // Trigger shutdown.
    shutdown.trigger();
    assert!(shutdown.is_shutting_down());

    // Save emergency checkpoint.
    let path = shutdown.emergency_checkpoint(&state, dir.path()).await.unwrap();
    assert!(path.exists());

    // Verify file contents.
    let data = std::fs::read(&path).unwrap();
    let restored: EmergencyCheckpointState = serde_json::from_slice(&data).unwrap();

    assert_eq!(restored.session_id, "resilience-test-session");
    assert_eq!(restored.party_index, 0);
    assert_eq!(restored.step, 42);
    assert_eq!(restored.beaver_cursor, 1000);
    assert_eq!(restored.auth_beaver_cursor, 500);
    assert_eq!(restored.reason, "graceful_shutdown");
    assert!(restored.mac_active);
}

// ============================================================================
// 5. Checkpoint Resume from Saved State
// ============================================================================

#[tokio::test]
async fn test_graceful_shutdown_resume() {
    let shutdown = GracefulShutdown::new();
    let dir = tempfile::tempdir().unwrap();

    // Create and save an emergency checkpoint.
    let original_w1 = vec![Fr::from_f64(1.0), Fr::from_f64(2.0)];
    let original_b1 = vec![Fr::from_f64(0.1)];
    let original_w2 = vec![Fr::from_f64(0.5)];
    let original_b2 = vec![Fr::from_f64(0.01)];

    let state = EmergencyCheckpointState::from_shares(
        "resume-session",
        1, // party 1
        100,
        &original_w1, &original_b1, &original_w2, &original_b2,
        &[], &[], &[], &[], // No MACs for this test.
        200, 100,
        "planned_shutdown",
    );

    let path = shutdown.emergency_checkpoint(&state, dir.path()).await.unwrap();

    // Load checkpoint back.
    let restored = load_emergency_checkpoint(&path).unwrap();
    assert_eq!(restored.session_id, "resume-session");
    assert_eq!(restored.party_index, 1);
    assert_eq!(restored.step, 100);
    assert_eq!(restored.beaver_cursor, 200);
    assert_eq!(restored.auth_beaver_cursor, 100);
    assert!(!restored.mac_active);

    // Verify weight shares roundtrip exactly.
    let (rw1, rb1, rw2, rb2) = restored.restore_shares().unwrap();
    assert_eq!(rw1.len(), 2);
    assert!((rw1[0].to_f64() - 1.0).abs() < 1e-6);
    assert!((rw1[1].to_f64() - 2.0).abs() < 1e-6);
    assert!((rb1[0].to_f64() - 0.1).abs() < 1e-6);
    assert!((rw2[0].to_f64() - 0.5).abs() < 1e-6);
    assert!((rb2[0].to_f64() - 0.01).abs() < 1e-6);
}

#[tokio::test]
async fn test_find_latest_checkpoint_selects_most_recent() {
    let dir = tempfile::tempdir().unwrap();
    let shutdown = GracefulShutdown::new();

    // Create checkpoint at step 10.
    let state10 = EmergencyCheckpointState::from_shares(
        "find-test", 0, 10,
        &[Fr::from_f64(1.0)], &[], &[], &[],
        &[], &[], &[], &[],
        0, 0, "test",
    );
    shutdown.emergency_checkpoint(&state10, dir.path()).await.unwrap();

    // Brief sleep for mtime distinction.
    std::thread::sleep(Duration::from_millis(20));

    // Create checkpoint at step 20.
    let state20 = EmergencyCheckpointState::from_shares(
        "find-test", 0, 20,
        &[Fr::from_f64(2.0)], &[], &[], &[],
        &[], &[], &[], &[],
        0, 0, "test",
    );
    shutdown.emergency_checkpoint(&state20, dir.path()).await.unwrap();

    // Brief sleep for mtime distinction.
    std::thread::sleep(Duration::from_millis(20));

    // Create checkpoint at step 30 (most recent).
    let state30 = EmergencyCheckpointState::from_shares(
        "find-test", 0, 30,
        &[Fr::from_f64(3.0)], &[], &[], &[],
        &[], &[], &[], &[],
        0, 0, "test",
    );
    shutdown.emergency_checkpoint(&state30, dir.path()).await.unwrap();

    // Find latest should return step 30.
    let latest = find_latest_checkpoint(dir.path()).unwrap();
    assert!(latest.is_some());
    let latest_path = latest.unwrap();
    assert!(
        latest_path.to_string_lossy().contains("step30"),
        "Expected latest checkpoint to be step 30, got: {}",
        latest_path.display()
    );

    // Verify content.
    let restored = load_emergency_checkpoint(&latest_path).unwrap();
    assert_eq!(restored.step, 30);
}

#[test]
fn test_find_latest_checkpoint_nonexistent_dir() {
    let result = find_latest_checkpoint(Path::new("/nonexistent/dir/for/testing"));
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn test_find_latest_checkpoint_empty_dir() {
    let dir = tempfile::tempdir().unwrap();
    let result = find_latest_checkpoint(dir.path()).unwrap();
    assert!(result.is_none());
}

// ============================================================================
// 6. Malformed Message Rejection
// ============================================================================

#[test]
fn test_malformed_message_rejected_too_short() {
    // A message that is too short to even contain the header.
    let result = AuthenticatedMessage::from_bytes(&[0u8; 10]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{}", err).contains("too short"),
        "Expected 'too short' error, got: {}",
        err,
    );
}

#[test]
fn test_malformed_message_rejected_size_mismatch() {
    // Message with payload length header that doesn't match actual size.
    let mut data = vec![0u8; 100];
    // Set payload length to 200 (larger than remaining bytes).
    data[0..4].copy_from_slice(&200u32.to_be_bytes());
    let result = AuthenticatedMessage::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_malformed_message_rejected_oversized() {
    // Message claiming to be larger than MAX_MESSAGE_SIZE.
    let mut data = vec![0u8; 44]; // minimum valid size
    // Set payload length to 9MB (exceeds 8MB max).
    data[0..4].copy_from_slice(&(9 * 1024 * 1024u32).to_be_bytes());
    let result = AuthenticatedMessage::from_bytes(&data);
    assert!(result.is_err());
    match result.unwrap_err() {
        MPCError::MessageTooLarge { size, max_size } => {
            assert!(size > max_size);
        }
        other => panic!("Expected MessageTooLarge, got: {:?}", other),
    }
}

#[test]
fn test_training_message_validation_empty() {
    assert!(validate_training_message(&[]).is_err());
}

#[test]
fn test_training_message_validation_too_short_for_tag() {
    assert!(validate_training_message(&[1, 2]).is_err());
}

#[test]
fn test_training_message_validation_unknown_variant() {
    let mut msg = vec![0u8; 100];
    msg[0..4].copy_from_slice(&99u32.to_le_bytes());
    assert!(validate_training_message(&msg).is_err());
}

#[test]
fn test_training_message_validation_valid_variants() {
    // All valid TrainingMessage variants (0-8).
    for variant in 0..=8u32 {
        let mut msg = vec![0u8; 100];
        msg[0..4].copy_from_slice(&variant.to_le_bytes());
        assert!(
            validate_training_message(&msg).is_ok(),
            "Variant {} should be valid",
            variant,
        );
    }
}

#[test]
fn test_field_element_validation_correct() {
    let data = vec![0u8; 96]; // 3 elements × 32 bytes
    assert!(validate_field_elements(&data, 3).is_ok());
}

#[test]
fn test_field_element_validation_wrong_size() {
    let data = vec![0u8; 97]; // wrong size for 3 elements
    assert!(validate_field_elements(&data, 3).is_err());
}

#[test]
fn test_field_element_validation_too_many() {
    assert!(validate_field_elements(&[], 1_000_001).is_err());
}

// ============================================================================
// 7. Message Authentication (HMAC-SHA256)
// ============================================================================

#[test]
fn test_message_authentication_roundtrip() {
    let key = derive_session_key(&[0x42u8; 32], "test-session");
    let payload = b"authenticated MPC message".to_vec();

    let msg = AuthenticatedMessage::create(payload.clone(), 1, &key).unwrap();
    let bytes = msg.to_bytes();
    let restored = AuthenticatedMessage::from_bytes(&bytes).unwrap();

    assert_eq!(restored.sequence, 1);
    assert_eq!(restored.payload, payload);
    restored.verify(&key).unwrap();
}

#[test]
fn test_message_authentication_wrong_key_fails() {
    let key1 = derive_session_key(&[0x42u8; 32], "session-1");
    let key2 = derive_session_key(&[0x43u8; 32], "session-2");

    let msg = AuthenticatedMessage::create(b"secret".to_vec(), 1, &key1).unwrap();
    let bytes = msg.to_bytes();
    let restored = AuthenticatedMessage::from_bytes(&bytes).unwrap();

    // Verification with wrong key should fail.
    assert!(restored.verify(&key2).is_err());
}

#[test]
fn test_message_authentication_tamper_detection() {
    let key = derive_session_key(&[0x42u8; 32], "tamper-test");
    let msg = AuthenticatedMessage::create(b"original data".to_vec(), 1, &key).unwrap();
    let mut bytes = msg.to_bytes();

    // Tamper with the payload.
    if bytes.len() > 12 {
        bytes[12] ^= 0xFF;
    }

    let restored = AuthenticatedMessage::from_bytes(&bytes).unwrap();
    assert!(restored.verify(&key).is_err(), "Tampered message should fail verification");
}

#[test]
fn test_message_authentication_sequence_integrity() {
    let key = [0xABu8; 32];

    // Create two messages with same payload but different sequence numbers.
    let msg1 = AuthenticatedMessage::create(b"data".to_vec(), 1, &key).unwrap();
    let msg2 = AuthenticatedMessage::create(b"data".to_vec(), 2, &key).unwrap();

    // Tags should differ (sequence is included in HMAC).
    assert_ne!(msg1.tag, msg2.tag, "Different sequences should produce different tags");
}

#[test]
fn test_validator_replay_protection() {
    let mut validator = MessageValidator::new(ValidationConfig {
        require_authentication: false,
        require_replay_protection: true,
        rate_limit_per_second: 0,
        ..Default::default()
    });
    let party = PartyId::from_index(0);
    let key = [0u8; 32];
    validator.register_session_key(&party, key);

    // Send messages with increasing sequence numbers.
    let msg1 = AuthenticatedMessage::create(b"first".to_vec(), 1, &key).unwrap();
    validator
        .validate_incoming(&msg1.to_bytes(), &party)
        .expect("First message should be accepted");

    let msg2 = AuthenticatedMessage::create(b"second".to_vec(), 2, &key).unwrap();
    validator
        .validate_incoming(&msg2.to_bytes(), &party)
        .expect("Second message should be accepted");

    // Replay of sequence 1 should fail.
    let replay = AuthenticatedMessage::create(b"replay".to_vec(), 1, &key).unwrap();
    let result = validator.validate_incoming(&replay.to_bytes(), &party);
    assert!(result.is_err(), "Replay should be rejected");
    match result.unwrap_err() {
        MPCError::ReplayAttack { sequence, .. } => {
            assert_eq!(sequence, 1);
        }
        other => panic!("Expected ReplayAttack, got: {:?}", other),
    }
}

#[test]
fn test_validator_rate_limiting() {
    let mut validator = MessageValidator::new(ValidationConfig {
        require_authentication: false,
        require_replay_protection: false,
        rate_limit_per_second: 5,
        rate_limit_window: Duration::from_secs(1),
        ..Default::default()
    });
    let party = PartyId::from_index(0);
    let key = [0u8; 32];

    // Send 5 messages (within limit).
    for i in 0..5 {
        let msg = AuthenticatedMessage::create(
            format!("msg-{}", i).into_bytes(),
            i + 1,
            &key,
        )
        .unwrap();
        assert!(
            validator.validate_incoming(&msg.to_bytes(), &party).is_ok(),
            "Message {} should be accepted (within rate limit)",
            i,
        );
    }

    // 6th message should be rate-limited.
    let overflow = AuthenticatedMessage::create(b"overflow".to_vec(), 6, &key).unwrap();
    assert!(
        validator.validate_incoming(&overflow.to_bytes(), &party).is_err(),
        "6th message should be rate-limited"
    );
}

#[test]
fn test_validator_full_authentication_flow() {
    let shared_secret = [0x42u8; 32];
    let session_key = derive_session_key(&shared_secret, "auth-flow-test");

    let mut validator = MessageValidator::new(ValidationConfig::default());
    let party_a = PartyId::from_index(0);
    let party_b = PartyId::from_index(1);

    validator.register_session_key(&party_a, session_key);
    validator.register_session_key(&party_b, session_key);

    // Party A creates an authenticated message.
    let msg = AuthenticatedMessage::create(b"authenticated data".to_vec(), 1, &session_key).unwrap();
    let raw = msg.to_bytes();

    // Party B validates it.
    let payload = validator.validate_incoming(&raw, &party_a).unwrap();
    assert_eq!(payload, b"authenticated data");
}

// ============================================================================
// 8. Error Message Sanitization
// ============================================================================

#[test]
fn test_error_message_sanitization() {
    // Communication errors should not leak addresses or secrets.
    let err = MPCError::CommunicationError(
        "tcp connection to 192.168.1.100:8080 failed: secret_share=0x1234abcdef5678901234abcdef".into(),
    );
    let sanitized = sanitize_error(&err);
    assert_eq!(sanitized, "Communication error with peer");
    assert!(!sanitized.contains("192.168"));
    assert!(!sanitized.contains("1234abcdef"));
    assert!(!sanitized.contains("secret_share"));
}

#[test]
fn test_error_sanitization_protocol_error() {
    let err = MPCError::ProtocolError(
        "internal state inconsistency: share_value=0xDEADBEEF01234567890ABCDEF".into(),
    );
    let sanitized = sanitize_error(&err);
    assert_eq!(sanitized, "Protocol error in MPC message");
    assert!(!sanitized.contains("DEADBEEF"));
    assert!(!sanitized.contains("internal state"));
}

#[test]
fn test_error_sanitization_mac_failure_preserves_step() {
    let err = MPCError::MACCheckFailed {
        step: 42,
        cheater: Some(2),
    };
    let sanitized = sanitize_error(&err);
    assert!(sanitized.contains("42"));
    assert!(sanitized.contains("2"));
    assert!(sanitized.contains("MAC check failed"));
}

#[test]
fn test_error_sanitization_beaver_exhaustion() {
    let err = MPCError::BeaverPoolExhausted {
        requested: 100,
        available: 0,
    };
    let sanitized = sanitize_error(&err);
    assert!(sanitized.contains("100"));
    assert!(sanitized.contains("0"));
}

#[test]
fn test_party_id_sanitization_prevents_injection() {
    let party = PartyId::from_index(0);
    let sanitized = sanitize_party_id(&party);
    // Should only contain alphanumeric, hyphens, underscores.
    assert!(
        sanitized.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_'),
        "Sanitized party ID should only contain safe characters: '{}'",
        sanitized,
    );
    assert!(!sanitized.is_empty());
    assert!(sanitized.len() <= 64);
}

#[test]
fn test_session_key_derivation_deterministic() {
    let secret = [0x42u8; 32];

    let key1 = derive_session_key(&secret, "session-1");
    let key2 = derive_session_key(&secret, "session-1");
    assert_eq!(key1, key2, "Same inputs should produce the same key");
}

#[test]
fn test_session_key_derivation_different_sessions() {
    let secret = [0x42u8; 32];

    let key1 = derive_session_key(&secret, "session-1");
    let key2 = derive_session_key(&secret, "session-2");
    assert_ne!(key1, key2, "Different sessions should produce different keys");
}

#[test]
fn test_session_key_derivation_different_secrets() {
    let secret1 = [0x42u8; 32];
    let secret2 = [0x43u8; 32];

    let key1 = derive_session_key(&secret1, "session");
    let key2 = derive_session_key(&secret2, "session");
    assert_ne!(key1, key2, "Different secrets should produce different keys");
}

// ============================================================================
// 9. Worker Disconnect — Honest Majority Lost
// ============================================================================

#[test]
fn test_worker_disconnect_majority_lost() {
    // Scenario: 3 parties, 2 disconnect → honest majority lost.
    let mut handler = DisconnectionHandler::new(3, 2, Duration::from_millis(5));

    // Let grace period expire.
    std::thread::sleep(Duration::from_millis(10));

    // Only party 0 sends heartbeat.
    handler.record_activity(0);
    let disconnected = handler.check_disconnections();

    // Parties 1 and 2 should be disconnected.
    assert!(disconnected.contains(&1));
    assert!(disconnected.contains(&2));
    assert_eq!(handler.active_count(), 1);
    assert!(
        !handler.has_honest_majority(),
        "Honest majority should be lost (1 active, 2 required)"
    );
}

#[test]
fn test_worker_disconnect_majority_lost_all_disconnect() {
    // Scenario: All parties disconnect (e.g., network partition).
    let mut handler = DisconnectionHandler::new(3, 2, Duration::from_millis(5));

    std::thread::sleep(Duration::from_millis(10));

    // Nobody sends a heartbeat.
    let disconnected = handler.check_disconnections();
    assert_eq!(disconnected.len(), 3);
    assert_eq!(handler.active_count(), 0);
    assert!(!handler.has_honest_majority());
}

// ============================================================================
// 10. Full Resilience Scenario — Multiple Failure Modes Combined
// ============================================================================

#[tokio::test]
async fn test_full_resilience_scenario() {
    // This test exercises multiple failure modes in sequence:
    // 1. Normal training starts with 3 parties
    // 2. Beaver triples exhaust and are replenished
    // 3. A chain submission fails and is retried
    // 4. A party disconnects (detected via grace period)
    // 5. An emergency checkpoint is saved

    // Phase 1: Setup components.
    let mut disconnect_handler = DisconnectionHandler::new(3, 2, Duration::from_millis(20));
    let mut triple_pool = ResilientTriplePool::new(Vec::new(), 5, 50, 42);
    let mut chain_retrier = ChainRetrier::new(RetryConfig {
        max_retries: 3,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(10),
        jitter_factor: 0.0,
    });
    let shutdown = GracefulShutdown::new();
    let checkpoint_dir = tempfile::tempdir().unwrap();

    assert_eq!(disconnect_handler.active_count(), 3);
    assert!(triple_pool.is_exhausted());

    // Phase 2: Beaver triple exhaustion → replenish.
    triple_pool.replenish_local();
    assert!(!triple_pool.is_exhausted());
    let stats = triple_pool.stats();
    assert_eq!(stats.remaining, 50);

    // Consume some triples (simulate training).
    for _ in 0..46 {
        triple_pool.take().unwrap();
    }
    assert!(triple_pool.needs_replenishment());

    // Phase 3: Chain submission fails → retry.
    chain_retrier.record_failure(PendingCheckpoint::new(10, "session-1", vec![0x01]));
    assert!(chain_retrier.has_pending());

    // First retry fails.
    let mut attempt = 0u32;
    chain_retrier
        .retry_pending(|_| {
            let a = attempt;
            attempt += 1;
            async move {
                if a == 0 {
                    Err("gas too low".to_string())
                } else {
                    Ok(())
                }
            }
        })
        .await;

    // Second retry succeeds.
    let succeeded = chain_retrier.retry_pending(|_| async { Ok(()) }).await;
    assert_eq!(succeeded, 1);
    assert!(!chain_retrier.has_pending());

    // Phase 4: Party 2 disconnects.
    std::thread::sleep(Duration::from_millis(30));
    disconnect_handler.record_activity(0);
    disconnect_handler.record_activity(1);
    let disconnected = disconnect_handler.check_disconnections();
    assert!(disconnected.contains(&2));
    assert!(disconnect_handler.has_honest_majority()); // 2 remaining, need 2

    // Phase 5: Emergency checkpoint save.
    let state = EmergencyCheckpointState::from_shares(
        "full-scenario",
        0,
        10,
        &[Fr::from_f64(1.0)],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        45, // beaver cursor
        0,
        "worker_disconnected",
    );

    let path = shutdown
        .emergency_checkpoint(&state, checkpoint_dir.path())
        .await
        .unwrap();
    assert!(path.exists());

    // Verify checkpoint can be loaded.
    let restored = load_emergency_checkpoint(&path).unwrap();
    assert_eq!(restored.step, 10);
    assert_eq!(restored.beaver_cursor, 45);
    assert_eq!(restored.reason, "worker_disconnected");
}

#[tokio::test]
async fn test_graceful_shutdown_multiple_subscribers() {
    // Verify shutdown signal reaches all subscribers.
    let shutdown = GracefulShutdown::new();

    let rx1 = shutdown.subscribe();
    let rx2 = shutdown.subscribe();
    let rx3 = shutdown.subscribe();

    assert!(!*rx1.borrow());
    assert!(!*rx2.borrow());
    assert!(!*rx3.borrow());

    shutdown.trigger();

    assert!(*rx1.borrow());
    assert!(*rx2.borrow());
    assert!(*rx3.borrow());
    assert!(shutdown.is_shutting_down());
}

#[test]
fn test_graceful_shutdown_idempotent() {
    let shutdown = GracefulShutdown::new();

    shutdown.trigger();
    shutdown.trigger();
    shutdown.trigger();

    assert!(shutdown.is_shutting_down());
    // Should not panic or misbehave.
}

#[tokio::test]
async fn test_emergency_checkpoint_mac_state_roundtrip() {
    // Verify that MAC state survives the checkpoint save/load cycle.
    let shutdown = GracefulShutdown::new();
    let dir = tempfile::tempdir().unwrap();

    let w1_macs = vec![Fr::from_f64(100.0), Fr::from_f64(200.0)];
    let b1_macs = vec![Fr::from_f64(10.0)];
    let w2_macs = vec![Fr::from_f64(50.0)];
    let b2_macs = vec![Fr::from_f64(5.0)];

    let state = EmergencyCheckpointState::from_shares(
        "mac-roundtrip",
        0,
        50,
        &[Fr::from_f64(1.0), Fr::from_f64(2.0)],
        &[Fr::from_f64(0.1)],
        &[Fr::from_f64(0.5)],
        &[Fr::from_f64(0.01)],
        &w1_macs,
        &b1_macs,
        &w2_macs,
        &b2_macs,
        500,
        250,
        "test",
    );

    let path = shutdown.emergency_checkpoint(&state, dir.path()).await.unwrap();
    let restored = load_emergency_checkpoint(&path).unwrap();

    assert!(restored.mac_active);
    let (rm1, rmb1, rm2, rmb2) = restored.restore_macs().unwrap();
    assert_eq!(rm1.len(), 2);
    assert!((rm1[0].to_f64() - 100.0).abs() < 1e-6);
    assert!((rm1[1].to_f64() - 200.0).abs() < 1e-6);
    assert!((rmb1[0].to_f64() - 10.0).abs() < 1e-6);
    assert!((rm2[0].to_f64() - 50.0).abs() < 1e-6);
    assert!((rmb2[0].to_f64() - 5.0).abs() < 1e-6);
}

#[test]
fn test_resilient_training_config_defaults() {
    let config = ResilientTrainingConfig::default();
    assert_eq!(config.max_steps, 500);
    assert_eq!(config.mac_check_interval, 1);
    assert_eq!(config.checkpoint_interval, 100);
    assert_eq!(config.disconnect_grace_period, Duration::from_secs(30));
    assert_eq!(config.min_parties, 2);
    assert!(config.listen_for_signals);
    assert!(config.validate_messages);
}

#[test]
fn test_resilient_training_config_for_testing() {
    let config = ResilientTrainingConfig::for_testing(50);
    assert_eq!(config.max_steps, 50);
    assert!(!config.listen_for_signals);
    assert!(!config.validate_messages);
    assert_eq!(config.disconnect_grace_period, Duration::from_millis(100));
}

#[test]
fn test_training_state_enum_completeness() {
    // Verify all states exist and are comparable.
    let states = vec![
        TrainingState::Initializing,
        TrainingState::Training,
        TrainingState::PausedTripleExhaustion,
        TrainingState::PausedWorkerDisconnect,
        TrainingState::PausedMajorityLost,
        TrainingState::RecoveringMACFailure,
        TrainingState::RecoveringRedistribution,
        TrainingState::ShuttingDown,
        TrainingState::Completed,
        TrainingState::Failed,
    ];

    // All states should be distinct.
    for (i, a) in states.iter().enumerate() {
        for (j, b) in states.iter().enumerate() {
            if i == j {
                assert_eq!(a, b);
            } else {
                assert_ne!(a, b);
            }
        }
    }
}

#[tokio::test]
async fn test_drain_pending_on_shutdown() {
    // Scenario: On shutdown, remaining pending checkpoints are drained for local storage.
    let mut retrier = ChainRetrier::new(RetryConfig {
        max_retries: 5,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(10),
        jitter_factor: 0.0,
    });

    retrier.record_failure(PendingCheckpoint::new(10, "s1", vec![]));
    retrier.record_failure(PendingCheckpoint::new(20, "s1", vec![]));
    retrier.record_failure(PendingCheckpoint::new(30, "s1", vec![]));

    assert_eq!(retrier.pending_count(), 3);

    let drained = retrier.drain_pending();
    assert_eq!(drained.len(), 3);
    assert_eq!(retrier.pending_count(), 0);

    // Verify drained checkpoints contain expected data.
    let steps: Vec<u64> = drained.iter().map(|c| c.step).collect();
    assert!(steps.contains(&10));
    assert!(steps.contains(&20));
    assert!(steps.contains(&30));
}
