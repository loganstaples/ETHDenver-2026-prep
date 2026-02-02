//! Integration tests for session establishment and network communication.
//!
//! These tests verify the complete session establishment flow including:
//! - Multi-party key exchange and authentication
//! - TLS-secured network communication
//! - Error handling and recovery
//! - Concurrent session management
//! - Session lifecycle (establishment, usage, rotation, teardown)

#![cfg(test)]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Barrier;
use tokio::time::timeout;

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

use super::channel::{LocalChannel, MPCChannel, Message, MessageType};
use super::establishment::{
    AuthenticationMessage, EstablishedSession, KeyExchangeMessage, SessionConfig,
    SessionEstablishment, SessionPhase, simulate_session_establishment,
};
use super::network::{AsyncMPCChannel, ConnectionState, NetworkChannel, NetworkConfig, TlsConfig};

/// Helper to create test party IDs.
fn test_parties(n: usize) -> Vec<PartyId> {
    (0..n).map(PartyId::from_index).collect()
}

/// Helper to create a test network config.
fn test_network_config(party_id: PartyId, listen_addr: SocketAddr) -> NetworkConfig {
    NetworkConfig {
        listen_addr,
        peer_addrs: HashMap::new(),
        party_id,
        use_tls: false,
        tls_config: None,
        connect_timeout: Duration::from_secs(5),
        max_message_size: 64 * 1024 * 1024,
        max_reconnect_attempts: 3,
        reconnect_delay: Duration::from_millis(100),
    }
}

// =============================================================================
// Session Establishment Tests
// =============================================================================

#[test]
fn test_two_party_session_establishment() {
    let parties = test_parties(2);
    let sessions = simulate_session_establishment(&parties, Duration::from_secs(3600)).unwrap();

    assert_eq!(sessions.len(), 2);

    // Verify session IDs match
    assert_eq!(sessions[0].session_id, sessions[1].session_id);

    // Verify pairwise keys exist and are symmetric
    let key_0_to_1 = sessions[0].pairwise_key(&parties[1].0).unwrap();
    let key_1_to_0 = sessions[1].pairwise_key(&parties[0].0).unwrap();
    assert_eq!(key_0_to_1, key_1_to_0);
}

#[test]
fn test_three_party_session_establishment() {
    let parties = test_parties(3);
    let sessions = simulate_session_establishment(&parties, Duration::from_secs(3600)).unwrap();

    assert_eq!(sessions.len(), 3);

    // All session IDs should match
    let session_id = sessions[0].session_id;
    for s in &sessions[1..] {
        assert_eq!(s.session_id, session_id);
    }

    // All pairwise keys should exist and be symmetric
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                let key_i_to_j = sessions[i].pairwise_key(&parties[j].0);
                let key_j_to_i = sessions[j].pairwise_key(&parties[i].0);
                assert!(key_i_to_j.is_some());
                assert!(key_j_to_i.is_some());
                assert_eq!(key_i_to_j, key_j_to_i);
            }
        }
    }
}

#[test]
fn test_five_party_session_establishment() {
    let parties = test_parties(5);
    let sessions = simulate_session_establishment(&parties, Duration::from_secs(3600)).unwrap();

    assert_eq!(sessions.len(), 5);

    // Verify all session IDs match
    let session_id = sessions[0].session_id;
    for s in &sessions {
        assert_eq!(s.session_id, session_id);
    }

    // Verify derived keys are consistent
    let enc_key = sessions[0].derive_key("encryption");
    for s in &sessions[1..] {
        assert_eq!(s.derive_key("encryption"), enc_key);
    }
}

#[test]
fn test_session_key_derivation_consistency() {
    let parties = test_parties(3);
    let sessions = simulate_session_establishment(&parties, Duration::from_secs(3600)).unwrap();

    // All parties should derive the same keys for the same purpose
    let purposes = ["encryption", "authentication", "mac", "beaver"];
    for purpose in &purposes {
        let key = sessions[0].derive_key(purpose);
        for s in &sessions[1..] {
            assert_eq!(s.derive_key(purpose), key, "Key mismatch for purpose: {}", purpose);
        }
    }

    // Different purposes should give different keys
    let enc_key = sessions[0].derive_key("encryption");
    let auth_key = sessions[0].derive_key("authentication");
    assert_ne!(enc_key, auth_key);
}

#[test]
fn test_session_expiry() {
    let parties = test_parties(2);

    // Create session with very short expiry
    let sessions = simulate_session_establishment(&parties, Duration::from_millis(10)).unwrap();

    // Should not be expired immediately
    assert!(!sessions[0].is_expired());

    // Wait for expiry
    std::thread::sleep(Duration::from_millis(20));
    assert!(sessions[0].is_expired());
}

#[test]
fn test_session_age_tracking() {
    let parties = test_parties(2);
    let sessions = simulate_session_establishment(&parties, Duration::from_secs(3600)).unwrap();

    let initial_age = sessions[0].age();
    std::thread::sleep(Duration::from_millis(50));
    let later_age = sessions[0].age();

    assert!(later_age >= initial_age + Duration::from_millis(40));
}

// =============================================================================
// Session Establishment State Machine Tests
// =============================================================================

#[test]
fn test_session_establishment_phases() {
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use ed25519_dalek::SigningKey;

    let mut rng = ChaCha20Rng::from_entropy();
    let parties = test_parties(2);

    // Generate keys
    let mut signing_keys = Vec::new();
    let mut public_keys = HashMap::new();
    for party in &parties {
        let mut key_bytes = [0u8; 32];
        rng.fill_bytes(&mut key_bytes);
        let signing_key = SigningKey::from_bytes(&key_bytes);
        public_keys.insert(party.0.clone(), signing_key.verifying_key().to_bytes());
        signing_keys.push(signing_key);
    }

    // Create establishments
    let config0 = SessionConfig {
        party_id: parties[0].clone(),
        all_parties: parties.clone(),
        timeout: Duration::from_secs(60),
        signing_key: signing_keys[0].to_bytes(),
        party_public_keys: public_keys.clone(),
    };

    let config1 = SessionConfig {
        party_id: parties[1].clone(),
        all_parties: parties.clone(),
        timeout: Duration::from_secs(60),
        signing_key: signing_keys[1].to_bytes(),
        party_public_keys: public_keys.clone(),
    };

    let mut est0 = SessionEstablishment::new(config0).unwrap();
    let mut est1 = SessionEstablishment::new(config1).unwrap();

    // Initial phase
    assert_eq!(est0.phase(), SessionPhase::Initial);
    assert_eq!(est1.phase(), SessionPhase::Initial);

    // Generate key exchanges
    let ke0 = est0.generate_key_exchange();
    let ke1 = est1.generate_key_exchange();

    assert_eq!(est0.phase(), SessionPhase::KeyExchangeSent);
    assert_eq!(est1.phase(), SessionPhase::KeyExchangeSent);

    // Exchange key exchange messages
    est0.receive_key_exchange(ke1).unwrap();
    est1.receive_key_exchange(ke0).unwrap();

    assert_eq!(est0.phase(), SessionPhase::KeyExchangeComplete);
    assert_eq!(est1.phase(), SessionPhase::KeyExchangeComplete);

    // Generate authentication messages
    let auth0 = est0.generate_authentication().unwrap();
    let auth1 = est1.generate_authentication().unwrap();

    assert_eq!(est0.phase(), SessionPhase::AuthenticationSent);
    assert_eq!(est1.phase(), SessionPhase::AuthenticationSent);

    // Exchange authentication messages
    est0.receive_authentication(auth1).unwrap();
    est1.receive_authentication(auth0).unwrap();

    assert_eq!(est0.phase(), SessionPhase::AuthenticationComplete);
    assert_eq!(est1.phase(), SessionPhase::AuthenticationComplete);

    // Finalize
    let session0 = est0.finalize(Duration::from_secs(3600)).unwrap();
    let session1 = est1.finalize(Duration::from_secs(3600)).unwrap();

    assert_eq!(est0.phase(), SessionPhase::Established);
    assert_eq!(est1.phase(), SessionPhase::Established);

    // Verify sessions match
    assert_eq!(session0.session_id, session1.session_id);
}

#[test]
fn test_invalid_signature_rejection() {
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use ed25519_dalek::SigningKey;

    let mut rng = ChaCha20Rng::from_entropy();
    let parties = test_parties(2);

    // Generate keys
    let mut key_bytes = [0u8; 32];
    rng.fill_bytes(&mut key_bytes);
    let signing_key0 = SigningKey::from_bytes(&key_bytes);

    rng.fill_bytes(&mut key_bytes);
    let signing_key1 = SigningKey::from_bytes(&key_bytes);

    // Create wrong public keys map (give party 0 the wrong key for party 1)
    let mut wrong_public_keys = HashMap::new();
    wrong_public_keys.insert(parties[0].0.clone(), signing_key0.verifying_key().to_bytes());
    // Use a random (wrong) key for party 1
    rng.fill_bytes(&mut key_bytes);
    let wrong_key = SigningKey::from_bytes(&key_bytes);
    wrong_public_keys.insert(parties[1].0.clone(), wrong_key.verifying_key().to_bytes());

    let config0 = SessionConfig {
        party_id: parties[0].clone(),
        all_parties: parties.clone(),
        timeout: Duration::from_secs(60),
        signing_key: signing_key0.to_bytes(),
        party_public_keys: wrong_public_keys.clone(),
    };

    let correct_public_keys = {
        let mut map = HashMap::new();
        map.insert(parties[0].0.clone(), signing_key0.verifying_key().to_bytes());
        map.insert(parties[1].0.clone(), signing_key1.verifying_key().to_bytes());
        map
    };

    let config1 = SessionConfig {
        party_id: parties[1].clone(),
        all_parties: parties.clone(),
        timeout: Duration::from_secs(60),
        signing_key: signing_key1.to_bytes(),
        party_public_keys: correct_public_keys,
    };

    let mut est0 = SessionEstablishment::new(config0).unwrap();
    let mut est1 = SessionEstablishment::new(config1).unwrap();

    let _ke0 = est0.generate_key_exchange();
    let ke1 = est1.generate_key_exchange();

    // Party 0 should reject party 1's message due to signature mismatch
    let result = est0.receive_key_exchange(ke1);
    assert!(result.is_err());
    match result {
        Err(MPCError::MaliciousBehavior { description, .. }) => {
            assert!(description.contains("signature"));
        }
        _ => panic!("Expected MaliciousBehavior error"),
    }
}

// =============================================================================
// Local Channel Tests
// =============================================================================

#[test]
fn test_local_channel_message_ordering() {
    let parties = test_parties(3);
    let channel = LocalChannel::new(&parties);

    // Send multiple messages
    for seq in 0..10 {
        let msg = Message {
            from: parties[0].clone(),
            to: parties[1].clone(),
            msg_type: MessageType::OpenShare,
            payload: vec![seq as u8],
            sequence: seq,
        };
        channel.send(msg).unwrap();
    }

    // Receive all messages
    let received = channel.receive(&parties[1]);
    assert_eq!(received.len(), 10);

    // Verify order preserved
    for (i, msg) in received.iter().enumerate() {
        assert_eq!(msg.sequence, i as u64);
        assert_eq!(msg.payload, vec![i as u8]);
    }
}

#[test]
fn test_local_channel_broadcast_exclusion() {
    let parties = test_parties(4);
    let channel = LocalChannel::new(&parties);

    // Broadcast from party 0
    channel.broadcast(&parties[0], MessageType::Control("ready".into()), vec![1, 2, 3]);

    // Party 0 should not receive the broadcast
    assert_eq!(channel.pending_count(&parties[0]), 0);

    // All other parties should receive it
    for party in &parties[1..] {
        assert_eq!(channel.pending_count(party), 1);
        let msgs = channel.receive(party);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].from, parties[0]);
        assert_eq!(msgs[0].payload, vec![1, 2, 3]);
    }
}

#[test]
fn test_local_channel_multiple_receivers() {
    let parties = test_parties(3);
    let channel = LocalChannel::new(&parties);

    // Party 0 sends to parties 1 and 2
    for &to_idx in &[1, 2] {
        let msg = Message {
            from: parties[0].clone(),
            to: parties[to_idx].clone(),
            msg_type: MessageType::WeightShare,
            payload: vec![to_idx as u8],
            sequence: 0,
        };
        channel.send(msg).unwrap();
    }

    // Each party should only receive their own message
    let msgs1 = channel.receive(&parties[1]);
    let msgs2 = channel.receive(&parties[2]);

    assert_eq!(msgs1.len(), 1);
    assert_eq!(msgs2.len(), 1);
    assert_eq!(msgs1[0].payload, vec![1]);
    assert_eq!(msgs2[0].payload, vec![2]);
}

// =============================================================================
// Network Channel Tests
// =============================================================================

#[tokio::test]
async fn test_network_channel_start_and_shutdown() {
    let parties = test_parties(2);
    let config = test_network_config(parties[0].clone(), "127.0.0.1:0".parse().unwrap());

    let mut channel = NetworkChannel::new(config, &parties);

    // Start should succeed
    channel.start().await.unwrap();

    // Give it time to bind
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Shutdown should be clean
    channel.shutdown().await;
}

#[tokio::test]
async fn test_network_channel_with_tls() {
    // Install rustls crypto provider for TLS operations
    let _ = rustls::crypto::ring::default_provider().install_default();

    let parties = test_parties(2);
    let tls_config = TlsConfig::generate_self_signed(&parties[0].0).unwrap();

    let config = NetworkConfig {
        listen_addr: "127.0.0.1:0".parse().unwrap(),
        peer_addrs: HashMap::new(),
        party_id: parties[0].clone(),
        use_tls: true,
        tls_config: Some(tls_config),
        connect_timeout: Duration::from_secs(5),
        max_message_size: 64 * 1024 * 1024,
        max_reconnect_attempts: 3,
        reconnect_delay: Duration::from_millis(100),
    };

    let mut channel = NetworkChannel::new(config, &parties);

    // Start with TLS should succeed
    channel.start().await.unwrap();

    tokio::time::sleep(Duration::from_millis(50)).await;
    channel.shutdown().await;
}

#[tokio::test]
async fn test_network_channel_connection_state() {
    let parties = test_parties(2);
    let config = test_network_config(parties[0].clone(), "127.0.0.1:0".parse().unwrap());

    let channel = NetworkChannel::new(config, &parties);

    // Before connection, state should be disconnected
    assert_eq!(channel.connection_state(&parties[1]), ConnectionState::Disconnected);
}

// =============================================================================
// Session + Channel Integration Tests
// =============================================================================

#[test]
fn test_session_with_local_channel() {
    let parties = test_parties(3);
    let channel = LocalChannel::new(&parties);

    // Establish session
    let sessions = simulate_session_establishment(&parties, Duration::from_secs(3600)).unwrap();

    // Use session keys to derive encryption keys for channel
    let enc_keys: Vec<[u8; 32]> = sessions.iter().map(|s| s.derive_key("channel")).collect();

    // All derived keys should be identical
    assert_eq!(enc_keys[0], enc_keys[1]);
    assert_eq!(enc_keys[1], enc_keys[2]);

    // Simulate secure message exchange
    for sender_idx in 0..3 {
        for receiver_idx in 0..3 {
            if sender_idx != receiver_idx {
                let msg = Message {
                    from: parties[sender_idx].clone(),
                    to: parties[receiver_idx].clone(),
                    msg_type: MessageType::WeightShare,
                    payload: vec![sender_idx as u8, receiver_idx as u8],
                    sequence: 0,
                };
                channel.send(msg).unwrap();
            }
        }
    }

    // Each party should receive messages from 2 other parties
    for (i, party) in parties.iter().enumerate() {
        let msgs = channel.receive(party);
        assert_eq!(msgs.len(), 2, "Party {} should receive 2 messages", i);
    }
}

#[test]
fn test_concurrent_session_establishment_simulation() {
    // Simulate multiple concurrent sessions
    let results: Vec<_> = (0..5)
        .map(|session_num| {
            let parties = test_parties(3);
            let result = simulate_session_establishment(&parties, Duration::from_secs(3600));
            (session_num, result)
        })
        .collect();

    // All sessions should establish successfully
    for (session_num, result) in results {
        assert!(result.is_ok(), "Session {} failed to establish", session_num);
        let sessions = result.unwrap();

        // Each session should have unique session ID
        let session_id = sessions[0].session_id;
        for s in &sessions[1..] {
            assert_eq!(s.session_id, session_id);
        }
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_session_establishment_with_unknown_party() {
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use ed25519_dalek::SigningKey;

    let mut rng = ChaCha20Rng::from_entropy();
    let parties = test_parties(2);

    // Generate keys for party 0 only
    let mut key_bytes = [0u8; 32];
    rng.fill_bytes(&mut key_bytes);
    let signing_key0 = SigningKey::from_bytes(&key_bytes);

    let mut public_keys = HashMap::new();
    public_keys.insert(parties[0].0.clone(), signing_key0.verifying_key().to_bytes());
    // Don't include party 1's key

    let config0 = SessionConfig {
        party_id: parties[0].clone(),
        all_parties: parties.clone(),
        timeout: Duration::from_secs(60),
        signing_key: signing_key0.to_bytes(),
        party_public_keys: public_keys,
    };

    let mut est0 = SessionEstablishment::new(config0).unwrap();
    let _ke0 = est0.generate_key_exchange();

    // Create a fake key exchange from party 1
    rng.fill_bytes(&mut key_bytes);
    let fake_signing_key = SigningKey::from_bytes(&key_bytes);
    let mut ephemeral = [0u8; 32];
    rng.fill_bytes(&mut ephemeral);
    let mut random = [0u8; 32];
    rng.fill_bytes(&mut random);

    let fake_ke = KeyExchangeMessage::new(
        parties[1].clone(),
        ephemeral,
        &random,
        &fake_signing_key,
    );

    // Should fail because party 1 is unknown
    let result = est0.receive_key_exchange(fake_ke);
    assert!(result.is_err());
    match result {
        Err(MPCError::UnknownParty(_)) => {}
        other => panic!("Expected UnknownParty error, got {:?}", other),
    }
}

#[test]
fn test_session_establishment_commitment_mismatch() {
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use ed25519_dalek::SigningKey;
    use sha2::{Digest, Sha256};

    let mut rng = ChaCha20Rng::from_entropy();
    let parties = test_parties(2);

    // Generate keys
    let mut key_bytes = [0u8; 32];
    rng.fill_bytes(&mut key_bytes);
    let signing_key0 = SigningKey::from_bytes(&key_bytes);

    rng.fill_bytes(&mut key_bytes);
    let signing_key1 = SigningKey::from_bytes(&key_bytes);

    let mut public_keys = HashMap::new();
    public_keys.insert(parties[0].0.clone(), signing_key0.verifying_key().to_bytes());
    public_keys.insert(parties[1].0.clone(), signing_key1.verifying_key().to_bytes());

    let config0 = SessionConfig {
        party_id: parties[0].clone(),
        all_parties: parties.clone(),
        timeout: Duration::from_secs(60),
        signing_key: signing_key0.to_bytes(),
        party_public_keys: public_keys.clone(),
    };

    let config1 = SessionConfig {
        party_id: parties[1].clone(),
        all_parties: parties.clone(),
        timeout: Duration::from_secs(60),
        signing_key: signing_key1.to_bytes(),
        party_public_keys: public_keys,
    };

    let mut est0 = SessionEstablishment::new(config0).unwrap();
    let mut est1 = SessionEstablishment::new(config1).unwrap();

    let ke0 = est0.generate_key_exchange();
    let ke1 = est1.generate_key_exchange();

    est0.receive_key_exchange(ke1.clone()).unwrap();
    est1.receive_key_exchange(ke0).unwrap();

    let auth0 = est0.generate_authentication().unwrap();
    let _auth1 = est1.generate_authentication().unwrap();

    // Create a fake auth message with wrong random value (commitment won't match)
    let mut wrong_random = [0u8; 32];
    rng.fill_bytes(&mut wrong_random);

    let fake_auth = AuthenticationMessage::new(
        parties[1].clone(),
        wrong_random, // Different from what was committed
        auth0.session_id,
        &signing_key1,
    );

    // Should fail because random value doesn't match commitment
    let result = est0.receive_authentication(fake_auth);
    assert!(result.is_err());
    match result {
        Err(MPCError::MaliciousBehavior { description, .. }) => {
            assert!(description.contains("commitment") || description.contains("match"));
        }
        other => panic!("Expected MaliciousBehavior error, got {:?}", other),
    }
}

// =============================================================================
// Performance Tests
// =============================================================================

#[test]
fn test_session_establishment_performance() {
    use std::time::Instant;

    let iterations = 10;
    let party_counts = [2, 3, 5, 10];

    for &n in &party_counts {
        let parties = test_parties(n);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = simulate_session_establishment(&parties, Duration::from_secs(3600)).unwrap();
        }
        let elapsed = start.elapsed();

        let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
        println!(
            "Session establishment for {} parties: {:.2} ms average",
            n, avg_ms
        );

        // Performance assertion: should complete within reasonable time
        // (100ms per party is generous for testing)
        assert!(
            avg_ms < (n as f64 * 100.0),
            "{}-party session took {:.2}ms (expected < {}ms)",
            n,
            avg_ms,
            n * 100
        );
    }
}

#[test]
fn test_key_derivation_performance() {
    use std::time::Instant;

    let parties = test_parties(3);
    let sessions = simulate_session_establishment(&parties, Duration::from_secs(3600)).unwrap();
    let session = &sessions[0];

    let iterations = 10000;
    let start = Instant::now();
    for i in 0..iterations {
        let _ = session.derive_key(&format!("purpose-{}", i));
    }
    let elapsed = start.elapsed();

    let per_derivation_us = elapsed.as_micros() as f64 / iterations as f64;
    println!("Key derivation: {:.2} us per key", per_derivation_us);

    // Should be very fast (< 100us per derivation)
    assert!(
        per_derivation_us < 100.0,
        "Key derivation too slow: {:.2}us",
        per_derivation_us
    );
}

// =============================================================================
// Message Type Coverage Tests
// =============================================================================

#[test]
fn test_all_message_types_through_channel() {
    let parties = test_parties(2);
    let channel = LocalChannel::new(&parties);

    let message_types = vec![
        MessageType::OpenShare,
        MessageType::OpenedValue,
        MessageType::ReshareContribution,
        MessageType::BeaverTriple,
        MessageType::Commitment,
        MessageType::VerificationResult,
        MessageType::Control("test".into()),
        MessageType::GradientShare,
        MessageType::WeightShare,
    ];

    for (seq, msg_type) in message_types.into_iter().enumerate() {
        let msg = Message {
            from: parties[0].clone(),
            to: parties[1].clone(),
            msg_type: msg_type.clone(),
            payload: vec![seq as u8],
            sequence: seq as u64,
        };
        channel.send(msg).unwrap();
    }

    let received = channel.receive(&parties[1]);
    assert_eq!(received.len(), 9);
}
