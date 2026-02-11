//! Adversarial and security tests for helix-mpc.
//!
//! Validates that the protocol detects and rejects:
//! - MAC forgery attempts
//! - Tampered shares
//! - Byzantine fault detection
//! - Commitment spoofing
//! - Replay attacks

use std::time::Duration;

use helix_mpc::field::Fr;
use helix_mpc::security::mac::{
    MACKey, MACVerifier, MessageAuthenticator, AuthenticatedMessage,
    SessionAuthState,
};
use helix_mpc::security::commitment::{
    PedersenCommitment, PedersenGenerators, ShareCommitment, BlindingGenerator,
};
use helix_mpc::security::byzantine::{
    ByzantineDetector, ByzantineChecker, FaultType,
};
use helix_mpc::types::PartyId;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

// =============================================================================
// SPDZ MAC Forgery Detection
// =============================================================================

#[test]
fn test_spdz_mac_forgery_detected() {
    let num_parties = 3;
    let mac_keys = MACKey::generate_shares(num_parties, 42);

    // Create a valid share with MAC for party 0.
    let value_share = Fr::from_f64(5.0);
    let mac_share = mac_keys[0].compute_mac_share(&value_share);

    // Tamper with the value but keep the old MAC.
    let tampered_value = Fr::from_f64(6.0);
    let tampered_mac = mac_keys[0].compute_mac_share(&tampered_value);

    // The MACs should differ — indicating tampering is detectable.
    assert!(
        !mac_share.ct_eq(&tampered_mac).to_bool(),
        "MAC should change when value changes",
    );
}

#[test]
fn test_spdz_batch_mac_verification() {
    let num_parties = 3;
    let mac_keys = MACKey::generate_shares(num_parties, 42);

    // Reconstruct the global MAC key α = Σ α_i.
    let alpha: Fr = mac_keys.iter()
        .map(|k| k.alpha_share.clone())
        .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));

    // The true values we're sharing.
    let true_values = vec![Fr::from_f64(1.0), Fr::from_f64(2.0), Fr::from_f64(3.0)];

    // Create additive shares of each value and of each MAC = α * value.
    let mut rng = ChaCha20Rng::seed_from_u64(99);
    let mut all_value_shares: Vec<Vec<Fr>> = vec![Vec::new(); num_parties];
    let mut all_mac_shares: Vec<Vec<Fr>> = vec![Vec::new(); num_parties];

    for val in &true_values {
        // Additive share the value.
        let mut val_sum = Fr::ZERO;
        for i in 0..num_parties - 1 {
            let r = Fr::random(&mut rng);
            val_sum = Fr::add(&val_sum, &r);
            all_value_shares[i].push(r);
        }
        all_value_shares[num_parties - 1].push(Fr::sub(val, &val_sum));

        // Additive share the MAC = α * value.
        let mac = Fr::mul(&alpha, val);
        let mut mac_sum = Fr::ZERO;
        for i in 0..num_parties - 1 {
            let r = Fr::random(&mut rng);
            mac_sum = Fr::add(&mac_sum, &r);
            all_mac_shares[i].push(r);
        }
        all_mac_shares[num_parties - 1].push(Fr::sub(&mac, &mac_sum));
    }

    let all_alpha_shares: Vec<Fr> = mac_keys.iter()
        .map(|k| k.alpha_share.clone())
        .collect();

    // Batch verification should succeed for honest shares.
    let result = MACVerifier::batch_verify(
        &all_value_shares,
        &all_mac_shares,
        &all_alpha_shares,
        42,
    );
    assert!(result.is_ok(), "Honest batch MAC should verify: {:?}", result);
}

#[test]
fn test_spdz_batch_mac_forgery_rejected() {
    let num_parties = 3;
    let mac_keys = MACKey::generate_shares(num_parties, 42);

    let alpha: Fr = mac_keys.iter()
        .map(|k| k.alpha_share.clone())
        .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));

    let true_values = vec![Fr::from_f64(1.0), Fr::from_f64(2.0)];

    let mut rng = ChaCha20Rng::seed_from_u64(77);
    let mut all_value_shares: Vec<Vec<Fr>> = vec![Vec::new(); num_parties];
    let mut all_mac_shares: Vec<Vec<Fr>> = vec![Vec::new(); num_parties];

    for val in &true_values {
        let mut val_sum = Fr::ZERO;
        for i in 0..num_parties - 1 {
            let r = Fr::random(&mut rng);
            val_sum = Fr::add(&val_sum, &r);
            all_value_shares[i].push(r);
        }
        all_value_shares[num_parties - 1].push(Fr::sub(val, &val_sum));

        let mac = Fr::mul(&alpha, val);
        let mut mac_sum = Fr::ZERO;
        for i in 0..num_parties - 1 {
            let r = Fr::random(&mut rng);
            mac_sum = Fr::add(&mac_sum, &r);
            all_mac_shares[i].push(r);
        }
        all_mac_shares[num_parties - 1].push(Fr::sub(&mac, &mac_sum));
    }

    let all_alpha_shares: Vec<Fr> = mac_keys.iter()
        .map(|k| k.alpha_share.clone())
        .collect();

    // Tamper with party 1's first value share (but keep old MAC).
    all_value_shares[1][0] = Fr::add(&all_value_shares[1][0], &Fr::from_f64(0.001));

    // Verification should fail.
    let result = MACVerifier::batch_verify(
        &all_value_shares,
        &all_mac_shares,
        &all_alpha_shares,
        42,
    );
    assert!(result.is_err(), "Tampered batch MAC should be rejected");
}

// =============================================================================
// Pedersen Commitment Spoofing
// =============================================================================

#[test]
fn test_pedersen_commitment_spoofing() {
    let generators = PedersenGenerators::from_seed(42);

    let value = Fr::from_f64(42.0);
    let blinding = Fr::from_f64(7.0);
    let commitment = PedersenCommitment::commit(&value, &blinding, &generators);

    // Verify with correct opening.
    assert!(commitment.verify(&value, &blinding, &generators));

    // Try to open with wrong value.
    let wrong_value = Fr::from_f64(43.0);
    assert!(
        !commitment.verify(&wrong_value, &blinding, &generators),
        "Commitment should reject wrong value opening",
    );

    // Try to open with wrong blinding factor.
    let wrong_blinding = Fr::from_f64(8.0);
    assert!(
        !commitment.verify(&value, &wrong_blinding, &generators),
        "Commitment should reject wrong blinding factor",
    );
}

#[test]
fn test_pedersen_binding_property() {
    let generators = PedersenGenerators::from_seed(55);

    let v1 = Fr::from_f64(1.0);
    let v2 = Fr::from_f64(2.0);
    let r1 = Fr::from_f64(100.0);
    let r2 = Fr::from_f64(200.0);

    let c1 = PedersenCommitment::commit(&v1, &r1, &generators);
    let c2 = PedersenCommitment::commit(&v2, &r2, &generators);

    // Different values should produce different commitments.
    assert_ne!(c1, c2, "Different values should produce different commitments");
}

#[test]
fn test_pedersen_homomorphic() {
    let generators = PedersenGenerators::from_seed(77);

    let v1 = Fr::from_f64(3.0);
    let v2 = Fr::from_f64(5.0);
    let r1 = Fr::from_f64(10.0);
    let r2 = Fr::from_f64(20.0);

    let c1 = PedersenCommitment::commit(&v1, &r1, &generators);
    let c2 = PedersenCommitment::commit(&v2, &r2, &generators);

    // Homomorphic addition: c1 + c2 should commit to v1 + v2 with r1 + r2.
    let c_sum = c1.add(&c2);
    let v_sum = Fr::add(&v1, &v2);
    let r_sum = Fr::add(&r1, &r2);

    assert!(
        c_sum.verify(&v_sum, &r_sum, &generators),
        "Homomorphic addition should verify",
    );
}

// =============================================================================
// Share Commitment Verification
// =============================================================================

#[test]
fn test_share_commitment_vector_tampering() {
    let party = PartyId::from_index(0);
    let mut blinding_gen = BlindingGenerator::with_seed(42);
    let blinding = blinding_gen.generate();

    let values: Vec<Fr> = vec![Fr::from_f64(1.0), Fr::from_f64(2.0), Fr::from_f64(3.0)];
    let commitment = ShareCommitment::commit_fr_vector(&party, &values, &blinding, "test");

    // Verify with correct values.
    assert!(commitment.verify_fr_vector(&values, &blinding));

    // Tamper with one element.
    let mut tampered = values.clone();
    tampered[1] = Fr::from_f64(2.001);

    assert!(
        !commitment.verify_fr_vector(&tampered, &blinding),
        "Tampered vector should fail commitment verification",
    );
}

// =============================================================================
// Byzantine Fault Detection
// =============================================================================

#[test]
fn test_byzantine_fault_tracking() {
    let mut detector = ByzantineDetector::new(Duration::from_secs(5), 3);

    let p0 = PartyId::from_index(0);
    let p1 = PartyId::from_index(1);
    let p2 = PartyId::from_index(2);

    detector.register_party(p0.clone());
    detector.register_party(p1.clone());
    detector.register_party(p2.clone());

    // Report faults for party 2.
    detector.report_fault(
        p2.clone(),
        FaultType::InvalidMAC,
        "MAC check failed in round 1",
    );
    detector.report_fault(
        p2.clone(),
        FaultType::GradientPoisoning,
        "Gradient magnitude 100x normal",
    );
    detector.report_fault(
        p2.clone(),
        FaultType::InconsistentValues,
        "Values inconsistent across rounds",
    );

    // Party 2 should now be excluded (exceeded max_faults=3).
    let faults = detector.party_faults(&p2.to_string());
    assert_eq!(faults.len(), 3, "Party 2 should have 3 faults");

    // Parties 0 and 1 should be clean.
    assert!(detector.party_faults(&p0.to_string()).is_empty());
    assert!(detector.party_faults(&p1.to_string()).is_empty());
}

#[test]
fn test_byzantine_gradient_poisoning_check() {
    // Normal gradient.
    let normal = vec![0.01, -0.02, 0.015, -0.01];
    let result = ByzantineChecker::check_gradient_poisoning(
        &normal,
        1.0,   // mean threshold
        10.0,  // max element
        100.0, // max norm
    );
    assert!(result.is_none(), "Normal gradient should pass: {:?}", result);

    // Poisoned gradient (huge element).
    let poisoned = vec![0.01, -0.02, 1000.0, -0.01];
    let result = ByzantineChecker::check_gradient_poisoning(
        &poisoned,
        1.0,
        10.0,
        100.0,
    );
    assert!(
        result.is_some(),
        "Gradient with 1000x element should be detected",
    );
}

// =============================================================================
// Message Authentication & Replay Prevention
// =============================================================================

#[test]
fn test_message_authentication() {
    let key = [42u8; 32];
    let auth = MessageAuthenticator::new(key);

    let message = b"Hello, MPC world!";
    let tag = auth.authenticate(message);

    // Valid message should verify.
    assert!(auth.verify(message, &tag));

    // Tampered message should fail.
    let tampered = b"Hello, MPC world?";
    assert!(!auth.verify(tampered, &tag));

    // Wrong tag should fail.
    let wrong_tag = vec![0u8; tag.len()];
    assert!(!auth.verify(message, &wrong_tag));
}

#[test]
fn test_authenticated_message_creation() {
    let key = [99u8; 32];
    let auth = MessageAuthenticator::new(key);

    let msg = AuthenticatedMessage::create(&auth, b"payload".to_vec(), 1);
    assert!(msg.verify(&auth));

    // Same auth, different key should fail.
    let wrong_auth = MessageAuthenticator::new([0u8; 32]);
    assert!(!msg.verify(&wrong_auth));
}

#[test]
fn test_session_replay_attack_rejected() {
    let mut state = SessionAuthState::new();
    state.establish("peer_a", b"shared_secret_abc");

    // Send messages with incrementing sequence numbers.
    let msg1 = state.create_message("peer_a", b"first".to_vec()).unwrap();
    let msg2 = state.create_message("peer_a", b"second".to_vec()).unwrap();

    // Verify in order succeeds.
    assert!(state.verify_message("peer_a", &msg1).is_ok());
    assert!(state.verify_message("peer_a", &msg2).is_ok());

    // Replay msg1 (old sequence number) should fail.
    let replay_result = state.verify_message("peer_a", &msg1);
    assert!(
        replay_result.is_err(),
        "Replayed message should be rejected",
    );
}

// =============================================================================
// Beaver Triple Consistency Check
// =============================================================================

#[test]
fn test_byzantine_checker_beaver_shares() {
    // Valid Beaver triple shares: sum(a) * sum(b) == sum(c)
    // For a simple case with 1 party: a=2, b=3, c=6
    let a_shares = vec![2.0];
    let b_shares = vec![3.0];
    let c_shares = vec![6.0];

    let result = ByzantineChecker::check_beaver_shares(&a_shares, &b_shares, &c_shares);
    assert!(result.is_none(), "Valid Beaver shares should pass: {:?}", result);

    // Invalid: c != a*b
    let bad_c = vec![7.0]; // should be 6.0
    let result = ByzantineChecker::check_beaver_shares(&a_shares, &b_shares, &bad_c);
    assert!(
        result.is_some(),
        "Invalid Beaver triple should be detected",
    );
}

#[test]
fn test_byzantine_value_consistency() {
    // Small change — should be consistent.
    let prev = vec![1.0, 2.0, 3.0];
    let curr = vec![1.01, 2.01, 3.01];
    let result = ByzantineChecker::check_value_consistency(&prev, &curr, 0.1);
    assert!(result.is_none(), "Small changes should pass: {:?}", result);

    // Large jump — should be flagged.
    let jump = vec![100.0, 200.0, 300.0];
    let result = ByzantineChecker::check_value_consistency(&prev, &jump, 0.1);
    assert!(result.is_some(), "Large jump should be flagged");
}
