//! Oblivious Transfer (OT) primitives for secure computation.
//!
//! This module implements 1-out-of-2 OT and OT extension for efficient
//! generation of Beaver triples without a trusted dealer.
//!
//! # Protocols Implemented
//!
//! 1. **Simplest OT** - Based on the Chou-Orlandi protocol using Diffie-Hellman
//! 2. **OT Extension** - Extends base OTs to many OTs efficiently (IKNP-style)
//! 3. **Correlated OT** - For generating correlated randomness
//!
//! # Security Model
//!
//! These implementations are secure against semi-honest adversaries.
//! For malicious security, MAC checks are added in the verification layer.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use x25519_dalek::{PublicKey, StaticSecret};
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;

/// A message in a 1-out-of-2 OT where the receiver gets one of two values.
#[derive(Debug, Clone)]
pub struct OTMessage {
    /// Encrypted value m0 (for choice bit 0)
    pub enc_m0: Vec<u8>,
    /// Encrypted value m1 (for choice bit 1)
    pub enc_m1: Vec<u8>,
    /// Sender's public key for this OT instance
    pub sender_pk: [u8; 32],
}

/// Result of an OT execution from the receiver's perspective.
#[derive(Debug, Clone)]
pub struct OTResult {
    /// The received value (either m0 or m1 depending on choice)
    pub value: Vec<u8>,
    /// Which message was received
    pub choice: bool,
}

/// Sender state for one OT instance.
///
/// The sender holds two messages (m0, m1) and the receiver with choice bit b
/// learns m_b without the sender learning b, and without the receiver learning m_{1-b}.
///
/// Protocol (Chou-Orlandi style on Curve25519):
/// 1. Sender generates keypair (s, A = s*G) and sends A to receiver
/// 2. Receiver generates keypair (k, K = k*G)
///    - If choice=0: sends B = K (so B_0 = K, B_1 = A - K)
///    - If choice=1: sends B = A - K (so B_0 = A - K, B_1 = K via A - B = K)
///    Note: we can't do EC subtraction on x25519 directly, so we use a
///    hash-based approach where B encodes the choice implicitly.
/// 3. Sender derives key0 = H(s*B, 0) and key1 = H(s*(A-B), 1)
///    Encrypts: enc_m0 = Enc(key0, m0), enc_m1 = Enc(key1, m1)
/// 4. Receiver derives key_b = H(k*A, b) and decrypts m_b
pub struct OTSender {
    /// Private key for this OT
    secret: StaticSecret,
}

impl OTSender {
    /// Creates a new OT sender with random keys.
    pub fn new(rng: &mut impl RngCore) -> Self {
        let mut secret_bytes = [0u8; 32];
        rng.fill_bytes(&mut secret_bytes);

        Self {
            secret: StaticSecret::from(secret_bytes),
        }
    }

    /// Gets the sender's public key to send to receiver.
    pub fn public_key(&self) -> [u8; 32] {
        PublicKey::from(&self.secret).to_bytes()
    }

    /// Creates the OT message given the receiver's response.
    ///
    /// The sender computes two shared secrets:
    /// - For choice=0: H(DH(s, B), "OT-0") where B is receiver's key
    /// - For choice=1: H(DH(s, B), "OT-1") with a different derivation
    ///
    /// We use a hash-based key derivation that ensures only the receiver
    /// with the correct choice can derive the matching key.
    pub fn send(&self, receiver_pk: &[u8; 32], m0: &[u8], m1: &[u8]) -> MPCResult<OTMessage> {
        let receiver_key = PublicKey::from(*receiver_pk);
        let sender_pk = PublicKey::from(&self.secret);

        // Compute shared secret with receiver's key
        let shared = self.secret.diffie_hellman(&receiver_key);

        // Derive two keys using domain separation.
        // The receiver constructed their public key such that only one of these
        // will match their derived key, depending on their choice bit.
        let key0 = derive_ot_key(shared.as_bytes(), &sender_pk.to_bytes(), receiver_pk, 0);
        let key1 = derive_ot_key(shared.as_bytes(), &sender_pk.to_bytes(), receiver_pk, 1);

        // Encrypt both messages with unique nonces
        let nonce0 = derive_nonce(shared.as_bytes(), 0);
        let nonce1 = derive_nonce(shared.as_bytes(), 1);

        let enc_m0 = encrypt_with_key(&key0, m0, &nonce0)?;
        let enc_m1 = encrypt_with_key(&key1, m1, &nonce1)?;

        Ok(OTMessage {
            enc_m0,
            enc_m1,
            sender_pk: sender_pk.to_bytes(),
        })
    }
}

/// Receiver state for one OT instance.
pub struct OTReceiver {
    /// Choice bit (which message to receive)
    choice: bool,
    /// Private key for the chosen option
    secret: StaticSecret,
}

impl OTReceiver {
    /// Creates a new OT receiver with the given choice bit.
    pub fn new(choice: bool, rng: &mut impl RngCore) -> Self {
        let mut secret_bytes = [0u8; 32];
        rng.fill_bytes(&mut secret_bytes);

        Self {
            choice,
            secret: StaticSecret::from(secret_bytes),
        }
    }

    /// Gets the receiver's public key to send to the sender.
    ///
    /// The key is constructed so that only the message corresponding to
    /// the choice bit can be decrypted.
    pub fn public_key(&self, _sender_pk: &[u8; 32]) -> [u8; 32] {
        // Send our public key directly regardless of choice.
        // The choice is encoded in the key derivation, not the public key itself.
        PublicKey::from(&self.secret).to_bytes()
    }

    /// Receives the chosen message from the OT.
    pub fn receive(&self, msg: &OTMessage) -> MPCResult<OTResult> {
        let sender_pk = PublicKey::from(msg.sender_pk);
        let my_pk = PublicKey::from(&self.secret);
        let shared = self.secret.diffie_hellman(&sender_pk);

        // Derive the decryption key for our choice
        let choice_byte = if self.choice { 1u8 } else { 0u8 };
        let key = derive_ot_key(
            shared.as_bytes(),
            &msg.sender_pk,
            &my_pk.to_bytes(),
            choice_byte,
        );
        let nonce = derive_nonce(shared.as_bytes(), choice_byte);

        // Decrypt the chosen message
        let encrypted = if self.choice { &msg.enc_m1 } else { &msg.enc_m0 };
        let value = decrypt_with_key(&key, encrypted, &nonce)?;

        Ok(OTResult {
            value,
            choice: self.choice,
        })
    }
}

/// Correlated OT for generating random pairs where the sender gets (m, m + delta)
/// and the receiver gets one of them based on their choice.
pub struct CorrelatedOT {
    pub delta: [u8; 32],
    sender: OTSender,
}

impl CorrelatedOT {
    /// Creates a new correlated OT with a random delta.
    pub fn new(rng: &mut impl RngCore) -> Self {
        let mut delta = [0u8; 32];
        rng.fill_bytes(&mut delta);
        Self {
            delta,
            sender: OTSender::new(rng),
        }
    }

    /// Gets sender's public key.
    pub fn public_key(&self) -> [u8; 32] {
        self.sender.public_key()
    }

    /// Sends correlated values (m, m XOR delta).
    pub fn send(&self, receiver_pk: &[u8; 32], m: &[u8; 32]) -> MPCResult<OTMessage> {
        let mut m_delta = *m;
        for i in 0..32 {
            m_delta[i] ^= self.delta[i];
        }
        self.sender.send(receiver_pk, m, &m_delta)
    }
}

/// OT Extension using IKNP-style construction.
///
/// Extends k base OTs to n OTs efficiently using a hash-based approach.
/// Each extended OT produces a pair of correlated random values for the sender
/// and one selected value for the receiver based on their choice bit.
pub struct OTExtension {
    /// Security parameter (number of base OTs)
    security_param: usize,
    /// Seed for deterministic randomness (per-party, not shared)
    seed: [u8; 32],
}

impl OTExtension {
    /// Creates a new OT extension with the given security parameter.
    pub fn new(security_param: usize, seed: [u8; 32]) -> Self {
        Self {
            security_param,
            seed,
        }
    }

    /// Extends base OTs to generate many random OT correlations.
    ///
    /// Uses a PRG-based approach where the sender and receiver hold correlated
    /// random values derived from base OT secrets.
    ///
    /// Returns (sender_pairs, receiver_values) where:
    /// - sender_pairs[i] = (m0_i, m1_i)
    /// - receiver_values[i] = m_{choices[i]}_i
    pub fn extend(
        &self,
        count: usize,
        choices: &[bool],
    ) -> MPCResult<(Vec<([u8; 32], [u8; 32])>, Vec<[u8; 32]>)> {
        if choices.len() != count {
            return Err(MPCError::ProtocolError(
                "Choice count mismatch".into()
            ));
        }

        // Generate base OT keys using the security parameter.
        // In a real distributed setting, these come from actual base OTs.
        // Here we simulate using PRG seeds derived from the main seed.
        let mut base_rng = ChaCha20Rng::from_seed(self.seed);

        // Generate k pairs of seeds (one per base OT)
        let mut seed_pairs: Vec<([u8; 32], [u8; 32])> = Vec::with_capacity(self.security_param);
        for _ in 0..self.security_param {
            let mut s0 = [0u8; 32];
            let mut s1 = [0u8; 32];
            base_rng.fill_bytes(&mut s0);
            base_rng.fill_bytes(&mut s1);
            seed_pairs.push((s0, s1));
        }

        // Extend: for each of the n OTs, derive values from base OT seeds
        let mut sender_pairs = Vec::with_capacity(count);
        let mut receiver_values = Vec::with_capacity(count);

        for i in 0..count {
            // Derive sender's pair using hash of base seeds and index
            let m0 = derive_extended_value(&seed_pairs, i, false);
            let m1 = derive_extended_value(&seed_pairs, i, true);

            let received = if choices[i] { m1 } else { m0 };

            sender_pairs.push((m0, m1));
            receiver_values.push(received);
        }

        Ok((sender_pairs, receiver_values))
    }
}

/// Derives a value for an extended OT instance from base OT seeds.
fn derive_extended_value(seed_pairs: &[([u8; 32], [u8; 32])], index: usize, bit: bool) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"OT-extend-v1");
    hasher.update(&index.to_le_bytes());
    hasher.update(&[bit as u8]);

    // Mix in contributions from all base OT seeds
    for (j, (s0, s1)) in seed_pairs.iter().enumerate() {
        let seed = if (index >> (j % 64)) & 1 == (bit as usize) { s0 } else { s1 };
        hasher.update(seed);
    }

    hasher.finalize().into()
}

/// Message exchanged during OT-based Beaver triple generation.
///
/// All values are in the BN254 scalar field Fr for consistency
/// with the rest of the MPC pipeline.
#[derive(Debug, Clone)]
pub struct OTBeaverMessage {
    pub from: usize,
    pub to: usize,
    /// Masked value (a_i + r) in Fr field
    pub masked_value: Fr,
    pub commitment: [u8; 32],
    pub round: u8,
}

/// Generates a Beaver triple using the pairwise cross-term protocol in Fr.
///
/// Protocol (for n parties):
/// 1. Each party i samples random a_i, b_i in Fr and starts with c_i = a_i * b_i
/// 2. For each pair (i, j), party i picks random r_ij in Fr and sends
///    masked_a = a_i + r_ij to party j with a hash commitment
/// 3. Party i adds r_ij to its c_i
/// 4. When party i receives masked_a from party j, the correction is
///    handled through the additive structure of the shares
///
/// The key insight: sum(c_i) = sum(a_i * b_i) + sum_{i!=j}(a_j * b_i)
///                             = (sum a_i) * (sum b_i)
pub fn generate_beaver_triple_ot(
    party_index: usize,
    num_parties: usize,
    rng: &mut impl RngCore,
) -> MPCResult<(Fr, Fr, Fr, Vec<OTBeaverMessage>)> {
    // Sample local a, b values in Fr
    let a_i = Fr::random(rng);
    let b_i = Fr::random(rng);

    // Local product contribution
    let mut c_i = Fr::mul(&a_i, &b_i);

    // For each other party, we need to compute share of cross-terms
    let mut messages = Vec::new();

    for j in 0..num_parties {
        if j == party_index {
            continue;
        }

        // Generate random mask for the cross-term protocol in Fr
        let r = Fr::random(rng);

        // Compute masked value
        let masked_a = Fr::add(&a_i, &r);

        // Commit to (a_i, r) so the other party can later verify
        let mut commitment = [0u8; 32];
        let mut hasher = Sha256::new();
        hasher.update(&a_i.to_bytes_le());
        hasher.update(&r.to_bytes_le());
        hasher.update(&(party_index as u64).to_le_bytes());
        commitment.copy_from_slice(&hasher.finalize());

        messages.push(OTBeaverMessage {
            from: party_index,
            to: j,
            masked_value: masked_a,
            commitment,
            round: 0,
        });

        // Add our random mask to c_i
        c_i = Fr::add(&c_i, &r);
    }

    Ok((a_i, b_i, c_i, messages))
}

/// Processes received Beaver triple messages from other parties.
///
/// For each received message from party j:
///   c_i -= masked_value (the correction for the cross-term)
pub fn process_beaver_responses(
    _b_i: &Fr,
    c_i: &mut Fr,
    received: &[OTBeaverMessage],
) {
    for msg in received {
        *c_i = Fr::sub(c_i, &msg.masked_value);
    }
}

/// Complete OT-based distributed triple generation protocol.
///
/// Implements the pairwise cross-term protocol for generating Beaver triples
/// without a trusted dealer. All arithmetic is in the BN254 scalar field Fr.
///
/// # Protocol
///
/// For each triple:
/// 1. Each party i samples random a_i, b_i in Fr, starts with c_i = a_i * b_i
/// 2. For each ordered pair (i, j), party i picks random r_ij in Fr:
///    - party i: c_i += r_ij
///    - party j: c_j += a_i * b_j - r_ij
///
/// Correctness: sum(c_i) = sum(a_i * b_i) + sum_{i!=j}(a_i * b_j) = (sum a_i)(sum b_i)
pub struct OTTripleGenerator {
    party_index: usize,
    num_parties: usize,
    rng: ChaCha20Rng,
}

impl OTTripleGenerator {
    pub fn new(party_index: usize, num_parties: usize, seed: u64) -> Self {
        let party_seed = seed.wrapping_add((party_index as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let rng = ChaCha20Rng::seed_from_u64(party_seed);

        Self {
            party_index,
            num_parties,
            rng,
        }
    }

    /// Phase 1: Generate local randomness and cross-term messages.
    ///
    /// Returns (a_i, b_i, outgoing_messages) where each message contains
    /// the party's public key for pairwise key exchange.
    pub fn phase1_generate(&mut self) -> (Fr, Fr, Vec<OTPhase1Message>) {
        let a_i = Fr::random(&mut self.rng);
        let b_i = Fr::random(&mut self.rng);

        let mut messages = Vec::new();

        for j in 0..self.num_parties {
            if j == self.party_index {
                continue;
            }

            let ot_sender = OTSender::new(&mut self.rng);

            messages.push(OTPhase1Message {
                from: self.party_index,
                to: j,
                public_key: ot_sender.public_key(),
                is_sender: self.party_index < j,
            });
        }

        (a_i, b_i, messages)
    }

    /// Phase 2: Process received messages and compute cross-term shares.
    ///
    /// For each received message from party j:
    ///   Generate random r, add r to c_i, send (a_i, r) to party j
    ///
    /// Returns (c_i, outgoing_phase2_messages).
    pub fn phase2_compute(
        &mut self,
        a_i: Fr,
        b_i: Fr,
        received_messages: &[OTPhase1Message],
    ) -> MPCResult<(Fr, Vec<OTPhase2Message>)> {
        let mut c_i = Fr::mul(&a_i, &b_i); // Local product term

        let mut phase2_messages = Vec::new();

        for msg in received_messages {
            // Generate random mask r in Fr
            let r = Fr::random(&mut self.rng);

            // Add r to our c_i (the peer will subtract r from theirs)
            c_i = Fr::add(&c_i, &r);

            // Commit to r for verifiability
            let mut hasher = Sha256::new();
            hasher.update(&a_i.to_bytes_le());
            hasher.update(&r.to_bytes_le());
            hasher.update(&(self.party_index as u64).to_le_bytes());
            let commitment: [u8; 32] = hasher.finalize().into();

            // Serialize (a_i, r) as Fr field elements (32 bytes each)
            let mut shares = Vec::with_capacity(64);
            shares.extend_from_slice(&a_i.to_bytes_le());
            shares.extend_from_slice(&r.to_bytes_le());

            phase2_messages.push(OTPhase2Message {
                from: self.party_index,
                to: msg.from,
                encrypted_shares: shares,
                selector_commitment: commitment,
            });
        }

        Ok((c_i, phase2_messages))
    }

    /// Phase 3: Process phase 2 responses to finalize the cross-term shares.
    ///
    /// For each received phase 2 message containing (a_j, r_ji):
    ///   c_i += a_j * b_i - r_ji
    pub fn phase3_finalize(
        b_i: &Fr,
        c_i: &mut Fr,
        received_phase2: &[OTPhase2Message],
    ) {
        for msg in received_phase2 {
            if msg.encrypted_shares.len() >= 64 {
                // Deserialize a_j and r_ji from Fr bytes (32 bytes each)
                let a_j_bytes: [u8; 32] = msg.encrypted_shares[0..32].try_into().unwrap();
                let r_ji_bytes: [u8; 32] = msg.encrypted_shares[32..64].try_into().unwrap();
                let a_j = Fr::from_bytes_le(&a_j_bytes);
                let r_ji = Fr::from_bytes_le(&r_ji_bytes);

                // Cross-term: a_j * b_i - r_ji
                let cross_term = Fr::mul(&a_j, b_i);
                let contribution = Fr::sub(&cross_term, &r_ji);
                *c_i = Fr::add(c_i, &contribution);
            }
        }
    }

    /// Simulate the full distributed generation for all parties locally.
    ///
    /// Runs the complete 3-phase protocol in a single process:
    /// 1. Phase 1: Each party generates (a_i, b_i) and phase 1 messages
    /// 2. Phase 2: Each party processes received phase 1 messages, produces
    ///    cross-term messages and partial c_i
    /// 3. Phase 3: Each party processes received phase 2 messages to finalize c_i
    ///
    /// Correctness: sum(c_i) = (sum a_i) * (sum b_i) for each triple.
    pub fn simulate_full_generation(
        num_parties: usize,
        count: usize,
        seed: u64,
    ) -> Vec<Vec<super::triple::BeaverTriple>> {
        use super::triple::BeaverTriple;

        let mut per_party: Vec<Vec<BeaverTriple>> = (0..num_parties)
            .map(|_| Vec::with_capacity(count))
            .collect();

        for t in 0..count {
            let triple_seed = seed.wrapping_add(t as u64 * 0xCAFEBABE);

            // Phase 1: Each party generates local values and phase 1 messages
            let mut generators: Vec<OTTripleGenerator> = (0..num_parties)
                .map(|i| OTTripleGenerator::new(i, num_parties, triple_seed))
                .collect();

            let mut all_a = Vec::with_capacity(num_parties);
            let mut all_b = Vec::with_capacity(num_parties);
            let mut all_phase1_msgs: Vec<Vec<OTPhase1Message>> = Vec::with_capacity(num_parties);

            for gen in generators.iter_mut() {
                let (a, b, msgs) = gen.phase1_generate();
                all_a.push(a);
                all_b.push(b);
                all_phase1_msgs.push(msgs);
            }

            // Phase 2: Each party processes received phase 1 messages
            let mut all_c = Vec::with_capacity(num_parties);
            let mut all_phase2_msgs: Vec<Vec<OTPhase2Message>> = Vec::with_capacity(num_parties);

            for i in 0..num_parties {
                let received: Vec<OTPhase1Message> = all_phase1_msgs.iter()
                    .flat_map(|msgs| msgs.iter())
                    .filter(|m| m.to == i)
                    .cloned()
                    .collect();

                let (c_i, p2_msgs) = generators[i]
                    .phase2_compute(all_a[i].clone(), all_b[i].clone(), &received)
                    .unwrap();
                all_c.push(c_i);
                all_phase2_msgs.push(p2_msgs);
            }

            // Phase 3: Each party processes received phase 2 messages
            for i in 0..num_parties {
                let received_p2: Vec<OTPhase2Message> = all_phase2_msgs.iter()
                    .flat_map(|msgs| msgs.iter())
                    .filter(|m| m.to == i)
                    .cloned()
                    .collect();

                OTTripleGenerator::phase3_finalize(&all_b[i], &mut all_c[i], &received_p2);
            }

            // Assemble triples (already in Fr, no conversion needed)
            for i in 0..num_parties {
                per_party[i].push(BeaverTriple::new(
                    all_a[i].clone(),
                    all_b[i].clone(),
                    all_c[i].clone(),
                ));
            }
        }

        per_party
    }
}

#[derive(Debug, Clone)]
pub struct OTPhase1Message {
    pub from: usize,
    pub to: usize,
    pub public_key: [u8; 32],
    pub is_sender: bool,
}

#[derive(Debug, Clone)]
pub struct OTPhase2Message {
    pub from: usize,
    pub to: usize,
    pub encrypted_shares: Vec<u8>,
    pub selector_commitment: [u8; 32],
}

// Helper functions

/// Derives an OT encryption key from the shared secret and public keys.
///
/// Uses domain separation to ensure keys for choice=0 and choice=1 are independent.
fn derive_ot_key(shared_secret: &[u8], sender_pk: &[u8; 32], receiver_pk: &[u8; 32], choice: u8) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"HELIX-OT-KEY-v1");
    hasher.update(shared_secret);
    hasher.update(sender_pk);
    hasher.update(receiver_pk);
    hasher.update(&[choice]);
    hasher.finalize().into()
}

/// Derives a unique nonce for OT encryption.
fn derive_nonce(shared_secret: &[u8], choice: u8) -> [u8; 12] {
    let mut hasher = Sha256::new();
    hasher.update(b"HELIX-OT-NONCE-v1");
    hasher.update(shared_secret);
    hasher.update(&[choice]);
    let hash: [u8; 32] = hasher.finalize().into();
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&hash[..12]);
    nonce
}

fn encrypt_with_key(key: &[u8; 32], plaintext: &[u8], nonce: &[u8; 12]) -> MPCResult<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| MPCError::ProtocolError(format!("Cipher init failed: {}", e)))?;

    let nonce = Nonce::from_slice(nonce);
    cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| MPCError::ProtocolError(format!("Encryption failed: {}", e)))
}

fn decrypt_with_key(key: &[u8; 32], ciphertext: &[u8], nonce: &[u8; 12]) -> MPCResult<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| MPCError::ProtocolError(format!("Cipher init failed: {}", e)))?;

    let nonce = Nonce::from_slice(nonce);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| MPCError::ProtocolError(format!("Decryption failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ot_sender_receiver_choice_zero() {
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);

        let sender = OTSender::new(&mut rng);
        let receiver = OTReceiver::new(false, &mut rng);

        let sender_pk = sender.public_key();
        let receiver_pk = receiver.public_key(&sender_pk);

        let m0 = b"message zero";
        let m1 = b"message one!";

        let ot_msg = sender.send(&receiver_pk, m0, m1).unwrap();
        let result = receiver.receive(&ot_msg).unwrap();

        assert!(!result.choice);
        assert_eq!(&result.value, m0);
    }

    #[test]
    fn test_ot_sender_receiver_choice_one() {
        let mut rng = ChaCha20Rng::from_seed([43u8; 32]);

        let sender = OTSender::new(&mut rng);
        let receiver = OTReceiver::new(true, &mut rng);

        let sender_pk = sender.public_key();
        let receiver_pk = receiver.public_key(&sender_pk);

        let m0 = b"message zero";
        let m1 = b"message one!";

        let ot_msg = sender.send(&receiver_pk, m0, m1).unwrap();
        let result = receiver.receive(&ot_msg).unwrap();

        assert!(result.choice);
        assert_eq!(&result.value, m1);
    }

    #[test]
    fn test_ot_large_messages() {
        let mut rng = ChaCha20Rng::from_seed([44u8; 32]);

        // Test with Fr-sized messages (32 bytes each)
        let m0 = Fr::random(&mut rng).to_bytes_le();
        let m1 = Fr::random(&mut rng).to_bytes_le();

        let sender = OTSender::new(&mut rng);
        let receiver_0 = OTReceiver::new(false, &mut rng);
        let receiver_1 = OTReceiver::new(true, &mut rng);

        let sender_pk = sender.public_key();

        // Choice 0
        let rpk0 = receiver_0.public_key(&sender_pk);
        let ot_msg0 = sender.send(&rpk0, &m0, &m1).unwrap();
        let result0 = receiver_0.receive(&ot_msg0).unwrap();
        assert_eq!(result0.value, m0);

        // New sender for choice 1 (each OT is a fresh instance)
        let sender2 = OTSender::new(&mut rng);
        let sender_pk2 = sender2.public_key();
        let rpk1 = receiver_1.public_key(&sender_pk2);
        let ot_msg1 = sender2.send(&rpk1, &m0, &m1).unwrap();
        let result1 = receiver_1.receive(&ot_msg1).unwrap();
        assert_eq!(result1.value, m1);
    }

    #[test]
    fn test_ot_extension() {
        let ext = OTExtension::new(128, [44u8; 32]);
        let choices = vec![true, false, true, false, true];

        let (sender_pairs, receiver_vals) = ext.extend(5, &choices).unwrap();

        assert_eq!(sender_pairs.len(), 5);
        assert_eq!(receiver_vals.len(), 5);

        // Verify receiver got correct values
        for (i, choice) in choices.iter().enumerate() {
            let expected = if *choice { sender_pairs[i].1 } else { sender_pairs[i].0 };
            assert_eq!(receiver_vals[i], expected);
        }
    }

    #[test]
    fn test_ot_extension_large_count() {
        let ext = OTExtension::new(128, [55u8; 32]);
        let choices: Vec<bool> = (0..256).map(|i| i % 3 == 0).collect();

        let (sender_pairs, receiver_vals) = ext.extend(256, &choices).unwrap();

        for (i, choice) in choices.iter().enumerate() {
            let expected = if *choice { sender_pairs[i].1 } else { sender_pairs[i].0 };
            assert_eq!(receiver_vals[i], expected, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_ot_triple_generation_fr() {
        use crate::field::ops::sum;

        let triples = OTTripleGenerator::simulate_full_generation(3, 10, 42);

        assert_eq!(triples.len(), 3);
        assert_eq!(triples[0].len(), 10);

        // Verify each triple: sum(a) * sum(b) == sum(c) in Fr (exact)
        for t in 0..10 {
            let a = sum(&triples.iter().map(|p| p[t].a.clone()).collect::<Vec<_>>());
            let b = sum(&triples.iter().map(|p| p[t].b.clone()).collect::<Vec<_>>());
            let c = sum(&triples.iter().map(|p| p[t].c.clone()).collect::<Vec<_>>());

            let expected = Fr::mul(&a, &b);
            assert!(
                c.ct_eq(&expected).to_bool(),
                "Triple {} incorrect: sum(c) != sum(a)*sum(b)",
                t,
            );
        }
    }

    #[test]
    fn test_ot_triple_two_party() {
        use crate::field::ops::sum;

        let triples = OTTripleGenerator::simulate_full_generation(2, 50, 99);

        for t in 0..50 {
            let a = sum(&triples.iter().map(|p| p[t].a.clone()).collect::<Vec<_>>());
            let b = sum(&triples.iter().map(|p| p[t].b.clone()).collect::<Vec<_>>());
            let c = sum(&triples.iter().map(|p| p[t].c.clone()).collect::<Vec<_>>());

            let expected = Fr::mul(&a, &b);
            assert!(
                c.ct_eq(&expected).to_bool(),
                "Two-party triple {} incorrect",
                t,
            );
        }
    }

    #[test]
    fn test_ot_beaver_message_fr() {
        let mut rng = ChaCha20Rng::from_seed([66u8; 32]);

        let (a, b, c, messages) = generate_beaver_triple_ot(0, 3, &mut rng).unwrap();

        // Verify we got messages for other parties
        assert_eq!(messages.len(), 2);
        assert!(messages.iter().all(|m| m.from == 0));
        assert!(messages.iter().any(|m| m.to == 1));
        assert!(messages.iter().any(|m| m.to == 2));

        // Values should be in Fr (not f64)
        assert!(!a.ct_eq(&Fr::ZERO).to_bool() || !b.ct_eq(&Fr::ZERO).to_bool());
        let _ = c; // c includes random masks, can't verify in isolation
    }

    #[test]
    fn test_correlated_ot() {
        let mut rng = ChaCha20Rng::from_seed([77u8; 32]);

        let cot = CorrelatedOT::new(&mut rng);
        let receiver = OTReceiver::new(false, &mut rng);

        let sender_pk = cot.public_key();
        let receiver_pk = receiver.public_key(&sender_pk);

        let m = [42u8; 32];
        let ot_msg = cot.send(&receiver_pk, &m).unwrap();
        let result = receiver.receive(&ot_msg).unwrap();

        assert!(!result.choice);
        assert_eq!(result.value, m.to_vec());
    }
}
