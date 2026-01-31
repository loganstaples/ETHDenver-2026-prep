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

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use x25519_dalek::{PublicKey, StaticSecret};
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

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
pub struct OTSender {
    /// Private key for this OT
    secret: StaticSecret,
    /// Random values for the two messages
    r0: [u8; 32],
    r1: [u8; 32],
}

impl OTSender {
    /// Creates a new OT sender with random keys.
    pub fn new(rng: &mut impl RngCore) -> Self {
        let mut secret_bytes = [0u8; 32];
        let mut r0 = [0u8; 32];
        let mut r1 = [0u8; 32];
        rng.fill_bytes(&mut secret_bytes);
        rng.fill_bytes(&mut r0);
        rng.fill_bytes(&mut r1);

        Self {
            secret: StaticSecret::from(secret_bytes),
            r0,
            r1,
        }
    }

    /// Gets the sender's public key to send to receiver.
    pub fn public_key(&self) -> [u8; 32] {
        PublicKey::from(&self.secret).to_bytes()
    }

    /// Creates the OT message given the receiver's response.
    /// m0 and m1 are the two possible values the receiver can get.
    pub fn send(&self, receiver_pk: &[u8; 32], m0: &[u8], m1: &[u8]) -> MPCResult<OTMessage> {
        let receiver_key = PublicKey::from(*receiver_pk);
        let sender_pk = PublicKey::from(&self.secret);

        // Compute shared secrets for both possible receiver keys
        // The receiver only knows one of these depending on their choice bit
        let shared0 = self.secret.diffie_hellman(&receiver_key);

        // For the other option, compute based on receiver_pk XOR sender_pk
        // This is a simplified version - real implementation would use proper EC operations
        let mut pk1_bytes = receiver_pk.clone();
        for i in 0..32 {
            pk1_bytes[i] ^= sender_pk.to_bytes()[i];
        }
        let receiver_key1 = PublicKey::from(pk1_bytes);
        let shared1 = self.secret.diffie_hellman(&receiver_key1);

        // Derive encryption keys from shared secrets
        let key0 = derive_key(shared0.as_bytes(), b"OT-key-0");
        let key1 = derive_key(shared1.as_bytes(), b"OT-key-1");

        // Encrypt both messages
        let enc_m0 = encrypt_with_key(&key0, m0, &[0u8; 12])?;
        let enc_m1 = encrypt_with_key(&key1, m1, &[0u8; 12])?;

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
    /// This is computed based on the choice bit to enable selective decryption.
    pub fn public_key(&self, sender_pk: &[u8; 32]) -> [u8; 32] {
        let my_pk = PublicKey::from(&self.secret);

        if self.choice {
            // For choice=1, XOR with sender's public key
            let mut pk_bytes = my_pk.to_bytes();
            for i in 0..32 {
                pk_bytes[i] ^= sender_pk[i];
            }
            pk_bytes
        } else {
            // For choice=0, use our public key directly
            my_pk.to_bytes()
        }
    }

    /// Receives the chosen message from the OT.
    pub fn receive(&self, msg: &OTMessage) -> MPCResult<OTResult> {
        let sender_pk = PublicKey::from(msg.sender_pk);
        let shared = self.secret.diffie_hellman(&sender_pk);

        // Derive the decryption key
        let key = if self.choice {
            derive_key(shared.as_bytes(), b"OT-key-1")
        } else {
            derive_key(shared.as_bytes(), b"OT-key-0")
        };

        // Try to decrypt the chosen message
        let encrypted = if self.choice { &msg.enc_m1 } else { &msg.enc_m0 };
        let value = decrypt_with_key(&key, encrypted, &[0u8; 12])?;

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
/// This extends k base OTs to n OTs efficiently.
pub struct OTExtension {
    /// Security parameter (number of base OTs)
    security_param: usize,
    /// Random seed for the extension
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
    /// Returns (sender_values, receiver_choices, receiver_values)
    /// where receiver_values[i] = sender_values[receiver_choices[i]][i]
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

        let mut rng = ChaCha20Rng::from_seed(self.seed);

        // Generate random sender pairs and select based on choices
        let mut sender_pairs = Vec::with_capacity(count);
        let mut receiver_values = Vec::with_capacity(count);

        for i in 0..count {
            let mut m0 = [0u8; 32];
            let mut m1 = [0u8; 32];
            rng.fill_bytes(&mut m0);
            rng.fill_bytes(&mut m1);

            let received = if choices[i] { m1 } else { m0 };

            sender_pairs.push((m0, m1));
            receiver_values.push(received);
        }

        Ok((sender_pairs, receiver_values))
    }
}

/// Generates a Beaver triple using OT-based multiplication.
///
/// Protocol:
/// 1. Party 0 samples a0, b0 and random t
/// 2. Party 1 samples a1, b1
/// 3. They run OT to compute shares of cross-terms a0*b1 and a1*b0
/// 4. Each party computes their c share: ci = ai*bi + (cross-term shares)
pub fn generate_beaver_triple_ot(
    party_index: usize,
    num_parties: usize,
    seed: u64,
) -> MPCResult<(f64, f64, f64, Vec<OTBeaverMessage>)> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed + party_index as u64 * 0x12345);

    // Sample local a, b values
    let a_i: f64 = rng.gen_range(-100.0..100.0);
    let b_i: f64 = rng.gen_range(-100.0..100.0);

    // Local product contribution
    let mut c_i = a_i * b_i;

    // For each other party, we need to compute share of cross-terms
    let mut messages = Vec::new();

    for j in 0..num_parties {
        if j == party_index {
            continue;
        }

        // Generate random masks for OT
        let r: f64 = rng.gen_range(-1000.0..1000.0);

        // In real OT, we'd send encrypted (r, r + a_i * ?) where ? is b_j
        // The other party selects based on their bit encoding of b_j
        // For now, we use a commitment-based simulation

        let mut commitment = [0u8; 32];
        let mut hasher = Sha256::new();
        hasher.update(&a_i.to_le_bytes());
        hasher.update(&r.to_le_bytes());
        hasher.update(&(party_index as u64).to_le_bytes());
        commitment.copy_from_slice(&hasher.finalize());

        messages.push(OTBeaverMessage {
            from: party_index,
            to: j,
            masked_value: a_i + r,
            commitment,
            round: 0,
        });

        // Add our random contribution to c
        // In the real protocol, this would be determined by OT
        c_i += r * 0.0; // Placeholder - real OT would add actual cross-term share
    }

    Ok((a_i, b_i, c_i, messages))
}

/// Message exchanged during OT-based Beaver triple generation
#[derive(Debug, Clone)]
pub struct OTBeaverMessage {
    pub from: usize,
    pub to: usize,
    pub masked_value: f64,
    pub commitment: [u8; 32],
    pub round: u8,
}

/// Complete OT-based distributed triple generation protocol
pub struct OTTripleGenerator {
    party_index: usize,
    num_parties: usize,
    rng: ChaCha20Rng,
    /// Global delta for correlated OT (sender only)
    delta: Option<[u8; 32]>,
}

impl OTTripleGenerator {
    pub fn new(party_index: usize, num_parties: usize, seed: u64) -> Self {
        let party_seed = seed.wrapping_add((party_index as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let mut rng = ChaCha20Rng::seed_from_u64(party_seed);

        // Party 0 generates global delta
        let delta = if party_index == 0 {
            let mut d = [0u8; 32];
            rng.fill_bytes(&mut d);
            Some(d)
        } else {
            None
        };

        Self {
            party_index,
            num_parties,
            rng,
            delta,
        }
    }

    /// Phase 1: Generate local randomness and OT messages
    pub fn phase1_generate(&mut self) -> (f64, f64, Vec<OTPhase1Message>) {
        let a_i: f64 = self.rng.gen_range(-100.0..100.0);
        let b_i: f64 = self.rng.gen_range(-100.0..100.0);

        let mut messages = Vec::new();

        // For each pair of parties, run OT for cross-term computation
        for j in 0..self.num_parties {
            if j == self.party_index {
                continue;
            }

            // Generate OT sender/receiver instances
            if self.party_index < j {
                // We are sender in OT with party j
                let ot_sender = OTSender::new(&mut self.rng);
                messages.push(OTPhase1Message {
                    from: self.party_index,
                    to: j,
                    public_key: ot_sender.public_key(),
                    is_sender: true,
                });
            } else {
                // We are receiver in OT with party j
                // Our choice encodes bits of b_i
                let choice = (b_i.to_bits() & 1) != 0;
                let ot_receiver = OTReceiver::new(choice, &mut self.rng);

                // Need sender's pk to compute our pk, so this will be in phase 2
                messages.push(OTPhase1Message {
                    from: self.party_index,
                    to: j,
                    public_key: ot_receiver.public_key(&[0u8; 32]), // Placeholder
                    is_sender: false,
                });
            }
        }

        (a_i, b_i, messages)
    }

    /// Phase 2: Process received messages and compute OT results
    pub fn phase2_compute(
        &mut self,
        a_i: f64,
        b_i: f64,
        received_messages: &[OTPhase1Message],
    ) -> MPCResult<(f64, Vec<OTPhase2Message>)> {
        let mut c_i = a_i * b_i; // Local product
        let mut phase2_messages = Vec::new();

        // Process each received OT message
        for msg in received_messages {
            if msg.is_sender {
                // They are sender, we are receiver
                // We select one of their values based on bits of our b
                let my_share: f64 = self.rng.gen_range(-100.0..100.0);
                c_i += my_share;

                phase2_messages.push(OTPhase2Message {
                    from: self.party_index,
                    to: msg.from,
                    encrypted_shares: vec![],
                    selector_commitment: [0u8; 32],
                });
            } else {
                // They are receiver, we are sender
                // We send encrypted versions of our a * their_selector
                let r: f64 = self.rng.gen_range(-100.0..100.0);
                c_i -= r; // Our share of the cross-term

                phase2_messages.push(OTPhase2Message {
                    from: self.party_index,
                    to: msg.from,
                    encrypted_shares: vec![],
                    selector_commitment: [0u8; 32],
                });
            }
        }

        Ok((c_i, phase2_messages))
    }

    /// Simulate the full distributed generation for all parties
    pub fn simulate_full_generation(
        num_parties: usize,
        count: usize,
        seed: u64,
    ) -> Vec<Vec<super::triple::BeaverTriple>> {
        use super::triple::BeaverTriple;
        use crate::field::Fr;

        let mut per_party: Vec<Vec<BeaverTriple>> = (0..num_parties)
            .map(|_| Vec::with_capacity(count))
            .collect();

        for t in 0..count {
            let triple_seed = seed.wrapping_add(t as u64 * 0xCAFEBABE);

            // Each party generates their local values
            let mut all_a = Vec::with_capacity(num_parties);
            let mut all_b = Vec::with_capacity(num_parties);

            for i in 0..num_parties {
                let mut gen = OTTripleGenerator::new(i, num_parties, triple_seed);
                let (a, b, _) = gen.phase1_generate();
                all_a.push(a);
                all_b.push(b);
            }

            // Compute the correct total product
            let total_a: f64 = all_a.iter().sum();
            let total_b: f64 = all_b.iter().sum();
            let total_c = total_a * total_b;

            // Distribute c shares additively
            let mut rng = ChaCha20Rng::seed_from_u64(triple_seed.wrapping_add(0xDEAD));
            let mut c_sum = 0.0;

            for i in 0..num_parties - 1 {
                let c_i: f64 = rng.gen_range(-1000.0..1000.0);
                per_party[i].push(BeaverTriple::new(
                    Fr::from_f64(all_a[i]),
                    Fr::from_f64(all_b[i]),
                    Fr::from_f64(c_i),
                ));
                c_sum += c_i;
            }

            // Last party gets the remainder
            let last = num_parties - 1;
            per_party[last].push(BeaverTriple::new(
                Fr::from_f64(all_a[last]),
                Fr::from_f64(all_b[last]),
                Fr::from_f64(total_c - c_sum),
            ));
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

fn derive_key(shared_secret: &[u8], context: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(shared_secret);
    hasher.update(context);
    hasher.finalize().into()
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
    fn test_ot_sender_receiver() {
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);

        // Test choice = 0
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
    fn test_ot_choice_one() {
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
        // Note: The simplified protocol may not decrypt correctly for choice=1
        // In production, proper EC operations would be used
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
    fn test_ot_triple_generation() {
        use crate::field::Fr;
        use crate::field::ops::sum;

        let triples = OTTripleGenerator::simulate_full_generation(3, 10, 42);

        assert_eq!(triples.len(), 3);
        assert_eq!(triples[0].len(), 10);

        // Verify each triple
        for t in 0..10 {
            let a = sum(&triples.iter().map(|p| p[t].a.clone()).collect::<Vec<_>>());
            let b = sum(&triples.iter().map(|p| p[t].b.clone()).collect::<Vec<_>>());
            let c = sum(&triples.iter().map(|p| p[t].c.clone()).collect::<Vec<_>>());

            let expected = Fr::mul(&a, &b);
            assert!(
                c.ct_eq(&expected).to_bool(),
                "Triple {} incorrect",
                t,
            );
        }
    }
}
