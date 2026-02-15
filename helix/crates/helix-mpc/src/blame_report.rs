//! Ethereum ECDSA signing for MAC failure blame reports.
//!
//! When SPDZ MAC verification fails and a cheater is identified, the honest
//! workers must produce a signed blame report that can be submitted to
//! `HelixCoordinatorV4.reportMACFailure()` on-chain.
//!
//! # Message Format
//!
//! The on-chain contract expects:
//! ```text
//! keccak256(abi.encodePacked("HELIX_MAC_FAILURE", jobId, stepNumber, cheater, evidence))
//! ```
//!
//! This message is then wrapped with the EIP-191 prefix:
//! ```text
//! "\x19Ethereum Signed Message:\n32" + messageHash
//! ```
//!
//! Each honest worker signs the `ethSignedHash` with their Ethereum private key.
//! The contract recovers signers via `ECDSA.recover()` and verifies they are
//! registered, non-slashed workers.
//!
//! # Signing Flow
//!
//! 1. All honest parties compute the same `MACFailureReport` from the sigma protocol.
//! 2. Each party serializes the evidence and computes the EIP-191 hash.
//! 3. Each party signs with their Ethereum private key.
//! 4. One party (coordinator) collects all signatures.
//! 5. The coordinator submits `reportMACFailure()` with the collected signatures.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, info};

use crate::error::{MPCError, MPCResult};
use crate::mac_verification::{CheaterEvidence, MACFailureReport};
use crate::session::transport::MPCTransport;
use crate::types::PartyId;

// ============================================================================
// Ethereum signing primitives (pure Rust, no ethers dependency required)
// ============================================================================

/// Keccak-256 hash (Ethereum's hash function).
///
/// This is the same as `keccak256()` in Solidity. Note: Ethereum uses
/// Keccak-256, NOT the NIST SHA3-256 standard (which adds domain separation).
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    use tiny_keccak::{Hasher, Keccak};
    let mut hasher = Keccak::v256();
    hasher.update(data);
    let mut output = [0u8; 32];
    hasher.finalize(&mut output);
    output
}

/// Computes the EIP-191 signed message hash.
///
/// Equivalent to Solidity's `MessageHashUtils.toEthSignedMessageHash(bytes32)`:
/// ```text
/// keccak256("\x19Ethereum Signed Message:\n32" + messageHash)
/// ```
pub fn to_eth_signed_message_hash(message_hash: &[u8; 32]) -> [u8; 32] {
    let mut prefixed = Vec::with_capacity(28 + 32);
    prefixed.extend_from_slice(b"\x19Ethereum Signed Message:\n32");
    prefixed.extend_from_slice(message_hash);
    keccak256(&prefixed)
}

/// Builds the MAC failure report message hash matching the V4 contract.
///
/// Equivalent to:
/// ```solidity
/// keccak256(abi.encodePacked("HELIX_MAC_FAILURE", jobId, stepNumber, cheater, evidence))
/// ```
///
/// # Arguments
///
/// * `job_id` - The on-chain job ID (uint256, 32 bytes big-endian).
/// * `step_number` - The training step where cheating was detected (uint256, 32 bytes BE).
/// * `cheater` - The cheater's Ethereum address (20 bytes).
/// * `evidence` - Encoded evidence bytes (variable length).
pub fn build_mac_failure_message(
    job_id: u64,
    step_number: u64,
    cheater: &[u8; 20],
    evidence: &[u8],
) -> [u8; 32] {
    // abi.encodePacked concatenates without padding for fixed types, but
    // uint256 values are always 32 bytes, and address is 20 bytes.
    let mut packed = Vec::new();

    // "HELIX_MAC_FAILURE" as raw bytes (no length prefix — it's a string literal in encodePacked)
    packed.extend_from_slice(b"HELIX_MAC_FAILURE");

    // jobId as uint256 (32 bytes, big-endian, zero-padded)
    let mut job_id_bytes = [0u8; 32];
    job_id_bytes[24..].copy_from_slice(&job_id.to_be_bytes());
    packed.extend_from_slice(&job_id_bytes);

    // stepNumber as uint256 (32 bytes, big-endian, zero-padded)
    let mut step_bytes = [0u8; 32];
    step_bytes[24..].copy_from_slice(&step_number.to_be_bytes());
    packed.extend_from_slice(&step_bytes);

    // cheater as address (20 bytes, no padding in encodePacked)
    packed.extend_from_slice(cheater);

    // evidence as bytes (raw, no length prefix in encodePacked)
    packed.extend_from_slice(evidence);

    keccak256(&packed)
}

/// Signs a 32-byte hash with an Ethereum private key using secp256k1 ECDSA.
///
/// Returns the 65-byte signature in Ethereum's `(r, s, v)` format where:
/// - `r`: 32 bytes
/// - `s`: 32 bytes
/// - `v`: 1 byte (27 or 28)
///
/// This signature is compatible with `ECDSA.recover()` in OpenZeppelin.
pub fn sign_hash(hash: &[u8; 32], private_key: &[u8; 32]) -> MPCResult<[u8; 65]> {
    use k256::ecdsa::{SigningKey, signature::hazmat::PrehashSigner};

    let signing_key = SigningKey::from_bytes(private_key.into())
        .map_err(|e| MPCError::ProtocolError(format!("invalid private key: {}", e)))?;

    let (signature, recovery_id) = signing_key
        .sign_prehash(hash)
        .map_err(|e| MPCError::ProtocolError(format!("ECDSA signing failed: {}", e)))?;

    let mut result = [0u8; 65];
    result[..64].copy_from_slice(&signature.to_bytes());
    // Ethereum uses v = 27 + recovery_id (0 or 1)
    result[64] = 27 + recovery_id.to_byte();

    Ok(result)
}

/// Derives the Ethereum address (20 bytes) from a secp256k1 private key.
pub fn address_from_private_key(private_key: &[u8; 32]) -> MPCResult<[u8; 20]> {
    use k256::ecdsa::SigningKey;

    let signing_key = SigningKey::from_bytes(private_key.into())
        .map_err(|e| MPCError::ProtocolError(format!("invalid private key: {}", e)))?;

    let public_key = signing_key.verifying_key();
    // Uncompressed public key: 04 || x || y (65 bytes)
    let encoded = public_key.to_encoded_point(false);
    let pubkey_bytes = &encoded.as_bytes()[1..]; // Skip the 0x04 prefix

    // Ethereum address = last 20 bytes of keccak256(public_key_bytes)
    let hash = keccak256(pubkey_bytes);
    let mut address = [0u8; 20];
    address.copy_from_slice(&hash[12..32]);
    Ok(address)
}

/// Serializes a `MACFailureReport` into evidence bytes for on-chain submission.
///
/// The evidence is a compact encoding of the sigma values and pairwise check
/// results. It doesn't need to be ABI-decodable on-chain — the contract stores
/// it as opaque `bytes` and uses it only for the message hash.
pub fn serialize_evidence(report: &MACFailureReport) -> Vec<u8> {
    // Use a deterministic, compact format:
    // [session_id_len(4) | session_id | step_number(8) | cheater(1) |
    //  num_sigmas(4) | sigma_0 | ... | sigma_n |
    //  num_pairwise(4) | (party_a(1) | party_b(1) | consistent(1)) | ... ]
    let mut buf = Vec::new();

    // Session ID
    let sid = report.session_id.as_bytes();
    buf.extend_from_slice(&(sid.len() as u32).to_be_bytes());
    buf.extend_from_slice(sid);

    // Step number
    buf.extend_from_slice(&report.step_number.to_be_bytes());

    // Identified cheater (0xFF if unknown)
    buf.push(report.identified_cheater.map_or(0xFF, |c| c as u8));

    // Sigma values
    buf.extend_from_slice(&(report.sigma_values.len() as u32).to_be_bytes());
    for sigma in &report.sigma_values {
        buf.extend_from_slice(&(sigma.len() as u32).to_be_bytes());
        buf.extend_from_slice(sigma);
    }

    // Pairwise results
    let pairwise = &report.evidence.pairwise_results;
    buf.extend_from_slice(&(pairwise.len() as u32).to_be_bytes());
    for result in pairwise {
        buf.push(result.party_a as u8);
        buf.push(result.party_b as u8);
        buf.push(if result.consistent { 1 } else { 0 });
    }

    // Round 1 and Round 2 sigmas (for audit trail)
    buf.extend_from_slice(&(report.evidence.round1_sigmas.len() as u32).to_be_bytes());
    for s in &report.evidence.round1_sigmas {
        buf.extend_from_slice(&(s.len() as u32).to_be_bytes());
        buf.extend_from_slice(s);
    }
    buf.extend_from_slice(&(report.evidence.round2_sigmas.len() as u32).to_be_bytes());
    for s in &report.evidence.round2_sigmas {
        buf.extend_from_slice(&(s.len() as u32).to_be_bytes());
        buf.extend_from_slice(s);
    }

    buf
}

// ============================================================================
// Blame Report types
// ============================================================================

/// A signed blame report ready for on-chain submission.
#[derive(Debug, Clone)]
pub struct SignedBlameReport {
    /// The on-chain job ID.
    pub job_id: u64,
    /// Training step where cheating was detected.
    pub step_number: u64,
    /// Ethereum address of the identified cheater.
    pub cheater_address: [u8; 20],
    /// Serialized evidence bytes.
    pub evidence: Vec<u8>,
    /// The message hash (before EIP-191 prefix).
    pub message_hash: [u8; 32],
    /// The EIP-191 signed message hash.
    pub eth_signed_hash: [u8; 32],
    /// Collected ECDSA signatures from honest workers (65 bytes each).
    pub signatures: Vec<[u8; 65]>,
    /// Ethereum addresses of the signers (in order matching signatures).
    pub signer_addresses: Vec<[u8; 20]>,
}

/// Per-worker Ethereum identity for signing blame reports.
#[derive(Debug, Clone)]
pub struct WorkerEthIdentity {
    /// The worker's MPC party index.
    pub party_index: usize,
    /// Ethereum private key (secp256k1, 32 bytes).
    pub private_key: [u8; 32],
    /// Ethereum address (derived from private key).
    pub address: [u8; 20],
}

impl WorkerEthIdentity {
    /// Creates a new identity from a private key.
    pub fn from_private_key(party_index: usize, private_key: [u8; 32]) -> MPCResult<Self> {
        let address = address_from_private_key(&private_key)?;
        Ok(Self {
            party_index,
            private_key,
            address,
        })
    }

    /// Signs a blame report hash.
    pub fn sign_blame(&self, eth_signed_hash: &[u8; 32]) -> MPCResult<[u8; 65]> {
        sign_hash(eth_signed_hash, &self.private_key)
    }
}

// ============================================================================
// Transport messages for blame report exchange
// ============================================================================

/// Message types for blame report exchange over MPC transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlameMessage {
    /// A party's signature for the blame report.
    BlameSignature {
        party_index: usize,
        signature: Vec<u8>,
        signer_address: Vec<u8>,
    },
}

impl BlameMessage {
    fn encode(&self) -> Vec<u8> {
        bincode::serialize(self).expect("BlameMessage encode")
    }
    fn decode(data: &[u8]) -> MPCResult<Self> {
        bincode::deserialize(data)
            .map_err(|e| MPCError::ProtocolError(format!("decode blame message: {e}")))
    }
}

// ============================================================================
// Blame report construction and signing
// ============================================================================

/// Creates a signed blame report by having all honest workers sign the failure message.
///
/// # Protocol
///
/// 1. All honest parties independently compute the message hash from the failure report.
/// 2. Each party signs the EIP-191 wrapped hash with their Ethereum key.
/// 3. Signatures are exchanged via the MPC transport.
/// 4. Party 0 (or the designated coordinator) collects all signatures.
///
/// # Arguments
///
/// * `report` - The MAC failure report from the sigma protocol.
/// * `job_id` - The on-chain training job ID.
/// * `cheater_address` - The Ethereum address of the identified cheater.
/// * `identity` - This worker's Ethereum signing identity.
/// * `transport` - MPC transport for exchanging signatures.
/// * `num_parties` - Total number of parties (including the cheater).
/// * `cheater_party_index` - The MPC party index of the cheater.
///
/// # Returns
///
/// A `SignedBlameReport` with all honest workers' signatures, ready for on-chain submission.
pub async fn create_signed_blame_report<T: MPCTransport>(
    report: &MACFailureReport,
    job_id: u64,
    cheater_address: &[u8; 20],
    identity: &WorkerEthIdentity,
    transport: &T,
    num_parties: usize,
    cheater_party_index: usize,
) -> MPCResult<SignedBlameReport> {
    // Step 1: Serialize evidence and compute message hash.
    let evidence = serialize_evidence(report);
    let message_hash = build_mac_failure_message(
        job_id,
        report.step_number,
        cheater_address,
        &evidence,
    );
    let eth_signed_hash = to_eth_signed_message_hash(&message_hash);

    debug!(
        party = identity.party_index,
        message_hash = hex::encode(message_hash),
        "Computed blame report message hash"
    );

    // Step 2: Sign the hash.
    let my_signature = identity.sign_blame(&eth_signed_hash)?;

    info!(
        party = identity.party_index,
        "Signed blame report"
    );

    // Step 3: Exchange signatures with all honest peers.
    let my_msg = BlameMessage::BlameSignature {
        party_index: identity.party_index,
        signature: my_signature.to_vec(),
        signer_address: identity.address.to_vec(),
    };
    let msg_bytes = my_msg.encode();

    // Send to all honest peers (skip the cheater — they're disconnected).
    let peers = transport.peers();
    for peer in &peers {
        let peer_idx = party_index_from_id(peer);
        if peer_idx == cheater_party_index {
            continue; // Don't send to the cheater
        }
        transport.send(peer, &msg_bytes).await?;
    }

    // Step 4: Collect signatures from honest peers.
    let mut signatures = Vec::with_capacity(num_parties - 1);
    let mut signer_addresses = Vec::with_capacity(num_parties - 1);

    // Add our own signature first.
    signatures.push(my_signature);
    signer_addresses.push(identity.address);

    for peer in &peers {
        let peer_idx = party_index_from_id(peer);

        // Skip the cheater entirely — they're disconnected and won't send.
        if peer_idx == cheater_party_index {
            continue;
        }

        let data = transport.recv(peer).await?;
        let msg = BlameMessage::decode(&data)?;

        if let BlameMessage::BlameSignature {
            party_index: _,
            signature,
            signer_address,
        } = msg
        {
            if signature.len() == 65 {
                let mut sig = [0u8; 65];
                sig.copy_from_slice(&signature);
                signatures.push(sig);
            }
            if signer_address.len() == 20 {
                let mut addr = [0u8; 20];
                addr.copy_from_slice(&signer_address);
                signer_addresses.push(addr);
            }
        }
    }

    info!(
        party = identity.party_index,
        num_signatures = signatures.len(),
        "Blame report signatures collected"
    );

    Ok(SignedBlameReport {
        job_id,
        step_number: report.step_number,
        cheater_address: *cheater_address,
        evidence,
        message_hash,
        eth_signed_hash,
        signatures,
        signer_addresses,
    })
}

// ============================================================================
// Helpers
// ============================================================================

/// Extracts party index from PartyId (e.g., "party-2" → 2).
fn party_index_from_id(party: &PartyId) -> usize {
    party
        .0
        .strip_prefix("party-")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mac_verification::{CheaterEvidence, PairwiseCheckResult};
    use crate::session::transport::LocalTransport;

    #[test]
    fn test_keccak256_empty() {
        // Known keccak256("") = 0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
        let hash = keccak256(b"");
        let expected = hex::decode("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470").unwrap();
        assert_eq!(hash[..], expected[..], "keccak256 of empty string should match known value");
    }

    #[test]
    fn test_keccak256_hello_world() {
        // Known: keccak256("Hello, World!") = 0xacaf3289d7b601cbd114fb36c4d29c85bbfd5e133f14cb355c3fd8d99367964f
        let hash = keccak256(b"Hello, World!");
        let expected = hex::decode("acaf3289d7b601cbd114fb36c4d29c85bbfd5e133f14cb355c3fd8d99367964f").unwrap();
        assert_eq!(hash[..], expected[..], "keccak256 of 'Hello, World!' should match known value");
    }

    #[test]
    fn test_eip191_prefix() {
        let msg = [0u8; 32];
        let result = to_eth_signed_message_hash(&msg);
        // The result should be different from the input (prefix added).
        assert_ne!(result, msg);
        // Should be deterministic.
        let result2 = to_eth_signed_message_hash(&msg);
        assert_eq!(result, result2);
    }

    #[test]
    fn test_build_mac_failure_message_deterministic() {
        let cheater = [0xABu8; 20];
        let evidence = b"test evidence";

        let hash1 = build_mac_failure_message(1, 10, &cheater, evidence);
        let hash2 = build_mac_failure_message(1, 10, &cheater, evidence);
        assert_eq!(hash1, hash2, "same inputs should produce same hash");

        let hash3 = build_mac_failure_message(2, 10, &cheater, evidence);
        assert_ne!(hash1, hash3, "different job_id should produce different hash");
    }

    #[test]
    fn test_sign_and_recover_address() {
        // Use a known private key.
        let private_key = [1u8; 32];
        let address = address_from_private_key(&private_key).unwrap();
        assert_eq!(address.len(), 20);

        // Sign a message.
        let message = keccak256(b"test message");
        let eth_hash = to_eth_signed_message_hash(&message);
        let signature = sign_hash(&eth_hash, &private_key).unwrap();
        assert_eq!(signature.len(), 65);
        assert!(signature[64] == 27 || signature[64] == 28, "v should be 27 or 28");
    }

    #[test]
    fn test_worker_eth_identity() {
        let pk = [42u8; 32];
        let identity = WorkerEthIdentity::from_private_key(0, pk).unwrap();
        assert_eq!(identity.party_index, 0);
        assert_eq!(identity.private_key, pk);
        assert_eq!(identity.address.len(), 20);
    }

    #[test]
    fn test_serialize_evidence_roundtrip() {
        let report = MACFailureReport {
            session_id: "test-session".to_string(),
            step_number: 42,
            identified_cheater: Some(2),
            sigma_values: vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            commitments: vec![[0u8; 32]; 3],
            evidence: CheaterEvidence {
                pairwise_results: vec![
                    PairwiseCheckResult { party_a: 0, party_b: 1, consistent: true },
                    PairwiseCheckResult { party_a: 0, party_b: 2, consistent: false },
                    PairwiseCheckResult { party_a: 1, party_b: 2, consistent: false },
                ],
                round1_sigmas: vec![vec![10, 11], vec![12, 13], vec![14, 15]],
                round2_sigmas: vec![vec![20, 21], vec![22, 23], vec![24, 25]],
            },
        };

        let bytes = serialize_evidence(&report);
        assert!(!bytes.is_empty());
        // The evidence should be deterministic.
        let bytes2 = serialize_evidence(&report);
        assert_eq!(bytes, bytes2);
    }

    #[tokio::test]
    async fn test_blame_report_signing_3_parties() {
        let num_parties = 3;
        let cheater_party = 2;

        // Generate deterministic private keys for each worker.
        let identities: Vec<WorkerEthIdentity> = (0..num_parties)
            .map(|i| {
                let mut pk = [0u8; 32];
                pk[31] = (i + 1) as u8; // Simple but valid private keys
                WorkerEthIdentity::from_private_key(i, pk).unwrap()
            })
            .collect();

        let report = MACFailureReport {
            session_id: "blame-test".to_string(),
            step_number: 30,
            identified_cheater: Some(cheater_party),
            sigma_values: vec![vec![1]; 3],
            commitments: vec![[0u8; 32]; 3],
            evidence: CheaterEvidence {
                pairwise_results: vec![
                    PairwiseCheckResult { party_a: 0, party_b: 2, consistent: false },
                    PairwiseCheckResult { party_a: 1, party_b: 2, consistent: false },
                ],
                round1_sigmas: vec![vec![1]; 3],
                round2_sigmas: vec![vec![1]; 3],
            },
        };

        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        // Only honest parties sign — the cheater is disconnected.
        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            if i == cheater_party {
                continue; // Cheater doesn't participate
            }

            let report_clone = report.clone();
            let identity = identities[i].clone();
            let cheater_addr = identities[cheater_party].address;

            handles.push(tokio::spawn(async move {
                create_signed_blame_report(
                    &report_clone,
                    1, // job_id
                    &cheater_addr,
                    &identity,
                    &transport,
                    num_parties,
                    cheater_party,
                )
                .await
            }));
        }

        // Collect results.
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap().expect("blame report should succeed");
            results.push(result);
        }

        // All honest parties should compute the same message hash.
        let expected_hash = results[0].message_hash;
        for r in &results {
            assert_eq!(r.message_hash, expected_hash, "all parties should agree on message hash");
        }

        // Honest parties should have collected 2 signatures.
        for r in &results {
            assert_eq!(
                r.signatures.len(),
                2,
                "should have 2 honest signatures, got {}",
                r.signatures.len()
            );
        }
    }
}
