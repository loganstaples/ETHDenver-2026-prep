//! Batch commitment scheme optimization.
//!
//! This module provides optimized batch operations for commitment generation
//! and verification, significantly reducing overhead for large-scale MPC.
//!
//! # Features
//!
//! - **Parallel Commitment Generation**: Multi-threaded commitment creation
//! - **Batch Verification**: Verify multiple commitments with single operation
//! - **Vector Commitments**: Efficient commitments to vectors using Merkle trees
//! - **Incremental Hashing**: Reuse hash state across related commitments
//! - **Deferred Verification**: Accumulate and verify at end of round

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

/// Configuration for batch commitment operations.
#[derive(Debug, Clone)]
pub struct BatchCommitmentConfig {
    /// Number of worker threads for parallel processing.
    pub num_workers: usize,
    /// Batch size for parallel processing.
    pub batch_size: usize,
    /// Enable SIMD optimizations where available.
    pub enable_simd: bool,
    /// Maximum pending verifications before forced flush.
    pub max_pending_verifications: usize,
}

impl Default for BatchCommitmentConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            batch_size: 1000,
            enable_simd: true,
            max_pending_verifications: 10000,
        }
    }
}

/// A batch of commitments to be created or verified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentBatch {
    /// Batch identifier.
    pub batch_id: u64,
    /// Individual commitment entries.
    pub entries: Vec<CommitmentEntry>,
    /// Root hash of the batch (for batch verification).
    pub batch_root: [u8; 32],
    /// Creation timestamp.
    pub created_at: u64,
}

/// Single commitment entry within a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentEntry {
    /// Entry index within batch.
    pub index: usize,
    /// The commitment hash.
    pub hash: [u8; 32],
    /// Party this commitment is for.
    pub party: PartyId,
    /// Description/label.
    pub label: String,
}

/// Parallel batch commitment generator.
pub struct BatchCommitmentGenerator {
    /// Configuration.
    config: BatchCommitmentConfig,
    /// Next batch ID.
    next_batch_id: AtomicU64,
    /// Worker request channel.
    request_tx: Option<Sender<WorkerRequest>>,
    /// Worker result channel.
    result_rx: Option<Receiver<WorkerResult>>,
    /// Worker handles.
    workers: Vec<thread::JoinHandle<()>>,
}

enum WorkerRequest {
    CommitScalars {
        batch_id: u64,
        start_index: usize,
        values: Vec<f64>,
        blindings: Vec<[u8; 32]>,
    },
    CommitVectors {
        batch_id: u64,
        start_index: usize,
        vectors: Vec<Vec<f64>>,
        blindings: Vec<[u8; 32]>,
    },
    Shutdown,
}

#[allow(dead_code)]
struct WorkerResult {
    batch_id: u64,
    start_index: usize,
    hashes: Vec<[u8; 32]>,
}

impl BatchCommitmentGenerator {
    /// Creates a new batch commitment generator.
    pub fn new(config: BatchCommitmentConfig) -> Self {
        let (request_tx, request_rx): (Sender<WorkerRequest>, Receiver<WorkerRequest>) = bounded(1000);
        let (result_tx, result_rx): (Sender<WorkerResult>, Receiver<WorkerResult>) = bounded(1000);

        let mut workers = Vec::new();
        for _ in 0..config.num_workers {
            let rx = request_rx.clone();
            let tx = result_tx.clone();
            let handle = thread::spawn(move || {
                Self::worker_loop(rx, tx);
            });
            workers.push(handle);
        }

        Self {
            config,
            next_batch_id: AtomicU64::new(0),
            request_tx: Some(request_tx),
            result_rx: Some(result_rx),
            workers,
        }
    }

    fn worker_loop(request_rx: Receiver<WorkerRequest>, result_tx: Sender<WorkerResult>) {
        loop {
            match request_rx.recv() {
                Ok(WorkerRequest::CommitScalars { batch_id, start_index, values, blindings }) => {
                    let hashes: Vec<[u8; 32]> = values
                        .iter()
                        .zip(blindings.iter())
                        .map(|(v, b)| Self::hash_scalar(*v, b))
                        .collect();

                    let _ = result_tx.send(WorkerResult {
                        batch_id,
                        start_index,
                        hashes,
                    });
                }
                Ok(WorkerRequest::CommitVectors { batch_id, start_index, vectors, blindings }) => {
                    let hashes: Vec<[u8; 32]> = vectors
                        .iter()
                        .zip(blindings.iter())
                        .map(|(v, b)| Self::hash_vector(v, b))
                        .collect();

                    let _ = result_tx.send(WorkerResult {
                        batch_id,
                        start_index,
                        hashes,
                    });
                }
                Ok(WorkerRequest::Shutdown) | Err(_) => break,
            }
        }
    }

    fn hash_scalar(value: f64, blinding: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&value.to_le_bytes());
        hasher.update(blinding);
        hasher.finalize().into()
    }

    fn hash_vector(values: &[f64], blinding: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for v in values {
            hasher.update(&v.to_le_bytes());
        }
        hasher.update(blinding);
        hasher.finalize().into()
    }

    /// Commits a batch of scalar values in parallel.
    pub fn commit_scalars(
        &self,
        party: &PartyId,
        values: &[f64],
        label_prefix: &str,
    ) -> MPCResult<CommitmentBatch> {
        let batch_id = self.next_batch_id.fetch_add(1, Ordering::SeqCst);
        let mut rng = ChaCha20Rng::from_entropy();

        // Generate blindings
        let blindings: Vec<[u8; 32]> = (0..values.len())
            .map(|_| {
                let mut b = [0u8; 32];
                rng.fill(&mut b);
                b
            })
            .collect();

        // Split into chunks and dispatch to workers
        let chunks: Vec<_> = values.chunks(self.config.batch_size).collect();
        let blinding_chunks: Vec<_> = blindings.chunks(self.config.batch_size).collect();

        let request_tx = self.request_tx.as_ref()
            .ok_or_else(|| MPCError::ProtocolError("Generator not initialized".into()))?;

        let mut expected_results = 0;
        for (chunk_idx, (value_chunk, blinding_chunk)) in chunks.iter().zip(blinding_chunks.iter()).enumerate() {
            request_tx
                .send(WorkerRequest::CommitScalars {
                    batch_id,
                    start_index: chunk_idx * self.config.batch_size,
                    values: value_chunk.to_vec(),
                    blindings: blinding_chunk.to_vec(),
                })
                .map_err(|_| MPCError::CommunicationError("Worker channel closed".into()))?;
            expected_results += 1;
        }

        // Collect results
        let result_rx = self.result_rx.as_ref()
            .ok_or_else(|| MPCError::ProtocolError("Generator not initialized".into()))?;

        let mut all_hashes: Vec<(usize, [u8; 32])> = Vec::with_capacity(values.len());
        for _ in 0..expected_results {
            let result = result_rx.recv()
                .map_err(|_| MPCError::CommunicationError("Worker channel closed".into()))?;

            for (i, hash) in result.hashes.into_iter().enumerate() {
                all_hashes.push((result.start_index + i, hash));
            }
        }

        // Sort by index
        all_hashes.sort_by_key(|(idx, _)| *idx);

        // Build entries
        let entries: Vec<CommitmentEntry> = all_hashes
            .into_iter()
            .map(|(idx, hash)| CommitmentEntry {
                index: idx,
                hash,
                party: party.clone(),
                label: format!("{}-{}", label_prefix, idx),
            })
            .collect();

        // Compute batch root
        let batch_root = Self::compute_batch_root(&entries);

        Ok(CommitmentBatch {
            batch_id,
            entries,
            batch_root,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        })
    }

    /// Commits multiple vectors in parallel.
    pub fn commit_vectors(
        &self,
        party: &PartyId,
        vectors: &[Vec<f64>],
        label_prefix: &str,
    ) -> MPCResult<CommitmentBatch> {
        let batch_id = self.next_batch_id.fetch_add(1, Ordering::SeqCst);
        let mut rng = ChaCha20Rng::from_entropy();

        // Generate blindings
        let blindings: Vec<[u8; 32]> = (0..vectors.len())
            .map(|_| {
                let mut b = [0u8; 32];
                rng.fill(&mut b);
                b
            })
            .collect();

        // For vector commitments, process in smaller batches
        let request_tx = self.request_tx.as_ref()
            .ok_or_else(|| MPCError::ProtocolError("Generator not initialized".into()))?;

        let chunk_size = self.config.batch_size.max(1);
        let chunks: Vec<_> = vectors.chunks(chunk_size).collect();
        let blinding_chunks: Vec<_> = blindings.chunks(chunk_size).collect();

        let mut expected_results = 0;
        for (chunk_idx, (vec_chunk, blind_chunk)) in chunks.iter().zip(blinding_chunks.iter()).enumerate() {
            request_tx
                .send(WorkerRequest::CommitVectors {
                    batch_id,
                    start_index: chunk_idx * chunk_size,
                    vectors: vec_chunk.to_vec(),
                    blindings: blind_chunk.to_vec(),
                })
                .map_err(|_| MPCError::CommunicationError("Worker channel closed".into()))?;
            expected_results += 1;
        }

        // Collect results
        let result_rx = self.result_rx.as_ref()
            .ok_or_else(|| MPCError::ProtocolError("Generator not initialized".into()))?;

        let mut all_hashes: Vec<(usize, [u8; 32])> = Vec::with_capacity(vectors.len());
        for _ in 0..expected_results {
            let result = result_rx.recv()
                .map_err(|_| MPCError::CommunicationError("Worker channel closed".into()))?;

            for (i, hash) in result.hashes.into_iter().enumerate() {
                all_hashes.push((result.start_index + i, hash));
            }
        }

        all_hashes.sort_by_key(|(idx, _)| *idx);

        let entries: Vec<CommitmentEntry> = all_hashes
            .into_iter()
            .map(|(idx, hash)| CommitmentEntry {
                index: idx,
                hash,
                party: party.clone(),
                label: format!("{}-{}", label_prefix, idx),
            })
            .collect();

        let batch_root = Self::compute_batch_root(&entries);

        Ok(CommitmentBatch {
            batch_id,
            entries,
            batch_root,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        })
    }

    /// Computes the batch root hash.
    fn compute_batch_root(entries: &[CommitmentEntry]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for entry in entries {
            hasher.update(&entry.hash);
        }
        hasher.finalize().into()
    }

    /// Shuts down the generator.
    pub fn shutdown(&mut self) {
        if let Some(tx) = self.request_tx.take() {
            for _ in 0..self.config.num_workers {
                let _ = tx.send(WorkerRequest::Shutdown);
            }
        }

        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for BatchCommitmentGenerator {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Batch verification accumulator.
///
/// Accumulates commitments and values during a round, then verifies
/// all at once for efficiency.
pub struct BatchVerifier {
    /// Pending verifications.
    pending: RwLock<Vec<PendingVerification>>,
    /// Maximum pending before auto-flush.
    max_pending: usize,
    /// Statistics.
    verified_count: AtomicU64,
    failed_count: AtomicU64,
}

struct PendingVerification {
    commitment: [u8; 32],
    value: Vec<u8>,
    blinding: [u8; 32],
    party: PartyId,
    label: String,
}

impl BatchVerifier {
    /// Creates a new batch verifier.
    pub fn new(max_pending: usize) -> Self {
        Self {
            pending: RwLock::new(Vec::new()),
            max_pending,
            verified_count: AtomicU64::new(0),
            failed_count: AtomicU64::new(0),
        }
    }

    /// Queues a scalar for verification.
    pub fn queue_scalar(
        &self,
        commitment: [u8; 32],
        value: f64,
        blinding: [u8; 32],
        party: PartyId,
        label: String,
    ) {
        let mut pending = self.pending.write();
        pending.push(PendingVerification {
            commitment,
            value: value.to_le_bytes().to_vec(),
            blinding,
            party,
            label,
        });

        if pending.len() >= self.max_pending {
            drop(pending);
            let _ = self.flush();
        }
    }

    /// Queues a vector for verification.
    pub fn queue_vector(
        &self,
        commitment: [u8; 32],
        values: &[f64],
        blinding: [u8; 32],
        party: PartyId,
        label: String,
    ) {
        let mut value_bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            value_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let mut pending = self.pending.write();
        pending.push(PendingVerification {
            commitment,
            value: value_bytes,
            blinding,
            party,
            label,
        });

        if pending.len() >= self.max_pending {
            drop(pending);
            let _ = self.flush();
        }
    }

    /// Verifies all pending commitments.
    pub fn flush(&self) -> MPCResult<Vec<VerificationFailure>> {
        let pending = std::mem::take(&mut *self.pending.write());
        let mut failures = Vec::new();

        for p in pending {
            let mut hasher = Sha256::new();
            hasher.update(&p.value);
            hasher.update(&p.blinding);
            let expected: [u8; 32] = hasher.finalize().into();

            if expected != p.commitment {
                failures.push(VerificationFailure {
                    party: p.party,
                    label: p.label,
                    expected,
                    actual: p.commitment,
                });
                self.failed_count.fetch_add(1, Ordering::SeqCst);
            } else {
                self.verified_count.fetch_add(1, Ordering::SeqCst);
            }
        }

        Ok(failures)
    }

    /// Returns verification statistics.
    pub fn stats(&self) -> VerificationStats {
        VerificationStats {
            verified: self.verified_count.load(Ordering::SeqCst),
            failed: self.failed_count.load(Ordering::SeqCst),
            pending: self.pending.read().len(),
        }
    }

    /// Returns pending count.
    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
    }
}

/// A verification failure.
#[derive(Debug, Clone)]
pub struct VerificationFailure {
    pub party: PartyId,
    pub label: String,
    pub expected: [u8; 32],
    pub actual: [u8; 32],
}

/// Verification statistics.
#[derive(Debug, Clone)]
pub struct VerificationStats {
    pub verified: u64,
    pub failed: u64,
    pub pending: usize,
}

/// Incremental hasher for related commitments.
///
/// Reuses hash state for commitments that share a common prefix.
pub struct IncrementalHasher {
    /// Base hasher state.
    base_state: Option<Sha256>,
    /// Common prefix.
    prefix: Vec<u8>,
}

impl IncrementalHasher {
    /// Creates a new incremental hasher with a common prefix.
    pub fn with_prefix(prefix: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(prefix);
        Self {
            base_state: Some(hasher),
            prefix: prefix.to_vec(),
        }
    }

    /// Computes hash with the common prefix + suffix.
    pub fn hash(&self, suffix: &[u8]) -> [u8; 32] {
        if let Some(ref base) = self.base_state {
            let mut hasher = base.clone();
            hasher.update(suffix);
            hasher.finalize().into()
        } else {
            let mut hasher = Sha256::new();
            hasher.update(&self.prefix);
            hasher.update(suffix);
            hasher.finalize().into()
        }
    }

    /// Computes multiple hashes efficiently.
    pub fn hash_batch(&self, suffixes: &[&[u8]]) -> Vec<[u8; 32]> {
        suffixes.iter().map(|s| self.hash(s)).collect()
    }
}

/// Vector commitment using a Merkle tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorCommitment {
    /// Root hash.
    pub root: [u8; 32],
    /// Number of elements.
    pub size: usize,
    /// Tree depth.
    pub depth: usize,
}

impl VectorCommitment {
    /// Creates a commitment to a vector of values.
    pub fn commit(values: &[f64]) -> Self {
        let leaves: Vec<[u8; 32]> = values.iter().map(|v| {
            let mut hasher = Sha256::new();
            hasher.update(&v.to_le_bytes());
            hasher.finalize().into()
        }).collect();

        let (root, depth) = Self::build_tree(&leaves);

        Self {
            root,
            size: values.len(),
            depth,
        }
    }

    /// Creates a commitment with precomputed leaves.
    pub fn from_leaves(leaves: &[[u8; 32]]) -> Self {
        let (root, depth) = Self::build_tree(leaves);
        Self {
            root,
            size: leaves.len(),
            depth,
        }
    }

    fn build_tree(leaves: &[[u8; 32]]) -> ([u8; 32], usize) {
        if leaves.is_empty() {
            return ([0u8; 32], 0);
        }

        let mut layer: Vec<[u8; 32]> = leaves.to_vec();

        // Pad to power of 2
        let target = layer.len().next_power_of_two();
        while layer.len() < target {
            layer.push([0u8; 32]);
        }

        let mut depth = 0;
        while layer.len() > 1 {
            let mut next = Vec::with_capacity(layer.len() / 2);
            for chunk in layer.chunks(2) {
                let mut hasher = Sha256::new();
                hasher.update(&chunk[0]);
                hasher.update(&chunk[1]);
                next.push(hasher.finalize().into());
            }
            layer = next;
            depth += 1;
        }

        (layer[0], depth)
    }

    /// Generates a proof for an element.
    pub fn prove(values: &[f64], index: usize) -> Option<VectorProof> {
        if index >= values.len() {
            return None;
        }

        let leaves: Vec<[u8; 32]> = values.iter().map(|v| {
            let mut hasher = Sha256::new();
            hasher.update(&v.to_le_bytes());
            hasher.finalize().into()
        }).collect();

        let mut layer = leaves.clone();
        let target = layer.len().next_power_of_two();
        while layer.len() < target {
            layer.push([0u8; 32]);
        }

        let mut siblings = Vec::new();
        let mut idx = index;

        while layer.len() > 1 {
            let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
            siblings.push(layer.get(sibling_idx).copied().unwrap_or([0u8; 32]));

            let mut next = Vec::with_capacity(layer.len() / 2);
            for chunk in layer.chunks(2) {
                let mut hasher = Sha256::new();
                hasher.update(&chunk[0]);
                hasher.update(&chunk[1]);
                next.push(hasher.finalize().into());
            }
            layer = next;
            idx /= 2;
        }

        Some(VectorProof {
            index,
            value: values[index],
            siblings,
        })
    }

    /// Verifies a proof.
    pub fn verify(&self, proof: &VectorProof) -> bool {
        let mut hasher = Sha256::new();
        hasher.update(&proof.value.to_le_bytes());
        let mut current: [u8; 32] = hasher.finalize().into();

        let mut idx = proof.index;
        for sibling in &proof.siblings {
            let mut hasher = Sha256::new();
            if idx % 2 == 0 {
                hasher.update(&current);
                hasher.update(sibling);
            } else {
                hasher.update(sibling);
                hasher.update(&current);
            }
            current = hasher.finalize().into();
            idx /= 2;
        }

        current == self.root
    }
}

/// Proof of a single element in a vector commitment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorProof {
    pub index: usize,
    pub value: f64,
    pub siblings: Vec<[u8; 32]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_commitment_generator_scalars() {
        let config = BatchCommitmentConfig {
            num_workers: 2,
            batch_size: 10,
            ..Default::default()
        };
        let mut generator = BatchCommitmentGenerator::new(config);

        let party = PartyId::from_index(0);
        let values: Vec<f64> = (0..25).map(|i| i as f64).collect();

        let batch = generator.commit_scalars(&party, &values, "test").unwrap();

        assert_eq!(batch.entries.len(), 25);
        assert_ne!(batch.batch_root, [0u8; 32]);

        generator.shutdown();
    }

    #[test]
    fn test_batch_commitment_generator_vectors() {
        let config = BatchCommitmentConfig {
            num_workers: 2,
            batch_size: 5,
            ..Default::default()
        };
        let mut generator = BatchCommitmentGenerator::new(config);

        let party = PartyId::from_index(0);
        let vectors: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64; 5]).collect();

        let batch = generator.commit_vectors(&party, &vectors, "weights").unwrap();

        assert_eq!(batch.entries.len(), 10);

        generator.shutdown();
    }

    #[test]
    fn test_batch_verifier() {
        let verifier = BatchVerifier::new(100);
        let party = PartyId::from_index(0);

        // Create commitment for a value
        let value = 42.0f64;
        let blinding = [1u8; 32];

        let mut hasher = Sha256::new();
        hasher.update(&value.to_le_bytes());
        hasher.update(&blinding);
        let commitment: [u8; 32] = hasher.finalize().into();

        // Queue for verification
        verifier.queue_scalar(commitment, value, blinding, party.clone(), "test".into());

        // Flush and verify
        let failures = verifier.flush().unwrap();
        assert!(failures.is_empty());

        let stats = verifier.stats();
        assert_eq!(stats.verified, 1);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_batch_verifier_failure() {
        let verifier = BatchVerifier::new(100);
        let party = PartyId::from_index(0);

        // Create wrong commitment
        let value = 42.0f64;
        let blinding = [1u8; 32];
        let wrong_commitment = [0u8; 32]; // Wrong!

        verifier.queue_scalar(wrong_commitment, value, blinding, party.clone(), "test".into());

        let failures = verifier.flush().unwrap();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].party, party);
    }

    #[test]
    fn test_incremental_hasher() {
        let prefix = b"common_prefix:";
        let hasher = IncrementalHasher::with_prefix(prefix);

        let hash1 = hasher.hash(b"suffix1");
        let hash2 = hasher.hash(b"suffix2");

        assert_ne!(hash1, hash2);

        // Verify against manual computation
        let mut manual = Sha256::new();
        manual.update(prefix);
        manual.update(b"suffix1");
        let expected: [u8; 32] = manual.finalize().into();

        assert_eq!(hash1, expected);
    }

    #[test]
    fn test_vector_commitment() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let commitment = VectorCommitment::commit(&values);

        assert_eq!(commitment.size, 8);
        assert!(commitment.depth > 0);

        // Generate and verify proofs
        for i in 0..values.len() {
            let proof = VectorCommitment::prove(&values, i).unwrap();
            assert!(commitment.verify(&proof), "Proof for index {} failed", i);
        }
    }

    #[test]
    fn test_vector_commitment_invalid_proof() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let commitment = VectorCommitment::commit(&values);

        let mut proof = VectorCommitment::prove(&values, 0).unwrap();
        proof.value = 999.0; // Tamper with value

        assert!(!commitment.verify(&proof));
    }

    #[test]
    fn test_vector_commitment_empty() {
        let values: Vec<f64> = vec![];
        let commitment = VectorCommitment::commit(&values);

        assert_eq!(commitment.size, 0);
        assert_eq!(commitment.root, [0u8; 32]);
    }
}
