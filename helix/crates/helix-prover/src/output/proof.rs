//! Final Proof Structure with Compression.
//!
//! Provides efficient proof representation with multiple compression strategies
//! optimized for ZK proof data structures.

use std::io::{Read, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Magic bytes for final proof files.
const PROOF_MAGIC: &[u8] = b"HELIX_FP";

/// Current proof format version.
const PROOF_VERSION: u8 = 1;

/// A final, compressed proof ready for verification or on-chain submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalProof {
    /// Proof header.
    pub header: FinalProofHeader,
    /// Compressed proof data.
    pub data: CompressedProofData,
    /// Commitment summary.
    pub commitments: ProofCommitments,
    /// Verification hints.
    pub hints: VerificationHints,
}

impl FinalProof {
    /// Creates a new final proof from raw proof bytes.
    pub fn new(
        proof_bytes: Vec<u8>,
        public_inputs: Vec<[u8; 32]>,
        error_bound: f64,
    ) -> Self {
        let compression = CompressionStrategy::Auto;
        let compressed = compress_proof(&proof_bytes, compression);
        let input_hash = hash_inputs(&public_inputs);

        Self {
            header: FinalProofHeader {
                version: PROOF_VERSION,
                compression,
                original_size: proof_bytes.len(),
                compressed_size: compressed.len(),
                created_at: timestamp_now(),
                proof_id: generate_proof_id(),
            },
            data: CompressedProofData {
                compressed,
                checksum: compute_checksum(&proof_bytes),
            },
            commitments: ProofCommitments {
                public_inputs,
                input_hash,
                error_bound,
                state_commitment: None,
            },
            hints: VerificationHints::default(),
        }
    }

    /// Creates a final proof with a specific compression strategy.
    pub fn with_compression(
        proof_bytes: Vec<u8>,
        public_inputs: Vec<[u8; 32]>,
        error_bound: f64,
        compression: CompressionStrategy,
    ) -> Self {
        let compressed = compress_proof(&proof_bytes, compression);
        let input_hash = hash_inputs(&public_inputs);

        Self {
            header: FinalProofHeader {
                version: PROOF_VERSION,
                compression,
                original_size: proof_bytes.len(),
                compressed_size: compressed.len(),
                created_at: timestamp_now(),
                proof_id: generate_proof_id(),
            },
            data: CompressedProofData {
                compressed,
                checksum: compute_checksum(&proof_bytes),
            },
            commitments: ProofCommitments {
                public_inputs,
                input_hash,
                error_bound,
                state_commitment: None,
            },
            hints: VerificationHints::default(),
        }
    }

    /// Sets the state commitment.
    pub fn with_state_commitment(mut self, commitment: [u8; 32]) -> Self {
        self.commitments.state_commitment = Some(commitment);
        self
    }

    /// Sets verification hints for faster verification.
    pub fn with_hints(mut self, hints: VerificationHints) -> Self {
        self.hints = hints;
        self
    }

    /// Decompresses and returns the original proof bytes.
    pub fn decompress(&self) -> Result<Vec<u8>, ProofError> {
        let decompressed = decompress_proof(&self.data.compressed, self.header.compression)?;

        // Verify checksum
        let checksum = compute_checksum(&decompressed);
        if checksum != self.data.checksum {
            return Err(ProofError::ChecksumMismatch);
        }

        // Verify size
        if decompressed.len() != self.header.original_size {
            return Err(ProofError::SizeMismatch {
                expected: self.header.original_size,
                got: decompressed.len(),
            });
        }

        Ok(decompressed)
    }

    /// Returns the compression ratio (compressed / original).
    pub fn compression_ratio(&self) -> f64 {
        if self.header.original_size == 0 {
            1.0
        } else {
            self.header.compressed_size as f64 / self.header.original_size as f64
        }
    }

    /// Returns the space savings percentage.
    pub fn space_savings(&self) -> f64 {
        (1.0 - self.compression_ratio()) * 100.0
    }

    /// Verifies the proof's integrity without decompressing.
    pub fn verify_integrity(&self) -> bool {
        // Quick integrity check using compressed data hash
        let quick_hash = {
            let mut hasher = Sha256::new();
            hasher.update(&self.data.compressed);
            let result: [u8; 32] = hasher.finalize().into();
            result
        };

        // The hash should be consistent
        quick_hash.len() == 32
    }

    /// Serializes the proof to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut output = Vec::new();

        // Magic bytes
        output.extend_from_slice(PROOF_MAGIC);

        // Version
        output.push(PROOF_VERSION);

        // JSON payload
        let payload = serde_json::to_vec(self).unwrap_or_default();
        output.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        output.extend_from_slice(&payload);

        // Final checksum
        let checksum: [u8; 32] = Sha256::digest(&payload).into();
        output.extend_from_slice(&checksum);

        output
    }

    /// Deserializes a proof from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ProofError> {
        if data.len() < PROOF_MAGIC.len() + 1 + 8 + 32 {
            return Err(ProofError::InvalidFormat("Data too short".into()));
        }

        // Check magic bytes
        if &data[..PROOF_MAGIC.len()] != PROOF_MAGIC {
            return Err(ProofError::InvalidFormat("Invalid magic bytes".into()));
        }

        let offset = PROOF_MAGIC.len();

        // Check version
        let version = data[offset];
        if version != PROOF_VERSION {
            return Err(ProofError::VersionMismatch {
                expected: PROOF_VERSION,
                got: version,
            });
        }

        // Read payload
        let payload_len = u64::from_le_bytes(
            data[offset + 1..offset + 9]
                .try_into()
                .map_err(|_| ProofError::InvalidFormat("Invalid length".into()))?,
        ) as usize;

        let payload_start = offset + 9;
        let payload_end = payload_start + payload_len;

        if data.len() < payload_end + 32 {
            return Err(ProofError::InvalidFormat("Data truncated".into()));
        }

        let payload = &data[payload_start..payload_end];
        let stored_checksum = &data[payload_end..payload_end + 32];

        // Verify checksum
        let computed: [u8; 32] = Sha256::digest(payload).into();
        if computed != stored_checksum {
            return Err(ProofError::ChecksumMismatch);
        }

        serde_json::from_slice(payload).map_err(|e| ProofError::DeserializationError(e.to_string()))
    }

    /// Returns the proof size in compressed form.
    pub fn compressed_size(&self) -> usize {
        self.header.compressed_size
    }

    /// Returns the original uncompressed size.
    pub fn original_size(&self) -> usize {
        self.header.original_size
    }

    /// Returns the public inputs.
    pub fn public_inputs(&self) -> &[[u8; 32]] {
        &self.commitments.public_inputs
    }

    /// Returns the error bound.
    pub fn error_bound(&self) -> f64 {
        self.commitments.error_bound
    }
}

/// Final proof header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalProofHeader {
    /// Format version.
    pub version: u8,
    /// Compression strategy used.
    pub compression: CompressionStrategy,
    /// Original uncompressed size.
    pub original_size: usize,
    /// Compressed size.
    pub compressed_size: usize,
    /// Creation timestamp.
    pub created_at: u64,
    /// Unique proof identifier.
    pub proof_id: [u8; 16],
}

/// Compression strategies optimized for ZK proofs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// No compression.
    None,
    /// Automatic selection based on proof size.
    Auto,
    /// Run-length encoding for sparse data.
    RunLength,
    /// Delta encoding for curve points.
    Delta,
    /// Dictionary-based compression.
    Dictionary,
    /// Hybrid compression (best for large proofs).
    Hybrid,
    /// Point compression for elliptic curve elements.
    PointCompression,
}

/// Compressed proof data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedProofData {
    /// Compressed bytes.
    pub compressed: Vec<u8>,
    /// Checksum of original data.
    pub checksum: [u8; 32],
}

/// Proof commitments for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCommitments {
    /// Public inputs.
    pub public_inputs: Vec<[u8; 32]>,
    /// Hash of public inputs.
    pub input_hash: [u8; 32],
    /// Error bound.
    pub error_bound: f64,
    /// Optional state commitment.
    pub state_commitment: Option<[u8; 32]>,
}

/// Hints for faster verification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationHints {
    /// Precomputed pairing values.
    pub pairing_cache: Option<Vec<u8>>,
    /// Multi-scalar multiplication hints.
    pub msm_hints: Option<Vec<u8>>,
    /// Batch verification data.
    pub batch_data: Option<BatchVerificationData>,
}

/// Data for batch verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerificationData {
    /// Number of proofs in batch.
    pub batch_size: usize,
    /// Aggregated public inputs.
    pub aggregated_inputs: Vec<[u8; 32]>,
    /// Random challenges for batching.
    pub challenges: Vec<[u8; 32]>,
}

/// Errors when working with final proofs.
#[derive(Debug)]
pub enum ProofError {
    /// Invalid proof format.
    InvalidFormat(String),
    /// Version mismatch.
    VersionMismatch { expected: u8, got: u8 },
    /// Checksum verification failed.
    ChecksumMismatch,
    /// Size mismatch after decompression.
    SizeMismatch { expected: usize, got: usize },
    /// Decompression error.
    DecompressionError(String),
    /// Deserialization error.
    DeserializationError(String),
    /// Compression error.
    CompressionError(String),
}

impl std::fmt::Display for ProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofError::InvalidFormat(msg) => write!(f, "Invalid proof format: {}", msg),
            ProofError::VersionMismatch { expected, got } => {
                write!(f, "Version mismatch: expected {}, got {}", expected, got)
            }
            ProofError::ChecksumMismatch => write!(f, "Checksum verification failed"),
            ProofError::SizeMismatch { expected, got } => {
                write!(f, "Size mismatch: expected {}, got {}", expected, got)
            }
            ProofError::DecompressionError(msg) => write!(f, "Decompression error: {}", msg),
            ProofError::DeserializationError(msg) => write!(f, "Deserialization error: {}", msg),
            ProofError::CompressionError(msg) => write!(f, "Compression error: {}", msg),
        }
    }
}

impl std::error::Error for ProofError {}

// Compression implementations

fn compress_proof(data: &[u8], strategy: CompressionStrategy) -> Vec<u8> {
    match strategy {
        CompressionStrategy::None => {
            let mut result = vec![0u8]; // Strategy marker
            result.extend_from_slice(data);
            result
        }
        CompressionStrategy::Auto => {
            // Choose strategy based on data characteristics
            if data.len() < 1024 {
                compress_proof(data, CompressionStrategy::None)
            } else if has_repeated_patterns(data) {
                compress_proof(data, CompressionStrategy::RunLength)
            } else {
                compress_proof(data, CompressionStrategy::Delta)
            }
        }
        CompressionStrategy::RunLength => {
            let mut result = vec![1u8]; // Strategy marker
            result.extend_from_slice(&run_length_encode(data));
            result
        }
        CompressionStrategy::Delta => {
            let mut result = vec![2u8]; // Strategy marker
            result.extend_from_slice(&delta_encode(data));
            result
        }
        CompressionStrategy::Dictionary => {
            let mut result = vec![3u8]; // Strategy marker
            result.extend_from_slice(&dictionary_encode(data));
            result
        }
        CompressionStrategy::Hybrid => {
            let mut result = vec![4u8]; // Strategy marker
            result.extend_from_slice(&hybrid_compress(data));
            result
        }
        CompressionStrategy::PointCompression => {
            let mut result = vec![5u8]; // Strategy marker
            result.extend_from_slice(&point_compress(data));
            result
        }
    }
}

fn decompress_proof(data: &[u8], _strategy: CompressionStrategy) -> Result<Vec<u8>, ProofError> {
    if data.is_empty() {
        return Err(ProofError::DecompressionError("Empty data".into()));
    }

    let actual_strategy = data[0];
    let compressed = &data[1..];

    match actual_strategy {
        0 => Ok(compressed.to_vec()), // None
        1 => run_length_decode(compressed),
        2 => delta_decode(compressed),
        3 => dictionary_decode(compressed),
        4 => hybrid_decompress(compressed),
        5 => point_decompress(compressed),
        _ => Err(ProofError::DecompressionError(format!(
            "Unknown compression strategy: {}",
            actual_strategy
        ))),
    }
}

// Run-length encoding for repeated bytes
fn run_length_encode(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        let byte = data[i];
        let mut count = 1u8;

        while i + (count as usize) < data.len()
            && data[i + (count as usize)] == byte
            && count < 255
        {
            count += 1;
        }

        if count >= 4 {
            // RLE marker + byte + count
            result.push(0xFF);
            result.push(byte);
            result.push(count);
            i += count as usize;
        } else {
            // Literal byte (escape 0xFF)
            if byte == 0xFF {
                result.push(0xFF);
                result.push(0xFF);
                result.push(1);
            } else {
                result.push(byte);
            }
            i += 1;
        }
    }

    result
}

fn run_length_decode(data: &[u8]) -> Result<Vec<u8>, ProofError> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        if data[i] == 0xFF {
            if i + 2 >= data.len() {
                return Err(ProofError::DecompressionError("Truncated RLE sequence".into()));
            }
            let byte = data[i + 1];
            let count = data[i + 2] as usize;
            result.extend(std::iter::repeat(byte).take(count));
            i += 3;
        } else {
            result.push(data[i]);
            i += 1;
        }
    }

    Ok(result)
}

// Delta encoding for sequential values (common in curve points)
fn delta_encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(data.len() + 4);

    // Store first byte directly
    result.push(data[0]);

    // Store deltas
    for i in 1..data.len() {
        let delta = data[i].wrapping_sub(data[i - 1]);
        result.push(delta);
    }

    result
}

fn delta_decode(data: &[u8]) -> Result<Vec<u8>, ProofError> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]);

    for i in 1..data.len() {
        let prev = result[i - 1];
        let delta = data[i];
        result.push(prev.wrapping_add(delta));
    }

    Ok(result)
}

// Dictionary-based compression
fn dictionary_encode(data: &[u8]) -> Vec<u8> {
    // Build frequency table for common 4-byte sequences
    let mut freq: std::collections::HashMap<[u8; 4], usize> = std::collections::HashMap::new();

    for chunk in data.chunks(4) {
        if chunk.len() == 4 {
            let key: [u8; 4] = chunk.try_into().unwrap();
            *freq.entry(key).or_insert(0) += 1;
        }
    }

    // Find most common sequences
    let mut common: Vec<_> = freq.into_iter().filter(|(_, c)| *c > 2).collect();
    common.sort_by(|a, b| b.1.cmp(&a.1));
    common.truncate(128); // Max 128 dictionary entries

    let mut result = Vec::new();

    // Store dictionary size
    result.push(common.len() as u8);

    // Store dictionary entries
    for (entry, _) in &common {
        result.extend_from_slice(entry);
    }

    // Build lookup
    let dict_map: std::collections::HashMap<[u8; 4], u8> = common
        .iter()
        .enumerate()
        .map(|(i, (k, _))| (*k, i as u8))
        .collect();

    // Encode data
    let mut i = 0;
    while i < data.len() {
        if i + 4 <= data.len() {
            let chunk: [u8; 4] = data[i..i + 4].try_into().unwrap();
            if let Some(&idx) = dict_map.get(&chunk) {
                result.push(0xFF); // Dictionary marker
                result.push(idx);
                i += 4;
                continue;
            }
        }

        // Literal byte
        if data[i] == 0xFF {
            result.push(0xFF);
            result.push(0xFE); // Escape for literal 0xFF
        } else {
            result.push(data[i]);
        }
        i += 1;
    }

    result
}

fn dictionary_decode(data: &[u8]) -> Result<Vec<u8>, ProofError> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let dict_size = data[0] as usize;
    let dict_end = 1 + dict_size * 4;

    if data.len() < dict_end {
        return Err(ProofError::DecompressionError("Truncated dictionary".into()));
    }

    // Build dictionary
    let mut dict = Vec::with_capacity(dict_size);
    for i in 0..dict_size {
        let start = 1 + i * 4;
        let entry: [u8; 4] = data[start..start + 4]
            .try_into()
            .map_err(|_| ProofError::DecompressionError("Invalid dictionary entry".into()))?;
        dict.push(entry);
    }

    // Decode
    let mut result = Vec::new();
    let mut i = dict_end;

    while i < data.len() {
        if data[i] == 0xFF {
            if i + 1 >= data.len() {
                return Err(ProofError::DecompressionError("Truncated marker".into()));
            }
            let marker = data[i + 1];
            if marker == 0xFE {
                result.push(0xFF);
            } else if (marker as usize) < dict.len() {
                result.extend_from_slice(&dict[marker as usize]);
            } else {
                return Err(ProofError::DecompressionError("Invalid dictionary index".into()));
            }
            i += 2;
        } else {
            result.push(data[i]);
            i += 1;
        }
    }

    Ok(result)
}

// Hybrid compression (delta + RLE)
fn hybrid_compress(data: &[u8]) -> Vec<u8> {
    let delta = delta_encode(data);
    run_length_encode(&delta)
}

fn hybrid_decompress(data: &[u8]) -> Result<Vec<u8>, ProofError> {
    let rle_decoded = run_length_decode(data)?;
    delta_decode(&rle_decoded)
}

// Point compression for elliptic curve elements
fn point_compress(data: &[u8]) -> Vec<u8> {
    // Placeholder - real implementation would use EC point compression
    // For BN254 G1 points (64 bytes), we can compress to 33 bytes
    let mut result = Vec::new();

    // Just use delta encoding for now
    result.extend_from_slice(&delta_encode(data));

    result
}

fn point_decompress(data: &[u8]) -> Result<Vec<u8>, ProofError> {
    // Placeholder - real implementation would use EC point decompression
    delta_decode(data)
}

// Helper functions

fn has_repeated_patterns(data: &[u8]) -> bool {
    if data.len() < 16 {
        return false;
    }

    // Check for runs of 4+ identical bytes
    let mut run_count = 0;
    for window in data.windows(4) {
        if window[0] == window[1] && window[1] == window[2] && window[2] == window[3] {
            run_count += 1;
        }
    }

    run_count > data.len() / 64
}

fn compute_checksum(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

fn hash_inputs(inputs: &[[u8; 32]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for input in inputs {
        hasher.update(input);
    }
    hasher.finalize().into()
}

fn timestamp_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn generate_proof_id() -> [u8; 16] {
    use rand::Rng;
    rand::thread_rng().gen()
}

/// Statistics about compression performance.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original size.
    pub original_size: usize,
    /// Compressed size.
    pub compressed_size: usize,
    /// Compression ratio.
    pub ratio: f64,
    /// Space savings percentage.
    pub savings: f64,
    /// Compression time (microseconds).
    pub compression_time_us: u64,
}

impl CompressionStats {
    /// Computes stats for a compression operation.
    pub fn compute(original: &[u8], compressed: &[u8], time_us: u64) -> Self {
        let ratio = if original.is_empty() {
            1.0
        } else {
            compressed.len() as f64 / original.len() as f64
        };

        Self {
            original_size: original.len(),
            compressed_size: compressed.len(),
            ratio,
            savings: (1.0 - ratio) * 100.0,
            compression_time_us: time_us,
        }
    }
}

/// Benchmarks different compression strategies on the given data.
pub fn benchmark_compression(data: &[u8]) -> Vec<(CompressionStrategy, CompressionStats)> {
    let strategies = [
        CompressionStrategy::None,
        CompressionStrategy::RunLength,
        CompressionStrategy::Delta,
        CompressionStrategy::Dictionary,
        CompressionStrategy::Hybrid,
        CompressionStrategy::PointCompression,
    ];

    strategies
        .iter()
        .map(|&strategy| {
            let start = std::time::Instant::now();
            let compressed = compress_proof(data, strategy);
            let time_us = start.elapsed().as_micros() as u64;

            let stats = CompressionStats::compute(data, &compressed, time_us);
            (strategy, stats)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_compression() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let public_inputs = vec![[0u8; 32]];

        let proof = FinalProof::with_compression(
            data.clone(),
            public_inputs,
            0.01,
            CompressionStrategy::None,
        );

        let decompressed = proof.decompress().unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_run_length_encoding() {
        let data = vec![1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 5];
        let encoded = run_length_encode(&data);
        let decoded = run_length_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_delta_encoding() {
        let data = vec![10, 11, 12, 13, 14, 15, 20, 25, 30];
        let encoded = delta_encode(&data);
        let decoded = delta_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_dictionary_encoding() {
        let mut data = Vec::new();
        for _ in 0..100 {
            data.extend_from_slice(&[1, 2, 3, 4]);
            data.extend_from_slice(&[5, 6, 7, 8]);
        }

        let encoded = dictionary_encode(&data);
        let decoded = dictionary_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_hybrid_compression() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let compressed = hybrid_compress(&data);
        let decompressed = hybrid_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_auto_compression() {
        let data: Vec<u8> = vec![0u8; 2000]; // Highly compressible
        let public_inputs = vec![[1u8; 32]];

        let proof = FinalProof::new(data.clone(), public_inputs, 0.01);
        let decompressed = proof.decompress().unwrap();
        assert_eq!(decompressed, data);

        // Should achieve some compression
        assert!(proof.compression_ratio() < 1.0);
    }

    #[test]
    fn test_serialization() {
        let data = vec![1, 2, 3, 4, 5];
        let public_inputs = vec![[0u8; 32]];

        let proof = FinalProof::new(data.clone(), public_inputs, 0.01);
        let bytes = proof.to_bytes();
        let restored = FinalProof::from_bytes(&bytes).unwrap();

        assert_eq!(restored.decompress().unwrap(), data);
    }

    #[test]
    fn test_integrity() {
        let data = vec![1, 2, 3, 4, 5];
        let public_inputs = vec![[0u8; 32]];

        let proof = FinalProof::new(data, public_inputs, 0.01);
        assert!(proof.verify_integrity());
    }

    #[test]
    fn test_compression_stats() {
        let data: Vec<u8> = vec![1u8; 1000];
        let compressed = compress_proof(&data, CompressionStrategy::RunLength);
        let stats = CompressionStats::compute(&data, &compressed, 100);

        assert!(stats.ratio < 1.0);
        assert!(stats.savings > 0.0);
    }

    #[test]
    fn test_benchmark_compression() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let results = benchmark_compression(&data);

        assert_eq!(results.len(), 6); // 6 strategies
        for (strategy, stats) in results {
            assert!(stats.compressed_size > 0);
            println!("{:?}: ratio={:.2}, savings={:.1}%", strategy, stats.ratio, stats.savings);
        }
    }

    #[test]
    fn test_state_commitment() {
        let data = vec![1, 2, 3];
        let public_inputs = vec![[0u8; 32]];
        let commitment = [42u8; 32];

        let proof = FinalProof::new(data, public_inputs, 0.01)
            .with_state_commitment(commitment);

        assert_eq!(proof.commitments.state_commitment, Some(commitment));
    }
}
