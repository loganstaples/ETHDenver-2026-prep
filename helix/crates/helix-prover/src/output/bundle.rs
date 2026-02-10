//! Proof Bundle Serialization.
//!
//! Provides bundling of proofs with public inputs, metadata, and verification data
//! for efficient storage, transmission, and verification.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::aggregation::AggregatedProof;
use crate::chunking::ChunkId;
use crate::ivc::IVCState;
use crate::parallel::ChunkProof;

/// Magic bytes for bundle files.
const BUNDLE_MAGIC: &[u8] = b"HELIX_BUNDLE";

/// Current bundle format version.
const BUNDLE_VERSION: u8 = 1;

/// A complete proof bundle with all data needed for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofBundle {
    /// Bundle header with metadata.
    pub header: BundleHeader,
    /// The proof data.
    pub proof: BundleProof,
    /// Public inputs for verification.
    pub public_inputs: PublicInputs,
    /// Verification data.
    pub verification: VerificationData,
    /// Optional attached metadata.
    pub metadata: BundleMetadata,
    /// Digital signature (if signed).
    pub signature: Option<BundleSignature>,
}

impl ProofBundle {
    /// Creates a new bundle from a chunk proof.
    pub fn from_chunk_proof(proof: &ChunkProof) -> Self {
        let now = timestamp_now();

        Self {
            header: BundleHeader {
                version: BUNDLE_VERSION,
                proof_type: ProofType::Chunk,
                created_at: now,
                bundle_id: generate_bundle_id(),
            },
            proof: BundleProof::Chunk(ChunkProofData {
                chunk_id: proof.chunk_id.0,
                proof: proof.proof.clone(),
                error_bound: proof.error_bound,
                generation_time_ms: proof.generation_time_ms,
            }),
            public_inputs: PublicInputs {
                inputs: proof.public_inputs.iter().map(|i| i.to_vec()).collect(),
                input_hash: hash_public_inputs(&proof.public_inputs),
            },
            verification: VerificationData::default(),
            metadata: BundleMetadata::default(),
            signature: None,
        }
    }

    /// Creates a new bundle from an aggregated proof.
    pub fn from_aggregated_proof(proof: &AggregatedProof) -> Self {
        let now = timestamp_now();

        Self {
            header: BundleHeader {
                version: BUNDLE_VERSION,
                proof_type: ProofType::Aggregated,
                created_at: now,
                bundle_id: generate_bundle_id(),
            },
            proof: BundleProof::Aggregated(AggregatedProofData {
                aggregation_id: proof.id.0,
                proof: proof.proof.clone(),
                root_commitment: proof.root_commitment,
                total_error_bound: proof.total_error_bound,
                chunk_ids: proof.chunk_ids.iter().map(|c| c.0).collect(),
                num_layers: proof.num_layers,
                depth: proof.depth,
            }),
            public_inputs: PublicInputs {
                inputs: proof.public_inputs.iter().map(|i| i.to_vec()).collect(),
                input_hash: hash_commitments(&proof.public_inputs),
            },
            verification: VerificationData::default(),
            metadata: BundleMetadata::default(),
            signature: None,
        }
    }

    /// Creates a new bundle from an IVC state.
    pub fn from_ivc_state(state: &IVCState) -> Self {
        let now = timestamp_now();

        Self {
            header: BundleHeader {
                version: BUNDLE_VERSION,
                proof_type: ProofType::IVC,
                created_at: now,
                bundle_id: generate_bundle_id(),
            },
            proof: BundleProof::IVC(IVCProofData {
                step: state.step,
                state_commitment: state.state_commitment,
                accumulated_error: state.accumulated_error,
                proof: state.proof.clone().unwrap_or_default(),
                prev_proof_hash: state.prev_proof_hash,
            }),
            public_inputs: PublicInputs {
                inputs: vec![state.state_commitment.to_vec()],
                input_hash: state.state_commitment,
            },
            verification: VerificationData::default(),
            metadata: BundleMetadata::default(),
            signature: None,
        }
    }

    /// Creates a training step bundle combining all proof types.
    pub fn training_step(
        chunk_proofs: Vec<ChunkProof>,
        aggregated: Option<AggregatedProof>,
        ivc_state: Option<IVCState>,
        step_number: u64,
    ) -> Self {
        let now = timestamp_now();

        let chunk_data: Vec<ChunkProofData> = chunk_proofs
            .iter()
            .map(|p| ChunkProofData {
                chunk_id: p.chunk_id.0,
                proof: p.proof.clone(),
                error_bound: p.error_bound,
                generation_time_ms: p.generation_time_ms,
            })
            .collect();

        let aggregated_data = aggregated.as_ref().map(|a| AggregatedProofData {
            aggregation_id: a.id.0,
            proof: a.proof.clone(),
            root_commitment: a.root_commitment,
            total_error_bound: a.total_error_bound,
            chunk_ids: a.chunk_ids.iter().map(|c| c.0).collect(),
            num_layers: a.num_layers,
            depth: a.depth,
        });

        let ivc_data = ivc_state.as_ref().map(|s| IVCProofData {
            step: s.step,
            state_commitment: s.state_commitment,
            accumulated_error: s.accumulated_error,
            proof: s.proof.clone().unwrap_or_default(),
            prev_proof_hash: s.prev_proof_hash,
        });

        // Collect all public inputs
        let mut all_inputs: Vec<Vec<u8>> = chunk_proofs
            .iter()
            .flat_map(|p| p.public_inputs.iter().map(|i| i.to_vec()))
            .collect();

        if let Some(ref a) = aggregated {
            all_inputs.extend(a.public_inputs.iter().map(|i| i.to_vec()));
        }

        let input_hash = {
            let mut hasher = Sha256::new();
            for input in &all_inputs {
                hasher.update(input);
            }
            hasher.finalize().into()
        };

        Self {
            header: BundleHeader {
                version: BUNDLE_VERSION,
                proof_type: ProofType::TrainingStep,
                created_at: now,
                bundle_id: generate_bundle_id(),
            },
            proof: BundleProof::TrainingStep(TrainingStepData {
                step_number,
                chunks: chunk_data,
                aggregated: aggregated_data,
                ivc: ivc_data,
            }),
            public_inputs: PublicInputs {
                inputs: all_inputs,
                input_hash,
            },
            verification: VerificationData::default(),
            metadata: BundleMetadata::default(),
            signature: None,
        }
    }

    /// Sets the verification key data.
    pub fn with_verification_key(mut self, vk_hash: [u8; 32], circuit_id: String) -> Self {
        self.verification.vk_hash = Some(vk_hash);
        self.verification.circuit_id = Some(circuit_id);
        self
    }

    /// Sets the model commitment.
    pub fn with_model_commitment(mut self, commitment: [u8; 32]) -> Self {
        self.metadata.model_commitment = Some(commitment);
        self
    }

    /// Adds custom metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.custom.insert(key.into(), value.into());
        self
    }

    /// Signs the bundle with a private key (placeholder).
    pub fn sign(mut self, signer_id: String, _private_key: &[u8]) -> Self {
        let bundle_hash = self.compute_hash();
        self.signature = Some(BundleSignature {
            signer_id,
            signature: bundle_hash.to_vec(), // Placeholder - real impl would use EC signature
            timestamp: timestamp_now(),
        });
        self
    }

    /// Computes the bundle hash for integrity verification.
    pub fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();

        // Hash header
        hasher.update(&[self.header.version]);
        hasher.update(&[self.header.proof_type as u8]);
        hasher.update(&self.header.created_at.to_le_bytes());

        // Hash public inputs
        hasher.update(&self.public_inputs.input_hash);

        // Hash proof data
        match &self.proof {
            BundleProof::Chunk(c) => hasher.update(&c.proof),
            BundleProof::Aggregated(a) => hasher.update(&a.proof),
            BundleProof::IVC(i) => hasher.update(&i.proof),
            BundleProof::TrainingStep(t) => {
                for chunk in &t.chunks {
                    hasher.update(&chunk.proof);
                }
            }
        }

        hasher.finalize().into()
    }

    /// Verifies the bundle integrity.
    pub fn verify_integrity(&self) -> bool {
        let computed_hash = self.compute_hash();

        if let Some(ref sig) = self.signature {
            // Verify signature matches hash
            sig.signature == computed_hash.to_vec()
        } else {
            true // No signature to verify
        }
    }

    /// Serializes the bundle to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut output = Vec::new();

        // Magic bytes
        output.extend_from_slice(BUNDLE_MAGIC);

        // Version
        output.push(BUNDLE_VERSION);

        // JSON payload
        let payload = serde_json::to_vec(self).unwrap_or_default();
        output.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        output.extend_from_slice(&payload);

        // Checksum
        let checksum: [u8; 32] = Sha256::digest(&payload).into();
        output.extend_from_slice(&checksum);

        output
    }

    /// Deserializes a bundle from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, BundleError> {
        if data.len() < BUNDLE_MAGIC.len() + 1 + 8 + 32 {
            return Err(BundleError::InvalidFormat("Data too short".into()));
        }

        // Check magic bytes
        if &data[..BUNDLE_MAGIC.len()] != BUNDLE_MAGIC {
            return Err(BundleError::InvalidFormat("Invalid magic bytes".into()));
        }

        let offset = BUNDLE_MAGIC.len();

        // Check version
        let version = data[offset];
        if version != BUNDLE_VERSION {
            return Err(BundleError::VersionMismatch {
                expected: BUNDLE_VERSION,
                got: version,
            });
        }

        // Read payload length
        let payload_len = u64::from_le_bytes(
            data[offset + 1..offset + 9]
                .try_into()
                .map_err(|_| BundleError::InvalidFormat("Invalid length".into()))?,
        ) as usize;

        let payload_start = offset + 9;
        let payload_end = payload_start + payload_len;

        if data.len() < payload_end + 32 {
            return Err(BundleError::InvalidFormat("Data truncated".into()));
        }

        let payload = &data[payload_start..payload_end];
        let checksum = &data[payload_end..payload_end + 32];

        // Verify checksum
        let computed: [u8; 32] = Sha256::digest(payload).into();
        if computed != checksum {
            return Err(BundleError::ChecksumMismatch);
        }

        // Deserialize
        serde_json::from_slice(payload).map_err(|e| BundleError::DeserializationError(e.to_string()))
    }

    /// Writes the bundle to a file.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        fs::write(path, self.to_bytes())
    }

    /// Reads a bundle from a file.
    pub fn read_from_file(path: impl AsRef<Path>) -> Result<Self, BundleError> {
        let data = fs::read(path).map_err(|e| BundleError::IoError(e.to_string()))?;
        Self::from_bytes(&data)
    }

    /// Returns the proof type.
    pub fn proof_type(&self) -> ProofType {
        self.header.proof_type
    }

    /// Returns the bundle size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.to_bytes().len()
    }

    /// Extracts chunk proofs from a training step bundle.
    pub fn extract_chunk_proofs(&self) -> Vec<ChunkProof> {
        match &self.proof {
            BundleProof::Chunk(c) => vec![ChunkProof {
                chunk_id: ChunkId(c.chunk_id),
                proof: c.proof.clone(),
                public_inputs: self.public_inputs.inputs.iter()
                    .filter_map(|i| i.clone().try_into().ok())
                    .collect(),
                error_bound: c.error_bound,
                generation_time_ms: c.generation_time_ms,
            }],
            BundleProof::TrainingStep(t) => t.chunks.iter().map(|c| ChunkProof {
                chunk_id: ChunkId(c.chunk_id),
                proof: c.proof.clone(),
                public_inputs: Vec::new(),
                error_bound: c.error_bound,
                generation_time_ms: c.generation_time_ms,
            }).collect(),
            _ => Vec::new(),
        }
    }
}

/// Bundle header with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleHeader {
    /// Format version.
    pub version: u8,
    /// Proof type.
    pub proof_type: ProofType,
    /// Creation timestamp.
    pub created_at: u64,
    /// Unique bundle identifier.
    pub bundle_id: [u8; 16],
}

/// Type of proof in the bundle.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ProofType {
    /// Single chunk proof.
    Chunk = 0,
    /// Aggregated proof.
    Aggregated = 1,
    /// IVC proof.
    IVC = 2,
    /// Complete training step.
    TrainingStep = 3,
}

/// Proof data variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BundleProof {
    /// Single chunk proof.
    Chunk(ChunkProofData),
    /// Aggregated proof.
    Aggregated(AggregatedProofData),
    /// IVC proof.
    IVC(IVCProofData),
    /// Training step with all components.
    TrainingStep(TrainingStepData),
}

/// Chunk proof data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkProofData {
    /// Chunk ID.
    pub chunk_id: u64,
    /// Proof bytes.
    pub proof: Vec<u8>,
    /// Error bound.
    pub error_bound: f64,
    /// Generation time in milliseconds.
    pub generation_time_ms: u64,
}

/// Aggregated proof data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedProofData {
    /// Aggregation ID.
    pub aggregation_id: u64,
    /// Proof bytes.
    pub proof: Vec<u8>,
    /// Root commitment.
    pub root_commitment: [u8; 32],
    /// Total error bound.
    pub total_error_bound: f64,
    /// Included chunk IDs.
    pub chunk_ids: Vec<u64>,
    /// Number of layers.
    pub num_layers: usize,
    /// Aggregation depth.
    pub depth: u32,
}

/// IVC proof data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVCProofData {
    /// Step number.
    pub step: u64,
    /// State commitment.
    pub state_commitment: [u8; 32],
    /// Accumulated error.
    pub accumulated_error: f64,
    /// Proof bytes.
    pub proof: Vec<u8>,
    /// Previous proof hash.
    pub prev_proof_hash: Option<[u8; 32]>,
}

/// Training step data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStepData {
    /// Step number.
    pub step_number: u64,
    /// Individual chunk proofs.
    pub chunks: Vec<ChunkProofData>,
    /// Optional aggregated proof.
    pub aggregated: Option<AggregatedProofData>,
    /// Optional IVC state.
    pub ivc: Option<IVCProofData>,
}

/// Public inputs for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicInputs {
    /// Raw public input bytes.
    pub inputs: Vec<Vec<u8>>,
    /// Hash of all inputs.
    pub input_hash: [u8; 32],
}

/// Verification data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationData {
    /// Hash of the verification key.
    pub vk_hash: Option<[u8; 32]>,
    /// Circuit identifier.
    pub circuit_id: Option<String>,
    /// SRS parameters (for on-chain verification).
    pub srs_hash: Option<[u8; 32]>,
    /// Verifier contract address (if deployed).
    pub verifier_address: Option<String>,
}

/// Bundle metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BundleMetadata {
    /// Model commitment (for MPC training).
    pub model_commitment: Option<[u8; 32]>,
    /// Training job ID.
    pub job_id: Option<String>,
    /// Worker ID.
    pub worker_id: Option<String>,
    /// Batch number.
    pub batch_number: Option<u64>,
    /// Custom key-value metadata.
    pub custom: HashMap<String, String>,
}

/// Digital signature for the bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleSignature {
    /// Signer identifier.
    pub signer_id: String,
    /// Signature bytes.
    pub signature: Vec<u8>,
    /// Signing timestamp.
    pub timestamp: u64,
}

/// Errors when working with bundles.
#[derive(Debug)]
pub enum BundleError {
    /// Invalid bundle format.
    InvalidFormat(String),
    /// Version mismatch.
    VersionMismatch { expected: u8, got: u8 },
    /// Checksum verification failed.
    ChecksumMismatch,
    /// Deserialization error.
    DeserializationError(String),
    /// IO error.
    IoError(String),
}

impl std::fmt::Display for BundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BundleError::InvalidFormat(msg) => write!(f, "Invalid bundle format: {}", msg),
            BundleError::VersionMismatch { expected, got } => {
                write!(f, "Version mismatch: expected {}, got {}", expected, got)
            }
            BundleError::ChecksumMismatch => write!(f, "Checksum verification failed"),
            BundleError::DeserializationError(msg) => write!(f, "Deserialization error: {}", msg),
            BundleError::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for BundleError {}

/// A collection of bundles for batch processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleCollection {
    /// Collection header.
    pub header: CollectionHeader,
    /// Individual bundles.
    pub bundles: Vec<ProofBundle>,
}

impl BundleCollection {
    /// Creates a new empty collection.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            header: CollectionHeader {
                name: name.into(),
                created_at: timestamp_now(),
                bundle_count: 0,
            },
            bundles: Vec::new(),
        }
    }

    /// Adds a bundle to the collection.
    pub fn add(&mut self, bundle: ProofBundle) {
        self.bundles.push(bundle);
        self.header.bundle_count = self.bundles.len();
    }

    /// Returns the total size of all bundles.
    pub fn total_size_bytes(&self) -> usize {
        self.bundles.iter().map(|b| b.size_bytes()).sum()
    }

    /// Serializes the collection to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserializes a collection from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, BundleError> {
        serde_json::from_slice(data).map_err(|e| BundleError::DeserializationError(e.to_string()))
    }
}

/// Collection header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionHeader {
    /// Collection name.
    pub name: String,
    /// Creation timestamp.
    pub created_at: u64,
    /// Number of bundles.
    pub bundle_count: usize,
}

// Helper functions

fn timestamp_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn generate_bundle_id() -> [u8; 16] {
    use rand::Rng;
    rand::thread_rng().gen()
}

fn hash_public_inputs(inputs: &[[u8; 32]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for input in inputs {
        hasher.update(input);
    }
    hasher.finalize().into()
}

fn hash_commitments(commitments: &[[u8; 32]]) -> [u8; 32] {
    hash_public_inputs(commitments)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_chunk_proof() -> ChunkProof {
        ChunkProof {
            chunk_id: ChunkId(42),
            proof: vec![1, 2, 3, 4, 5],
            public_inputs: vec![[1; 32], [2; 32]],
            error_bound: 0.001,
            generation_time_ms: 100,
        }
    }

    #[test]
    fn test_bundle_from_chunk_proof() {
        let proof = make_test_chunk_proof();
        let bundle = ProofBundle::from_chunk_proof(&proof);

        assert_eq!(bundle.header.proof_type, ProofType::Chunk);
        assert_eq!(bundle.header.version, BUNDLE_VERSION);

        assert!(
            matches!(&bundle.proof, BundleProof::Chunk(c) if c.chunk_id == 42 && c.proof == proof.proof),
            "Expected Chunk proof with chunk_id=42, got {:?}", bundle.proof_type()
        );
    }

    #[test]
    fn test_bundle_serialization() {
        let proof = make_test_chunk_proof();
        let bundle = ProofBundle::from_chunk_proof(&proof);

        let bytes = bundle.to_bytes();
        let restored = ProofBundle::from_bytes(&bytes).unwrap();

        assert_eq!(restored.header.proof_type, bundle.header.proof_type);
        assert_eq!(restored.public_inputs.input_hash, bundle.public_inputs.input_hash);
    }

    #[test]
    fn test_bundle_integrity() {
        let proof = make_test_chunk_proof();
        let bundle = ProofBundle::from_chunk_proof(&proof)
            .sign("test_signer".into(), &[]);

        assert!(bundle.verify_integrity());
    }

    #[test]
    fn test_bundle_metadata() {
        let proof = make_test_chunk_proof();
        let bundle = ProofBundle::from_chunk_proof(&proof)
            .with_verification_key([0; 32], "test_circuit".into())
            .with_model_commitment([1; 32])
            .with_metadata("key", "value");

        assert!(bundle.verification.vk_hash.is_some());
        assert!(bundle.metadata.model_commitment.is_some());
        assert_eq!(bundle.metadata.custom.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_training_step_bundle() {
        let chunks = vec![make_test_chunk_proof(), make_test_chunk_proof()];
        let bundle = ProofBundle::training_step(chunks, None, None, 1);

        assert_eq!(bundle.header.proof_type, ProofType::TrainingStep);

        assert!(
            matches!(&bundle.proof, BundleProof::TrainingStep(t) if t.step_number == 1 && t.chunks.len() == 2),
            "Expected TrainingStep proof with step_number=1 and 2 chunks, got {:?}", bundle.proof_type()
        );
    }

    #[test]
    fn test_bundle_collection() {
        let mut collection = BundleCollection::new("test_collection");

        for i in 0..3 {
            let mut proof = make_test_chunk_proof();
            proof.chunk_id = ChunkId(i);
            collection.add(ProofBundle::from_chunk_proof(&proof));
        }

        assert_eq!(collection.bundles.len(), 3);
        assert_eq!(collection.header.bundle_count, 3);
    }

    #[test]
    fn test_invalid_bundle() {
        let result = ProofBundle::from_bytes(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }
}
