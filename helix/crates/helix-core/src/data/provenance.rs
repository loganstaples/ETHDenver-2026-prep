//! Data Provenance Tracking for HELIX.
//!
//! Provides comprehensive tracking of training data lineage:
//! - Data origin and source attribution
//! - Transformation history
//! - Custody chain for audit trails
//! - Cryptographic attestations
//! - On-chain anchoring support

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::time::SystemTime;

use super::merkle::{Hash, MerkleHasher, MerkleTree, MerkleTreeBuilder, Sha256Hasher, HASH_SIZE};
use crate::traits::BinarySerializable;
use crate::traits::serializable::SerializeError;

/// Unique identifier for a provenance record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProvenanceId(pub [u8; 16]);

impl ProvenanceId {
    /// Creates a new random provenance ID.
    pub fn new() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut id = [0u8; 16];
        rng.fill(&mut id);
        Self(id)
    }

    /// Creates from bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Converts to hex string.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

impl Default for ProvenanceId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ProvenanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Origin types for training data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataOrigin {
    /// Data from IPFS with CID.
    Ipfs {
        cid: String,
        gateway: Option<String>,
    },
    /// Data from Filecoin with deal ID.
    Filecoin {
        deal_id: String,
        miner_id: String,
    },
    /// Data from Arweave with transaction ID.
    Arweave { tx_id: String },
    /// Data from S3-compatible storage.
    S3 {
        bucket: String,
        key: String,
        region: String,
        endpoint: Option<String>,
    },
    /// Data from HTTP URL.
    Http { url: String, etag: Option<String> },
    /// Data generated synthetically.
    Synthetic {
        generator: String,
        seed: Option<u64>,
        parameters: HashMap<String, String>,
    },
    /// Data from on-chain source.
    OnChain {
        chain_id: u64,
        contract: String,
        block_range: (u64, u64),
    },
    /// Custom/unknown origin.
    Custom {
        source_type: String,
        identifier: String,
        metadata: HashMap<String, String>,
    },
}

impl DataOrigin {
    /// Returns a canonical identifier for this origin.
    pub fn canonical_id(&self) -> String {
        match self {
            DataOrigin::Ipfs { cid, .. } => format!("ipfs://{}", cid),
            DataOrigin::Filecoin { deal_id, .. } => format!("fil://deal/{}", deal_id),
            DataOrigin::Arweave { tx_id } => format!("ar://{}", tx_id),
            DataOrigin::S3 { bucket, key, .. } => format!("s3://{}/{}", bucket, key),
            DataOrigin::Http { url, .. } => url.clone(),
            DataOrigin::Synthetic { generator, seed, .. } => {
                format!("synthetic://{}?seed={:?}", generator, seed)
            }
            DataOrigin::OnChain {
                chain_id,
                contract,
                block_range,
            } => format!(
                "chain://{}:{}@{}-{}",
                chain_id, contract, block_range.0, block_range.1
            ),
            DataOrigin::Custom {
                source_type,
                identifier,
                ..
            } => format!("{}://{}", source_type, identifier),
        }
    }

    /// Computes a hash of this origin for commitment.
    pub fn hash(&self) -> Hash {
        let canonical = self.canonical_id();
        Sha256Hasher.hash_leaf(canonical.as_bytes())
    }
}

/// A transformation applied to data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransformation {
    /// Unique ID for this transformation.
    pub id: ProvenanceId,
    /// Type of transformation.
    pub transform_type: TransformationType,
    /// Timestamp when applied.
    pub timestamp: u64,
    /// Hash of input data.
    pub input_hash: Hash,
    /// Hash of output data.
    pub output_hash: Hash,
    /// Parameters used.
    pub parameters: HashMap<String, String>,
    /// Who applied the transformation.
    pub applied_by: Option<String>,
}

impl DataTransformation {
    /// Creates a new transformation record.
    pub fn new(
        transform_type: TransformationType,
        input_hash: Hash,
        output_hash: Hash,
        parameters: HashMap<String, String>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            id: ProvenanceId::new(),
            transform_type,
            timestamp,
            input_hash,
            output_hash,
            parameters,
            applied_by: None,
        }
    }

    /// Computes a hash of this transformation.
    pub fn hash(&self) -> Hash {
        let mut data = Vec::new();
        data.extend_from_slice(&self.id.0);
        data.extend_from_slice(&self.timestamp.to_le_bytes());
        data.extend_from_slice(self.input_hash.as_bytes());
        data.extend_from_slice(self.output_hash.as_bytes());
        Sha256Hasher.hash_leaf(&data)
    }
}

/// Types of data transformations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransformationType {
    /// Normalization (min-max, z-score, etc.).
    Normalization { method: String },
    /// Data augmentation.
    Augmentation { method: String },
    /// Filtering/selection.
    Filter { criteria: String },
    /// Sampling.
    Sample { method: String, ratio: f64 },
    /// Shuffling.
    Shuffle { seed: Option<u64> },
    /// Format conversion.
    FormatConversion { from: String, to: String },
    /// Tokenization (for text data).
    Tokenization { tokenizer: String },
    /// Batching.
    Batching { batch_size: usize },
    /// Sharding.
    Sharding { num_shards: usize, strategy: String },
    /// Custom transformation.
    Custom { name: String, description: String },
}

/// A custody record in the data chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustodyRecord {
    /// Unique ID.
    pub id: ProvenanceId,
    /// Timestamp of custody transfer.
    pub timestamp: u64,
    /// Previous custodian (if any).
    pub from: Option<Custodian>,
    /// New custodian.
    pub to: Custodian,
    /// Data hash at transfer.
    pub data_hash: Hash,
    /// Purpose of transfer.
    pub purpose: String,
    /// Attestation from transferring party.
    pub attestation: Option<Attestation>,
}

impl CustodyRecord {
    /// Creates a new custody record.
    pub fn new(from: Option<Custodian>, to: Custodian, data_hash: Hash, purpose: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            id: ProvenanceId::new(),
            timestamp,
            from,
            to,
            data_hash,
            purpose,
            attestation: None,
        }
    }

    /// Computes a hash of this record.
    pub fn hash(&self) -> Hash {
        let mut data = Vec::new();
        data.extend_from_slice(&self.id.0);
        data.extend_from_slice(&self.timestamp.to_le_bytes());
        data.extend_from_slice(self.data_hash.as_bytes());
        data.extend_from_slice(self.purpose.as_bytes());
        Sha256Hasher.hash_leaf(&data)
    }
}

/// A custodian of data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Custodian {
    /// Type of custodian.
    pub custodian_type: CustodianType,
    /// Identifier (address, name, etc.).
    pub identifier: String,
    /// Public key for verification (hex encoded).
    pub public_key: Option<String>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

impl Custodian {
    /// Creates a new custodian.
    pub fn new(custodian_type: CustodianType, identifier: String) -> Self {
        Self {
            custodian_type,
            identifier,
            public_key: None,
            metadata: HashMap::new(),
        }
    }

    /// Creates a worker custodian.
    pub fn worker(address: String) -> Self {
        Self::new(CustodianType::Worker, address)
    }

    /// Creates a model owner custodian.
    pub fn model_owner(address: String) -> Self {
        Self::new(CustodianType::ModelOwner, address)
    }
}

/// Types of custodians.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CustodianType {
    /// Data provider/creator.
    DataProvider,
    /// Model owner.
    ModelOwner,
    /// Compute worker.
    Worker,
    /// Aggregator node.
    Aggregator,
    /// Storage provider.
    StorageProvider,
    /// Smart contract.
    Contract,
    /// Custom type.
    Custom(String),
}

/// A cryptographic attestation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attestation {
    /// Type of attestation.
    pub attestation_type: AttestationType,
    /// The attester.
    pub attester: String,
    /// Statement being attested.
    pub statement_hash: Hash,
    /// Signature (hex encoded).
    pub signature: String,
    /// Timestamp.
    pub timestamp: u64,
    /// Optional on-chain transaction hash.
    pub tx_hash: Option<String>,
}

impl Attestation {
    /// Creates a new attestation.
    pub fn new(
        attestation_type: AttestationType,
        attester: String,
        statement_hash: Hash,
        signature: String,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            attestation_type,
            attester,
            statement_hash,
            signature,
            timestamp,
            tx_hash: None,
        }
    }

    /// Returns the attestation hash.
    pub fn hash(&self) -> Hash {
        let mut data = Vec::new();
        data.extend_from_slice(self.attester.as_bytes());
        data.extend_from_slice(self.statement_hash.as_bytes());
        data.extend_from_slice(&self.timestamp.to_le_bytes());
        Sha256Hasher.hash_leaf(&data)
    }
}

/// Types of attestations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AttestationType {
    /// Origin verification.
    OriginVerification,
    /// Data integrity.
    DataIntegrity,
    /// Custody transfer.
    CustodyTransfer,
    /// Computation verification.
    ComputationVerification,
    /// On-chain proof.
    OnChainProof,
    /// Custom attestation.
    Custom(String),
}

/// Complete provenance record for a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    /// Unique ID.
    pub id: ProvenanceId,
    /// Data origin.
    pub origin: DataOrigin,
    /// Hash of the original data.
    pub original_hash: Hash,
    /// Current hash (after transformations).
    pub current_hash: Hash,
    /// Ordered list of transformations.
    pub transformations: Vec<DataTransformation>,
    /// Custody chain.
    pub custody_chain: Vec<CustodyRecord>,
    /// Attestations.
    pub attestations: Vec<Attestation>,
    /// Creation timestamp.
    pub created_at: u64,
    /// Last update timestamp.
    pub updated_at: u64,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

impl ProvenanceRecord {
    /// Creates a new provenance record.
    pub fn new(origin: DataOrigin, original_hash: Hash) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            id: ProvenanceId::new(),
            origin,
            original_hash,
            current_hash: original_hash,
            transformations: Vec::new(),
            custody_chain: Vec::new(),
            attestations: Vec::new(),
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Records a transformation.
    pub fn add_transformation(&mut self, transform: DataTransformation) {
        self.current_hash = transform.output_hash;
        self.transformations.push(transform);
        self.touch();
    }

    /// Records a custody transfer.
    pub fn add_custody_transfer(&mut self, record: CustodyRecord) {
        self.custody_chain.push(record);
        self.touch();
    }

    /// Adds an attestation.
    pub fn add_attestation(&mut self, attestation: Attestation) {
        self.attestations.push(attestation);
        self.touch();
    }

    /// Updates the timestamp.
    fn touch(&mut self) {
        self.updated_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }

    /// Returns the current custodian.
    pub fn current_custodian(&self) -> Option<&Custodian> {
        self.custody_chain.last().map(|r| &r.to)
    }

    /// Computes a Merkle root of the provenance chain.
    pub fn provenance_root(&self) -> Hash {
        let mut builder = MerkleTreeBuilder::with_sha256();

        // Add origin hash
        builder = builder.add_hash(self.origin.hash());

        // Add transformation hashes
        for t in &self.transformations {
            builder = builder.add_hash(t.hash());
        }

        // Add custody hashes
        for c in &self.custody_chain {
            builder = builder.add_hash(c.hash());
        }

        // Add attestation hashes
        for a in &self.attestations {
            builder = builder.add_hash(a.hash());
        }

        match builder.build() {
            Ok(tree) => tree.root().unwrap_or(Hash::zero()),
            Err(_) => Hash::zero(),
        }
    }

    /// Verifies the transformation chain.
    pub fn verify_transformation_chain(&self) -> bool {
        if self.transformations.is_empty() {
            return self.original_hash == self.current_hash;
        }

        // First transformation should start from original hash
        if self.transformations[0].input_hash != self.original_hash {
            return false;
        }

        // Each transformation output should match next input
        for window in self.transformations.windows(2) {
            if window[0].output_hash != window[1].input_hash {
                return false;
            }
        }

        // Last transformation output should match current hash
        self.transformations.last().map(|t| t.output_hash) == Some(self.current_hash)
    }

    /// Returns a summary for on-chain storage.
    pub fn to_chain_summary(&self) -> ProvenanceChainSummary {
        ProvenanceChainSummary {
            id: self.id,
            origin_hash: self.origin.hash(),
            original_hash: self.original_hash,
            current_hash: self.current_hash,
            provenance_root: self.provenance_root(),
            transformation_count: self.transformations.len() as u32,
            custody_transfers: self.custody_chain.len() as u32,
            attestation_count: self.attestations.len() as u32,
            created_at: self.created_at,
            updated_at: self.updated_at,
        }
    }
}

/// Compact provenance summary for on-chain storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceChainSummary {
    /// Provenance ID.
    pub id: ProvenanceId,
    /// Hash of the origin.
    pub origin_hash: Hash,
    /// Original data hash.
    pub original_hash: Hash,
    /// Current data hash.
    pub current_hash: Hash,
    /// Merkle root of all provenance data.
    pub provenance_root: Hash,
    /// Number of transformations.
    pub transformation_count: u32,
    /// Number of custody transfers.
    pub custody_transfers: u32,
    /// Number of attestations.
    pub attestation_count: u32,
    /// Creation timestamp.
    pub created_at: u64,
    /// Last update timestamp.
    pub updated_at: u64,
}

impl ProvenanceChainSummary {
    /// Size in bytes when serialized compactly.
    pub const COMPACT_SIZE: usize = 16 + 32 + 32 + 32 + 32 + 4 + 4 + 4 + 8 + 8; // 172 bytes

    /// Serializes to compact bytes.
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::COMPACT_SIZE);
        bytes.extend_from_slice(&self.id.0);
        bytes.extend_from_slice(self.origin_hash.as_bytes());
        bytes.extend_from_slice(self.original_hash.as_bytes());
        bytes.extend_from_slice(self.current_hash.as_bytes());
        bytes.extend_from_slice(self.provenance_root.as_bytes());
        bytes.extend_from_slice(&self.transformation_count.to_le_bytes());
        bytes.extend_from_slice(&self.custody_transfers.to_le_bytes());
        bytes.extend_from_slice(&self.attestation_count.to_le_bytes());
        bytes.extend_from_slice(&self.created_at.to_le_bytes());
        bytes.extend_from_slice(&self.updated_at.to_le_bytes());
        bytes
    }

    /// Parses from compact bytes.
    pub fn from_compact_bytes(bytes: &[u8]) -> Result<Self, ProvenanceError> {
        if bytes.len() < Self::COMPACT_SIZE {
            return Err(ProvenanceError::InvalidFormat("Bytes too short".into()));
        }

        let mut id = [0u8; 16];
        id.copy_from_slice(&bytes[0..16]);

        let mut origin_hash = [0u8; 32];
        origin_hash.copy_from_slice(&bytes[16..48]);

        let mut original_hash = [0u8; 32];
        original_hash.copy_from_slice(&bytes[48..80]);

        let mut current_hash = [0u8; 32];
        current_hash.copy_from_slice(&bytes[80..112]);

        let mut provenance_root = [0u8; 32];
        provenance_root.copy_from_slice(&bytes[112..144]);

        let mut tc = [0u8; 4];
        tc.copy_from_slice(&bytes[144..148]);
        let transformation_count = u32::from_le_bytes(tc);

        let mut ct = [0u8; 4];
        ct.copy_from_slice(&bytes[148..152]);
        let custody_transfers = u32::from_le_bytes(ct);

        let mut ac = [0u8; 4];
        ac.copy_from_slice(&bytes[152..156]);
        let attestation_count = u32::from_le_bytes(ac);

        let mut ca = [0u8; 8];
        ca.copy_from_slice(&bytes[156..164]);
        let created_at = u64::from_le_bytes(ca);

        let mut ua = [0u8; 8];
        ua.copy_from_slice(&bytes[164..172]);
        let updated_at = u64::from_le_bytes(ua);

        Ok(Self {
            id: ProvenanceId(id),
            origin_hash: Hash(origin_hash),
            original_hash: Hash(original_hash),
            current_hash: Hash(current_hash),
            provenance_root: Hash(provenance_root),
            transformation_count,
            custody_transfers,
            attestation_count,
            created_at,
            updated_at,
        })
    }
}

/// Manages provenance records for multiple datasets.
pub struct ProvenanceRegistry {
    /// All records indexed by ID.
    records: HashMap<ProvenanceId, ProvenanceRecord>,
    /// Index by original hash.
    by_original_hash: HashMap<Hash, Vec<ProvenanceId>>,
    /// Index by current hash.
    by_current_hash: HashMap<Hash, Vec<ProvenanceId>>,
}

impl ProvenanceRegistry {
    /// Creates a new registry.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            by_original_hash: HashMap::new(),
            by_current_hash: HashMap::new(),
        }
    }

    /// Registers a provenance record.
    pub fn register(&mut self, record: ProvenanceRecord) {
        let id = record.id;
        let original = record.original_hash;
        let current = record.current_hash;

        self.records.insert(id, record);
        self.by_original_hash
            .entry(original)
            .or_default()
            .push(id);
        self.by_current_hash.entry(current).or_default().push(id);
    }

    /// Gets a record by ID.
    pub fn get(&self, id: &ProvenanceId) -> Option<&ProvenanceRecord> {
        self.records.get(id)
    }

    /// Gets a mutable record by ID.
    pub fn get_mut(&mut self, id: &ProvenanceId) -> Option<&mut ProvenanceRecord> {
        self.records.get_mut(id)
    }

    /// Finds records by original hash.
    pub fn find_by_original(&self, hash: &Hash) -> Vec<&ProvenanceRecord> {
        self.by_original_hash
            .get(hash)
            .map(|ids| ids.iter().filter_map(|id| self.records.get(id)).collect())
            .unwrap_or_default()
    }

    /// Finds records by current hash.
    pub fn find_by_current(&self, hash: &Hash) -> Vec<&ProvenanceRecord> {
        self.by_current_hash
            .get(hash)
            .map(|ids| ids.iter().filter_map(|id| self.records.get(id)).collect())
            .unwrap_or_default()
    }

    /// Returns all records.
    pub fn all(&self) -> impl Iterator<Item = &ProvenanceRecord> {
        self.records.values()
    }

    /// Returns the number of records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl Default for ProvenanceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating provenance records.
pub struct ProvenanceBuilder {
    origin: DataOrigin,
    original_hash: Hash,
    transformations: Vec<DataTransformation>,
    custody_chain: Vec<CustodyRecord>,
    attestations: Vec<Attestation>,
    metadata: HashMap<String, String>,
}

impl ProvenanceBuilder {
    /// Creates a new builder.
    pub fn new(origin: DataOrigin, original_hash: Hash) -> Self {
        Self {
            origin,
            original_hash,
            transformations: Vec::new(),
            custody_chain: Vec::new(),
            attestations: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Adds a transformation.
    pub fn add_transformation(mut self, transform: DataTransformation) -> Self {
        self.transformations.push(transform);
        self
    }

    /// Adds a custody transfer.
    pub fn add_custody(mut self, record: CustodyRecord) -> Self {
        self.custody_chain.push(record);
        self
    }

    /// Adds an attestation.
    pub fn add_attestation(mut self, attestation: Attestation) -> Self {
        self.attestations.push(attestation);
        self
    }

    /// Adds metadata.
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Builds the provenance record.
    pub fn build(self) -> ProvenanceRecord {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let current_hash = self
            .transformations
            .last()
            .map(|t| t.output_hash)
            .unwrap_or(self.original_hash);

        ProvenanceRecord {
            id: ProvenanceId::new(),
            origin: self.origin,
            original_hash: self.original_hash,
            current_hash,
            transformations: self.transformations,
            custody_chain: self.custody_chain,
            attestations: self.attestations,
            created_at: now,
            updated_at: now,
            metadata: self.metadata,
        }
    }
}

/// Errors for provenance operations.
#[derive(Debug, Clone)]
pub enum ProvenanceError {
    /// Invalid format.
    InvalidFormat(String),
    /// Record not found.
    NotFound(ProvenanceId),
    /// Verification failed.
    VerificationFailed(String),
    /// Chain broken.
    ChainBroken { index: usize, reason: String },
}

impl std::fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProvenanceError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            ProvenanceError::NotFound(id) => write!(f, "Record not found: {}", id),
            ProvenanceError::VerificationFailed(msg) => write!(f, "Verification failed: {}", msg),
            ProvenanceError::ChainBroken { index, reason } => {
                write!(f, "Chain broken at {}: {}", index, reason)
            }
        }
    }
}

impl std::error::Error for ProvenanceError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_id() {
        let id1 = ProvenanceId::new();
        let id2 = ProvenanceId::new();
        assert_ne!(id1, id2);
        assert_eq!(id1.to_hex().len(), 32);
    }

    #[test]
    fn test_data_origin() {
        let origin = DataOrigin::Ipfs {
            cid: "QmTest123".to_string(),
            gateway: None,
        };
        assert!(origin.canonical_id().starts_with("ipfs://"));
        assert!(!origin.hash().is_zero());
    }

    #[test]
    fn test_transformation() {
        let input = Hash::from_slice(b"input");
        let output = Hash::from_slice(b"output");

        let transform = DataTransformation::new(
            TransformationType::Normalization {
                method: "z-score".to_string(),
            },
            input,
            output,
            HashMap::new(),
        );

        assert_eq!(transform.input_hash, input);
        assert_eq!(transform.output_hash, output);
        assert!(!transform.hash().is_zero());
    }

    #[test]
    fn test_custody_record() {
        let from = Custodian::new(CustodianType::DataProvider, "provider".to_string());
        let to = Custodian::worker("0x1234".to_string());
        let hash = Hash::from_slice(b"data");

        let record = CustodyRecord::new(Some(from), to, hash, "training".to_string());

        assert_eq!(record.data_hash, hash);
        assert!(!record.hash().is_zero());
    }

    #[test]
    fn test_provenance_record() {
        let origin = DataOrigin::Synthetic {
            generator: "test".to_string(),
            seed: Some(42),
            parameters: HashMap::new(),
        };
        let hash = Hash::from_slice(b"data");

        let record = ProvenanceRecord::new(origin, hash);

        assert_eq!(record.original_hash, hash);
        assert_eq!(record.current_hash, hash);
        assert!(record.verify_transformation_chain());
    }

    #[test]
    fn test_provenance_with_transformations() {
        let origin = DataOrigin::Synthetic {
            generator: "test".to_string(),
            seed: Some(42),
            parameters: HashMap::new(),
        };
        let original = Hash::from_slice(b"original");
        let t1_out = Hash::from_slice(b"after_norm");
        let t2_out = Hash::from_slice(b"after_shuffle");

        let mut record = ProvenanceRecord::new(origin, original);

        record.add_transformation(DataTransformation::new(
            TransformationType::Normalization {
                method: "minmax".to_string(),
            },
            original,
            t1_out,
            HashMap::new(),
        ));

        record.add_transformation(DataTransformation::new(
            TransformationType::Shuffle { seed: Some(42) },
            t1_out,
            t2_out,
            HashMap::new(),
        ));

        assert!(record.verify_transformation_chain());
        assert_eq!(record.current_hash, t2_out);
    }

    #[test]
    fn test_broken_transformation_chain() {
        let origin = DataOrigin::Synthetic {
            generator: "test".to_string(),
            seed: Some(42),
            parameters: HashMap::new(),
        };
        let original = Hash::from_slice(b"original");
        let t1_out = Hash::from_slice(b"after_norm");
        let wrong_input = Hash::from_slice(b"wrong");
        let t2_out = Hash::from_slice(b"after_shuffle");

        let mut record = ProvenanceRecord::new(origin, original);

        record.transformations.push(DataTransformation::new(
            TransformationType::Normalization {
                method: "minmax".to_string(),
            },
            original,
            t1_out,
            HashMap::new(),
        ));

        // Add transformation with wrong input hash
        record.transformations.push(DataTransformation::new(
            TransformationType::Shuffle { seed: Some(42) },
            wrong_input, // This breaks the chain
            t2_out,
            HashMap::new(),
        ));

        record.current_hash = t2_out;

        assert!(!record.verify_transformation_chain());
    }

    #[test]
    fn test_provenance_chain_summary() {
        let origin = DataOrigin::Ipfs {
            cid: "QmTest".to_string(),
            gateway: None,
        };
        let hash = Hash::from_slice(b"data");
        let record = ProvenanceRecord::new(origin, hash);

        let summary = record.to_chain_summary();

        assert_eq!(summary.original_hash, hash);
        assert_eq!(summary.current_hash, hash);
        assert_eq!(summary.transformation_count, 0);
    }

    #[test]
    fn test_provenance_chain_summary_serialization() {
        let origin = DataOrigin::Ipfs {
            cid: "QmTest".to_string(),
            gateway: None,
        };
        let hash = Hash::from_slice(b"data");
        let record = ProvenanceRecord::new(origin, hash);

        let summary = record.to_chain_summary();
        let bytes = summary.to_compact_bytes();
        assert_eq!(bytes.len(), ProvenanceChainSummary::COMPACT_SIZE);

        let restored = ProvenanceChainSummary::from_compact_bytes(&bytes).unwrap();
        assert_eq!(summary.original_hash, restored.original_hash);
        assert_eq!(summary.current_hash, restored.current_hash);
    }

    #[test]
    fn test_provenance_registry() {
        let mut registry = ProvenanceRegistry::new();

        let origin = DataOrigin::Synthetic {
            generator: "test".to_string(),
            seed: Some(42),
            parameters: HashMap::new(),
        };
        let hash = Hash::from_slice(b"data");
        let record = ProvenanceRecord::new(origin, hash);
        let id = record.id;

        registry.register(record);

        assert_eq!(registry.len(), 1);
        assert!(registry.get(&id).is_some());
        assert_eq!(registry.find_by_original(&hash).len(), 1);
    }

    #[test]
    fn test_provenance_builder() {
        let origin = DataOrigin::Ipfs {
            cid: "QmTest".to_string(),
            gateway: None,
        };
        let original = Hash::from_slice(b"original");
        let transformed = Hash::from_slice(b"transformed");

        let record = ProvenanceBuilder::new(origin, original)
            .add_transformation(DataTransformation::new(
                TransformationType::Normalization {
                    method: "z-score".to_string(),
                },
                original,
                transformed,
                HashMap::new(),
            ))
            .with_metadata("version".to_string(), "1.0".to_string())
            .build();

        assert_eq!(record.original_hash, original);
        assert_eq!(record.current_hash, transformed);
        assert_eq!(record.transformations.len(), 1);
    }
}
