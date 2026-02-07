//! Dataset Commitment Registry for HELIX.
//!
//! Provides a registry for tracking and verifying dataset commitments
//! in decentralized ML training. This enables:
//!
//! - Verifiable proof that training used the committed dataset
//! - Tracking of dataset provenance and transformations
//! - Multi-party coordination on shared datasets
//! - Audit trail for compliance and reproducibility
//!
//! Each dataset entry includes:
//! - Merkle root commitment
//! - Sample count and metadata
//! - Shuffle seed commitment (if shuffled)
//! - Provenance chain
//! - Attestations from data providers

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::data::merkle::{Hash, MerkleHasher, Sha256Hasher};
use crate::data::shuffling::{ShuffleSeed, ShuffleSchedule};
use crate::data::provenance::DataOrigin;

/// Unique identifier for a dataset.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DatasetId(pub String);

impl DatasetId {
    /// Creates a new dataset ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generates a deterministic ID from the Merkle root.
    pub fn from_root(root: &Hash) -> Self {
        Self(format!("ds_{}", &root.to_hex()[..16]))
    }

    /// Generates a random-ish ID.
    pub fn generate() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        Self(format!("ds_{:016x}", now))
    }
}

impl std::fmt::Display for DatasetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Status of a dataset in the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DatasetStatus {
    /// Dataset is pending verification.
    Pending,
    /// Dataset is verified and active.
    Active,
    /// Dataset has been deprecated.
    Deprecated,
    /// Dataset failed verification.
    Invalid,
    /// Dataset is being processed.
    Processing,
}

/// Metadata about a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Human-readable name.
    pub name: String,
    /// Description.
    pub description: Option<String>,
    /// Version string.
    pub version: String,
    /// Data type (e.g., "images", "text", "tabular").
    pub data_type: String,
    /// Number of samples.
    pub sample_count: usize,
    /// Number of features per sample.
    pub feature_count: Option<usize>,
    /// Number of classes (for classification).
    pub num_classes: Option<usize>,
    /// Total size in bytes.
    pub size_bytes: u64,
    /// Creation timestamp.
    pub created_at: u64,
    /// Last modified timestamp.
    pub modified_at: u64,
    /// Custom tags.
    pub tags: Vec<String>,
    /// Custom key-value metadata.
    pub extra: HashMap<String, String>,
}

impl DatasetMetadata {
    /// Creates basic metadata.
    pub fn new(name: &str, sample_count: usize) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            name: name.to_string(),
            description: None,
            version: "1.0".to_string(),
            data_type: "unknown".to_string(),
            sample_count,
            feature_count: None,
            num_classes: None,
            size_bytes: 0,
            created_at: now,
            modified_at: now,
            tags: Vec::new(),
            extra: HashMap::new(),
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Sets the data type.
    pub fn with_data_type(mut self, dtype: &str) -> Self {
        self.data_type = dtype.to_string();
        self
    }

    /// Sets the feature count.
    pub fn with_features(mut self, count: usize) -> Self {
        self.feature_count = Some(count);
        self
    }

    /// Sets the number of classes.
    pub fn with_classes(mut self, count: usize) -> Self {
        self.num_classes = Some(count);
        self
    }

    /// Adds a tag.
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }
}

/// Commitment to a dataset including Merkle root and shuffle info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetCommitment {
    /// Merkle root of the dataset.
    pub merkle_root: Hash,
    /// Shuffle seed commitment (if shuffled).
    pub shuffle_commitment: Option<Hash>,
    /// Shuffle schedule (if epoch-varying shuffle).
    pub shuffle_schedule: Option<ShuffleSchedule>,
    /// Height of the Merkle tree.
    pub tree_height: usize,
    /// Commitment timestamp.
    pub committed_at: u64,
    /// Commitment signature (optional, from data provider).
    pub signature: Option<Vec<u8>>,
}

impl DatasetCommitment {
    /// Creates a new commitment from a Merkle root.
    pub fn new(merkle_root: Hash, tree_height: usize) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            merkle_root,
            shuffle_commitment: None,
            shuffle_schedule: None,
            tree_height,
            committed_at: now,
            signature: None,
        }
    }

    /// Adds a shuffle commitment.
    pub fn with_shuffle(mut self, seed_commitment: Hash) -> Self {
        self.shuffle_commitment = Some(seed_commitment);
        self
    }

    /// Adds a shuffle schedule.
    pub fn with_shuffle_schedule(mut self, schedule: ShuffleSchedule) -> Self {
        self.shuffle_schedule = Some(schedule);
        self
    }

    /// Computes a combined commitment hash.
    pub fn combined_hash(&self) -> Hash {
        let hasher = Sha256Hasher;
        let mut data = self.merkle_root.0.to_vec();

        if let Some(shuffle) = &self.shuffle_commitment {
            data.extend_from_slice(&shuffle.0);
        }

        data.extend_from_slice(&(self.tree_height as u64).to_le_bytes());
        data.extend_from_slice(&self.committed_at.to_le_bytes());

        hasher.hash_leaf(&data)
    }
}

/// Attestation from a data provider or verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetAttestation {
    /// Attestation ID.
    pub id: String,
    /// Attester identifier (address, name, etc.).
    pub attester: String,
    /// Attester type.
    pub attester_type: AttesterType,
    /// What is being attested.
    pub attestation_type: AttestationType,
    /// The commitment being attested.
    pub commitment_hash: Hash,
    /// Attestation message.
    pub message: Option<String>,
    /// Attestation signature.
    pub signature: Option<Vec<u8>>,
    /// Timestamp.
    pub timestamp: u64,
    /// Validity period (optional).
    pub valid_until: Option<u64>,
}

/// Type of attester.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttesterType {
    /// Original data provider.
    DataProvider,
    /// Third-party verifier.
    Verifier,
    /// Auditor.
    Auditor,
    /// Governance committee.
    Governance,
    /// Automated system.
    System,
}

/// Type of attestation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationType {
    /// Attests that the data exists and matches commitment.
    Existence,
    /// Attests to the data quality.
    Quality,
    /// Attests to the data source/provenance.
    Provenance,
    /// Attests compliance with regulations.
    Compliance,
    /// Attests that data is suitable for training.
    TrainingSuitability,
    /// General validity attestation.
    Validity,
}

/// Complete dataset entry in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetEntry {
    /// Unique identifier.
    pub id: DatasetId,
    /// Dataset metadata.
    pub metadata: DatasetMetadata,
    /// Dataset commitment.
    pub commitment: DatasetCommitment,
    /// Current status.
    pub status: DatasetStatus,
    /// Data origin/provenance.
    pub origin: Option<DataOrigin>,
    /// Attestations.
    pub attestations: Vec<DatasetAttestation>,
    /// Parent dataset (if derived).
    pub parent_id: Option<DatasetId>,
    /// Child datasets (derived from this one).
    pub child_ids: Vec<DatasetId>,
    /// Registration timestamp.
    pub registered_at: u64,
    /// Last update timestamp.
    pub updated_at: u64,
}

impl DatasetEntry {
    /// Creates a new dataset entry.
    pub fn new(id: DatasetId, metadata: DatasetMetadata, commitment: DatasetCommitment) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            id,
            metadata,
            commitment,
            status: DatasetStatus::Pending,
            origin: None,
            attestations: Vec::new(),
            parent_id: None,
            child_ids: Vec::new(),
            registered_at: now,
            updated_at: now,
        }
    }

    /// Sets the status.
    pub fn with_status(mut self, status: DatasetStatus) -> Self {
        self.status = status;
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self
    }

    /// Sets the origin.
    pub fn with_origin(mut self, origin: DataOrigin) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Sets the parent.
    pub fn with_parent(mut self, parent_id: DatasetId) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    /// Adds an attestation.
    pub fn add_attestation(&mut self, attestation: DatasetAttestation) {
        self.attestations.push(attestation);
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }

    /// Gets the number of attestations.
    pub fn attestation_count(&self) -> usize {
        self.attestations.len()
    }

    /// Checks if the dataset has a specific attestation type.
    pub fn has_attestation(&self, atype: AttestationType) -> bool {
        self.attestations.iter().any(|a| a.attestation_type == atype)
    }

    /// Gets the combined commitment hash.
    pub fn commitment_hash(&self) -> Hash {
        self.commitment.combined_hash()
    }
}

/// Configuration for the dataset registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Maximum number of datasets.
    pub max_datasets: usize,
    /// Require attestation for activation.
    pub require_attestation: bool,
    /// Minimum attestations required.
    pub min_attestations: usize,
    /// Allow dataset deprecation.
    pub allow_deprecation: bool,
    /// Enable automatic verification.
    pub auto_verify: bool,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_datasets: 10000,
            require_attestation: false,
            min_attestations: 1,
            allow_deprecation: true,
            auto_verify: false,
        }
    }
}

/// Error types for registry operations.
#[derive(Debug, Clone)]
pub enum RegistryError {
    /// Dataset not found.
    NotFound(DatasetId),
    /// Dataset already exists.
    AlreadyExists(DatasetId),
    /// Registry is full.
    RegistryFull,
    /// Invalid commitment.
    InvalidCommitment(String),
    /// Attestation required.
    AttestationRequired,
    /// Invalid attestation.
    InvalidAttestation(String),
    /// Operation not allowed.
    NotAllowed(String),
    /// Verification failed.
    VerificationFailed(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::NotFound(id) => write!(f, "Dataset not found: {}", id),
            RegistryError::AlreadyExists(id) => write!(f, "Dataset already exists: {}", id),
            RegistryError::RegistryFull => write!(f, "Registry is full"),
            RegistryError::InvalidCommitment(msg) => write!(f, "Invalid commitment: {}", msg),
            RegistryError::AttestationRequired => write!(f, "Attestation required"),
            RegistryError::InvalidAttestation(msg) => write!(f, "Invalid attestation: {}", msg),
            RegistryError::NotAllowed(msg) => write!(f, "Operation not allowed: {}", msg),
            RegistryError::VerificationFailed(msg) => write!(f, "Verification failed: {}", msg),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Result type for registry operations.
pub type RegistryResult<T> = Result<T, RegistryError>;

/// Dataset commitment registry.
pub struct DatasetCommitmentRegistry {
    /// Configuration.
    config: RegistryConfig,
    /// Registered datasets.
    datasets: HashMap<DatasetId, DatasetEntry>,
    /// Index by Merkle root for fast lookup.
    by_root: HashMap<Hash, DatasetId>,
    /// Index by status.
    by_status: HashMap<DatasetStatus, Vec<DatasetId>>,
}

impl DatasetCommitmentRegistry {
    /// Creates a new registry.
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            config,
            datasets: HashMap::new(),
            by_root: HashMap::new(),
            by_status: HashMap::new(),
        }
    }

    /// Creates with default configuration.
    pub fn default_registry() -> Self {
        Self::new(RegistryConfig::default())
    }

    /// Registers a new dataset.
    pub fn register(&mut self, entry: DatasetEntry) -> RegistryResult<()> {
        // Check capacity
        if self.datasets.len() >= self.config.max_datasets {
            return Err(RegistryError::RegistryFull);
        }

        // Check for duplicate ID
        if self.datasets.contains_key(&entry.id) {
            return Err(RegistryError::AlreadyExists(entry.id.clone()));
        }

        // Check for duplicate root
        if self.by_root.contains_key(&entry.commitment.merkle_root) {
            return Err(RegistryError::InvalidCommitment(
                "Merkle root already registered".to_string(),
            ));
        }

        // Add to indices
        self.by_root.insert(entry.commitment.merkle_root, entry.id.clone());
        self.by_status
            .entry(entry.status)
            .or_insert_with(Vec::new)
            .push(entry.id.clone());

        // Add entry
        self.datasets.insert(entry.id.clone(), entry);

        Ok(())
    }

    /// Gets a dataset by ID.
    pub fn get(&self, id: &DatasetId) -> RegistryResult<&DatasetEntry> {
        self.datasets.get(id).ok_or_else(|| RegistryError::NotFound(id.clone()))
    }

    /// Gets a mutable dataset by ID.
    pub fn get_mut(&mut self, id: &DatasetId) -> RegistryResult<&mut DatasetEntry> {
        self.datasets
            .get_mut(id)
            .ok_or_else(|| RegistryError::NotFound(id.clone()))
    }

    /// Finds a dataset by Merkle root.
    pub fn find_by_root(&self, root: &Hash) -> Option<&DatasetEntry> {
        self.by_root.get(root).and_then(|id| self.datasets.get(id))
    }

    /// Lists datasets by status.
    pub fn list_by_status(&self, status: DatasetStatus) -> Vec<&DatasetEntry> {
        self.by_status
            .get(&status)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.datasets.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Lists all datasets.
    pub fn list_all(&self) -> Vec<&DatasetEntry> {
        self.datasets.values().collect()
    }

    /// Updates dataset status.
    pub fn update_status(&mut self, id: &DatasetId, new_status: DatasetStatus) -> RegistryResult<()> {
        let entry = self.datasets.get_mut(id).ok_or_else(|| RegistryError::NotFound(id.clone()))?;

        let old_status = entry.status;

        // Remove from old status index
        if let Some(ids) = self.by_status.get_mut(&old_status) {
            ids.retain(|i| i != id);
        }

        // Update status
        entry.status = new_status;
        entry.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Add to new status index
        self.by_status
            .entry(new_status)
            .or_insert_with(Vec::new)
            .push(id.clone());

        Ok(())
    }

    /// Activates a dataset (requires attestation if configured).
    pub fn activate(&mut self, id: &DatasetId) -> RegistryResult<()> {
        let entry = self.get(id)?;

        // Check attestation requirement
        if self.config.require_attestation && entry.attestations.len() < self.config.min_attestations {
            return Err(RegistryError::AttestationRequired);
        }

        self.update_status(id, DatasetStatus::Active)
    }

    /// Deprecates a dataset.
    pub fn deprecate(&mut self, id: &DatasetId) -> RegistryResult<()> {
        if !self.config.allow_deprecation {
            return Err(RegistryError::NotAllowed("Deprecation not allowed".to_string()));
        }

        self.update_status(id, DatasetStatus::Deprecated)
    }

    /// Adds an attestation to a dataset.
    pub fn add_attestation(&mut self, id: &DatasetId, attestation: DatasetAttestation) -> RegistryResult<()> {
        // Read config values before mutable borrow
        let require_attestation = self.config.require_attestation;
        let min_attestations = self.config.min_attestations;

        let entry = self.get_mut(id)?;

        // Verify attestation matches commitment
        if attestation.commitment_hash != entry.commitment.combined_hash() {
            return Err(RegistryError::InvalidAttestation(
                "Commitment hash mismatch".to_string(),
            ));
        }

        entry.add_attestation(attestation);

        // Auto-activate if requirements met
        if require_attestation
            && entry.status == DatasetStatus::Pending
            && entry.attestations.len() >= min_attestations
        {
            entry.status = DatasetStatus::Active;
        }

        Ok(())
    }

    /// Verifies a dataset's commitment against provided data.
    pub fn verify_commitment<H: MerkleHasher>(
        &self,
        id: &DatasetId,
        merkle_root: &Hash,
        hasher: &H,
    ) -> RegistryResult<bool> {
        let entry = self.get(id)?;

        if entry.commitment.merkle_root != *merkle_root {
            return Err(RegistryError::VerificationFailed(
                "Merkle root mismatch".to_string(),
            ));
        }

        Ok(true)
    }

    /// Verifies a shuffle seed against the dataset's commitment.
    pub fn verify_shuffle(
        &self,
        id: &DatasetId,
        seed: &ShuffleSeed,
        epoch: usize,
    ) -> RegistryResult<bool> {
        let entry = self.get(id)?;

        if let Some(ref schedule) = entry.commitment.shuffle_schedule {
            Ok(schedule.verify_epoch_seed(seed, epoch))
        } else if let Some(ref commitment) = entry.commitment.shuffle_commitment {
            let hasher = Sha256Hasher;
            let computed = hasher.hash_leaf(seed.as_bytes());
            Ok(computed == *commitment)
        } else {
            Err(RegistryError::InvalidCommitment(
                "No shuffle commitment".to_string(),
            ))
        }
    }

    /// Links a child dataset to its parent.
    pub fn link_child(&mut self, parent_id: &DatasetId, child_id: &DatasetId) -> RegistryResult<()> {
        // Verify both exist
        self.get(parent_id)?;
        self.get(child_id)?;

        // Update parent
        let parent = self.get_mut(parent_id)?;
        if !parent.child_ids.contains(child_id) {
            parent.child_ids.push(child_id.clone());
        }

        Ok(())
    }

    /// Gets registry statistics.
    pub fn stats(&self) -> RegistryStats {
        let mut stats = RegistryStats {
            total_datasets: self.datasets.len(),
            ..Default::default()
        };

        for entry in self.datasets.values() {
            match entry.status {
                DatasetStatus::Active => stats.active_datasets += 1,
                DatasetStatus::Pending => stats.pending_datasets += 1,
                DatasetStatus::Deprecated => stats.deprecated_datasets += 1,
                DatasetStatus::Invalid => stats.invalid_datasets += 1,
                DatasetStatus::Processing => {}
            }

            stats.total_samples += entry.metadata.sample_count;
            stats.total_attestations += entry.attestations.len();
        }

        stats
    }

    /// Returns the number of datasets.
    pub fn len(&self) -> usize {
        self.datasets.len()
    }

    /// Returns true if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }

    /// Saves the registry to a JSON file.
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let snapshot = DatasetRegistrySnapshot {
            entries: self.datasets.values().cloned().collect(),
            config: self.config.clone(),
        };
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Loads the registry from a JSON file, rebuilding indices.
    pub fn load_from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let snapshot: DatasetRegistrySnapshot = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut registry = Self::new(snapshot.config);
        for entry in snapshot.entries {
            // Direct insert bypassing validation (data was already validated on first register)
            let id = entry.id.clone();
            let root = entry.commitment.merkle_root;
            let status = entry.status;
            registry.by_root.insert(root, id.clone());
            registry.by_status.entry(status).or_default().push(id.clone());
            registry.datasets.insert(id, entry);
        }
        Ok(registry)
    }
}

/// Serializable snapshot of a DatasetCommitmentRegistry for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetRegistrySnapshot {
    entries: Vec<DatasetEntry>,
    config: RegistryConfig,
}

/// Registry statistics.
#[derive(Debug, Clone, Default)]
pub struct RegistryStats {
    /// Total datasets.
    pub total_datasets: usize,
    /// Active datasets.
    pub active_datasets: usize,
    /// Pending datasets.
    pub pending_datasets: usize,
    /// Deprecated datasets.
    pub deprecated_datasets: usize,
    /// Invalid datasets.
    pub invalid_datasets: usize,
    /// Total samples across all datasets.
    pub total_samples: usize,
    /// Total attestations.
    pub total_attestations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_id() {
        let id = DatasetId::new("test-dataset");
        assert_eq!(id.0, "test-dataset");

        let root = Hash::from_slice(b"test root");
        let id_from_root = DatasetId::from_root(&root);
        assert!(id_from_root.0.starts_with("ds_"));
    }

    #[test]
    fn test_dataset_metadata() {
        let meta = DatasetMetadata::new("MNIST", 60000)
            .with_description("Handwritten digits")
            .with_data_type("images")
            .with_features(784)
            .with_classes(10)
            .with_tag("classification");

        assert_eq!(meta.name, "MNIST");
        assert_eq!(meta.sample_count, 60000);
        assert_eq!(meta.feature_count, Some(784));
        assert_eq!(meta.num_classes, Some(10));
    }

    #[test]
    fn test_dataset_commitment() {
        let root = Hash::from_slice(b"merkle root");
        let commitment = DatasetCommitment::new(root, 20);

        assert_eq!(commitment.merkle_root, root);
        assert_eq!(commitment.tree_height, 20);

        let combined = commitment.combined_hash();
        assert!(!combined.is_zero());
    }

    #[test]
    fn test_registry_register() {
        let mut registry = DatasetCommitmentRegistry::default_registry();

        let root = Hash::from_slice(b"test root");
        let metadata = DatasetMetadata::new("test", 1000);
        let commitment = DatasetCommitment::new(root, 10);
        let entry = DatasetEntry::new(DatasetId::new("test-1"), metadata, commitment);

        assert!(registry.register(entry).is_ok());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_duplicate() {
        let mut registry = DatasetCommitmentRegistry::default_registry();

        let root = Hash::from_slice(b"test root");
        let metadata = DatasetMetadata::new("test", 1000);
        let commitment = DatasetCommitment::new(root, 10);
        let entry = DatasetEntry::new(DatasetId::new("test-1"), metadata, commitment);

        assert!(registry.register(entry.clone()).is_ok());

        // Same ID should fail
        assert!(matches!(
            registry.register(entry),
            Err(RegistryError::AlreadyExists(_))
        ));
    }

    #[test]
    fn test_find_by_root() {
        let mut registry = DatasetCommitmentRegistry::default_registry();

        let root = Hash::from_slice(b"unique root");
        let metadata = DatasetMetadata::new("findme", 500);
        let commitment = DatasetCommitment::new(root, 10);
        let entry = DatasetEntry::new(DatasetId::new("findme-1"), metadata, commitment);

        registry.register(entry).unwrap();

        let found = registry.find_by_root(&root);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id.0, "findme-1");
    }

    #[test]
    fn test_status_update() {
        let mut registry = DatasetCommitmentRegistry::default_registry();

        let root = Hash::from_slice(b"status root");
        let metadata = DatasetMetadata::new("status-test", 100);
        let commitment = DatasetCommitment::new(root, 5);
        let entry = DatasetEntry::new(DatasetId::new("status-1"), metadata, commitment);

        registry.register(entry).unwrap();

        let id = DatasetId::new("status-1");
        assert_eq!(registry.get(&id).unwrap().status, DatasetStatus::Pending);

        registry.update_status(&id, DatasetStatus::Active).unwrap();
        assert_eq!(registry.get(&id).unwrap().status, DatasetStatus::Active);

        let active = registry.list_by_status(DatasetStatus::Active);
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn test_attestation() {
        let mut registry = DatasetCommitmentRegistry::default_registry();

        let root = Hash::from_slice(b"attest root");
        let metadata = DatasetMetadata::new("attest-test", 100);
        let commitment = DatasetCommitment::new(root, 5);
        let id = DatasetId::new("attest-1");
        let entry = DatasetEntry::new(id.clone(), metadata, commitment);

        registry.register(entry).unwrap();

        let commitment_hash = registry.get(&id).unwrap().commitment_hash();

        let attestation = DatasetAttestation {
            id: "attest-001".to_string(),
            attester: "verifier-1".to_string(),
            attester_type: AttesterType::Verifier,
            attestation_type: AttestationType::Validity,
            commitment_hash,
            message: Some("Verified correct".to_string()),
            signature: None,
            timestamp: 12345,
            valid_until: None,
        };

        registry.add_attestation(&id, attestation).unwrap();

        let entry = registry.get(&id).unwrap();
        assert_eq!(entry.attestation_count(), 1);
        assert!(entry.has_attestation(AttestationType::Validity));
    }

    #[test]
    fn test_shuffle_verification() {
        let mut registry = DatasetCommitmentRegistry::default_registry();

        let root = Hash::from_slice(b"shuffle root");
        let base_seed = ShuffleSeed::from_u64(42);
        let schedule = ShuffleSchedule::new(&base_seed, 1000, 10);

        let metadata = DatasetMetadata::new("shuffle-test", 1000);
        let commitment = DatasetCommitment::new(root, 10)
            .with_shuffle_schedule(schedule);
        let id = DatasetId::new("shuffle-1");
        let entry = DatasetEntry::new(id.clone(), metadata, commitment);

        registry.register(entry).unwrap();

        // Verify correct seed
        let epoch_seed = ShuffleSeed::for_epoch(&base_seed, 5);
        assert!(registry.verify_shuffle(&id, &epoch_seed, 5).unwrap());

        // Verify wrong seed fails
        let wrong_seed = ShuffleSeed::from_u64(999);
        assert!(!registry.verify_shuffle(&id, &wrong_seed, 5).unwrap());
    }

    #[test]
    fn test_registry_stats() {
        let mut registry = DatasetCommitmentRegistry::default_registry();

        for i in 0..5 {
            let root = Hash::from_slice(&format!("root-{}", i).into_bytes());
            let metadata = DatasetMetadata::new(&format!("dataset-{}", i), (i + 1) * 100);
            let commitment = DatasetCommitment::new(root, 10);
            let entry = DatasetEntry::new(DatasetId::new(format!("ds-{}", i)), metadata, commitment)
                .with_status(if i % 2 == 0 {
                    DatasetStatus::Active
                } else {
                    DatasetStatus::Pending
                });

            registry.register(entry).unwrap();
        }

        let stats = registry.stats();
        assert_eq!(stats.total_datasets, 5);
        assert_eq!(stats.active_datasets, 3);
        assert_eq!(stats.pending_datasets, 2);
        assert_eq!(stats.total_samples, 100 + 200 + 300 + 400 + 500);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let mut registry = DatasetCommitmentRegistry::default_registry();

        let root1 = Hash::from_slice(b"root-1-data-hash-pad!");
        let root2 = Hash::from_slice(b"root-2-data-hash-pad!");

        let entry1 = DatasetEntry::new(
            DatasetId::new("ds-1"),
            DatasetMetadata::new("dataset-1", 100),
            DatasetCommitment::new(root1, 10),
        ).with_status(DatasetStatus::Active);

        let entry2 = DatasetEntry::new(
            DatasetId::new("ds-2"),
            DatasetMetadata::new("dataset-2", 200),
            DatasetCommitment::new(root2, 12),
        ).with_status(DatasetStatus::Pending);

        registry.register(entry1).unwrap();
        registry.register(entry2).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dataset_registry.json");

        registry.save_to_file(&path).unwrap();
        let loaded = DatasetCommitmentRegistry::load_from_file(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert!(loaded.get(&DatasetId::new("ds-1")).is_ok());
        assert!(loaded.get(&DatasetId::new("ds-2")).is_ok());
        assert_eq!(loaded.get(&DatasetId::new("ds-1")).unwrap().metadata.sample_count, 100);
        assert_eq!(loaded.list_by_status(DatasetStatus::Active).len(), 1);
        assert_eq!(loaded.list_by_status(DatasetStatus::Pending).len(), 1);
    }
}
