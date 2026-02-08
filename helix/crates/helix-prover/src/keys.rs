//! Key Management for ZK Proving.
//!
//! Manages proving and verification keys, including generation, storage,
//! serialization, and secure handling.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Identifier for a key set (circuit-specific).
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct KeyId(pub String);

impl KeyId {
    /// Creates a new key ID from circuit name and version.
    pub fn new(circuit_name: &str, version: u32) -> Self {
        Self(format!("{}_v{}", circuit_name, version))
    }

    /// Creates a key ID from a string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

impl std::fmt::Display for KeyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Metadata about a key set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetadata {
    /// Key identifier.
    pub id: KeyId,
    /// Circuit name.
    pub circuit_name: String,
    /// Circuit version.
    pub version: u32,
    /// Creation timestamp.
    pub created_at: u64,
    /// K parameter (circuit size).
    pub k: u32,
    /// Hash of the proving key.
    pub pk_hash: [u8; 32],
    /// Hash of the verification key.
    pub vk_hash: [u8; 32],
    /// Size of the proving key in bytes.
    pub pk_size: usize,
    /// Size of the verification key in bytes.
    pub vk_size: usize,
}

/// Result of key operations.
pub type KeyResult<T> = Result<T, KeyError>;

/// Errors in key management.
#[derive(Debug)]
pub enum KeyError {
    /// Key not found.
    NotFound(KeyId),
    /// IO error.
    IoError(std::io::Error),
    /// Serialization error.
    SerializationError(String),
    /// Key verification failed.
    VerificationFailed,
    /// Key already exists.
    AlreadyExists(KeyId),
}

impl std::fmt::Display for KeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyError::NotFound(id) => write!(f, "Key not found: {}", id),
            KeyError::IoError(e) => write!(f, "IO error: {}", e),
            KeyError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            KeyError::VerificationFailed => write!(f, "Key verification failed"),
            KeyError::AlreadyExists(id) => write!(f, "Key already exists: {}", id),
        }
    }
}

impl std::error::Error for KeyError {}

impl From<std::io::Error> for KeyError {
    fn from(e: std::io::Error) -> Self {
        KeyError::IoError(e)
    }
}

/// In-memory key storage.
#[derive(Default)]
pub struct InMemoryKeyStore {
    /// Proving keys.
    proving_keys: HashMap<KeyId, Vec<u8>>,
    /// Verification keys.
    verification_keys: HashMap<KeyId, Vec<u8>>,
    /// Key metadata.
    metadata: HashMap<KeyId, KeyMetadata>,
}

impl InMemoryKeyStore {
    /// Creates a new empty key store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores a key pair.
    pub fn store(
        &mut self,
        id: KeyId,
        pk: Vec<u8>,
        vk: Vec<u8>,
        metadata: KeyMetadata,
    ) -> KeyResult<()> {
        if self.proving_keys.contains_key(&id) {
            return Err(KeyError::AlreadyExists(id));
        }

        self.proving_keys.insert(id.clone(), pk);
        self.verification_keys.insert(id.clone(), vk);
        self.metadata.insert(id, metadata);

        Ok(())
    }

    /// Gets the proving key.
    pub fn get_pk(&self, id: &KeyId) -> KeyResult<&[u8]> {
        self.proving_keys
            .get(id)
            .map(|v| v.as_slice())
            .ok_or_else(|| KeyError::NotFound(id.clone()))
    }

    /// Gets the verification key.
    pub fn get_vk(&self, id: &KeyId) -> KeyResult<&[u8]> {
        self.verification_keys
            .get(id)
            .map(|v| v.as_slice())
            .ok_or_else(|| KeyError::NotFound(id.clone()))
    }

    /// Gets key metadata.
    pub fn get_metadata(&self, id: &KeyId) -> Option<&KeyMetadata> {
        self.metadata.get(id)
    }

    /// Lists all key IDs.
    pub fn list(&self) -> Vec<&KeyId> {
        self.metadata.keys().collect()
    }

    /// Removes a key pair.
    pub fn remove(&mut self, id: &KeyId) -> KeyResult<()> {
        self.proving_keys.remove(id);
        self.verification_keys.remove(id);
        self.metadata.remove(id);
        Ok(())
    }
}

/// File-based key storage.
pub struct FileKeyStore {
    /// Base directory for key storage.
    base_dir: PathBuf,
    /// In-memory cache.
    cache: InMemoryKeyStore,
    /// Metadata index.
    index: HashMap<KeyId, PathBuf>,
}

impl FileKeyStore {
    /// Creates a new file-based key store.
    pub fn new(base_dir: impl AsRef<Path>) -> KeyResult<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        fs::create_dir_all(&base_dir)?;

        let mut store = Self {
            base_dir,
            cache: InMemoryKeyStore::new(),
            index: HashMap::new(),
        };

        store.load_index()?;

        Ok(store)
    }

    /// Stores a key pair to disk.
    pub fn store(
        &mut self,
        id: KeyId,
        pk: Vec<u8>,
        vk: Vec<u8>,
        circuit_name: String,
        version: u32,
        k: u32,
    ) -> KeyResult<()> {
        let key_dir = self.base_dir.join(id.0.clone());
        fs::create_dir_all(&key_dir)?;

        // Compute hashes
        let pk_hash = Self::hash_bytes(&pk);
        let vk_hash = Self::hash_bytes(&vk);

        // Create metadata
        let metadata = KeyMetadata {
            id: id.clone(),
            circuit_name,
            version,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            k,
            pk_hash,
            vk_hash,
            pk_size: pk.len(),
            vk_size: vk.len(),
        };

        // Write files
        let pk_path = key_dir.join("proving_key.bin");
        let vk_path = key_dir.join("verification_key.bin");
        let meta_path = key_dir.join("metadata.json");

        fs::write(&pk_path, &pk)?;
        fs::write(&vk_path, &vk)?;
        fs::write(&meta_path, serde_json::to_string_pretty(&metadata)
            .map_err(|e| KeyError::SerializationError(e.to_string()))?)?;

        // Update cache and index
        self.cache.store(id.clone(), pk, vk, metadata)?;
        self.index.insert(id, key_dir);

        self.save_index()?;

        Ok(())
    }

    /// Loads a proving key from disk.
    pub fn load_pk(&mut self, id: &KeyId) -> KeyResult<Vec<u8>> {
        // Check cache first
        if let Ok(pk) = self.cache.get_pk(id) {
            return Ok(pk.to_vec());
        }

        // Load from disk
        let key_dir = self.index.get(id)
            .ok_or_else(|| KeyError::NotFound(id.clone()))?;
        
        let pk_path = key_dir.join("proving_key.bin");
        let pk = fs::read(&pk_path)?;

        Ok(pk)
    }

    /// Loads a verification key from disk.
    pub fn load_vk(&mut self, id: &KeyId) -> KeyResult<Vec<u8>> {
        // Check cache first
        if let Ok(vk) = self.cache.get_vk(id) {
            return Ok(vk.to_vec());
        }

        // Load from disk
        let key_dir = self.index.get(id)
            .ok_or_else(|| KeyError::NotFound(id.clone()))?;
        
        let vk_path = key_dir.join("verification_key.bin");
        let vk = fs::read(&vk_path)?;

        Ok(vk)
    }

    /// Loads key metadata.
    pub fn load_metadata(&self, id: &KeyId) -> KeyResult<KeyMetadata> {
        let key_dir = self.index.get(id)
            .ok_or_else(|| KeyError::NotFound(id.clone()))?;
        
        let meta_path = key_dir.join("metadata.json");
        let meta_str = fs::read_to_string(&meta_path)?;
        
        serde_json::from_str(&meta_str)
            .map_err(|e| KeyError::SerializationError(e.to_string()))
    }

    /// Lists all stored key IDs.
    pub fn list(&self) -> Vec<KeyId> {
        self.index.keys().cloned().collect()
    }

    /// Verifies key integrity.
    pub fn verify(&self, id: &KeyId) -> KeyResult<bool> {
        let key_dir = self.index.get(id)
            .ok_or_else(|| KeyError::NotFound(id.clone()))?;
        
        let pk_path = key_dir.join("proving_key.bin");
        let vk_path = key_dir.join("verification_key.bin");
        
        let pk = fs::read(&pk_path)?;
        let vk = fs::read(&vk_path)?;
        
        let metadata = self.load_metadata(id)?;
        
        let pk_hash = Self::hash_bytes(&pk);
        let vk_hash = Self::hash_bytes(&vk);
        
        Ok(pk_hash == metadata.pk_hash && vk_hash == metadata.vk_hash)
    }

    /// Deletes a key pair.
    pub fn delete(&mut self, id: &KeyId) -> KeyResult<()> {
        let key_dir = self.index.remove(id)
            .ok_or_else(|| KeyError::NotFound(id.clone()))?;
        
        fs::remove_dir_all(&key_dir)?;
        self.cache.remove(id)?;
        self.save_index()?;
        
        Ok(())
    }

    fn load_index(&mut self) -> KeyResult<()> {
        let index_path = self.base_dir.join("index.json");
        
        if index_path.exists() {
            let index_str = fs::read_to_string(&index_path)?;
            let index_data: HashMap<String, String> = serde_json::from_str(&index_str)
                .map_err(|e| KeyError::SerializationError(e.to_string()))?;
            
            for (id, path) in index_data {
                self.index.insert(KeyId(id), PathBuf::from(path));
            }
        }
        
        Ok(())
    }

    fn save_index(&self) -> KeyResult<()> {
        let index_path = self.base_dir.join("index.json");
        
        let index_data: HashMap<String, String> = self.index
            .iter()
            .map(|(k, v)| (k.0.clone(), v.to_string_lossy().to_string()))
            .collect();
        
        let index_str = serde_json::to_string_pretty(&index_data)
            .map_err(|e| KeyError::SerializationError(e.to_string()))?;
        
        fs::write(&index_path, index_str)?;
        
        Ok(())
    }

    fn hash_bytes(data: &[u8]) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        Sha256::digest(data).into()
    }
}

/// Generates a key set for a circuit (stub — returns deterministic placeholder bytes).
///
/// For real Halo2 key generation use [`generate_keys_for_circuit`].
pub fn generate_keys<C>(
    circuit_name: &str,
    version: u32,
    k: u32,
) -> (KeyId, Vec<u8>, Vec<u8>, KeyMetadata) {
    let id = KeyId::new(circuit_name, version);

    // Deterministic placeholder keys derived from circuit identity
    let pk = format!("HELIX_PK:{}:k={}", id.0, k).into_bytes();
    let vk = format!("HELIX_VK:{}:k={}", id.0, k).into_bytes();

    let pk_hash = FileKeyStore::hash_bytes(&pk);
    let vk_hash = FileKeyStore::hash_bytes(&vk);

    let metadata = KeyMetadata {
        id: id.clone(),
        circuit_name: circuit_name.to_string(),
        version,
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        k,
        pk_hash,
        vk_hash,
        pk_size: pk.len(),
        vk_size: vk.len(),
    };

    (id, pk, vk, metadata)
}

/// Result of real Halo2 key generation.
pub struct CircuitKeys {
    /// Key identifier.
    pub id: KeyId,
    /// SRS parameters (KZG commitment scheme on BN254).
    pub params: helix_circuits::halo2_proofs::poly::kzg::commitment::ParamsKZG<helix_circuits::halo2curves::bn256::Bn256>,
    /// Proving key.
    pub pk: helix_circuits::halo2_proofs::plonk::ProvingKey<helix_circuits::halo2curves::bn256::G1Affine>,
    /// Verification key.
    pub vk: helix_circuits::halo2_proofs::plonk::VerifyingKey<helix_circuits::halo2curves::bn256::G1Affine>,
}

/// Default deterministic seed for the HELIX SRS.
///
/// All provers MUST use this seed (or one explicitly agreed upon) so that they
/// share the same SRS and can verify each other's proofs. Using `OsRng` for
/// `ParamsKZG::setup()` produces a different SRS every time, which makes
/// cross-prover verification impossible.
pub const HELIX_SRS_SEED: [u8; 32] = *b"HELIX_DETERMINISTIC_SRS_SEED_v1!";

/// Generates real Halo2 proving/verification keys for a concrete circuit.
///
/// Uses a deterministic SRS derived from [`HELIX_SRS_SEED`] so that all
/// provers in the network generate compatible keys for the same circuit.
pub fn generate_keys_for_circuit<C: helix_circuits::halo2_proofs::plonk::Circuit<helix_circuits::halo2curves::bn256::Fr>>(
    circuit: &C,
    circuit_name: &str,
    version: u32,
    k: u32,
) -> CircuitKeys {
    generate_keys_for_circuit_with_seed(circuit, circuit_name, version, k, HELIX_SRS_SEED)
}

/// Generates real Halo2 proving/verification keys with a custom SRS seed.
///
/// Use this when you need a different SRS than the default (e.g., for testing
/// or for per-model SRS isolation).
pub fn generate_keys_for_circuit_with_seed<C: helix_circuits::halo2_proofs::plonk::Circuit<helix_circuits::halo2curves::bn256::Fr>>(
    circuit: &C,
    circuit_name: &str,
    version: u32,
    k: u32,
    srs_seed: [u8; 32],
) -> CircuitKeys {
    use helix_circuits::halo2_proofs::poly::kzg::commitment::ParamsKZG;
    use helix_circuits::halo2_proofs::plonk::{keygen_pk, keygen_vk};
    use helix_circuits::halo2curves::bn256::Bn256;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let id = KeyId::new(circuit_name, version);

    // Deterministic trusted setup (KZG) — same seed → same SRS
    let rng = StdRng::from_seed(srs_seed);
    let params = ParamsKZG::<Bn256>::setup(k, rng);

    // Generate verification key then proving key
    let vk = keygen_vk(&params, circuit).expect("keygen_vk failed");
    let pk = keygen_pk(&params, vk.clone(), circuit).expect("keygen_pk failed");

    CircuitKeys { id, params, pk, vk }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_store() {
        let mut store = InMemoryKeyStore::new();
        
        let id = KeyId::new("test_circuit", 1);
        let pk = vec![1, 2, 3, 4];
        let vk = vec![5, 6, 7, 8];
        
        let metadata = KeyMetadata {
            id: id.clone(),
            circuit_name: "test_circuit".to_string(),
            version: 1,
            created_at: 0,
            k: 10,
            pk_hash: [0; 32],
            vk_hash: [0; 32],
            pk_size: 4,
            vk_size: 4,
        };
        
        store.store(id.clone(), pk, vk, metadata).unwrap();
        
        assert_eq!(store.get_pk(&id).unwrap(), &[1, 2, 3, 4]);
        assert_eq!(store.get_vk(&id).unwrap(), &[5, 6, 7, 8]);
    }

    #[test]
    fn test_key_id() {
        let id = KeyId::new("my_circuit", 3);
        assert_eq!(id.0, "my_circuit_v3");
    }

    #[test]
    fn test_generate_keys() {
        let (id, pk, vk, metadata) = generate_keys::<()>("test", 1, 10);
        
        assert!(pk.len() > 0);
        assert!(vk.len() > 0);
        assert_eq!(metadata.k, 10);
    }
}
