//! Local File-Based Storage.
//!
//! Provides persistent storage for node state, checkpoints, and peer lists
//! using the local filesystem. Writes are atomic (write to temp, then rename).
//!
//! Features:
//! - Atomic writes with unique temp-file names (PID + random suffix)
//! - SHA-256 integrity checksums stored alongside each file
//! - Checkpoint garbage collection (keep last N per model)

use std::path::{Path, PathBuf};
use std::{fs, io};

use sha2::{Digest, Sha256};

use crate::network::messages::PeerInfo;
use super::StorageBackend;

/// Local file-based storage backend.
pub struct LocalStorage {
    /// Base data directory.
    data_dir: PathBuf,
    /// Maximum checkpoints to retain per model (0 = unlimited).
    max_checkpoints: usize,
}

/// Errors from local storage operations.
#[derive(Debug)]
pub enum LocalStorageError {
    /// IO error.
    Io(io::Error),
    /// Serialization error.
    Serialization(String),
    /// Integrity check failed.
    IntegrityError { path: String, expected: String, actual: String },
}

impl std::fmt::Display for LocalStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Storage IO error: {}", e),
            Self::Serialization(msg) => write!(f, "Storage serialization error: {}", msg),
            Self::IntegrityError { path, expected, actual } => {
                write!(f, "Integrity check failed for {}: expected {}, got {}", path, expected, actual)
            }
        }
    }
}

impl std::error::Error for LocalStorageError {}

impl From<io::Error> for LocalStorageError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl LocalStorage {
    /// Creates a new local storage at the given data directory.
    /// Creates the directory structure if it doesn't exist.
    pub fn new(data_dir: impl Into<PathBuf>) -> Result<Self, LocalStorageError> {
        let data_dir = data_dir.into();
        fs::create_dir_all(data_dir.join("state"))?;
        fs::create_dir_all(data_dir.join("checkpoints"))?;
        Ok(Self { data_dir, max_checkpoints: 10 })
    }

    /// Creates local storage with a custom checkpoint retention limit.
    pub fn with_max_checkpoints(data_dir: impl Into<PathBuf>, max_checkpoints: usize) -> Result<Self, LocalStorageError> {
        let data_dir = data_dir.into();
        fs::create_dir_all(data_dir.join("state"))?;
        fs::create_dir_all(data_dir.join("checkpoints"))?;
        Ok(Self { data_dir, max_checkpoints })
    }

    /// Returns the base data directory.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Generates a unique temp-file path to prevent collisions between
    /// concurrent processes and threads.
    fn unique_tmp_path(base: &Path) -> PathBuf {
        use rand::Rng;
        let pid = std::process::id();
        let rand_suffix: u32 = rand::thread_rng().gen();
        let tmp_name = format!(
            ".{}.{}.{}.tmp",
            base.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("data"),
            pid,
            rand_suffix,
        );
        base.with_file_name(tmp_name)
    }

    /// Computes SHA-256 checksum of data and returns hex string.
    fn compute_checksum(data: &[u8]) -> String {
        let hash = Sha256::digest(data);
        hex::encode(hash)
    }

    /// Returns the path where the integrity checksum is stored for a given file.
    fn checksum_path(file_path: &Path) -> PathBuf {
        file_path.with_extension("sha256")
    }

    /// Atomically writes data to a file (write to unique temp, then rename).
    /// Also writes a SHA-256 checksum sidecar file.
    fn atomic_write(path: &Path, data: &[u8]) -> Result<(), LocalStorageError> {
        let tmp_path = Self::unique_tmp_path(path);
        fs::write(&tmp_path, data)?;
        fs::rename(&tmp_path, path)?;

        // Write checksum sidecar
        let checksum = Self::compute_checksum(data);
        let checksum_path = Self::checksum_path(path);
        fs::write(&checksum_path, checksum.as_bytes())?;

        Ok(())
    }

    /// Verifies the integrity of a file against its checksum sidecar.
    /// Returns Ok(()) if valid, Err if mismatch or missing checksum.
    pub fn verify_integrity(&self, path: &Path) -> Result<(), LocalStorageError> {
        let data = fs::read(path)?;
        let checksum_path = Self::checksum_path(path);

        match fs::read_to_string(&checksum_path) {
            Ok(expected) => {
                let actual = Self::compute_checksum(&data);
                if expected.trim() == actual {
                    Ok(())
                } else {
                    Err(LocalStorageError::IntegrityError {
                        path: path.display().to_string(),
                        expected: expected.trim().to_string(),
                        actual,
                    })
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // No checksum file — cannot verify
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Saves a checkpoint for a specific round.
    pub fn save_checkpoint(&self, round: u64, data: &[u8]) -> Result<(), LocalStorageError> {
        let path = self.data_dir.join("checkpoints").join(format!("round_{}", round));
        Self::atomic_write(&path, data)?;

        // Run garbage collection
        if self.max_checkpoints > 0 {
            self.gc_checkpoints()?;
        }

        Ok(())
    }

    /// Loads a checkpoint for a specific round, verifying integrity.
    pub fn load_checkpoint(&self, round: u64) -> Result<Option<Vec<u8>>, LocalStorageError> {
        let path = self.data_dir.join("checkpoints").join(format!("round_{}", round));
        match fs::read(&path) {
            Ok(data) => {
                // Verify checksum if available
                self.verify_integrity(&path)?;
                Ok(Some(data))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Garbage-collects old checkpoints, keeping only the most recent
    /// `max_checkpoints` files.
    fn gc_checkpoints(&self) -> Result<(), LocalStorageError> {
        let checkpoint_dir = self.data_dir.join("checkpoints");

        // List all checkpoint files (pattern: round_<N>)
        let mut entries: Vec<(u64, PathBuf)> = Vec::new();
        for entry in fs::read_dir(&checkpoint_dir)? {
            let entry = entry?;
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if let Some(round_str) = name.strip_prefix("round_") {
                // Skip checksum sidecar files
                if round_str.contains('.') {
                    continue;
                }
                if let Ok(round) = round_str.parse::<u64>() {
                    entries.push((round, entry.path()));
                }
            }
        }

        if entries.len() <= self.max_checkpoints {
            return Ok(());
        }

        // Sort by round number (ascending), remove oldest
        entries.sort_by_key(|(round, _)| *round);
        let to_remove = entries.len() - self.max_checkpoints;

        for (round, path) in entries.iter().take(to_remove) {
            tracing::info!("GC: removing old checkpoint round_{}", round);
            let _ = fs::remove_file(path);
            let _ = fs::remove_file(Self::checksum_path(path));
        }

        Ok(())
    }

    /// Lists all available checkpoint round numbers (sorted ascending).
    pub fn list_checkpoints(&self) -> Result<Vec<u64>, LocalStorageError> {
        let checkpoint_dir = self.data_dir.join("checkpoints");
        let mut rounds = Vec::new();

        for entry in fs::read_dir(&checkpoint_dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(round_str) = name.strip_prefix("round_") {
                if !round_str.contains('.') {
                    if let Ok(round) = round_str.parse::<u64>() {
                        rounds.push(round);
                    }
                }
            }
        }

        rounds.sort();
        Ok(rounds)
    }

    /// Saves the peer list for restart recovery.
    pub fn save_peers(&self, peers: &[PeerInfo]) -> Result<(), LocalStorageError> {
        let data = serde_json::to_vec(peers)
            .map_err(|e| LocalStorageError::Serialization(e.to_string()))?;
        let path = self.data_dir.join("state").join("peers.json");
        Self::atomic_write(&path, &data)
    }

    /// Loads the saved peer list.
    pub fn load_peers(&self) -> Result<Option<Vec<PeerInfo>>, LocalStorageError> {
        let path = self.data_dir.join("state").join("peers.json");
        match fs::read(&path) {
            Ok(data) => {
                let peers: Vec<PeerInfo> = serde_json::from_slice(&data)
                    .map_err(|e| LocalStorageError::Serialization(e.to_string()))?;
                Ok(Some(peers))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

impl StorageBackend for LocalStorage {
    type Error = LocalStorageError;

    fn save_state(&self, key: &str, data: &[u8]) -> Result<(), Self::Error> {
        let path = self.data_dir.join("state").join(key);
        Self::atomic_write(&path, data)
    }

    fn load_state(&self, key: &str) -> Result<Option<Vec<u8>>, Self::Error> {
        let path = self.data_dir.join("state").join(key);
        match fs::read(&path) {
            Ok(data) => {
                self.verify_integrity(&path)?;
                Ok(Some(data))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn delete_state(&self, key: &str) -> Result<(), Self::Error> {
        let path = self.data_dir.join("state").join(key);
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        // Also remove checksum sidecar
        let _ = fs::remove_file(Self::checksum_path(&path));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::messages::{NodeCapabilities, PeerId};

    #[test]
    fn test_save_load_state_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();

        let data = b"hello world";
        storage.save_state("test_key", data).unwrap();

        let loaded = storage.load_state("test_key").unwrap();
        assert_eq!(loaded.unwrap(), data);
    }

    #[test]
    fn test_load_missing_state_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();

        let loaded = storage.load_state("nonexistent").unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_delete_state() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();

        storage.save_state("to_delete", b"data").unwrap();
        assert!(storage.load_state("to_delete").unwrap().is_some());

        storage.delete_state("to_delete").unwrap();
        assert!(storage.load_state("to_delete").unwrap().is_none());
    }

    #[test]
    fn test_delete_nonexistent_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();
        storage.delete_state("nonexistent").unwrap();
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();

        let checkpoint_data = vec![1u8, 2, 3, 4, 5];
        storage.save_checkpoint(42, &checkpoint_data).unwrap();

        let loaded = storage.load_checkpoint(42).unwrap();
        assert_eq!(loaded.unwrap(), checkpoint_data);
    }

    #[test]
    fn test_load_missing_checkpoint_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();

        let loaded = storage.load_checkpoint(999).unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_peer_persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();

        let peers = vec![
            PeerInfo {
                id: PeerId::from_string("peer-1"),
                address: "127.0.0.1:9001".to_string(),
                capabilities: NodeCapabilities::default(),
                last_seen: 12345,
                reputation: 100,
            },
            PeerInfo {
                id: PeerId::from_string("peer-2"),
                address: "127.0.0.1:9002".to_string(),
                capabilities: NodeCapabilities::default(),
                last_seen: 12346,
                reputation: 90,
            },
        ];

        storage.save_peers(&peers).unwrap();
        let loaded = storage.load_peers().unwrap().unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id.0, "peer-1");
        assert_eq!(loaded[1].id.0, "peer-2");
        assert_eq!(loaded[0].address, "127.0.0.1:9001");
    }

    #[test]
    fn test_load_peers_when_none_saved() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();

        let loaded = storage.load_peers().unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_creates_directory_structure() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("nested").join("data");
        let storage = LocalStorage::new(&base).unwrap();

        assert!(base.join("state").exists());
        assert!(base.join("checkpoints").exists());
        assert_eq!(storage.data_dir(), base);
    }

    #[test]
    fn test_checksum_written_and_verified() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();

        let data = b"important data";
        storage.save_state("checksummed", data).unwrap();

        // Checksum file should exist
        let checksum_path = dir.path().join("state").join("checksummed.sha256");
        assert!(checksum_path.exists());

        // Verify succeeds
        let path = dir.path().join("state").join("checksummed");
        storage.verify_integrity(&path).unwrap();
    }

    #[test]
    fn test_checksum_detects_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();

        storage.save_state("corrupted", b"original").unwrap();

        // Corrupt the data file
        let path = dir.path().join("state").join("corrupted");
        fs::write(&path, b"tampered").unwrap();

        // Verify should fail
        let result = storage.verify_integrity(&path);
        assert!(matches!(result, Err(LocalStorageError::IntegrityError { .. })));
    }

    #[test]
    fn test_checkpoint_gc() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::with_max_checkpoints(dir.path(), 3).unwrap();

        // Save 5 checkpoints
        for i in 1..=5 {
            storage.save_checkpoint(i, &[i as u8; 10]).unwrap();
        }

        // Should only have 3 remaining (rounds 3, 4, 5)
        let rounds = storage.list_checkpoints().unwrap();
        assert_eq!(rounds.len(), 3);
        assert_eq!(rounds, vec![3, 4, 5]);

        // Oldest should be gone
        assert!(storage.load_checkpoint(1).unwrap().is_none());
        assert!(storage.load_checkpoint(2).unwrap().is_none());

        // Newest should be intact
        assert_eq!(storage.load_checkpoint(5).unwrap().unwrap(), vec![5u8; 10]);
    }

    #[test]
    fn test_unique_tmp_paths_are_different() {
        let path = PathBuf::from("/tmp/test_file");
        let t1 = LocalStorage::unique_tmp_path(&path);
        let t2 = LocalStorage::unique_tmp_path(&path);
        assert_ne!(t1, t2);
    }
}
