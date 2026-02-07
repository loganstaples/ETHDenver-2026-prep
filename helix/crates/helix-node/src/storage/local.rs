//! Local File-Based Storage.
//!
//! Provides persistent storage for node state, checkpoints, and peer lists
//! using the local filesystem. Writes are atomic (write to .tmp then rename).

use std::path::{Path, PathBuf};
use std::{fs, io};

use crate::network::messages::PeerInfo;
use super::StorageBackend;

/// Local file-based storage backend.
pub struct LocalStorage {
    /// Base data directory.
    data_dir: PathBuf,
}

/// Errors from local storage operations.
#[derive(Debug)]
pub enum LocalStorageError {
    /// IO error.
    Io(io::Error),
    /// Serialization error.
    Serialization(String),
}

impl std::fmt::Display for LocalStorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Storage IO error: {}", e),
            Self::Serialization(msg) => write!(f, "Storage serialization error: {}", msg),
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
        Ok(Self { data_dir })
    }

    /// Returns the base data directory.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Atomically writes data to a file (write to .tmp, then rename).
    fn atomic_write(path: &Path, data: &[u8]) -> Result<(), LocalStorageError> {
        let tmp_path = path.with_extension("tmp");
        fs::write(&tmp_path, data)?;
        fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Saves a checkpoint for a specific round.
    pub fn save_checkpoint(&self, round: u64, data: &[u8]) -> Result<(), LocalStorageError> {
        let path = self.data_dir.join("checkpoints").join(format!("round_{}", round));
        Self::atomic_write(&path, data)
    }

    /// Loads a checkpoint for a specific round.
    pub fn load_checkpoint(&self, round: u64) -> Result<Option<Vec<u8>>, LocalStorageError> {
        let path = self.data_dir.join("checkpoints").join(format!("round_{}", round));
        match fs::read(&path) {
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
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
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn delete_state(&self, key: &str) -> Result<(), Self::Error> {
        let path = self.data_dir.join("state").join(key);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
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
}
