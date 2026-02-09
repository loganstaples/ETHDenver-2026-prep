//! Memory-Mapped Storage Backend.
//!
//! Provides memory-mapped file access for large cache entries (proving keys,
//! SRS parameters) to reduce heap allocation and enable OS-level page caching.
//!
//! This module is a placeholder for future mmap-based storage. Currently,
//! all cache entries are stored in-memory via the other cache modules.
//! When models exceed available RAM, mmap storage will allow the OS to
//! page in/out cache entries transparently.

use std::path::{Path, PathBuf};
use std::fs;
use std::io;

/// Configuration for memory-mapped storage.
#[derive(Debug, Clone)]
pub struct MmapStorageConfig {
    /// Base directory for mmap files.
    pub base_dir: PathBuf,
    /// Maximum total size of mmap files (bytes).
    pub max_total_bytes: u64,
}

impl Default for MmapStorageConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("/tmp/helix-mmap-cache"),
            max_total_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
        }
    }
}

/// Errors from mmap storage operations.
#[derive(Debug)]
pub enum MmapError {
    Io(io::Error),
    CapacityExceeded { requested: u64, available: u64 },
}

impl std::fmt::Display for MmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Mmap IO error: {}", e),
            Self::CapacityExceeded { requested, available } => {
                write!(f, "Mmap capacity exceeded: requested {} bytes, {} available", requested, available)
            }
        }
    }
}

impl std::error::Error for MmapError {}

impl From<io::Error> for MmapError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Memory-mapped storage backend for large cache entries.
///
/// Currently provides file-based read/write as a stepping stone to
/// full mmap support. The interface is designed to be swap-compatible
/// with a true mmap implementation.
pub struct MmapStorage {
    config: MmapStorageConfig,
    total_bytes: u64,
}

impl MmapStorage {
    /// Creates a new mmap storage backend.
    pub fn new(config: MmapStorageConfig) -> Result<Self, MmapError> {
        fs::create_dir_all(&config.base_dir)?;
        Ok(Self {
            config,
            total_bytes: 0,
        })
    }

    /// Stores data to a file-backed entry.
    pub fn store(&mut self, key: &str, data: &[u8]) -> Result<PathBuf, MmapError> {
        let size = data.len() as u64;
        if self.total_bytes + size > self.config.max_total_bytes {
            return Err(MmapError::CapacityExceeded {
                requested: size,
                available: self.config.max_total_bytes.saturating_sub(self.total_bytes),
            });
        }

        let path = self.config.base_dir.join(key);
        fs::write(&path, data)?;
        self.total_bytes += size;

        Ok(path)
    }

    /// Loads data from a file-backed entry.
    pub fn load(&self, key: &str) -> Result<Option<Vec<u8>>, MmapError> {
        let path = self.config.base_dir.join(key);
        match fs::read(&path) {
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Removes a file-backed entry.
    pub fn remove(&mut self, key: &str) -> Result<bool, MmapError> {
        let path = self.config.base_dir.join(key);
        match fs::metadata(&path) {
            Ok(meta) => {
                self.total_bytes = self.total_bytes.saturating_sub(meta.len());
                fs::remove_file(&path)?;
                Ok(true)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// Returns the total bytes currently stored.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Returns the base directory.
    pub fn base_dir(&self) -> &Path {
        &self.config.base_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let config = MmapStorageConfig {
            base_dir: dir.path().to_path_buf(),
            max_total_bytes: 1024,
        };
        let mut storage = MmapStorage::new(config).unwrap();

        storage.store("test_key", b"hello world").unwrap();
        let loaded = storage.load("test_key").unwrap().unwrap();
        assert_eq!(loaded, b"hello world");
    }

    #[test]
    fn test_load_missing() {
        let dir = tempfile::tempdir().unwrap();
        let config = MmapStorageConfig {
            base_dir: dir.path().to_path_buf(),
            max_total_bytes: 1024,
        };
        let storage = MmapStorage::new(config).unwrap();
        assert!(storage.load("nonexistent").unwrap().is_none());
    }

    #[test]
    fn test_capacity_exceeded() {
        let dir = tempfile::tempdir().unwrap();
        let config = MmapStorageConfig {
            base_dir: dir.path().to_path_buf(),
            max_total_bytes: 10,
        };
        let mut storage = MmapStorage::new(config).unwrap();
        let result = storage.store("big", &[0u8; 20]);
        assert!(matches!(result, Err(MmapError::CapacityExceeded { .. })));
    }
}
