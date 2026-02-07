pub mod ethereum;
pub mod ipfs;
pub mod local;

/// Trait for storage backends.
pub trait StorageBackend {
    /// Error type for this backend.
    type Error: std::fmt::Display;

    /// Saves arbitrary state by key.
    fn save_state(&self, key: &str, data: &[u8]) -> Result<(), Self::Error>;

    /// Loads state by key. Returns None if not found.
    fn load_state(&self, key: &str) -> Result<Option<Vec<u8>>, Self::Error>;

    /// Deletes state by key.
    fn delete_state(&self, key: &str) -> Result<(), Self::Error>;
}
