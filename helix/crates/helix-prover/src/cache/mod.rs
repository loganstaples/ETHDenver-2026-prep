//! Caching infrastructure for the HELIX prover.
//!
//! This module provides multiple caching layers:
//!
//! - **key_cache**: LRU cache for proving/verification keys
//! - **proof_cache**: Incremental proof cache for session-based proving
//! - **witness_cache**: Content-addressed cache for witness-based proof deduplication
//! - **mmap_storage**: Memory-mapped storage for large cache entries

pub mod key_cache;
pub mod mmap_storage;
pub mod proof_cache;
pub mod witness_cache;

// Re-export commonly used types
pub use key_cache::{KeyCache, KeyCacheConfig, KeyCacheStats};
pub use proof_cache::{
    CachedProof, ProofCache, ProofCacheConfig, ProofCacheStats, ProvingSession, SessionState,
    SharedProofCache, shared_cache, shared_cache_with_config,
};
pub use witness_cache::{
    EvictionStrategy, SharedWitnessCache, WitnessCache, WitnessCacheConfig, WitnessCacheStats,
    WitnessCachedProof, WitnessHash, WitnessHashBuilder, shared_witness_cache,
    shared_witness_cache_with_config,
};
