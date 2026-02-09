//! Ethereum On-Chain State Storage.
//!
//! This module is intentionally empty. On-chain state (model commitments,
//! round results, stake info) is managed via [`crate::sc_client::SCClient`]
//! which provides typed bindings to the `HelixCoordinatorV2` smart contract.
//!
//! A separate `StorageBackend` implementation for Ethereum is not needed
//! because:
//! 1. Ethereum storage is inherently transactional (not key-value).
//! 2. All on-chain reads/writes are already handled by `SCClient`.
//! 3. Large data (checkpoints, model weights) should use IPFS, not on-chain
//!    storage (which is prohibitively expensive at ~20k gas per 32 bytes).
//!
//! If you need to persist data on-chain, use `SCClient` directly.
