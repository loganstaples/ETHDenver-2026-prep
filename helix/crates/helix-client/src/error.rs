//! Unified error type for the HELIX Client SDK.

use thiserror::Error;

use crate::rpc::client::RpcError;

/// Top-level error type for all HELIX SDK operations.
#[derive(Debug, Error)]
pub enum HelixError {
    /// Failed to connect to a HELIX node.
    #[error("connection error: {0}")]
    Connection(String),

    /// An RPC call to the node failed.
    #[error("RPC error: {0}")]
    Rpc(#[from] RpcError),

    /// A training operation failed.
    #[error("training error: {0}")]
    Training(String),

    /// An on-chain / contract interaction failed.
    #[error("chain error: {0}")]
    Chain(String),

    /// Model validation or lookup error.
    #[error("model error: {0}")]
    Model(String),

    /// Proof generation or verification failed.
    #[error("proof error: {0}")]
    Proof(String),

    /// Invalid configuration.
    #[error("config error: {0}")]
    Config(String),

    /// Serialization / deserialization failure.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Catch-all for unexpected errors.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl HelixError {
    /// Convenience constructor for connection errors.
    pub fn connection(msg: impl Into<String>) -> Self {
        Self::Connection(msg.into())
    }

    /// Convenience constructor for training errors.
    pub fn training(msg: impl Into<String>) -> Self {
        Self::Training(msg.into())
    }

    /// Convenience constructor for model errors.
    pub fn model(msg: impl Into<String>) -> Self {
        Self::Model(msg.into())
    }

    /// Convenience constructor for config errors.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
}
