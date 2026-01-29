//! Verifier Node Role.
//!
//! Implements the verifier node role which validates proofs
//! and maintains network consensus.

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::network::messages::{NodeCapabilities, PeerId};

/// Verifier state.
#[derive(Debug, Clone, PartialEq)]
pub enum VerifierState {
    /// Ready to verify.
    Ready,
    /// Currently verifying a proof.
    Verifying { proof_id: String },
    /// Error state.
    Error { message: String },
}

/// Configuration for verifier node.
#[derive(Debug, Clone)]
pub struct VerifierConfig {
    /// Maximum concurrent verifications.
    pub max_concurrent: usize,
    /// Verification timeout (seconds).
    pub timeout_secs: u64,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            timeout_secs: 60,
        }
    }
}

/// Verifier node statistics.
#[derive(Debug, Clone, Default)]
pub struct VerifierStats {
    /// Total proofs verified.
    pub proofs_verified: u64,
    /// Proofs accepted.
    pub proofs_accepted: u64,
    /// Proofs rejected.
    pub proofs_rejected: u64,
    /// Average verification time (ms).
    pub avg_verify_time_ms: f64,
}

/// Verifier node role.
pub struct VerifierNode {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: VerifierConfig,
    /// Current state.
    state: Arc<RwLock<VerifierState>>,
    /// Statistics.
    stats: Arc<RwLock<VerifierStats>>,
}

impl VerifierNode {
    /// Creates a new verifier node.
    pub fn new(local_id: PeerId, config: VerifierConfig) -> Self {
        Self {
            local_id,
            config,
            state: Arc::new(RwLock::new(VerifierState::Ready)),
            stats: Arc::new(RwLock::new(VerifierStats::default())),
        }
    }

    /// Gets the node's capabilities.
    pub fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities {
            can_train: false,
            can_aggregate: false,
            can_prove: true,
            gpu_memory_mb: 0,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4),
            storage_gb: 100,
        }
    }

    /// Gets current state.
    pub async fn get_state(&self) -> VerifierState {
        self.state.read().await.clone()
    }

    /// Verifies a proof.
    pub async fn verify_proof(&self, proof_id: String, proof: &[u8]) -> bool {
        // Update state
        {
            let mut state = self.state.write().await;
            *state = VerifierState::Verifying { proof_id: proof_id.clone() };
        }

        // Simulate verification
        let valid = !proof.is_empty();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.proofs_verified += 1;
            if valid {
                stats.proofs_accepted += 1;
            } else {
                stats.proofs_rejected += 1;
            }
        }

        // Return to ready state
        {
            let mut state = self.state.write().await;
            *state = VerifierState::Ready;
        }

        valid
    }

    /// Gets statistics.
    pub async fn get_stats(&self) -> VerifierStats {
        self.stats.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_verifier_init() {
        let local_id = PeerId::random();
        let node = VerifierNode::new(local_id, VerifierConfig::default());
        
        assert!(matches!(node.get_state().await, VerifierState::Ready));
    }

    #[tokio::test]
    async fn test_verify_proof() {
        let local_id = PeerId::random();
        let node = VerifierNode::new(local_id, VerifierConfig::default());
        
        let result = node.verify_proof("proof1".to_string(), &[1, 2, 3]).await;
        assert!(result);

        let stats = node.get_stats().await;
        assert_eq!(stats.proofs_verified, 1);
        assert_eq!(stats.proofs_accepted, 1);
    }
}
