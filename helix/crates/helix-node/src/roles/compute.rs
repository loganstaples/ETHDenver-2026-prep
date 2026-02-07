//! Compute Node Role.
//!
//! Implements the compute node role which performs local training
//! and generates proofs of correct computation.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

use crate::network::messages::{
    GradientMessage, MessagePayload, NetworkMessage, NodeCapabilities, PeerId, TrainingMessage,
    TrainingParams,
};

/// Compute node state.
#[derive(Debug, Clone, PartialEq)]
pub enum ComputeState {
    /// Idle, waiting for work.
    Idle,
    /// Registered for a training round.
    Registered { round_id: u64, shard_id: u32 },
    /// Currently training.
    Training { round_id: u64, progress: f32 },
    /// Training complete, generating proof.
    Proving { round_id: u64 },
    /// Ready to submit.
    Ready { round_id: u64 },
    /// Error state.
    Error { message: String },
}

/// Configuration for compute node.
#[derive(Debug, Clone)]
pub struct ComputeConfig {
    /// Maximum concurrent training tasks.
    pub max_concurrent_tasks: usize,
    /// GPU memory limit (MB).
    pub gpu_memory_limit_mb: u32,
    /// Enable proof generation.
    pub generate_proofs: bool,
    /// Local epochs before submitting gradient.
    pub local_epochs: u32,
    /// Batch size.
    pub batch_size: u32,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 1,
            gpu_memory_limit_mb: 8192,
            generate_proofs: true,
            local_epochs: 5,
            batch_size: 32,
        }
    }
}

/// Training result from a compute node.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Round ID.
    pub round_id: u64,
    /// Gradient commitment.
    pub gradient_commitment: [u8; 32],
    /// Serialized gradients.
    pub gradients: Vec<u8>,
    /// Error bound for this computation.
    pub error_bound: f64,
    /// Proof of correct computation.
    pub proof: Vec<u8>,
    /// Computation time (ms).
    pub compute_time_ms: u64,
}

/// Compute node role.
pub struct ComputeNode {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: ComputeConfig,
    /// Current state.
    state: Arc<RwLock<ComputeState>>,
    /// Current training parameters.
    current_params: Arc<RwLock<Option<TrainingParams>>>,
    /// Completed results.
    results: Arc<RwLock<HashMap<u64, TrainingResult>>>,
    /// Statistics.
    stats: Arc<RwLock<ComputeStats>>,
}

/// Compute node statistics.
#[derive(Debug, Clone, Default)]
pub struct ComputeStats {
    /// Total rounds participated.
    pub rounds_participated: u64,
    /// Successful rounds.
    pub rounds_successful: u64,
    /// Total compute time (ms).
    pub total_compute_time_ms: u64,
    /// Total gradients computed.
    pub gradients_computed: u64,
    /// Total proofs generated.
    pub proofs_generated: u64,
}

impl ComputeNode {
    /// Creates a new compute node.
    pub fn new(local_id: PeerId, config: ComputeConfig) -> Self {
        Self {
            local_id,
            config,
            state: Arc::new(RwLock::new(ComputeState::Idle)),
            current_params: Arc::new(RwLock::new(None)),
            results: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ComputeStats::default())),
        }
    }

    /// Gets the node's capabilities.
    pub fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities {
            can_train: true,
            can_aggregate: false,
            can_prove: self.config.generate_proofs,
            gpu_memory_mb: self.config.gpu_memory_limit_mb,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4),
            storage_gb: 100,
        }
    }

    /// Gets current state.
    pub async fn get_state(&self) -> ComputeState {
        self.state.read().await.clone()
    }

    /// Registers for a training round.
    pub async fn register_for_round(&self, round_id: u64, shard_id: u32, params: TrainingParams) {
        let mut state = self.state.write().await;
        *state = ComputeState::Registered { round_id, shard_id };

        let mut current = self.current_params.write().await;
        *current = Some(params);

        let mut stats = self.stats.write().await;
        stats.rounds_participated += 1;
    }

    /// Starts training.
    pub async fn start_training(&self, round_id: u64) -> Result<(), String> {
        let current_state = self.get_state().await;
        
        match current_state {
            ComputeState::Registered { round_id: r, .. } if r == round_id => {
                let mut state = self.state.write().await;
                *state = ComputeState::Training {
                    round_id,
                    progress: 0.0,
                };
                Ok(())
            }
            _ => Err("Not registered for this round".to_string()),
        }
    }

    /// Updates training progress.
    pub async fn update_progress(&self, round_id: u64, progress: f32) {
        let mut state = self.state.write().await;
        if let ComputeState::Training { round_id: r, .. } = &*state {
            if *r == round_id {
                *state = ComputeState::Training { round_id, progress };
            }
        }
    }

    /// Completes training with result.
    pub async fn complete_training(&self, result: TrainingResult) {
        let round_id = result.round_id;
        
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.rounds_successful += 1;
            stats.gradients_computed += 1;
            stats.total_compute_time_ms += result.compute_time_ms;
            if !result.proof.is_empty() {
                stats.proofs_generated += 1;
            }
        }

        // Store result
        {
            let mut results = self.results.write().await;
            results.insert(round_id, result);
        }

        // Update state
        let mut state = self.state.write().await;
        *state = ComputeState::Ready { round_id };
    }

    /// Gets a training result.
    pub async fn get_result(&self, round_id: u64) -> Option<TrainingResult> {
        self.results.read().await.get(&round_id).cloned()
    }

    /// Creates a gradient share message.
    pub async fn create_gradient_message(&self, round_id: u64) -> Option<NetworkMessage> {
        let result = self.get_result(round_id).await?;

        Some(NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::Gradient(GradientMessage::ShareGradient {
                round_id,
                gradient_commitment: result.gradient_commitment,
                error_bound: result.error_bound,
                proof: result.proof,
            }),
        ))
    }

    /// Creates a participation request.
    pub fn create_participate_request(&self, round_id: u64) -> NetworkMessage {
        NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::Training(TrainingMessage::ParticipateRequest { round_id }),
        )
    }

    /// Handles a training message.
    pub async fn handle_training_message(&self, msg: TrainingMessage) -> Option<NetworkMessage> {
        match msg {
            TrainingMessage::RoundStart {
                round_id,
                model_hash: _,
                params,
            } => {
                // Auto-request participation if idle
                if matches!(*self.state.read().await, ComputeState::Idle) {
                    return Some(self.create_participate_request(round_id));
                }
                None
            }
            TrainingMessage::ParticipateResponse {
                round_id,
                accepted,
                shard_id,
            } => {
                if accepted {
                    if let Some(shard) = shard_id {
                        let params = self.current_params.read().await.clone()
                            .unwrap_or(TrainingParams {
                                learning_rate: 0.001,
                                batch_size: 32,
                                local_epochs: 5,
                                max_error_bound: 0.1,
                                d_in: 4,
                                d_hid: 8,
                                d_out: 2,
                                model_seed: 42,
                            });
                        self.register_for_round(round_id, shard, params).await;
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Resets to idle state.
    pub async fn reset(&self) {
        let mut state = self.state.write().await;
        *state = ComputeState::Idle;

        let mut current = self.current_params.write().await;
        *current = None;
    }

    /// Gets statistics.
    pub async fn get_stats(&self) -> ComputeStats {
        self.stats.read().await.clone()
    }
}

/// Returns the number of available CPUs.
fn num_cpus() -> impl std::ops::Deref<Target = usize> {
    struct NumCpus(usize);
    impl std::ops::Deref for NumCpus {
        type Target = usize;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    NumCpus(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compute_node_init() {
        let local_id = PeerId::random();
        let node = ComputeNode::new(local_id, ComputeConfig::default());
        
        assert!(matches!(node.get_state().await, ComputeState::Idle));
    }

    #[tokio::test]
    async fn test_register_for_round() {
        let local_id = PeerId::random();
        let node = ComputeNode::new(local_id, ComputeConfig::default());
        
        let params = TrainingParams {
            learning_rate: 0.001,
            batch_size: 32,
            local_epochs: 5,
            max_error_bound: 0.1,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            model_seed: 42,
        };
        
        node.register_for_round(1, 0, params).await;
        
        match node.get_state().await {
            ComputeState::Registered { round_id, shard_id } => {
                assert_eq!(round_id, 1);
                assert_eq!(shard_id, 0);
            }
            _ => panic!("Expected Registered state"),
        }
    }

    #[tokio::test]
    async fn test_start_training() {
        let local_id = PeerId::random();
        let node = ComputeNode::new(local_id, ComputeConfig::default());
        
        let params = TrainingParams {
            learning_rate: 0.001,
            batch_size: 32,
            local_epochs: 5,
            max_error_bound: 0.1,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            model_seed: 42,
        };
        
        node.register_for_round(1, 0, params).await;
        node.start_training(1).await.unwrap();
        
        match node.get_state().await {
            ComputeState::Training { round_id, progress } => {
                assert_eq!(round_id, 1);
                assert_eq!(progress, 0.0);
            }
            _ => panic!("Expected Training state"),
        }
    }
}
