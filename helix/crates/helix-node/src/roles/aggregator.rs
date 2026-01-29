//! Aggregator Node Role.
//!
//! Implements the aggregator node role which collects and aggregates
//! gradients from compute nodes and coordinates training rounds.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

use crate::network::messages::{
    GradientMessage, MessagePayload, NetworkMessage, NodeCapabilities, PeerId, TrainingMessage,
    TrainingParams,
};

/// Aggregator state.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregatorState {
    /// Idle, not coordinating.
    Idle,
    /// Recruiting participants for a round.
    Recruiting { round_id: u64, participants: usize },
    /// Collecting gradients.
    Collecting { round_id: u64, received: usize, expected: usize },
    /// Aggregating collected gradients.
    Aggregating { round_id: u64 },
    /// Round complete, ready to finalize.
    Complete { round_id: u64 },
    /// Error state.
    Error { message: String },
}

/// Configuration for aggregator node.
#[derive(Debug, Clone)]
pub struct AggregatorConfig {
    /// Minimum participants per round.
    pub min_participants: usize,
    /// Maximum participants per round.
    pub max_participants: usize,
    /// Timeout for collection (seconds).
    pub collection_timeout_secs: u64,
    /// Whether to generate aggregation proof.
    pub generate_proof: bool,
    /// Maximum error bound for aggregated result.
    pub max_error_bound: f64,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            min_participants: 3,
            max_participants: 100,
            collection_timeout_secs: 300,
            generate_proof: true,
            max_error_bound: 0.1,
        }
    }
}

/// Collected gradient from a participant.
#[derive(Debug, Clone)]
pub struct CollectedGradient {
    /// Participant peer ID.
    pub participant: PeerId,
    /// Gradient commitment.
    pub commitment: [u8; 32],
    /// Error bound.
    pub error_bound: f64,
    /// Proof of computation.
    pub proof: Vec<u8>,
    /// Received timestamp.
    pub received_at: u64,
}

/// Aggregated result.
#[derive(Debug, Clone)]
pub struct AggregatedResult {
    /// Round ID.
    pub round_id: u64,
    /// Aggregated commitment.
    pub commitment: [u8; 32],
    /// Total error bound.
    pub total_error_bound: f64,
    /// Number of participants.
    pub num_participants: usize,
    /// Aggregation proof.
    pub proof: Vec<u8>,
}

/// Aggregator node role.
pub struct AggregatorNode {
    /// Our peer ID.
    local_id: PeerId,
    /// Configuration.
    config: AggregatorConfig,
    /// Current state.
    state: Arc<RwLock<AggregatorState>>,
    /// Current round.
    current_round: Arc<RwLock<u64>>,
    /// Registered participants.
    participants: Arc<RwLock<HashMap<PeerId, u32>>>,
    /// Collected gradients.
    gradients: Arc<RwLock<HashMap<PeerId, CollectedGradient>>>,
    /// Completed rounds.
    completed_rounds: Arc<RwLock<HashMap<u64, AggregatedResult>>>,
    /// Statistics.
    stats: Arc<RwLock<AggregatorStats>>,
}

/// Aggregator statistics.
#[derive(Debug, Clone, Default)]
pub struct AggregatorStats {
    /// Total rounds coordinated.
    pub rounds_coordinated: u64,
    /// Successful rounds.
    pub rounds_successful: u64,
    /// Total gradients aggregated.
    pub gradients_aggregated: u64,
    /// Average participants per round.
    pub avg_participants: f64,
}

impl AggregatorNode {
    /// Creates a new aggregator node.
    pub fn new(local_id: PeerId, config: AggregatorConfig) -> Self {
        Self {
            local_id,
            config,
            state: Arc::new(RwLock::new(AggregatorState::Idle)),
            current_round: Arc::new(RwLock::new(0)),
            participants: Arc::new(RwLock::new(HashMap::new())),
            gradients: Arc::new(RwLock::new(HashMap::new())),
            completed_rounds: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(AggregatorStats::default())),
        }
    }

    /// Gets the node's capabilities.
    pub fn capabilities(&self) -> NodeCapabilities {
        NodeCapabilities {
            can_train: false,
            can_aggregate: true,
            can_prove: self.config.generate_proof,
            gpu_memory_mb: 0,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4),
            storage_gb: 500,
        }
    }

    /// Gets current state.
    pub async fn get_state(&self) -> AggregatorState {
        self.state.read().await.clone()
    }

    /// Starts a new training round.
    pub async fn start_round(
        &self,
        model_hash: [u8; 32],
        params: TrainingParams,
    ) -> NetworkMessage {
        let round_id = {
            let mut round = self.current_round.write().await;
            *round += 1;
            *round
        };

        // Clear previous round data
        {
            let mut participants = self.participants.write().await;
            participants.clear();
        }
        {
            let mut gradients = self.gradients.write().await;
            gradients.clear();
        }

        // Update state
        {
            let mut state = self.state.write().await;
            *state = AggregatorState::Recruiting {
                round_id,
                participants: 0,
            };
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.rounds_coordinated += 1;
        }

        // Create round start message
        NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::Training(TrainingMessage::RoundStart {
                round_id,
                model_hash,
                params,
            }),
        )
    }

    /// Handles a participation request.
    pub async fn handle_participate_request(
        &self,
        from: PeerId,
        round_id: u64,
    ) -> Option<NetworkMessage> {
        let current_round = *self.current_round.read().await;
        if round_id != current_round {
            return Some(self.create_participate_response(round_id, false, None));
        }

        let mut participants = self.participants.write().await;
        if participants.len() >= self.config.max_participants {
            return Some(self.create_participate_response(round_id, false, None));
        }

        let shard_id = participants.len() as u32;
        participants.insert(from.clone(), shard_id);

        // Update state
        {
            let mut state = self.state.write().await;
            *state = AggregatorState::Recruiting {
                round_id,
                participants: participants.len(),
            };
        }

        Some(self.create_participate_response(round_id, true, Some(shard_id)))
    }

    /// Transitions to collection phase.
    pub async fn start_collection(&self) {
        let round_id = *self.current_round.read().await;
        let num_participants = self.participants.read().await.len();

        let mut state = self.state.write().await;
        *state = AggregatorState::Collecting {
            round_id,
            received: 0,
            expected: num_participants,
        };
    }

    /// Handles a gradient share.
    pub async fn handle_gradient_share(
        &self,
        from: PeerId,
        round_id: u64,
        commitment: [u8; 32],
        error_bound: f64,
        proof: Vec<u8>,
    ) -> bool {
        let current_round = *self.current_round.read().await;
        if round_id != current_round {
            return false;
        }

        // Check if from registered participant
        let is_participant = self.participants.read().await.contains_key(&from);
        if !is_participant {
            return false;
        }

        // Store gradient
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let gradient = CollectedGradient {
            participant: from.clone(),
            commitment,
            error_bound,
            proof,
            received_at: now,
        };

        {
            let mut gradients = self.gradients.write().await;
            gradients.insert(from, gradient);
        }

        // Update state
        let received = self.gradients.read().await.len();
        let expected = self.participants.read().await.len();

        {
            let mut state = self.state.write().await;
            *state = AggregatorState::Collecting {
                round_id,
                received,
                expected,
            };
        }

        // Check if all received
        received >= expected
    }

    /// Aggregates collected gradients.
    pub async fn aggregate(&self) -> Option<AggregatedResult> {
        let round_id = *self.current_round.read().await;

        // Update state
        {
            let mut state = self.state.write().await;
            *state = AggregatorState::Aggregating { round_id };
        }

        let gradients = self.gradients.read().await;
        if gradients.is_empty() {
            return None;
        }

        // Compute aggregated commitment (simplified)
        let mut combined_commitment = [0u8; 32];
        let mut total_error = 0.0;

        for (i, grad) in gradients.values().enumerate() {
            for (j, byte) in grad.commitment.iter().enumerate() {
                combined_commitment[j] ^= byte;
            }
            total_error += grad.error_bound;
        }

        // Average error
        total_error /= gradients.len() as f64;

        let result = AggregatedResult {
            round_id,
            commitment: combined_commitment,
            total_error_bound: total_error,
            num_participants: gradients.len(),
            proof: vec![], // Would generate proof here
        };

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.rounds_successful += 1;
            stats.gradients_aggregated += gradients.len() as u64;
            
            // Update rolling average
            let total_rounds = stats.rounds_successful as f64;
            stats.avg_participants = (stats.avg_participants * (total_rounds - 1.0) 
                + gradients.len() as f64) / total_rounds;
        }

        // Store result
        {
            let mut completed = self.completed_rounds.write().await;
            completed.insert(round_id, result.clone());
        }

        // Update state
        {
            let mut state = self.state.write().await;
            *state = AggregatorState::Complete { round_id };
        }

        Some(result)
    }

    /// Creates an aggregated gradient message.
    pub async fn create_aggregated_message(&self, round_id: u64) -> Option<NetworkMessage> {
        let completed = self.completed_rounds.read().await;
        let result = completed.get(&round_id)?;

        Some(NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::Gradient(GradientMessage::AggregatedGradient {
                round_id,
                commitment: result.commitment,
                proof: result.proof.clone(),
            }),
        ))
    }

    /// Gets a completed round result.
    pub async fn get_result(&self, round_id: u64) -> Option<AggregatedResult> {
        self.completed_rounds.read().await.get(&round_id).cloned()
    }

    /// Resets to idle state.
    pub async fn reset(&self) {
        let mut state = self.state.write().await;
        *state = AggregatorState::Idle;

        let mut participants = self.participants.write().await;
        participants.clear();

        let mut gradients = self.gradients.write().await;
        gradients.clear();
    }

    /// Gets statistics.
    pub async fn get_stats(&self) -> AggregatorStats {
        self.stats.read().await.clone()
    }

    fn create_participate_response(
        &self,
        round_id: u64,
        accepted: bool,
        shard_id: Option<u32>,
    ) -> NetworkMessage {
        NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::Training(TrainingMessage::ParticipateResponse {
                round_id,
                accepted,
                shard_id,
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_aggregator_init() {
        let local_id = PeerId::random();
        let node = AggregatorNode::new(local_id, AggregatorConfig::default());
        
        assert!(matches!(node.get_state().await, AggregatorState::Idle));
    }

    #[tokio::test]
    async fn test_start_round() {
        let local_id = PeerId::random();
        let node = AggregatorNode::new(local_id, AggregatorConfig::default());
        
        let params = TrainingParams {
            learning_rate: 0.001,
            batch_size: 32,
            local_epochs: 5,
            max_error_bound: 0.1,
        };
        
        let _msg = node.start_round([1; 32], params).await;
        
        match node.get_state().await {
            AggregatorState::Recruiting { round_id, .. } => {
                assert_eq!(round_id, 1);
            }
            _ => panic!("Expected Recruiting state"),
        }
    }

    #[tokio::test]
    async fn test_handle_participate() {
        let local_id = PeerId::random();
        let node = AggregatorNode::new(local_id, AggregatorConfig::default());
        
        let params = TrainingParams {
            learning_rate: 0.001,
            batch_size: 32,
            local_epochs: 5,
            max_error_bound: 0.1,
        };
        
        node.start_round([1; 32], params).await;
        
        let participant = PeerId::random();
        let response = node.handle_participate_request(participant, 1).await;
        
        assert!(response.is_some());
    }
}
