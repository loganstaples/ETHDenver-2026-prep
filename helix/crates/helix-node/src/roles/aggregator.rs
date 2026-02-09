//! Aggregator Node Role.
//!
//! Implements the aggregator node role which collects and aggregates
//! gradients from compute nodes and coordinates training rounds.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

use sha2::{Digest, Sha256};

use crate::network::messages::{
    GradientMessage, MessagePayload, NetworkMessage, NodeCapabilities, PeerId, TrainingMessage,
    TrainingParams,
};

/// Strategy for combining gradient commitments during aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitmentAggregation {
    /// SHA-256 hash of sorted commitments (deterministic, collision-resistant).
    HashBased,
    /// Pedersen commitment addition (homomorphic, from helix-mpc).
    /// Requires that individual commitments are valid Pedersen commitments
    /// on the BN254 G1 curve.
    Pedersen,
}

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
    /// How to combine gradient commitments during aggregation.
    pub commitment_aggregation: CommitmentAggregation,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            min_participants: 3,
            max_participants: 100,
            collection_timeout_secs: 300,
            generate_proof: true,
            max_error_bound: 0.1,
            commitment_aggregation: CommitmentAggregation::HashBased,
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
    ///
    /// Combines individual gradient commitments using the configured
    /// [`CommitmentAggregation`] strategy:
    /// - **HashBased**: SHA-256 of sorted commitments (deterministic,
    ///   collision-resistant, matches the orchestrator's aggregation).
    /// - **Pedersen**: Homomorphic addition of commitments on BN254 G1.
    ///   Requires that each commitment is a serialized Pedersen point.
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

        // Collect and sort commitments for deterministic aggregation
        let mut commitments: Vec<[u8; 32]> = gradients.values()
            .map(|g| g.commitment)
            .collect();
        commitments.sort();

        let combined_commitment = match self.config.commitment_aggregation {
            CommitmentAggregation::HashBased => {
                // SHA-256 hash of sorted, validated commitments
                let mut hasher = Sha256::new();
                for commitment in &commitments {
                    hasher.update(commitment);
                }
                let result: [u8; 32] = hasher.finalize().into();
                result
            }
            CommitmentAggregation::Pedersen => {
                // Homomorphic addition of Pedersen commitments on BN254 G1.
                //
                // Each 32-byte commitment is interpreted as a compressed G1Affine
                // point. If any commitment fails to deserialize (e.g. because the
                // wire protocol currently sends SHA-256 hashes rather than EC
                // points), we fall back to hash-based aggregation with a warning.
                use helix_mpc::security::commitment::PedersenCommitment;
                use halo2curves::bn256::G1Affine;
                use halo2curves::serde::SerdeObject;

                let mut acc: Option<PedersenCommitment> = None;
                let mut pedersen_ok = true;

                for grad in gradients.values() {
                    // G1Affine compressed is 32 bytes on BN254
                    match G1Affine::from_raw_bytes(&grad.commitment) {
                        Some(point) => {
                            let pc = PedersenCommitment { point };
                            acc = Some(match acc {
                                Some(a) => a.add(&pc),
                                None => pc,
                            });
                        }
                        None => {
                            log::warn!(
                                "Failed to parse Pedersen commitment from {}, falling back to hash-based",
                                grad.participant,
                            );
                            pedersen_ok = false;
                            break;
                        }
                    }
                }

                if !pedersen_ok || acc.is_none() {
                    // Fallback: SHA-256 hash of sorted commitments
                    let mut hasher = Sha256::new();
                    for c in &commitments {
                        hasher.update(c);
                    }
                    hasher.finalize().into()
                } else {
                    // Serialize the aggregated G1 point back to 32 bytes
                    let agg = acc.unwrap();
                    let mut bytes = [0u8; 32];
                    let raw = agg.point.to_raw_bytes();
                    let len = raw.len().min(32);
                    bytes[..len].copy_from_slice(&raw[..len]);
                    bytes
                }
            }
        };

        let mut total_error = 0.0;
        for grad in gradients.values() {
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

    /// Helper: builds an AggregatedResult, stores it, and updates stats.
    async fn build_result(
        &self,
        round_id: u64,
        commitment: [u8; 32],
        gradients: &HashMap<PeerId, CollectedGradient>,
    ) -> AggregatedResult {
        let mut total_error = 0.0;
        for grad in gradients.values() {
            total_error += grad.error_bound;
        }
        total_error /= gradients.len() as f64;

        let result = AggregatedResult {
            round_id,
            commitment,
            total_error_bound: total_error,
            num_participants: gradients.len(),
            proof: vec![],
        };

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.rounds_successful += 1;
            stats.gradients_aggregated += gradients.len() as u64;
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

        result
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
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            model_seed: 42,
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
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            model_seed: 42,
        };

        node.start_round([1; 32], params).await;

        let participant = PeerId::random();
        let response = node.handle_participate_request(participant, 1).await;

        assert!(response.is_some());
    }

    #[tokio::test]
    async fn test_hash_based_aggregation_is_deterministic() {
        let local_id = PeerId::random();
        let config = AggregatorConfig {
            min_participants: 1,
            commitment_aggregation: CommitmentAggregation::HashBased,
            ..Default::default()
        };
        let node = AggregatorNode::new(local_id, config);

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

        node.start_round([1; 32], params).await;

        let p1 = PeerId::random();
        let p2 = PeerId::random();
        node.handle_participate_request(p1.clone(), 1).await;
        node.handle_participate_request(p2.clone(), 1).await;
        node.start_collection().await;

        node.handle_gradient_share(p1, 1, [0xAA; 32], 0.01, vec![1, 2, 3]).await;
        node.handle_gradient_share(p2, 1, [0xBB; 32], 0.02, vec![4, 5, 6]).await;

        let result = node.aggregate().await.expect("aggregation should succeed");

        // Recompute expected: SHA-256 of sorted commitments
        use sha2::{Digest, Sha256};
        let mut commitments = vec![[0xAA; 32], [0xBB; 32]];
        commitments.sort();
        let mut hasher = Sha256::new();
        for c in &commitments {
            hasher.update(c);
        }
        let expected: [u8; 32] = hasher.finalize().into();

        assert_eq!(result.commitment, expected);
        assert_eq!(result.num_participants, 2);
    }
}
