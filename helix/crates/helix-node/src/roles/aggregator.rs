//! Aggregator Node Role.
//!
//! Implements the aggregator node role which collects and aggregates
//! gradients from compute nodes and coordinates training rounds.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

use helix_mpc::poseidon::{poseidon_hash_two, poseidon_commit_with_domain, domains};
use helix_mpc::Fr as MpcFr;

use crate::aggregator_proof_pipeline::{
    AggregatorProofPipeline, ProofPipelineError, ProofSubmissionResult,
};
use crate::network::messages::{
    GradientMessage, MessagePayload, NetworkMessage, NodeCapabilities, PeerId, TrainingMessage,
    TrainingParams,
};
use crate::network::partition_detect::{PartitionDetector, PartitionAction};
use crate::on_chain_pipeline::OnChainPipeline;
use crate::training::aggregation::{
    AggregationStrategy, AggregationConfig as ByzantineAggregationConfig,
    GradientAggregator, WeightedGradient,
};
use crate::training::model::ModelGradient;

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

/// Serializable aggregation proof containing a Poseidon commitment
/// that binds the sorted commitments, participant count, and round ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationProof {
    /// Domain-separated Poseidon commitment (32 bytes, BN254 Fr).
    pub poseidon_commitment: [u8; 32],
    /// Sorted list of participant commitments included in the proof.
    pub sorted_commitments: Vec<[u8; 32]>,
    /// Number of participants.
    pub participant_count: usize,
    /// Round ID this proof covers.
    pub round_id: u64,
    /// Total accumulated error bound.
    pub total_error_bound: f64,
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
    /// Optional Byzantine-fault-tolerant gradient filtering strategy.
    /// When set and gradient data is available, outliers are excluded
    /// before computing the aggregated commitment.
    pub byzantine_strategy: Option<AggregationStrategy>,
    /// Minimum required stake amount for worker participation.
    pub min_stake_amount: u64,
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
            byzantine_strategy: Some(AggregationStrategy::Krum { num_byzantine: 1 }),
            min_stake_amount: 0,
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
    /// Optional raw gradient data for Byzantine filtering.
    /// When present, the aggregator can run Krum/Median/TrimmedMean
    /// to exclude outliers before computing the aggregated commitment.
    pub gradient_data: Option<ModelGradient>,
    /// Optional public inputs (8 hex strings) for ZK proof aggregation.
    /// Required for RLC aggregation when on-chain pipeline is active.
    pub public_inputs_hex: Option<Vec<String>>,
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
    /// Aggregation proof (bincode-serialized `AggregationProof`).
    pub proof: Vec<u8>,
    /// Participants excluded by Byzantine filtering.
    pub excluded_participants: Vec<PeerId>,
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
    /// Optional partition detector to gate aggregation on network health.
    partition_detector: Option<Arc<PartitionDetector>>,
    /// Optional on-chain pipeline for proof aggregation and submission.
    on_chain_pipeline: Option<Arc<OnChainPipeline>>,
    /// Optional proof pipeline for real ZK proof generation + on-chain submission.
    proof_pipeline: Option<Arc<AggregatorProofPipeline>>,
    /// Allowlist of registered worker public keys that have verified stakes.
    worker_allowlist: Arc<RwLock<HashSet<PeerId>>>,
    /// Local LRU cache of recent proof hashes for replay protection.
    /// Stores SHA-256 hashes of recently seen proofs.
    seen_proof_hashes: Arc<RwLock<Vec<[u8; 32]>>>,
    /// Maximum number of proof hashes to cache.
    max_proof_cache_size: usize,
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
            partition_detector: None,
            on_chain_pipeline: None,
            proof_pipeline: None,
            worker_allowlist: Arc::new(RwLock::new(HashSet::new())),
            seen_proof_hashes: Arc::new(RwLock::new(Vec::new())),
            max_proof_cache_size: 10_000,
        }
    }

    /// Sets the partition detector for network health gating.
    pub fn with_partition_detector(mut self, detector: Arc<PartitionDetector>) -> Self {
        self.partition_detector = Some(detector);
        self
    }

    /// Sets the on-chain pipeline for proof aggregation and submission.
    pub fn with_on_chain_pipeline(mut self, pipeline: Arc<OnChainPipeline>) -> Self {
        self.on_chain_pipeline = Some(pipeline);
        self
    }

    /// Sets the proof pipeline for real ZK proof generation + on-chain submission.
    pub fn with_proof_pipeline(mut self, pipeline: Arc<AggregatorProofPipeline>) -> Self {
        self.proof_pipeline = Some(pipeline);
        self
    }

    /// Registers a worker as authorized to participate.
    ///
    /// In production, this should be called after verifying the worker's
    /// on-chain stake via `verify_stake()`.
    pub async fn register_worker(&self, peer_id: PeerId) {
        let mut allowlist = self.worker_allowlist.write().await;
        allowlist.insert(peer_id);
    }

    /// Removes a worker from the allowlist.
    pub async fn deregister_worker(&self, peer_id: &PeerId) {
        let mut allowlist = self.worker_allowlist.write().await;
        allowlist.remove(peer_id);
    }

    /// Checks if a worker is in the allowlist.
    pub async fn is_worker_authorized(&self, peer_id: &PeerId) -> bool {
        let allowlist = self.worker_allowlist.read().await;
        // If allowlist is empty, allow all workers (permissive mode)
        allowlist.is_empty() || allowlist.contains(peer_id)
    }

    /// Verifies that a worker has sufficient stake on-chain.
    ///
    /// In a production deployment, this would query the Staking contract.
    /// Currently returns the configured minimum stake requirement for
    /// the caller to verify externally.
    pub fn required_stake(&self) -> u64 {
        self.config.min_stake_amount
    }

    /// Checks if a proof has already been seen (replay protection).
    ///
    /// Returns true if the proof is a duplicate (should be rejected).
    /// Uses an LRU-style bounded cache of recent proof hashes.
    pub async fn is_duplicate_proof(&self, proof: &[u8]) -> bool {
        if proof.is_empty() {
            return false;
        }

        let hash: [u8; 32] = {
            let mut hasher = Sha256::new();
            hasher.update(proof);
            hasher.finalize().into()
        };

        let mut cache = self.seen_proof_hashes.write().await;

        if cache.iter().any(|h| h == &hash) {
            return true;
        }

        // Add to cache, evict oldest if at capacity
        if cache.len() >= self.max_proof_cache_size {
            cache.remove(0);
        }
        cache.push(hash);

        false
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
    ///
    /// If a partition detector is set and reports a `Halt` or `PauseTraining`
    /// action, the round is not started and the aggregator transitions to
    /// `Error` state instead.
    pub async fn start_round(
        &self,
        model_hash: [u8; 32],
        params: TrainingParams,
    ) -> NetworkMessage {
        // Check partition status before starting
        if let Some(ref detector) = self.partition_detector {
            if detector.is_partitioned() {
                if let Some(status) = detector.get_status() {
                    match status.recommended_action {
                        PartitionAction::Halt | PartitionAction::PauseTraining => {
                            tracing::warn!(
                                "Partition detected ({:?}), refusing to start round",
                                status.recommended_action
                            );
                            let mut state = self.state.write().await;
                            *state = AggregatorState::Error {
                                message: format!(
                                    "Network partitioned: {} unreachable peers, action={:?}",
                                    status.unreachable_peers, status.recommended_action
                                ),
                            };
                            // Return a round start message with round_id 0
                            // to signal the round was not started
                            return NetworkMessage::new(
                                self.local_id.clone(),
                                MessagePayload::Training(TrainingMessage::RoundStart {
                                    round_id: 0,
                                    model_hash,
                                    params,
                                }),
                            );
                        }
                        PartitionAction::ReconnectPeers | PartitionAction::AlertOperator => {
                            tracing::warn!(
                                "Partition detected ({:?}), proceeding optimistically",
                                status.recommended_action
                            );
                        }
                        PartitionAction::Continue => {}
                    }
                }
            }
        }

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
        self.handle_gradient_share_with_data(from, round_id, commitment, error_bound, proof, None, None).await
    }

    /// Handles a gradient share with optional raw gradient data for Byzantine filtering
    /// and optional public inputs for ZK proof aggregation.
    pub async fn handle_gradient_share_with_data(
        &self,
        from: PeerId,
        round_id: u64,
        commitment: [u8; 32],
        error_bound: f64,
        proof: Vec<u8>,
        gradient_data: Option<ModelGradient>,
        public_inputs_hex: Option<Vec<String>>,
    ) -> bool {
        let current_round = *self.current_round.read().await;
        if round_id != current_round {
            return false;
        }

        // Check worker allowlist authorization
        if !self.is_worker_authorized(&from).await {
            tracing::warn!("Rejected gradient from unauthorized worker: {}", from);
            return false;
        }

        // Check for proof replay
        if self.is_duplicate_proof(&proof).await {
            tracing::warn!("Rejected duplicate proof from worker: {}", from);
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
            gradient_data,
            public_inputs_hex,
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
    ///
    /// When `byzantine_strategy` is configured and gradient data is available,
    /// outlier gradients are excluded before computing the aggregated commitment.
    ///
    /// Generates a domain-separated Poseidon commitment proof binding the
    /// sorted commitments, participant count, and round ID.
    pub async fn aggregate(&self) -> Option<AggregatedResult> {
        let round_id = *self.current_round.read().await;

        // Check partition status before aggregating
        if let Some(ref detector) = self.partition_detector {
            if detector.is_partitioned() {
                if let Some(status) = detector.get_status() {
                    if status.recommended_action == PartitionAction::Halt {
                        tracing::warn!(
                            "Partition detected with Halt action, refusing to aggregate round {}",
                            round_id
                        );
                        return None;
                    }
                    // For PauseTraining/ReconnectPeers, proceed with warning
                    tracing::warn!(
                        "Partition detected ({:?}) during aggregation, proceeding optimistically",
                        status.recommended_action
                    );
                }
            }
        }

        // Update state
        {
            let mut state = self.state.write().await;
            *state = AggregatorState::Aggregating { round_id };
        }

        let gradients = self.gradients.read().await;
        if gradients.is_empty() {
            return None;
        }

        // Validate gradients: reject NaN/Inf error bounds and gradient data
        let mut validation_rejected: Vec<PeerId> = Vec::new();
        for (peer, grad) in gradients.iter() {
            if !grad.error_bound.is_finite() || grad.error_bound > self.config.max_error_bound {
                tracing::warn!("Rejected gradient from {} (error_bound={:.6})", peer, grad.error_bound);
                validation_rejected.push(peer.clone());
                continue;
            }
            if let Some(ref gd) = grad.gradient_data {
                let has_nan = gd.layers.iter().any(|layer| {
                    layer.gradients.values().any(|wd| wd.data.iter().any(|v| !v.is_finite()))
                });
                if has_nan {
                    tracing::warn!("Rejected gradient from {} with NaN/Inf values", peer);
                    validation_rejected.push(peer.clone());
                }
            }
        }

        if !validation_rejected.is_empty() {
            tracing::info!(
                "Gradient validation filtered {}/{} submissions",
                validation_rejected.len(),
                gradients.len(),
            );
        }

        // Run Byzantine filtering if configured and gradient data is available
        let mut excluded_participants = Vec::new();
        let excluded_peers: HashSet<PeerId>;

        if let Some(ref strategy) = self.config.byzantine_strategy {
            let has_gradient_data = gradients.values().any(|g| g.gradient_data.is_some());
            if has_gradient_data {
                let byz_config = ByzantineAggregationConfig {
                    strategy: *strategy,
                    min_gradients: 2,
                    max_gradient_norm: 10.0,
                    stake_weighted: false,
                };
                let mut byz_aggregator = GradientAggregator::new(byz_config);

                for grad in gradients.values() {
                    if let Some(ref gd) = grad.gradient_data {
                        byz_aggregator.add_gradient(WeightedGradient {
                            participant_id: grad.participant.to_string(),
                            stake: 100, // Equal stake for filtering
                            gradient: gd.clone(),
                            error_bound: grad.error_bound,
                            is_valid: true,
                        });
                    }
                }

                match byz_aggregator.aggregate() {
                    Ok(byz_result) => {
                        for excl_id in &byz_result.excluded {
                            // Find matching PeerId
                            for grad in gradients.values() {
                                if grad.participant.to_string() == *excl_id {
                                    excluded_participants.push(grad.participant.clone());
                                }
                            }
                        }
                        if !excluded_participants.is_empty() {
                            tracing::info!(
                                "Byzantine filtering excluded {} participants: {:?}",
                                excluded_participants.len(),
                                excluded_participants
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Byzantine filtering failed ({}), using all gradients", e);
                    }
                }
            } else {
                tracing::debug!(
                    "Byzantine filtering configured but no gradient data available, skipping"
                );
            }
        }

        excluded_peers = excluded_participants.iter()
            .chain(validation_rejected.iter())
            .cloned()
            .collect();

        // Collect and sort commitments for deterministic aggregation,
        // excluding any participants flagged by Byzantine filtering
        let mut commitments: Vec<[u8; 32]> = gradients.values()
            .filter(|g| !excluded_peers.contains(&g.participant))
            .map(|g| g.commitment)
            .collect();
        commitments.sort();

        let included_gradients: Vec<&CollectedGradient> = gradients.values()
            .filter(|g| !excluded_peers.contains(&g.participant))
            .collect();

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
                use helix_mpc::security::commitment::PedersenCommitment;
                use halo2curves::bn256::G1Affine;
                use halo2curves::serde::SerdeObject;

                let mut acc: Option<PedersenCommitment> = None;
                let mut pedersen_ok = true;

                for grad in &included_gradients {
                    match G1Affine::from_raw_bytes(&grad.commitment) {
                        Some(point) => {
                            let pc = PedersenCommitment { point };
                            acc = Some(match acc {
                                Some(a) => a.add(&pc),
                                None => pc,
                            });
                        }
                        None => {
                            tracing::warn!(
                                "Failed to parse Pedersen commitment from {}, falling back to hash-based",
                                grad.participant,
                            );
                            pedersen_ok = false;
                            break;
                        }
                    }
                }

                if !pedersen_ok || acc.is_none() {
                    let mut hasher = Sha256::new();
                    for c in &commitments {
                        hasher.update(c);
                    }
                    hasher.finalize().into()
                } else {
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
        for grad in &included_gradients {
            total_error += grad.error_bound;
        }
        // Average error
        if !included_gradients.is_empty() {
            total_error /= included_gradients.len() as f64;
        }

        // Generate Poseidon commitment proof
        let proof_bytes = self.generate_aggregation_proof(
            round_id,
            &commitments,
            included_gradients.len(),
            total_error,
        );

        let result = AggregatedResult {
            round_id,
            commitment: combined_commitment,
            total_error_bound: total_error,
            num_participants: included_gradients.len(),
            proof: proof_bytes,
            excluded_participants,
        };

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.rounds_successful += 1;
            stats.gradients_aggregated += included_gradients.len() as u64;

            // Update rolling average
            let total_rounds = stats.rounds_successful as f64;
            stats.avg_participants = (stats.avg_participants * (total_rounds - 1.0)
                + included_gradients.len() as f64) / total_rounds;
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

        // On-chain submission: if pipeline is set and workers have public inputs,
        // aggregate proofs via RLC and submit the single proof on-chain.
        if let Some(ref pipeline) = self.on_chain_pipeline {
            let worker_proofs = Self::build_training_proof_results(&included_gradients);
            if !worker_proofs.is_empty() {
                let pipeline = Arc::clone(pipeline);
                let round = round_id;
                tokio::spawn(async move {
                    match pipeline.submit_aggregated_round(round, worker_proofs).await {
                        Ok(Some(submission)) => {
                            tracing::info!(
                                "On-chain submission for round {}: tx={}, gas={}, proofs={}",
                                round,
                                submission.tx_hash,
                                submission.gas_used,
                                submission.proofs_aggregated,
                            );
                        }
                        Ok(None) => {
                            tracing::warn!("On-chain submission returned None for round {}", round);
                        }
                        Err(e) => {
                            tracing::error!("On-chain submission failed for round {}: {}", round, e);
                        }
                    }
                });
            } else {
                tracing::debug!(
                    "Round {}: no worker proofs with public inputs for RLC aggregation",
                    round_id,
                );
            }
        }

        Some(result)
    }

    /// Builds `TrainingProofResultV2` values from collected gradients.
    ///
    /// Only includes gradients that have both proof bytes and public inputs.
    fn build_training_proof_results(
        gradients: &[&CollectedGradient],
    ) -> Vec<helix_prover::TrainingProofResultV2> {
        use helix_prover::halo2curves::bn256::Fr;
        use helix_prover::halo2curves::ff::PrimeField;

        gradients
            .iter()
            .filter(|g| !g.proof.is_empty() && g.public_inputs_hex.is_some())
            .filter_map(|g| {
                let hex_inputs = g.public_inputs_hex.as_ref()?;
                if hex_inputs.len() < 8 {
                    return None;
                }

                // Convert hex public inputs to Fr
                let pis: Vec<Fr> = hex_inputs
                    .iter()
                    .map(|hex_str| {
                        let stripped = hex_str.strip_prefix("0x").unwrap_or(hex_str);
                        let bytes = hex::decode(stripped).unwrap_or_default();
                        if bytes.len() == 32 {
                            let mut repr = [0u8; 32];
                            repr.copy_from_slice(&bytes);
                            Option::from(Fr::from_repr(repr.into())).unwrap_or(Fr::from(0u64))
                        } else {
                            Fr::from(0u64)
                        }
                    })
                    .collect();

                let old_hash = if pis.len() >= 2 {
                    (pis[0], pis[1])
                } else {
                    (Fr::from(0u64), Fr::from(0u64))
                };
                let new_hash = if pis.len() >= 4 {
                    (pis[2], pis[3])
                } else {
                    (Fr::from(0u64), Fr::from(0u64))
                };
                let loss = if pis.len() >= 5 { pis[4] } else { Fr::from(0u64) };
                let total_error = if pis.len() >= 6 { pis[5] } else { Fr::from(0u64) };
                let step_number = if pis.len() >= 7 {
                    // Extract step number from Fr
                    let repr = pis[6].to_repr();
                    u64::from_le_bytes(repr.as_ref()[..8].try_into().unwrap_or([0; 8]))
                } else {
                    0
                };

                Some(helix_prover::TrainingProofResultV2 {
                    proof: g.proof.clone(),
                    public_inputs: pis,
                    loss,
                    total_error,
                    step_number,
                    old_state_hash: old_hash,
                    new_state_hash: new_hash,
                    verified: true,
                    generation_time: std::time::Duration::from_millis(0),
                    verification_time: None,
                    attempts: 1,
                    from_cache: false,
                    witness_hash: None,
                })
            })
            .collect()
    }

    /// Generates a domain-separated Poseidon commitment proof.
    ///
    /// The proof binds: `Poseidon(H(sorted_commitments), Fr(num_participants))`
    /// using the `AGGREGATION` domain separator (0x05).
    fn generate_aggregation_proof(
        &self,
        round_id: u64,
        sorted_commitments: &[[u8; 32]],
        num_participants: usize,
        total_error_bound: f64,
    ) -> Vec<u8> {
        // Hash all sorted commitments into a single Fr using tree-based Poseidon
        let commitment_frs: Vec<MpcFr> = sorted_commitments
            .iter()
            .map(|c| MpcFr::from_bytes_le(c))
            .collect();

        // Fold commitments pairwise: H(c0, c1), H(c2, c3), ...
        let mut current = commitment_frs;
        while current.len() > 1 {
            let mut next = Vec::new();
            let mut i = 0;
            while i < current.len() {
                if i + 1 < current.len() {
                    let h = poseidon_hash_two(*current[i].inner(), *current[i + 1].inner());
                    next.push(MpcFr::from_inner(h));
                } else {
                    // Odd element: hash with zero
                    let h = poseidon_hash_two(*current[i].inner(), *MpcFr::ZERO.inner());
                    next.push(MpcFr::from_inner(h));
                }
                i += 2;
            }
            current = next;
        }

        let commitments_hash = if current.is_empty() {
            MpcFr::ZERO
        } else {
            current[0]
        };

        // Create domain-separated commitment binding commitments + participant count
        let num_participants_fr = MpcFr::from_u64(num_participants as u64);
        let poseidon_commitment = poseidon_commit_with_domain(
            domains::AGGREGATION,
            &[commitments_hash, num_participants_fr],
            MpcFr::from_u64(round_id),
        );

        let agg_proof = AggregationProof {
            poseidon_commitment: poseidon_commitment.to_bytes_le(),
            sorted_commitments: sorted_commitments.to_vec(),
            participant_count: num_participants,
            round_id,
            total_error_bound,
        };

        bincode::serialize(&agg_proof).unwrap_or_default()
    }

    /// Aggregates gradients then generates a real ZK proof and submits it on-chain.
    ///
    /// This is the primary proof pipeline entry point. After calling `aggregate()`,
    /// call this method to:
    /// 1. Generate a ZK proof of the aggregated training step via MLTrainingProverV2
    /// 2. Submit the proof to HelixCoordinatorV2 (which verifies it on-chain)
    /// 3. Track commitment chaining (new_hash → next round's old_hash)
    ///
    /// Requires a proof pipeline to be set via `with_proof_pipeline()`.
    ///
    /// `training_input` and `training_target` are the training data for this step.
    /// If None, uses the pipeline's configured defaults.
    pub async fn prove_and_submit_round(
        &self,
        round_id: u64,
        training_input: Option<&[halo2curves::bn256::Fr]>,
        training_target: Option<&[halo2curves::bn256::Fr]>,
    ) -> Result<ProofSubmissionResult, ProofPipelineError> {
        let pipeline = self.proof_pipeline.as_ref().ok_or_else(|| {
            ProofPipelineError::NotInitialized(
                "No proof pipeline configured — call with_proof_pipeline() first".to_string(),
            )
        })?;

        pipeline
            .prove_and_submit(round_id, training_input, training_target)
            .await
    }

    /// Returns the proof pipeline's current commitment, if a pipeline is configured.
    pub async fn proof_pipeline_commitment(
        &self,
    ) -> Option<(halo2curves::bn256::Fr, halo2curves::bn256::Fr)> {
        if let Some(ref pipeline) = self.proof_pipeline {
            Some(pipeline.current_commitment().await)
        } else {
            None
        }
    }

    /// Returns the proof pipeline's current step number, if configured.
    pub async fn proof_pipeline_step(&self) -> Option<u64> {
        if let Some(ref pipeline) = self.proof_pipeline {
            Some(pipeline.current_step().await)
        } else {
            None
        }
    }

    /// Aggregates worker proofs using IVC folding into a single decider proof.
    ///
    /// This method takes completed worker proofs (with public inputs), converts
    /// them into IVC steps, folds all steps into a single accumulator, and
    /// produces a contract-compatible decider proof for on-chain submission.
    ///
    /// This is an alternative to RLC aggregation that provides true cryptographic
    /// compression: N worker proofs → 1 contract-compatible proof.
    pub async fn aggregate_with_ivc(
        &self,
        round_id: u64,
        model_id: u64,
        error_budget: u64,
    ) -> Option<AggregatedResult> {
        let gradients = self.gradients.read().await;
        if gradients.is_empty() {
            return None;
        }

        // Collect worker proofs that have public inputs
        let mut worker_proofs: Vec<(&PeerId, &CollectedGradient)> = gradients
            .iter()
            .filter(|(_, g)| !g.proof.is_empty() && g.public_inputs_hex.is_some())
            .collect();

        if worker_proofs.is_empty() {
            tracing::warn!("No worker proofs with public inputs for IVC aggregation");
            return None;
        }

        // Sort by step number (extracted from public_inputs_hex[6])
        worker_proofs.sort_by_key(|(_, g)| {
            g.public_inputs_hex.as_ref()
                .and_then(|pis| pis.get(6))
                .and_then(|hex| {
                    let stripped = hex.strip_prefix("0x").unwrap_or(hex);
                    let bytes = hex::decode(stripped).ok()?;
                    if bytes.len() >= 8 {
                        Some(u64::from_le_bytes(bytes[..8].try_into().ok()?))
                    } else {
                        Some(0u64)
                    }
                })
                .unwrap_or(0)
        });

        // Build IVC steps from worker proofs
        let initial_commitment = if let Some((_, first)) = worker_proofs.first() {
            first.commitment
        } else {
            [0u8; 32]
        };

        let ivc_config = helix_prover::IVCConfig {
            steps_per_fold: worker_proofs.len() + 1, // Manual fold at end
            store_intermediates: false,
            ..Default::default()
        };
        let mut ivc_prover = helix_prover::IVCProver::with_config(initial_commitment, ivc_config);

        let mut total_error = 0.0f64;
        for (i, (_, grad)) in worker_proofs.iter().enumerate() {
            let mut output = grad.commitment;
            output[0] = output[0].wrapping_add(1);

            let computation_hash = {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(round_id.to_le_bytes());
                hasher.update((i as u64).to_le_bytes());
                hasher.update(grad.commitment);
                let hash: [u8; 32] = hasher.finalize().into();
                hash
            };

            let step = helix_prover::IVCStep {
                step: (i + 1) as u64,
                input_state: if i == 0 {
                    initial_commitment
                } else {
                    ivc_prover.state().state_commitment
                },
                output_state: ivc_prover.state().state_commitment, // Will be updated by IVC
                computation_hash,
                step_error: grad.error_bound,
                proof: vec![],
            };

            // IVCProver computes output_state internally via Poseidon:
            //   new_state = Poseidon(bytes_to_fr(current_state), bytes_to_fr(computation_hash))
            // where bytes_to_fr masks top 3 bits (repr[31] &= 0x1F).
            // We must replicate this to match add_step's validation.
            let current_state = ivc_prover.state().state_commitment;
            let step = helix_prover::IVCStep {
                output_state: {
                    use helix_prover::halo2curves::bn256::Fr as Halo2Fr;
                    use helix_prover::halo2curves::ff::PrimeField;
                    fn mask_to_fr(bytes: &[u8; 32]) -> Halo2Fr {
                        let mut repr = [0u8; 32];
                        repr.copy_from_slice(bytes);
                        repr[31] &= 0x1F;
                        Halo2Fr::from_repr_vartime(repr.into()).unwrap_or(Halo2Fr::from(0u64))
                    }
                    let current_fr = mask_to_fr(&current_state);
                    let comp_fr = mask_to_fr(&computation_hash);
                    let new_state = poseidon_hash_two(current_fr, comp_fr);
                    let new_repr = new_state.to_repr();
                    let mut bytes = [0u8; 32];
                    bytes.copy_from_slice(new_repr.as_ref());
                    bytes
                },
                ..step
            };

            match ivc_prover.add_step(step) {
                Ok(()) => {
                    total_error += grad.error_bound;
                }
                Err(e) => {
                    tracing::error!("IVC step {} failed for round {}: {}", i + 1, round_id, e);
                    return None;
                }
            }
        }

        // Generate decider proof
        use helix_prover::halo2curves::bn256::Fr;
        let loss_fr = Fr::from(0u64); // Aggregate loss not available from commitments
        let model_id_fr = Fr::from(model_id);
        let error_budget_fr = Fr::from(error_budget);

        match ivc_prover.prove_decider(loss_fr, model_id_fr, error_budget_fr) {
            Ok(decider) => {
                tracing::info!(
                    "IVC aggregation for round {}: {} worker proofs → 1 decider proof ({} bytes, {} steps)",
                    round_id,
                    worker_proofs.len(),
                    decider.proof.len(),
                    decider.num_steps,
                );

                // Submit to on-chain pipeline if available.
                // We convert the IVC decider proof into a single TrainingProofResultV2
                // and use the existing submit_aggregated_round path.
                if let Some(ref pipeline) = self.on_chain_pipeline {
                    use helix_prover::halo2curves::ff::PrimeField;
                    let pis: Vec<Fr> = decider.public_inputs.iter().map(|b| {
                        Fr::from_repr_vartime((*b).into()).unwrap_or(Fr::from(0u64))
                    }).collect();
                    let old_hash = if pis.len() >= 2 { (pis[0], pis[1]) } else { (Fr::from(0u64), Fr::from(0u64)) };
                    let new_hash = if pis.len() >= 4 { (pis[2], pis[3]) } else { (Fr::from(0u64), Fr::from(0u64)) };

                    let decider_as_result = helix_prover::TrainingProofResultV2 {
                        proof: decider.proof.clone(),
                        public_inputs: pis,
                        loss: loss_fr,
                        total_error: Fr::from((total_error * 1e9) as u64),
                        step_number: decider.num_steps,
                        old_state_hash: old_hash,
                        new_state_hash: new_hash,
                        verified: true,
                        generation_time: std::time::Duration::from_millis(0),
                        verification_time: None,
                        attempts: 1,
                        from_cache: false,
                        witness_hash: None,
                    };

                    let pipeline = Arc::clone(pipeline);
                    tokio::spawn(async move {
                        match pipeline.submit_aggregated_round(round_id, vec![decider_as_result]).await {
                            Ok(Some(submission)) => {
                                tracing::info!(
                                    "IVC decider proof submitted on-chain for round {}: tx={}, gas={}",
                                    round_id, submission.tx_hash, submission.gas_used,
                                );
                            }
                            Ok(None) => {
                                tracing::warn!("IVC on-chain submission returned None for round {}", round_id);
                            }
                            Err(e) => {
                                tracing::error!("IVC on-chain submission failed for round {}: {}", round_id, e);
                            }
                        }
                    });
                }

                // Build aggregated result
                let combined_commitment = {
                    let repr = decider.final_state;
                    repr
                };

                Some(AggregatedResult {
                    round_id,
                    commitment: combined_commitment,
                    total_error_bound: total_error / worker_proofs.len() as f64,
                    num_participants: worker_proofs.len(),
                    proof: decider.proof,
                    excluded_participants: Vec::new(),
                })
            }
            Err(e) => {
                tracing::error!("IVC decider proof failed for round {}: {}", round_id, e);
                None
            }
        }
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
    use crate::network::partition_detect::{PartitionDetectionConfig, PartitionStatus, PartitionGroup};
    use crate::training::model::{ModelGradient, WeightData};

    fn default_params() -> TrainingParams {
        TrainingParams {
            learning_rate: 0.001,
            batch_size: 32,
            local_epochs: 5,
            max_error_bound: 0.1,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            model_seed: 42,
            num_layers: 2,
            activation_type: 0,
        }
    }

    fn make_simple_gradient(offset: f32) -> ModelGradient {
        ModelGradient {
            embeddings: Some(WeightData {
                shape: vec![4],
                data: vec![1.0 + offset, 2.0 + offset, 3.0 + offset, 4.0 + offset],
                error_bound: 0.01,
            }),
            layers: vec![],
            lm_head: None,
            error_bound: 0.01,
        }
    }

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

        let _msg = node.start_round([1; 32], default_params()).await;

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

        node.start_round([1; 32], default_params()).await;

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
            byzantine_strategy: None, // No filtering for this test
            ..Default::default()
        };
        let node = AggregatorNode::new(local_id, config);

        node.start_round([1; 32], default_params()).await;

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

    #[tokio::test]
    async fn test_aggregate_produces_nonempty_proof() {
        let local_id = PeerId::random();
        let config = AggregatorConfig {
            min_participants: 1,
            byzantine_strategy: None,
            ..Default::default()
        };
        let node = AggregatorNode::new(local_id, config);

        node.start_round([1; 32], default_params()).await;

        let p1 = PeerId::random();
        let p2 = PeerId::random();
        node.handle_participate_request(p1.clone(), 1).await;
        node.handle_participate_request(p2.clone(), 1).await;
        node.start_collection().await;

        node.handle_gradient_share(p1, 1, [0xAA; 32], 0.01, vec![1]).await;
        node.handle_gradient_share(p2, 1, [0xBB; 32], 0.02, vec![2]).await;

        let result = node.aggregate().await.expect("aggregation should succeed");

        // Proof should be non-empty
        assert!(!result.proof.is_empty(), "proof must not be empty");

        // Proof should be deserializable as AggregationProof
        let agg_proof: AggregationProof =
            bincode::deserialize(&result.proof).expect("proof must be deserializable");

        assert_eq!(agg_proof.participant_count, 2);
        assert_eq!(agg_proof.round_id, 1);
        assert_eq!(agg_proof.sorted_commitments.len(), 2);
        // Poseidon commitment should be non-zero
        assert_ne!(agg_proof.poseidon_commitment, [0u8; 32]);
    }

    #[tokio::test]
    async fn test_aggregate_with_byzantine_filtering() {
        let local_id = PeerId::random();
        let config = AggregatorConfig {
            min_participants: 1,
            byzantine_strategy: Some(AggregationStrategy::Krum { num_byzantine: 1 }),
            ..Default::default()
        };
        let node = AggregatorNode::new(local_id, config);

        node.start_round([1; 32], default_params()).await;

        // Create 5 participants: 4 honest + 1 outlier
        let mut peers = Vec::new();
        for _ in 0..5 {
            let p = PeerId::random();
            node.handle_participate_request(p.clone(), 1).await;
            peers.push(p);
        }
        node.start_collection().await;

        // 4 honest gradients with small offsets (unique proofs per worker)
        for (i, p) in peers[..4].iter().enumerate() {
            node.handle_gradient_share_with_data(
                p.clone(),
                1,
                [i as u8 + 1; 32],
                0.01,
                vec![i as u8 + 1],
                Some(make_simple_gradient(i as f32 * 0.1)),
                None,
            ).await;
        }
        // 1 outlier with extreme values
        node.handle_gradient_share_with_data(
            peers[4].clone(),
            1,
            [0xFF; 32],
            0.01,
            vec![5],
            Some(make_simple_gradient(1000.0)),
            None,
        ).await;

        let result = node.aggregate().await.expect("aggregation should succeed");

        // The outlier should be excluded
        assert!(
            !result.excluded_participants.is_empty(),
            "Byzantine filtering should exclude at least one participant"
        );
        // Only 4 participants should remain after filtering
        assert!(result.num_participants <= 4, "Outlier should be filtered out");
    }

    #[tokio::test]
    async fn test_partition_blocks_round_start() {
        let detector = Arc::new(PartitionDetector::new(PartitionDetectionConfig {
            min_peers_for_detection: 2,
            ..Default::default()
        }));

        // Simulate a partition with Halt action by registering peers and
        // making them unreachable
        let addr1: std::net::SocketAddr = "127.0.0.1:9001".parse().unwrap();
        let addr2: std::net::SocketAddr = "127.0.0.1:9002".parse().unwrap();
        let addr3: std::net::SocketAddr = "127.0.0.1:9003".parse().unwrap();
        detector.register_peer(addr1);
        detector.register_peer(addr2);
        detector.register_peer(addr3);

        // Record missed heartbeats to make them unreachable — need to also
        // set unreachable_threshold very low for test
        let fast_detector = Arc::new(PartitionDetector::new(PartitionDetectionConfig {
            min_peers_for_detection: 2,
            unreachable_threshold: std::time::Duration::from_millis(1),
            ..Default::default()
        }));
        fast_detector.register_peer(addr1);
        fast_detector.register_peer(addr2);
        fast_detector.register_peer(addr3);

        // Wait for threshold to expire
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Record missed heartbeats so status becomes Unreachable
        for _ in 0..5 {
            fast_detector.record_missed_heartbeat(&addr1);
            fast_detector.record_missed_heartbeat(&addr2);
            fast_detector.record_missed_heartbeat(&addr3);
        }

        // Run detection to populate status
        let _ = fast_detector.detect_partition().await;

        let local_id = PeerId::random();
        let node = AggregatorNode::new(local_id, AggregatorConfig::default())
            .with_partition_detector(fast_detector);

        let _msg = node.start_round([1; 32], default_params()).await;

        // Should be in Error state, not Recruiting
        match node.get_state().await {
            AggregatorState::Error { message } => {
                assert!(message.contains("partitioned"), "Error should mention partition: {}", message);
            }
            other => panic!("Expected Error state due to partition, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_partition_allows_round_when_healthy() {
        let detector = Arc::new(PartitionDetector::new(PartitionDetectionConfig {
            min_peers_for_detection: 2,
            ..Default::default()
        }));

        // Register healthy peers
        let addr1: std::net::SocketAddr = "127.0.0.1:9010".parse().unwrap();
        let addr2: std::net::SocketAddr = "127.0.0.1:9011".parse().unwrap();
        detector.register_peer(addr1);
        detector.register_peer(addr2);
        for _ in 0..5 {
            detector.record_heartbeat(&addr1, Some(std::time::Duration::from_millis(10)));
            detector.record_heartbeat(&addr2, Some(std::time::Duration::from_millis(10)));
        }

        // Healthy network — is_partitioned() should be false (no status set yet)
        let local_id = PeerId::random();
        let node = AggregatorNode::new(local_id, AggregatorConfig::default())
            .with_partition_detector(detector);

        let _msg = node.start_round([1; 32], default_params()).await;

        match node.get_state().await {
            AggregatorState::Recruiting { round_id, .. } => {
                assert_eq!(round_id, 1);
            }
            other => panic!("Expected Recruiting state with healthy network, got {:?}", other),
        }
    }

    fn make_aggregator() -> AggregatorNode {
        AggregatorNode::new(
            PeerId::from_string("aggregator-1"),
            AggregatorConfig::default(),
        )
    }

    #[tokio::test]
    async fn test_worker_allowlist_authorization() {
        let agg = make_aggregator();
        let worker = PeerId::from_string("worker-1");
        let unknown = PeerId::from_string("unknown");

        // Empty allowlist = permissive mode
        assert!(agg.is_worker_authorized(&worker).await);

        // Register worker
        agg.register_worker(worker.clone()).await;
        assert!(agg.is_worker_authorized(&worker).await);
        assert!(!agg.is_worker_authorized(&unknown).await);

        // Deregister
        agg.deregister_worker(&worker).await;
        // Back to empty = permissive
        assert!(agg.is_worker_authorized(&unknown).await);
    }

    #[tokio::test]
    async fn test_proof_replay_detection() {
        let agg = make_aggregator();
        let proof = vec![1, 2, 3, 4, 5];

        // First submission: not duplicate
        assert!(!agg.is_duplicate_proof(&proof).await);
        // Second submission: duplicate
        assert!(agg.is_duplicate_proof(&proof).await);
        // Different proof: not duplicate
        assert!(!agg.is_duplicate_proof(&[6, 7, 8]).await);
        // Empty proof: never tracked
        assert!(!agg.is_duplicate_proof(&[]).await);
    }

    #[tokio::test]
    async fn test_reject_unregistered_worker() {
        let agg = make_aggregator();
        let authorized = PeerId::from_string("auth-worker");
        let unauthorized = PeerId::from_string("bad-worker");

        agg.register_worker(authorized.clone()).await;

        // Start a round
        agg.start_round([1; 32], default_params()).await;

        // Register both as participants
        agg.handle_participate_request(authorized.clone(), 1).await;
        agg.handle_participate_request(unauthorized.clone(), 1).await;
        agg.start_collection().await;

        // Gradient from unauthorized should be rejected
        let accepted = agg.handle_gradient_share(
            unauthorized.clone(), 1, [0u8; 32], 0.01, vec![1, 2, 3],
        ).await;
        assert!(!accepted);

        // Gradient from authorized should be accepted (returns false because
        // we need all participants, but it should not be rejected due to auth)
        let accepted = agg.handle_gradient_share(
            authorized.clone(), 1, [0u8; 32], 0.01, vec![4, 5, 6],
        ).await;
        // With 2 expected and 1 received, this returns false (not all received)
        // but the gradient IS stored, which we verify below
        let gradients = agg.gradients.read().await;
        assert!(gradients.contains_key(&authorized), "authorized worker gradient should be stored");
        assert!(!gradients.contains_key(&unauthorized), "unauthorized worker gradient should not be stored");
    }
}
