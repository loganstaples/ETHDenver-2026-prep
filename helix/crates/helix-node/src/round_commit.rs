//! Distributed Round Commit Integration.
//!
//! This module provides integration between the distributed training coordinator
//! and on-chain round commits. It handles:
//! - Aggregating proofs from multiple workers
//! - Submitting combined round proofs to smart contracts
//! - Tracking on-chain round state
//! - Handling commit failures and retries
//!
//! The round commit system ensures that completed training rounds are properly
//! recorded on-chain with verifiable proofs.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ethers::types::{Address, H256, U256};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, oneshot};

use crate::network::messages::PeerId;
use crate::sc_client::{ModelState, RoundState, SCClient, TrainingProofInputs};
use crate::training::{
    DistributedRoundId, DistributedRoundState, GradientShare,
};

// ============================================================================
// Round Commit Types
// ============================================================================

/// Unique identifier for a round commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RoundCommitId {
    /// Session ID.
    pub session_id: u64,
    /// Round number.
    pub round_number: u64,
    /// Timestamp.
    pub timestamp: u64,
}

impl RoundCommitId {
    /// Creates a new round commit ID.
    pub fn new(session_id: u64, round_number: u64) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            session_id,
            round_number,
            timestamp,
        }
    }

    /// Creates from a distributed round ID.
    pub fn from_distributed_round(round_id: DistributedRoundId) -> Self {
        Self::new(round_id.session_id, round_id.round_number)
    }
}

impl std::fmt::Display for RoundCommitId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "commit-s{}-r{}", self.session_id, self.round_number)
    }
}

/// Status of a round commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundCommitStatus {
    /// Collecting proofs from workers.
    Collecting,
    /// Proofs collected, preparing submission.
    Preparing,
    /// Submitting to chain.
    Submitting,
    /// Waiting for confirmation.
    Confirming,
    /// Successfully committed.
    Committed,
    /// Commit failed.
    Failed,
    /// Commit cancelled.
    Cancelled,
}

/// Worker proof contribution for a round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerProof {
    /// Worker ID.
    pub worker_id: PeerId,
    /// Proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs.
    pub public_inputs: TrainingProofInputs,
    /// Error bound for this proof.
    pub error_bound: f64,
    /// Share index in MPC scheme.
    pub share_index: usize,
    /// Gradient commitment.
    pub gradient_commitment: [u8; 32],
    /// Timestamp of submission.
    pub submitted_at: u64,
}

/// Aggregated round proof ready for submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedRoundProof {
    /// Round commit ID.
    pub commit_id: RoundCommitId,
    /// Combined proof bytes.
    pub proof: Vec<u8>,
    /// Combined public inputs.
    pub public_inputs: TrainingProofInputs,
    /// Total error bound.
    pub total_error_bound: f64,
    /// Number of contributing workers.
    pub num_contributors: usize,
    /// Model commitment before round.
    pub old_commitment: [u8; 32],
    /// Model commitment after round.
    pub new_commitment: [u8; 32],
    /// Aggregated gradient commitment.
    pub gradient_commitment: [u8; 32],
    /// Contributing worker IDs.
    pub contributors: Vec<PeerId>,
}

/// Result of a round commit.
#[derive(Debug, Clone)]
pub struct RoundCommitResult {
    /// Commit ID.
    pub commit_id: RoundCommitId,
    /// Transaction hash.
    pub tx_hash: H256,
    /// Block number.
    pub block_number: u64,
    /// Gas used.
    pub gas_used: U256,
    /// New on-chain commitment.
    pub new_commitment: U256,
}

/// Events from round commit system.
#[derive(Debug, Clone)]
pub enum RoundCommitEvent {
    /// Proof collection started.
    CollectionStarted {
        commit_id: RoundCommitId,
        expected_proofs: usize,
    },
    /// Worker proof received.
    ProofReceived {
        commit_id: RoundCommitId,
        worker_id: PeerId,
        proofs_collected: usize,
    },
    /// Proof aggregation completed.
    ProofsAggregated {
        commit_id: RoundCommitId,
        num_proofs: usize,
    },
    /// Submission started.
    SubmissionStarted {
        commit_id: RoundCommitId,
    },
    /// Commit successful.
    CommitSuccessful {
        commit_id: RoundCommitId,
        tx_hash: H256,
        block_number: u64,
    },
    /// Commit failed.
    CommitFailed {
        commit_id: RoundCommitId,
        reason: String,
        retry_count: u32,
    },
    /// Commit confirmed on-chain.
    CommitConfirmed {
        commit_id: RoundCommitId,
        confirmations: u64,
    },
}

// ============================================================================
// Round Commit Configuration
// ============================================================================

/// Configuration for round commits.
#[derive(Debug, Clone)]
pub struct RoundCommitConfig {
    /// Model ID on-chain.
    pub model_id: u64,
    /// Minimum proofs required for commit.
    pub min_proofs: usize,
    /// Timeout for proof collection.
    pub collection_timeout: Duration,
    /// Maximum retries for failed submissions.
    pub max_retries: u32,
    /// Delay between retries.
    pub retry_delay: Duration,
    /// Required confirmations before considering commit final.
    pub required_confirmations: u64,
    /// Whether to aggregate proofs (vs submit individually).
    pub aggregate_proofs: bool,
    /// Gas limit for submission.
    pub gas_limit: Option<U256>,
    /// Gas price multiplier (for faster confirmation).
    pub gas_price_multiplier: f64,
}

impl Default for RoundCommitConfig {
    fn default() -> Self {
        Self {
            model_id: 0,
            min_proofs: 2,
            collection_timeout: Duration::from_secs(120),
            max_retries: 3,
            retry_delay: Duration::from_secs(10),
            required_confirmations: 2,
            aggregate_proofs: true,
            gas_limit: None,
            gas_price_multiplier: 1.1,
        }
    }
}

// ============================================================================
// Proof Collector
// ============================================================================

/// Collects proofs from workers for a round.
pub struct ProofCollector {
    /// Commit ID.
    commit_id: RoundCommitId,
    /// Required workers.
    required_workers: Vec<PeerId>,
    /// Collected proofs.
    proofs: HashMap<PeerId, WorkerProof>,
    /// Minimum proofs required.
    min_proofs: usize,
    /// Collection started at.
    started_at: Instant,
    /// Timeout.
    timeout: Duration,
    /// Status.
    status: RoundCommitStatus,
}

impl ProofCollector {
    /// Creates a new proof collector.
    pub fn new(
        commit_id: RoundCommitId,
        workers: Vec<PeerId>,
        min_proofs: usize,
        timeout: Duration,
    ) -> Self {
        Self {
            commit_id,
            required_workers: workers,
            proofs: HashMap::new(),
            min_proofs,
            started_at: Instant::now(),
            timeout,
            status: RoundCommitStatus::Collecting,
        }
    }

    /// Adds a proof from a worker.
    pub fn add_proof(&mut self, proof: WorkerProof) -> Result<(), RoundCommitError> {
        if self.status != RoundCommitStatus::Collecting {
            return Err(RoundCommitError::NotCollecting);
        }

        if !self.required_workers.contains(&proof.worker_id) {
            return Err(RoundCommitError::UnexpectedWorker(proof.worker_id));
        }

        if self.proofs.contains_key(&proof.worker_id) {
            return Err(RoundCommitError::DuplicateProof(proof.worker_id));
        }

        self.proofs.insert(proof.worker_id, proof);
        Ok(())
    }

    /// Checks if enough proofs have been collected.
    pub fn has_enough_proofs(&self) -> bool {
        self.proofs.len() >= self.min_proofs
    }

    /// Checks if all proofs have been collected.
    pub fn is_complete(&self) -> bool {
        self.required_workers
            .iter()
            .all(|w| self.proofs.contains_key(w))
    }

    /// Checks if collection has timed out.
    pub fn is_timed_out(&self) -> bool {
        self.started_at.elapsed() > self.timeout
    }

    /// Returns missing workers.
    pub fn missing_workers(&self) -> Vec<PeerId> {
        self.required_workers
            .iter()
            .filter(|w| !self.proofs.contains_key(w))
            .cloned()
            .collect()
    }

    /// Returns collected proofs.
    pub fn proofs(&self) -> &HashMap<PeerId, WorkerProof> {
        &self.proofs
    }

    /// Takes all collected proofs.
    pub fn take_proofs(self) -> HashMap<PeerId, WorkerProof> {
        self.proofs
    }

    /// Finalizes collection and returns status.
    pub fn finalize(&mut self) -> RoundCommitStatus {
        if self.has_enough_proofs() {
            self.status = RoundCommitStatus::Preparing;
        } else if self.is_timed_out() {
            self.status = RoundCommitStatus::Failed;
        }
        self.status
    }
}

// ============================================================================
// Proof Aggregator
// ============================================================================

/// Aggregates proofs from multiple workers into a single submission.
pub struct ProofAggregator;

impl ProofAggregator {
    /// Aggregates multiple worker proofs into a single round proof.
    pub fn aggregate(
        commit_id: RoundCommitId,
        proofs: HashMap<PeerId, WorkerProof>,
        old_commitment: [u8; 32],
        new_commitment: [u8; 32],
    ) -> Result<AggregatedRoundProof, RoundCommitError> {
        if proofs.is_empty() {
            return Err(RoundCommitError::NoProofs);
        }

        let contributors: Vec<PeerId> = proofs.keys().cloned().collect();
        let num_contributors = contributors.len();

        // Calculate total error bound
        let total_error_bound: f64 = proofs.values()
            .map(|p| p.error_bound)
            .sum();

        // Combine proof bytes (simplified - in production use proper aggregation)
        let mut combined_proof = Vec::new();
        for proof in proofs.values() {
            combined_proof.extend(&proof.proof);
        }

        // Take the first proof's public inputs as base (in production, combine properly)
        let first_proof = proofs.values().next()
            .ok_or(RoundCommitError::NoProofs)?;

        // Compute gradient commitment (XOR of all commitments for simplicity)
        let gradient_commitment = proofs.values()
            .fold([0u8; 32], |acc, p| {
                let mut result = [0u8; 32];
                for i in 0..32 {
                    result[i] = acc[i] ^ p.gradient_commitment[i];
                }
                result
            });

        Ok(AggregatedRoundProof {
            commit_id,
            proof: combined_proof,
            public_inputs: first_proof.public_inputs.clone(),
            total_error_bound,
            num_contributors,
            old_commitment,
            new_commitment,
            gradient_commitment,
            contributors,
        })
    }
}

// ============================================================================
// Round Commit Manager
// ============================================================================

/// Manages round commits for distributed training.
pub struct RoundCommitManager {
    /// Configuration.
    config: RoundCommitConfig,
    /// Smart contract client.
    sc_client: Option<Arc<SCClient>>,
    /// Active proof collectors.
    collectors: HashMap<RoundCommitId, ProofCollector>,
    /// Pending submissions.
    pending_submissions: HashMap<RoundCommitId, AggregatedRoundProof>,
    /// Completed commits.
    completed_commits: Vec<RoundCommitResult>,
    /// Event broadcaster.
    event_tx: broadcast::Sender<RoundCommitEvent>,
    /// Current on-chain model state.
    model_state: Option<ModelState>,
}

impl RoundCommitManager {
    /// Creates a new round commit manager.
    pub fn new(config: RoundCommitConfig) -> Self {
        let (event_tx, _) = broadcast::channel(64);

        Self {
            config,
            sc_client: None,
            collectors: HashMap::new(),
            pending_submissions: HashMap::new(),
            completed_commits: Vec::new(),
            event_tx,
            model_state: None,
        }
    }

    /// Creates a new round commit manager with a smart contract client.
    pub fn with_client(config: RoundCommitConfig, client: Arc<SCClient>) -> Self {
        let (event_tx, _) = broadcast::channel(64);

        Self {
            config,
            sc_client: Some(client),
            collectors: HashMap::new(),
            pending_submissions: HashMap::new(),
            completed_commits: Vec::new(),
            event_tx,
            model_state: None,
        }
    }

    /// Sets the smart contract client.
    pub fn set_client(&mut self, client: Arc<SCClient>) {
        self.sc_client = Some(client);
    }

    /// Subscribes to commit events.
    pub fn subscribe(&self) -> broadcast::Receiver<RoundCommitEvent> {
        self.event_tx.subscribe()
    }

    /// Refreshes on-chain model state.
    pub async fn refresh_model_state(&mut self) -> Result<ModelState, RoundCommitError> {
        let client = self.sc_client.as_ref()
            .ok_or(RoundCommitError::NoClient)?;

        let state = client.get_model_state(self.config.model_id).await
            .map_err(|e| RoundCommitError::ChainError(e.to_string()))?;

        self.model_state = Some(state.clone());
        Ok(state)
    }

    /// Starts proof collection for a round.
    pub fn start_collection(
        &mut self,
        round_id: DistributedRoundId,
        workers: Vec<PeerId>,
    ) -> Result<RoundCommitId, RoundCommitError> {
        let commit_id = RoundCommitId::from_distributed_round(round_id);

        if self.collectors.contains_key(&commit_id) {
            return Err(RoundCommitError::CollectionAlreadyStarted(commit_id));
        }

        let collector = ProofCollector::new(
            commit_id,
            workers.clone(),
            self.config.min_proofs,
            self.config.collection_timeout,
        );

        self.collectors.insert(commit_id, collector);

        let _ = self.event_tx.send(RoundCommitEvent::CollectionStarted {
            commit_id,
            expected_proofs: workers.len(),
        });

        Ok(commit_id)
    }

    /// Submits a proof from a worker.
    pub fn submit_proof(
        &mut self,
        commit_id: RoundCommitId,
        proof: WorkerProof,
    ) -> Result<(), RoundCommitError> {
        let collector = self.collectors.get_mut(&commit_id)
            .ok_or(RoundCommitError::CollectionNotFound(commit_id))?;

        let worker_id = proof.worker_id;
        collector.add_proof(proof)?;

        let proofs_collected = collector.proofs().len();

        let _ = self.event_tx.send(RoundCommitEvent::ProofReceived {
            commit_id,
            worker_id,
            proofs_collected,
        });

        Ok(())
    }

    /// Finalizes collection and prepares for submission.
    pub fn finalize_collection(
        &mut self,
        commit_id: RoundCommitId,
        old_commitment: [u8; 32],
        new_commitment: [u8; 32],
    ) -> Result<AggregatedRoundProof, RoundCommitError> {
        let mut collector = self.collectors.remove(&commit_id)
            .ok_or(RoundCommitError::CollectionNotFound(commit_id))?;

        let status = collector.finalize();
        if status == RoundCommitStatus::Failed {
            return Err(RoundCommitError::InsufficientProofs {
                required: self.config.min_proofs,
                collected: collector.proofs().len(),
            });
        }

        let proofs = collector.take_proofs();
        let num_proofs = proofs.len();

        let aggregated = ProofAggregator::aggregate(
            commit_id,
            proofs,
            old_commitment,
            new_commitment,
        )?;

        let _ = self.event_tx.send(RoundCommitEvent::ProofsAggregated {
            commit_id,
            num_proofs,
        });

        // Store for submission
        self.pending_submissions.insert(commit_id, aggregated.clone());

        Ok(aggregated)
    }

    /// Submits an aggregated proof to the smart contract.
    pub async fn submit_to_chain(
        &mut self,
        commit_id: RoundCommitId,
    ) -> Result<RoundCommitResult, RoundCommitError> {
        let client = self.sc_client.as_ref()
            .ok_or(RoundCommitError::NoClient)?;

        let aggregated = self.pending_submissions.get(&commit_id)
            .ok_or(RoundCommitError::NoPendingSubmission(commit_id))?
            .clone();

        let _ = self.event_tx.send(RoundCommitEvent::SubmissionStarted { commit_id });

        let mut retry_count = 0;
        let result = loop {
            match client.submit_proof_raw(
                self.config.model_id,
                commit_id.round_number,
                aggregated.proof.clone(),
                aggregated.public_inputs.to_vec(),
            ).await {
                Ok(receipt) => {
                    let result = RoundCommitResult {
                        commit_id,
                        tx_hash: receipt.transaction_hash,
                        block_number: receipt.block_number.unwrap_or_default().as_u64(),
                        gas_used: receipt.gas_used.unwrap_or_default(),
                        new_commitment: aggregated.public_inputs.new_hash_lo,
                    };

                    let _ = self.event_tx.send(RoundCommitEvent::CommitSuccessful {
                        commit_id,
                        tx_hash: result.tx_hash,
                        block_number: result.block_number,
                    });

                    break Ok(result);
                }
                Err(e) => {
                    retry_count += 1;

                    let _ = self.event_tx.send(RoundCommitEvent::CommitFailed {
                        commit_id,
                        reason: e.to_string(),
                        retry_count,
                    });

                    if retry_count >= self.config.max_retries {
                        break Err(RoundCommitError::SubmissionFailed(e.to_string()));
                    }

                    tokio::time::sleep(self.config.retry_delay).await;
                }
            }
        };

        if let Ok(ref res) = result {
            self.pending_submissions.remove(&commit_id);
            self.completed_commits.push(res.clone());
        }

        result
    }

    /// Gets the status of a commit.
    pub fn get_status(&self, commit_id: RoundCommitId) -> Option<RoundCommitStatus> {
        if self.completed_commits.iter().any(|c| c.commit_id == commit_id) {
            return Some(RoundCommitStatus::Committed);
        }

        if self.pending_submissions.contains_key(&commit_id) {
            return Some(RoundCommitStatus::Preparing);
        }

        if let Some(collector) = self.collectors.get(&commit_id) {
            return Some(collector.status);
        }

        None
    }

    /// Returns completed commits.
    pub fn completed_commits(&self) -> &[RoundCommitResult] {
        &self.completed_commits
    }

    /// Cancels an in-progress collection.
    pub fn cancel_collection(
        &mut self,
        commit_id: RoundCommitId,
    ) -> Result<(), RoundCommitError> {
        self.collectors.remove(&commit_id);
        self.pending_submissions.remove(&commit_id);
        Ok(())
    }
}

// ============================================================================
// Round Commit Coordinator (High-Level Interface)
// ============================================================================

/// High-level coordinator for round commits in distributed training.
pub struct RoundCommitCoordinator {
    /// Commit manager.
    manager: Arc<RwLock<RoundCommitManager>>,
    /// This worker's ID.
    worker_id: PeerId,
    /// Whether this worker is the commit leader.
    is_leader: bool,
}

impl RoundCommitCoordinator {
    /// Creates a new round commit coordinator.
    pub fn new(
        config: RoundCommitConfig,
        worker_id: PeerId,
    ) -> Self {
        let manager = Arc::new(RwLock::new(RoundCommitManager::new(config)));

        Self {
            manager,
            worker_id,
            is_leader: false,
        }
    }

    /// Creates a coordinator with a smart contract client.
    pub fn with_client(
        config: RoundCommitConfig,
        worker_id: PeerId,
        client: Arc<SCClient>,
    ) -> Self {
        let manager = Arc::new(RwLock::new(
            RoundCommitManager::with_client(config, client)
        ));

        Self {
            manager,
            worker_id,
            is_leader: false,
        }
    }

    /// Sets whether this worker is the commit leader.
    pub fn set_leader(&mut self, is_leader: bool) {
        self.is_leader = is_leader;
    }

    /// Returns whether this worker is the commit leader.
    pub fn is_leader(&self) -> bool {
        self.is_leader
    }

    /// Subscribes to commit events.
    pub fn subscribe(&self) -> broadcast::Receiver<RoundCommitEvent> {
        self.manager.read().subscribe()
    }

    /// Creates a worker proof from round results.
    pub fn create_proof(
        &self,
        proof_bytes: Vec<u8>,
        old_commitment: [u8; 32],
        new_commitment: [u8; 32],
        error_bound: f64,
        share_index: usize,
        gradient_commitment: [u8; 32],
    ) -> WorkerProof {
        WorkerProof {
            worker_id: self.worker_id,
            proof: proof_bytes,
            public_inputs: TrainingProofInputs {
                old_hash_lo: U256::from_big_endian(&old_commitment[..16]),
                old_hash_hi: U256::from_big_endian(&old_commitment[16..]),
                new_hash_lo: U256::from_big_endian(&new_commitment[..16]),
                new_hash_hi: U256::from_big_endian(&new_commitment[16..]),
                loss: U256::zero(),
                error_bound: U256::from((error_bound * 1e18) as u64),
                step_number: U256::from(share_index),
            },
            error_bound,
            share_index,
            gradient_commitment,
            submitted_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Starts a round commit (leader only).
    pub fn start_round_commit(
        &self,
        round_id: DistributedRoundId,
        workers: Vec<PeerId>,
    ) -> Result<RoundCommitId, RoundCommitError> {
        if !self.is_leader {
            return Err(RoundCommitError::NotLeader);
        }

        self.manager.write().start_collection(round_id, workers)
    }

    /// Submits this worker's proof.
    pub fn submit_proof(
        &self,
        commit_id: RoundCommitId,
        proof: WorkerProof,
    ) -> Result<(), RoundCommitError> {
        self.manager.write().submit_proof(commit_id, proof)
    }

    /// Finalizes and submits to chain (leader only).
    pub async fn finalize_and_submit(
        &self,
        commit_id: RoundCommitId,
        old_commitment: [u8; 32],
        new_commitment: [u8; 32],
    ) -> Result<RoundCommitResult, RoundCommitError> {
        if !self.is_leader {
            return Err(RoundCommitError::NotLeader);
        }

        // Finalize collection
        {
            let mut manager = self.manager.write();
            manager.finalize_collection(commit_id, old_commitment, new_commitment)?;
        }

        // Submit to chain
        self.manager.write().submit_to_chain(commit_id).await
    }

    /// Gets commit status.
    pub fn get_status(&self, commit_id: RoundCommitId) -> Option<RoundCommitStatus> {
        self.manager.read().get_status(commit_id)
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Round commit errors.
#[derive(Debug)]
pub enum RoundCommitError {
    /// No smart contract client configured.
    NoClient,
    /// Not the commit leader.
    NotLeader,
    /// Collection not in progress.
    NotCollecting,
    /// Collection already started.
    CollectionAlreadyStarted(RoundCommitId),
    /// Collection not found.
    CollectionNotFound(RoundCommitId),
    /// Unexpected worker.
    UnexpectedWorker(PeerId),
    /// Duplicate proof from worker.
    DuplicateProof(PeerId),
    /// No proofs collected.
    NoProofs,
    /// Insufficient proofs.
    InsufficientProofs { required: usize, collected: usize },
    /// No pending submission.
    NoPendingSubmission(RoundCommitId),
    /// Submission failed.
    SubmissionFailed(String),
    /// Chain error.
    ChainError(String),
    /// Timeout.
    Timeout,
}

impl std::fmt::Display for RoundCommitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoClient => write!(f, "No smart contract client configured"),
            Self::NotLeader => write!(f, "Not the commit leader"),
            Self::NotCollecting => write!(f, "Collection not in progress"),
            Self::CollectionAlreadyStarted(id) => write!(f, "Collection already started: {}", id),
            Self::CollectionNotFound(id) => write!(f, "Collection not found: {}", id),
            Self::UnexpectedWorker(id) => write!(f, "Unexpected worker: {}", id),
            Self::DuplicateProof(id) => write!(f, "Duplicate proof from: {}", id),
            Self::NoProofs => write!(f, "No proofs collected"),
            Self::InsufficientProofs { required, collected } => {
                write!(f, "Insufficient proofs: need {}, have {}", required, collected)
            }
            Self::NoPendingSubmission(id) => write!(f, "No pending submission: {}", id),
            Self::SubmissionFailed(msg) => write!(f, "Submission failed: {}", msg),
            Self::ChainError(msg) => write!(f, "Chain error: {}", msg),
            Self::Timeout => write!(f, "Timeout"),
        }
    }
}

impl std::error::Error for RoundCommitError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_peer_id(id: u8) -> PeerId {
        let mut bytes = [0u8; 32];
        bytes[0] = id;
        PeerId(bytes)
    }

    fn create_test_config() -> RoundCommitConfig {
        RoundCommitConfig {
            model_id: 1,
            min_proofs: 2,
            collection_timeout: Duration::from_secs(60),
            ..Default::default()
        }
    }

    fn create_test_proof(worker_id: PeerId, share_index: usize) -> WorkerProof {
        WorkerProof {
            worker_id,
            proof: vec![1, 2, 3, 4],
            public_inputs: TrainingProofInputs {
                old_hash_lo: U256::from(1),
                old_hash_hi: U256::from(2),
                new_hash_lo: U256::from(3),
                new_hash_hi: U256::from(4),
                loss: U256::from(100),
                error_bound: U256::from(10),
                step_number: U256::from(share_index),
            },
            error_bound: 0.01,
            share_index,
            gradient_commitment: [0u8; 32],
            submitted_at: 12345,
        }
    }

    #[test]
    fn test_round_commit_id() {
        let id = RoundCommitId::new(1, 5);
        assert_eq!(id.session_id, 1);
        assert_eq!(id.round_number, 5);
        assert!(id.timestamp > 0);

        let display = format!("{}", id);
        assert!(display.starts_with("commit-s1-r5"));
    }

    #[test]
    fn test_proof_collector() {
        let commit_id = RoundCommitId::new(1, 1);
        let workers = vec![
            create_test_peer_id(1),
            create_test_peer_id(2),
            create_test_peer_id(3),
        ];

        let mut collector = ProofCollector::new(
            commit_id,
            workers,
            2,
            Duration::from_secs(60),
        );

        assert!(!collector.has_enough_proofs());
        assert!(!collector.is_complete());
        assert_eq!(collector.missing_workers().len(), 3);

        // Add first proof
        let proof1 = create_test_proof(create_test_peer_id(1), 0);
        collector.add_proof(proof1).unwrap();

        assert!(!collector.has_enough_proofs());
        assert_eq!(collector.proofs().len(), 1);

        // Add second proof
        let proof2 = create_test_proof(create_test_peer_id(2), 1);
        collector.add_proof(proof2).unwrap();

        assert!(collector.has_enough_proofs());
        assert!(!collector.is_complete());

        // Add third proof
        let proof3 = create_test_proof(create_test_peer_id(3), 2);
        collector.add_proof(proof3).unwrap();

        assert!(collector.is_complete());
        assert!(collector.missing_workers().is_empty());
    }

    #[test]
    fn test_proof_collector_duplicate() {
        let commit_id = RoundCommitId::new(1, 1);
        let workers = vec![create_test_peer_id(1), create_test_peer_id(2)];

        let mut collector = ProofCollector::new(
            commit_id,
            workers,
            2,
            Duration::from_secs(60),
        );

        let proof = create_test_proof(create_test_peer_id(1), 0);
        collector.add_proof(proof.clone()).unwrap();

        // Duplicate should fail
        let result = collector.add_proof(proof);
        assert!(matches!(result, Err(RoundCommitError::DuplicateProof(_))));
    }

    #[test]
    fn test_proof_collector_unexpected_worker() {
        let commit_id = RoundCommitId::new(1, 1);
        let workers = vec![create_test_peer_id(1), create_test_peer_id(2)];

        let mut collector = ProofCollector::new(
            commit_id,
            workers,
            2,
            Duration::from_secs(60),
        );

        // Worker 3 is not expected
        let proof = create_test_proof(create_test_peer_id(3), 0);
        let result = collector.add_proof(proof);
        assert!(matches!(result, Err(RoundCommitError::UnexpectedWorker(_))));
    }

    #[test]
    fn test_proof_aggregator() {
        let commit_id = RoundCommitId::new(1, 1);
        let mut proofs = HashMap::new();

        proofs.insert(
            create_test_peer_id(1),
            create_test_proof(create_test_peer_id(1), 0),
        );
        proofs.insert(
            create_test_peer_id(2),
            create_test_proof(create_test_peer_id(2), 1),
        );

        let old_commitment = [1u8; 32];
        let new_commitment = [2u8; 32];

        let aggregated = ProofAggregator::aggregate(
            commit_id,
            proofs,
            old_commitment,
            new_commitment,
        ).unwrap();

        assert_eq!(aggregated.commit_id, commit_id);
        assert_eq!(aggregated.num_contributors, 2);
        assert_eq!(aggregated.old_commitment, old_commitment);
        assert_eq!(aggregated.new_commitment, new_commitment);
        assert!(!aggregated.proof.is_empty());
    }

    #[test]
    fn test_round_commit_manager() {
        let config = create_test_config();
        let mut manager = RoundCommitManager::new(config);

        let round_id = DistributedRoundId {
            session_id: 1,
            round_number: 1,
        };
        let workers = vec![
            create_test_peer_id(1),
            create_test_peer_id(2),
        ];

        // Start collection
        let commit_id = manager.start_collection(round_id, workers).unwrap();

        assert!(manager.get_status(commit_id).is_some());
        assert_eq!(
            manager.get_status(commit_id),
            Some(RoundCommitStatus::Collecting)
        );

        // Submit proofs
        let proof1 = create_test_proof(create_test_peer_id(1), 0);
        manager.submit_proof(commit_id, proof1).unwrap();

        let proof2 = create_test_proof(create_test_peer_id(2), 1);
        manager.submit_proof(commit_id, proof2).unwrap();

        // Finalize
        let old_commitment = [1u8; 32];
        let new_commitment = [2u8; 32];
        let aggregated = manager.finalize_collection(
            commit_id,
            old_commitment,
            new_commitment,
        ).unwrap();

        assert_eq!(aggregated.num_contributors, 2);
        assert_eq!(
            manager.get_status(commit_id),
            Some(RoundCommitStatus::Preparing)
        );
    }

    #[test]
    fn test_round_commit_coordinator_create_proof() {
        let config = create_test_config();
        let worker_id = create_test_peer_id(1);
        let coordinator = RoundCommitCoordinator::new(config, worker_id);

        let proof = coordinator.create_proof(
            vec![1, 2, 3, 4],
            [1u8; 32],
            [2u8; 32],
            0.01,
            0,
            [3u8; 32],
        );

        assert_eq!(proof.worker_id, worker_id);
        assert_eq!(proof.share_index, 0);
        assert_eq!(proof.error_bound, 0.01);
    }

    #[test]
    fn test_round_commit_error_display() {
        let err = RoundCommitError::InsufficientProofs {
            required: 3,
            collected: 1,
        };
        assert_eq!(
            format!("{}", err),
            "Insufficient proofs: need 3, have 1"
        );

        let err = RoundCommitError::NotLeader;
        assert_eq!(format!("{}", err), "Not the commit leader");
    }
}
