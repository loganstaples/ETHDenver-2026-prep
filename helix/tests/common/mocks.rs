//! Mock implementations for integration testing.
//!
//! Provides mock network, workers, and coordinators for testing without
//! real network infrastructure.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::{Mutex, RwLock};
use tokio::sync::mpsc;

use helix_circuits::halo2curves::bn256::Fr;
use helix_mpc::types::PartyId;

/// Simulated network message.
#[derive(Debug, Clone)]
pub enum NetworkMessage {
    /// Gradient share from a worker.
    GradientShare {
        from: PartyId,
        step: u64,
        gradients: Vec<f64>,
        commitment: [u8; 32],
    },
    /// Proof submission.
    ProofSubmission {
        from: PartyId,
        step: u64,
        proof: Vec<u8>,
        public_inputs: Vec<Fr>,
    },
    /// Verification result.
    VerificationResult {
        step: u64,
        valid: bool,
        message: Option<String>,
    },
    /// Checkpoint notification.
    Checkpoint {
        step: u64,
        weights_hash: [u8; 32],
    },
    /// Network partition simulation.
    PartitionStart {
        partitioned_nodes: Vec<PartyId>,
    },
    /// Network partition end.
    PartitionEnd,
    /// Heartbeat for liveness.
    Heartbeat {
        from: PartyId,
        timestamp: u64,
    },
}

/// Network partition state.
#[derive(Debug, Clone, Default)]
pub struct PartitionState {
    pub active: bool,
    pub partitioned_nodes: Vec<PartyId>,
}

/// Mock network for simulating distributed communication.
#[derive(Debug)]
pub struct MockNetwork {
    /// Message queue per party.
    queues: Arc<RwLock<HashMap<PartyId, Vec<NetworkMessage>>>>,
    /// Network latency simulation (milliseconds).
    latency_ms: u64,
    /// Packet loss probability (0.0 - 1.0).
    packet_loss: f64,
    /// Current partition state.
    partition: Arc<RwLock<PartitionState>>,
    /// Delivered message count.
    delivered_count: Arc<Mutex<u64>>,
    /// Dropped message count.
    dropped_count: Arc<Mutex<u64>>,
}

impl MockNetwork {
    /// Creates a new mock network.
    pub fn new() -> Self {
        Self {
            queues: Arc::new(RwLock::new(HashMap::new())),
            latency_ms: 10,
            packet_loss: 0.0,
            partition: Arc::new(RwLock::new(PartitionState::default())),
            delivered_count: Arc::new(Mutex::new(0)),
            dropped_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Sets network latency.
    pub fn with_latency(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Sets packet loss probability.
    pub fn with_packet_loss(mut self, prob: f64) -> Self {
        self.packet_loss = prob.clamp(0.0, 1.0);
        self
    }

    /// Registers a party with the network.
    pub fn register_party(&self, party: PartyId) {
        self.queues.write().insert(party, Vec::new());
    }

    /// Starts a network partition.
    pub fn start_partition(&self, partitioned_nodes: Vec<PartyId>) {
        let mut state = self.partition.write();
        state.active = true;
        state.partitioned_nodes = partitioned_nodes;
    }

    /// Ends the network partition.
    pub fn end_partition(&self) {
        let mut state = self.partition.write();
        state.active = false;
        state.partitioned_nodes.clear();
    }

    /// Checks if a party is partitioned.
    pub fn is_partitioned(&self, party: &PartyId) -> bool {
        let state = self.partition.read();
        state.active && state.partitioned_nodes.contains(party)
    }

    /// Sends a message to a party.
    pub fn send(&self, to: &PartyId, message: NetworkMessage) -> bool {
        // Check partition
        if self.is_partitioned(to) {
            *self.dropped_count.lock() += 1;
            return false;
        }

        // Simulate packet loss
        if self.packet_loss > 0.0 {
            let r: f64 = rand::random();
            if r < self.packet_loss {
                *self.dropped_count.lock() += 1;
                return false;
            }
        }

        // Deliver message
        if let Some(queue) = self.queues.write().get_mut(to) {
            queue.push(message);
            *self.delivered_count.lock() += 1;
            true
        } else {
            *self.dropped_count.lock() += 1;
            false
        }
    }

    /// Broadcasts a message to all parties except the sender.
    pub fn broadcast(&self, from: &PartyId, message: NetworkMessage) -> usize {
        let mut delivered = 0;
        let parties: Vec<PartyId> = self.queues.read().keys().cloned().collect();

        for party in parties {
            if &party != from && self.send(&party, message.clone()) {
                delivered += 1;
            }
        }

        delivered
    }

    /// Receives all pending messages for a party.
    pub fn receive(&self, party: &PartyId) -> Vec<NetworkMessage> {
        self.queues
            .write()
            .get_mut(party)
            .map(|q| std::mem::take(q))
            .unwrap_or_default()
    }

    /// Returns network statistics.
    pub fn stats(&self) -> NetworkStats {
        NetworkStats {
            delivered: *self.delivered_count.lock(),
            dropped: *self.dropped_count.lock(),
            partition_active: self.partition.read().active,
        }
    }
}

impl Default for MockNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Network statistics.
#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub delivered: u64,
    pub dropped: u64,
    pub partition_active: bool,
}

/// Mock worker that simulates gradient computation.
#[derive(Debug)]
pub struct MockWorker {
    pub party: PartyId,
    pub index: usize,
    /// Whether this worker is adversarial.
    pub adversarial: bool,
    /// Type of adversarial behavior.
    pub adversary_type: AdversaryType,
    /// Network reference.
    network: Arc<MockNetwork>,
    /// Completed steps.
    completed_steps: Arc<Mutex<Vec<u64>>>,
    /// Slashed flag.
    slashed: Arc<Mutex<bool>>,
}

/// Types of adversarial behavior for testing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdversaryType {
    /// Honest behavior.
    Honest,
    /// Submits random garbage gradients.
    RandomGradients,
    /// Submits zero gradients (lazy).
    ZeroGradients,
    /// Submits gradients with wrong commitment.
    WrongCommitment,
    /// Submits invalid proof.
    InvalidProof,
    /// Submits gradients exceeding norm bounds.
    LargeGradients,
    /// Delays submissions.
    Delayed,
    /// Stops responding.
    Unresponsive,
}

impl MockWorker {
    /// Creates a new honest mock worker.
    pub fn new(party: PartyId, index: usize, network: Arc<MockNetwork>) -> Self {
        network.register_party(party.clone());
        Self {
            party,
            index,
            adversarial: false,
            adversary_type: AdversaryType::Honest,
            network,
            completed_steps: Arc::new(Mutex::new(Vec::new())),
            slashed: Arc::new(Mutex::new(false)),
        }
    }

    /// Creates an adversarial worker.
    pub fn adversarial(
        party: PartyId,
        index: usize,
        network: Arc<MockNetwork>,
        adversary_type: AdversaryType,
    ) -> Self {
        network.register_party(party.clone());
        Self {
            party,
            index,
            adversarial: true,
            adversary_type,
            network,
            completed_steps: Arc::new(Mutex::new(Vec::new())),
            slashed: Arc::new(Mutex::new(false)),
        }
    }

    /// Simulates computing and submitting gradients.
    pub fn submit_gradients(&self, step: u64, true_gradients: &[f64]) -> GradientSubmission {
        if *self.slashed.lock() {
            return GradientSubmission {
                party: self.party.clone(),
                step,
                gradients: vec![],
                commitment: [0u8; 32],
                valid: false,
            };
        }

        let (gradients, commitment, valid) = match self.adversary_type {
            AdversaryType::Honest => {
                let grads = true_gradients.to_vec();
                let commit = self.compute_commitment(&grads);
                (grads, commit, true)
            }
            AdversaryType::RandomGradients => {
                let grads: Vec<f64> = (0..true_gradients.len())
                    .map(|_| rand::random::<f64>() * 1000.0)
                    .collect();
                let commit = self.compute_commitment(&grads);
                (grads, commit, false)
            }
            AdversaryType::ZeroGradients => {
                let grads = vec![0.0; true_gradients.len()];
                let commit = self.compute_commitment(&grads);
                (grads, commit, false)
            }
            AdversaryType::WrongCommitment => {
                let grads = true_gradients.to_vec();
                let commit = [0xDEu8; 32]; // Wrong commitment
                (grads, commit, false)
            }
            AdversaryType::InvalidProof => {
                let grads = true_gradients.to_vec();
                let commit = self.compute_commitment(&grads);
                (grads, commit, false) // Proof will fail
            }
            AdversaryType::LargeGradients => {
                let grads: Vec<f64> = true_gradients.iter().map(|&g| g * 1000.0).collect();
                let commit = self.compute_commitment(&grads);
                (grads, commit, false)
            }
            AdversaryType::Delayed | AdversaryType::Unresponsive => {
                // Don't submit anything
                return GradientSubmission {
                    party: self.party.clone(),
                    step,
                    gradients: vec![],
                    commitment: [0u8; 32],
                    valid: false,
                };
            }
        };

        self.completed_steps.lock().push(step);

        GradientSubmission {
            party: self.party.clone(),
            step,
            gradients,
            commitment,
            valid,
        }
    }

    /// Computes commitment hash for gradients.
    fn compute_commitment(&self, gradients: &[f64]) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for g in gradients {
            hasher.update(g.to_le_bytes());
        }
        hasher.finalize().into()
    }

    /// Marks this worker as slashed.
    pub fn slash(&self) {
        *self.slashed.lock() = true;
    }

    /// Checks if worker is slashed.
    pub fn is_slashed(&self) -> bool {
        *self.slashed.lock()
    }

    /// Returns completed step count.
    pub fn completed_step_count(&self) -> usize {
        self.completed_steps.lock().len()
    }
}

/// Gradient submission result.
#[derive(Debug, Clone)]
pub struct GradientSubmission {
    pub party: PartyId,
    pub step: u64,
    pub gradients: Vec<f64>,
    pub commitment: [u8; 32],
    pub valid: bool,
}

/// Mock coordinator for testing aggregation.
#[derive(Debug)]
pub struct MockCoordinator {
    /// Network reference.
    network: Arc<MockNetwork>,
    /// Current step.
    step: Arc<Mutex<u64>>,
    /// Registered workers.
    workers: Arc<RwLock<Vec<PartyId>>>,
    /// Slashed workers.
    slashed: Arc<RwLock<Vec<PartyId>>>,
    /// Aggregated gradients per step.
    aggregated: Arc<RwLock<HashMap<u64, Vec<f64>>>>,
    /// Timeout for gradient collection.
    timeout: Duration,
}

impl MockCoordinator {
    /// Creates a new mock coordinator.
    pub fn new(network: Arc<MockNetwork>) -> Self {
        let party = PartyId::from_index(999); // Coordinator has special ID
        network.register_party(party);

        Self {
            network,
            step: Arc::new(Mutex::new(0)),
            workers: Arc::new(RwLock::new(Vec::new())),
            slashed: Arc::new(RwLock::new(Vec::new())),
            aggregated: Arc::new(RwLock::new(HashMap::new())),
            timeout: Duration::from_secs(5),
        }
    }

    /// Registers a worker.
    pub fn register_worker(&self, party: PartyId) {
        self.workers.write().push(party);
    }

    /// Returns active (non-slashed) worker count.
    pub fn active_worker_count(&self) -> usize {
        let workers = self.workers.read();
        let slashed = self.slashed.read();
        workers.iter().filter(|w| !slashed.contains(w)).count()
    }

    /// Advances to the next step.
    pub fn advance_step(&self) -> u64 {
        let mut step = self.step.lock();
        *step += 1;
        *step
    }

    /// Slashes a worker.
    pub fn slash_worker(&self, party: &PartyId, reason: &str) {
        self.slashed.write().push(party.clone());
        tracing::warn!("Slashed worker {:?}: {}", party, reason);
    }

    /// Checks if a worker is slashed.
    pub fn is_slashed(&self, party: &PartyId) -> bool {
        self.slashed.read().contains(party)
    }

    /// Aggregates gradient submissions.
    pub fn aggregate_gradients(&self, submissions: &[GradientSubmission]) -> Option<Vec<f64>> {
        let valid_submissions: Vec<_> = submissions
            .iter()
            .filter(|s| s.valid && !self.is_slashed(&s.party))
            .collect();

        if valid_submissions.is_empty() {
            return None;
        }

        let len = valid_submissions[0].gradients.len();
        let mut aggregated = vec![0.0; len];

        for sub in &valid_submissions {
            for (i, g) in sub.gradients.iter().enumerate() {
                aggregated[i] += g;
            }
        }

        // Average
        for g in &mut aggregated {
            *g /= valid_submissions.len() as f64;
        }

        let step = *self.step.lock();
        self.aggregated.write().insert(step, aggregated.clone());

        Some(aggregated)
    }
}

/// Mock EVM verifier for testing on-chain verification.
#[derive(Debug)]
pub struct MockEVMVerifier {
    /// Verification results (proof hash -> result).
    results: Arc<RwLock<HashMap<[u8; 32], bool>>>,
    /// Gas used per verification.
    gas_used: Arc<Mutex<u64>>,
    /// Total verifications.
    verification_count: Arc<Mutex<u64>>,
}

impl MockEVMVerifier {
    pub fn new() -> Self {
        Self {
            results: Arc::new(RwLock::new(HashMap::new())),
            gas_used: Arc::new(Mutex::new(0)),
            verification_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Simulates EVM proof verification.
    pub fn verify(&self, proof: &[u8], public_inputs: &[Fr]) -> EVMVerificationResult {
        use sha2::{Digest, Sha256};

        // Compute proof hash
        let mut hasher = Sha256::new();
        hasher.update(proof);
        let proof_hash: [u8; 32] = hasher.finalize().into();

        // Simulate gas cost based on proof size
        let gas = 200_000 + (proof.len() as u64 * 100) + (public_inputs.len() as u64 * 10_000);
        *self.gas_used.lock() += gas;
        *self.verification_count.lock() += 1;

        // For testing, we check if proof is non-empty and has valid structure
        let valid = !proof.is_empty() && proof.len() >= 64 && public_inputs.len() >= 7;

        self.results.write().insert(proof_hash, valid);

        EVMVerificationResult {
            valid,
            gas_used: gas,
            proof_hash,
        }
    }

    /// Returns total gas used.
    pub fn total_gas_used(&self) -> u64 {
        *self.gas_used.lock()
    }

    /// Returns verification count.
    pub fn verification_count(&self) -> u64 {
        *self.verification_count.lock()
    }
}

impl Default for MockEVMVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// EVM verification result.
#[derive(Debug, Clone)]
pub struct EVMVerificationResult {
    pub valid: bool,
    pub gas_used: u64,
    pub proof_hash: [u8; 32],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_network_delivery() {
        let network = MockNetwork::new();
        let party1 = PartyId::from_index(0);
        let party2 = PartyId::from_index(1);

        network.register_party(party1.clone());
        network.register_party(party2.clone());

        let msg = NetworkMessage::Heartbeat {
            from: party1.clone(),
            timestamp: 123,
        };

        assert!(network.send(&party2, msg));

        let received = network.receive(&party2);
        assert_eq!(received.len(), 1);
    }

    #[test]
    fn test_mock_network_partition() {
        let network = MockNetwork::new();
        let party1 = PartyId::from_index(0);
        let party2 = PartyId::from_index(1);

        network.register_party(party1.clone());
        network.register_party(party2.clone());

        // Partition party2
        network.start_partition(vec![party2.clone()]);

        let msg = NetworkMessage::Heartbeat {
            from: party1.clone(),
            timestamp: 123,
        };

        assert!(!network.send(&party2, msg));
        assert!(network.is_partitioned(&party2));

        network.end_partition();
        assert!(!network.is_partitioned(&party2));
    }

    #[test]
    fn test_mock_worker_honest() {
        let network = Arc::new(MockNetwork::new());
        let worker = MockWorker::new(PartyId::from_index(0), 0, network);

        let gradients = vec![1.0, 2.0, 3.0];
        let submission = worker.submit_gradients(1, &gradients);

        assert!(submission.valid);
        assert_eq!(submission.gradients, gradients);
    }

    #[test]
    fn test_mock_worker_adversarial() {
        let network = Arc::new(MockNetwork::new());
        let worker = MockWorker::adversarial(
            PartyId::from_index(0),
            0,
            network,
            AdversaryType::RandomGradients,
        );

        let gradients = vec![1.0, 2.0, 3.0];
        let submission = worker.submit_gradients(1, &gradients);

        assert!(!submission.valid);
        assert_ne!(submission.gradients, gradients);
    }

    #[test]
    fn test_mock_coordinator_aggregation() {
        let network = Arc::new(MockNetwork::new());
        let coordinator = MockCoordinator::new(network);

        let submissions = vec![
            GradientSubmission {
                party: PartyId::from_index(0),
                step: 1,
                gradients: vec![1.0, 2.0],
                commitment: [0u8; 32],
                valid: true,
            },
            GradientSubmission {
                party: PartyId::from_index(1),
                step: 1,
                gradients: vec![3.0, 4.0],
                commitment: [0u8; 32],
                valid: true,
            },
        ];

        let aggregated = coordinator.aggregate_gradients(&submissions);
        assert!(aggregated.is_some());
        let agg = aggregated.unwrap();
        assert_eq!(agg, vec![2.0, 3.0]); // Average
    }
}
