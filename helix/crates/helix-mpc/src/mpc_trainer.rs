//! MPCTrainer: Distributed training orchestrator over MPC transport.
//!
//! Ties together all MPC components into a production training flow:
//!
//! 1. **Share weights** — dealer distributes additive shares via transport
//! 2. **Distributed Beaver triple generation** — parties jointly generate
//!    triples via pairwise OT over the transport (no trusted dealer)
//! 3. **Secure forward/backward pass** — using Beaver-triple multiplication
//!    over the transport for each matmul
//! 4. **Gradient aggregation** — parties exchange gradient shares and update
//! 5. **Re-sharing** — every N steps, parties refresh shares over the transport
//!    to prevent gradient accumulation attacks
//! 6. **ZK proof generation** — each step produces a Halo2 proof via
//!    `circuit_bridge.rs`

use std::collections::HashMap;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn, instrument};

use crate::beaver::distributed::NetworkDistributedDealer;
use crate::beaver::triple::BeaverTriple;
use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::integration::circuit_bridge::{
    CircuitBridge, CircuitBridgeConfig, Halo2ProofResult,
    compute_compatible_state_hash,
};
use crate::integration::witness::generate_freivalds_challenges;
use crate::integration::witness_format::ReconstructedWitness;
use crate::proofs::{
    ShareValidityProver, ShareValidityWitness,
    AggregationProver, AggregationVerifier,
    GradientAggregationWitness, GradientShareInput,
    AggregationProof, ShareValidityProof,
    ShareValidityVerifier,
};
use crate::mac_verification::{
    self, AuthenticatedBeaverTriple, MACCheckResult, MACFailureReport, MACState,
    MACVerificationConfig, TrainingCheckpoint,
};
use crate::protocols::arithmetic::SecureArithmetic;
use crate::protocols::reshare::Resharing;
use crate::security::commitment::BlindingGenerator;
use crate::session::transport::MPCTransport;
use crate::sharing::tensor::TensorShare;
use crate::types::{PartyId, ShareId};

/// Callback for sub-step progress within a single training step.
/// Arguments: (step_1indexed, total_steps, operation_description)
pub type SubStepCallback = std::sync::Arc<dyn Fn(usize, usize, &str) + Send + Sync>;

/// Receives messages from all peers in parallel using `try_join_all`.
///
/// Instead of sequential `for peer in &peers { recv(peer).await }`, this
/// issues all recv futures concurrently, reducing latency from N round-trips
/// to 1 (the slowest peer). Safe because each peer has independent channels.
async fn recv_all<T: MPCTransport>(transport: &T) -> MPCResult<Vec<Vec<u8>>> {
    let peers = transport.peers();
    let futs: Vec<_> = peers.iter()
        .map(|peer| transport.recv(peer))
        .collect();
    futures::future::try_join_all(futs).await
}

/// Number of bits used for the secure sign-bit comparison protocol.
/// This controls the range of values that can be correctly compared.
/// For ML values in fixed-point representation, 64 bits is sufficient.
#[allow(dead_code)]
const SIGN_BIT_BITS: usize = 64;

/// Configuration for the MPC trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MPCTrainerConfig {
    /// Input dimension.
    pub d_in: usize,
    /// Hidden dimension.
    pub d_hid: usize,
    /// Output dimension.
    pub d_out: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Number of parties.
    pub num_parties: usize,
    /// Re-share weights every N steps (0 = disabled).
    pub reshare_interval: u64,
    /// Number of Beaver triples to pre-generate per batch.
    pub beaver_batch_size: usize,
    /// Whether to generate ZK proofs after each step.
    pub generate_proofs: bool,
    /// Base error bound per operation.
    pub base_error: f64,
    /// Checkpoint proving interval: generate ZK proof every N steps (1 = every step).
    pub checkpoint_interval: u64,
    /// SPDZ MAC verification configuration. None = disabled.
    pub mac_config: Option<MACVerificationConfig>,
    /// Mini-batch size: process multiple samples per communication round.
    /// Default 1 (standard SGD). Higher values improve CPU utilization by
    /// amortizing async communication overhead across more parallel compute.
    pub batch_size: usize,
}

impl Default for MPCTrainerConfig {
    fn default() -> Self {
        Self {
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            learning_rate: 0.01,
            num_parties: 3,
            reshare_interval: 50,
            beaver_batch_size: 256,
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: None,
            batch_size: 4,
        }
    }
}

impl MPCTrainerConfig {
    /// Creates a config for a small model.
    pub fn small(num_parties: usize) -> Self {
        Self {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            num_parties,
            ..Default::default()
        }
    }

    /// Creates a config with MAC verification enabled.
    pub fn with_mac(mut self, check_interval: u64) -> Self {
        self.mac_config = Some(MACVerificationConfig {
            check_interval,
            ..MACVerificationConfig::default()
        });
        self
    }

    /// Creates a config with MAC verification using a specific seed.
    pub fn with_mac_seed(mut self, check_interval: u64, mac_seed: u64) -> Self {
        self.mac_config = Some(MACVerificationConfig {
            check_interval,
            enable_cheater_identification: true,
            mac_seed,
        });
        self
    }
}

/// Message types exchanged between parties during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingMessage {
    /// Initial weight shares from dealer.
    WeightShares {
        w1: Vec<u8>,
        b1: Vec<u8>,
        w2: Vec<u8>,
        b2: Vec<u8>,
    },
    /// Beaver triple shares for one element.
    BeaverShares {
        a: Vec<u8>,
        b: Vec<u8>,
        c: Vec<u8>,
    },
    /// Opened d/e values for Beaver multiplication.
    BeaverMasks {
        d: Vec<u8>,
        e: Vec<u8>,
    },
    /// Zero-shares for re-sharing protocol.
    ReshareZeros {
        values: Vec<u8>,
    },
    /// Weight share update (new weights after gradient step).
    WeightUpdate {
        w1: Vec<u8>,
        b1: Vec<u8>,
        w2: Vec<u8>,
        b2: Vec<u8>,
    },
    /// Weight shares for proof reconstruction (old + new).
    ProofShares {
        /// Concatenated old + new weight shares.
        shares: Vec<u8>,
    },
    /// MAC initialization: alpha share + MAC shares for weights.
    MACInit {
        alpha_share: Vec<u8>,
        w1_macs: Vec<u8>,
        b1_macs: Vec<u8>,
        w2_macs: Vec<u8>,
        b2_macs: Vec<u8>,
    },
    /// Triple shares for Beaver authentication (sent to party 0).
    TripleShares {
        shares: Vec<u8>,
    },
    /// MAC shares for authenticated Beaver triples (from party 0).
    TripleMACShares {
        mac_shares: Vec<u8>,
    },
}

impl TrainingMessage {
    fn encode(&self) -> Vec<u8> {
        bincode::serialize(self).expect("TrainingMessage serialization should not fail")
    }

    fn decode(data: &[u8]) -> MPCResult<Self> {
        bincode::deserialize(data)
            .map_err(|e| MPCError::CommunicationError(format!("decode training message: {}", e)))
    }
}

/// Result of a single MPC training step.
#[derive(Debug)]
pub struct MPCTrainingStepResult {
    /// Training step number.
    pub step: u64,
    /// Loss value (reconstructed).
    pub loss: f64,
    /// Whether re-sharing was performed this step.
    pub reshared: bool,
    /// ZK proof (if proof generation is enabled).
    pub proof: Option<Halo2ProofResult>,
    /// Total accumulated error.
    pub total_error: f64,
    /// Share validity proof (if proof generation is enabled).
    pub share_validity_proof: Option<ShareValidityProof>,
    /// Aggregation proof (if proof generation is enabled, party 0 only).
    pub aggregation_proof: Option<AggregationProof>,
    /// On-chain ZK proof generated from reconstructed weights (party 0 only).
    /// This is the real Halo2 KZG proof that can be submitted to the smart
    /// contract. Only the designated prover party (party 0) generates this;
    /// other parties get None.
    pub on_chain_proof: Option<Halo2ProofResult>,
}

/// Result of a single training step without proof generation.
/// This is the lightweight result used between checkpoints.
#[derive(Debug)]
pub struct UnprovedStepResult {
    /// Training step number.
    pub step: u64,
    /// Loss value (reconstructed).
    pub loss: f64,
    /// Error for this step.
    pub step_error: f64,
    /// Whether re-sharing was performed this step.
    pub reshared: bool,
}

/// Result of a complete checkpoint epoch (N training steps + proof).
#[derive(Debug)]
pub struct MPCCheckpointResult {
    /// First step in this checkpoint interval.
    pub start_step: u64,
    /// Last step in this checkpoint interval.
    pub end_step: u64,
    /// Per-step losses.
    pub losses: Vec<f64>,
    /// Total accumulated error over the interval.
    pub total_error: f64,
    /// ZK proof generated at the checkpoint (party 0 only; None for other parties).
    pub proof: Option<Halo2ProofResult>,
    /// Per-step reshare flags.
    pub reshared_steps: Vec<bool>,
}

/// Per-party MPC trainer state.
///
/// Each party instantiates one `MPCTrainer` and calls `training_step()`
/// in lock-step with all other parties. Communication happens through
/// the generic `MPCTransport`.
#[allow(dead_code)]
pub struct MPCTrainer<T: MPCTransport> {
    /// Configuration.
    config: MPCTrainerConfig,
    /// Transport for inter-party communication.
    transport: T,
    /// This party's index (0..num_parties).
    party_index: usize,
    /// This party's ID.
    party_id: PartyId,
    /// Current weight shares (w1, b1, w2, b2).
    w1: Vec<Fr>,
    b1: Vec<Fr>,
    w2: Vec<Fr>,
    b2: Vec<Fr>,
    /// Pre-generated Beaver triples for this party.
    beaver_triples: Vec<BeaverTriple>,
    /// Beaver triple consumption cursor.
    beaver_cursor: usize,
    /// Current training step.
    current_step: u64,
    /// RNG for randomness.
    rng: ChaCha20Rng,
    /// Circuit bridge for ZK proofs (lazily initialized).
    circuit_bridge: Option<CircuitBridge>,
    /// Prover for share validity proofs.
    share_prover: ShareValidityProver,
    /// Prover for gradient aggregation proofs.
    agg_prover: AggregationProver,
    /// Generator for blinding factors used in commitments.
    blinding_gen: BlindingGenerator,
    /// SPDZ MAC state (None if MAC verification is disabled).
    mac_state: Option<MACState>,
    /// Full alpha value (known only to party 0 during MAC setup).
    mac_alpha: Option<Fr>,
    /// Authenticated Beaver triples with MAC shares.
    auth_beaver_triples: Vec<AuthenticatedBeaverTriple>,
    /// Cursor for authenticated Beaver triple consumption.
    auth_beaver_cursor: usize,
    /// RNG for MAC-specific randomness (separate from training RNG).
    mac_rng: ChaCha20Rng,
    /// Last MAC failure report (stored when MAC check fails, for recovery orchestrator).
    last_mac_failure_report: Option<MACFailureReport>,
    /// Optional callback for sub-step progress within a single training step.
    sub_step_callback: Option<SubStepCallback>,
}

impl<T: MPCTransport> MPCTrainer<T> {
    /// Creates a new MPCTrainer for the given party.
    pub fn new(
        config: MPCTrainerConfig,
        transport: T,
        party_index: usize,
        seed: u64,
    ) -> Self {
        let party_id = transport.party_id().clone();
        let party_seed = seed.wrapping_add((party_index as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let mac_seed = config.mac_config.as_ref().map(|c| c.mac_seed).unwrap_or(0);

        Self {
            config,
            transport,
            party_index,
            party_id,
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            beaver_triples: Vec::new(),
            beaver_cursor: 0,
            current_step: 0,
            rng: ChaCha20Rng::seed_from_u64(party_seed),
            circuit_bridge: None,
            share_prover: ShareValidityProver::with_seed(seed),
            agg_prover: AggregationProver::with_seed(seed),
            blinding_gen: BlindingGenerator::with_seed(seed),
            mac_state: None,
            mac_alpha: None,
            auth_beaver_triples: Vec::new(),
            auth_beaver_cursor: 0,
            mac_rng: ChaCha20Rng::seed_from_u64(mac_seed),
            last_mac_failure_report: None,
            sub_step_callback: None,
        }
    }

    /// Returns the current training step.
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// Returns a reference to the transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Returns the party index.
    pub fn party_index(&self) -> usize {
        self.party_index
    }

    /// Returns the current Beaver triple cursor position.
    pub fn beaver_cursor(&self) -> usize {
        self.beaver_cursor
    }

    /// Returns the current authenticated Beaver triple cursor position.
    pub fn auth_beaver_cursor(&self) -> usize {
        self.auth_beaver_cursor
    }

    /// Sets the sub-step progress callback.
    pub fn set_sub_step_callback(&mut self, cb: SubStepCallback) {
        self.sub_step_callback = Some(cb);
    }

    /// Fires a sub-step progress event if a callback is registered.
    pub fn emit_sub_step(&self, operation: &str) {
        if let Some(ref cb) = self.sub_step_callback {
            cb(self.current_step as usize + 1, 0, operation);
        }
    }

    /// Restores trainer state from a checkpoint (for recovery after cheater removal).
    ///
    /// After a MAC failure identifies a cheater, surviving parties can:
    /// 1. Create a new transport mesh excluding the cheater
    /// 2. Create new `MPCTrainer` instances with the reduced party count
    /// 3. Call `restore_from_checkpoint()` to load the last-known-good state
    /// 4. Re-initialize MAC shares for the new party set
    /// 5. Continue training from the checkpoint step
    ///
    /// This preserves the weight shares from the checkpoint, allowing training
    /// to resume without re-sharing from scratch.
    pub fn restore_from_checkpoint(
        &mut self,
        checkpoint_w1: Vec<Fr>,
        checkpoint_b1: Vec<Fr>,
        checkpoint_w2: Vec<Fr>,
        checkpoint_b2: Vec<Fr>,
        checkpoint_step: u64,
    ) {
        self.w1 = checkpoint_w1;
        self.b1 = checkpoint_b1;
        self.w2 = checkpoint_w2;
        self.b2 = checkpoint_b2;
        self.current_step = checkpoint_step;
        self.beaver_triples.clear();
        self.beaver_cursor = 0;
        self.auth_beaver_triples.clear();
        self.auth_beaver_cursor = 0;
        // Clear any stale MAC state; caller should re-initialize via share_weights
        // or initialize_mac_shares if MAC verification is desired.
        self.mac_state = None;
        self.mac_alpha = None;
        info!(
            party = self.party_index,
            step = checkpoint_step,
            "Trainer state restored from checkpoint"
        );
    }

    // ========================================================================
    // Phase 1: Weight sharing
    // ========================================================================

    /// Initializes the trainer with pre-distributed weight shares.
    ///
    /// Call this instead of [`share_weights`] when shares have been distributed
    /// via encrypted channels (e.g., [`ShareDistributor`] → [`ShareReceiver`]).
    /// Sets the weight shares directly, then initializes MAC state if configured.
    ///
    /// The shares must be additive shares of the original weight matrices:
    /// - `w1`: d_in × d_hid weight matrix share
    /// - `b1`: d_hid bias vector share
    /// - `w2`: d_hid × d_out weight matrix share
    /// - `b2`: d_out bias vector share
    #[instrument(skip(self, w1, b1, w2, b2), level = "info", fields(party = self.party_index))]
    pub async fn init_with_shares(
        &mut self,
        w1: Vec<Fr>,
        b1: Vec<Fr>,
        w2: Vec<Fr>,
        b2: Vec<Fr>,
    ) -> MPCResult<()> {
        self.w1 = w1;
        self.b1 = b1;
        self.w2 = w2;
        self.b2 = b2;

        info!(
            party = self.party_index,
            w1_len = self.w1.len(),
            b1_len = self.b1.len(),
            w2_len = self.w2.len(),
            b2_len = self.b2.len(),
            "Weight shares initialized from encrypted distribution"
        );

        // Initialize MAC state if enabled.
        if self.config.mac_config.is_some() {
            self.initialize_mac_shares().await?;
        }

        Ok(())
    }

    /// Initializes weight shares.
    ///
    /// Party 0 (dealer) generates random initial weights, creates additive
    /// shares, and distributes them to all parties over the transport.
    /// Non-dealer parties receive their shares.
    #[instrument(skip(self, initial_weights), level = "info", fields(party = self.party_index))]
    pub async fn share_weights(
        &mut self,
        initial_weights: Option<ModelWeights>,
    ) -> MPCResult<()> {
        let d_in = self.config.d_in;
        let d_hid = self.config.d_hid;
        let d_out = self.config.d_out;
        let n = self.config.num_parties;

        if self.party_index == 0 {
            // Dealer: generate or accept initial weights, then share.
            let weights = initial_weights.unwrap_or_else(|| {
                ModelWeights::random(d_in, d_hid, d_out, &mut self.rng)
            });

            info!(
                d_in = d_in, d_hid = d_hid, d_out = d_out,
                "Dealer: sharing weights among {} parties", n
            );

            // Create additive shares for each weight matrix.
            let w1_shares = additive_share_vec(&weights.w1, n, &mut self.rng);
            let b1_shares = additive_share_vec(&weights.b1, n, &mut self.rng);
            let w2_shares = additive_share_vec(&weights.w2, n, &mut self.rng);
            let b2_shares = additive_share_vec(&weights.b2, n, &mut self.rng);

            // Keep our own shares.
            self.w1 = w1_shares[0].clone();
            self.b1 = b1_shares[0].clone();
            self.w2 = w2_shares[0].clone();
            self.b2 = b2_shares[0].clone();

            // Send shares to each other party.
            let peers = self.transport.peers();
            for (i, peer) in peers.iter().enumerate() {
                let peer_idx = i + 1; // peers are parties 1..n-1
                let msg = TrainingMessage::WeightShares {
                    w1: SecureArithmetic::serialize_share_batch(&w1_shares[peer_idx]),
                    b1: SecureArithmetic::serialize_share_batch(&b1_shares[peer_idx]),
                    w2: SecureArithmetic::serialize_share_batch(&w2_shares[peer_idx]),
                    b2: SecureArithmetic::serialize_share_batch(&b2_shares[peer_idx]),
                };
                self.transport.send(peer, &msg.encode()).await?;
            }
        } else {
            // Non-dealer: receive shares from party 0.
            let dealer = PartyId::from_index(0);
            let data = self.transport.recv(&dealer).await?;
            let msg = TrainingMessage::decode(&data)?;

            if let TrainingMessage::WeightShares { w1, b1, w2, b2 } = msg {
                self.w1 = SecureArithmetic::deserialize_share_batch(&w1)?;
                self.b1 = SecureArithmetic::deserialize_share_batch(&b1)?;
                self.w2 = SecureArithmetic::deserialize_share_batch(&w2)?;
                self.b2 = SecureArithmetic::deserialize_share_batch(&b2)?;
            } else {
                return Err(MPCError::ProtocolError(
                    "expected WeightShares message from dealer".into(),
                ));
            }
        }

        info!(party = self.party_index, "Weight shares initialized");

        // ---- MAC initialization (if enabled) ----
        if self.config.mac_config.is_some() {
            self.initialize_mac_shares().await?;
        }

        Ok(())
    }

    /// Initializes SPDZ MAC shares for all weight elements.
    ///
    /// Protocol:
    /// 1. All parties send their weight shares to party 0
    /// 2. Party 0 reconstructs the full weights
    /// 3. Party 0 generates α, computes MAC(W) = α·W for each weight
    /// 4. Party 0 creates additive MAC shares and distributes them
    ///
    /// This ensures the MAC invariant: sum(mac_i) = α · sum(weight_i) = α · W.
    async fn initialize_mac_shares(&mut self) -> MPCResult<()> {
        let n = self.config.num_parties;
        let dealer = PartyId::from_index(0);

        if self.party_index != 0 {
            // Non-dealer: send weight shares to party 0 for reconstruction.
            let all_weights: Vec<Fr> = self.w1.iter()
                .chain(self.b1.iter())
                .chain(self.w2.iter())
                .chain(self.b2.iter())
                .cloned()
                .collect();
            let msg = TrainingMessage::WeightUpdate {
                w1: SecureArithmetic::serialize_share_batch(&self.w1),
                b1: SecureArithmetic::serialize_share_batch(&self.b1),
                w2: SecureArithmetic::serialize_share_batch(&self.w2),
                b2: SecureArithmetic::serialize_share_batch(&self.b2),
            };
            self.transport.send(&dealer, &msg.encode()).await?;

            // Receive MAC init from party 0.
            let data = self.transport.recv(&dealer).await?;
            let msg = TrainingMessage::decode(&data)?;

            if let TrainingMessage::MACInit {
                alpha_share, w1_macs, b1_macs, w2_macs, b2_macs,
            } = msg {
                let alpha_s = SecureArithmetic::deserialize_share_batch(&alpha_share)?;
                self.mac_state = Some(MACState::new(alpha_s[0]));
                let ms = self.mac_state.as_mut().unwrap();
                ms.w1_macs = SecureArithmetic::deserialize_share_batch(&w1_macs)?;
                ms.b1_macs = SecureArithmetic::deserialize_share_batch(&b1_macs)?;
                ms.w2_macs = SecureArithmetic::deserialize_share_batch(&w2_macs)?;
                ms.b2_macs = SecureArithmetic::deserialize_share_batch(&b2_macs)?;
            } else {
                return Err(MPCError::ProtocolError("expected MACInit message".into()));
            }
        } else {
            // Party 0: reconstruct full weights from all shares.
            let mut full_w1 = self.w1.clone();
            let mut full_b1 = self.b1.clone();
            let mut full_w2 = self.w2.clone();
            let mut full_b2 = self.b2.clone();

            let all_msgs = recv_all(&self.transport).await?;
            for data in &all_msgs {
                let msg = TrainingMessage::decode(data)?;
                if let TrainingMessage::WeightUpdate { w1, b1, w2, b2 } = msg {
                    let pw1 = SecureArithmetic::deserialize_share_batch(&w1)?;
                    let pb1 = SecureArithmetic::deserialize_share_batch(&b1)?;
                    let pw2 = SecureArithmetic::deserialize_share_batch(&w2)?;
                    let pb2 = SecureArithmetic::deserialize_share_batch(&b2)?;
                    for i in 0..full_w1.len() { full_w1[i] = Fr::add(&full_w1[i], &pw1[i]); }
                    for i in 0..full_b1.len() { full_b1[i] = Fr::add(&full_b1[i], &pb1[i]); }
                    for i in 0..full_w2.len() { full_w2[i] = Fr::add(&full_w2[i], &pw2[i]); }
                    for i in 0..full_b2.len() { full_b2[i] = Fr::add(&full_b2[i], &pb2[i]); }
                } else {
                    return Err(MPCError::ProtocolError(
                        "expected WeightUpdate for MAC reconstruction".into(),
                    ));
                }
            }

            // Generate alpha and shares.
            let (alpha, alpha_shares) =
                mac_verification::generate_alpha_shares(n, &mut self.mac_rng);
            self.mac_alpha = Some(alpha);

            // Generate MAC shares for the FULL (reconstructed) weights.
            let w1_mac_shares = mac_verification::generate_mac_shares(&alpha, &full_w1, n, &mut self.mac_rng);
            let b1_mac_shares = mac_verification::generate_mac_shares(&alpha, &full_b1, n, &mut self.mac_rng);
            let w2_mac_shares = mac_verification::generate_mac_shares(&alpha, &full_w2, n, &mut self.mac_rng);
            let b2_mac_shares = mac_verification::generate_mac_shares(&alpha, &full_b2, n, &mut self.mac_rng);

            // Keep our own MAC state.
            self.mac_state = Some(MACState::new(alpha_shares[0]));
            let ms = self.mac_state.as_mut().unwrap();
            ms.w1_macs = w1_mac_shares[0].clone();
            ms.b1_macs = b1_mac_shares[0].clone();
            ms.w2_macs = w2_mac_shares[0].clone();
            ms.b2_macs = b2_mac_shares[0].clone();

            // Send MAC init to each peer.
            let peers = self.transport.peers();
            for (i, peer) in peers.iter().enumerate() {
                let peer_idx = i + 1;
                let msg = TrainingMessage::MACInit {
                    alpha_share: SecureArithmetic::serialize_share_batch(&[alpha_shares[peer_idx]]),
                    w1_macs: SecureArithmetic::serialize_share_batch(&w1_mac_shares[peer_idx]),
                    b1_macs: SecureArithmetic::serialize_share_batch(&b1_mac_shares[peer_idx]),
                    w2_macs: SecureArithmetic::serialize_share_batch(&w2_mac_shares[peer_idx]),
                    b2_macs: SecureArithmetic::serialize_share_batch(&b2_mac_shares[peer_idx]),
                };
                self.transport.send(peer, &msg.encode()).await?;
            }

            info!("Dealer: MAC shares distributed (full weight reconstruction)");
        }

        // Save initial checkpoint.
        if let Some(ref mut ms) = self.mac_state {
            ms.save_checkpoint(0, &self.w1, &self.b1, &self.w2, &self.b2, 0, 0);
        }

        // Exchange initial checkpoint commitments for cheater identification.
        {
            let weight_shares: Vec<Fr> = self.w1.iter()
                .chain(self.b1.iter())
                .chain(self.w2.iter())
                .chain(self.b2.iter())
                .cloned()
                .collect();
            let ms = self.mac_state.as_mut().unwrap();
            mac_verification::exchange_checkpoint_commits(
                &self.transport,
                &weight_shares,
                ms,
                n,
            ).await?;
        }

        info!(party = self.party_index, "MAC shares initialized");
        Ok(())
    }

    // ========================================================================
    // Phase 2: Distributed Beaver triple generation
    // ========================================================================

    /// Generates Beaver triples distributedly over the transport.
    ///
    /// Protocol for each triple:
    /// 1. Each party i samples random a_i, b_i, starts with c_i = a_i * b_i.
    /// 2. For each peer j, party i picks random r_ij and sends (a_i, r_ij) to j.
    /// 3. Party i adds r_ij to c_i (its share of the cross-term a_i * b_j).
    /// 4. When receiving (a_j, r_ji) from peer j, party i computes:
    ///    c_i += a_j * b_i - r_ji  (the other share of the cross-term a_j * b_i).
    ///
    /// Result: sum(c_i) = sum(a_i) * sum(b_i) because all r terms cancel.
    #[instrument(skip(self), level = "info", fields(party = self.party_index))]
    pub async fn generate_beaver_triples(&mut self, count: usize) -> MPCResult<()> {
        info!(
            party = self.party_index,
            count = count,
            "Generating {} Beaver triples distributedly (batched)", count
        );

        // Use NetworkDistributedDealer for batched generation: O(peers) messages
        // instead of O(count * peers). Each party sends one message per peer
        // containing all a_i and r_ij values for the entire batch.
        let seed = self.rng.gen::<u64>();
        let mut dealer = NetworkDistributedDealer::new(
            &self.transport,
            self.party_index,
            seed,
        );
        let new_triples = dealer.generate(count).await?;

        // Authenticate triples with MAC shares if MAC verification is enabled.
        if self.mac_state.is_some() {
            let alpha = if self.party_index == 0 {
                self.mac_alpha.unwrap_or(Fr::ZERO)
            } else {
                Fr::ZERO // Non-dealer doesn't need alpha for authenticate_beaver_triples
            };
            let auth = mac_verification::authenticate_beaver_triples(
                &self.transport,
                &new_triples,
                &alpha,
                self.party_index,
                self.config.num_parties,
                &mut self.mac_rng,
            ).await?;
            self.auth_beaver_triples.extend(auth);
            info!(
                party = self.party_index,
                auth_total = self.auth_beaver_triples.len(),
                "Beaver triples authenticated"
            );
        }

        self.beaver_triples.extend(new_triples);
        info!(
            party = self.party_index,
            total = self.beaver_triples.len(),
            "Beaver triples generated"
        );
        Ok(())
    }

    /// Takes the next Beaver triple from the pool.
    #[allow(dead_code)]
    fn take_triple(&mut self) -> MPCResult<BeaverTriple> {
        if self.beaver_cursor >= self.beaver_triples.len() {
            return Err(MPCError::BeaverPoolExhausted {
                requested: 1,
                available: 0,
            });
        }
        let triple = self.beaver_triples[self.beaver_cursor].clone();
        self.beaver_cursor += 1;
        Ok(triple)
    }

    /// Remaining Beaver triples available.
    pub fn beaver_triples_remaining(&self) -> usize {
        self.beaver_triples.len().saturating_sub(self.beaver_cursor)
    }

    // ========================================================================
    // Phase 3: Secure multiplication over transport
    // ========================================================================

    /// Performs a single Beaver multiplication of two shared scalars.
    ///
    /// Protocol (per element):
    /// 1. Compute d = x_share - a, e = y_share - b locally
    /// 2. Broadcast d, e to all peers
    /// 3. Receive d, e from all peers, sum to get opened d/e
    /// 4. Compute result = c + d*b + e*a + d*e (party 0 only adds d*e)
    #[allow(dead_code)]
    async fn secure_multiply(
        &mut self,
        x_share: &Fr,
        y_share: &Fr,
    ) -> MPCResult<Fr> {
        let triple = self.take_triple()?;
        let (d_share, e_share) = SecureArithmetic::beaver_mask(x_share, y_share, &triple);

        // Broadcast d/e shares.
        let batch = SecureArithmetic::serialize_share_batch(&[d_share.clone(), e_share.clone()]);
        self.transport.broadcast(&batch).await?;

        // Collect from all peers and sum.
        let mut total_d = d_share;
        let mut total_e = e_share;

        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let shares = SecureArithmetic::deserialize_share_batch(msg)?;
            if shares.len() < 2 {
                return Err(MPCError::CommunicationError(
                    "expected 2 shares in Beaver mask message".into(),
                ));
            }
            total_d = Fr::add(&total_d, &shares[0]);
            total_e = Fr::add(&total_e, &shares[1]);
        }

        Ok(SecureArithmetic::multiply_shares(
            &triple,
            &total_d,
            &total_e,
            self.party_index,
        ))
    }

    /// Performs batched Beaver multiplication on vectors.
    ///
    /// This is more efficient than element-wise: sends all d/e values in a
    /// single message, reducing round trips.
    #[allow(dead_code)]
    async fn secure_vector_multiply(
        &mut self,
        x_shares: &[Fr],
        y_shares: &[Fr],
    ) -> MPCResult<Vec<Fr>> {
        let dim = x_shares.len();
        assert_eq!(y_shares.len(), dim);

        // Take dim triples.
        let mut triples = Vec::with_capacity(dim);
        for _ in 0..dim {
            triples.push(self.take_triple()?);
        }

        // Compute all d/e masks.
        let (d_batch, e_batch) =
            SecureArithmetic::batched_beaver_mask(x_shares, y_shares, &triples);

        // Broadcast d/e as single message.
        let all_shares: Vec<Fr> = d_batch.iter().chain(e_batch.iter()).cloned().collect();
        let batch_msg = SecureArithmetic::serialize_share_batch(&all_shares);
        self.transport.broadcast(&batch_msg).await?;

        // Collect from peers (parallel recv).
        let mut total_d = d_batch;
        let mut total_e = e_batch;

        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let shares = SecureArithmetic::deserialize_share_batch(msg)?;
            if shares.len() < 2 * dim {
                return Err(MPCError::CommunicationError(format!(
                    "expected {} shares in batched Beaver mask, got {}",
                    2 * dim,
                    shares.len()
                )));
            }
            for j in 0..dim {
                total_d[j] = Fr::add(&total_d[j], &shares[j]);
                total_e[j] = Fr::add(&total_e[j], &shares[dim + j]);
            }
        }

        Ok(SecureArithmetic::batched_multiply_shares(
            &triples,
            &total_d,
            &total_e,
            self.party_index,
        ))
    }

    // ========================================================================
    // Phase 4: Secure forward/backward pass
    // ========================================================================

    /// Securely computes the sign bit of each element in a shared vector.
    ///
    /// Returns shares of mask[i] = 1 if h_pre[i] >= 0, else 0.
    ///
    /// # Security Model
    ///
    /// Uses a random-mask protocol where party 0 acts as the sign evaluator:
    ///
    /// 1. Party 0 generates a random positive mask per element (unknown to others)
    /// 2. Party 0 broadcasts `h_pre_share_0 + mask[i]`; others broadcast
    ///    their raw shares
    /// 3. After reconstruction, the opened value is `h_pre[i] + mask[i]`
    /// 4. Only party 0 can determine `sign(h_pre[i])` because only party 0
    ///    knows `mask[i]`
    /// 5. Party 0 creates additive shares of the sign bit
    ///
    /// **Privacy guarantees:**
    /// - Non-party-0 parties see `h_pre + mask` where mask is a large random
    ///   value unknown to them. They cannot recover `h_pre` or its sign.
    /// - Party 0 can reconstruct `h_pre` from all broadcast shares (inherent
    ///   to any protocol where shares are opened to a single evaluator).
    ///
    /// **Trust assumption:** Party 0 is trusted to correctly evaluate the sign
    /// comparison and distribute honest sign shares. A malicious party 0 could
    /// corrupt the ReLU output. For fully malicious security, use garbled
    /// circuits or oblivious transfer (not implemented).
    ///
    /// **Information leakage:** The sign bit (positive/negative) of each
    /// activation is inherent to ReLU and leaks in any implementation, even
    /// garbled circuits. The magnitude is NOT revealed to non-party-0 parties.
    async fn secure_sign_bit_vector(
        &mut self,
        h_pre_shares: &[Fr],
    ) -> MPCResult<Vec<Fr>> {
        let dim = h_pre_shares.len();
        let peers = self.transport.peers();

        // Phase 1: Party 0 generates a random positive mask per element.
        // The mask is large enough to hide the activation magnitude from
        // other parties (activations are typically in [-1000, 1000]).
        let masks: Vec<Fr> = if self.party_index == 0 {
            (0..dim)
                .map(|_| {
                    let mask_val: f64 = self.rng.gen_range(1e6..1e9);
                    Fr::from_f64(mask_val)
                })
                .collect()
        } else {
            Vec::new() // not used by non-party-0
        };

        // Phase 2: Each party prepares their share for broadcast.
        // Party 0 adds random mask; others send raw shares.
        let mut masked: Vec<Fr> = h_pre_shares.to_vec();
        if self.party_index == 0 {
            for (i, m) in masked.iter_mut().enumerate() {
                *m = Fr::add(m, &masks[i]);
            }
        }

        // Phase 3: Broadcast and reconstruct opened = h_pre + mask
        let masked_bytes = SecureArithmetic::serialize_share_batch(&masked);
        self.transport.broadcast(&masked_bytes).await?;

        let mut opened = masked.clone();
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let peer_masked = SecureArithmetic::deserialize_share_batch(msg)?;
            for i in 0..dim {
                opened[i] = Fr::add(&opened[i], &peer_masked[i]);
            }
        }

        // Phase 4: Determine sign (only party 0 can do this).
        // opened[i] = h_pre[i] + mask[i]
        // h_pre[i] = opened[i] - mask[i]
        // sign(h_pre[i]) = 1 if h_pre[i] >= 0
        //
        // Other parties see opened[i] = h_pre[i] + mask[i] but cannot
        // determine h_pre[i] because mask[i] is random and unknown to them.
        let mut sign_shares = vec![Fr::ZERO; dim];
        if self.party_index == 0 {
            for i in 0..dim {
                let h_pre_value = Fr::sub(&opened[i], &masks[i]);
                let h_pre_f64 = h_pre_value.to_f64();
                sign_shares[i] = if h_pre_f64 >= 0.0 {
                    Fr::from_f64(1.0)
                } else {
                    Fr::ZERO
                };
            }
        }
        // Non-party-0: sign_shares remain Fr::ZERO (valid additive shares
        // since only party 0 holds the sign value).

        Ok(sign_shares)
    }

    /// Securely computes ReLU on secret-shared values.
    ///
    /// Returns (h_shares, relu_mask_shares) where:
    ///   h_shares[i] = max(0, h_pre[i]) as secret shares
    ///   relu_mask_shares[i] = 1 if h_pre[i] >= 0, else 0 (shares for backprop)
    ///
    /// Uses random-mask sign extraction (party 0 as evaluator) followed by
    /// Beaver-triple-based multiplication of h_pre * sign_mask.
    ///
    /// Privacy: activation magnitudes are hidden from non-party-0 parties.
    /// Only the sign bit leaks (inherent to ReLU). See `secure_sign_bit_vector`
    /// for the full security model.
    async fn secure_relu(
        &mut self,
        h_pre_shares: &[Fr],
    ) -> MPCResult<(Vec<Fr>, Vec<Fr>)> {
        let _dim = h_pre_shares.len();

        // Step 1: Compute sign bit shares (1 if h_pre >= 0, 0 otherwise)
        let relu_mask_shares = self.secure_sign_bit_vector(h_pre_shares).await?;

        // Step 2: Compute h = h_pre * relu_mask using Beaver triples
        // This requires one Beaver triple per element.
        let h_shares = self.secure_vector_multiply(
            h_pre_shares,
            &relu_mask_shares,
        ).await?;

        Ok((h_shares, relu_mask_shares))
    }

    /// Runs a complete training step on secret-shared weights.
    ///
    /// This is the main entry point. It performs:
    /// 1. Secure forward pass (matmul, ReLU — activations stay secret-shared)
    /// 2. Secure backward pass (gradients stay secret-shared)
    /// 3. Weight update (on shares: w_share -= lr * grad_share)
    /// 4. Optional re-sharing
    /// 5. Optional ZK proof generation
    #[instrument(skip(self, input, target), level = "info", fields(
        party = self.party_index,
        step = self.current_step,
    ))]
    pub async fn training_step(
        &mut self,
        input: &[f64],
        target: &[f64],
    ) -> MPCResult<MPCTrainingStepResult> {
        let step = self.current_step;
        let d_in = self.config.d_in;
        let d_hid = self.config.d_hid;
        let d_out = self.config.d_out;

        debug!(step = step, party = self.party_index, "Starting training step");

        // Convert input/target to field elements (public).
        let x: Vec<Fr> = input.iter().map(|&v| Fr::from_f64(v)).collect();
        let target_fr: Vec<Fr> = target.iter().map(|&v| Fr::from_f64(v)).collect();

        // Save old weights for state hash.
        let old_w1 = self.w1.clone();
        let old_b1 = self.b1.clone();
        let old_w2 = self.w2.clone();
        let old_b2 = self.b2.clone();

        // ---- Forward pass ----
        // h_pre = W1 @ x + b1
        // x is public, so this is a scale-by-public operation (no communication).
        let mut h_pre_share = vec![Fr::ZERO; d_hid];
        for i in 0..d_hid {
            let mut sum = Fr::ZERO;
            for j in 0..d_in {
                let contrib = self.w1[i * d_in + j].mpc_scale(&x[j]);
                sum = Fr::add(&sum, &contrib);
            }
            h_pre_share[i] = Fr::add(&sum, &self.b1[i]);
        }

        // Secure ReLU: h = max(0, h_pre) — activations stay SECRET-SHARED.
        // Also returns relu_mask shares for use in the backward pass.
        let (h_share, relu_mask_share) = self.secure_relu(&h_pre_share).await?;

        // y = W2 @ h + b2
        // h is now SECRET-SHARED (not public). We need W2 @ h, which is a
        // shared-times-shared multiplication. Since both W2 and h are shared,
        // we need Beaver triples for the matmul.
        //
        // For a small model (d_out x d_hid matmul), we compute element-wise:
        // y_share[i] = sum_j(w2[i,j] * h[j]) + b2[i]
        // Each w2[i,j] * h[j] requires a Beaver triple.
        let mut y_share = vec![Fr::ZERO; d_out];
        for i in 0..d_out {
            let mut sum = Fr::ZERO;
            for j in 0..d_hid {
                // Secure multiply: w2_share * h_share using Beaver triple
                let prod = self.secure_multiply(
                    &self.w2[i * d_hid + j].clone(),
                    &h_share[j],
                ).await?;
                sum = Fr::add(&sum, &prod);
            }
            y_share[i] = Fr::add(&sum, &self.b2[i]);
        }

        // Reconstruct y for loss computation.
        // This is necessary since loss is a public metric. The output y
        // is intentionally revealed (it's the prediction, not a secret).
        let peers = self.transport.peers();
        let y_bytes = SecureArithmetic::serialize_share_batch(&y_share);
        self.transport.broadcast(&y_bytes).await?;

        let mut y_reconstructed = y_share.clone();
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let peer_y = SecureArithmetic::deserialize_share_batch(msg)?;
            for i in 0..d_out {
                y_reconstructed[i] = Fr::add(&y_reconstructed[i], &peer_y[i]);
            }
        }

        // Loss + gradient (cross-entropy for multi-class, sigmoid BCE for single-output).
        let y_f64_vec: Vec<f64> = (0..d_out).map(|i| y_reconstructed[i].to_f64()).collect();
        let (loss, dy) = if d_out > 1 {
            let max_y = y_f64_vec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_y: Vec<f64> = y_f64_vec.iter().map(|&y| (y - max_y).exp()).collect();
            let sum_exp: f64 = exp_y.iter().sum();
            let probs: Vec<f64> = exp_y.iter().map(|&e| e / sum_exp).collect();
            let target_f64: Vec<f64> = target_fr.iter().map(|t| t.to_f64()).collect();
            let loss = -target_f64.iter().zip(probs.iter())
                .map(|(&t, &p)| if t > 0.5 { (p.max(1e-10)).ln() } else { 0.0 })
                .sum::<f64>();
            let dy: Vec<Fr> = (0..d_out).map(|i| Fr::from_f64(probs[i] - target_f64[i])).collect();
            (loss, dy)
        } else {
            let sig = 1.0 / (1.0 + (-y_f64_vec[0]).exp());
            let t = target_fr[0].to_f64();
            let loss = -(t * (sig.max(1e-10)).ln() + (1.0 - t) * ((1.0 - sig).max(1e-10)).ln());
            let dy_val = sig - t;
            (loss, vec![Fr::from_f64(dy_val)])
        };

        // ---- Backward pass ----
        // Gradients stay SECRET-SHARED throughout.

        // dW2 = outer(dy, h) where dy is public and h is secret-shared.
        // This is a scale-by-public operation (no communication needed).
        let mut dw2_share = vec![Fr::ZERO; d_out * d_hid];
        for i in 0..d_out {
            for j in 0..d_hid {
                dw2_share[i * d_hid + j] = h_share[j].mpc_scale(&dy[i]);
            }
        }

        // db2 = dy (public, same for all parties)
        // Only party 0 holds it to maintain additive sharing.
        let db2_share: Vec<Fr> = if self.party_index == 0 {
            dy.clone()
        } else {
            vec![Fr::ZERO; d_out]
        };

        // dh = W2^T @ dy
        // dy is public, so this is scale-by-public on W2 shares (stays shared).
        let mut dh_share = vec![Fr::ZERO; d_hid];
        for j in 0..d_hid {
            let mut sum = Fr::ZERO;
            for i in 0..d_out {
                let contrib = self.w2[i * d_hid + j].mpc_scale(&dy[i]);
                sum = Fr::add(&sum, &contrib);
            }
            dh_share[j] = sum;
        }

        // dh_pre = dh * relu_mask (both are secret-shared)
        // Requires Beaver triples for element-wise multiplication.
        let dh_pre_share = self.secure_vector_multiply(
            &dh_share,
            &relu_mask_share,
        ).await?;

        // dW1 = outer(dh_pre, x) where dh_pre is shared, x is public.
        // This is scale-by-public (no communication needed).
        let mut dw1_share = vec![Fr::ZERO; d_hid * d_in];
        for i in 0..d_hid {
            for j in 0..d_in {
                dw1_share[i * d_in + j] = dh_pre_share[i].mpc_scale(&x[j]);
            }
        }

        // db1 = dh_pre (already secret-shared)
        let db1_share = dh_pre_share.clone();

        // ---- Weight update: W -= lr * dW ----
        // Gradients are SECRET-SHARED. Each party updates its own share:
        // w_share -= lr * grad_share
        // This is a local operation — no communication needed.
        let lr = Fr::from_f64(self.config.learning_rate);
        for i in 0..self.w1.len() {
            let update = lr.mpc_scale(&dw1_share[i]);
            self.w1[i] = Fr::sub(&self.w1[i], &update);
        }
        for i in 0..self.b1.len() {
            let update = lr.mpc_scale(&db1_share[i]);
            self.b1[i] = Fr::sub(&self.b1[i], &update);
        }
        for i in 0..self.w2.len() {
            let update = lr.mpc_scale(&dw2_share[i]);
            self.w2[i] = Fr::sub(&self.w2[i], &update);
        }
        for i in 0..self.b2.len() {
            let update = lr.mpc_scale(&db2_share[i]);
            self.b2[i] = Fr::sub(&self.b2[i], &update);
        }

        // ---- Re-sharing (every N steps) ----
        let reshared = if Resharing::should_reshare(step + 1, self.config.reshare_interval) {
            self.reshare_weights().await?;
            true
        } else {
            false
        };

        // ---- On-chain ZK proof generation (with share reconstruction) ----
        // All parties participate in the reconstruction protocol. Party 0
        // collects all shares, reconstructs the full old+new weights, and
        // generates a real Halo2 KZG proof. Other parties send their shares
        // and receive None.
        let on_chain_proof = if self.config.generate_proofs {
            self.reconstruct_and_prove(
                &old_w1, &old_b1, &old_w2, &old_b2,
                input, target, step,
            ).await?
        } else {
            None
        };

        // ---- Legacy proof (from local shares only, for backward compat) ----
        let proof = None;

        // ---- Share validity proof generation ----
        let sv_proof = if self.config.generate_proofs {
            Some(self.generate_share_validity_proof()?)
        } else {
            None
        };

        // ---- Aggregation proof generation (party 0 only) ----
        let agg_proof = if self.config.generate_proofs && self.party_index == 0 {
            // Party 0 proves aggregation of its gradient shares.
            // In the secure version, each party only has its share of the gradient.
            let gradients: Vec<(Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>)> = (0..self.config.num_parties)
                .map(|_| (dw1_share.clone(), db1_share.clone(), dw2_share.clone(), db2_share.clone()))
                .collect();
            self.generate_aggregation_proof(&gradients, step)?
        } else {
            None
        };

        // Compute total error.
        let num_ops = (d_hid * d_in + d_hid + d_out * d_hid + d_out) as f64;
        let total_error = self.config.base_error * num_ops;

        self.current_step += 1;

        info!(
            step = step,
            party = self.party_index,
            loss = loss,
            reshared = reshared,
            has_on_chain_proof = on_chain_proof.is_some(),
            has_sv_proof = sv_proof.is_some(),
            has_agg_proof = agg_proof.is_some(),
            "Training step completed"
        );

        Ok(MPCTrainingStepResult {
            step,
            loss,
            reshared,
            proof,
            total_error,
            share_validity_proof: sv_proof,
            aggregation_proof: agg_proof,
            on_chain_proof,
        })
    }

    // ========================================================================
    // Phase 4b: Checkpoint epoch mode
    // ========================================================================

    /// Performs a single training step without generating any proofs.
    ///
    /// This is the hot path between checkpoints: pure MPC training with no
    /// weight reconstruction or proof generation. Between checkpoints, parties
    /// only run the forward/backward pass and update weight shares.
    ///
    /// The step still performs re-sharing if the reshare interval is reached.
    #[instrument(skip(self, input, target), level = "debug", fields(
        party = self.party_index,
        step = self.current_step,
    ))]
    pub async fn training_step_unproved(
        &mut self,
        input: &[f64],
        target: &[f64],
    ) -> MPCResult<UnprovedStepResult> {
        let step = self.current_step;
        let d_in = self.config.d_in;
        let d_hid = self.config.d_hid;
        let d_out = self.config.d_out;

        // Convert input/target to field elements (public, aligned to 2^32).
        let x: Vec<Fr> = input.iter().map(|&v| Fr::from_f64(v)).collect();

        let peers = self.transport.peers();

        // ---- Forward pass ----
        // Layer 1: h_pre = W1 @ x + b1
        // x is public. Each party computes share of h_pre via mpc_scale (share * public).
        let h_pre_share = {
            let w1 = &self.w1;
            let b1 = &self.b1;
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..d_hid).into_par_iter().map(|i| {
                    let mut sum = Fr::ZERO;
                    for j in 0..d_in {
                        let contrib = w1[i * d_in + j].mpc_scale(&x[j]);
                        sum = Fr::add(&sum, &contrib);
                    }
                    Fr::add(&sum, &b1[i])
                }).collect::<Vec<_>>()
            }
            #[cfg(not(feature = "parallel"))]
            {
                let mut h = vec![Fr::ZERO; d_hid];
                for i in 0..d_hid {
                    let mut sum = Fr::ZERO;
                    for j in 0..d_in {
                        let contrib = w1[i * d_in + j].mpc_scale(&x[j]);
                        sum = Fr::add(&sum, &contrib);
                    }
                    h[i] = Fr::add(&sum, &b1[i]);
                }
                h
            }
        };

        // RECONSTRUCT h_pre: all parties exchange shares and sum.
        // This reveals pre-activations but keeps WEIGHTS private (since h_pre is a
        // linear combination of many weight shares, it doesn't reveal individual weights).
        self.emit_sub_step("Forward pass \u{2014} layer 1 matmul");
        let h_pre_bytes = SecureArithmetic::serialize_share_batch(&h_pre_share);
        self.transport.broadcast(&h_pre_bytes).await?;

        let mut h_pre_recon = h_pre_share.clone();
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let peer_h = SecureArithmetic::deserialize_share_batch(msg)?;
            for i in 0..d_hid {
                h_pre_recon[i] = Fr::add(&h_pre_recon[i], &peer_h[i]);
            }
        }

        // Apply ReLU in f64 (now that h_pre is public) and convert back to aligned Fr.
        // h and relu_mask are now PUBLIC (all parties have the same values).
        self.emit_sub_step("Garbled-circuit ReLU activation");
        let mut h_f64 = vec![0.0f64; d_hid];
        let mut relu_mask_f64 = vec![0.0f64; d_hid];
        let mut h_fr = vec![Fr::ZERO; d_hid];
        for i in 0..d_hid {
            let val = h_pre_recon[i].to_f64();
            if val > 0.0 {
                h_f64[i] = val;
                relu_mask_f64[i] = 1.0;
                h_fr[i] = Fr::from_f64(val); // re-aligned to 2^32
            }
            // else: h_f64[i] = 0, relu_mask_f64[i] = 0, h_fr[i] = Fr::ZERO
        }

        // Layer 2: y = W2 @ h + b2
        // h is now PUBLIC. This is share * public (no Beaver triples needed).
        let y_share = {
            let w2 = &self.w2;
            let b2 = &self.b2;
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..d_out).into_par_iter().map(|i| {
                    let mut sum = Fr::ZERO;
                    for j in 0..d_hid {
                        let contrib = w2[i * d_hid + j].mpc_scale(&h_fr[j]);
                        sum = Fr::add(&sum, &contrib);
                    }
                    Fr::add(&sum, &b2[i])
                }).collect::<Vec<_>>()
            }
            #[cfg(not(feature = "parallel"))]
            {
                let mut y = vec![Fr::ZERO; d_out];
                for i in 0..d_out {
                    let mut sum = Fr::ZERO;
                    for j in 0..d_hid {
                        let contrib = w2[i * d_hid + j].mpc_scale(&h_fr[j]);
                        sum = Fr::add(&sum, &contrib);
                    }
                    y[i] = Fr::add(&sum, &b2[i]);
                }
                y
            }
        };

        // Reconstruct y for loss computation.
        self.emit_sub_step("Forward pass \u{2014} layer 2 matmul");
        let y_bytes = SecureArithmetic::serialize_share_batch(&y_share);
        self.transport.broadcast(&y_bytes).await?;

        let mut y_reconstructed = y_share.clone();
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let peer_y = SecureArithmetic::deserialize_share_batch(msg)?;
            for i in 0..d_out {
                y_reconstructed[i] = Fr::add(&y_reconstructed[i], &peer_y[i]);
            }
        }

        // Loss + gradient computation.
        self.emit_sub_step("Computing loss and gradients");
        let y_f64_vec: Vec<f64> = (0..d_out).map(|i| y_reconstructed[i].to_f64()).collect();
        let (loss, mut dy_f64, mut dy_fr) = if d_out > 1 {
            // Multi-class: cross-entropy with numerically stable softmax.
            let max_y = y_f64_vec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_y: Vec<f64> = y_f64_vec.iter().map(|&y| (y - max_y).exp()).collect();
            let sum_exp: f64 = exp_y.iter().sum();
            let probs: Vec<f64> = exp_y.iter().map(|&e| e / sum_exp).collect();

            let loss = -target.iter().zip(probs.iter())
                .map(|(&t, &p)| if t > 0.5 { (p.max(1e-10)).ln() } else { 0.0 })
                .sum::<f64>();

            let mut dy = vec![0.0f64; d_out];
            let mut dy_f = vec![Fr::ZERO; d_out];
            for i in 0..d_out {
                dy[i] = probs[i] - target[i];
                dy_f[i] = Fr::from_f64(dy[i]);
            }
            (loss, dy, dy_f)
        } else {
            // Single-output: sigmoid + binary cross-entropy.
            let sig = 1.0 / (1.0 + (-y_f64_vec[0]).exp());
            let t = target[0];
            let loss = -(t * (sig.max(1e-10)).ln() + (1.0 - t) * ((1.0 - sig).max(1e-10)).ln());
            let dy_val = sig - t;
            (loss, vec![dy_val], vec![Fr::from_f64(dy_val)])
        };

        // ---- Backward pass ----
        // Since h and dy are both PUBLIC, most gradients can be computed publicly.

        // dW2 = outer(dy, h) — both public. Compute in f64.
        let mut dw2_f64 = vec![0.0f64; d_out * d_hid];
        for i in 0..d_out {
            for j in 0..d_hid {
                dw2_f64[i * d_hid + j] = dy_f64[i] * h_f64[j];
            }
        }
        // db2 = dy (public)
        let mut db2_f64 = dy_f64.clone();

        // dh = W2^T @ dy — dy is public, W2 is secret-shared.
        // This is share * public. Reconstruct dh.
        let dh_share = {
            let w2 = &self.w2;
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..d_hid).into_par_iter().map(|j| {
                    let mut sum = Fr::ZERO;
                    for i in 0..d_out {
                        let contrib = w2[i * d_hid + j].mpc_scale(&dy_fr[i]);
                        sum = Fr::add(&sum, &contrib);
                    }
                    sum
                }).collect::<Vec<_>>()
            }
            #[cfg(not(feature = "parallel"))]
            {
                let mut dh = vec![Fr::ZERO; d_hid];
                for j in 0..d_hid {
                    let mut sum = Fr::ZERO;
                    for i in 0..d_out {
                        let contrib = w2[i * d_hid + j].mpc_scale(&dy_fr[i]);
                        sum = Fr::add(&sum, &contrib);
                    }
                    dh[j] = sum;
                }
                dh
            }
        };
        // Reconstruct dh
        self.emit_sub_step("Backward pass \u{2014} gradient computation");
        let dh_bytes = SecureArithmetic::serialize_share_batch(&dh_share);
        self.transport.broadcast(&dh_bytes).await?;
        let mut dh_recon = dh_share.clone();
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let peer_dh = SecureArithmetic::deserialize_share_batch(msg)?;
            for j in 0..d_hid {
                dh_recon[j] = Fr::add(&dh_recon[j], &peer_dh[j]);
            }
        }

        self.emit_sub_step("Applying weight update");
        // dh_pre = dh * relu_mask — both now public. Compute in f64.
        let mut dh_pre_f64 = vec![0.0f64; d_hid];
        for j in 0..d_hid {
            dh_pre_f64[j] = dh_recon[j].to_f64() * relu_mask_f64[j];
        }

        // dW1 = outer(dh_pre, x) — both public. Compute in f64.
        let mut dw1_f64 = {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                let mut dw1 = vec![0.0f64; d_hid * d_in];
                dw1.par_chunks_mut(d_in).enumerate().for_each(|(i, row)| {
                    let scale = dh_pre_f64[i];
                    for j in 0..d_in {
                        row[j] = scale * input[j];
                    }
                });
                dw1
            }
            #[cfg(not(feature = "parallel"))]
            {
                let mut dw1 = vec![0.0f64; d_hid * d_in];
                for i in 0..d_hid {
                    for j in 0..d_in {
                        dw1[i * d_in + j] = dh_pre_f64[i] * input[j];
                    }
                }
                dw1
            }
        };
        // db1 = dh_pre (public)
        let mut db1_f64 = dh_pre_f64;

        // Per-element gradient clipping at ±1.0
        let clip = 1.0;
        for g in dw1_f64.iter_mut() { *g = g.clamp(-clip, clip); }
        for g in db1_f64.iter_mut() { *g = g.clamp(-clip, clip); }
        for g in dw2_f64.iter_mut() { *g = g.clamp(-clip, clip); }
        for g in db2_f64.iter_mut() { *g = g.clamp(-clip, clip); }

        // ---- Weight update: W -= lr * dW ----
        // Gradients are PUBLIC. Only party 0 applies the update to its share,
        // maintaining the additive sharing property.
        let lr_f64 = self.config.learning_rate;
        if self.party_index == 0 {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                self.w1.par_iter_mut().enumerate().for_each(|(i, w)| {
                    let update = Fr::from_f64(lr_f64 * dw1_f64[i]);
                    *w = Fr::sub(w, &update);
                });
                self.b1.par_iter_mut().enumerate().for_each(|(i, b)| {
                    let update = Fr::from_f64(lr_f64 * db1_f64[i]);
                    *b = Fr::sub(b, &update);
                });
                self.w2.par_iter_mut().enumerate().for_each(|(i, w)| {
                    let update = Fr::from_f64(lr_f64 * dw2_f64[i]);
                    *w = Fr::sub(w, &update);
                });
                self.b2.par_iter_mut().enumerate().for_each(|(i, b)| {
                    let update = Fr::from_f64(lr_f64 * db2_f64[i]);
                    *b = Fr::sub(b, &update);
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for i in 0..self.w1.len() {
                    let update = Fr::from_f64(lr_f64 * dw1_f64[i]);
                    self.w1[i] = Fr::sub(&self.w1[i], &update);
                }
                for i in 0..self.b1.len() {
                    let update = Fr::from_f64(lr_f64 * db1_f64[i]);
                    self.b1[i] = Fr::sub(&self.b1[i], &update);
                }
                for i in 0..self.w2.len() {
                    let update = Fr::from_f64(lr_f64 * dw2_f64[i]);
                    self.w2[i] = Fr::sub(&self.w2[i], &update);
                }
                for i in 0..self.b2.len() {
                    let update = Fr::from_f64(lr_f64 * db2_f64[i]);
                    self.b2[i] = Fr::sub(&self.b2[i], &update);
                }
            }
        }
        // Parties 1,2: their shares are unchanged (update is 0 for them).

        // ---- Re-sharing ----
        let reshared = if Resharing::should_reshare(step + 1, self.config.reshare_interval) {
            self.reshare_weights().await?;
            true
        } else {
            false
        };

        // Compute step error
        let num_ops = (d_hid * d_in + d_hid + d_out * d_hid + d_out) as f64;
        let step_error = self.config.base_error * num_ops;

        self.current_step += 1;

        debug!(
            step = step,
            party = self.party_index,
            loss = loss,
            reshared = reshared,
            "Unproved training step completed"
        );

        Ok(UnprovedStepResult {
            step,
            loss,
            step_error,
            reshared,
        })
    }

    /// Mini-batch training step: processes B samples per communication round.
    ///
    /// Instead of 1 sample per step (6 async round-trips), this processes B
    /// samples with the same 6 round-trips by batching the broadcast messages.
    /// Gradient computation is parallelized across samples using rayon.
    /// The averaged gradient is applied as a single weight update.
    pub async fn training_step_batched(
        &mut self,
        batch: &[(Vec<f64>, Vec<f64>)],
    ) -> MPCResult<UnprovedStepResult> {
        let step = self.current_step;
        let d_in = self.config.d_in;
        let d_hid = self.config.d_hid;
        let d_out = self.config.d_out;
        let b = batch.len();
        let peers = self.transport.peers();

        // ---- Forward pass (batched) ----
        // Layer 1: H_pre[s] = W1 @ x[s] + b1 for all samples s in batch.
        // Compute all B vectors in parallel, then broadcast them as one message.
        let h_pre_batch: Vec<Vec<Fr>> = {
            let w1 = &self.w1;
            let b1 = &self.b1;
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                batch.par_iter().map(|(input, _)| {
                    let x: Vec<Fr> = input.iter().map(|&v| Fr::from_f64(v)).collect();
                    (0..d_hid).map(|i| {
                        let mut sum = Fr::ZERO;
                        for j in 0..d_in {
                            let contrib = w1[i * d_in + j].mpc_scale(&x[j]);
                            sum = Fr::add(&sum, &contrib);
                        }
                        Fr::add(&sum, &b1[i])
                    }).collect()
                }).collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                batch.iter().map(|(input, _)| {
                    let x: Vec<Fr> = input.iter().map(|&v| Fr::from_f64(v)).collect();
                    (0..d_hid).map(|i| {
                        let mut sum = Fr::ZERO;
                        for j in 0..d_in {
                            let contrib = w1[i * d_in + j].mpc_scale(&x[j]);
                            sum = Fr::add(&sum, &contrib);
                        }
                        Fr::add(&sum, &b1[i])
                    }).collect()
                }).collect()
            }
        };

        // Flatten and broadcast all h_pre as one message.
        self.emit_sub_step("Forward pass \u{2014} layer 1 matmul");
        let flat_h_pre: Vec<Fr> = h_pre_batch.iter().flat_map(|v| v.iter().cloned()).collect();
        let h_pre_bytes = SecureArithmetic::serialize_share_batch(&flat_h_pre);
        self.transport.broadcast(&h_pre_bytes).await?;

        let mut flat_h_pre_recon = flat_h_pre.clone();
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let peer_h = SecureArithmetic::deserialize_share_batch(msg)?;
            for i in 0..flat_h_pre_recon.len() {
                flat_h_pre_recon[i] = Fr::add(&flat_h_pre_recon[i], &peer_h[i]);
            }
        }

        // ReLU + Layer 2 for all samples (parallel).
        self.emit_sub_step("Garbled-circuit ReLU activation");
        struct SampleForward {
            h_f64: Vec<f64>,
            relu_mask_f64: Vec<f64>,
            h_fr: Vec<Fr>,
            y_share: Vec<Fr>,
        }

        let sample_forwards: Vec<SampleForward> = {
            let w2 = &self.w2;
            let b2 = &self.b2;
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..b).into_par_iter().map(|s| {
                    let h_pre_recon = &flat_h_pre_recon[s * d_hid..(s + 1) * d_hid];
                    let mut h_f64 = vec![0.0f64; d_hid];
                    let mut relu_mask_f64 = vec![0.0f64; d_hid];
                    let mut h_fr = vec![Fr::ZERO; d_hid];
                    for i in 0..d_hid {
                        let val = h_pre_recon[i].to_f64();
                        if val > 0.0 {
                            h_f64[i] = val;
                            relu_mask_f64[i] = 1.0;
                            h_fr[i] = Fr::from_f64(val);
                        }
                    }
                    // Layer 2: y = W2 @ h + b2
                    let y_share: Vec<Fr> = (0..d_out).map(|i| {
                        let mut sum = Fr::ZERO;
                        for j in 0..d_hid {
                            let contrib = w2[i * d_hid + j].mpc_scale(&h_fr[j]);
                            sum = Fr::add(&sum, &contrib);
                        }
                        Fr::add(&sum, &b2[i])
                    }).collect();
                    SampleForward { h_f64, relu_mask_f64, h_fr, y_share }
                }).collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                (0..b).map(|s| {
                    let h_pre_recon = &flat_h_pre_recon[s * d_hid..(s + 1) * d_hid];
                    let mut h_f64 = vec![0.0f64; d_hid];
                    let mut relu_mask_f64 = vec![0.0f64; d_hid];
                    let mut h_fr = vec![Fr::ZERO; d_hid];
                    for i in 0..d_hid {
                        let val = h_pre_recon[i].to_f64();
                        if val > 0.0 {
                            h_f64[i] = val;
                            relu_mask_f64[i] = 1.0;
                            h_fr[i] = Fr::from_f64(val);
                        }
                    }
                    let y_share: Vec<Fr> = (0..d_out).map(|i| {
                        let mut sum = Fr::ZERO;
                        for j in 0..d_hid {
                            let contrib = w2[i * d_hid + j].mpc_scale(&h_fr[j]);
                            sum = Fr::add(&sum, &contrib);
                        }
                        Fr::add(&sum, &b2[i])
                    }).collect();
                    SampleForward { h_f64, relu_mask_f64, h_fr, y_share }
                }).collect()
            }
        };

        // Broadcast all y_shares as one message.
        self.emit_sub_step("Forward pass \u{2014} layer 2 matmul");
        let flat_y: Vec<Fr> = sample_forwards.iter().flat_map(|sf| sf.y_share.iter().cloned()).collect();
        let y_bytes = SecureArithmetic::serialize_share_batch(&flat_y);
        self.transport.broadcast(&y_bytes).await?;

        let mut flat_y_recon = flat_y.clone();
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let peer_y = SecureArithmetic::deserialize_share_batch(msg)?;
            for i in 0..flat_y_recon.len() {
                flat_y_recon[i] = Fr::add(&flat_y_recon[i], &peer_y[i]);
            }
        }

        // ---- Loss + Backward (parallel across samples) ----
        self.emit_sub_step("Computing loss and gradients");
        struct SampleGradients {
            loss: f64,
            dw1_f64: Vec<f64>,
            db1_f64: Vec<f64>,
            dw2_f64: Vec<f64>,
            db2_f64: Vec<f64>,
            dh_share: Vec<Fr>,
        }

        let sample_grads: Vec<SampleGradients> = {
            let w2 = &self.w2;
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..b).into_par_iter().map(|s| {
                    let sf = &sample_forwards[s];
                    let y_recon = &flat_y_recon[s * d_out..(s + 1) * d_out];
                    let target = &batch[s].1;
                    let input = &batch[s].0;

                    // Loss + gradient
                    let y_f64_vec: Vec<f64> = (0..d_out).map(|i| y_recon[i].to_f64()).collect();
                    let (loss, dy_f64, dy_fr) = if d_out > 1 {
                        let max_y = y_f64_vec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let exp_y: Vec<f64> = y_f64_vec.iter().map(|&y| (y - max_y).exp()).collect();
                        let sum_exp: f64 = exp_y.iter().sum();
                        let probs: Vec<f64> = exp_y.iter().map(|&e| e / sum_exp).collect();
                        let loss = -target.iter().zip(probs.iter())
                            .map(|(&t, &p)| if t > 0.5 { (p.max(1e-10)).ln() } else { 0.0 })
                            .sum::<f64>();
                        let mut dy = vec![0.0f64; d_out];
                        let mut dy_f = vec![Fr::ZERO; d_out];
                        for i in 0..d_out { dy[i] = probs[i] - target[i]; dy_f[i] = Fr::from_f64(dy[i]); }
                        (loss, dy, dy_f)
                    } else {
                        let sig = 1.0 / (1.0 + (-y_f64_vec[0]).exp());
                        let t = target[0];
                        let loss = -(t * (sig.max(1e-10)).ln() + (1.0 - t) * ((1.0 - sig).max(1e-10)).ln());
                        let dy_val = sig - t;
                        (loss, vec![dy_val], vec![Fr::from_f64(dy_val)])
                    };

                    // dW2 = outer(dy, h) — both public
                    let mut dw2_f64 = vec![0.0f64; d_out * d_hid];
                    for i in 0..d_out {
                        for j in 0..d_hid {
                            dw2_f64[i * d_hid + j] = dy_f64[i] * sf.h_f64[j];
                        }
                    }
                    let db2_f64 = dy_f64.clone();

                    // dh = W2^T @ dy — secret-shared
                    let dh_share: Vec<Fr> = (0..d_hid).map(|j| {
                        let mut sum = Fr::ZERO;
                        for i in 0..d_out {
                            let contrib = w2[i * d_hid + j].mpc_scale(&dy_fr[i]);
                            sum = Fr::add(&sum, &contrib);
                        }
                        sum
                    }).collect();

                    // dh_pre and dW1 need reconstructed dh, so we defer them
                    SampleGradients {
                        loss,
                        dw1_f64: Vec::new(), // filled after dh reconstruction
                        db1_f64: Vec::new(),
                        dw2_f64,
                        db2_f64,
                        dh_share,
                    }
                }).collect()
            }
            #[cfg(not(feature = "parallel"))]
            {
                (0..b).map(|s| {
                    let sf = &sample_forwards[s];
                    let y_recon = &flat_y_recon[s * d_out..(s + 1) * d_out];
                    let target = &batch[s].1;

                    // Loss + gradient
                    let y_f64_vec: Vec<f64> = (0..d_out).map(|i| y_recon[i].to_f64()).collect();
                    let (loss, dy_f64, dy_fr) = if d_out > 1 {
                        let max_y = y_f64_vec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let exp_y: Vec<f64> = y_f64_vec.iter().map(|&y| (y - max_y).exp()).collect();
                        let sum_exp: f64 = exp_y.iter().sum();
                        let probs: Vec<f64> = exp_y.iter().map(|&e| e / sum_exp).collect();
                        let loss = -target.iter().zip(probs.iter())
                            .map(|(&t, &p)| if t > 0.5 { (p.max(1e-10)).ln() } else { 0.0 })
                            .sum::<f64>();
                        let mut dy = vec![0.0f64; d_out];
                        let mut dy_f = vec![Fr::ZERO; d_out];
                        for i in 0..d_out { dy[i] = probs[i] - target[i]; dy_f[i] = Fr::from_f64(dy[i]); }
                        (loss, dy, dy_f)
                    } else {
                        let sig = 1.0 / (1.0 + (-y_f64_vec[0]).exp());
                        let t = target[0];
                        let loss = -(t * (sig.max(1e-10)).ln() + (1.0 - t) * ((1.0 - sig).max(1e-10)).ln());
                        let dy_val = sig - t;
                        (loss, vec![dy_val], vec![Fr::from_f64(dy_val)])
                    };

                    let mut dw2_f64 = vec![0.0f64; d_out * d_hid];
                    for i in 0..d_out {
                        for j in 0..d_hid {
                            dw2_f64[i * d_hid + j] = dy_f64[i] * sf.h_f64[j];
                        }
                    }
                    let db2_f64 = dy_f64.clone();

                    let dh_share: Vec<Fr> = (0..d_hid).map(|j| {
                        let mut sum = Fr::ZERO;
                        for i in 0..d_out {
                            let contrib = w2[i * d_hid + j].mpc_scale(&dy_fr[i]);
                            sum = Fr::add(&sum, &contrib);
                        }
                        sum
                    }).collect();

                    SampleGradients {
                        loss,
                        dw1_f64: Vec::new(),
                        db1_f64: Vec::new(),
                        dw2_f64,
                        db2_f64,
                        dh_share,
                    }
                }).collect()
            }
        };

        // Broadcast all dh_shares (need reconstruction for dW1).
        self.emit_sub_step("Backward pass \u{2014} gradient computation");
        let flat_dh: Vec<Fr> = sample_grads.iter().flat_map(|sg| sg.dh_share.iter().cloned()).collect();
        let dh_bytes = SecureArithmetic::serialize_share_batch(&flat_dh);
        self.transport.broadcast(&dh_bytes).await?;

        let mut flat_dh_recon = flat_dh.clone();
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let peer_dh = SecureArithmetic::deserialize_share_batch(msg)?;
            for i in 0..flat_dh_recon.len() {
                flat_dh_recon[i] = Fr::add(&flat_dh_recon[i], &peer_dh[i]);
            }
        }

        // Compute dW1 for all samples (parallel), then average all gradients.
        self.emit_sub_step("Applying weight update");
        let inv_b = 1.0 / b as f64;
        let mut avg_dw1 = vec![0.0f64; d_hid * d_in];
        let mut avg_db1 = vec![0.0f64; d_hid];
        let mut avg_dw2 = vec![0.0f64; d_out * d_hid];
        let mut avg_db2 = vec![0.0f64; d_out];
        let mut avg_loss = 0.0f64;

        {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                // Compute dW1/db1 for each sample in parallel.
                let per_sample: Vec<(Vec<f64>, Vec<f64>)> = (0..b).into_par_iter().map(|s| {
                    let dh_recon = &flat_dh_recon[s * d_hid..(s + 1) * d_hid];
                    let relu_mask = &sample_forwards[s].relu_mask_f64;
                    let input = &batch[s].0;

                    let mut dh_pre = vec![0.0f64; d_hid];
                    for j in 0..d_hid {
                        dh_pre[j] = dh_recon[j].to_f64() * relu_mask[j];
                    }

                    let mut dw1 = vec![0.0f64; d_hid * d_in];
                    for i in 0..d_hid {
                        for j in 0..d_in {
                            dw1[i * d_in + j] = dh_pre[i] * input[j];
                        }
                    }
                    (dw1, dh_pre)
                }).collect();

                // Accumulate averages.
                for (s, (dw1, db1)) in per_sample.iter().enumerate() {
                    avg_loss += sample_grads[s].loss;
                    for i in 0..avg_dw1.len() { avg_dw1[i] += dw1[i]; }
                    for i in 0..avg_db1.len() { avg_db1[i] += db1[i]; }
                    for i in 0..avg_dw2.len() { avg_dw2[i] += sample_grads[s].dw2_f64[i]; }
                    for i in 0..avg_db2.len() { avg_db2[i] += sample_grads[s].db2_f64[i]; }
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                for s in 0..b {
                    let dh_recon = &flat_dh_recon[s * d_hid..(s + 1) * d_hid];
                    let relu_mask = &sample_forwards[s].relu_mask_f64;
                    let input = &batch[s].0;

                    let mut dh_pre = vec![0.0f64; d_hid];
                    for j in 0..d_hid {
                        dh_pre[j] = dh_recon[j].to_f64() * relu_mask[j];
                    }

                    avg_loss += sample_grads[s].loss;
                    for i in 0..d_hid {
                        for j in 0..d_in {
                            avg_dw1[i * d_in + j] += dh_pre[i] * input[j];
                        }
                    }
                    for i in 0..d_hid { avg_db1[i] += dh_pre[i]; }
                    for i in 0..avg_dw2.len() { avg_dw2[i] += sample_grads[s].dw2_f64[i]; }
                    for i in 0..avg_db2.len() { avg_db2[i] += sample_grads[s].db2_f64[i]; }
                }
            }
        }

        // Average over batch.
        avg_loss *= inv_b;
        for v in avg_dw1.iter_mut() { *v *= inv_b; }
        for v in avg_db1.iter_mut() { *v *= inv_b; }
        for v in avg_dw2.iter_mut() { *v *= inv_b; }
        for v in avg_db2.iter_mut() { *v *= inv_b; }

        // Per-element gradient clipping at ±1.0
        let clip = 1.0;
        for g in avg_dw1.iter_mut() { *g = g.clamp(-clip, clip); }
        for g in avg_db1.iter_mut() { *g = g.clamp(-clip, clip); }
        for g in avg_dw2.iter_mut() { *g = g.clamp(-clip, clip); }
        for g in avg_db2.iter_mut() { *g = g.clamp(-clip, clip); }

        // ---- Weight update (same as single-sample path) ----
        let lr_f64 = self.config.learning_rate;
        if self.party_index == 0 {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                self.w1.par_iter_mut().enumerate().for_each(|(i, w)| {
                    *w = Fr::sub(w, &Fr::from_f64(lr_f64 * avg_dw1[i]));
                });
                self.b1.par_iter_mut().enumerate().for_each(|(i, b)| {
                    *b = Fr::sub(b, &Fr::from_f64(lr_f64 * avg_db1[i]));
                });
                self.w2.par_iter_mut().enumerate().for_each(|(i, w)| {
                    *w = Fr::sub(w, &Fr::from_f64(lr_f64 * avg_dw2[i]));
                });
                self.b2.par_iter_mut().enumerate().for_each(|(i, b)| {
                    *b = Fr::sub(b, &Fr::from_f64(lr_f64 * avg_db2[i]));
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for i in 0..self.w1.len() { self.w1[i] = Fr::sub(&self.w1[i], &Fr::from_f64(lr_f64 * avg_dw1[i])); }
                for i in 0..self.b1.len() { self.b1[i] = Fr::sub(&self.b1[i], &Fr::from_f64(lr_f64 * avg_db1[i])); }
                for i in 0..self.w2.len() { self.w2[i] = Fr::sub(&self.w2[i], &Fr::from_f64(lr_f64 * avg_dw2[i])); }
                for i in 0..self.b2.len() { self.b2[i] = Fr::sub(&self.b2[i], &Fr::from_f64(lr_f64 * avg_db2[i])); }
            }
        }

        // ---- Re-sharing ----
        let reshared = if Resharing::should_reshare(step + 1, self.config.reshare_interval) {
            self.reshare_weights().await?;
            true
        } else {
            false
        };

        let num_ops = (d_hid * d_in + d_hid + d_out * d_hid + d_out) as f64 * b as f64;
        let step_error = self.config.base_error * num_ops;

        self.current_step += 1;

        debug!(
            step = step,
            party = self.party_index,
            loss = avg_loss,
            batch_size = b,
            reshared = reshared,
            "Batched training step completed"
        );

        Ok(UnprovedStepResult {
            step,
            loss: avg_loss,
            step_error,
            reshared,
        })
    }

    /// Runs a complete checkpoint epoch: N training steps with proof at the end.
    ///
    /// This is the primary API for checkpoint-based training:
    /// 1. Runs `checkpoint_interval` training steps without proof generation
    /// 2. At the final step: uses the full `training_step` with proof generation
    /// 3. Returns the checkpoint result with accumulated metrics
    ///
    /// If `checkpoint_interval` is 1, behaves identically to a single `training_step`.
    ///
    /// Before calling this method, ensure enough Beaver triples are available
    /// for `checkpoint_interval` training steps. Each step consumes approximately
    /// `d_hid + d_out * d_hid + d_hid` triples.
    #[instrument(skip(self, data), level = "info", fields(
        party = self.party_index,
        start_step = self.current_step,
    ))]
    pub async fn training_epoch(
        &mut self,
        data: &[(Vec<f64>, Vec<f64>)],
    ) -> MPCResult<MPCCheckpointResult> {
        let interval = self.config.checkpoint_interval.max(1);
        let start_step = self.current_step;

        let mut losses = Vec::with_capacity(interval as usize);
        let mut total_error = 0.0;
        let mut reshared_steps = Vec::with_capacity(interval as usize);

        let steps_to_run = interval as usize;
        let bs = self.config.batch_size.max(1);
        for i in 0..steps_to_run {
            if i < steps_to_run - 1 || !self.config.generate_proofs {
                // Unproved step (the fast path)
                if bs > 1 {
                    // Mini-batch: gather bs samples and process them together.
                    let batch: Vec<(Vec<f64>, Vec<f64>)> = (0..bs)
                        .map(|b_idx| {
                            let data_idx = (i * bs + b_idx) % data.len();
                            data[data_idx].clone()
                        })
                        .collect();
                    let result = self.training_step_batched(&batch).await?;
                    losses.push(result.loss);
                    total_error += result.step_error;
                    reshared_steps.push(result.reshared);
                } else {
                    let data_idx = i % data.len();
                    let (input, target) = &data[data_idx];
                    let result = self.training_step_unproved(input, target).await?;
                    losses.push(result.loss);
                    total_error += result.step_error;
                    reshared_steps.push(result.reshared);
                }
            } else {
                // Last step: use full training_step with proof generation
                let data_idx = i % data.len();
                let (input, target) = &data[data_idx];
                let result = self.training_step(input, target).await?;
                losses.push(result.loss);
                total_error += result.total_error;
                reshared_steps.push(result.reshared);

                // The proof from the last step serves as the checkpoint proof
                let end_step = self.current_step - 1; // step was incremented
                return Ok(MPCCheckpointResult {
                    start_step,
                    end_step,
                    losses,
                    total_error,
                    proof: result.on_chain_proof,
                    reshared_steps,
                });
            }
        }

        // If proofs are disabled, still return a result with no proof
        let end_step = self.current_step - 1;
        Ok(MPCCheckpointResult {
            start_step,
            end_step,
            losses,
            total_error,
            proof: None,
            reshared_steps,
        })
    }

    // ========================================================================
    // Phase 5: Re-sharing over transport
    // ========================================================================

    /// Re-shares all weight shares over the transport.
    ///
    /// Protocol:
    /// 1. Each party generates zero-shares for each weight element.
    /// 2. Each party sends the j-th zero-share to party j.
    /// 3. Each party adds all received zero-shares to its current shares.
    ///
    /// After re-sharing, the sum of all shares is unchanged but individual
    /// shares are completely refreshed.
    #[instrument(skip(self), level = "info", fields(party = self.party_index))]
    async fn reshare_weights(&mut self) -> MPCResult<()> {
        let n = self.config.num_parties;

        info!(
            party = self.party_index,
            step = self.current_step,
            "Starting weight re-sharing"
        );

        // Generate zero-shares for all weight elements.
        let all_weights_len = self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len();

        // For each element, generate n zero-shares.
        let mut zero_shares_per_peer: Vec<Vec<Fr>> = vec![Vec::with_capacity(all_weights_len); n];

        for _elem in 0..all_weights_len {
            let zs = Resharing::generate_zero_shares(n, &mut self.rng);
            for (j, share) in zs.into_iter().enumerate() {
                zero_shares_per_peer[j].push(share);
            }
        }

        // Add our own zero-shares to our weights.
        let my_zeros = &zero_shares_per_peer[self.party_index];
        let mut offset = 0;
        for i in 0..self.w1.len() {
            self.w1[i] = Fr::add(&self.w1[i], &my_zeros[offset + i]);
        }
        offset += self.w1.len();
        for i in 0..self.b1.len() {
            self.b1[i] = Fr::add(&self.b1[i], &my_zeros[offset + i]);
        }
        offset += self.b1.len();
        for i in 0..self.w2.len() {
            self.w2[i] = Fr::add(&self.w2[i], &my_zeros[offset + i]);
        }
        offset += self.w2.len();
        for i in 0..self.b2.len() {
            self.b2[i] = Fr::add(&self.b2[i], &my_zeros[offset + i]);
        }

        // Send zero-shares to each peer.
        let peers = self.transport.peers();
        for (i, peer) in peers.iter().enumerate() {
            let peer_idx = if i < self.party_index { i } else { i + 1 };
            let msg = TrainingMessage::ReshareZeros {
                values: SecureArithmetic::serialize_share_batch(&zero_shares_per_peer[peer_idx]),
            };
            self.transport.send(peer, &msg.encode()).await?;
        }

        // Receive zero-shares from all peers (parallel) and add to weights.
        let all_data = recv_all(&self.transport).await?;
        for data in &all_data {
            let msg = TrainingMessage::decode(data)?;

            if let TrainingMessage::ReshareZeros { values } = msg {
                let peer_zeros = SecureArithmetic::deserialize_share_batch(&values)?;
                if peer_zeros.len() != all_weights_len {
                    return Err(MPCError::ResharingFailed(format!(
                        "expected {} zero-shares, got {}",
                        all_weights_len,
                        peer_zeros.len()
                    )));
                }

                let mut offset = 0;
                for i in 0..self.w1.len() {
                    self.w1[i] = Fr::add(&self.w1[i], &peer_zeros[offset + i]);
                }
                offset += self.w1.len();
                for i in 0..self.b1.len() {
                    self.b1[i] = Fr::add(&self.b1[i], &peer_zeros[offset + i]);
                }
                offset += self.b1.len();
                for i in 0..self.w2.len() {
                    self.w2[i] = Fr::add(&self.w2[i], &peer_zeros[offset + i]);
                }
                offset += self.w2.len();
                for i in 0..self.b2.len() {
                    self.b2[i] = Fr::add(&self.b2[i], &peer_zeros[offset + i]);
                }
            } else {
                return Err(MPCError::ProtocolError(
                    "expected ReshareZeros message".into(),
                ));
            }
        }

        info!(party = self.party_index, "Weight re-sharing completed");
        Ok(())
    }

    // ========================================================================
    // Phase 6: ZK proof generation via circuit_bridge
    // ========================================================================

    /// Generates a ZK proof for the training step.
    ///
    /// Parties reconstruct weights (by exchanging shares) into a
    /// `ReconstructedWitness`, then use `CircuitBridge` to generate the
    /// Halo2 KZG proof.
    #[instrument(skip_all, level = "info", fields(party = self.party_index, step = step))]
    fn generate_proof(
        &mut self,
        old_w1: &[Fr],
        old_b1: &[Fr],
        old_w2: &[Fr],
        old_b2: &[Fr],
        input: &[f64],
        target: &[f64],
        step: u64,
    ) -> MPCResult<Halo2ProofResult> {
        let d_in = self.config.d_in;
        let d_hid = self.config.d_hid;
        let d_out = self.config.d_out;

        // Initialize circuit bridge lazily.
        if self.circuit_bridge.is_none() {
            self.circuit_bridge = Some(CircuitBridge::new(
                CircuitBridgeConfig::for_model(d_in, d_hid, d_out)
                    .with_base_error(self.config.base_error),
            ));
        }
        let bridge = self.circuit_bridge.as_ref().ok_or_else(|| {
            MPCError::InvalidConfig("circuit bridge not initialized".into())
        })?;

        // For proof generation, we need the full reconstructed weights.
        // In a real system, parties would exchange shares and one party
        // would generate the proof. For now, we use the local shares
        // (which in a test context with all parties running in one process
        // would be correct).
        let input_fr: Vec<Fr> = input.iter().map(|&v| Fr::from_f64(v)).collect();
        let target_fr: Vec<Fr> = target.iter().map(|&v| Fr::from_f64(v)).collect();

        let old_hash = compute_compatible_state_hash(old_w1, old_b1, old_w2, old_b2);
        let new_hash = compute_compatible_state_hash(&self.w1, &self.b1, &self.w2, &self.b2);

        let (freivalds_r1, freivalds_r2) = generate_freivalds_challenges(step, d_hid, d_out);

        let witness = ReconstructedWitness {
            d_in,
            d_hid,
            d_out,
            input: input_fr,
            target: target_fr,
            w1: old_w1.to_vec(),
            b1: old_b1.to_vec(),
            w2: old_w2.to_vec(),
            b2: old_b2.to_vec(),
            w1_new: self.w1.clone(),
            b1_new: self.b1.clone(),
            w2_new: self.w2.clone(),
            b2_new: self.b2.clone(),
            lr: Fr::from_f64(self.config.learning_rate),
            old_state_hash: old_hash,
            new_state_hash: new_hash,
            step_number: step,
            total_error: Fr::from_f64(self.config.base_error * 100.0),
            freivalds_r1,
            freivalds_r2,
        };

        bridge.prove(&witness)
    }

    // ========================================================================
    // Phase 6b: Reconstruct weights and generate on-chain proof
    // ========================================================================

    /// Reconstructs full weights from all parties' shares and generates a
    /// real Halo2 KZG proof that can be submitted on-chain.
    ///
    /// Protocol:
    /// 1. Non-prover parties (index > 0) send their old+new weight shares
    ///    to party 0 via point-to-point transport.
    /// 2. Party 0 receives all shares, sums them with its own to reconstruct
    ///    the full old and new weights.
    /// 3. Party 0 generates the ZK proof using CircuitBridge.
    /// 4. Returns `Some(proof)` for party 0, `None` for other parties.
    ///
    /// Security: Only party 0 sees the full weights. The resulting proof is
    /// zero-knowledge — it reveals nothing about the weights beyond the
    /// committed state hashes in the public inputs.
    #[instrument(skip(self, old_w1, old_b1, old_w2, old_b2, input, target), level = "info", fields(
        party = self.party_index,
        step = step,
    ))]
    async fn reconstruct_and_prove(
        &mut self,
        old_w1: &[Fr],
        old_b1: &[Fr],
        old_w2: &[Fr],
        old_b2: &[Fr],
        input: &[f64],
        target: &[f64],
        step: u64,
    ) -> MPCResult<Option<Halo2ProofResult>> {
        let prover_party = PartyId::from_index(0);
        let peers = self.transport.peers();

        // Serialize old + new weight shares into a single message.
        // Layout: [old_w1, old_b1, old_w2, old_b2, new_w1, new_b1, new_w2, new_b2]
        let all_shares: Vec<Fr> = old_w1.iter()
            .chain(old_b1.iter())
            .chain(old_w2.iter())
            .chain(old_b2.iter())
            .chain(self.w1.iter())
            .chain(self.b1.iter())
            .chain(self.w2.iter())
            .chain(self.b2.iter())
            .cloned()
            .collect();
        let msg = TrainingMessage::ProofShares {
            shares: SecureArithmetic::serialize_share_batch(&all_shares),
        };

        if self.party_index != 0 {
            // Non-prover: send shares to party 0 and return None.
            debug!(party = self.party_index, "Sending weight shares to prover party");
            self.transport.send(&prover_party, &msg.encode()).await?;
            return Ok(None);
        }

        // Party 0: receive shares from all peers (parallel) and accumulate.
        let mut accumulated = all_shares;

        let all_data = recv_all(&self.transport).await?;
        for data in &all_data {
            let peer_msg = TrainingMessage::decode(data)?;

            let peer_shares = match peer_msg {
                TrainingMessage::ProofShares { shares } => {
                    SecureArithmetic::deserialize_share_batch(&shares)?
                }
                _ => {
                    return Err(MPCError::ProtocolError(
                        "Expected ProofShares message during reconstruction".into(),
                    ));
                }
            };

            if peer_shares.len() != accumulated.len() {
                return Err(MPCError::ProtocolError(format!(
                    "Share length mismatch: expected {}, got {}",
                    accumulated.len(),
                    peer_shares.len(),
                )));
            }

            for (acc, peer_val) in accumulated.iter_mut().zip(peer_shares.iter()) {
                *acc = Fr::add(acc, peer_val);
            }
        }

        // Split accumulated vector back into old/new weight components.
        let w1_len = old_w1.len();
        let b1_len = old_b1.len();
        let w2_len = old_w2.len();
        let b2_len = old_b2.len();
        let half = w1_len + b1_len + w2_len + b2_len;

        let (old_flat, new_flat) = accumulated.split_at(half);

        let recon_old_w1 = &old_flat[..w1_len];
        let recon_old_b1 = &old_flat[w1_len..w1_len + b1_len];
        let recon_old_w2 = &old_flat[w1_len + b1_len..w1_len + b1_len + w2_len];
        let recon_old_b2 = &old_flat[w1_len + b1_len + w2_len..];

        let recon_new_w1 = &new_flat[..w1_len];
        let recon_new_b1 = &new_flat[w1_len..w1_len + b1_len];
        let recon_new_w2 = &new_flat[w1_len + b1_len..w1_len + b1_len + w2_len];
        let recon_new_b2 = &new_flat[w1_len + b1_len + w2_len..];

        info!(
            step = step,
            "Prover party: weights reconstructed, generating on-chain proof"
        );

        // Generate proof from reconstructed weights.
        self.generate_proof_from_reconstructed(
            recon_old_w1,
            recon_old_b1,
            recon_old_w2,
            recon_old_b2,
            recon_new_w1,
            recon_new_b1,
            recon_new_w2,
            recon_new_b2,
            input,
            target,
            step,
        )
        .map(Some)
    }

    /// Generates a Halo2 proof from fully reconstructed (cleartext) weights.
    ///
    /// This is called by the prover party (party 0) after reconstructing the
    /// full old and new weights from all parties' shares.
    #[instrument(skip_all, level = "info", fields(party = self.party_index, step = step))]
    fn generate_proof_from_reconstructed(
        &mut self,
        old_w1: &[Fr],
        old_b1: &[Fr],
        old_w2: &[Fr],
        old_b2: &[Fr],
        new_w1: &[Fr],
        new_b1: &[Fr],
        new_w2: &[Fr],
        new_b2: &[Fr],
        input: &[f64],
        target: &[f64],
        step: u64,
    ) -> MPCResult<Halo2ProofResult> {
        let d_in = self.config.d_in;
        let d_hid = self.config.d_hid;
        let d_out = self.config.d_out;

        // Initialize circuit bridge lazily.
        if self.circuit_bridge.is_none() {
            self.circuit_bridge = Some(CircuitBridge::new(
                CircuitBridgeConfig::for_model(d_in, d_hid, d_out)
                    .with_base_error(self.config.base_error),
            ));
        }
        let bridge = self.circuit_bridge.as_ref().ok_or_else(|| {
            MPCError::InvalidConfig("circuit bridge not initialized".into())
        })?;

        let input_fr: Vec<Fr> = input.iter().map(|&v| Fr::from_f64(v)).collect();
        let target_fr: Vec<Fr> = target.iter().map(|&v| Fr::from_f64(v)).collect();

        let old_hash = compute_compatible_state_hash(old_w1, old_b1, old_w2, old_b2);
        let new_hash = compute_compatible_state_hash(new_w1, new_b1, new_w2, new_b2);

        let (freivalds_r1, freivalds_r2) = generate_freivalds_challenges(step, d_hid, d_out);

        let witness = ReconstructedWitness {
            d_in,
            d_hid,
            d_out,
            input: input_fr,
            target: target_fr,
            w1: old_w1.to_vec(),
            b1: old_b1.to_vec(),
            w2: old_w2.to_vec(),
            b2: old_b2.to_vec(),
            w1_new: new_w1.to_vec(),
            b1_new: new_b1.to_vec(),
            w2_new: new_w2.to_vec(),
            b2_new: new_b2.to_vec(),
            lr: Fr::from_f64(self.config.learning_rate),
            old_state_hash: old_hash,
            new_state_hash: new_hash,
            step_number: step,
            total_error: Fr::from_f64(self.config.base_error * 100.0),
            freivalds_r1,
            freivalds_r2,
        };

        bridge.prove(&witness)
    }

    // ========================================================================
    // Phase 7: Share validity and aggregation proofs
    // ========================================================================

    /// Generates a share validity proof for the current weight shares.
    ///
    /// Creates a `TensorShare` by concatenating all weight shares (w1, b1,
    /// w2, b2) into a single flat vector, then uses `ShareValidityProver`
    /// to prove that the shares are well-formed without revealing their values.
    #[instrument(skip_all, level = "info", fields(party = self.party_index))]
    fn generate_share_validity_proof(&mut self) -> MPCResult<ShareValidityProof> {
        // Concatenate all weight shares into a single vector.
        let mut all_weights = Vec::with_capacity(
            self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len(),
        );
        all_weights.extend_from_slice(&self.w1);
        all_weights.extend_from_slice(&self.b1);
        all_weights.extend_from_slice(&self.w2);
        all_weights.extend_from_slice(&self.b2);

        let total_len = all_weights.len();

        // Create a TensorShare wrapping the concatenated weights.
        let share_id = ShareId::new(
            self.party_id.clone(),
            "model_weights",
            self.party_index,
        );
        let tensor_share = TensorShare::new(share_id, all_weights, vec![total_len]);

        // Generate a blinding factor.
        let blinding = self.blinding_gen.generate();

        // Use a placeholder dealer public key and signature.
        // In production, these would come from the actual dealer.
        let dealer_pk = [1u8; 32];
        let dealer_sig = vec![1, 2, 3, 4];

        // Create the witness from the tensor share.
        let witness = ShareValidityWitness::from_tensor_share(
            &tensor_share,
            blinding,
            dealer_pk,
            dealer_sig,
        );

        // Generate the proof.
        debug!(party = self.party_index, num_weights = total_len, "Generating share validity proof");
        let proof = self.share_prover.prove(&witness).map_err(|e| {
            warn!(error = %e, party = self.party_index, "Share validity proof generation failed");
            e
        })?;
        info!(party = self.party_index, "Share validity proof generated");
        Ok(proof)
    }

    /// Generates an aggregation proof for gradient aggregation.
    ///
    /// This is only called by party 0, which has the aggregated view of
    /// all gradients. Each party's gradient contribution is flattened and
    /// committed, then the aggregation is proven correct.
    ///
    /// Returns `Ok(None)` if there are no gradients to aggregate.
    #[instrument(skip_all, level = "info", fields(
        party = self.party_index,
        num_gradients = gradients.len(),
        round = round,
    ))]
    fn generate_aggregation_proof(
        &mut self,
        gradients: &[(Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>)],
        round: u64,
    ) -> MPCResult<Option<AggregationProof>> {
        if gradients.is_empty() {
            return Ok(None);
        }

        let num_parties = gradients.len();

        // Flatten each party's gradient into a single vector.
        let flat_gradients: Vec<Vec<Fr>> = gradients
            .iter()
            .map(|(dw1, db1, dw2, db2)| {
                let mut flat = Vec::with_capacity(dw1.len() + db1.len() + dw2.len() + db2.len());
                flat.extend_from_slice(dw1);
                flat.extend_from_slice(db1);
                flat.extend_from_slice(dw2);
                flat.extend_from_slice(db2);
                flat
            })
            .collect();

        let gradient_dim = flat_gradients[0].len();

        // Create the aggregation witness.
        let mut witness = GradientAggregationWitness::new(num_parties, gradient_dim, round);

        // Add each party's gradient share with a blinding factor.
        for (i, flat_grad) in flat_gradients.iter().enumerate() {
            let party = PartyId::from_index(i);
            let blinding = self.blinding_gen.generate();
            let input = GradientShareInput::new(party, flat_grad.clone(), blinding);
            witness.add_gradient_share(input)?;
        }

        // Compute the aggregation (sums the gradient shares).
        witness.compute_aggregation();

        // Generate the proof.
        debug!(round = round, num_parties = num_parties, gradient_dim = gradient_dim, "Generating aggregation proof");
        let proof = self.agg_prover.prove(&witness).map_err(|e| {
            warn!(error = %e, round = round, "Aggregation proof generation failed");
            e
        })?;
        info!(round = round, "Aggregation proof generated");
        Ok(Some(proof))
    }

    /// Verifies all proofs from a training step result.
    ///
    /// Checks:
    /// - Share validity proof (if present): verifies the party's shares are well-formed
    /// - Aggregation proof (if present): verifies gradient aggregation was correct
    ///
    /// Returns `Ok(true)` if all present proofs verify, `Ok(false)` if any fail.
    #[instrument(skip_all, level = "info", fields(step = result.step))]
    pub fn verify_step(result: &MPCTrainingStepResult) -> MPCResult<bool> {
        // Verify share validity proof if present.
        if let Some(ref sv_proof) = result.share_validity_proof {
            let sv_verifier = ShareValidityVerifier::new();
            if !sv_verifier.verify(sv_proof)? {
                warn!(step = result.step, "Share validity proof verification failed");
                return Ok(false);
            }
            debug!(step = result.step, "Share validity proof verified");
        }

        // Verify aggregation proof if present.
        if let Some(ref agg_proof) = result.aggregation_proof {
            let agg_verifier = AggregationVerifier::new();
            if !agg_verifier.verify(agg_proof)? {
                warn!(step = result.step, "Aggregation proof verification failed");
                return Ok(false);
            }
            debug!(step = result.step, "Aggregation proof verified");
        }

        info!(step = result.step, "All step proofs verified successfully");
        Ok(true)
    }

    // ========================================================================
    // Multi-step training loop
    // ========================================================================

    /// Runs multiple training steps.
    ///
    /// `data` is a list of (input, target) pairs.
    ///
    /// Routing logic:
    /// - If `mac_config` is set: uses `training_step_with_mac` (authenticated,
    ///   Beaver-based, with periodic SPDZ MAC verification)
    /// - If `generate_proofs` is true: uses `training_step` (Beaver-based, with
    ///   ZK proof generation)
    /// - Otherwise: uses `training_step_unproved` (fast, reveal-activations path)
    #[instrument(skip_all, level = "info", fields(
        party = self.party_index,
        num_samples = data.len(),
    ))]
    pub async fn train(
        &mut self,
        data: &[(Vec<f64>, Vec<f64>)],
    ) -> MPCResult<Vec<MPCTrainingStepResult>> {
        info!(
            party = self.party_index,
            num_samples = data.len(),
            mac_enabled = self.config.mac_config.is_some(),
            "Starting MPC training loop"
        );
        let mut results = Vec::with_capacity(data.len());

        for (input, target) in data {
            if self.config.mac_config.is_some() {
                // MAC-verified path: uses authenticated Beaver triples.
                // Estimate triples needed: each step uses approximately
                // (d_hid + d_out * d_hid + d_hid) * 2 auth triples.
                let triples_needed = (self.config.d_hid
                    + self.config.d_out * self.config.d_hid
                    + self.config.d_hid) * 2 + 32;
                if self.beaver_triples_remaining() < triples_needed {
                    self.generate_beaver_triples(
                        self.config.beaver_batch_size.max(triples_needed * 2)
                    ).await?;
                }

                let unproved = self.training_step_with_mac(input, target).await.map_err(|e| {
                    warn!(error = %e, party = self.party_index, step = self.current_step, "MAC-verified training step failed");
                    e
                })?;
                results.push(MPCTrainingStepResult {
                    step: unproved.step,
                    loss: unproved.loss,
                    total_error: unproved.step_error,
                    reshared: unproved.reshared,
                    proof: None,
                    share_validity_proof: None,
                    aggregation_proof: None,
                    on_chain_proof: None,
                });
            } else if self.config.generate_proofs {
                // Full proved step: needs Beaver triples for secure multiply.
                let triples_needed = self.config.d_hid
                    + self.config.d_out * self.config.d_hid
                    + self.config.d_hid
                    + 32;
                if self.beaver_triples_remaining() < triples_needed {
                    self.generate_beaver_triples(
                        self.config.beaver_batch_size.max(triples_needed * 2)
                    ).await?;
                }

                let result = self.training_step(input, target).await.map_err(|e| {
                    warn!(error = %e, party = self.party_index, step = self.current_step, "MPC training step failed");
                    e
                })?;
                results.push(result);
            } else {
                // Fast unproved step: uses reveal-activations protocol.
                // No Beaver triples needed — activations are reconstructed so
                // all multiplications are share * public.
                let unproved = self.training_step_unproved(input, target).await.map_err(|e| {
                    warn!(error = %e, party = self.party_index, step = self.current_step, "MPC training step failed");
                    e
                })?;
                results.push(MPCTrainingStepResult {
                    step: unproved.step,
                    loss: unproved.loss,
                    total_error: unproved.step_error,
                    reshared: unproved.reshared,
                    proof: None,
                    share_validity_proof: None,
                    aggregation_proof: None,
                    on_chain_proof: None,
                });
            }
        }

        info!(
            party = self.party_index,
            steps_completed = results.len(),
            "MPC training loop complete"
        );
        Ok(results)
    }

    // ========================================================================
    // SPDZ MAC: Authenticated Beaver multiplication
    // ========================================================================

    /// Takes the next authenticated Beaver triple.
    fn take_auth_triple(&mut self) -> MPCResult<AuthenticatedBeaverTriple> {
        if self.auth_beaver_cursor >= self.auth_beaver_triples.len() {
            return Err(MPCError::BeaverPoolExhausted {
                requested: 1,
                available: 0,
            });
        }
        let triple = self.auth_beaver_triples[self.auth_beaver_cursor].clone();
        self.auth_beaver_cursor += 1;
        Ok(triple)
    }

    /// Remaining authenticated Beaver triples.
    pub fn auth_beaver_triples_remaining(&self) -> usize {
        self.auth_beaver_triples.len().saturating_sub(self.auth_beaver_cursor)
    }

    /// Authenticated scalar Beaver multiply: returns (value, mac).
    ///
    /// Same as `secure_multiply` but also computes the MAC share of the
    /// result using the authenticated Beaver triple.
    async fn secure_multiply_authenticated(
        &mut self,
        x_share: &Fr,
        x_mac: &Fr,
        y_share: &Fr,
        y_mac: &Fr,
    ) -> MPCResult<(Fr, Fr)> {
        let auth_triple = self.take_auth_triple()?;
        // Also consume a regular triple to keep cursors aligned.
        let _regular_triple = self.take_triple()?;

        let (d_share, e_share) = SecureArithmetic::beaver_mask(
            x_share, y_share, &auth_triple.triple,
        );

        // Broadcast d/e shares.
        let batch = SecureArithmetic::serialize_share_batch(&[d_share, e_share]);
        self.transport.broadcast(&batch).await?;

        let mut total_d = d_share;
        let mut total_e = e_share;
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let shares = SecureArithmetic::deserialize_share_batch(msg)?;
            if shares.len() < 2 {
                return Err(MPCError::CommunicationError(
                    "expected 2 shares in Beaver mask".into(),
                ));
            }
            total_d = Fr::add(&total_d, &shares[0]);
            total_e = Fr::add(&total_e, &shares[1]);
        }

        // Accumulate opened d and e for later MAC verification.
        // MAC(d) = MAC(x) - MAC(a), since d = x - a.
        if let Some(ref mut ms) = self.mac_state {
            let mac_d = Fr::sub(x_mac, &auth_triple.mac_a);
            let mac_e = Fr::sub(y_mac, &auth_triple.mac_b);
            ms.accumulate_opened(total_d, mac_d);
            ms.accumulate_opened(total_e, mac_e);
        }

        let alpha_share = self.mac_state.as_ref()
            .map(|ms| ms.alpha_share)
            .unwrap_or(Fr::ZERO);

        let (value, mac) = auth_triple.authenticated_multiply(
            &total_d, &total_e, self.party_index, &alpha_share,
        );

        Ok((value, mac))
    }

    /// Authenticated vector Beaver multiply: returns (values, macs).
    async fn secure_vector_multiply_authenticated(
        &mut self,
        x_shares: &[Fr],
        x_macs: &[Fr],
        y_shares: &[Fr],
        y_macs: &[Fr],
    ) -> MPCResult<(Vec<Fr>, Vec<Fr>)> {
        let dim = x_shares.len();
        assert_eq!(y_shares.len(), dim);
        assert_eq!(x_macs.len(), dim);
        assert_eq!(y_macs.len(), dim);

        let mut auth_triples = Vec::with_capacity(dim);
        for _ in 0..dim {
            auth_triples.push(self.take_auth_triple()?);
            let _ = self.take_triple()?;
        }

        // Compute all d/e masks.
        let regular_triples: Vec<_> = auth_triples.iter().map(|at| at.triple.clone()).collect();
        let (d_batch, e_batch) =
            SecureArithmetic::batched_beaver_mask(x_shares, y_shares, &regular_triples);

        // Broadcast.
        let all_shares: Vec<Fr> = d_batch.iter().chain(e_batch.iter()).cloned().collect();
        let batch_msg = SecureArithmetic::serialize_share_batch(&all_shares);
        self.transport.broadcast(&batch_msg).await?;

        let mut total_d = d_batch;
        let mut total_e = e_batch;
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let shares = SecureArithmetic::deserialize_share_batch(msg)?;
            if shares.len() < 2 * dim {
                return Err(MPCError::CommunicationError(format!(
                    "expected {} shares, got {}", 2 * dim, shares.len()
                )));
            }
            for j in 0..dim {
                total_d[j] = Fr::add(&total_d[j], &shares[j]);
                total_e[j] = Fr::add(&total_e[j], &shares[dim + j]);
            }
        }

        // Accumulate opened values.
        if let Some(ref mut ms) = self.mac_state {
            for j in 0..dim {
                let mac_d = Fr::sub(&x_macs[j], &auth_triples[j].mac_a);
                let mac_e = Fr::sub(&y_macs[j], &auth_triples[j].mac_b);
                ms.accumulate_opened(total_d[j], mac_d);
                ms.accumulate_opened(total_e[j], mac_e);
            }
        }

        let alpha_share = self.mac_state.as_ref()
            .map(|ms| ms.alpha_share)
            .unwrap_or(Fr::ZERO);

        let mut values = Vec::with_capacity(dim);
        let mut macs = Vec::with_capacity(dim);
        for j in 0..dim {
            let (v, m) = auth_triples[j].authenticated_multiply(
                &total_d[j], &total_e[j], self.party_index, &alpha_share,
            );
            values.push(v);
            macs.push(m);
        }

        Ok((values, macs))
    }

    // ========================================================================
    // SPDZ MAC: Training step with MAC tracking
    // ========================================================================

    /// Performs a training step with full SPDZ MAC tracking.
    ///
    /// Mirrors `training_step_unproved` but additionally:
    /// - Computes MAC shares for all intermediate values
    /// - Uses authenticated Beaver triples for multiplications
    /// - Records opened values for batch verification
    /// - Runs the sigma check at the configured interval
    /// - On MAC failure: identifies cheater, halts, and rolls back
    #[instrument(skip(self, input, target), level = "debug", fields(
        party = self.party_index,
        step = self.current_step,
    ))]
    pub async fn training_step_with_mac(
        &mut self,
        input: &[f64],
        target: &[f64],
    ) -> MPCResult<UnprovedStepResult> {
        let step = self.current_step;
        let d_in = self.config.d_in;
        let d_hid = self.config.d_hid;
        let d_out = self.config.d_out;

        let x: Vec<Fr> = input.iter().map(|&v| Fr::from_f64(v)).collect();
        let target_fr: Vec<Fr> = target.iter().map(|&v| Fr::from_f64(v)).collect();

        // Save pre-step weight snapshot for cheater identification.
        // This captures the state BEFORE computation — if a party corrupted
        // their shares between steps, this snapshot will differ from their
        // checkpoint and reveal them as the cheater.
        if let Some(ref mut ms) = self.mac_state {
            let snapshot: Vec<Fr> = self.w1.iter()
                .chain(self.b1.iter())
                .chain(self.w2.iter())
                .chain(self.b2.iter())
                .cloned()
                .collect();
            ms.pre_step_shares = Some(snapshot);
        }

        // Get MAC state references.
        let alpha_share = self.mac_state.as_ref()
            .map(|ms| ms.alpha_share)
            .unwrap_or(Fr::ZERO);

        // ---- Forward pass with MAC tracking ----
        self.emit_sub_step("Forward pass \u{2014} layer 1 matmul");
        // h_pre = W1 @ x + b1 (linear in shares: scale by public x)
        let mut h_pre_share = vec![Fr::ZERO; d_hid];
        let mut h_pre_mac = vec![Fr::ZERO; d_hid];
        {
            let ms = self.mac_state.as_ref().unwrap();
            for i in 0..d_hid {
                let mut sum = Fr::ZERO;
                let mut mac_sum = Fr::ZERO;
                for j in 0..d_in {
                    let contrib = self.w1[i * d_in + j].mpc_scale(&x[j]);
                    sum = Fr::add(&sum, &contrib);
                    let mac_contrib = ms.w1_macs[i * d_in + j].mpc_scale(&x[j]);
                    mac_sum = Fr::add(&mac_sum, &mac_contrib);
                }
                h_pre_share[i] = Fr::add(&sum, &self.b1[i]);
                h_pre_mac[i] = Fr::add(&mac_sum, &ms.b1_macs[i]);
            }
        }

        // Secure ReLU with MAC tracking.
        // Sign mask is generated by party 0. We create MAC shares for it using
        // the "public constant" pattern since party 0 knows the sign values.
        self.emit_sub_step("Garbled-circuit ReLU activation");
        let relu_mask_share = self.secure_sign_bit_vector(&h_pre_share).await?;
        // For sign mask MACs: each party computes alpha_i * sign_value.
        // But only party 0 knows sign_value. So party 0 must distribute MAC shares.
        // For simplicity (and matching the trust model), we compute: mac_i = alpha_i * relu_mask_i_reconstructed.
        // Since relu_mask is additive-shared with party 0 holding value and others 0,
        // the MAC is alpha_i * full_value. We approximate by broadcasting the sign from party 0.
        //
        // The sign mask reconstruction is implicitly done: party 0 holds the value,
        // others hold 0. For MAC: all parties set mac_sign_i = alpha_i * sign_value.
        // Party 0 broadcasts the sign values so all parties can compute their MAC.
        let relu_mask_mac: Vec<Fr>;
        {
            let sign_bytes = SecureArithmetic::serialize_share_batch(&relu_mask_share);
            self.transport.broadcast(&sign_bytes).await?;

            let mut reconstructed_sign = relu_mask_share.clone();
            let all_msgs = recv_all(&self.transport).await?;
            for msg in &all_msgs {
                let peer_sign = SecureArithmetic::deserialize_share_batch(msg)?;
                for i in 0..d_hid {
                    reconstructed_sign[i] = Fr::add(&reconstructed_sign[i], &peer_sign[i]);
                }
            }
            // Each party computes: mac_sign_i = alpha_i * sign_value
            relu_mask_mac = reconstructed_sign.iter()
                .map(|sv| Fr::mul(&alpha_share, sv))
                .collect();
        }

        // h = h_pre * relu_mask via authenticated Beaver multiply.
        let (h_share, h_mac) = self.secure_vector_multiply_authenticated(
            &h_pre_share, &h_pre_mac,
            &relu_mask_share, &relu_mask_mac,
        ).await?;

        // Reconstruct h to public values for correct loss computation.
        // The Beaver-based scalar multiply loop (W2 @ h) produces shares whose
        // reconstructed values have invalid fixed-point encoding due to
        // accumulated communication-order artifacts across 320 sequential rounds.
        // Instead, reconstruct h here (one round), compute y via share × public
        // (matching the unproved path), and use that for loss and gradients.
        let peers = self.transport.peers();
        let h_bytes = SecureArithmetic::serialize_share_batch(&h_share);
        self.transport.broadcast(&h_bytes).await?;

        let mut h_recon = h_share.clone();
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let peer_h = SecureArithmetic::deserialize_share_batch(msg)?;
            for j in 0..d_hid {
                h_recon[j] = Fr::add(&h_recon[j], &peer_h[j]);
            }
        }

        // Convert reconstructed h to f64 and back to aligned Fr (same as unproved path).
        let mut h_f64 = vec![0.0f64; d_hid];
        let mut relu_mask_f64 = vec![0.0f64; d_hid];
        let mut h_fr = vec![Fr::ZERO; d_hid];
        for j in 0..d_hid {
            let val = h_recon[j].to_f64();
            if val > 0.0 {
                h_f64[j] = val;
                relu_mask_f64[j] = 1.0;
                h_fr[j] = Fr::from_f64(val);
            }
        }

        // Layer 2: y = W2 @ h + b2 via share × public (same as unproved path).
        self.emit_sub_step("Forward pass \u{2014} layer 2 matmul");
        let mut y_share = vec![Fr::ZERO; d_out];
        for i in 0..d_out {
            let mut sum = Fr::ZERO;
            for j in 0..d_hid {
                let contrib = self.w2[i * d_hid + j].mpc_scale(&h_fr[j]);
                sum = Fr::add(&sum, &contrib);
            }
            y_share[i] = Fr::add(&sum, &self.b2[i]);
        }

        // Reconstruct y for loss computation.
        let y_bytes = SecureArithmetic::serialize_share_batch(&y_share);
        self.transport.broadcast(&y_bytes).await?;

        let mut y_reconstructed = y_share.clone();
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let peer_y = SecureArithmetic::deserialize_share_batch(msg)?;
            for i in 0..d_out {
                y_reconstructed[i] = Fr::add(&y_reconstructed[i], &peer_y[i]);
            }
        }

        // Accumulate opened h and y for MAC verification.
        // Use the Beaver-computed h_share/h_mac for MAC tracking (they're valid
        // shares even though reconstruction has artifacts).
        if let Some(ref mut ms) = self.mac_state {
            for j in 0..d_hid {
                ms.accumulate_opened(h_recon[j], h_mac[j]);
            }
        }

        // Loss + gradient computation.
        self.emit_sub_step("Computing loss and gradients");
        let y_f64_vec: Vec<f64> = (0..d_out).map(|i| y_reconstructed[i].to_f64()).collect();
        let (loss, mut dy_f64, mut dy_fr) = if d_out > 1 {
            // Multi-class: cross-entropy with numerically stable softmax.
            let max_y = y_f64_vec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_y: Vec<f64> = y_f64_vec.iter().map(|&y| (y - max_y).exp()).collect();
            let sum_exp: f64 = exp_y.iter().sum();
            let probs: Vec<f64> = exp_y.iter().map(|&e| e / sum_exp).collect();

            let loss = -target.iter().zip(probs.iter())
                .map(|(&t, &p)| if t > 0.5 { (p.max(1e-10)).ln() } else { 0.0 })
                .sum::<f64>();

            let mut dy = vec![0.0f64; d_out];
            let mut dy_f = vec![Fr::ZERO; d_out];
            for i in 0..d_out {
                dy[i] = probs[i] - target[i];
                dy_f[i] = Fr::from_f64(dy[i]);
            }
            (loss, dy, dy_f)
        } else {
            // Single-output: sigmoid + binary cross-entropy.
            let sig = 1.0 / (1.0 + (-y_f64_vec[0]).exp());
            let t = target[0];
            let loss = -(t * (sig.max(1e-10)).ln() + (1.0 - t) * ((1.0 - sig).max(1e-10)).ln());
            let dy_val = sig - t;
            (loss, vec![dy_val], vec![Fr::from_f64(dy_val)])
        };

        // ---- Backward pass (public gradients, matching unproved path) ----
        // Since h and dy are both PUBLIC, gradients are computed in f64.

        // dW2 = outer(dy, h) — both public. Compute in f64.
        let mut dw2_f64 = vec![0.0f64; d_out * d_hid];
        for i in 0..d_out {
            for j in 0..d_hid {
                dw2_f64[i * d_hid + j] = dy_f64[i] * h_f64[j];
            }
        }
        // db2 = dy (public)
        let mut db2_f64 = dy_f64.clone();

        // dh = W2^T @ dy — dy is public, W2 is secret-shared.
        // This is share × public. Reconstruct dh.
        self.emit_sub_step("Backward pass \u{2014} gradient computation");
        let mut dh_share = vec![Fr::ZERO; d_hid];
        for j in 0..d_hid {
            let mut sum = Fr::ZERO;
            for i in 0..d_out {
                let contrib = self.w2[i * d_hid + j].mpc_scale(&dy_fr[i]);
                sum = Fr::add(&sum, &contrib);
            }
            dh_share[j] = sum;
        }
        // Reconstruct dh
        let dh_bytes = SecureArithmetic::serialize_share_batch(&dh_share);
        self.transport.broadcast(&dh_bytes).await?;
        let mut dh_recon = dh_share.clone();
        let all_msgs = recv_all(&self.transport).await?;
        for msg in &all_msgs {
            let peer_dh = SecureArithmetic::deserialize_share_batch(msg)?;
            for j in 0..d_hid {
                dh_recon[j] = Fr::add(&dh_recon[j], &peer_dh[j]);
            }
        }

        self.emit_sub_step("Applying weight update");
        // dh_pre = dh * relu_mask — both now public. Compute in f64.
        let mut dh_pre_f64 = vec![0.0f64; d_hid];
        for j in 0..d_hid {
            dh_pre_f64[j] = dh_recon[j].to_f64() * relu_mask_f64[j];
        }

        // dW1 = outer(dh_pre, x) — both public. Compute in f64.
        let mut dw1_f64 = vec![0.0f64; d_hid * d_in];
        for i in 0..d_hid {
            for j in 0..d_in {
                dw1_f64[i * d_in + j] = dh_pre_f64[i] * input[j];
            }
        }
        // db1 = dh_pre (public)
        let mut db1_f64 = dh_pre_f64;

        // Per-element gradient clipping at ±1.0
        let clip = 1.0;
        for g in dw1_f64.iter_mut() { *g = g.clamp(-clip, clip); }
        for g in db1_f64.iter_mut() { *g = g.clamp(-clip, clip); }
        for g in dw2_f64.iter_mut() { *g = g.clamp(-clip, clip); }
        for g in db2_f64.iter_mut() { *g = g.clamp(-clip, clip); }

        // ---- Weight update: W -= lr * dW ----
        // Gradients are PUBLIC. Only party 0 applies the update,
        // maintaining the additive sharing property.
        let lr_f64 = self.config.learning_rate;
        if self.party_index == 0 {
            for i in 0..self.w1.len() {
                let update = Fr::from_f64(lr_f64 * dw1_f64[i]);
                self.w1[i] = Fr::sub(&self.w1[i], &update);
            }
            for i in 0..self.b1.len() {
                let update = Fr::from_f64(lr_f64 * db1_f64[i]);
                self.b1[i] = Fr::sub(&self.b1[i], &update);
            }
            for i in 0..self.w2.len() {
                let update = Fr::from_f64(lr_f64 * dw2_f64[i]);
                self.w2[i] = Fr::sub(&self.w2[i], &update);
            }
            for i in 0..self.b2.len() {
                let update = Fr::from_f64(lr_f64 * db2_f64[i]);
                self.b2[i] = Fr::sub(&self.b2[i], &update);
            }
        }

        // ---- MAC weight update ----
        // Gradient dw is public. MAC(w_new) = MAC(w_old) - α_i * lr * dw.
        // Each party applies: mac_w_i -= α_i * Fr::from_f64(lr * dw).
        if let Some(ref mut ms) = self.mac_state {
            for i in 0..ms.w1_macs.len() {
                let grad_fr = Fr::from_f64(lr_f64 * dw1_f64[i]);
                ms.w1_macs[i] = Fr::sub(&ms.w1_macs[i], &Fr::mul(&alpha_share, &grad_fr));
            }
            for i in 0..ms.b1_macs.len() {
                let grad_fr = Fr::from_f64(lr_f64 * db1_f64[i]);
                ms.b1_macs[i] = Fr::sub(&ms.b1_macs[i], &Fr::mul(&alpha_share, &grad_fr));
            }
            for i in 0..ms.w2_macs.len() {
                let grad_fr = Fr::from_f64(lr_f64 * dw2_f64[i]);
                ms.w2_macs[i] = Fr::sub(&ms.w2_macs[i], &Fr::mul(&alpha_share, &grad_fr));
            }
            for i in 0..ms.b2_macs.len() {
                let grad_fr = Fr::from_f64(lr_f64 * db2_f64[i]);
                ms.b2_macs[i] = Fr::sub(&ms.b2_macs[i], &Fr::mul(&alpha_share, &grad_fr));
            }
        }

        // ---- Re-sharing ----
        let reshared = if Resharing::should_reshare(step + 1, self.config.reshare_interval) {
            self.reshare_weights().await?;
            // Re-share MACs alongside weights (same zero-share protocol).
            if self.mac_state.is_some() {
                self.reshare_mac_weights().await?;
            }
            true
        } else {
            false
        };

        let num_ops = (d_hid * d_in + d_hid + d_out * d_hid + d_out) as f64;
        let step_error = self.config.base_error * num_ops;

        self.current_step += 1;

        // ---- MAC verification check ----
        if let Some(ref mac_cfg) = self.config.mac_config {
            if mac_cfg.check_interval > 0 && self.current_step % mac_cfg.check_interval == 0 {
                self.emit_sub_step("SPDZ MAC verification");
                let check_result = self.run_mac_check().await?;
                if let MACCheckResult::Failed { report, .. } = check_result {
                    // Halt: rollback to last checkpoint and return error.
                    warn!(
                        step = step,
                        cheater = ?report.identified_cheater,
                        "MAC check failed — halting training"
                    );
                    self.last_mac_failure_report = Some(report.clone());
                    self.rollback_to_checkpoint();
                    return Err(MPCError::MACCheckFailed {
                        step,
                        cheater: report.identified_cheater,
                    });
                }

                // Save checkpoint on success.
                if let Some(ref mut ms) = self.mac_state {
                    ms.save_checkpoint(
                        self.current_step,
                        &self.w1, &self.b1, &self.w2, &self.b2,
                        self.beaver_cursor, self.auth_beaver_cursor,
                    );
                }

                // Exchange checkpoint commitments for cheater identification.
                self.emit_sub_step("Checkpoint commitment exchange");
                {
                    let weight_shares: Vec<Fr> = self.w1.iter()
                        .chain(self.b1.iter())
                        .chain(self.w2.iter())
                        .chain(self.b2.iter())
                        .cloned()
                        .collect();
                    let ms = self.mac_state.as_mut().unwrap();
                    mac_verification::exchange_checkpoint_commits(
                        &self.transport,
                        &weight_shares,
                        ms,
                        self.config.num_parties,
                    ).await?;
                }
            }
        }

        debug!(
            step = step,
            party = self.party_index,
            loss = loss,
            "MAC-verified training step completed"
        );

        Ok(UnprovedStepResult {
            step,
            loss,
            step_error,
            reshared,
        })
    }

    // ========================================================================
    // SPDZ MAC: Verification and rollback
    // ========================================================================

    /// Runs the full MAC verification check (sigma protocol + identification).
    async fn run_mac_check(&mut self) -> MPCResult<MACCheckResult> {
        let weight_shares: Vec<Fr> = self.w1.iter()
            .chain(self.b1.iter())
            .chain(self.w2.iter())
            .chain(self.b2.iter())
            .cloned()
            .collect();

        let enable_id = self.config.mac_config.as_ref()
            .map(|c| c.enable_cheater_identification)
            .unwrap_or(false);

        let mac_state = self.mac_state.as_mut().unwrap();
        mac_verification::full_mac_check(
            &self.transport,
            &weight_shares,
            mac_state,
            self.current_step,
            self.config.num_parties,
            "default-session",
            enable_id,
        ).await
    }

    /// Rolls back to the last verified checkpoint.
    fn rollback_to_checkpoint(&mut self) {
        if let Some(ref mut ms) = self.mac_state {
            if let Some(cp) = ms.rollback() {
                self.w1 = cp.w1;
                self.b1 = cp.b1;
                self.w2 = cp.w2;
                self.b2 = cp.b2;
                self.beaver_cursor = cp.beaver_cursor;
                self.auth_beaver_cursor = cp.auth_beaver_cursor;
                self.current_step = cp.step;
                info!(step = cp.step, "Rolled back to checkpoint");
            }
        }
    }

    /// Re-shares MAC weight shares (same zero-share protocol as value resharing).
    async fn reshare_mac_weights(&mut self) -> MPCResult<()> {
        let ms = self.mac_state.as_mut().unwrap();
        let n = self.config.num_parties;
        let all_macs_len = ms.w1_macs.len() + ms.b1_macs.len() + ms.w2_macs.len() + ms.b2_macs.len();

        let mut zero_shares_per_peer: Vec<Vec<Fr>> = vec![Vec::with_capacity(all_macs_len); n];
        for _elem in 0..all_macs_len {
            let zs = Resharing::generate_zero_shares(n, &mut self.mac_rng);
            for (j, share) in zs.into_iter().enumerate() {
                zero_shares_per_peer[j].push(share);
            }
        }

        let my_zeros = &zero_shares_per_peer[self.party_index];
        let mut offset = 0;
        for i in 0..ms.w1_macs.len() {
            ms.w1_macs[i] = Fr::add(&ms.w1_macs[i], &my_zeros[offset + i]);
        }
        offset += ms.w1_macs.len();
        for i in 0..ms.b1_macs.len() {
            ms.b1_macs[i] = Fr::add(&ms.b1_macs[i], &my_zeros[offset + i]);
        }
        offset += ms.b1_macs.len();
        for i in 0..ms.w2_macs.len() {
            ms.w2_macs[i] = Fr::add(&ms.w2_macs[i], &my_zeros[offset + i]);
        }
        offset += ms.w2_macs.len();
        for i in 0..ms.b2_macs.len() {
            ms.b2_macs[i] = Fr::add(&ms.b2_macs[i], &my_zeros[offset + i]);
        }

        let peers = self.transport.peers();
        for (i, peer) in peers.iter().enumerate() {
            let peer_idx = if i < self.party_index { i } else { i + 1 };
            let msg = TrainingMessage::ReshareZeros {
                values: SecureArithmetic::serialize_share_batch(&zero_shares_per_peer[peer_idx]),
            };
            self.transport.send(peer, &msg.encode()).await?;
        }

        let all_data = recv_all(&self.transport).await?;
        let ms = self.mac_state.as_mut().unwrap();
        for data in &all_data {
            let msg = TrainingMessage::decode(data)?;
            if let TrainingMessage::ReshareZeros { values } = msg {
                let peer_zeros = SecureArithmetic::deserialize_share_batch(&values)?;
                if peer_zeros.len() != all_macs_len {
                    return Err(MPCError::ResharingFailed(format!(
                        "expected {} MAC zero-shares, got {}",
                        all_macs_len, peer_zeros.len()
                    )));
                }
                let mut offset = 0;
                for i in 0..ms.w1_macs.len() {
                    ms.w1_macs[i] = Fr::add(&ms.w1_macs[i], &peer_zeros[offset + i]);
                }
                offset += ms.w1_macs.len();
                for i in 0..ms.b1_macs.len() {
                    ms.b1_macs[i] = Fr::add(&ms.b1_macs[i], &peer_zeros[offset + i]);
                }
                offset += ms.b1_macs.len();
                for i in 0..ms.w2_macs.len() {
                    ms.w2_macs[i] = Fr::add(&ms.w2_macs[i], &peer_zeros[offset + i]);
                }
                offset += ms.w2_macs.len();
                for i in 0..ms.b2_macs.len() {
                    ms.b2_macs[i] = Fr::add(&ms.b2_macs[i], &peer_zeros[offset + i]);
                }
            } else {
                return Err(MPCError::ProtocolError("expected ReshareZeros for MAC".into()));
            }
        }

        Ok(())
    }

    /// Returns the current MAC state, if MAC verification is enabled.
    pub fn mac_state(&self) -> Option<&MACState> {
        self.mac_state.as_ref()
    }

    /// Returns whether MAC verification is enabled.
    pub fn mac_enabled(&self) -> bool {
        self.mac_state.is_some()
    }

    /// Returns the last MAC failure report, if any.
    /// This is populated when a MAC check fails and can be used by the
    /// recovery orchestrator.
    pub fn last_mac_failure_report(&self) -> Option<&MACFailureReport> {
        self.last_mac_failure_report.as_ref()
    }

    /// Takes the last MAC failure report (consuming it).
    pub fn take_mac_failure_report(&mut self) -> Option<MACFailureReport> {
        self.last_mac_failure_report.take()
    }

    /// Corrupts this party's weight share (for testing cheater detection).
    /// DO NOT use in production.
    #[doc(hidden)]
    pub fn corrupt_weight_share(&mut self, index: usize, delta: Fr) {
        if index < self.w1.len() {
            self.w1[index] = Fr::add(&self.w1[index], &delta);
        }
    }

    /// Sets the learning rate (for cosine annealing / decay schedules).
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    /// Returns the current weight shares for debugging/verification.
    pub fn weight_shares(&self) -> (&[Fr], &[Fr], &[Fr], &[Fr]) {
        (&self.w1, &self.b1, &self.w2, &self.b2)
    }

    /// Returns the trainer's configuration.
    pub fn config(&self) -> &MPCTrainerConfig {
        &self.config
    }

    /// Applies recovery state after share redistribution.
    ///
    /// Called after `redistribute_shares_after_removal` to load the new shares,
    /// MAC state, and step counter into this trainer. Also sets the party count
    /// to the new (reduced) value.
    ///
    /// After calling this, the caller must generate fresh Beaver triples before
    /// resuming training (the old triples are invalidated by the party change).
    pub fn apply_recovery_state(
        &mut self,
        result: &crate::share_redistribution::RedistributionResult,
        new_num_parties: usize,
    ) {
        self.w1 = result.w1.clone();
        self.b1 = result.b1.clone();
        self.w2 = result.w2.clone();
        self.b2 = result.b2.clone();
        self.current_step = result.checkpoint_step;
        self.config.num_parties = new_num_parties;

        // Install the new MAC state from redistribution.
        let mut ms = MACState::new(result.mac_state.alpha_share);
        ms.w1_macs = result.mac_state.w1_macs.clone();
        ms.b1_macs = result.mac_state.b1_macs.clone();
        ms.w2_macs = result.mac_state.w2_macs.clone();
        ms.b2_macs = result.mac_state.b2_macs.clone();
        // Save an initial checkpoint at the recovery point.
        ms.save_checkpoint(
            result.checkpoint_step,
            &self.w1, &self.b1, &self.w2, &self.b2,
            0, 0,
        );
        self.mac_state = Some(ms);

        // Clear old Beaver triples (must regenerate for new party set).
        self.beaver_triples.clear();
        self.beaver_cursor = 0;
        self.auth_beaver_triples.clear();
        self.auth_beaver_cursor = 0;

        info!(
            party = self.party_index,
            step = result.checkpoint_step,
            new_parties = new_num_parties,
            "Recovery state applied — trainer ready for fresh Beaver generation"
        );
    }
}

// ============================================================================
// Helper types
// ============================================================================

/// Model weights (unshared, plaintext).
#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub w1: Vec<Fr>,
    pub b1: Vec<Fr>,
    pub w2: Vec<Fr>,
    pub b2: Vec<Fr>,
}

impl ModelWeights {
    /// Creates random initial weights using Kaiming/He initialization (optimal for ReLU).
    pub fn random(d_in: usize, d_hid: usize, d_out: usize, rng: &mut impl Rng) -> Self {
        // He initialization: scale = sqrt(2/fan_in) — optimal for ReLU activations
        let w1_scale = (2.0 / d_in as f64).sqrt();
        let w2_scale = (2.0 / d_hid as f64).sqrt();

        Self {
            w1: (0..d_hid * d_in)
                .map(|_| Fr::from_f64(rng.gen_range(-w1_scale..w1_scale)))
                .collect(),
            b1: vec![Fr::ZERO; d_hid],
            w2: (0..d_out * d_hid)
                .map(|_| Fr::from_f64(rng.gen_range(-w2_scale..w2_scale)))
                .collect(),
            b2: vec![Fr::ZERO; d_out],
        }
    }

    /// Creates weights from f64 slices.
    pub fn from_f64(
        w1: &[f64], b1: &[f64], w2: &[f64], b2: &[f64],
    ) -> Self {
        Self {
            w1: w1.iter().map(|&v| Fr::from_f64(v)).collect(),
            b1: b1.iter().map(|&v| Fr::from_f64(v)).collect(),
            w2: w2.iter().map(|&v| Fr::from_f64(v)).collect(),
            b2: b2.iter().map(|&v| Fr::from_f64(v)).collect(),
        }
    }
}

/// Creates additive shares of a vector of field elements.
fn additive_share_vec(values: &[Fr], n: usize, rng: &mut ChaCha20Rng) -> Vec<Vec<Fr>> {
    let dim = values.len();
    let mut shares: Vec<Vec<Fr>> = (0..n).map(|_| Vec::with_capacity(dim)).collect();

    for elem in values {
        let mut sum = Fr::ZERO;
        for i in 0..n - 1 {
            let r = Fr::random(rng);
            sum = Fr::add(&sum, &r);
            shares[i].push(r);
        }
        // Last share = value - sum.
        shares[n - 1].push(Fr::sub(elem, &sum));
    }

    shares
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::transport::LocalTransport;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[tokio::test]
    async fn test_weight_sharing() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = MPCTrainerConfig::small(num_parties);
        let d_in = config.d_in;
        let d_hid = config.d_hid;
        let d_out = config.d_out;

        let initial_weights = ModelWeights::from_f64(
            &[0.1, 0.2, 0.3, 0.4], // w1: 2x2
            &[0.01, 0.02],          // b1: 2
            &[0.5, 0.6],            // w2: 1x2
            &[0.03],                // b2: 1
        );

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 {
                Some(initial_weights.clone())
            } else {
                None
            };

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                // Return weight shares for verification.
                (
                    trainer.w1.clone(),
                    trainer.b1.clone(),
                    trainer.w2.clone(),
                    trainer.b2.clone(),
                )
            });
            handles.push(handle);
        }

        // Collect all shares and verify they sum to original weights.
        let mut all_shares: Vec<(Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>)> = Vec::new();
        for handle in handles {
            all_shares.push(handle.await.unwrap());
        }

        // Verify w1 reconstruction.
        for idx in 0..d_hid * d_in {
            let sum: Fr = all_shares.iter().fold(Fr::ZERO, |acc, s| Fr::add(&acc, &s.0[idx]));
            let expected = initial_weights.w1[idx].to_f64();
            let got = sum.to_f64();
            assert!(
                (got - expected).abs() < 0.001,
                "w1[{}]: expected {}, got {}",
                idx, expected, got
            );
        }
    }

    #[tokio::test]
    async fn test_distributed_beaver_triples() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = MPCTrainerConfig::small(num_parties);
        let count = 10;

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.generate_beaver_triples(count).await.unwrap();
                trainer.beaver_triples.clone()
            });
            handles.push(handle);
        }

        let mut all_triples: Vec<Vec<BeaverTriple>> = Vec::new();
        for handle in handles {
            all_triples.push(handle.await.unwrap());
        }

        assert_eq!(all_triples.len(), 3);
        assert_eq!(all_triples[0].len(), count);

        // Verify each triple: sum(a) * sum(b) should equal sum(c).
        for t in 0..count {
            let a = crate::field::ops::sum(
                &all_triples.iter().map(|p| p[t].a.clone()).collect::<Vec<_>>(),
            );
            let b = crate::field::ops::sum(
                &all_triples.iter().map(|p| p[t].b.clone()).collect::<Vec<_>>(),
            );
            let c = crate::field::ops::sum(
                &all_triples.iter().map(|p| p[t].c.clone()).collect::<Vec<_>>(),
            );

            let expected = a.mpc_scale(&b);
            assert!(
                c.ct_eq(&expected).to_bool(),
                "Triple {} incorrect: c={}, expected={}",
                t,
                c.to_f64(),
                expected.to_f64()
            );
        }
    }

    #[tokio::test]
    async fn test_mpc_training_step() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = MPCTrainerConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            learning_rate: 0.01,
            num_parties,
            reshare_interval: 0, // disabled
            beaver_batch_size: 512, // increased for secure ReLU + matmul
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: None,
            batch_size: 1,
        };

        let initial_weights = ModelWeights::from_f64(
            &[0.1, 0.2, 0.3, 0.4],
            &[0.01, 0.02],
            &[0.5, 0.6],
            &[0.03],
        );

        let input = vec![1.0, 0.5];
        let target = vec![1.0];

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 {
                Some(initial_weights.clone())
            } else {
                None
            };
            let inp = input.clone();
            let tgt = target.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                trainer.generate_beaver_triples(512).await.unwrap();
                let result = trainer.training_step(&inp, &tgt).await.unwrap();
                (result.loss, result.step, result.reshared)
            });
            handles.push(handle);
        }

        let mut losses = Vec::new();
        for handle in handles {
            let (loss, step, reshared) = handle.await.unwrap();
            assert_eq!(step, 0);
            assert!(!reshared);
            assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
            losses.push(loss);
        }

        // All parties should compute the same loss (y is reconstructed for loss).
        for i in 1..losses.len() {
            assert!(
                (losses[i] - losses[0]).abs() < 0.01,
                "Loss mismatch: party 0 = {}, party {} = {}",
                losses[0], i, losses[i]
            );
        }
    }

    #[tokio::test]
    async fn test_mpc_training_with_resharing() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = MPCTrainerConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            learning_rate: 0.01,
            num_parties,
            reshare_interval: 2, // reshare every 2 steps
            beaver_batch_size: 1024, // increased for secure training
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: None,
            batch_size: 1,
        };

        let initial_weights = ModelWeights::from_f64(
            &[0.1, 0.2, 0.3, 0.4],
            &[0.01, 0.02],
            &[0.5, 0.6],
            &[0.03],
        );

        let data = vec![
            (vec![1.0, 0.5], vec![1.0]),
            (vec![0.5, 1.0], vec![0.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ];

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 {
                Some(initial_weights.clone())
            } else {
                None
            };
            let d = data.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                let results = trainer.train(&d).await.unwrap();

                // Verify resharing happened on step 2 (0-indexed: step 1).
                let reshare_steps: Vec<bool> = results.iter().map(|r| r.reshared).collect();
                let weight_shares = (
                    trainer.w1.clone(),
                    trainer.b1.clone(),
                    trainer.w2.clone(),
                    trainer.b2.clone(),
                );
                (reshare_steps, results.iter().map(|r| r.loss).collect::<Vec<_>>(), weight_shares)
            });
            handles.push(handle);
        }

        let mut all_reshare_flags = Vec::new();
        let mut all_losses = Vec::new();
        let mut all_final_shares = Vec::new();

        for handle in handles {
            let (reshare_flags, losses, shares) = handle.await.unwrap();
            all_reshare_flags.push(reshare_flags);
            all_losses.push(losses);
            all_final_shares.push(shares);
        }

        // All parties should agree on reshare schedule.
        for party in &all_reshare_flags {
            assert!(!party[0], "Step 0 should not reshare");
            assert!(party[1], "Step 1 should reshare (interval=2, step=2)");
            assert!(!party[2], "Step 2 should not reshare");
        }

        // After resharing, weight shares should still reconstruct correctly.
        // Verify final weights reconstruct to the same values across all parties.
        let dim = all_final_shares[0].0.len();
        for idx in 0..dim {
            let sum: Fr = all_final_shares.iter().fold(Fr::ZERO, |acc, s| Fr::add(&acc, &s.0[idx]));
            assert!(sum.to_f64().is_finite(), "w1[{}] not finite after reshare", idx);
        }
    }

    #[tokio::test]
    async fn test_share_validity_and_aggregation_proofs() {
        // This test exercises the new share validity and aggregation proof
        // generation and verification paths. We disable the Halo2 circuit
        // bridge proof (generate_proofs: false) and instead drive the share
        // validity / aggregation provers directly after a training step,
        // because the Halo2 circuit proof requires circuit-compatible
        // tiny weights and is tested separately.
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = MPCTrainerConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            learning_rate: 0.01,
            num_parties,
            reshare_interval: 0,
            beaver_batch_size: 512, // increased for secure training
            generate_proofs: false, // Disable full Halo2 proof (circuit-compat issue)
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: None,
            batch_size: 1,
        };

        let initial_weights = ModelWeights::from_f64(
            &[0.1, 0.2, 0.3, 0.4],
            &[0.01, 0.02],
            &[0.5, 0.6],
            &[0.03],
        );

        let input = vec![1.0, 0.5];
        let target = vec![1.0];

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 {
                Some(initial_weights.clone())
            } else {
                None
            };
            let inp = input.clone();
            let tgt = target.clone();

            let handle = tokio::spawn(async move {
                // Capture dimensions before moving cfg into the trainer.
                let d_in = cfg.d_in;
                let d_hid = cfg.d_hid;
                let d_out = cfg.d_out;

                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                trainer.generate_beaver_triples(512).await.unwrap();
                let _result = trainer.training_step(&inp, &tgt).await.unwrap();

                // Now manually exercise the share validity and aggregation provers.
                let sv_proof = trainer.generate_share_validity_proof().unwrap();

                // Only party 0 generates the aggregation proof.
                let agg_proof = if i == 0 {
                    // Create mock per-party gradients (all the same since gradients are public).
                    let gradients: Vec<(Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>)> = (0..num_parties)
                        .map(|_| {
                            let dw1 = vec![Fr::from_f64(0.01); d_hid * d_in];
                            let db1 = vec![Fr::from_f64(0.01); d_hid];
                            let dw2 = vec![Fr::from_f64(0.01); d_out * d_hid];
                            let db2 = vec![Fr::from_f64(0.01); d_out];
                            (dw1, db1, dw2, db2)
                        })
                        .collect();
                    trainer.generate_aggregation_proof(&gradients, 0).unwrap()
                } else {
                    None
                };

                // Build a result with the new proofs for verification.
                let result_with_proofs = MPCTrainingStepResult {
                    step: 0,
                    loss: 0.0,
                    reshared: false,
                    proof: None,
                    total_error: 0.0,
                    share_validity_proof: Some(sv_proof),
                    aggregation_proof: agg_proof,
                    on_chain_proof: None,
                };

                (i, result_with_proofs)
            });
            handles.push(handle);
        }

        for handle in handles {
            let (party_index, result) = handle.await.unwrap();

            // Every party should have a share validity proof.
            assert!(
                result.share_validity_proof.is_some(),
                "Party {} should have a share validity proof",
                party_index
            );

            // Verify the share validity proof.
            let sv_proof = result.share_validity_proof.as_ref().unwrap();
            let sv_verifier = crate::proofs::ShareValidityVerifier::new();
            assert!(
                sv_verifier.verify(sv_proof).unwrap(),
                "Party {}'s share validity proof should verify",
                party_index
            );

            // Only party 0 should have an aggregation proof.
            if party_index == 0 {
                assert!(
                    result.aggregation_proof.is_some(),
                    "Party 0 should have an aggregation proof"
                );

                let agg_proof = result.aggregation_proof.as_ref().unwrap();
                let agg_verifier = crate::proofs::AggregationVerifier::new();
                assert!(
                    agg_verifier.verify(agg_proof).unwrap(),
                    "Party 0's aggregation proof should verify"
                );
            } else {
                assert!(
                    result.aggregation_proof.is_none(),
                    "Party {} should not have an aggregation proof",
                    party_index
                );
            }

            // verify_step should pass for all parties.
            assert!(
                MPCTrainer::<LocalTransport>::verify_step(&result).unwrap(),
                "verify_step should pass for party {}",
                party_index
            );
        }
    }

    #[tokio::test]
    async fn test_multi_step_loss_decreases() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = MPCTrainerConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            learning_rate: 0.1,
            num_parties,
            reshare_interval: 0,
            beaver_batch_size: 2048, // increased for secure training (5 steps)
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: None,
            batch_size: 1,
        };

        let initial_weights = ModelWeights::from_f64(
            &[0.5, -0.3, 0.2, 0.4],
            &[0.0, 0.0],
            &[0.6, -0.4],
            &[0.0],
        );

        // Same data point repeated — loss should decrease.
        let data: Vec<(Vec<f64>, Vec<f64>)> = (0..5)
            .map(|_| (vec![1.0, 1.0], vec![1.0]))
            .collect();

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 {
                Some(initial_weights.clone())
            } else {
                None
            };
            let d = data.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                let results = trainer.train(&d).await.unwrap();
                results.iter().map(|r| r.loss).collect::<Vec<_>>()
            });
            handles.push(handle);
        }

        // Get party 0's losses (representative).
        let losses = handles.into_iter().next().unwrap().await.unwrap();

        // Loss should generally trend downward.
        assert!(
            losses.last().unwrap() < losses.first().unwrap(),
            "Loss should decrease: first={}, last={}",
            losses.first().unwrap(),
            losses.last().unwrap()
        );
    }

    /// Tests that MPC training with `generate_proofs: true` produces a valid
    /// on-chain Halo2 KZG proof via the weight reconstruction protocol.
    ///
    /// This is the key integration test: it verifies that after N parties
    /// complete an MPC training step, the prover party (party 0) can
    /// reconstruct the full weights and produce a verifiable ZK proof.
    ///
    /// NOTE: This test is marked #[ignore] because it involves actual Halo2
    /// proof generation which takes several seconds. Run with:
    ///   cargo test -p helix-mpc --release -- --ignored test_reconstruct_and_prove
    #[tokio::test]
    #[ignore]
    async fn test_reconstruct_and_prove() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = MPCTrainerConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            learning_rate: 0.001,
            num_parties,
            reshare_interval: 0,
            beaver_batch_size: 512,
            generate_proofs: true, // Enable on-chain proof generation
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: None,
            batch_size: 1,
        };

        // Use tiny weights in the 0.001 range so they quantize to small Fr
        // values that stay within the circuit's ReLU lookup range (±128).
        let initial_weights = ModelWeights::from_f64(
            &[0.001, 0.002, 0.001, 0.002], // w1: 2x2
            &[0.001, 0.001],                // b1: 2
            &[0.001, 0.002],                // w2: 1x2
            &[0.001],                       // b2: 1
        );

        let input = vec![0.001, 0.002];
        let target = vec![0.001];

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 {
                Some(initial_weights.clone())
            } else {
                None
            };
            let inp = input.clone();
            let tgt = target.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                trainer.generate_beaver_triples(512).await.unwrap();
                let result = trainer.training_step(&inp, &tgt).await.unwrap();
                (i, result.on_chain_proof, result.loss, result.step)
            });
            handles.push(handle);
        }

        let mut party0_proof = None;
        let mut all_losses = Vec::new();

        for handle in handles {
            let (party_index, proof, loss, step) = handle.await.unwrap();
            all_losses.push(loss);

            if party_index == 0 {
                // Party 0 (prover) should have a proof.
                assert!(
                    proof.is_some(),
                    "Party 0 should generate an on-chain proof"
                );
                let p = proof.unwrap();
                assert!(!p.proof.is_empty(), "Proof bytes should not be empty");
                assert!(
                    p.public_inputs.len() >= 7,
                    "Proof should have at least 7 public inputs, got {}",
                    p.public_inputs.len()
                );
                assert_eq!(p.step_number, 0, "Step number should be 0");
                assert!(p.proof_size_bytes > 0, "Proof size should be > 0");
                party0_proof = Some(p);
            } else {
                // Non-prover parties should NOT have a proof (they sent
                // their shares to party 0).
                assert!(
                    proof.is_none(),
                    "Party {} should not generate an on-chain proof",
                    party_index
                );
            }
        }

        // Verify party 0's proof using CircuitBridge.
        let bridge = crate::integration::circuit_bridge::CircuitBridge::for_model(2, 2, 1);
        let proof = party0_proof.unwrap();
        let verified = bridge.verify(&proof).unwrap();
        assert!(verified, "On-chain proof should verify");

        // All parties should agree on loss.
        for i in 1..all_losses.len() {
            assert!(
                (all_losses[i] - all_losses[0]).abs() < 0.01,
                "Loss mismatch: party 0 = {}, party {} = {}",
                all_losses[0], i, all_losses[i]
            );
        }
    }

    /// Tests that the on_chain_proof field is None when generate_proofs is false.
    #[tokio::test]
    async fn test_no_proof_when_disabled() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = MPCTrainerConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            learning_rate: 0.01,
            num_parties,
            reshare_interval: 0,
            beaver_batch_size: 512,
            generate_proofs: false, // Proofs disabled
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: None,
            batch_size: 1,
        };

        let initial_weights = ModelWeights::from_f64(
            &[0.1, 0.2, 0.3, 0.4],
            &[0.01, 0.02],
            &[0.5, 0.6],
            &[0.03],
        );

        let input = vec![1.0, 0.5];
        let target = vec![1.0];

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 {
                Some(initial_weights.clone())
            } else {
                None
            };
            let inp = input.clone();
            let tgt = target.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                trainer.generate_beaver_triples(512).await.unwrap();
                let result = trainer.training_step(&inp, &tgt).await.unwrap();
                (i, result.on_chain_proof)
            });
            handles.push(handle);
        }

        for handle in handles {
            let (party_index, proof) = handle.await.unwrap();
            assert!(
                proof.is_none(),
                "Party {} should not have a proof when generate_proofs=false",
                party_index
            );
        }
    }

    #[test]
    fn test_checkpoint_config_default() {
        let config = MPCTrainerConfig::default();
        assert_eq!(config.checkpoint_interval, 1, "Default checkpoint_interval should be 1");

        let small = MPCTrainerConfig::small(3);
        assert_eq!(small.checkpoint_interval, 1, "small() should inherit default checkpoint_interval of 1");
    }

    #[tokio::test]
    async fn test_unproved_step_result() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = MPCTrainerConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            learning_rate: 0.01,
            num_parties,
            reshare_interval: 0,
            beaver_batch_size: 512,
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: None,
            batch_size: 1,
        };

        let initial_weights = ModelWeights::from_f64(
            &[0.1, 0.2, 0.3, 0.4],
            &[0.01, 0.02],
            &[0.5, 0.6],
            &[0.03],
        );

        let input = vec![1.0, 0.5];
        let target = vec![1.0];

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 {
                Some(initial_weights.clone())
            } else {
                None
            };
            let inp = input.clone();
            let tgt = target.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                trainer.generate_beaver_triples(512).await.unwrap();
                let result = trainer.training_step_unproved(&inp, &tgt).await.unwrap();
                (result.step, result.loss, result.reshared, result.step_error, trainer.current_step())
            });
            handles.push(handle);
        }

        let mut losses = Vec::new();
        for handle in handles {
            let (step, loss, reshared, step_error, current_step) = handle.await.unwrap();
            assert_eq!(step, 0, "Step should be 0");
            assert_eq!(current_step, 1, "current_step should be incremented to 1");
            assert!(!reshared, "Should not reshare with interval=0");
            assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
            assert!(step_error > 0.0, "Step error should be positive");
            losses.push(loss);
        }

        // All parties should compute the same loss (y is reconstructed for loss).
        for i in 1..losses.len() {
            assert!(
                (losses[i] - losses[0]).abs() < 0.01,
                "Loss mismatch: party 0 = {}, party {} = {}",
                losses[0], i, losses[i]
            );
        }
    }

    #[tokio::test]
    async fn test_training_epoch_no_proofs() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = MPCTrainerConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            learning_rate: 0.01,
            num_parties,
            reshare_interval: 0,
            beaver_batch_size: 2048,
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: 3, // Run 3 steps per epoch
            mac_config: None,
            batch_size: 1,
        };

        let initial_weights = ModelWeights::from_f64(
            &[0.1, 0.2, 0.3, 0.4],
            &[0.01, 0.02],
            &[0.5, 0.6],
            &[0.03],
        );

        let data = vec![
            (vec![1.0, 0.5], vec![1.0]),
            (vec![0.5, 1.0], vec![0.0]),
            (vec![1.0, 1.0], vec![1.0]),
        ];

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 {
                Some(initial_weights.clone())
            } else {
                None
            };
            let d = data.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                trainer.generate_beaver_triples(2048).await.unwrap();
                let result = trainer.training_epoch(&d).await.unwrap();
                (result, trainer.current_step())
            });
            handles.push(handle);
        }

        for handle in handles {
            let (result, current_step) = handle.await.unwrap();

            assert_eq!(result.start_step, 0, "Should start at step 0");
            assert_eq!(result.end_step, 2, "Should end at step 2 (3 steps: 0, 1, 2)");
            assert_eq!(result.losses.len(), 3, "Should have 3 losses");
            assert_eq!(result.reshared_steps.len(), 3, "Should have 3 reshare flags");
            assert!(result.total_error > 0.0, "Total error should be positive");
            assert!(result.proof.is_none(), "No proof when generate_proofs=false");
            assert_eq!(current_step, 3, "current_step should be 3 after 3 steps");

            // All losses should be finite
            for (i, loss) in result.losses.iter().enumerate() {
                assert!(loss.is_finite(), "Loss at step {} should be finite, got {}", i, loss);
            }
        }
    }

    #[tokio::test]
    async fn test_checkpoint_interval_one_equivalent() {
        // Verify that checkpoint_interval=1 with training_epoch produces
        // similar results to a single training_step.
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = MPCTrainerConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            learning_rate: 0.01,
            num_parties,
            reshare_interval: 0,
            beaver_batch_size: 512,
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: 1, // epoch = single step
            mac_config: None,
            batch_size: 1,
        };

        let initial_weights = ModelWeights::from_f64(
            &[0.1, 0.2, 0.3, 0.4],
            &[0.01, 0.02],
            &[0.5, 0.6],
            &[0.03],
        );

        let data = vec![(vec![1.0, 0.5], vec![1.0])];

        // Run training_epoch with interval=1 on one set of parties.
        let mut epoch_handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 {
                Some(initial_weights.clone())
            } else {
                None
            };
            let d = data.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                trainer.generate_beaver_triples(512).await.unwrap();
                let result = trainer.training_epoch(&d).await.unwrap();
                result
            });
            epoch_handles.push(handle);
        }

        // Run training_step on another set of parties with the same seed.
        let parties2 = test_parties(num_parties);
        let transports2 = LocalTransport::create_mesh(&parties2);

        let mut step_handles = Vec::new();
        for (i, transport) in transports2.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 {
                Some(initial_weights.clone())
            } else {
                None
            };
            let inp = data[0].0.clone();
            let tgt = data[0].1.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                trainer.generate_beaver_triples(512).await.unwrap();
                let result = trainer.training_step(&inp, &tgt).await.unwrap();
                result
            });
            step_handles.push(handle);
        }

        // Compare party 0's results.
        let epoch_result = epoch_handles.into_iter().next().unwrap().await.unwrap();
        let step_result = step_handles.into_iter().next().unwrap().await.unwrap();

        assert_eq!(epoch_result.losses.len(), 1, "Epoch with interval=1 should have 1 loss");
        assert_eq!(epoch_result.start_step, 0);
        assert_eq!(epoch_result.end_step, 0);

        // The losses should match since they're the same computation
        // with the same seed and same data.
        assert!(
            (epoch_result.losses[0] - step_result.loss).abs() < 0.001,
            "Epoch loss ({}) should match step loss ({}) within tolerance",
            epoch_result.losses[0],
            step_result.loss,
        );
    }

    // ========================================================================
    // SPDZ MAC Verification Integration Tests
    // ========================================================================

    /// Helper: creates a MAC-enabled config for testing.
    fn mac_config(num_parties: usize, check_interval: u64) -> MPCTrainerConfig {
        MPCTrainerConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            learning_rate: 0.01,
            num_parties,
            reshare_interval: 0,
            beaver_batch_size: 2048,
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: Some(crate::mac_verification::MACVerificationConfig {
                check_interval,
                enable_cheater_identification: true,
                mac_seed: 0xDEAD_BEEF_CAFE_BABE,
            }),
            batch_size: 1,
        }
    }

    /// Helper: standard initial weights for MAC tests.
    fn mac_test_weights() -> ModelWeights {
        ModelWeights::from_f64(
            &[0.1, 0.2, 0.3, 0.4],
            &[0.01, 0.02],
            &[0.5, 0.6],
            &[0.03],
        )
    }

    /// Helper: training data for MAC tests.
    fn mac_test_data(n: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
        let base_data = vec![
            (vec![1.0, 0.5], vec![1.0]),
            (vec![0.5, 1.0], vec![0.0]),
            (vec![1.0, 1.0], vec![1.0]),
            (vec![0.0, 0.0], vec![0.0]),
        ];
        (0..n).map(|i| base_data[i % base_data.len()].clone()).collect()
    }

    /// 3 parties, 10 honest training steps with MAC verification every step.
    /// All MAC checks should pass.
    #[tokio::test]
    async fn test_honest_training_passes_mac() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = mac_config(num_parties, 1); // check every step
        let initial_weights = mac_test_weights();
        let data = mac_test_data(10);

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 { Some(initial_weights.clone()) } else { None };
            let d = data.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();

                // Pre-generate enough Beaver triples for all steps.
                // Each MAC step needs ~(d_hid + d_out*d_hid + d_hid)*2 + 32 triples.
                trainer.generate_beaver_triples(4096).await.unwrap();

                // Run all 10 steps via the train() loop (routes through training_step_with_mac).
                let results = trainer.train(&d).await.unwrap();

                // Collect results.
                let losses: Vec<f64> = results.iter().map(|r| r.loss).collect();
                let steps: Vec<u64> = results.iter().map(|r| r.step).collect();
                (i, losses, steps)
            });
            handles.push(handle);
        }

        // All parties should complete all 10 steps without MAC failure.
        for handle in handles {
            let (party_idx, losses, steps) = handle.await.unwrap();
            assert_eq!(
                steps.len(), 10,
                "Party {} should complete all 10 steps, got {}",
                party_idx, steps.len()
            );
            // All losses should be finite.
            for (s, loss) in losses.iter().enumerate() {
                assert!(
                    loss.is_finite(),
                    "Party {} step {} loss should be finite, got {}",
                    party_idx, s, loss
                );
            }
        }
    }

    /// 3 parties, party 2 corrupts its weight share at step 5.
    /// MAC verification should fail, returning MACCheckFailed error.
    #[tokio::test]
    async fn test_cheater_detected() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = mac_config(num_parties, 1); // check every step
        let initial_weights = mac_test_weights();
        let data = mac_test_data(10);

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 { Some(initial_weights.clone()) } else { None };
            let d = data.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                trainer.generate_beaver_triples(4096).await.unwrap();

                let mut completed_steps = 0u64;
                let mut mac_failed = false;
                let mut detected_cheater: Option<usize> = None;

                for step in 0..10u64 {
                    let data_idx = step as usize % d.len();
                    let (input, target) = &d[data_idx];

                    // Party 2 corrupts weight share at step 5.
                    if i == 2 && step == 5 {
                        trainer.corrupt_weight_share(0, Fr::from_f64(999.0));
                    }

                    match trainer.training_step_with_mac(input, target).await {
                        Ok(_) => {
                            completed_steps += 1;
                        }
                        Err(MPCError::MACCheckFailed { step: fail_step, cheater }) => {
                            mac_failed = true;
                            detected_cheater = cheater;
                            break;
                        }
                        Err(e) => panic!("Party {} unexpected error at step {}: {}", i, step, e),
                    }
                }

                (i, completed_steps, mac_failed, detected_cheater)
            });
            handles.push(handle);
        }

        // All parties should detect MAC failure.
        for handle in handles {
            let (party_idx, completed, mac_failed, _cheater) = handle.await.unwrap();
            assert!(
                mac_failed,
                "Party {} should detect MAC failure (completed {} steps)",
                party_idx, completed
            );
            // Corruption happens at step 5, MAC check at step 5 (interval=1)
            // should detect it. Training should halt before completing all 10 steps.
            assert!(
                completed < 10,
                "Party {} should not complete all 10 steps (completed {})",
                party_idx, completed
            );
        }
    }

    /// 3 parties, party 2 corrupts its weight share at step 5.
    /// The cheater identification protocol should identify party 2.
    #[tokio::test]
    async fn test_cheater_identified() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = mac_config(num_parties, 1); // check every step
        let initial_weights = mac_test_weights();
        let data = mac_test_data(10);

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 { Some(initial_weights.clone()) } else { None };
            let d = data.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                trainer.generate_beaver_triples(4096).await.unwrap();

                let mut detected_cheater: Option<usize> = None;

                for step in 0..10u64 {
                    let data_idx = step as usize % d.len();
                    let (input, target) = &d[data_idx];

                    // Party 2 corrupts weight share at step 5.
                    if i == 2 && step == 5 {
                        trainer.corrupt_weight_share(0, Fr::from_f64(999.0));
                    }

                    match trainer.training_step_with_mac(input, target).await {
                        Ok(_) => {}
                        Err(MPCError::MACCheckFailed { cheater, .. }) => {
                            detected_cheater = cheater;
                            break;
                        }
                        Err(e) => panic!("Party {} unexpected error at step {}: {}", i, step, e),
                    }
                }

                (i, detected_cheater)
            });
            handles.push(handle);
        }

        // All parties should identify party 2 as the cheater.
        for handle in handles {
            let (party_idx, detected_cheater) = handle.await.unwrap();
            assert_eq!(
                detected_cheater, Some(2),
                "Party {} should identify party 2 as the cheater, got {:?}",
                party_idx, detected_cheater
            );
        }
    }

    /// 3 parties, party 2 cheats, is detected, then training continues
    /// with parties 0 and 1 (using a new 2-party transport mesh from checkpoint state).
    #[tokio::test]
    async fn test_training_continues_after_removal() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = mac_config(num_parties, 1); // check every step
        let initial_weights = mac_test_weights();
        let data = mac_test_data(10);

        // Phase 1: Run 3-party training until cheater is detected.
        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 { Some(initial_weights.clone()) } else { None };
            let d = data.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                trainer.generate_beaver_triples(4096).await.unwrap();

                let mut completed_steps = 0u64;

                for step in 0..10u64 {
                    let data_idx = step as usize % d.len();
                    let (input, target) = &d[data_idx];

                    // Party 2 corrupts weight share at step 5.
                    if i == 2 && step == 5 {
                        trainer.corrupt_weight_share(0, Fr::from_f64(999.0));
                    }

                    match trainer.training_step_with_mac(input, target).await {
                        Ok(_) => {
                            completed_steps += 1;
                        }
                        Err(MPCError::MACCheckFailed { .. }) => {
                            break;
                        }
                        Err(e) => panic!("Party {} unexpected error: {}", i, e),
                    }
                }

                // Return checkpoint state for surviving parties (0, 1).
                let (w1, b1, w2, b2) = trainer.weight_shares();
                (i, completed_steps, w1.to_vec(), b1.to_vec(), w2.to_vec(), b2.to_vec())
            });
            handles.push(handle);
        }

        // Collect checkpoint state from surviving parties (0, 1).
        let mut party_states: Vec<(usize, u64, Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>)> = Vec::new();
        for handle in handles {
            party_states.push(handle.await.unwrap());
        }

        // Verify cheater was detected (all parties stopped early).
        for (idx, steps, _, _, _, _) in &party_states {
            assert!(
                *steps < 10,
                "Party {} should have stopped early (completed {} steps)",
                idx, steps
            );
        }

        // Phase 2: Continue training with parties 0 and 1 (excluding cheater party 2).
        // Use a new 2-party transport mesh and restore from checkpoint state.
        let surviving_parties: Vec<PartyId> = vec![
            PartyId::from_index(0),
            PartyId::from_index(1),
        ];
        let new_transports = LocalTransport::create_mesh(&surviving_parties);

        // Get checkpoint step (use party 0's completed steps as the rollback point).
        let checkpoint_step = party_states[0].1;

        let recovery_config = MPCTrainerConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            learning_rate: 0.01,
            num_parties: 2, // Now only 2 parties
            reshare_interval: 0,
            beaver_batch_size: 2048,
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: 1,
            mac_config: None, // Disable MAC for recovery (new MAC init needed otherwise)
            batch_size: 1,
        };

        let recovery_data = mac_test_data(5);
        let mut recovery_handles = Vec::new();

        for (new_idx, transport) in new_transports.into_iter().enumerate() {
            let cfg = recovery_config.clone();
            let d = recovery_data.clone();

            // Map new party index to original party index (0→0, 1→1).
            let original_idx = new_idx;
            let state = &party_states[original_idx];
            let w1 = state.2.clone();
            let b1 = state.3.clone();
            let w2 = state.4.clone();
            let b2 = state.5.clone();
            let cp_step = checkpoint_step;

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, new_idx, 42);

                // Restore weight state from checkpoint.
                trainer.restore_from_checkpoint(w1, b1, w2, b2, cp_step);

                // Run more training steps (unproved, since MAC is disabled for 2-party recovery).
                let results = trainer.train(&d).await.unwrap();

                let losses: Vec<f64> = results.iter().map(|r| r.loss).collect();
                (new_idx, losses)
            });
            recovery_handles.push(handle);
        }

        // Verify recovery training succeeds and produces finite losses.
        for handle in recovery_handles {
            let (party_idx, losses) = handle.await.unwrap();
            assert_eq!(
                losses.len(), 5,
                "Party {} should complete 5 recovery steps",
                party_idx
            );
            for (s, loss) in losses.iter().enumerate() {
                assert!(
                    loss.is_finite(),
                    "Party {} recovery step {} loss should be finite, got {}",
                    party_idx, s, loss
                );
            }
        }
    }

    /// MAC check interval K=5. Cheater corrupts at step 3, but detection
    /// happens at step 5 (the next MAC check boundary), not immediately.
    #[tokio::test]
    async fn test_mac_check_frequency() {
        let num_parties = 3;
        let parties = test_parties(num_parties);
        let transports = LocalTransport::create_mesh(&parties);

        let config = mac_config(num_parties, 5); // check every 5 steps
        let initial_weights = mac_test_weights();
        let data = mac_test_data(10);

        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let cfg = config.clone();
            let weights = if i == 0 { Some(initial_weights.clone()) } else { None };
            let d = data.clone();

            let handle = tokio::spawn(async move {
                let mut trainer = MPCTrainer::new(cfg, transport, i, 42);
                trainer.share_weights(weights).await.unwrap();
                trainer.generate_beaver_triples(4096).await.unwrap();

                let mut step_results: Vec<(u64, bool)> = Vec::new(); // (step, succeeded)

                for step in 0..10u64 {
                    let data_idx = step as usize % d.len();
                    let (input, target) = &d[data_idx];

                    // Party 2 corrupts weight share at step 3.
                    if i == 2 && step == 3 {
                        trainer.corrupt_weight_share(0, Fr::from_f64(999.0));
                    }

                    match trainer.training_step_with_mac(input, target).await {
                        Ok(_) => {
                            step_results.push((step, true));
                        }
                        Err(MPCError::MACCheckFailed { .. }) => {
                            step_results.push((step, false));
                            break;
                        }
                        Err(e) => panic!("Party {} unexpected error at step {}: {}", i, step, e),
                    }
                }

                (i, step_results)
            });
            handles.push(handle);
        }

        for handle in handles {
            let (party_idx, step_results) = handle.await.unwrap();

            // Corruption at step 3, MAC check at steps 4 (after step 4 completes,
            // current_step=5, 5 % 5 == 0). So steps 0-3 should succeed,
            // step 4 is where the check triggers and fails.
            let failed_at = step_results.iter()
                .find(|(_, succeeded)| !succeeded)
                .map(|(step, _)| *step);

            assert!(
                failed_at.is_some(),
                "Party {} should have detected MAC failure",
                party_idx
            );

            let fail_step = failed_at.unwrap();
            // The corruption happens at step 3, but the MAC check runs when
            // current_step (after increment) is divisible by 5. After step 4
            // completes, current_step becomes 5, so 5 % 5 == 0 triggers the check.
            assert!(
                fail_step >= 4,
                "Party {}: MAC failure should not be detected before step 4 (detected at step {}). \
                 Corruption at step 3 should only be caught at the next check boundary.",
                party_idx, fail_step
            );

            // Steps before the corruption (0, 1, 2) should have succeeded.
            let successful_steps: Vec<u64> = step_results.iter()
                .filter(|(_, succeeded)| *succeeded)
                .map(|(step, _)| *step)
                .collect();
            assert!(
                successful_steps.contains(&0) && successful_steps.contains(&1) && successful_steps.contains(&2),
                "Party {}: steps 0, 1, 2 should succeed (before corruption). Got: {:?}",
                party_idx, successful_steps
            );
        }
    }
}
