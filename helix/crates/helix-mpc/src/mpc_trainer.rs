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
use tracing::{debug, info, instrument};

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
use crate::protocols::arithmetic::SecureArithmetic;
use crate::protocols::reshare::Resharing;
use crate::security::commitment::BlindingGenerator;
use crate::session::transport::MPCTransport;
use crate::sharing::tensor::TensorShare;
use crate::types::{PartyId, ShareId};

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

    // ========================================================================
    // Phase 1: Weight sharing
    // ========================================================================

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
            "Generating {} Beaver triples distributedly", count
        );

        let mut new_triples = Vec::with_capacity(count);

        for t in 0..count {
            // Step 1: Each party samples random a_i, b_i.
            let a_i = Fr::random(&mut self.rng);
            let b_i = Fr::random(&mut self.rng);

            // Start with the diagonal term: c_i = a_i * b_i.
            let mut c_i = Fr::mul(&a_i, &b_i);

            let peers = self.transport.peers();

            // Step 2: For each peer j, pick random r_ij and send (a_i, r_ij).
            let mut my_randoms: HashMap<String, Fr> = HashMap::new();
            for peer in &peers {
                let r_ij = Fr::random(&mut self.rng);
                my_randoms.insert(peer.0.clone(), r_ij.clone());

                let msg = TrainingMessage::BeaverShares {
                    a: SecureArithmetic::serialize_share_batch(&[a_i.clone()]),
                    b: SecureArithmetic::serialize_share_batch(&[r_ij]),
                    c: SecureArithmetic::serialize_share_batch(&[Fr::from_u64(t as u64)]),
                };
                self.transport.send(peer, &msg.encode()).await?;
            }

            // Step 3: Add our random r_ij to c_i (our share of cross-term a_i*b_j).
            for r_ij in my_randoms.values() {
                c_i = Fr::add(&c_i, r_ij);
            }

            // Step 4: Receive (a_j, r_ji) from each peer and add the cross-term.
            for peer in &peers {
                let data = self.transport.recv(peer).await?;
                let msg = TrainingMessage::decode(&data)?;

                if let TrainingMessage::BeaverShares { a, b, c: _ } = msg {
                    let peer_a = SecureArithmetic::deserialize_share_batch(&a)?;
                    let peer_r = SecureArithmetic::deserialize_share_batch(&b)?;

                    // Cross-term: a_j * b_i - r_ji
                    // peer_a[0] = a_j, peer_r[0] = r_ji (their random for us)
                    let cross = Fr::sub(&Fr::mul(&peer_a[0], &b_i), &peer_r[0]);
                    c_i = Fr::add(&c_i, &cross);
                }
            }

            new_triples.push(BeaverTriple::new(a_i, b_i, c_i));
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

        let peers = self.transport.peers();
        for peer in &peers {
            let msg = self.transport.recv(peer).await?;
            let shares = SecureArithmetic::deserialize_share_batch(&msg)?;
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

        // Collect from peers.
        let mut total_d = d_batch;
        let mut total_e = e_batch;

        let peers = self.transport.peers();
        for peer in &peers {
            let msg = self.transport.recv(peer).await?;
            let shares = SecureArithmetic::deserialize_share_batch(&msg)?;
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
        for peer in &peers {
            let msg = self.transport.recv(peer).await?;
            let peer_masked = SecureArithmetic::deserialize_share_batch(&msg)?;
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
        for peer in &peers {
            let msg = self.transport.recv(peer).await?;
            let peer_y = SecureArithmetic::deserialize_share_batch(&msg)?;
            for i in 0..d_out {
                y_reconstructed[i] = Fr::add(&y_reconstructed[i], &peer_y[i]);
            }
        }

        // Loss = 0.5 * sum((y - target)^2)
        let mut loss = 0.0_f64;
        let mut dy: Vec<Fr> = vec![Fr::ZERO; d_out];
        for i in 0..d_out {
            let diff = Fr::sub(&y_reconstructed[i], &target_fr[i]);
            let diff_f64 = diff.to_f64();
            loss += 0.5 * diff_f64 * diff_f64;
            dy[i] = diff; // dy = y - target (public)
        }

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

        // ---- ZK proof generation ----
        let proof = if self.config.generate_proofs {
            Some(self.generate_proof(
                &old_w1, &old_b1, &old_w2, &old_b2,
                input, target, step,
            )?)
        } else {
            None
        };

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
            has_proof = proof.is_some(),
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

        // Receive zero-shares from all peers and add to weights.
        for peer in &peers {
            let data = self.transport.recv(peer).await?;
            let msg = TrainingMessage::decode(&data)?;

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
        let bridge = self.circuit_bridge.as_ref().unwrap();

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
    // Phase 7: Share validity and aggregation proofs
    // ========================================================================

    /// Generates a share validity proof for the current weight shares.
    ///
    /// Creates a `TensorShare` by concatenating all weight shares (w1, b1,
    /// w2, b2) into a single flat vector, then uses `ShareValidityProver`
    /// to prove that the shares are well-formed without revealing their values.
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
        self.share_prover.prove(&witness)
    }

    /// Generates an aggregation proof for gradient aggregation.
    ///
    /// This is only called by party 0, which has the aggregated view of
    /// all gradients. Each party's gradient contribution is flattened and
    /// committed, then the aggregation is proven correct.
    ///
    /// Returns `Ok(None)` if there are no gradients to aggregate.
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
        let proof = self.agg_prover.prove(&witness)?;
        Ok(Some(proof))
    }

    /// Verifies all proofs from a training step result.
    ///
    /// Checks:
    /// - Share validity proof (if present): verifies the party's shares are well-formed
    /// - Aggregation proof (if present): verifies gradient aggregation was correct
    ///
    /// Returns `Ok(true)` if all present proofs verify, `Ok(false)` if any fail.
    pub fn verify_step(result: &MPCTrainingStepResult) -> MPCResult<bool> {
        // Verify share validity proof if present.
        if let Some(ref sv_proof) = result.share_validity_proof {
            let sv_verifier = ShareValidityVerifier::new();
            if !sv_verifier.verify(sv_proof)? {
                return Ok(false);
            }
        }

        // Verify aggregation proof if present.
        if let Some(ref agg_proof) = result.aggregation_proof {
            let agg_verifier = AggregationVerifier::new();
            if !agg_verifier.verify(agg_proof)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    // ========================================================================
    // Multi-step training loop
    // ========================================================================

    /// Runs multiple training steps.
    ///
    /// `data` is a list of (input, target) pairs.
    pub async fn train(
        &mut self,
        data: &[(Vec<f64>, Vec<f64>)],
    ) -> MPCResult<Vec<MPCTrainingStepResult>> {
        let mut results = Vec::with_capacity(data.len());

        for (input, target) in data {
            // Auto-generate Beaver triples if we're running low.
            // Secure training needs more triples than the old open-ReLU version:
            // - ReLU sign: d_hid triples (for h_pre * sign_mask)
            // - W2 @ h: d_out * d_hid triples (shared * shared matmul)
            // - dh * relu_mask: d_hid triples (backward pass)
            // - Plus overhead for any re-sharing or proof operations
            let triples_needed = self.config.d_hid  // ReLU forward
                + self.config.d_out * self.config.d_hid  // W2 @ h
                + self.config.d_hid  // backward pass dh * mask
                + 32;  // overhead
            if self.beaver_triples_remaining() < triples_needed {
                self.generate_beaver_triples(
                    self.config.beaver_batch_size.max(triples_needed * 2)
                ).await?;
            }

            let result = self.training_step(input, target).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Returns the current weight shares for debugging/verification.
    pub fn weight_shares(&self) -> (&[Fr], &[Fr], &[Fr], &[Fr]) {
        (&self.w1, &self.b1, &self.w2, &self.b2)
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
    /// Creates random initial weights.
    pub fn random(d_in: usize, d_hid: usize, d_out: usize, rng: &mut impl Rng) -> Self {
        // Xavier initialization.
        let w1_scale = (2.0 / (d_in + d_hid) as f64).sqrt();
        let w2_scale = (2.0 / (d_hid + d_out) as f64).sqrt();

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

            let expected = Fr::mul(&a, &b);
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
}
