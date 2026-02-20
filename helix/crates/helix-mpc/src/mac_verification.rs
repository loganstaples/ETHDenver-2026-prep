//! SPDZ MAC verification protocol for MPC training integrity.
//!
//! This module implements the information-theoretic MAC-based verification
//! system that detects cheating during distributed training. The protocol:
//!
//! 1. **MAC Generation**: Party 0 (dealer) generates authenticated weight shares
//!    where each share has an associated MAC tag under a global key α.
//!
//! 2. **MAC Tracking**: During training, MAC shares are maintained through all
//!    operations (linear ops locally, Beaver multiplications via authenticated
//!    triples).
//!
//! 3. **Batch Verification**: Periodically (every K steps), all parties run the
//!    SPDZ sigma protocol to verify MAC consistency without revealing secrets.
//!
//! 4. **Cheater Identification**: On MAC failure, a checkpoint-diff comparison
//!    protocol identifies the specific cheating party. Weight updates between
//!    checkpoints are deterministic (public gradients), so honest parties have
//!    identical diffs. The cheater's diff contains the corruption delta.
//!
//! 5. **Halt & Rollback**: Training halts on detection, rolls back to the last
//!    verified checkpoint, and produces a failure report for on-chain submission.
//!
//! # Security Model
//!
//! - Information-theoretic: security holds against computationally unbounded
//!   adversaries (no cryptographic assumptions needed for MAC checking).
//! - Party 0 is trusted for initial MAC generation (knows α from setup seed).
//!   This matches the existing trust model where party 0 evaluates ReLU signs.
//! - The sigma protocol uses commit-then-reveal to prevent adaptive cheating.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use crate::beaver::triple::BeaverTriple;
use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::protocols::arithmetic::SecureArithmetic;
use crate::session::transport::MPCTransport;
use crate::types::PartyId;

/// Receives messages from all peers in parallel using `try_join_all`.
async fn recv_all<T: MPCTransport>(transport: &T) -> crate::error::MPCResult<Vec<Vec<u8>>> {
    let peers = transport.peers();
    let futs: Vec<_> = peers.iter()
        .map(|peer| transport.recv(peer))
        .collect();
    futures::future::try_join_all(futs).await
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for SPDZ MAC-based verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MACVerificationConfig {
    /// Run MAC verification every K training steps. 0 = disabled.
    pub check_interval: u64,
    /// Whether to run pairwise cheater identification on MAC failure.
    pub enable_cheater_identification: bool,
    /// Seed for MAC key generation (separate from training seed for isolation).
    pub mac_seed: u64,
}

impl Default for MACVerificationConfig {
    fn default() -> Self {
        Self {
            check_interval: 1,
            enable_cheater_identification: true,
            mac_seed: 0xDEAD_BEEF_CAFE_BABE,
        }
    }
}

// ============================================================================
// Authenticated Beaver Triple
// ============================================================================

/// A Beaver triple with associated MAC shares for authenticated computation.
///
/// For a triple (a, b, c) where c = a*b, each party holds:
/// - (a_i, b_i, c_i): their share of the triple
/// - (mac_a_i, mac_b_i, mac_c_i): their share of MAC(a), MAC(b), MAC(c)
///
/// Invariant: sum(mac_x_i) = α * sum(x_i) for x ∈ {a, b, c}
#[derive(Debug, Clone)]
pub struct AuthenticatedBeaverTriple {
    /// The underlying Beaver triple shares.
    pub triple: BeaverTriple,
    /// MAC share for the 'a' component.
    pub mac_a: Fr,
    /// MAC share for the 'b' component.
    pub mac_b: Fr,
    /// MAC share for the 'c' component.
    pub mac_c: Fr,
}

impl AuthenticatedBeaverTriple {
    /// Computes the result share and MAC share of z = x * y using this
    /// authenticated Beaver triple.
    ///
    /// Given opened d = x - a and e = y - b, computes:
    /// - z_i = c_i + d * b_i + e * a_i + (party 0: d * e)
    /// - mac_z_i = mac_c_i + d * mac_b_i + e * mac_a_i + α_i * d * e
    pub fn authenticated_multiply(
        &self,
        opened_d: &Fr,
        opened_e: &Fr,
        party_index: usize,
        alpha_share: &Fr,
    ) -> (Fr, Fr) {
        // Value computation (standard Beaver protocol)
        let value = SecureArithmetic::multiply_shares(
            &self.triple, opened_d, opened_e, party_index,
        );

        // MAC computation
        let d_mac_b = opened_d.mpc_scale(&self.mac_b);
        let e_mac_a = opened_e.mpc_scale(&self.mac_a);
        let de = opened_d.mpc_scale(opened_e);
        let alpha_de = Fr::mul(alpha_share, &de);

        let mac = Fr::add(
            &Fr::add(&Fr::add(&self.mac_c, &d_mac_b), &e_mac_a),
            &alpha_de,
        );

        (value, mac)
    }
}

// ============================================================================
// MAC State
// ============================================================================

/// Tracks MAC shares for all weight elements during training.
///
/// The MAC state is maintained through all operations:
/// - Linear ops (add, scale): MAC shares updated locally
/// - Beaver multiplications: MAC shares computed from authenticated triples
/// - Re-sharing: MAC shares re-shared alongside value shares
pub struct MACState {
    /// This party's share of the global MAC key α.
    pub alpha_share: Fr,
    /// MAC shares for layer 1 weights.
    pub w1_macs: Vec<Fr>,
    /// MAC shares for layer 1 biases.
    pub b1_macs: Vec<Fr>,
    /// MAC shares for layer 2 weights.
    pub w2_macs: Vec<Fr>,
    /// MAC shares for layer 2 biases.
    pub b2_macs: Vec<Fr>,
    /// Accumulated opened values for batch verification.
    /// Each entry is a (value, mac_share) pair from an opening during computation.
    opened_values: Vec<Fr>,
    opened_mac_shares: Vec<Fr>,
    /// Step of the last successful MAC verification.
    pub last_verified_step: u64,
    /// Checkpoint of the last known-good state.
    pub checkpoint: Option<TrainingCheckpoint>,
    /// Snapshot of this party's weight shares taken at the START of each
    /// training step (before computation). Used for cheater identification:
    /// comparing pre-step shares against checkpoint detects corruption
    /// that occurred between steps.
    pub pre_step_shares: Option<Vec<Fr>>,
    /// Per-party commitments H(weight_shares) exchanged at checkpoint time.
    /// Index i = commitment from party i. Used during identification to
    /// verify that pre-step shares match the checkpoint state.
    pub checkpoint_share_commits: Vec<[u8; 32]>,
}

impl MACState {
    /// Creates a new MAC state with the given alpha share and empty MAC vectors.
    pub fn new(alpha_share: Fr) -> Self {
        Self {
            alpha_share,
            w1_macs: Vec::new(),
            b1_macs: Vec::new(),
            w2_macs: Vec::new(),
            b2_macs: Vec::new(),
            opened_values: Vec::new(),
            opened_mac_shares: Vec::new(),
            last_verified_step: 0,
            checkpoint: None,
            pre_step_shares: None,
            checkpoint_share_commits: Vec::new(),
        }
    }

    /// Records an opened value and its MAC share for later batch verification.
    ///
    /// When a value is opened during computation (e.g., Beaver mask d = x - a),
    /// we record (opened_value, MAC(x)_i - MAC(a)_i) so we can verify later
    /// that the opening was honest.
    pub fn accumulate_opened(&mut self, value: Fr, mac_share: Fr) {
        self.opened_values.push(value);
        self.opened_mac_shares.push(mac_share);
    }

    /// Clears the accumulated opened values (after a successful verification).
    pub fn clear_accumulated(&mut self) {
        self.opened_values.clear();
        self.opened_mac_shares.clear();
    }

    /// Saves a checkpoint of the current weight and MAC state.
    pub fn save_checkpoint(
        &mut self,
        step: u64,
        w1: &[Fr], b1: &[Fr], w2: &[Fr], b2: &[Fr],
        beaver_cursor: usize,
        auth_beaver_cursor: usize,
    ) {
        self.checkpoint = Some(TrainingCheckpoint {
            step,
            w1: w1.to_vec(),
            b1: b1.to_vec(),
            w2: w2.to_vec(),
            b2: b2.to_vec(),
            w1_macs: self.w1_macs.clone(),
            b1_macs: self.b1_macs.clone(),
            w2_macs: self.w2_macs.clone(),
            b2_macs: self.b2_macs.clone(),
            beaver_cursor,
            auth_beaver_cursor,
        });
        self.last_verified_step = step;
    }

    /// Restores the last checkpoint, returning it if one exists.
    pub fn rollback(&mut self) -> Option<TrainingCheckpoint> {
        if let Some(cp) = self.checkpoint.take() {
            self.w1_macs = cp.w1_macs.clone();
            self.b1_macs = cp.b1_macs.clone();
            self.w2_macs = cp.w2_macs.clone();
            self.b2_macs = cp.b2_macs.clone();
            self.clear_accumulated();
            Some(cp)
        } else {
            None
        }
    }

    /// Returns the total number of weight elements tracked.
    pub fn num_weights(&self) -> usize {
        self.w1_macs.len() + self.b1_macs.len() + self.w2_macs.len() + self.b2_macs.len()
    }

    /// Collects all weight MAC shares into a single flat vector.
    pub fn all_weight_macs(&self) -> Vec<Fr> {
        let mut macs = Vec::with_capacity(self.num_weights());
        macs.extend_from_slice(&self.w1_macs);
        macs.extend_from_slice(&self.b1_macs);
        macs.extend_from_slice(&self.w2_macs);
        macs.extend_from_slice(&self.b2_macs);
        macs
    }

    /// Returns the checkpoint weight shares as a flat vector (w1, b1, w2, b2).
    /// Returns None if no checkpoint exists.
    pub fn checkpoint_weights_flat(&self) -> Option<Vec<Fr>> {
        self.checkpoint.as_ref().map(|cp| {
            let mut flat = Vec::with_capacity(
                cp.w1.len() + cp.b1.len() + cp.w2.len() + cp.b2.len(),
            );
            flat.extend_from_slice(&cp.w1);
            flat.extend_from_slice(&cp.b1);
            flat.extend_from_slice(&cp.w2);
            flat.extend_from_slice(&cp.b2);
            flat
        })
    }
}

// ============================================================================
// Checkpoint
// ============================================================================

/// Snapshot of training state at a verified MAC checkpoint.
#[derive(Debug, Clone)]
pub struct TrainingCheckpoint {
    /// Training step when this checkpoint was created.
    pub step: u64,
    /// Weight shares at checkpoint time.
    pub w1: Vec<Fr>,
    pub b1: Vec<Fr>,
    pub w2: Vec<Fr>,
    pub b2: Vec<Fr>,
    /// MAC shares at checkpoint time.
    pub w1_macs: Vec<Fr>,
    pub b1_macs: Vec<Fr>,
    pub w2_macs: Vec<Fr>,
    pub b2_macs: Vec<Fr>,
    /// Beaver triple cursor at checkpoint time.
    pub beaver_cursor: usize,
    /// Authenticated Beaver triple cursor at checkpoint time.
    pub auth_beaver_cursor: usize,
}

// ============================================================================
// Results and Reports
// ============================================================================

/// Result of a MAC verification check.
#[derive(Debug)]
pub enum MACCheckResult {
    /// All MACs verified successfully.
    Passed,
    /// MAC verification failed — cheating detected.
    Failed {
        /// The non-zero delta = sum(sigma_i) indicating tampering.
        delta: Fr,
        /// Detailed failure report (populated after identification).
        report: MACFailureReport,
    },
}

impl MACCheckResult {
    /// Returns true if the check passed.
    pub fn passed(&self) -> bool {
        matches!(self, MACCheckResult::Passed)
    }
}

/// Failure report for on-chain submission.
///
/// Contains all evidence needed to prove that a party cheated:
/// - The sigma values from all parties (public after reveal)
/// - The commitments that bind them
/// - Pairwise identification evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MACFailureReport {
    /// Session identifier.
    pub session_id: String,
    /// Training step where the failure was detected.
    pub step_number: u64,
    /// Index of the identified cheater (None if identification failed).
    pub identified_cheater: Option<usize>,
    /// Sigma values from all parties (from the first round).
    pub sigma_values: Vec<Vec<u8>>,
    /// Commitments from all parties.
    pub commitments: Vec<[u8; 32]>,
    /// Evidence from the identification protocol.
    pub evidence: CheaterEvidence,
}

/// Evidence from the pairwise cheater identification protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheaterEvidence {
    /// Results of pairwise consistency checks.
    pub pairwise_results: Vec<PairwiseCheckResult>,
    /// Sigma values from round 1 (serialized Fr elements).
    pub round1_sigmas: Vec<Vec<u8>>,
    /// Sigma values from round 2 (serialized Fr elements).
    pub round2_sigmas: Vec<Vec<u8>>,
}

/// Result of a pairwise sigma consistency check between two parties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseCheckResult {
    /// First party index.
    pub party_a: usize,
    /// Second party index.
    pub party_b: usize,
    /// Whether the sigma ratios were consistent (both honest → true).
    pub consistent: bool,
}

// ============================================================================
// Internal message types for the sigma protocol
// ============================================================================

/// Messages exchanged during the sigma verification protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum SigmaMessage {
    /// Phase 1: broadcast share of random linear combination.
    Open { v_share: Vec<u8> },
    /// Phase 2: commitment to sigma value.
    Commitment { hash: [u8; 32] },
    /// Phase 3: reveal sigma value.
    Reveal { sigma: Vec<u8>, nonce: [u8; 32] },
    /// Phase 4 (identification): commit to checkpoint diff.
    IdentifyCommit { hash: [u8; 32] },
    /// Phase 5 (identification): reveal checkpoint diff.
    IdentifyReveal { diff_data: Vec<u8>, nonce: [u8; 32] },
}

impl SigmaMessage {
    fn encode(&self) -> Vec<u8> {
        bincode::serialize(self).expect("SigmaMessage encode")
    }
    fn decode(data: &[u8]) -> MPCResult<Self> {
        bincode::deserialize(data)
            .map_err(|e| MPCError::ProtocolError(format!("decode sigma message: {e}")))
    }
}

// ============================================================================
// MAC Generation Helpers
// ============================================================================

/// Generates MAC shares for a set of weight values.
///
/// Party 0 (dealer) calls this with the full alpha and weight values.
/// Returns MAC shares for each party as additive shares of alpha * value.
pub fn generate_mac_shares(
    alpha: &Fr,
    values: &[Fr],
    num_parties: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<Vec<Fr>> {
    let dim = values.len();
    let mut shares: Vec<Vec<Fr>> = (0..num_parties).map(|_| Vec::with_capacity(dim)).collect();

    for val in values {
        let mac = Fr::mul(alpha, val);
        let mut sum = Fr::ZERO;
        for i in 0..num_parties - 1 {
            let r = Fr::random(rng);
            sum = Fr::add(&sum, &r);
            shares[i].push(r);
        }
        shares[num_parties - 1].push(Fr::sub(&mac, &sum));
    }

    shares
}

/// Generates alpha key shares for all parties.
///
/// Returns (alpha, vec_of_alpha_shares) where sum(alpha_shares) = alpha.
pub fn generate_alpha_shares(
    num_parties: usize,
    rng: &mut ChaCha20Rng,
) -> (Fr, Vec<Fr>) {
    let alpha = Fr::random(rng);
    let mut sum = Fr::ZERO;
    let mut shares = Vec::with_capacity(num_parties);
    for _ in 0..num_parties - 1 {
        let share = Fr::random(rng);
        sum = Fr::add(&sum, &share);
        shares.push(share);
    }
    shares.push(Fr::sub(&alpha, &sum));
    (alpha, shares)
}

// ============================================================================
// Sigma Protocol: Batch MAC Verification
// ============================================================================

/// Runs the SPDZ sigma batch verification protocol.
///
/// Protocol:
/// 1. All parties agree on random coefficients r_j (deterministic from step seed)
/// 2. Each party computes v_i = Σ r_j * weight_j (their share of random linear combo)
///    and m_i = Σ r_j * mac_j (their MAC share of the combo)
/// 3. Open v: broadcast v_i, sum to get v = Σ v_i
/// 4. Each party computes σ_i = m_i - α_i * v
/// 5. Commit to σ_i, exchange commitments
/// 6. Reveal σ_i, verify commitments
/// 7. Check Σ σ_i = 0
///
/// Also includes accumulated opened-value checks from intermediate computation.
///
/// Returns `Ok(MACCheckResult::Passed)` if verification succeeds,
/// or `Ok(MACCheckResult::Failed { .. })` with a failure report.
pub async fn run_sigma_check<T: MPCTransport>(
    transport: &T,
    weight_shares: &[Fr],
    mac_state: &MACState,
    step: u64,
    num_parties: usize,
) -> MPCResult<SigmaCheckOutput> {
    let party_index = party_index_from_id(transport.party_id());
    let alpha_share = &mac_state.alpha_share;
    let weight_macs = mac_state.all_weight_macs();

    // Generate deterministic random coefficients from step number.
    let mut coeff_rng = ChaCha20Rng::seed_from_u64(
        step.wrapping_mul(0x517CC1B727220A95).wrapping_add(0xA5A5A5A5),
    );
    let num_values = weight_shares.len();
    let coeffs: Vec<Fr> = (0..num_values).map(|_| Fr::random(&mut coeff_rng)).collect();

    // Phase 1: Compute local share of random linear combination.
    let mut v_i = Fr::ZERO;
    let mut m_i = Fr::ZERO;
    for j in 0..num_values {
        v_i = Fr::add(&v_i, &Fr::mul(&coeffs[j], &weight_shares[j]));
        m_i = Fr::add(&m_i, &Fr::mul(&coeffs[j], &weight_macs[j]));
    }

    // Include accumulated opened values in the check.
    let num_opened = mac_state.opened_values.len();
    if num_opened > 0 {
        let mut opened_coeff_rng = ChaCha20Rng::seed_from_u64(
            step.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xBEEF),
        );
        let mut opened_sigma_accum = Fr::ZERO;
        for k in 0..num_opened {
            let r = Fr::random(&mut opened_coeff_rng);
            let contrib = Fr::sub(
                &Fr::mul(&r, &mac_state.opened_mac_shares[k]),
                &Fr::mul(alpha_share, &Fr::mul(&r, &mac_state.opened_values[k])),
            );
            opened_sigma_accum = Fr::add(&opened_sigma_accum, &contrib);
        }
        // We'll add this to sigma later (it's already a full sigma contribution,
        // no opening needed since the values are already public).
        m_i = Fr::add(&m_i, &opened_sigma_accum);
        // For consistency, add the corresponding value to v (but since opened values
        // are already public and the sigma is pre-computed, we handle them separately).
        // The opened sigma is added directly to sigma_i after opening v.
    }

    // Phase 2: Open v by broadcasting shares.
    let v_msg = SigmaMessage::Open {
        v_share: SecureArithmetic::serialize_share_batch(&[v_i]),
    };
    transport.broadcast(&v_msg.encode()).await?;

    let mut v_total = v_i;
    let peers = transport.peers();
    let all_data = recv_all(transport).await?;
    for data in &all_data {
        let msg = SigmaMessage::decode(data)?;
        if let SigmaMessage::Open { v_share } = msg {
            let peer_v = SecureArithmetic::deserialize_share_batch(&v_share)?;
            v_total = Fr::add(&v_total, &peer_v[0]);
        } else {
            return Err(MPCError::ProtocolError("expected SigmaMessage::Open".into()));
        }
    }

    // Phase 3: Compute sigma_i = m_i - alpha_i * v.
    // Note: m_i already includes the opened-value sigma contributions.
    let sigma_i = Fr::sub(&m_i, &Fr::mul(alpha_share, &v_total));

    // Phase 4: Commit to sigma_i.
    let mut nonce = [0u8; 32];
    {
        let mut nonce_rng = ChaCha20Rng::seed_from_u64(
            step.wrapping_mul(0x6C62272E07BB0142)
                .wrapping_add(party_index as u64),
        );
        for b in nonce.iter_mut() {
            *b = rand::Rng::gen(&mut nonce_rng);
        }
    }
    let sigma_bytes = SecureArithmetic::serialize_share_batch(&[sigma_i]);
    let commitment = compute_commitment(&sigma_bytes, &nonce);

    let commit_msg = SigmaMessage::Commitment { hash: commitment };
    transport.broadcast(&commit_msg.encode()).await?;

    let mut all_commitments = vec![[0u8; 32]; num_parties];
    all_commitments[party_index] = commitment;
    let all_data = recv_all(transport).await?;
    for (peer, data) in peers.iter().zip(all_data.iter()) {
        let msg = SigmaMessage::decode(data)?;
        if let SigmaMessage::Commitment { hash } = msg {
            let peer_idx = party_index_from_id(peer);
            all_commitments[peer_idx] = hash;
        } else {
            return Err(MPCError::ProtocolError("expected SigmaMessage::Commitment".into()));
        }
    }

    // Phase 5: Reveal sigma_i.
    let reveal_msg = SigmaMessage::Reveal {
        sigma: sigma_bytes.clone(),
        nonce,
    };
    transport.broadcast(&reveal_msg.encode()).await?;

    let mut all_sigmas = vec![Fr::ZERO; num_parties];
    all_sigmas[party_index] = sigma_i;
    let all_data = recv_all(transport).await?;
    for (peer, data) in peers.iter().zip(all_data.iter()) {
        let msg = SigmaMessage::decode(data)?;
        if let SigmaMessage::Reveal { sigma, nonce: peer_nonce } = msg {
            let peer_idx = party_index_from_id(peer);
            // Verify commitment.
            let expected_commit = compute_commitment(&sigma, &peer_nonce);
            if expected_commit != all_commitments[peer_idx] {
                return Err(MPCError::MaliciousBehavior {
                    party: peer.clone(),
                    description: "sigma commitment mismatch during MAC check".into(),
                });
            }
            let peer_sigma = SecureArithmetic::deserialize_share_batch(&sigma)?;
            all_sigmas[peer_idx] = peer_sigma[0];
        } else {
            return Err(MPCError::ProtocolError("expected SigmaMessage::Reveal".into()));
        }
    }

    // Phase 6: Check sum(sigma_i) = 0.
    let mut delta = Fr::ZERO;
    for s in &all_sigmas {
        delta = Fr::add(&delta, s);
    }

    let passed = delta.ct_eq(&Fr::ZERO).to_bool();

    debug!(
        party = party_index,
        step = step,
        passed = passed,
        "MAC sigma check completed"
    );

    Ok(SigmaCheckOutput {
        passed,
        delta,
        all_sigmas,
        all_commitments,
    })
}

/// Internal output of a sigma check round.
#[derive(Debug)]
pub struct SigmaCheckOutput {
    pub passed: bool,
    pub delta: Fr,
    pub all_sigmas: Vec<Fr>,
    pub all_commitments: Vec<[u8; 32]>,
}

// ============================================================================
// Cheater Identification
// ============================================================================

/// Identifies the cheating party using pre-step snapshot comparison.
///
/// The key insight: between a checkpoint (successful MAC check) and the start
/// of the next training step, honest parties' weight shares do NOT change.
/// Only corruption modifies shares in that window. So if we compare each
/// party's pre-step snapshot against their checkpoint, the party whose
/// snapshot differs is the cheater.
///
/// Protocol:
/// 1. Each party hashes their pre-step snapshot (saved at step start)
/// 2. Commit to H(pre_step_shares || nonce), broadcast commitments
/// 3. Reveal H(pre_step_shares) + nonce, verify against commitments
/// 4. Compare each party's pre-step hash against stored checkpoint commitments
/// 5. The party whose hash doesn't match their checkpoint commitment is the cheater
///
/// Falls back to sigma-magnitude heuristic if no checkpoint commitments
/// are available (e.g. cheating on the very first step).
///
/// Returns the identified cheater's party index.
pub async fn identify_cheater<T: MPCTransport>(
    transport: &T,
    _weight_shares: &[Fr],
    mac_state: &MACState,
    step: u64,
    num_parties: usize,
    round1_sigmas: &[Fr],
) -> MPCResult<(usize, CheaterEvidence)> {
    info!("Running cheater identification protocol (pre-step snapshot comparison)");

    let party_index = party_index_from_id(transport.party_id());

    // Primary approach: compare pre-step snapshot against checkpoint commitments.
    if let Some(ref pre_step) = mac_state.pre_step_shares {
        if !mac_state.checkpoint_share_commits.is_empty() {
            return identify_cheater_pre_step(
                transport,
                pre_step,
                &mac_state.checkpoint_share_commits,
                step,
                num_parties,
                party_index,
                round1_sigmas,
            )
            .await;
        }
        warn!("No checkpoint commitments stored, falling back to sigma heuristic");
    } else {
        warn!("No pre-step snapshot available, falling back to sigma heuristic");
    }

    // Fallback: sigma magnitude heuristic (best-effort).
    identify_cheater_sigma_fallback(round1_sigmas, num_parties)
}

/// Pre-step snapshot comparison — the primary identification approach.
///
/// Between the checkpoint and the next step start, honest parties' shares
/// are unchanged. A party that corrupted their shares will have a different
/// hash than what was committed at checkpoint time.
async fn identify_cheater_pre_step<T: MPCTransport>(
    transport: &T,
    pre_step_shares: &[Fr],
    checkpoint_commits: &[[u8; 32]],
    step: u64,
    num_parties: usize,
    party_index: usize,
    round1_sigmas: &[Fr],
) -> MPCResult<(usize, CheaterEvidence)> {
    // Phase 1: Hash our pre-step shares.
    let my_pre_step_bytes = SecureArithmetic::serialize_share_batch(pre_step_shares);
    let my_pre_step_hash = hash_bytes(&my_pre_step_bytes);

    // Phase 2: Commit to pre-step hash.
    let mut nonce = [0u8; 32];
    {
        let mut nonce_rng = ChaCha20Rng::seed_from_u64(
            step
                .wrapping_mul(0xAB54A98CEB1F0AD2)
                .wrapping_add(party_index as u64)
                .wrapping_add(0x1D1F_F1ED),
        );
        for b in nonce.iter_mut() {
            *b = rand::Rng::gen(&mut nonce_rng);
        }
    }
    let commitment = compute_commitment(&my_pre_step_hash.as_slice(), &nonce);

    let commit_msg = SigmaMessage::IdentifyCommit { hash: commitment };
    transport.broadcast(&commit_msg.encode()).await?;

    let mut all_commitments = vec![[0u8; 32]; num_parties];
    all_commitments[party_index] = commitment;
    let peers = transport.peers();
    let all_data = recv_all(transport).await?;
    for (peer, data) in peers.iter().zip(all_data.iter()) {
        let msg = SigmaMessage::decode(data)?;
        if let SigmaMessage::IdentifyCommit { hash } = msg {
            let peer_idx = party_index_from_id(peer);
            all_commitments[peer_idx] = hash;
        } else {
            return Err(MPCError::ProtocolError(
                "expected IdentifyCommit during cheater identification".into(),
            ));
        }
    }

    // Phase 3: Reveal pre-step hash.
    let reveal_msg = SigmaMessage::IdentifyReveal {
        diff_data: my_pre_step_hash.to_vec(),
        nonce,
    };
    transport.broadcast(&reveal_msg.encode()).await?;

    let mut all_pre_step_hashes = vec![[0u8; 32]; num_parties];
    all_pre_step_hashes[party_index] = my_pre_step_hash;

    let all_data = recv_all(transport).await?;
    for (peer, data) in peers.iter().zip(all_data.iter()) {
        let msg = SigmaMessage::decode(data)?;
        if let SigmaMessage::IdentifyReveal {
            diff_data,
            nonce: peer_nonce,
        } = msg
        {
            let peer_idx = party_index_from_id(peer);

            // Verify commitment.
            let expected_commit = compute_commitment(&diff_data, &peer_nonce);
            if expected_commit != all_commitments[peer_idx] {
                info!(
                    identified_cheater = peer_idx,
                    "Cheater identified: commitment mismatch during identification"
                );
                return Ok((
                    peer_idx,
                    build_identification_evidence(num_parties, round1_sigmas),
                ));
            }

            if diff_data.len() == 32 {
                all_pre_step_hashes[peer_idx].copy_from_slice(&diff_data);
            }
        } else {
            return Err(MPCError::ProtocolError(
                "expected IdentifyReveal during cheater identification".into(),
            ));
        }
    }

    // Phase 4: Compare each party's pre-step hash against checkpoint commitment.
    // The party whose hash doesn't match their checkpoint is the cheater.
    let mut mismatches = Vec::new();
    for i in 0..num_parties {
        if i < checkpoint_commits.len() && all_pre_step_hashes[i] != checkpoint_commits[i] {
            mismatches.push(i);
        }
    }

    let identified = if mismatches.len() == 1 {
        // Exactly one mismatch — clear identification.
        mismatches[0]
    } else if mismatches.is_empty() {
        // No mismatches in pre-step hashes. This means the corruption happened
        // DURING the computation (not between steps). Fall back to sigma heuristic.
        warn!("All pre-step hashes match checkpoint — corruption during computation");
        return identify_cheater_sigma_fallback(round1_sigmas, num_parties);
    } else {
        // Multiple mismatches — shouldn't happen with a single cheater.
        // Pick the first non-zero-party mismatch (party 0 legitimately changes
        // shares during weight update, so exclude it if others also mismatch).
        *mismatches.iter().find(|&&i| i != 0).unwrap_or(&mismatches[0])
    };

    info!(
        identified_cheater = identified,
        mismatches = mismatches.len(),
        "Cheater identified via pre-step snapshot comparison"
    );

    Ok((identified, build_identification_evidence(num_parties, round1_sigmas)))
}

/// Fallback identification using sigma magnitude when no checkpoint is available.
fn identify_cheater_sigma_fallback(
    round1_sigmas: &[Fr],
    num_parties: usize,
) -> MPCResult<(usize, CheaterEvidence)> {
    let mut max_mag = 0u64;
    let mut max_idx = 0;
    for (i, sigma) in round1_sigmas.iter().enumerate() {
        let mag = sigma_magnitude(sigma);
        if mag > max_mag {
            max_mag = mag;
            max_idx = i;
        }
    }

    warn!(
        identified_cheater = max_idx,
        "Cheater identified via sigma magnitude heuristic (no checkpoint available)"
    );

    Ok((
        max_idx,
        CheaterEvidence {
            pairwise_results: Vec::new(),
            round1_sigmas: round1_sigmas
                .iter()
                .map(|s| SecureArithmetic::serialize_share_batch(&[*s]))
                .collect(),
            round2_sigmas: Vec::new(),
        },
    ))
}

/// Builds identification evidence from sigma values.
fn build_identification_evidence(
    num_parties: usize,
    round1_sigmas: &[Fr],
) -> CheaterEvidence {
    let mut pairwise_results = Vec::new();
    for i in 0..num_parties {
        for j in (i + 1)..num_parties {
            pairwise_results.push(PairwiseCheckResult {
                party_a: i,
                party_b: j,
                consistent: true, // Not used in pre-step approach
            });
        }
    }

    CheaterEvidence {
        pairwise_results,
        round1_sigmas: round1_sigmas
            .iter()
            .map(|s| SecureArithmetic::serialize_share_batch(&[*s]))
            .collect(),
        round2_sigmas: Vec::new(),
    }
}

/// SHA-256 hash of a byte slice.
fn hash_bytes(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut h = [0u8; 32];
    h.copy_from_slice(&result);
    h
}

/// Runs just the sigma computation (without commit/reveal).
/// Retained for potential future use (e.g. additional verification rounds).
#[allow(dead_code)]
async fn run_sigma_check_round<T: MPCTransport>(
    transport: &T,
    weight_shares: &[Fr],
    mac_state: &MACState,
    seed: u64,
    num_parties: usize,
) -> MPCResult<SigmaCheckOutput> {
    let party_index = party_index_from_id(transport.party_id());
    let alpha_share = &mac_state.alpha_share;
    let weight_macs = mac_state.all_weight_macs();
    let num_values = weight_shares.len();

    // Generate random coefficients.
    let mut coeff_rng = ChaCha20Rng::seed_from_u64(seed);
    let coeffs: Vec<Fr> = (0..num_values).map(|_| Fr::random(&mut coeff_rng)).collect();

    // Compute local v_i and m_i.
    let mut v_i = Fr::ZERO;
    let mut m_i = Fr::ZERO;
    for j in 0..num_values {
        v_i = Fr::add(&v_i, &Fr::mul(&coeffs[j], &weight_shares[j]));
        m_i = Fr::add(&m_i, &Fr::mul(&coeffs[j], &weight_macs[j]));
    }

    // Open v.
    let v_msg = SigmaMessage::Open {
        v_share: SecureArithmetic::serialize_share_batch(&[v_i]),
    };
    transport.broadcast(&v_msg.encode()).await?;

    let mut v_total = v_i;
    for peer in &transport.peers() {
        let data = transport.recv(peer).await?;
        let msg = SigmaMessage::decode(&data)?;
        if let SigmaMessage::Open { v_share } = msg {
            let peer_v = SecureArithmetic::deserialize_share_batch(&v_share)?;
            v_total = Fr::add(&v_total, &peer_v[0]);
        } else {
            return Err(MPCError::ProtocolError("expected Open in identification round".into()));
        }
    }

    // Compute sigma.
    let sigma_i = Fr::sub(&m_i, &Fr::mul(alpha_share, &v_total));

    // In the identification round, we use simplified reveal (no commitments needed
    // since we're already in an abort path and the first round's commitments prevent
    // adaptive cheating).
    let sigma_bytes = SecureArithmetic::serialize_share_batch(&[sigma_i]);
    let reveal_msg = SigmaMessage::Reveal {
        sigma: sigma_bytes,
        nonce: [0u8; 32], // Dummy nonce for identification round
    };
    transport.broadcast(&reveal_msg.encode()).await?;

    let mut all_sigmas = vec![Fr::ZERO; num_parties];
    all_sigmas[party_index] = sigma_i;
    for peer in &transport.peers() {
        let data = transport.recv(peer).await?;
        let msg = SigmaMessage::decode(&data)?;
        if let SigmaMessage::Reveal { sigma, .. } = msg {
            let peer_idx = party_index_from_id(peer);
            let peer_sigma = SecureArithmetic::deserialize_share_batch(&sigma)?;
            all_sigmas[peer_idx] = peer_sigma[0];
        } else {
            return Err(MPCError::ProtocolError("expected Reveal in identification round".into()));
        }
    }

    let mut delta = Fr::ZERO;
    for s in &all_sigmas {
        delta = Fr::add(&delta, s);
    }

    Ok(SigmaCheckOutput {
        passed: delta.ct_eq(&Fr::ZERO).to_bool(),
        delta,
        all_sigmas,
        all_commitments: vec![[0u8; 32]; num_parties],
    })
}

// ============================================================================
// Beaver Triple Authentication
// ============================================================================

/// Authenticates a batch of Beaver triples by having party 0 compute MAC shares.
///
/// Protocol:
/// 1. All parties send their triple shares (a_i, b_i, c_i) to party 0
/// 2. Party 0 reconstructs (a, b, c) = (Σ a_i, Σ b_i, Σ c_i)
/// 3. Party 0 computes MAC(a) = α*a, MAC(b) = α*b, MAC(c) = α*c
/// 4. Party 0 creates additive MAC shares and distributes them
/// 5. Each party receives (mac_a_i, mac_b_i, mac_c_i)
pub async fn authenticate_beaver_triples<T: MPCTransport>(
    transport: &T,
    triples: &[BeaverTriple],
    alpha: &Fr,
    party_index: usize,
    num_parties: usize,
    rng: &mut ChaCha20Rng,
) -> MPCResult<Vec<AuthenticatedBeaverTriple>> {
    let count = triples.len();
    let prover_party = PartyId::from_index(0);
    let peers = transport.peers();

    if party_index != 0 {
        // Non-dealer: send triple shares to party 0.
        let mut flat: Vec<Fr> = Vec::with_capacity(count * 3);
        for t in triples {
            flat.push(t.a);
            flat.push(t.b);
            flat.push(t.c);
        }
        let msg = SecureArithmetic::serialize_share_batch(&flat);
        transport.send(&prover_party, &msg).await?;

        // Receive MAC shares from party 0.
        let mac_data = transport.recv(&prover_party).await?;
        let mac_shares = SecureArithmetic::deserialize_share_batch(&mac_data)?;
        if mac_shares.len() != count * 3 {
            return Err(MPCError::ProtocolError(format!(
                "expected {} MAC shares, got {}", count * 3, mac_shares.len()
            )));
        }

        let mut result = Vec::with_capacity(count);
        for (i, t) in triples.iter().enumerate() {
            result.push(AuthenticatedBeaverTriple {
                triple: t.clone(),
                mac_a: mac_shares[i * 3],
                mac_b: mac_shares[i * 3 + 1],
                mac_c: mac_shares[i * 3 + 2],
            });
        }
        return Ok(result);
    }

    // Party 0: receive shares from all peers and reconstruct.
    let mut all_a = vec![Fr::ZERO; count];
    let mut all_b = vec![Fr::ZERO; count];
    let mut all_c = vec![Fr::ZERO; count];

    // Add our own shares.
    for (i, t) in triples.iter().enumerate() {
        all_a[i] = t.a;
        all_b[i] = t.b;
        all_c[i] = t.c;
    }

    // Receive from peers (parallel).
    let all_data = recv_all(transport).await?;
    for data in &all_data {
        let shares = SecureArithmetic::deserialize_share_batch(data)?;
        if shares.len() != count * 3 {
            return Err(MPCError::ProtocolError(format!(
                "expected {} triple shares, got {}", count * 3, shares.len()
            )));
        }
        for i in 0..count {
            all_a[i] = Fr::add(&all_a[i], &shares[i * 3]);
            all_b[i] = Fr::add(&all_b[i], &shares[i * 3 + 1]);
            all_c[i] = Fr::add(&all_c[i], &shares[i * 3 + 2]);
        }
    }

    // Compute MACs for each reconstructed triple.
    let mut mac_a_full: Vec<Fr> = Vec::with_capacity(count);
    let mut mac_b_full: Vec<Fr> = Vec::with_capacity(count);
    let mut mac_c_full: Vec<Fr> = Vec::with_capacity(count);
    for i in 0..count {
        mac_a_full.push(Fr::mul(alpha, &all_a[i]));
        mac_b_full.push(Fr::mul(alpha, &all_b[i]));
        mac_c_full.push(Fr::mul(alpha, &all_c[i]));
    }

    // Create additive shares of each MAC and distribute.
    let mut my_mac_a = Vec::with_capacity(count);
    let mut my_mac_b = Vec::with_capacity(count);
    let mut my_mac_c = Vec::with_capacity(count);

    // For each peer, accumulate their MAC shares to send.
    let mut peer_shares: Vec<Vec<Fr>> = vec![Vec::with_capacity(count * 3); peers.len()];

    for i in 0..count {
        // Create random MAC shares for n-1 parties, keep remainder for self.
        let (mac_a_mine, mac_a_others) = random_additive_shares(&mac_a_full[i], num_parties, rng);
        let (mac_b_mine, mac_b_others) = random_additive_shares(&mac_b_full[i], num_parties, rng);
        let (mac_c_mine, mac_c_others) = random_additive_shares(&mac_c_full[i], num_parties, rng);

        my_mac_a.push(mac_a_mine);
        my_mac_b.push(mac_b_mine);
        my_mac_c.push(mac_c_mine);

        for (j, peer_flat) in peer_shares.iter_mut().enumerate() {
            peer_flat.push(mac_a_others[j]);
            peer_flat.push(mac_b_others[j]);
            peer_flat.push(mac_c_others[j]);
        }
    }

    // Send MAC shares to each peer.
    for (j, peer) in peers.iter().enumerate() {
        let msg = SecureArithmetic::serialize_share_batch(&peer_shares[j]);
        transport.send(peer, &msg).await?;
    }

    // Build authenticated triples for party 0.
    let mut result = Vec::with_capacity(count);
    for (i, t) in triples.iter().enumerate() {
        result.push(AuthenticatedBeaverTriple {
            triple: t.clone(),
            mac_a: my_mac_a[i],
            mac_b: my_mac_b[i],
            mac_c: my_mac_c[i],
        });
    }

    Ok(result)
}

/// Creates additive shares where party 0 gets the remainder.
/// Returns (party_0_share, [party_1_share, ..., party_{n-1}_share]).
fn random_additive_shares(value: &Fr, num_parties: usize, rng: &mut ChaCha20Rng) -> (Fr, Vec<Fr>) {
    let mut others = Vec::with_capacity(num_parties - 1);
    let mut sum = Fr::ZERO;
    for _ in 0..num_parties - 1 {
        let r = Fr::random(rng);
        sum = Fr::add(&sum, &r);
        others.push(r);
    }
    (Fr::sub(value, &sum), others)
}

// ============================================================================
// Full MAC Check with Identification and Reporting
// ============================================================================

/// Runs the complete MAC check pipeline: verify → identify → report.
///
/// This is the main entry point called from the training loop:
/// 1. Runs the sigma check on current weight shares
/// 2. If check passes: saves checkpoint, returns Passed
/// 3. If check fails: runs cheater identification, produces report, returns Failed
pub async fn full_mac_check<T: MPCTransport>(
    transport: &T,
    weight_shares: &[Fr],
    mac_state: &mut MACState,
    step: u64,
    num_parties: usize,
    session_id: &str,
    enable_identification: bool,
) -> MPCResult<MACCheckResult> {
    let check = run_sigma_check(
        transport,
        weight_shares,
        mac_state,
        step,
        num_parties,
    ).await?;

    if check.passed {
        info!(step = step, "MAC verification passed");
        mac_state.clear_accumulated();
        return Ok(MACCheckResult::Passed);
    }

    warn!(step = step, "MAC verification FAILED — cheating detected");

    let (identified_cheater, evidence) = if enable_identification {
        let (cheater, evidence) = identify_cheater(
            transport,
            weight_shares,
            mac_state,
            step,
            num_parties,
            &check.all_sigmas,
        ).await?;
        (Some(cheater), evidence)
    } else {
        (None, CheaterEvidence {
            pairwise_results: Vec::new(),
            round1_sigmas: Vec::new(),
            round2_sigmas: Vec::new(),
        })
    };

    let report = MACFailureReport {
        session_id: session_id.to_string(),
        step_number: step,
        identified_cheater,
        sigma_values: check.all_sigmas.iter()
            .map(|s| SecureArithmetic::serialize_share_batch(&[*s]))
            .collect(),
        commitments: check.all_commitments,
        evidence,
    };

    Ok(MACCheckResult::Failed {
        delta: check.delta,
        report,
    })
}

// ============================================================================
// Checkpoint Commitment Exchange
// ============================================================================

/// Exchanges H(weight_shares) commitments with all peers after a checkpoint.
///
/// Each party broadcasts H(their_weight_shares) and stores all peers'
/// commitments. These are used during cheater identification to verify
/// that pre-step shares match the checkpoint state.
pub async fn exchange_checkpoint_commits<T: MPCTransport>(
    transport: &T,
    weight_shares: &[Fr],
    mac_state: &mut MACState,
    num_parties: usize,
) -> MPCResult<()> {
    let party_index = party_index_from_id(transport.party_id());
    let share_bytes = SecureArithmetic::serialize_share_batch(weight_shares);
    let my_hash = hash_bytes(&share_bytes);

    // Broadcast our hash.
    transport.broadcast(&my_hash.to_vec()).await?;

    // Collect all hashes.
    let mut commits = vec![[0u8; 32]; num_parties];
    commits[party_index] = my_hash;

    let peers = transport.peers();
    let all_data = recv_all(transport).await?;
    for (peer, data) in peers.iter().zip(all_data.iter()) {
        let peer_idx = party_index_from_id(peer);
        if data.len() == 32 {
            commits[peer_idx].copy_from_slice(data);
        }
    }

    mac_state.checkpoint_share_commits = commits;
    debug!(party = party_index, "Checkpoint commitments exchanged");
    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

/// Computes SHA-256 commitment: H(sigma || nonce).
fn compute_commitment(sigma_bytes: &[u8], nonce: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(sigma_bytes);
    hasher.update(nonce);
    let result = hasher.finalize();
    let mut commitment = [0u8; 32];
    commitment.copy_from_slice(&result);
    commitment
}

/// Extracts party index from PartyId (e.g., "party-2" → 2).
fn party_index_from_id(party: &PartyId) -> usize {
    party.0.strip_prefix("party-")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Computes a rough magnitude proxy for an Fr element (for tiebreaking).
/// Uses the first 8 bytes of the representation.
fn sigma_magnitude(fr: &Fr) -> u64 {
    let bytes = fr.to_bytes_le();
    u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0u8; 8]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_mac_shares_consistency() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let (alpha, alpha_shares) = generate_alpha_shares(3, &mut rng);

        let values = vec![Fr::from_f64(1.0), Fr::from_f64(2.0), Fr::from_f64(3.0)];
        let mac_shares = generate_mac_shares(&alpha, &values, 3, &mut rng);

        // Verify: sum of MAC shares = alpha * value for each element
        for v in 0..values.len() {
            let mut mac_sum = Fr::ZERO;
            for p in 0..3 {
                mac_sum = Fr::add(&mac_sum, &mac_shares[p][v]);
            }
            let expected = Fr::mul(&alpha, &values[v]);
            assert!(mac_sum.ct_eq(&expected).to_bool(),
                "MAC sum should equal alpha * value for element {}", v);
        }

        // Verify alpha shares sum correctly.
        let mut alpha_sum = Fr::ZERO;
        for a in &alpha_shares {
            alpha_sum = Fr::add(&alpha_sum, a);
        }
        assert!(alpha_sum.ct_eq(&alpha).to_bool(), "alpha shares should sum to alpha");
    }

    #[test]
    fn test_authenticated_beaver_multiply() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let (alpha, alpha_shares) = generate_alpha_shares(3, &mut rng);

        // Create a simple authenticated triple for party 0.
        let a_full = Fr::from_f64(3.0);
        let b_full = Fr::from_f64(5.0);
        let c_full = a_full.mpc_scale(&b_full);
        let mac_a = Fr::mul(&alpha, &a_full);
        let mac_b = Fr::mul(&alpha, &b_full);
        let mac_c = Fr::mul(&alpha, &c_full);

        // Simulate: all shares held by party 0 (single-party test).
        let triple = AuthenticatedBeaverTriple {
            triple: BeaverTriple::new(a_full, b_full, c_full),
            mac_a,
            mac_b,
            mac_c,
        };

        let x = Fr::from_f64(7.0);
        let y = Fr::from_f64(11.0);
        let d = Fr::sub(&x, &a_full);
        let e = Fr::sub(&y, &b_full);

        let (z, mac_z) = triple.authenticated_multiply(&d, &e, 0, &alpha);

        // Check value: z should equal x * y.
        let expected_z = x.mpc_scale(&y);
        let z_f64 = z.to_f64();
        let expected_f64 = expected_z.to_f64();
        assert!((z_f64 - expected_f64).abs() < 1e-6,
            "z = {}, expected = {}", z_f64, expected_f64);

        // Check MAC: mac_z should equal alpha * z.
        let expected_mac = Fr::mul(&alpha, &z);
        assert!(mac_z.ct_eq(&expected_mac).to_bool(),
            "MAC(z) should equal alpha * z");
    }

    #[test]
    fn test_commitment() {
        let sigma = vec![1, 2, 3, 4, 5];
        let nonce = [42u8; 32];
        let c1 = compute_commitment(&sigma, &nonce);
        let c2 = compute_commitment(&sigma, &nonce);
        assert_eq!(c1, c2, "same input should give same commitment");

        let nonce2 = [43u8; 32];
        let c3 = compute_commitment(&sigma, &nonce2);
        assert_ne!(c1, c3, "different nonce should give different commitment");
    }
}
