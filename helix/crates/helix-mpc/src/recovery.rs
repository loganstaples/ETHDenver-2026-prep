//! Checkpoint rollback and share redistribution after cheater removal.
//!
//! When a party is identified as a cheater (via MAC failure and pairwise
//! identification), the system needs to:
//!
//! 1. Roll back to the last verified checkpoint.
//! 2. Remove the cheater from the participant set.
//! 3. Redistribute weight shares among the remaining honest parties.
//! 4. Restore the trainer state from the checkpoint.
//!
//! # Share Redistribution
//!
//! For additive secret sharing, removing a party means its share is lost.
//! The model owner (party 0) must reconstruct the full weights from the
//! checkpoint and re-split them among the remaining parties.
//!
//! # Disconnection Handling
//!
//! The [`DisconnectionHandler`] provides a grace period before declaring a
//! party disconnected. If enough parties remain (honest majority), training
//! can continue.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::mac_verification::{TrainingCheckpoint, MACState, generate_alpha_shares, generate_mac_shares};
use crate::types::PartyId;

// ============================================================================
// Recovery Coordinator
// ============================================================================

/// Coordinates recovery after a cheater is identified.
///
/// Holds the checkpoint state and orchestrates share redistribution
/// among the remaining honest parties.
pub struct RecoveryCoordinator {
    /// The checkpoint to recover from.
    checkpoint: TrainingCheckpoint,
    /// The original number of parties.
    original_num_parties: usize,
    /// Parties that have been removed (identified cheaters or disconnected).
    removed_parties: HashSet<usize>,
    /// Session identifier for logging.
    session_id: String,
    /// RNG for share generation during redistribution.
    rng: ChaCha20Rng,
}

impl RecoveryCoordinator {
    /// Creates a recovery coordinator from a checkpoint.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - The last verified checkpoint to recover from.
    /// * `original_num_parties` - Total number of parties before removal.
    /// * `session_id` - Session identifier for logging.
    /// * `seed` - Random seed for share generation.
    pub fn from_checkpoint(
        checkpoint: TrainingCheckpoint,
        original_num_parties: usize,
        session_id: &str,
        seed: u64,
    ) -> Self {
        info!(
            step = checkpoint.step,
            session_id = session_id,
            "Recovery coordinator created from checkpoint at step {}",
            checkpoint.step
        );

        Self {
            checkpoint,
            original_num_parties,
            removed_parties: HashSet::new(),
            session_id: session_id.to_string(),
            rng: ChaCha20Rng::seed_from_u64(seed),
        }
    }

    /// Marks a party as removed (cheater or disconnected).
    pub fn remove_party(&mut self, party_index: usize) -> MPCResult<()> {
        if party_index >= self.original_num_parties {
            return Err(MPCError::IndexOutOfRange {
                index: party_index,
                num_parties: self.original_num_parties,
            });
        }

        warn!(
            party = party_index,
            session = %self.session_id,
            "Party {} removed from session",
            party_index
        );

        self.removed_parties.insert(party_index);
        Ok(())
    }

    /// Returns the number of remaining (non-removed) parties.
    pub fn remaining_party_count(&self) -> usize {
        self.original_num_parties - self.removed_parties.len()
    }

    /// Returns the indices of remaining (non-removed) parties.
    pub fn remaining_parties(&self) -> Vec<usize> {
        (0..self.original_num_parties)
            .filter(|i| !self.removed_parties.contains(i))
            .collect()
    }

    /// Returns the checkpoint this coordinator is recovering from.
    pub fn checkpoint(&self) -> &TrainingCheckpoint {
        &self.checkpoint
    }

    /// Returns the step number of the checkpoint.
    pub fn checkpoint_step(&self) -> u64 {
        self.checkpoint.step
    }

    /// Redistributes weight shares among the remaining honest parties.
    ///
    /// This is called by party 0 (the model owner) who reconstructs the
    /// full weights from the checkpoint and creates new additive shares
    /// for the remaining parties.
    ///
    /// # Arguments
    ///
    /// * `full_weights` - The full (reconstructed) weights at the checkpoint.
    ///   Party 0 reconstructs these by summing all parties' checkpoint shares.
    ///
    /// # Returns
    ///
    /// A vector of (w1, b1, w2, b2) share tuples, one per remaining party
    /// (in the order returned by `remaining_parties()`).
    pub fn redistribute_shares(
        &mut self,
        full_w1: &[Fr],
        full_b1: &[Fr],
        full_w2: &[Fr],
        full_b2: &[Fr],
    ) -> MPCResult<Vec<RedistributedShares>> {
        let remaining = self.remaining_parties();
        let n = remaining.len();

        if n < 2 {
            return Err(MPCError::InsufficientParties {
                required: 2,
                available: n,
            });
        }

        info!(
            remaining_parties = n,
            session = %self.session_id,
            "Redistributing shares among {} parties",
            n
        );

        let w1_shares = additive_share_vec(full_w1, n, &mut self.rng);
        let b1_shares = additive_share_vec(full_b1, n, &mut self.rng);
        let w2_shares = additive_share_vec(full_w2, n, &mut self.rng);
        let b2_shares = additive_share_vec(full_b2, n, &mut self.rng);

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            result.push(RedistributedShares {
                party_index: remaining[i],
                w1: w1_shares[i].clone(),
                b1: b1_shares[i].clone(),
                w2: w2_shares[i].clone(),
                b2: b2_shares[i].clone(),
            });
        }

        Ok(result)
    }

    /// Redistributes weight shares with fresh MAC shares.
    ///
    /// Like `redistribute_shares()`, but also generates new MAC key and shares
    /// for the remaining parties.
    ///
    /// # Returns
    ///
    /// A tuple of (redistributed shares, alpha, alpha shares per remaining party).
    pub fn redistribute_shares_with_macs(
        &mut self,
        full_w1: &[Fr],
        full_b1: &[Fr],
        full_w2: &[Fr],
        full_b2: &[Fr],
    ) -> MPCResult<(Vec<RedistributedSharesWithMACs>, Fr, Vec<Fr>)> {
        let remaining = self.remaining_parties();
        let n = remaining.len();

        if n < 2 {
            return Err(MPCError::InsufficientParties {
                required: 2,
                available: n,
            });
        }

        // Create new weight shares.
        let w1_shares = additive_share_vec(full_w1, n, &mut self.rng);
        let b1_shares = additive_share_vec(full_b1, n, &mut self.rng);
        let w2_shares = additive_share_vec(full_w2, n, &mut self.rng);
        let b2_shares = additive_share_vec(full_b2, n, &mut self.rng);

        // Generate fresh MAC key and shares.
        let (alpha, alpha_shares) = generate_alpha_shares(n, &mut self.rng);
        let w1_mac_shares = generate_mac_shares(&alpha, full_w1, n, &mut self.rng);
        let b1_mac_shares = generate_mac_shares(&alpha, full_b1, n, &mut self.rng);
        let w2_mac_shares = generate_mac_shares(&alpha, full_w2, n, &mut self.rng);
        let b2_mac_shares = generate_mac_shares(&alpha, full_b2, n, &mut self.rng);

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            result.push(RedistributedSharesWithMACs {
                party_index: remaining[i],
                w1: w1_shares[i].clone(),
                b1: b1_shares[i].clone(),
                w2: w2_shares[i].clone(),
                b2: b2_shares[i].clone(),
                w1_macs: w1_mac_shares[i].clone(),
                b1_macs: b1_mac_shares[i].clone(),
                w2_macs: w2_mac_shares[i].clone(),
                b2_macs: b2_mac_shares[i].clone(),
                alpha_share: alpha_shares[i],
            });
        }

        info!(
            remaining_parties = n,
            "Redistributed shares with fresh MACs"
        );

        Ok((result, alpha, alpha_shares))
    }

    /// Restores a trainer's state from the checkpoint.
    ///
    /// Returns the weight shares and MAC state from the checkpoint so the
    /// caller can apply them to a trainer. This is used after redistribution
    /// when the trainer needs to be reset to the checkpoint state with new shares.
    pub fn restore_state(&self) -> RestoredState {
        RestoredState {
            step: self.checkpoint.step,
            w1: self.checkpoint.w1.clone(),
            b1: self.checkpoint.b1.clone(),
            w2: self.checkpoint.w2.clone(),
            b2: self.checkpoint.b2.clone(),
            w1_macs: self.checkpoint.w1_macs.clone(),
            b1_macs: self.checkpoint.b1_macs.clone(),
            w2_macs: self.checkpoint.w2_macs.clone(),
            b2_macs: self.checkpoint.b2_macs.clone(),
            beaver_cursor: self.checkpoint.beaver_cursor,
            auth_beaver_cursor: self.checkpoint.auth_beaver_cursor,
        }
    }
}

/// Weight shares for a single party after redistribution.
#[derive(Debug, Clone)]
pub struct RedistributedShares {
    /// Which party these shares belong to.
    pub party_index: usize,
    /// Layer 1 weight shares.
    pub w1: Vec<Fr>,
    /// Layer 1 bias shares.
    pub b1: Vec<Fr>,
    /// Layer 2 weight shares.
    pub w2: Vec<Fr>,
    /// Layer 2 bias shares.
    pub b2: Vec<Fr>,
}

/// Weight shares with MAC shares for a single party after redistribution.
#[derive(Debug, Clone)]
pub struct RedistributedSharesWithMACs {
    /// Which party these shares belong to.
    pub party_index: usize,
    /// Layer 1 weight shares.
    pub w1: Vec<Fr>,
    /// Layer 1 bias shares.
    pub b1: Vec<Fr>,
    /// Layer 2 weight shares.
    pub w2: Vec<Fr>,
    /// Layer 2 bias shares.
    pub b2: Vec<Fr>,
    /// Layer 1 weight MAC shares.
    pub w1_macs: Vec<Fr>,
    /// Layer 1 bias MAC shares.
    pub b1_macs: Vec<Fr>,
    /// Layer 2 weight MAC shares.
    pub w2_macs: Vec<Fr>,
    /// Layer 2 bias MAC shares.
    pub b2_macs: Vec<Fr>,
    /// This party's share of the MAC key alpha.
    pub alpha_share: Fr,
}

/// State restored from a checkpoint.
#[derive(Debug, Clone)]
pub struct RestoredState {
    /// Training step at the checkpoint.
    pub step: u64,
    /// Weight shares.
    pub w1: Vec<Fr>,
    pub b1: Vec<Fr>,
    pub w2: Vec<Fr>,
    pub b2: Vec<Fr>,
    /// MAC shares.
    pub w1_macs: Vec<Fr>,
    pub b1_macs: Vec<Fr>,
    pub w2_macs: Vec<Fr>,
    pub b2_macs: Vec<Fr>,
    /// Beaver triple cursor position.
    pub beaver_cursor: usize,
    /// Authenticated Beaver triple cursor position.
    pub auth_beaver_cursor: usize,
}

// ============================================================================
// Disconnection Handler
// ============================================================================

/// Handles party disconnections with grace period logic.
///
/// When a party stops responding, the handler:
/// 1. Waits for a grace period before declaring disconnect.
/// 2. Checks if enough parties remain for honest majority.
/// 3. Initiates recovery if the disconnected party should be removed.
#[derive(Debug)]
pub struct DisconnectionHandler {
    /// Grace period before declaring a party disconnected.
    grace_period: Duration,
    /// When each party was last seen.
    last_seen: Vec<Option<Instant>>,
    /// Number of parties.
    num_parties: usize,
    /// Minimum number of honest parties required to continue.
    /// For additive sharing this must equal num_parties.
    /// For threshold schemes this is the threshold.
    min_honest: usize,
    /// Parties that have been declared disconnected.
    disconnected: HashSet<usize>,
}

impl DisconnectionHandler {
    /// Creates a new disconnection handler.
    ///
    /// # Arguments
    ///
    /// * `num_parties` - Total number of parties.
    /// * `min_honest` - Minimum honest parties needed to continue.
    /// * `grace_period` - How long to wait before declaring disconnect.
    pub fn new(
        num_parties: usize,
        min_honest: usize,
        grace_period: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            grace_period,
            last_seen: vec![Some(now); num_parties],
            num_parties,
            min_honest,
            disconnected: HashSet::new(),
        }
    }

    /// Records that a party was seen (heartbeat or message received).
    pub fn record_activity(&mut self, party_index: usize) {
        if party_index < self.num_parties {
            self.last_seen[party_index] = Some(Instant::now());
        }
    }

    /// Checks for disconnected parties and returns newly disconnected indices.
    pub fn check_disconnections(&mut self) -> Vec<usize> {
        let now = Instant::now();
        let mut newly_disconnected = Vec::new();

        for i in 0..self.num_parties {
            if self.disconnected.contains(&i) {
                continue;
            }

            if let Some(last) = self.last_seen[i] {
                if now.duration_since(last) > self.grace_period {
                    self.disconnected.insert(i);
                    newly_disconnected.push(i);
                    warn!(
                        party = i,
                        elapsed_ms = now.duration_since(last).as_millis() as u64,
                        "Party {} disconnected (grace period exceeded)",
                        i
                    );
                }
            }
        }

        newly_disconnected
    }

    /// Returns whether enough parties remain for the protocol to continue.
    pub fn has_honest_majority(&self) -> bool {
        let active = self.num_parties - self.disconnected.len();
        active >= self.min_honest
    }

    /// Returns the number of active (non-disconnected) parties.
    pub fn active_count(&self) -> usize {
        self.num_parties - self.disconnected.len()
    }

    /// Returns whether a specific party is disconnected.
    pub fn is_disconnected(&self, party_index: usize) -> bool {
        self.disconnected.contains(&party_index)
    }

    /// Returns all disconnected party indices.
    pub fn disconnected_parties(&self) -> Vec<usize> {
        self.disconnected.iter().copied().collect()
    }

    /// Marks a party as disconnected explicitly (e.g., after cheater identification).
    pub fn mark_disconnected(&mut self, party_index: usize) {
        self.disconnected.insert(party_index);
    }
}

// ============================================================================
// Helpers
// ============================================================================

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
        shares[n - 1].push(Fr::sub(elem, &sum));
    }

    shares
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_checkpoint(step: u64, dim: usize) -> TrainingCheckpoint {
        TrainingCheckpoint {
            step,
            w1: vec![Fr::from_f64(1.0); dim],
            b1: vec![Fr::from_f64(0.1); dim / 2],
            w2: vec![Fr::from_f64(0.5); dim / 2],
            b2: vec![Fr::from_f64(0.01); 1],
            w1_macs: vec![Fr::ZERO; dim],
            b1_macs: vec![Fr::ZERO; dim / 2],
            w2_macs: vec![Fr::ZERO; dim / 2],
            b2_macs: vec![Fr::ZERO; 1],
            beaver_cursor: 10,
            auth_beaver_cursor: 5,
        }
    }

    #[test]
    fn test_recovery_coordinator_creation() {
        let cp = make_checkpoint(10, 4);
        let coord = RecoveryCoordinator::from_checkpoint(cp, 3, "test-session", 42);

        assert_eq!(coord.remaining_party_count(), 3);
        assert_eq!(coord.checkpoint_step(), 10);
        assert_eq!(coord.remaining_parties(), vec![0, 1, 2]);
    }

    #[test]
    fn test_remove_party() {
        let cp = make_checkpoint(10, 4);
        let mut coord = RecoveryCoordinator::from_checkpoint(cp, 3, "test-session", 42);

        coord.remove_party(2).unwrap();
        assert_eq!(coord.remaining_party_count(), 2);
        assert_eq!(coord.remaining_parties(), vec![0, 1]);
    }

    #[test]
    fn test_remove_party_invalid_index() {
        let cp = make_checkpoint(10, 4);
        let mut coord = RecoveryCoordinator::from_checkpoint(cp, 3, "test-session", 42);

        let result = coord.remove_party(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_redistribute_shares() {
        let cp = make_checkpoint(10, 4);
        let mut coord = RecoveryCoordinator::from_checkpoint(cp, 3, "test-session", 42);

        coord.remove_party(2).unwrap();

        let full_w1 = vec![Fr::from_f64(1.0), Fr::from_f64(2.0), Fr::from_f64(3.0), Fr::from_f64(4.0)];
        let full_b1 = vec![Fr::from_f64(0.1), Fr::from_f64(0.2)];
        let full_w2 = vec![Fr::from_f64(0.5), Fr::from_f64(0.6)];
        let full_b2 = vec![Fr::from_f64(0.01)];

        let shares = coord.redistribute_shares(&full_w1, &full_b1, &full_w2, &full_b2).unwrap();

        assert_eq!(shares.len(), 2);
        assert_eq!(shares[0].party_index, 0);
        assert_eq!(shares[1].party_index, 1);

        // Verify shares sum to original weights.
        for idx in 0..4 {
            let sum = Fr::add(&shares[0].w1[idx], &shares[1].w1[idx]);
            let diff = (sum.to_f64() - full_w1[idx].to_f64()).abs();
            assert!(diff < 1e-6, "w1[{}] reconstruction failed: diff={}", idx, diff);
        }
    }

    #[test]
    fn test_redistribute_too_few_parties() {
        let cp = make_checkpoint(10, 4);
        let mut coord = RecoveryCoordinator::from_checkpoint(cp, 2, "test-session", 42);

        coord.remove_party(0).unwrap();
        coord.remove_party(1).unwrap();

        let result = coord.redistribute_shares(
            &[Fr::ZERO], &[Fr::ZERO], &[Fr::ZERO], &[Fr::ZERO],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_redistribute_with_macs() {
        let cp = make_checkpoint(10, 4);
        let mut coord = RecoveryCoordinator::from_checkpoint(cp, 3, "test-session", 42);

        coord.remove_party(1).unwrap();

        let full_w1 = vec![Fr::from_f64(1.0), Fr::from_f64(2.0)];
        let full_b1 = vec![Fr::from_f64(0.1)];
        let full_w2 = vec![Fr::from_f64(0.5)];
        let full_b2 = vec![Fr::from_f64(0.01)];

        let (shares, alpha, alpha_shares) = coord
            .redistribute_shares_with_macs(&full_w1, &full_b1, &full_w2, &full_b2)
            .unwrap();

        assert_eq!(shares.len(), 2);

        // Verify alpha shares sum to alpha.
        let alpha_sum = Fr::add(&alpha_shares[0], &alpha_shares[1]);
        assert!(alpha_sum.ct_eq(&alpha).to_bool(), "alpha shares should sum to alpha");

        // Verify MAC shares: sum(mac_shares) = alpha * value.
        for idx in 0..2 {
            let mac_sum = Fr::add(&shares[0].w1_macs[idx], &shares[1].w1_macs[idx]);
            let expected = Fr::mul(&alpha, &full_w1[idx]);
            assert!(
                mac_sum.ct_eq(&expected).to_bool(),
                "MAC sum for w1[{}] should equal alpha * value",
                idx
            );
        }
    }

    #[test]
    fn test_restore_state() {
        let cp = make_checkpoint(10, 4);
        let coord = RecoveryCoordinator::from_checkpoint(cp, 3, "test-session", 42);

        let state = coord.restore_state();
        assert_eq!(state.step, 10);
        assert_eq!(state.w1.len(), 4);
        assert_eq!(state.beaver_cursor, 10);
        assert_eq!(state.auth_beaver_cursor, 5);
    }

    #[test]
    fn test_disconnection_handler_creation() {
        let handler = DisconnectionHandler::new(3, 2, Duration::from_secs(10));
        assert_eq!(handler.active_count(), 3);
        assert!(handler.has_honest_majority());
    }

    #[test]
    fn test_disconnection_detection() {
        let mut handler = DisconnectionHandler::new(3, 2, Duration::from_millis(1));

        // Wait for grace period to expire.
        std::thread::sleep(Duration::from_millis(5));

        // Only party 0 is active.
        handler.record_activity(0);

        let disconnected = handler.check_disconnections();
        assert_eq!(disconnected.len(), 2);
        assert!(handler.is_disconnected(1));
        assert!(handler.is_disconnected(2));
        assert!(!handler.is_disconnected(0));
    }

    #[test]
    fn test_honest_majority_check() {
        let mut handler = DisconnectionHandler::new(3, 2, Duration::from_millis(1));

        std::thread::sleep(Duration::from_millis(5));
        handler.record_activity(0);
        handler.record_activity(1);
        handler.check_disconnections();

        // 2 active, min_honest=2 -> has majority.
        assert!(handler.has_honest_majority());
    }

    #[test]
    fn test_honest_majority_lost() {
        let mut handler = DisconnectionHandler::new(3, 3, Duration::from_millis(1));

        std::thread::sleep(Duration::from_millis(5));
        handler.record_activity(0);
        handler.check_disconnections();

        // 1 active, min_honest=3 -> no majority.
        assert!(!handler.has_honest_majority());
    }

    #[test]
    fn test_mark_disconnected() {
        let mut handler = DisconnectionHandler::new(3, 2, Duration::from_secs(60));
        handler.mark_disconnected(2);

        assert!(handler.is_disconnected(2));
        assert_eq!(handler.active_count(), 2);
    }

    #[test]
    fn test_additive_share_vec_correctness() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let values = vec![Fr::from_f64(1.0), Fr::from_f64(2.0), Fr::from_f64(3.0)];
        let shares = additive_share_vec(&values, 3, &mut rng);

        assert_eq!(shares.len(), 3);
        for idx in 0..3 {
            let sum = Fr::add(&Fr::add(&shares[0][idx], &shares[1][idx]), &shares[2][idx]);
            let diff = (sum.to_f64() - values[idx].to_f64()).abs();
            assert!(diff < 1e-6, "reconstruction failed for element {}", idx);
        }
    }
}
