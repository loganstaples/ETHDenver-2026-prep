//! Sybil Resistance for HELIX Network.
//!
//! Implements stake-weighted peer selection to prevent Sybil attacks where
//! an attacker creates many fake identities to gain disproportionate influence.
//!
//! Key mechanisms:
//! - Stake-weighted peer selection for consensus participation
//! - Minimum stake requirements for network participation
//! - Quadratic voting/selection for democratic influence limits
//! - Identity verification through staking commitment

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use std::time::{Duration, Instant};

use super::messages::PeerId;
use serde::{Deserialize, Serialize};

/// Configuration for Sybil resistance.
#[derive(Debug, Clone)]
pub struct SybilResistanceConfig {
    /// Minimum stake required to participate.
    pub min_stake: u64,
    /// Maximum stake influence (caps large stakers).
    pub max_stake_influence: u64,
    /// Whether to use quadratic weighting.
    pub use_quadratic_weighting: bool,
    /// Stake lock period in seconds.
    pub stake_lock_period: Duration,
    /// Whether to verify stake on-chain.
    pub verify_on_chain: bool,
    /// Maximum peers to select for a task.
    pub max_selected_peers: usize,
    /// Minimum peers required for a task.
    pub min_selected_peers: usize,
    /// Randomness factor for selection (0.0-1.0).
    pub randomness_factor: f64,
    /// Cooldown period between selections for same peer.
    pub selection_cooldown: Duration,
    /// Whether to penalize peers for bad behavior.
    pub enable_penalties: bool,
    /// Penalty decay rate per hour.
    pub penalty_decay_rate: f64,
}

impl Default for SybilResistanceConfig {
    fn default() -> Self {
        Self {
            min_stake: 100,
            max_stake_influence: 10000,
            use_quadratic_weighting: true,
            stake_lock_period: Duration::from_secs(3600), // 1 hour
            verify_on_chain: false,
            max_selected_peers: 10,
            min_selected_peers: 3,
            randomness_factor: 0.2,
            selection_cooldown: Duration::from_secs(60),
            enable_penalties: true,
            penalty_decay_rate: 0.1,
        }
    }
}

/// Stake information for a peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerStake {
    /// Peer ID.
    pub peer_id: PeerId,
    /// Total staked amount.
    pub stake: u64,
    /// Effective stake (after penalties).
    pub effective_stake: u64,
    /// Time when stake was locked.
    pub locked_at: u64,
    /// Whether stake is verified on-chain.
    pub verified: bool,
    /// Penalty factor (0.0-1.0, lower is worse).
    pub penalty_factor: f64,
    /// Last selection time (unix timestamp millis, 0 if never selected).
    #[serde(skip)]
    pub last_selected: Option<Instant>,
    /// Selection count in current epoch.
    pub selection_count: u32,
    /// On-chain stake verification hash.
    pub verification_hash: Option<[u8; 32]>,
}

impl PeerStake {
    /// Creates a new peer stake record.
    pub fn new(peer_id: PeerId, stake: u64) -> Self {
        Self {
            peer_id,
            stake,
            effective_stake: stake,
            locked_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            verified: false,
            penalty_factor: 1.0,
            last_selected: None,
            selection_count: 0,
            verification_hash: None,
        }
    }

    /// Updates the effective stake based on penalties.
    pub fn update_effective_stake(&mut self, max_influence: u64, use_quadratic: bool) {
        // Apply penalty factor
        let penalized = (self.stake as f64 * self.penalty_factor) as u64;

        // Cap at max influence
        let capped = penalized.min(max_influence);

        // Apply quadratic weighting if enabled
        self.effective_stake = if use_quadratic {
            (capped as f64).sqrt() as u64
        } else {
            capped
        };
    }

    /// Applies a penalty to this stake.
    pub fn apply_penalty(&mut self, amount: f64) {
        self.penalty_factor = (self.penalty_factor - amount).max(0.0);
    }

    /// Decays penalty over time.
    pub fn decay_penalty(&mut self, rate: f64, hours_elapsed: f64) {
        let decay = rate * hours_elapsed;
        self.penalty_factor = (self.penalty_factor + decay).min(1.0);
    }

    /// Returns whether this peer can be selected (cooldown check).
    pub fn can_be_selected(&self, cooldown: Duration) -> bool {
        match self.last_selected {
            Some(last) => last.elapsed() >= cooldown,
            None => true,
        }
    }

    /// Marks this peer as selected.
    pub fn mark_selected(&mut self) {
        self.last_selected = Some(Instant::now());
        self.selection_count += 1;
    }
}

/// Entry in the selection priority queue.
#[derive(Debug, Clone)]
struct SelectionCandidate {
    peer_id: PeerId,
    weight: f64,
    stake: u64,
}

impl PartialEq for SelectionCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl Eq for SelectionCandidate {}

impl PartialOrd for SelectionCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SelectionCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher weight = higher priority
        self.weight.partial_cmp(&other.weight).unwrap_or(Ordering::Equal)
    }
}

/// Result of peer selection.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Selected peers.
    pub selected: Vec<PeerId>,
    /// Total effective stake of selected peers.
    pub total_stake: u64,
    /// Peers that were excluded (and why).
    pub excluded: HashMap<PeerId, ExclusionReason>,
    /// Whether minimum peer count was met.
    pub sufficient_peers: bool,
    /// Selection entropy (measure of randomness).
    pub entropy: f64,
}

/// Reason for excluding a peer from selection.
#[derive(Debug, Clone)]
pub enum ExclusionReason {
    /// Stake below minimum.
    InsufficientStake { stake: u64, required: u64 },
    /// Stake not verified.
    Unverified,
    /// In cooldown period.
    Cooldown { remaining: Duration },
    /// Penalized too heavily.
    Penalized { factor: f64 },
    /// Already at selection limit.
    SelectionLimit,
}

/// Sybil-resistant peer selector.
///
/// Implements stake-weighted random selection with Sybil resistance.
pub struct SybilResistantSelector {
    /// Configuration.
    config: SybilResistanceConfig,
    /// Peer stakes.
    stakes: HashMap<PeerId, PeerStake>,
    /// Selection history (for fairness tracking).
    selection_history: Vec<SelectionRecord>,
    /// Maximum history size.
    max_history: usize,
    /// Random seed for deterministic selection.
    seed: u64,
}

/// Record of a selection event.
#[derive(Debug, Clone)]
struct SelectionRecord {
    selected: Vec<PeerId>,
    timestamp: Instant,
    total_stake: u64,
}

impl SybilResistantSelector {
    /// Creates a new selector.
    pub fn new(config: SybilResistanceConfig) -> Self {
        Self {
            config,
            stakes: HashMap::new(),
            selection_history: Vec::new(),
            max_history: 1000,
            seed: rand::random(),
        }
    }

    /// Registers a peer's stake.
    pub fn register_stake(&mut self, peer_id: PeerId, stake: u64) -> Result<(), SybilError> {
        if stake < self.config.min_stake {
            return Err(SybilError::InsufficientStake {
                provided: stake,
                required: self.config.min_stake,
            });
        }

        let mut peer_stake = PeerStake::new(peer_id.clone(), stake);
        peer_stake.update_effective_stake(
            self.config.max_stake_influence,
            self.config.use_quadratic_weighting,
        );

        self.stakes.insert(peer_id, peer_stake);
        Ok(())
    }

    /// Unregisters a peer's stake.
    pub fn unregister_stake(&mut self, peer_id: &PeerId) {
        self.stakes.remove(peer_id);
    }

    /// Updates a peer's stake.
    pub fn update_stake(&mut self, peer_id: &PeerId, new_stake: u64) -> Result<(), SybilError> {
        if new_stake < self.config.min_stake {
            return Err(SybilError::InsufficientStake {
                provided: new_stake,
                required: self.config.min_stake,
            });
        }

        if let Some(stake) = self.stakes.get_mut(peer_id) {
            stake.stake = new_stake;
            stake.update_effective_stake(
                self.config.max_stake_influence,
                self.config.use_quadratic_weighting,
            );
            Ok(())
        } else {
            Err(SybilError::PeerNotFound(peer_id.clone()))
        }
    }

    /// Applies a penalty to a peer.
    pub fn apply_penalty(&mut self, peer_id: &PeerId, amount: f64) {
        if let Some(stake) = self.stakes.get_mut(peer_id) {
            stake.apply_penalty(amount);
            stake.update_effective_stake(
                self.config.max_stake_influence,
                self.config.use_quadratic_weighting,
            );
        }
    }

    /// Decays all penalties.
    pub fn decay_all_penalties(&mut self, hours_elapsed: f64) {
        for stake in self.stakes.values_mut() {
            stake.decay_penalty(self.config.penalty_decay_rate, hours_elapsed);
            stake.update_effective_stake(
                self.config.max_stake_influence,
                self.config.use_quadratic_weighting,
            );
        }
    }

    /// Selects peers weighted by stake.
    ///
    /// Uses weighted random selection to pick peers, with:
    /// - Stake-proportional probability
    /// - Randomness factor for unpredictability
    /// - Cooldown enforcement
    /// - Minimum stake requirements
    pub fn select_peers(&mut self, count: usize) -> SelectionResult {
        let target_count = count.min(self.config.max_selected_peers);
        let mut selected = Vec::with_capacity(target_count);
        let mut excluded: HashMap<PeerId, ExclusionReason> = HashMap::new();
        let mut total_stake = 0u64;

        // Build candidate list
        let mut candidates: BinaryHeap<SelectionCandidate> = BinaryHeap::new();
        let mut total_effective_stake = 0u64;

        for (peer_id, stake) in &self.stakes {
            // Check minimum stake
            if stake.effective_stake < self.config.min_stake {
                excluded.insert(
                    peer_id.clone(),
                    ExclusionReason::InsufficientStake {
                        stake: stake.effective_stake,
                        required: self.config.min_stake,
                    },
                );
                continue;
            }

            // Check verification if required
            if self.config.verify_on_chain && !stake.verified {
                excluded.insert(peer_id.clone(), ExclusionReason::Unverified);
                continue;
            }

            // Check cooldown
            if !stake.can_be_selected(self.config.selection_cooldown) {
                if let Some(last) = stake.last_selected {
                    let remaining = self.config.selection_cooldown
                        .saturating_sub(last.elapsed());
                    excluded.insert(peer_id.clone(), ExclusionReason::Cooldown { remaining });
                }
                continue;
            }

            // Check penalty threshold
            if stake.penalty_factor < 0.1 {
                excluded.insert(
                    peer_id.clone(),
                    ExclusionReason::Penalized {
                        factor: stake.penalty_factor,
                    },
                );
                continue;
            }

            total_effective_stake += stake.effective_stake;

            // Calculate weight with randomness
            let base_weight = stake.effective_stake as f64;
            let random_factor = 1.0 + (rand::random::<f64>() - 0.5) * 2.0 * self.config.randomness_factor;
            let weight = base_weight * random_factor;

            candidates.push(SelectionCandidate {
                peer_id: peer_id.clone(),
                weight,
                stake: stake.effective_stake,
            });
        }

        // Select top candidates
        while selected.len() < target_count {
            match candidates.pop() {
                Some(candidate) => {
                    selected.push(candidate.peer_id.clone());
                    total_stake += candidate.stake;

                    // Mark as selected
                    if let Some(stake) = self.stakes.get_mut(&candidate.peer_id) {
                        stake.mark_selected();
                    }
                }
                None => break,
            }
        }

        // Calculate entropy (measure of selection randomness)
        let entropy = if total_effective_stake > 0 && !selected.is_empty() {
            let avg_stake = total_stake as f64 / selected.len() as f64;
            let variance: f64 = self.stakes
                .values()
                .filter(|s| selected.contains(&s.peer_id))
                .map(|s| (s.effective_stake as f64 - avg_stake).powi(2))
                .sum::<f64>()
                / selected.len() as f64;
            1.0 / (1.0 + variance.sqrt() / avg_stake)
        } else {
            0.0
        };

        let sufficient_peers = selected.len() >= self.config.min_selected_peers;

        // Record selection
        if !selected.is_empty() {
            self.record_selection(&selected, total_stake);
        }

        SelectionResult {
            selected,
            total_stake,
            excluded,
            sufficient_peers,
            entropy,
        }
    }

    /// Records a selection for history tracking.
    fn record_selection(&mut self, selected: &[PeerId], total_stake: u64) {
        self.selection_history.push(SelectionRecord {
            selected: selected.to_vec(),
            timestamp: Instant::now(),
            total_stake,
        });

        // Prune old history
        if self.selection_history.len() > self.max_history {
            self.selection_history.remove(0);
        }
    }

    /// Returns stake information for a peer.
    pub fn get_stake(&self, peer_id: &PeerId) -> Option<&PeerStake> {
        self.stakes.get(peer_id)
    }

    /// Returns total stake in the system.
    pub fn total_stake(&self) -> u64 {
        self.stakes.values().map(|s| s.stake).sum()
    }

    /// Returns total effective stake.
    pub fn total_effective_stake(&self) -> u64 {
        self.stakes.values().map(|s| s.effective_stake).sum()
    }

    /// Returns number of registered peers.
    pub fn peer_count(&self) -> usize {
        self.stakes.len()
    }

    /// Returns peers with stake above threshold.
    pub fn peers_above_threshold(&self, threshold: u64) -> Vec<PeerId> {
        self.stakes
            .iter()
            .filter(|(_, s)| s.effective_stake >= threshold)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Calculates fairness score (how evenly distributed selections are).
    pub fn fairness_score(&self) -> f64 {
        if self.stakes.is_empty() {
            return 1.0;
        }

        let selection_counts: Vec<u32> = self.stakes
            .values()
            .map(|s| s.selection_count)
            .collect();

        let total: u32 = selection_counts.iter().sum();
        if total == 0 {
            return 1.0;
        }

        let avg = total as f64 / selection_counts.len() as f64;
        let variance: f64 = selection_counts
            .iter()
            .map(|&c| (c as f64 - avg).powi(2))
            .sum::<f64>()
            / selection_counts.len() as f64;

        1.0 / (1.0 + variance.sqrt() / avg)
    }

    /// Resets selection counts for new epoch.
    pub fn reset_epoch(&mut self) {
        for stake in self.stakes.values_mut() {
            stake.selection_count = 0;
            stake.last_selected = None;
        }
        self.selection_history.clear();
    }

    /// Verifies stake on-chain (stub for integration).
    pub async fn verify_stake_on_chain(&mut self, peer_id: &PeerId) -> Result<bool, SybilError> {
        // In production, would call smart contract to verify
        // For now, just mark as verified if stake exists
        if let Some(stake) = self.stakes.get_mut(peer_id) {
            stake.verified = true;
            Ok(true)
        } else {
            Err(SybilError::PeerNotFound(peer_id.clone()))
        }
    }
}

/// Sybil resistance errors.
#[derive(Debug, Clone)]
pub enum SybilError {
    /// Stake below minimum.
    InsufficientStake { provided: u64, required: u64 },
    /// Peer not found.
    PeerNotFound(PeerId),
    /// Stake verification failed.
    VerificationFailed(String),
    /// Selection failed.
    SelectionFailed(String),
}

impl std::fmt::Display for SybilError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientStake { provided, required } => {
                write!(f, "Insufficient stake: {} < {}", provided, required)
            }
            Self::PeerNotFound(id) => write!(f, "Peer not found: {}", id),
            Self::VerificationFailed(msg) => write!(f, "Verification failed: {}", msg),
            Self::SelectionFailed(msg) => write!(f, "Selection failed: {}", msg),
        }
    }
}

impl std::error::Error for SybilError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stake_registration() {
        let config = SybilResistanceConfig::default();
        let mut selector = SybilResistantSelector::new(config);

        let peer = PeerId::from_string("peer1");
        selector.register_stake(peer.clone(), 1000).unwrap();

        assert!(selector.get_stake(&peer).is_some());
        assert_eq!(selector.total_stake(), 1000);
    }

    #[test]
    fn test_insufficient_stake() {
        let config = SybilResistanceConfig {
            min_stake: 100,
            ..Default::default()
        };
        let mut selector = SybilResistantSelector::new(config);

        let peer = PeerId::from_string("peer1");
        let result = selector.register_stake(peer, 50);

        assert!(matches!(result, Err(SybilError::InsufficientStake { .. })));
    }

    #[test]
    fn test_peer_selection() {
        let config = SybilResistanceConfig {
            min_stake: 100,
            selection_cooldown: Duration::from_secs(0), // No cooldown for test
            ..Default::default()
        };
        let mut selector = SybilResistantSelector::new(config);

        // Register several peers
        for i in 0..5 {
            let peer = PeerId::from_string(format!("peer{}", i));
            selector.register_stake(peer, 1000 + i * 100).unwrap();
        }

        let result = selector.select_peers(3);
        assert_eq!(result.selected.len(), 3);
        assert!(result.sufficient_peers);
    }

    #[test]
    fn test_quadratic_weighting() {
        let config = SybilResistanceConfig {
            use_quadratic_weighting: true,
            min_stake: 1,
            max_stake_influence: 10000,
            ..Default::default()
        };
        let mut selector = SybilResistantSelector::new(config);

        // Large staker
        let whale = PeerId::from_string("whale");
        selector.register_stake(whale.clone(), 10000).unwrap();

        // Small staker
        let small = PeerId::from_string("small");
        selector.register_stake(small.clone(), 100).unwrap();

        let whale_stake = selector.get_stake(&whale).unwrap();
        let small_stake = selector.get_stake(&small).unwrap();

        // With quadratic weighting, influence should be sqrt of stake
        // Whale: sqrt(10000) = 100, Small: sqrt(100) = 10
        // Ratio should be 10:1 instead of 100:1
        let ratio = whale_stake.effective_stake as f64 / small_stake.effective_stake as f64;
        assert!(ratio < 15.0); // Should be ~10
    }

    #[test]
    fn test_penalty_application() {
        let config = SybilResistanceConfig::default();
        let mut selector = SybilResistantSelector::new(config);

        let peer = PeerId::from_string("peer1");
        selector.register_stake(peer.clone(), 1000).unwrap();

        let original = selector.get_stake(&peer).unwrap().effective_stake;

        selector.apply_penalty(&peer, 0.5);

        let penalized = selector.get_stake(&peer).unwrap().effective_stake;
        assert!(penalized < original);
    }

    #[test]
    fn test_fairness_score() {
        let config = SybilResistanceConfig {
            selection_cooldown: Duration::from_secs(0),
            ..Default::default()
        };
        let mut selector = SybilResistantSelector::new(config);

        for i in 0..5 {
            let peer = PeerId::from_string(format!("peer{}", i));
            selector.register_stake(peer, 1000).unwrap();
        }

        // Initial fairness should be high (no selections yet)
        let initial_fairness = selector.fairness_score();
        assert!(initial_fairness > 0.9);

        // After some selections, fairness might decrease
        for _ in 0..10 {
            selector.select_peers(2);
        }

        // Fairness should still be reasonable
        let final_fairness = selector.fairness_score();
        assert!(final_fairness > 0.5);
    }
}
