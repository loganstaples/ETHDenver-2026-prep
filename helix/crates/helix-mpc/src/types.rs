//! Core MPC types used throughout the crate.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Identifies a party in the MPC protocol.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct PartyId(pub String);

impl PartyId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Creates a party ID from a numeric index (e.g., "party-0").
    pub fn from_index(idx: usize) -> Self {
        Self(format!("party-{}", idx))
    }
}

impl fmt::Display for PartyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identifies a specific secret share.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShareId {
    /// Which party holds this share.
    pub party: PartyId,
    /// Unique identifier for the shared secret this belongs to.
    pub secret_id: String,
    /// Share index (for Shamir: evaluation point; for additive: party index).
    pub index: usize,
}

impl ShareId {
    pub fn new(party: PartyId, secret_id: impl Into<String>, index: usize) -> Self {
        Self {
            party,
            secret_id: secret_id.into(),
            index,
        }
    }
}

impl fmt::Display for ShareId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}[{}]", self.party, self.secret_id, self.index)
    }
}

/// Configuration for MPC operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MPCConfig {
    /// Number of parties in the computation.
    pub num_parties: usize,
    /// Threshold for Shamir sharing (k-of-n). Must satisfy k <= n.
    /// For additive sharing, this is ignored.
    pub threshold: usize,
    /// Statistical security parameter (bits). Controls soundness of
    /// probabilistic checks. Typical values: 40 or 80.
    pub statistical_security: usize,
    /// Whether to use Shamir (true) or additive (false) sharing as default.
    pub use_shamir: bool,
    /// Maximum number of Beaver triples to pre-generate per batch.
    pub beaver_batch_size: usize,
    /// How often to re-share weights (in training steps) to prevent
    /// gradient accumulation attacks.
    pub reshare_interval: u64,
    /// Whether to verify share consistency using commitments.
    pub verify_shares: bool,
    /// Field modulus for finite field arithmetic. Using a large prime
    /// that fits in f64 without precision loss: 2^53 - 1 is not prime,
    /// so we use a nearby prime.
    pub field_prime: u64,
}

impl Default for MPCConfig {
    fn default() -> Self {
        Self {
            num_parties: 3,
            threshold: 2,
            statistical_security: 40,
            use_shamir: false,
            beaver_batch_size: 1024,
            reshare_interval: 50,
            verify_shares: true,
            // Large prime that fits comfortably in f64: 2^31 - 1 (Mersenne prime).
            // For real deployment a larger field would be used, but this suffices
            // for the demo and avoids f64 precision issues.
            field_prime: 2_147_483_647,
        }
    }
}

impl MPCConfig {
    /// Creates a config for a 2-party additive sharing setup.
    pub fn two_party() -> Self {
        Self {
            num_parties: 2,
            threshold: 2,
            use_shamir: false,
            ..Default::default()
        }
    }

    /// Creates a config for a 3-party setup (2-of-3 Shamir or 3-additive).
    pub fn three_party() -> Self {
        Self {
            num_parties: 3,
            threshold: 2,
            ..Default::default()
        }
    }

    /// Creates a config for N-party setup with t-of-n Shamir sharing.
    pub fn shamir(num_parties: usize, threshold: usize) -> Self {
        assert!(
            threshold <= num_parties,
            "threshold {} must be <= num_parties {}",
            threshold,
            num_parties,
        );
        assert!(threshold >= 1, "threshold must be >= 1");
        Self {
            num_parties,
            threshold,
            use_shamir: true,
            ..Default::default()
        }
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_parties < 2 {
            return Err("MPC requires at least 2 parties".into());
        }
        if self.use_shamir && self.threshold > self.num_parties {
            return Err(format!(
                "Shamir threshold {} exceeds num_parties {}",
                self.threshold, self.num_parties
            ));
        }
        if self.use_shamir && self.threshold < 1 {
            return Err("Shamir threshold must be at least 1".into());
        }
        if self.field_prime < 2 {
            return Err("Field prime must be >= 2".into());
        }
        Ok(())
    }
}

/// Represents the role a party plays in the MPC protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartyRole {
    /// Model owner who initiates sharing and collects results.
    Dealer,
    /// Compute worker who holds shares and performs computation.
    Worker,
    /// Aggregator who combines results from workers.
    Aggregator,
}

/// Tracks which phase the MPC computation is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MPCPhase {
    /// Preprocessing: generating Beaver triples, commitments.
    Preprocessing,
    /// Input sharing: dealer distributes shares to workers.
    InputSharing,
    /// Computation: workers compute on shares.
    Computation,
    /// Output reconstruction: workers send shares to reconstruct result.
    OutputReconstruction,
    /// Verification: checking correctness of computation.
    Verification,
    /// Complete: MPC protocol finished successfully.
    Complete,
    /// Failed: protocol aborted due to error.
    Failed,
}

impl fmt::Display for MPCPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preprocessing => write!(f, "Preprocessing"),
            Self::InputSharing => write!(f, "InputSharing"),
            Self::Computation => write!(f, "Computation"),
            Self::OutputReconstruction => write!(f, "OutputReconstruction"),
            Self::Verification => write!(f, "Verification"),
            Self::Complete => write!(f, "Complete"),
            Self::Failed => write!(f, "Failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_party_id() {
        let p = PartyId::from_index(0);
        assert_eq!(p.0, "party-0");
        assert_eq!(format!("{}", p), "party-0");
    }

    #[test]
    fn test_share_id() {
        let sid = ShareId::new(PartyId::from_index(1), "weights-layer0", 1);
        assert_eq!(format!("{}", sid), "party-1:weights-layer0[1]");
    }

    #[test]
    fn test_config_validation() {
        assert!(MPCConfig::default().validate().is_ok());
        assert!(MPCConfig::two_party().validate().is_ok());
        assert!(MPCConfig::shamir(5, 3).validate().is_ok());

        let bad = MPCConfig {
            num_parties: 1,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }
}
