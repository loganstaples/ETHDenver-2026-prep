//! Secret sharing schemes for MPC.
//!
//! Provides two sharing schemes:
//! - **Additive**: Split a secret s into n shares where s = s_1 + s_2 + ... + s_n.
//!   All shares are needed for reconstruction. Simple and efficient.
//! - **Shamir**: k-of-n threshold scheme based on polynomial interpolation.
//!   Any k shares suffice to reconstruct, fewer than k reveal nothing.
//!
//! Both schemes support sharing of scalars, vectors, tensors, and full model weights.

pub mod additive;
pub mod shamir;
pub mod tensor;
pub mod model;

pub use additive::AdditiveSharing;
pub use shamir::ShamirSharing;
pub use tensor::{TensorShare, TensorSharing};
pub use model::{ModelShare, ModelSharing};

use crate::error::MPCResult;
use crate::types::{PartyId, ShareId};
use serde::{Deserialize, Serialize};

/// A single scalar share held by one party.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarShare {
    /// Identifies this share (party, secret, index).
    pub id: ShareId,
    /// The share value.
    pub value: f64,
    /// Error bound introduced by the sharing process.
    pub error_bound: f64,
}

impl ScalarShare {
    pub fn new(id: ShareId, value: f64) -> Self {
        Self {
            id,
            value,
            error_bound: 0.0,
        }
    }

    pub fn with_error(mut self, error: f64) -> Self {
        self.error_bound = error;
        self
    }
}

/// A vector share: a share of an entire vector of values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorShare {
    /// Identifies this share.
    pub id: ShareId,
    /// The share values (one per element of the original vector).
    pub values: Vec<f64>,
    /// Error bound for the sharing process.
    pub error_bound: f64,
}

impl VectorShare {
    pub fn new(id: ShareId, values: Vec<f64>) -> Self {
        Self {
            id,
            values,
            error_bound: 0.0,
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn with_error(mut self, error: f64) -> Self {
        self.error_bound = error;
        self
    }
}

/// Trait for secret sharing schemes.
pub trait SecretSharingScheme {
    /// Splits a scalar secret into n shares.
    fn share_scalar(
        &self,
        secret: f64,
        secret_id: &str,
        parties: &[PartyId],
    ) -> MPCResult<Vec<ScalarShare>>;

    /// Reconstructs a scalar from shares.
    fn reconstruct_scalar(&self, shares: &[ScalarShare]) -> MPCResult<f64>;

    /// Splits a vector into n vector shares.
    fn share_vector(
        &self,
        secret: &[f64],
        secret_id: &str,
        parties: &[PartyId],
    ) -> MPCResult<Vec<VectorShare>>;

    /// Reconstructs a vector from vector shares.
    fn reconstruct_vector(&self, shares: &[VectorShare]) -> MPCResult<Vec<f64>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[test]
    fn test_scalar_share_creation() {
        let id = ShareId::new(PartyId::from_index(0), "test", 0);
        let share = ScalarShare::new(id.clone(), 42.0);
        assert_eq!(share.value, 42.0);
        assert_eq!(share.error_bound, 0.0);
    }

    #[test]
    fn test_vector_share_creation() {
        let id = ShareId::new(PartyId::from_index(0), "test", 0);
        let share = VectorShare::new(id, vec![1.0, 2.0, 3.0]);
        assert_eq!(share.len(), 3);
        assert!(!share.is_empty());
    }
}
