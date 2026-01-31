//! Secret sharing for tensors (BoundedTensor and WeightData).
//!
//! Extends the scalar/vector sharing schemes to work with the tensor types
//! used throughout HELIX. Each element of the tensor is shared independently.

use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::types::PartyId;

use super::{AdditiveSharing, ScalarShare, SecretSharingScheme, ShareId, ShamirSharing, VectorShare};

use serde::{Deserialize, Serialize};

/// A secret share of a tensor, held by one party.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorShare {
    /// Identifies this share.
    pub id: ShareId,
    /// The share data (flattened, same layout as the original tensor).
    pub data: Vec<Fr>,
    /// Shape of the original tensor.
    pub shape: Vec<usize>,
    /// Error bound from the sharing process.
    pub sharing_error: f64,
    /// Error bound from the original tensor.
    pub original_error: f64,
}

impl TensorShare {
    pub fn new(id: ShareId, data: Vec<Fr>, shape: Vec<usize>) -> Self {
        Self {
            id,
            data,
            shape,
            sharing_error: 0.0,
            original_error: 0.0,
        }
    }

    /// Creates a TensorShare from f64 data (convenience method).
    pub fn from_f64(id: ShareId, data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self {
            id,
            data: data.into_iter().map(Fr::from_f64).collect(),
            shape,
            sharing_error: 0.0,
            original_error: 0.0,
        }
    }

    /// Number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Converts this share to a BoundedTensor (for use in local computation).
    pub fn to_bounded_tensor(&self) -> BoundedTensor {
        let total_error = self.sharing_error + self.original_error;
        let data_f64: Vec<f64> = self.data.iter().map(|v| v.to_f64()).collect();
        BoundedTensor::from_approximate(data_f64, self.shape.clone(), total_error)
    }

    /// Element-wise addition of two tensor shares (linear homomorphism).
    pub fn add(&self, other: &TensorShare) -> MPCResult<TensorShare> {
        if self.shape != other.shape {
            return Err(MPCError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }

        let data: Vec<Fr> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| Fr::add(a, b))
            .collect();

        let mut result = TensorShare::new(self.id.clone(), data, self.shape.clone());
        result.sharing_error = self.sharing_error + other.sharing_error;
        result.original_error = self.original_error.max(other.original_error);
        Ok(result)
    }

    /// Element-wise subtraction of two tensor shares.
    pub fn sub(&self, other: &TensorShare) -> MPCResult<TensorShare> {
        if self.shape != other.shape {
            return Err(MPCError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }

        let data: Vec<Fr> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| Fr::sub(a, b))
            .collect();

        let mut result = TensorShare::new(self.id.clone(), data, self.shape.clone());
        result.sharing_error = self.sharing_error + other.sharing_error;
        result.original_error = self.original_error.max(other.original_error);
        Ok(result)
    }

    /// Scalar multiplication of a tensor share by a public constant.
    pub fn scale(&self, scalar: &Fr) -> TensorShare {
        let data: Vec<Fr> = self.data.iter().map(|v| Fr::mul(v, scalar)).collect();
        let scalar_f64 = scalar.to_f64().abs();
        let mut result = TensorShare::new(self.id.clone(), data, self.shape.clone());
        result.sharing_error = self.sharing_error * scalar_f64;
        result.original_error = self.original_error * scalar_f64;
        result
    }

    /// Scalar multiplication by f64 (convenience method).
    /// Uses fixed-point multiplication since scalar is converted to fixed-point representation.
    pub fn scale_f64(&self, scalar: f64) -> TensorShare {
        let scalar_fr = Fr::from_f64(scalar);
        // Use fixed_mul since both values are in fixed-point format
        let data: Vec<Fr> = self.data.iter().map(|v| v.fixed_mul(&scalar_fr)).collect();
        let scalar_f64 = scalar.abs();
        let mut result = TensorShare::new(self.id.clone(), data, self.shape.clone());
        result.sharing_error = self.sharing_error * scalar_f64;
        result.original_error = self.original_error * scalar_f64;
        result
    }

    /// Transposes a 2D tensor share.
    pub fn transpose(&self) -> MPCResult<TensorShare> {
        if self.shape.len() != 2 {
            return Err(MPCError::ShapeMismatch {
                expected: vec![0, 0], // indicates "expected 2D"
                got: self.shape.clone(),
            });
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![Fr::ZERO; self.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = self.data[i * cols + j].clone();
            }
        }

        let mut result = TensorShare::new(self.id.clone(), data, vec![cols, rows]);
        result.sharing_error = self.sharing_error;
        result.original_error = self.original_error;
        Ok(result)
    }
}

/// Operations for sharing and reconstructing tensors.
pub struct TensorSharing;

impl TensorSharing {
    /// Shares a BoundedTensor using additive sharing.
    pub fn share_additive(
        tensor: &BoundedTensor,
        secret_id: &str,
        parties: &[PartyId],
        sharing: &AdditiveSharing,
    ) -> MPCResult<Vec<TensorShare>> {
        let values: Vec<f64> = tensor.data().iter().map(|bv| bv.value()).collect();
        let shape = tensor.shape().clone();
        let original_error = tensor.max_error();

        let vector_shares = sharing.share_vector(&values, secret_id, parties)?;

        let tensor_shares = vector_shares
            .into_iter()
            .map(|vs| {
                let mut ts = TensorShare::from_f64(vs.id, vs.values, shape.clone());
                ts.original_error = original_error;
                ts
            })
            .collect();

        Ok(tensor_shares)
    }

    /// Shares a BoundedTensor using Shamir sharing.
    pub fn share_shamir(
        tensor: &BoundedTensor,
        secret_id: &str,
        parties: &[PartyId],
        sharing: &ShamirSharing,
    ) -> MPCResult<Vec<TensorShare>> {
        let values: Vec<f64> = tensor.data().iter().map(|bv| bv.value()).collect();
        let shape = tensor.shape().clone();
        let original_error = tensor.max_error();

        let vector_shares = sharing.share_vector(&values, secret_id, parties)?;

        let tensor_shares = vector_shares
            .into_iter()
            .map(|vs| {
                let mut ts = TensorShare::from_f64(vs.id, vs.values, shape.clone());
                ts.original_error = original_error;
                // Shamir has quantization error from field mapping.
                ts.sharing_error = 1e-6;
                ts
            })
            .collect();

        Ok(tensor_shares)
    }

    /// Reconstructs a BoundedTensor from additive tensor shares.
    pub fn reconstruct_additive(shares: &[TensorShare]) -> MPCResult<BoundedTensor> {
        if shares.is_empty() {
            return Err(MPCError::InsufficientShares {
                required: 2,
                available: 0,
            });
        }

        let shape = shares[0].shape.clone();
        let dim = shares[0].data.len();

        for s in shares {
            if s.shape != shape {
                return Err(MPCError::ShapeMismatch {
                    expected: shape,
                    got: s.shape.clone(),
                });
            }
        }

        let mut values = vec![Fr::ZERO; dim];
        for s in shares {
            for (i, v) in s.data.iter().enumerate() {
                values[i] = Fr::add(&values[i], v);
            }
        }

        let values_f64: Vec<f64> = values.iter().map(|v| v.to_f64()).collect();

        let max_sharing_error: f64 = shares.iter().map(|s| s.sharing_error).sum();
        let max_original_error: f64 = shares
            .iter()
            .map(|s| s.original_error)
            .fold(0.0, f64::max);
        let total_error = max_sharing_error + max_original_error;

        Ok(BoundedTensor::from_approximate(
            values_f64,
            shape,
            total_error,
        ))
    }

    /// Reconstructs a BoundedTensor from Shamir tensor shares.
    pub fn reconstruct_shamir(
        shares: &[TensorShare],
        sharing: &ShamirSharing,
    ) -> MPCResult<BoundedTensor> {
        if shares.len() < sharing.threshold() {
            return Err(MPCError::InsufficientShares {
                required: sharing.threshold(),
                available: shares.len(),
            });
        }

        let shape = shares[0].shape.clone();

        // Convert TensorShares to VectorShares for the Shamir reconstruction.
        // Convert Fr back to f64 for the VectorShare API.
        let vector_shares: Vec<VectorShare> = shares
            .iter()
            .map(|ts| {
                let values: Vec<f64> = ts.data.iter().map(|v| v.to_f64()).collect();
                VectorShare::new(ts.id.clone(), values)
            })
            .collect();

        let values = sharing.reconstruct_vector(&vector_shares)?;

        let max_error: f64 = shares
            .iter()
            .map(|s| s.sharing_error + s.original_error)
            .fold(0.0, f64::max);

        Ok(BoundedTensor::from_approximate(values, shape, max_error))
    }

    /// Shares raw weight data (Vec<f32> with shape) using additive sharing.
    pub fn share_weight_data_additive(
        data: &[f32],
        shape: &[usize],
        error_bound: f64,
        secret_id: &str,
        parties: &[PartyId],
        sharing: &AdditiveSharing,
    ) -> MPCResult<Vec<TensorShare>> {
        let values: Vec<f64> = data.iter().map(|v| *v as f64).collect();

        let vector_shares = sharing.share_vector(&values, secret_id, parties)?;

        let tensor_shares = vector_shares
            .into_iter()
            .map(|vs| {
                let mut ts = TensorShare::from_f64(vs.id, vs.values, shape.to_vec());
                ts.original_error = error_bound;
                ts
            })
            .collect();

        Ok(tensor_shares)
    }

    /// Reconstructs raw weight data from additive tensor shares.
    pub fn reconstruct_weight_data_additive(shares: &[TensorShare]) -> MPCResult<(Vec<f32>, f64)> {
        let tensor = Self::reconstruct_additive(shares)?;
        let data: Vec<f32> = tensor.data().iter().map(|bv| bv.value() as f32).collect();
        let error = tensor.max_error();
        Ok((data, error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parties(n: usize) -> Vec<PartyId> {
        (0..n).map(PartyId::from_index).collect()
    }

    #[test]
    fn test_tensor_share_additive() {
        let tensor = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let parties = test_parties(3);
        let sharing = AdditiveSharing::with_seed(42);

        let shares = TensorSharing::share_additive(&tensor, "test", &parties, &sharing).unwrap();
        assert_eq!(shares.len(), 3);
        assert_eq!(shares[0].shape, vec![2, 2]);
        assert_eq!(shares[0].numel(), 4);

        let recon = TensorSharing::reconstruct_additive(&shares).unwrap();
        assert_eq!(recon.shape(), &vec![2, 2]);

        let orig_vals = tensor.values();
        let recon_vals = recon.values();
        for (a, b) in orig_vals.iter().zip(recon_vals.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Tensor recon failed: {} vs {}",
                a,
                b,
            );
        }
    }

    #[test]
    fn test_tensor_share_shamir() {
        let tensor = BoundedTensor::from_exact(vec![10.0, 20.0, 30.0], vec![3]);
        let parties = test_parties(3);
        let sharing = ShamirSharing::with_seed(2, 42);

        let shares = TensorSharing::share_shamir(&tensor, "test", &parties, &sharing).unwrap();
        assert_eq!(shares.len(), 3);

        // Reconstruct with just 2 shares (2-of-3 threshold).
        let recon = TensorSharing::reconstruct_shamir(&shares[..2], &sharing).unwrap();
        let orig_vals = tensor.values();
        let recon_vals = recon.values();

        for (a, b) in orig_vals.iter().zip(recon_vals.iter()) {
            assert!(
                (a - b).abs() < 0.01,
                "Shamir tensor recon failed: {} vs {}",
                a,
                b,
            );
        }
    }

    #[test]
    fn test_tensor_share_linear_homomorphism() {
        let parties = test_parties(3);
        let sharing = AdditiveSharing::with_seed(42);

        let a = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let b = BoundedTensor::from_exact(vec![4.0, 5.0, 6.0], vec![3]);

        let shares_a = TensorSharing::share_additive(&a, "a", &parties, &sharing).unwrap();
        let sharing_b = AdditiveSharing::with_seed(99);
        let shares_b = TensorSharing::share_additive(&b, "b", &parties, &sharing_b).unwrap();

        // Add shares element-wise.
        let shares_c: Vec<TensorShare> = shares_a
            .iter()
            .zip(&shares_b)
            .map(|(sa, sb)| sa.add(sb).unwrap())
            .collect();

        let result = TensorSharing::reconstruct_additive(&shares_c).unwrap();
        let expected = vec![5.0, 7.0, 9.0];

        for (a, b) in result.values().iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Homomorphic add failed: {} vs {}",
                a,
                b,
            );
        }
    }

    #[test]
    fn test_tensor_share_scale() {
        let parties = test_parties(3);
        let sharing = AdditiveSharing::with_seed(42);

        let tensor = BoundedTensor::from_exact(vec![2.0, 4.0, 6.0], vec![3]);
        let shares = TensorSharing::share_additive(&tensor, "t", &parties, &sharing).unwrap();

        // Scale each share by 3.
        let scaled: Vec<TensorShare> = shares.iter().map(|s| s.scale_f64(3.0)).collect();

        let result = TensorSharing::reconstruct_additive(&scaled).unwrap();
        let expected = vec![6.0, 12.0, 18.0];

        for (a, b) in result.values().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tensor_share_transpose() {
        let parties = test_parties(3);
        let sharing = AdditiveSharing::with_seed(42);

        let tensor = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let shares = TensorSharing::share_additive(&tensor, "t", &parties, &sharing).unwrap();

        let transposed: Vec<TensorShare> = shares.iter().map(|s| s.transpose().unwrap()).collect();

        let result = TensorSharing::reconstruct_additive(&transposed).unwrap();
        assert_eq!(result.shape(), &vec![3, 2]);

        // Original [1,2,3; 4,5,6] transposed = [1,4; 2,5; 3,6]
        let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        for (a, b) in result.values().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_weight_data_sharing() {
        let parties = test_parties(3);
        let sharing = AdditiveSharing::with_seed(42);

        let data = vec![0.1f32, 0.2, 0.3, 0.4];
        let shape = vec![2, 2];

        let shares = TensorSharing::share_weight_data_additive(
            &data, &shape, 0.01, "w", &parties, &sharing,
        )
        .unwrap();

        let (recon_data, error) =
            TensorSharing::reconstruct_weight_data_additive(&shares).unwrap();

        for (a, b) in recon_data.iter().zip(data.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "Weight data recon failed: {} vs {}",
                a,
                b,
            );
        }
    }
}
