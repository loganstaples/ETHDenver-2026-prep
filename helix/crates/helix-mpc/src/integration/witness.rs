//! MPC to Circuit Witness Conversion.
//!
//! This module provides conversion utilities between MPC secret shares
//! and the witness format expected by helix-circuits' MLTrainingStepV2Circuit.
//!
//! The key challenge is that MPC shares are distributed across parties,
//! but the circuit needs a complete witness. This module handles:
//!
//! 1. Reconstructing full weights from shares (only in aggregator)
//! 2. Converting MPC data types to circuit-compatible field elements
//! 3. Computing required intermediate values for the circuit witness
//! 4. Generating Freivalds verification challenges

use sha2::{Digest, Sha256};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::integration::witness_format::{MPCTrainingWitness, ReconstructedWitness};
use crate::poseidon::{poseidon_state_hash, share_commitment};
use crate::sharing::model::ModelShare;
use crate::types::PartyId;

/// Converts reconstructed MPC witness to the format expected by helix-circuits.
///
/// This produces a `CircuitWitness` that can be passed directly to
/// `helix_prover::MLTrainingProverV2::prove()`.
#[derive(Debug, Clone)]
pub struct CircuitWitness {
    /// Model dimensions
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,

    /// Input features (d_in elements)
    pub x: Vec<Fr>,
    /// Target values (d_out elements)
    pub target: Vec<Fr>,

    /// Layer 1 weights (d_hid × d_in, row-major)
    pub w1: Vec<Fr>,
    /// Layer 1 biases (d_hid elements)
    pub b1: Vec<Fr>,
    /// Layer 2 weights (d_out × d_hid, row-major)
    pub w2: Vec<Fr>,
    /// Layer 2 biases (d_out elements)
    pub b2: Vec<Fr>,

    /// Forward pass intermediates
    pub h_pre: Vec<Fr>,
    pub h_pre_err: Vec<Fr>,
    pub h: Vec<Fr>,
    pub h_err: Vec<Fr>,
    pub y: Vec<Fr>,
    pub y_err: Vec<Fr>,
    pub loss: Fr,
    pub loss_err: Fr,

    /// Backward pass gradients
    pub dy: Vec<Fr>,
    pub dy_err: Vec<Fr>,
    pub dw2: Vec<Fr>,
    pub dw2_err: Vec<Fr>,
    pub db2: Vec<Fr>,
    pub db2_err: Vec<Fr>,
    pub dh: Vec<Fr>,
    pub dh_err: Vec<Fr>,
    pub relu_mask: Vec<Fr>,
    pub dh_pre: Vec<Fr>,
    pub dh_pre_err: Vec<Fr>,
    pub dw1: Vec<Fr>,
    pub dw1_err: Vec<Fr>,
    pub db1: Vec<Fr>,
    pub db1_err: Vec<Fr>,

    /// Learning rate
    pub lr: Fr,

    /// Updated weights
    pub w1_new: Vec<Fr>,
    pub b1_new: Vec<Fr>,
    pub w2_new: Vec<Fr>,
    pub b2_new: Vec<Fr>,

    /// Freivalds verification vectors
    pub freivalds_r1: Vec<Fr>,
    pub freivalds_r2: Vec<Fr>,

    /// Total accumulated error
    pub total_error: Fr,

    /// State hashes (lo, hi pairs)
    pub old_state_hash: (Fr, Fr),
    pub new_state_hash: (Fr, Fr),

    /// Training step number
    pub step_number: u64,
}

impl CircuitWitness {
    /// Creates a circuit witness from a reconstructed MPC witness.
    ///
    /// This is the main conversion function used after share aggregation.
    pub fn from_reconstructed(recon: &ReconstructedWitness, base_error: f64) -> MPCResult<Self> {
        // Compute forward pass
        let forward = Self::compute_forward_pass(
            recon.d_in,
            recon.d_hid,
            recon.d_out,
            &recon.w1,
            &recon.b1,
            &recon.w2,
            &recon.b2,
            &recon.input,
            &recon.target,
            base_error,
        )?;

        // Compute backward pass
        let backward = Self::compute_backward_pass(
            recon.d_in,
            recon.d_hid,
            recon.d_out,
            &forward,
            &recon.w1,
            &recon.w2,
            &recon.input,
            &recon.target,
            base_error,
        )?;

        // Compute updated weights
        let (w1_new, b1_new, w2_new, b2_new) = Self::apply_gradients(
            &recon.w1,
            &recon.b1,
            &recon.w2,
            &recon.b2,
            &backward,
            &recon.lr,
        );

        // Compute state hashes using Poseidon
        let old_hash = poseidon_state_hash(&recon.w1, &recon.b1, &recon.w2, &recon.b2);
        let new_hash = poseidon_state_hash(&w1_new, &b1_new, &w2_new, &b2_new);

        // Compute total error
        let total_error = Self::compute_total_error(&forward, &backward);

        Ok(Self {
            d_in: recon.d_in,
            d_hid: recon.d_hid,
            d_out: recon.d_out,
            x: recon.input.clone(),
            target: recon.target.clone(),
            w1: recon.w1.clone(),
            b1: recon.b1.clone(),
            w2: recon.w2.clone(),
            b2: recon.b2.clone(),
            h_pre: forward.h_pre,
            h_pre_err: forward.h_pre_err,
            h: forward.h,
            h_err: forward.h_err,
            y: forward.y,
            y_err: forward.y_err,
            loss: forward.loss,
            loss_err: forward.loss_err,
            dy: backward.dy,
            dy_err: backward.dy_err,
            dw2: backward.dw2,
            dw2_err: backward.dw2_err,
            db2: backward.db2,
            db2_err: backward.db2_err,
            dh: backward.dh,
            dh_err: backward.dh_err,
            relu_mask: backward.relu_mask,
            dh_pre: backward.dh_pre,
            dh_pre_err: backward.dh_pre_err,
            dw1: backward.dw1,
            dw1_err: backward.dw1_err,
            db1: backward.db1,
            db1_err: backward.db1_err,
            lr: recon.lr.clone(),
            w1_new,
            b1_new,
            w2_new,
            b2_new,
            freivalds_r1: recon.freivalds_r1.clone(),
            freivalds_r2: recon.freivalds_r2.clone(),
            total_error,
            old_state_hash: old_hash,
            new_state_hash: new_hash,
            step_number: recon.step_number,
        })
    }

    /// Returns the 7 public inputs expected by the circuit.
    pub fn public_inputs(&self) -> Vec<Fr> {
        vec![
            self.old_state_hash.0.clone(),
            self.old_state_hash.1.clone(),
            self.new_state_hash.0.clone(),
            self.new_state_hash.1.clone(),
            self.loss.clone(),
            self.total_error.clone(),
            Fr::from_u64(self.step_number),
        ]
    }

    /// Validates that the witness is internally consistent.
    pub fn validate(&self) -> MPCResult<()> {
        // Check dimensions
        if self.w1.len() != self.d_hid * self.d_in {
            return Err(MPCError::ShapeMismatch {
                expected: vec![self.d_hid * self.d_in],
                got: vec![self.w1.len()],
            });
        }
        if self.b1.len() != self.d_hid {
            return Err(MPCError::ShapeMismatch {
                expected: vec![self.d_hid],
                got: vec![self.b1.len()],
            });
        }
        if self.w2.len() != self.d_out * self.d_hid {
            return Err(MPCError::ShapeMismatch {
                expected: vec![self.d_out * self.d_hid],
                got: vec![self.w2.len()],
            });
        }
        if self.b2.len() != self.d_out {
            return Err(MPCError::ShapeMismatch {
                expected: vec![self.d_out],
                got: vec![self.b2.len()],
            });
        }

        // Check forward pass dimensions
        if self.h_pre.len() != self.d_hid || self.h.len() != self.d_hid {
            return Err(MPCError::ProtocolError("Forward pass dimension mismatch".into()));
        }
        if self.y.len() != self.d_out {
            return Err(MPCError::ProtocolError("Output dimension mismatch".into()));
        }

        Ok(())
    }

    fn compute_forward_pass(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        w1: &[Fr],
        b1: &[Fr],
        w2: &[Fr],
        b2: &[Fr],
        x: &[Fr],
        target: &[Fr],
        base_error: f64,
    ) -> MPCResult<ForwardResult> {
        let base_err = Fr::from_f64(base_error);

        // h_pre = W1 @ x + b1
        let mut h_pre = vec![Fr::ZERO; d_hid];
        let mut h_pre_err = vec![Fr::ZERO; d_hid];

        for i in 0..d_hid {
            let mut sum = Fr::ZERO;
            for j in 0..d_in {
                let w_ij = &w1[i * d_in + j];
                sum = Fr::add(&sum, &Fr::mul(w_ij, &x[j]));
            }
            h_pre[i] = Fr::add(&sum, &b1[i]);
            h_pre_err[i] = Fr::mul(&base_err, &Fr::from_u64(d_in as u64 + 1));
        }

        // h = ReLU(h_pre)
        let mut h = vec![Fr::ZERO; d_hid];
        let mut h_err = vec![Fr::ZERO; d_hid];

        for i in 0..d_hid {
            let val_f64 = h_pre[i].to_f64();
            if val_f64 > 0.0 {
                h[i] = h_pre[i].clone();
            } else {
                h[i] = Fr::ZERO;
            }
            h_err[i] = h_pre_err[i].clone();
        }

        // y = W2 @ h + b2
        let mut y = vec![Fr::ZERO; d_out];
        let mut y_err = vec![Fr::ZERO; d_out];

        for i in 0..d_out {
            let mut sum = Fr::ZERO;
            for j in 0..d_hid {
                let w_ij = &w2[i * d_hid + j];
                sum = Fr::add(&sum, &Fr::mul(w_ij, &h[j]));
            }
            y[i] = Fr::add(&sum, &b2[i]);
            y_err[i] = Fr::mul(&base_err, &Fr::from_u64(d_hid as u64 + 1));
        }

        // Loss = 0.5 * sum((y - target)^2)
        let mut loss = Fr::ZERO;
        for i in 0..d_out {
            let diff = Fr::sub(&y[i], &target[i]);
            let sq = Fr::mul(&diff, &diff);
            loss = Fr::add(&loss, &sq);
        }
        // Scale by 0.5
        let half = Fr::from_f64(0.5);
        loss = Fr::mul(&loss, &half);

        let loss_err = Fr::mul(&base_err, &Fr::from_u64(d_out as u64 * 2));

        Ok(ForwardResult {
            h_pre,
            h_pre_err,
            h,
            h_err,
            y,
            y_err,
            loss,
            loss_err,
        })
    }

    fn compute_backward_pass(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        forward: &ForwardResult,
        w1: &[Fr],
        w2: &[Fr],
        x: &[Fr],
        target: &[Fr],
        base_error: f64,
    ) -> MPCResult<BackwardResult> {
        let base_err = Fr::from_f64(base_error);

        // dy = y - target (gradient of MSE)
        let mut dy = vec![Fr::ZERO; d_out];
        let mut dy_err = vec![Fr::ZERO; d_out];
        for i in 0..d_out {
            dy[i] = Fr::sub(&forward.y[i], &target[i]);
            dy_err[i] = Fr::add(&forward.y_err[i], &base_err);
        }

        // dW2 = outer(dy, h)
        let mut dw2 = vec![Fr::ZERO; d_out * d_hid];
        let mut dw2_err = vec![Fr::ZERO; d_out * d_hid];
        for i in 0..d_out {
            for j in 0..d_hid {
                dw2[i * d_hid + j] = Fr::mul(&dy[i], &forward.h[j]);
                dw2_err[i * d_hid + j] = base_err.clone();
            }
        }

        // db2 = dy
        let db2 = dy.clone();
        let db2_err = dy_err.clone();

        // dh = W2^T @ dy
        let mut dh = vec![Fr::ZERO; d_hid];
        let mut dh_err = vec![Fr::ZERO; d_hid];
        for j in 0..d_hid {
            let mut sum = Fr::ZERO;
            for i in 0..d_out {
                let w_ij = &w2[i * d_hid + j];
                sum = Fr::add(&sum, &Fr::mul(w_ij, &dy[i]));
            }
            dh[j] = sum;
            dh_err[j] = Fr::mul(&base_err, &Fr::from_u64(d_out as u64));
        }

        // ReLU mask and dh_pre
        let mut relu_mask = vec![Fr::ZERO; d_hid];
        let mut dh_pre = vec![Fr::ZERO; d_hid];
        let mut dh_pre_err = vec![Fr::ZERO; d_hid];
        for j in 0..d_hid {
            let h_pre_val = forward.h_pre[j].to_f64();
            if h_pre_val > 0.0 {
                relu_mask[j] = Fr::ONE;
                dh_pre[j] = dh[j].clone();
            } else {
                relu_mask[j] = Fr::ZERO;
                dh_pre[j] = Fr::ZERO;
            }
            dh_pre_err[j] = dh_err[j].clone();
        }

        // dW1 = outer(dh_pre, x)
        let mut dw1 = vec![Fr::ZERO; d_hid * d_in];
        let mut dw1_err = vec![Fr::ZERO; d_hid * d_in];
        for i in 0..d_hid {
            for j in 0..d_in {
                dw1[i * d_in + j] = Fr::mul(&dh_pre[i], &x[j]);
                dw1_err[i * d_in + j] = base_err.clone();
            }
        }

        // db1 = dh_pre
        let db1 = dh_pre.clone();
        let db1_err = dh_pre_err.clone();

        Ok(BackwardResult {
            dy,
            dy_err,
            dw2,
            dw2_err,
            db2,
            db2_err,
            dh,
            dh_err,
            relu_mask,
            dh_pre,
            dh_pre_err,
            dw1,
            dw1_err,
            db1,
            db1_err,
        })
    }

    fn apply_gradients(
        w1: &[Fr],
        b1: &[Fr],
        w2: &[Fr],
        b2: &[Fr],
        backward: &BackwardResult,
        lr: &Fr,
    ) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>) {
        let w1_new: Vec<Fr> = w1
            .iter()
            .zip(&backward.dw1)
            .map(|(w, dw)| Fr::sub(w, &Fr::mul(lr, dw)))
            .collect();

        let b1_new: Vec<Fr> = b1
            .iter()
            .zip(&backward.db1)
            .map(|(b, db)| Fr::sub(b, &Fr::mul(lr, db)))
            .collect();

        let w2_new: Vec<Fr> = w2
            .iter()
            .zip(&backward.dw2)
            .map(|(w, dw)| Fr::sub(w, &Fr::mul(lr, dw)))
            .collect();

        let b2_new: Vec<Fr> = b2
            .iter()
            .zip(&backward.db2)
            .map(|(b, db)| Fr::sub(b, &Fr::mul(lr, db)))
            .collect();

        (w1_new, b1_new, w2_new, b2_new)
    }

    fn compute_total_error(forward: &ForwardResult, backward: &BackwardResult) -> Fr {
        let mut total = Fr::ZERO;

        for e in &forward.h_pre_err {
            total = Fr::add(&total, e);
        }
        for e in &forward.h_err {
            total = Fr::add(&total, e);
        }
        for e in &forward.y_err {
            total = Fr::add(&total, e);
        }
        total = Fr::add(&total, &forward.loss_err);

        for e in &backward.dy_err {
            total = Fr::add(&total, e);
        }
        for e in &backward.dw2_err {
            total = Fr::add(&total, e);
        }
        for e in &backward.db2_err {
            total = Fr::add(&total, e);
        }
        for e in &backward.dh_err {
            total = Fr::add(&total, e);
        }
        for e in &backward.dh_pre_err {
            total = Fr::add(&total, e);
        }
        for e in &backward.dw1_err {
            total = Fr::add(&total, e);
        }
        for e in &backward.db1_err {
            total = Fr::add(&total, e);
        }

        total
    }
}

/// Forward pass computation result.
#[derive(Clone)]
struct ForwardResult {
    h_pre: Vec<Fr>,
    h_pre_err: Vec<Fr>,
    h: Vec<Fr>,
    h_err: Vec<Fr>,
    y: Vec<Fr>,
    y_err: Vec<Fr>,
    loss: Fr,
    loss_err: Fr,
}

/// Backward pass computation result.
#[derive(Clone)]
struct BackwardResult {
    dy: Vec<Fr>,
    dy_err: Vec<Fr>,
    dw2: Vec<Fr>,
    dw2_err: Vec<Fr>,
    db2: Vec<Fr>,
    db2_err: Vec<Fr>,
    dh: Vec<Fr>,
    dh_err: Vec<Fr>,
    relu_mask: Vec<Fr>,
    dh_pre: Vec<Fr>,
    dh_pre_err: Vec<Fr>,
    dw1: Vec<Fr>,
    dw1_err: Vec<Fr>,
    db1: Vec<Fr>,
    db1_err: Vec<Fr>,
}

/// Commitment to a party's share for verification.
#[derive(Debug, Clone)]
pub struct ShareCommitmentData {
    /// Party that owns this share
    pub party: PartyId,
    /// Poseidon commitment to the share data
    pub commitment: Fr,
    /// Blinding factor (private to the party)
    pub blinding: Fr,
    /// Error bound for this share
    pub error_bound: Fr,
}

impl ShareCommitmentData {
    /// Creates a new share commitment.
    pub fn new(party: PartyId, share_data: &[Fr], blinding: Fr, error_bound: Fr) -> Self {
        let commitment = share_commitment(share_data, blinding.clone());
        Self {
            party,
            commitment,
            blinding,
            error_bound,
        }
    }

    /// Verifies the commitment against provided data.
    pub fn verify(&self, share_data: &[Fr]) -> bool {
        let computed = share_commitment(share_data, self.blinding.clone());
        self.commitment.ct_eq(&computed).to_bool()
    }
}

/// Generates Freivalds verification challenges deterministically.
pub fn generate_freivalds_challenges(step_number: u64, d_hid: usize, d_out: usize) -> (Vec<Fr>, Vec<Fr>) {
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    // Layer 1 challenges
    let seed1 = step_number.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut rng1 = ChaCha20Rng::seed_from_u64(seed1);
    let r1: Vec<Fr> = (0..d_hid).map(|_| Fr::random(&mut rng1)).collect();

    // Layer 2 challenges
    let seed2 = step_number.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(2);
    let mut rng2 = ChaCha20Rng::seed_from_u64(seed2);
    let r2: Vec<Fr> = (0..d_out).map(|_| Fr::random(&mut rng2)).collect();

    (r1, r2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_reconstructed_witness() -> ReconstructedWitness {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        ReconstructedWitness {
            d_in,
            d_hid,
            d_out,
            input: vec![Fr::from_f64(0.5), Fr::from_f64(0.5)],
            target: vec![Fr::from_f64(1.0)],
            w1: vec![Fr::from_f64(0.1); d_hid * d_in],
            b1: vec![Fr::from_f64(0.01); d_hid],
            w2: vec![Fr::from_f64(0.1); d_out * d_hid],
            b2: vec![Fr::from_f64(0.01); d_out],
            w1_new: vec![Fr::from_f64(0.1); d_hid * d_in],
            b1_new: vec![Fr::from_f64(0.01); d_hid],
            w2_new: vec![Fr::from_f64(0.1); d_out * d_hid],
            b2_new: vec![Fr::from_f64(0.01); d_out],
            lr: Fr::from_f64(0.01),
            old_state_hash: (Fr::from_u64(1), Fr::from_u64(2)),
            new_state_hash: (Fr::from_u64(3), Fr::from_u64(4)),
            step_number: 0,
            total_error: Fr::from_f64(0.001),
            freivalds_r1: vec![Fr::from_u64(42); 2],
            freivalds_r2: vec![Fr::from_u64(43); 1],
        }
    }

    #[test]
    fn test_circuit_witness_from_reconstructed() {
        let recon = create_test_reconstructed_witness();
        let witness = CircuitWitness::from_reconstructed(&recon, 1e-6).unwrap();

        assert_eq!(witness.d_in, 2);
        assert_eq!(witness.d_hid, 2);
        assert_eq!(witness.d_out, 1);
        assert_eq!(witness.step_number, 0);
    }

    #[test]
    fn test_circuit_witness_public_inputs() {
        let recon = create_test_reconstructed_witness();
        let witness = CircuitWitness::from_reconstructed(&recon, 1e-6).unwrap();

        let pi = witness.public_inputs();
        assert_eq!(pi.len(), 7);
    }

    #[test]
    fn test_circuit_witness_validation() {
        let recon = create_test_reconstructed_witness();
        let witness = CircuitWitness::from_reconstructed(&recon, 1e-6).unwrap();

        assert!(witness.validate().is_ok());
    }

    #[test]
    fn test_share_commitment_data() {
        let party = PartyId::from_index(0);
        let share_data = vec![Fr::from_f64(0.1), Fr::from_f64(0.2)];
        let blinding = Fr::from_u64(42);
        let error_bound = Fr::from_f64(0.001);

        let commitment = ShareCommitmentData::new(party, &share_data, blinding, error_bound);
        assert!(commitment.verify(&share_data));

        // Wrong data should fail
        let wrong_data = vec![Fr::from_f64(0.1), Fr::from_f64(0.3)];
        assert!(!commitment.verify(&wrong_data));
    }

    #[test]
    fn test_freivalds_challenges_deterministic() {
        let (r1_a, r2_a) = generate_freivalds_challenges(0, 4, 2);
        let (r1_b, r2_b) = generate_freivalds_challenges(0, 4, 2);

        assert_eq!(r1_a.len(), 4);
        assert_eq!(r2_a.len(), 2);

        // Same step should produce same challenges
        for (a, b) in r1_a.iter().zip(r1_b.iter()) {
            assert!(a.ct_eq(b).to_bool());
        }
        for (a, b) in r2_a.iter().zip(r2_b.iter()) {
            assert!(a.ct_eq(b).to_bool());
        }

        // Different step should produce different challenges
        let (r1_c, _) = generate_freivalds_challenges(1, 4, 2);
        assert!(!r1_a[0].ct_eq(&r1_c[0]).to_bool());
    }
}
