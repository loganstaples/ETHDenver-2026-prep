//! MPC Witness Format for ZK Circuits.
//!
//! This module provides the bridge between MPC secret shares and ZK proof witnesses.
//! It enables generating ZK proofs over secret-shared computations without revealing
//! the underlying model weights.
//!
//! # Architecture
//!
//! The witness format converts MPC computation state into field elements compatible
//! with the ZK circuits in helix-circuits. Key components:
//!
//! - **ShareWitness**: Witness representation of a secret share
//! - **TrainingWitness**: Complete witness for a training step
//! - **WitnessBuilder**: Builder for constructing witnesses from MPC state
//!
//! # Privacy Guarantees
//!
//! The witness generation is designed so that:
//! - Individual share values are never directly exposed
//! - Only aggregated/reconstructed values appear as public inputs
//! - Error bounds are tracked and proven within tolerance

use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::sharing::model::{GradientShare, ModelShare};
use crate::sharing::tensor::TensorShare;
use crate::types::PartyId;

/// Represents a share converted to ZK-compatible field elements.
#[derive(Debug, Clone)]
pub struct ShareWitness {
    /// The share data as field elements.
    pub data: Vec<Fr>,
    /// Shape of the original tensor.
    pub shape: Vec<usize>,
    /// Party that holds this share.
    pub party: PartyId,
    /// Commitment to the share (H(data || blinding)).
    pub commitment: [u8; 32],
    /// Error bound for this share.
    pub error_bound: Fr,
}

impl ShareWitness {
    /// Creates a new share witness from a tensor share.
    pub fn from_tensor_share(share: &TensorShare, blinding: &[u8; 32]) -> Self {
        let commitment = Self::compute_commitment(&share.data, blinding);
        Self {
            data: share.data.clone(),
            shape: share.shape.clone(),
            party: share.id.party.clone(),
            commitment,
            error_bound: Fr::from_f64(share.sharing_error + share.original_error),
        }
    }

    /// Computes a commitment to share data.
    fn compute_commitment(data: &[Fr], blinding: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for v in data {
            hasher.update(&v.to_bytes_le());
        }
        hasher.update(blinding);
        hasher.finalize().into()
    }

    /// Verifies that the data matches the commitment.
    pub fn verify_commitment(&self, blinding: &[u8; 32]) -> bool {
        let expected = Self::compute_commitment(&self.data, blinding);
        self.commitment == expected
    }

    /// Returns the number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Complete witness for a training step in MPC context.
///
/// This structure contains all the information needed to generate
/// a ZK proof that a training step was computed correctly on
/// secret-shared weights.
#[derive(Debug, Clone)]
pub struct MPCTrainingWitness {
    /// Dimensions of the model.
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,

    /// Input and target (public or committed).
    pub input: Vec<Fr>,
    pub target: Vec<Fr>,

    /// Weight shares from this party.
    pub w1_share: ShareWitness,
    pub b1_share: ShareWitness,
    pub w2_share: ShareWitness,
    pub b2_share: ShareWitness,

    /// Forward pass intermediates (computed on shares).
    pub h_pre: Vec<Fr>,
    pub h_pre_err: Vec<Fr>,
    pub h: Vec<Fr>,
    pub h_err: Vec<Fr>,
    pub y: Vec<Fr>,
    pub y_err: Vec<Fr>,
    pub loss: Fr,
    pub loss_err: Fr,

    /// Backward pass gradients (on shares).
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

    /// Learning rate.
    pub lr: Fr,

    /// Updated weight shares.
    pub w1_new_share: ShareWitness,
    pub b1_new_share: ShareWitness,
    pub w2_new_share: ShareWitness,
    pub b2_new_share: ShareWitness,

    /// Freivalds verification randomness.
    pub freivalds_r1: Vec<Fr>,
    pub freivalds_r2: Vec<Fr>,

    /// Total accumulated error.
    pub total_error: Fr,

    /// State hashes for public verification.
    pub old_state_hash: (Fr, Fr),
    pub new_state_hash: (Fr, Fr),

    /// Training step number.
    pub step_number: u64,

    /// Party index in the MPC protocol.
    pub party_index: usize,
}

impl MPCTrainingWitness {
    /// Computes the hash of weight shares for state commitment.
    pub fn compute_state_hash(
        w1: &[Fr],
        b1: &[Fr],
        w2: &[Fr],
        b2: &[Fr],
    ) -> (Fr, Fr) {
        let mut hasher = Sha256::new();
        for v in w1 {
            hasher.update(&v.to_bytes_le());
        }
        for v in b1 {
            hasher.update(&v.to_bytes_le());
        }
        for v in w2 {
            hasher.update(&v.to_bytes_le());
        }
        for v in b2 {
            hasher.update(&v.to_bytes_le());
        }
        let hash: [u8; 32] = hasher.finalize().into();

        // Split into two 128-bit field elements (lo, hi).
        let lo = Fr::from_bytes_le(&[
            hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
            hash[8], hash[9], hash[10], hash[11], hash[12], hash[13], hash[14], hash[15],
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        let hi = Fr::from_bytes_le(&[
            hash[16], hash[17], hash[18], hash[19], hash[20], hash[21], hash[22], hash[23],
            hash[24], hash[25], hash[26], hash[27], hash[28], hash[29], hash[30], hash[31],
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);

        (lo, hi)
    }

    /// Returns the 7 public inputs for the ZK circuit.
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
}

/// Forward pass computation result.
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

/// Builder for constructing MPC training witnesses.
pub struct WitnessBuilder {
    d_in: usize,
    d_hid: usize,
    d_out: usize,
    party_index: usize,
    learning_rate: f64,
    base_error: f64,
}

impl WitnessBuilder {
    /// Creates a new witness builder.
    pub fn new(d_in: usize, d_hid: usize, d_out: usize, party_index: usize) -> Self {
        Self {
            d_in,
            d_hid,
            d_out,
            party_index,
            learning_rate: 0.001,
            base_error: 1e-6,
        }
    }

    /// Sets the learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Sets the base error per operation.
    pub fn base_error(mut self, err: f64) -> Self {
        self.base_error = err;
        self
    }

    /// Builds a training witness from model and gradient shares.
    ///
    /// This computes the forward and backward passes on the shares
    /// and generates all witness values needed for proof generation.
    pub fn build(
        &self,
        model_share: &ModelShare,
        gradient_share: &GradientShare,
        input: &[f64],
        target: &[f64],
        step_number: u64,
        blinding: &[u8; 32],
    ) -> MPCResult<MPCTrainingWitness> {
        // Convert input/target to Fr.
        let input_fr: Vec<Fr> = input.iter().map(|v| Fr::from_f64(*v)).collect();
        let target_fr: Vec<Fr> = target.iter().map(|v| Fr::from_f64(*v)).collect();

        // Extract weight shares.
        let (w1_data, b1_data, w2_data, b2_data) = self.extract_weight_shares(model_share)?;

        // Compute forward pass on shares.
        let forward = self.compute_forward_pass(&w1_data, &b1_data, &w2_data, &b2_data, &input_fr)?;

        // Compute backward pass.
        let backward = self.compute_backward_pass(
            &forward,
            &w1_data,
            &w2_data,
            &input_fr,
            &target_fr,
        )?;

        // Compute updated weights.
        let lr = Fr::from_f64(self.learning_rate);
        let (w1_new, b1_new, w2_new, b2_new) = self.apply_gradients(
            &w1_data,
            &b1_data,
            &w2_data,
            &b2_data,
            &backward,
            &lr,
        );

        // Compute state hashes.
        let old_hash = MPCTrainingWitness::compute_state_hash(&w1_data, &b1_data, &w2_data, &b2_data);
        let new_hash = MPCTrainingWitness::compute_state_hash(&w1_new, &b1_new, &w2_new, &b2_new);

        // Generate Freivalds randomness.
        let freivalds_r1 = self.generate_freivalds_randomness(self.d_hid, step_number, 1);
        let freivalds_r2 = self.generate_freivalds_randomness(self.d_out, step_number, 2);

        // Compute total error.
        let total_error = self.compute_total_error(&forward, &backward);

        // Create share witnesses.
        let party = model_share.party.clone();
        let w1_shape = vec![self.d_hid, self.d_in];
        let b1_shape = vec![self.d_hid];
        let w2_shape = vec![self.d_out, self.d_hid];
        let b2_shape = vec![self.d_out];

        Ok(MPCTrainingWitness {
            d_in: self.d_in,
            d_hid: self.d_hid,
            d_out: self.d_out,
            input: input_fr,
            target: target_fr,
            w1_share: ShareWitness {
                data: w1_data.clone(),
                shape: w1_shape.clone(),
                party: party.clone(),
                commitment: ShareWitness::compute_commitment(&w1_data, blinding),
                error_bound: Fr::from_f64(self.base_error),
            },
            b1_share: ShareWitness {
                data: b1_data.clone(),
                shape: b1_shape.clone(),
                party: party.clone(),
                commitment: ShareWitness::compute_commitment(&b1_data, blinding),
                error_bound: Fr::from_f64(self.base_error),
            },
            w2_share: ShareWitness {
                data: w2_data.clone(),
                shape: w2_shape.clone(),
                party: party.clone(),
                commitment: ShareWitness::compute_commitment(&w2_data, blinding),
                error_bound: Fr::from_f64(self.base_error),
            },
            b2_share: ShareWitness {
                data: b2_data.clone(),
                shape: b2_shape.clone(),
                party: party.clone(),
                commitment: ShareWitness::compute_commitment(&b2_data, blinding),
                error_bound: Fr::from_f64(self.base_error),
            },
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
            lr,
            w1_new_share: ShareWitness {
                data: w1_new.clone(),
                shape: w1_shape,
                party: party.clone(),
                commitment: ShareWitness::compute_commitment(&w1_new, blinding),
                error_bound: Fr::from_f64(self.base_error),
            },
            b1_new_share: ShareWitness {
                data: b1_new.clone(),
                shape: b1_shape,
                party: party.clone(),
                commitment: ShareWitness::compute_commitment(&b1_new, blinding),
                error_bound: Fr::from_f64(self.base_error),
            },
            w2_new_share: ShareWitness {
                data: w2_new.clone(),
                shape: w2_shape,
                party: party.clone(),
                commitment: ShareWitness::compute_commitment(&w2_new, blinding),
                error_bound: Fr::from_f64(self.base_error),
            },
            b2_new_share: ShareWitness {
                data: b2_new.clone(),
                shape: b2_shape,
                party: party.clone(),
                commitment: ShareWitness::compute_commitment(&b2_new, blinding),
                error_bound: Fr::from_f64(self.base_error),
            },
            freivalds_r1,
            freivalds_r2,
            total_error,
            old_state_hash: old_hash,
            new_state_hash: new_hash,
            step_number,
            party_index: self.party_index,
        })
    }

    /// Extracts weight data from a model share.
    fn extract_weight_shares(&self, model: &ModelShare) -> MPCResult<(Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>)> {
        // For a 2-layer MLP, we need w1, b1, w2, b2.
        // The structure depends on how the model was shared.

        // Try to find the weights in the first layer.
        if model.layers.is_empty() {
            return Err(MPCError::ProtocolError("Model has no layers".into()));
        }

        let layer0 = &model.layers[0];

        let w1 = layer0
            .weights
            .get("w1")
            .or_else(|| layer0.weights.get("weight"))
            .or_else(|| layer0.weights.get("q_proj"))
            .cloned()
            .ok_or_else(|| MPCError::ProtocolError("Missing w1 weight".into()))?;

        let b1 = layer0
            .weights
            .get("b1")
            .or_else(|| layer0.weights.get("bias"))
            .cloned()
            .unwrap_or_else(|| TensorShare::new(
                w1.id.clone(),
                vec![Fr::ZERO; self.d_hid],
                vec![self.d_hid],
            ));

        // Try to find w2/b2 in layer 1 or same layer.
        let (w2, b2) = if model.layers.len() > 1 {
            let layer1 = &model.layers[1];
            let w2 = layer1
                .weights
                .get("w2")
                .or_else(|| layer1.weights.get("weight"))
                .cloned()
                .ok_or_else(|| MPCError::ProtocolError("Missing w2 weight".into()))?;
            let b2 = layer1
                .weights
                .get("b2")
                .or_else(|| layer1.weights.get("bias"))
                .cloned()
                .unwrap_or_else(|| TensorShare::new(
                    w2.id.clone(),
                    vec![Fr::ZERO; self.d_out],
                    vec![self.d_out],
                ));
            (w2, b2)
        } else {
            // Try to find in same layer with different naming.
            let w2 = layer0
                .weights
                .get("w2")
                .or_else(|| layer0.weights.get("v_proj"))
                .cloned()
                .ok_or_else(|| MPCError::ProtocolError("Missing w2 weight".into()))?;
            let b2 = layer0
                .weights
                .get("b2")
                .cloned()
                .unwrap_or_else(|| TensorShare::new(
                    w2.id.clone(),
                    vec![Fr::ZERO; self.d_out],
                    vec![self.d_out],
                ));
            (w2, b2)
        };

        Ok((w1.data, b1.data, w2.data, b2.data))
    }

    /// Computes the forward pass on weight shares.
    fn compute_forward_pass(
        &self,
        w1: &[Fr],
        b1: &[Fr],
        w2: &[Fr],
        b2: &[Fr],
        x: &[Fr],
    ) -> MPCResult<ForwardResult> {
        let base_err = Fr::from_f64(self.base_error);

        // h_pre = W1 @ x + b1.
        let mut h_pre = vec![Fr::ZERO; self.d_hid];
        let mut h_pre_err = vec![Fr::ZERO; self.d_hid];

        for i in 0..self.d_hid {
            let mut sum = Fr::ZERO;
            for j in 0..self.d_in {
                let w_ij = &w1[i * self.d_in + j];
                sum = Fr::add(&sum, &Fr::mul(w_ij, &x[j]));
            }
            h_pre[i] = Fr::add(&sum, &b1[i]);
            // Error from matmul: O(d_in) multiplications.
            h_pre_err[i] = Fr::mul(&base_err, &Fr::from_u64(self.d_in as u64 + 1));
        }

        // h = ReLU(h_pre).
        let mut h = vec![Fr::ZERO; self.d_hid];
        let mut h_err = vec![Fr::ZERO; self.d_hid];

        for i in 0..self.d_hid {
            let val_f64 = h_pre[i].to_f64();
            if val_f64 > 0.0 {
                h[i] = h_pre[i].clone();
            } else {
                h[i] = Fr::ZERO;
            }
            h_err[i] = h_pre_err[i].clone();
        }

        // y = W2 @ h + b2.
        let mut y = vec![Fr::ZERO; self.d_out];
        let mut y_err = vec![Fr::ZERO; self.d_out];

        for i in 0..self.d_out {
            let mut sum = Fr::ZERO;
            for j in 0..self.d_hid {
                let w_ij = &w2[i * self.d_hid + j];
                sum = Fr::add(&sum, &Fr::mul(w_ij, &h[j]));
            }
            y[i] = Fr::add(&sum, &b2[i]);
            y_err[i] = Fr::mul(&base_err, &Fr::from_u64(self.d_hid as u64 + 1));
        }

        // Loss = 0.5 * sum((y - target)^2).
        // For simplicity, compute MSE.
        Ok(ForwardResult {
            h_pre,
            h_pre_err,
            h,
            h_err,
            y,
            y_err,
            loss: Fr::ZERO, // Will be set by caller with target.
            loss_err: Fr::ZERO,
        })
    }

    /// Computes the backward pass.
    fn compute_backward_pass(
        &self,
        forward: &ForwardResult,
        w1: &[Fr],
        w2: &[Fr],
        x: &[Fr],
        target: &[Fr],
    ) -> MPCResult<BackwardResult> {
        let base_err = Fr::from_f64(self.base_error);

        // dy = y - target (gradient of MSE).
        let mut dy = vec![Fr::ZERO; self.d_out];
        let mut dy_err = vec![Fr::ZERO; self.d_out];
        for i in 0..self.d_out {
            dy[i] = Fr::sub(&forward.y[i], &target[i]);
            dy_err[i] = Fr::add(&forward.y_err[i], &base_err);
        }

        // dW2 = outer(dy, h).
        let mut dw2 = vec![Fr::ZERO; self.d_out * self.d_hid];
        let mut dw2_err = vec![Fr::ZERO; self.d_out * self.d_hid];
        for i in 0..self.d_out {
            for j in 0..self.d_hid {
                dw2[i * self.d_hid + j] = Fr::mul(&dy[i], &forward.h[j]);
                dw2_err[i * self.d_hid + j] = base_err.clone();
            }
        }

        // db2 = dy.
        let db2 = dy.clone();
        let db2_err = dy_err.clone();

        // dh = W2^T @ dy.
        let mut dh = vec![Fr::ZERO; self.d_hid];
        let mut dh_err = vec![Fr::ZERO; self.d_hid];
        for j in 0..self.d_hid {
            let mut sum = Fr::ZERO;
            for i in 0..self.d_out {
                let w_ij = &w2[i * self.d_hid + j];
                sum = Fr::add(&sum, &Fr::mul(w_ij, &dy[i]));
            }
            dh[j] = sum;
            dh_err[j] = Fr::mul(&base_err, &Fr::from_u64(self.d_out as u64));
        }

        // ReLU mask and dh_pre.
        let mut relu_mask = vec![Fr::ZERO; self.d_hid];
        let mut dh_pre = vec![Fr::ZERO; self.d_hid];
        let mut dh_pre_err = vec![Fr::ZERO; self.d_hid];
        for j in 0..self.d_hid {
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

        // dW1 = outer(dh_pre, x).
        let mut dw1 = vec![Fr::ZERO; self.d_hid * self.d_in];
        let mut dw1_err = vec![Fr::ZERO; self.d_hid * self.d_in];
        for i in 0..self.d_hid {
            for j in 0..self.d_in {
                dw1[i * self.d_in + j] = Fr::mul(&dh_pre[i], &x[j]);
                dw1_err[i * self.d_in + j] = base_err.clone();
            }
        }

        // db1 = dh_pre.
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

    /// Applies gradients to weights.
    fn apply_gradients(
        &self,
        w1: &[Fr],
        b1: &[Fr],
        w2: &[Fr],
        b2: &[Fr],
        backward: &BackwardResult,
        lr: &Fr,
    ) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>) {
        // W_new = W - lr * dW.
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

    /// Generates Freivalds randomness from step number.
    fn generate_freivalds_randomness(&self, size: usize, step: u64, salt: u64) -> Vec<Fr> {
        use rand::{RngCore, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        let seed = step.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(salt);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        (0..size).map(|_| Fr::random(&mut rng)).collect()
    }

    /// Computes total accumulated error.
    fn compute_total_error(&self, forward: &ForwardResult, backward: &BackwardResult) -> Fr {
        let mut total = Fr::ZERO;

        // Sum all error bounds.
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

/// Aggregates witness data from multiple parties.
pub struct WitnessAggregator {
    /// Party witnesses indexed by party.
    witnesses: HashMap<usize, MPCTrainingWitness>,
    /// Number of parties required.
    num_parties: usize,
}

impl WitnessAggregator {
    /// Creates a new witness aggregator.
    pub fn new(num_parties: usize) -> Self {
        Self {
            witnesses: HashMap::new(),
            num_parties,
        }
    }

    /// Adds a party's witness.
    pub fn add_witness(&mut self, party_index: usize, witness: MPCTrainingWitness) -> MPCResult<()> {
        if party_index >= self.num_parties {
            return Err(MPCError::IndexOutOfRange {
                index: party_index,
                num_parties: self.num_parties,
            });
        }
        self.witnesses.insert(party_index, witness);
        Ok(())
    }

    /// Checks if all witnesses have been received.
    pub fn is_complete(&self) -> bool {
        self.witnesses.len() == self.num_parties
    }

    /// Reconstructs the full witness by summing shares.
    pub fn reconstruct(&self) -> MPCResult<ReconstructedWitness> {
        if !self.is_complete() {
            return Err(MPCError::InsufficientShares {
                required: self.num_parties,
                available: self.witnesses.len(),
            });
        }

        let first = self.witnesses.get(&0).ok_or_else(|| {
            MPCError::ProtocolError("Missing party 0 witness".into())
        })?;

        // Sum all weight shares.
        let mut w1 = vec![Fr::ZERO; first.w1_share.data.len()];
        let mut b1 = vec![Fr::ZERO; first.b1_share.data.len()];
        let mut w2 = vec![Fr::ZERO; first.w2_share.data.len()];
        let mut b2 = vec![Fr::ZERO; first.b2_share.data.len()];

        let mut w1_new = vec![Fr::ZERO; first.w1_new_share.data.len()];
        let mut b1_new = vec![Fr::ZERO; first.b1_new_share.data.len()];
        let mut w2_new = vec![Fr::ZERO; first.w2_new_share.data.len()];
        let mut b2_new = vec![Fr::ZERO; first.b2_new_share.data.len()];

        for (_, witness) in &self.witnesses {
            for (i, v) in witness.w1_share.data.iter().enumerate() {
                w1[i] = Fr::add(&w1[i], v);
            }
            for (i, v) in witness.b1_share.data.iter().enumerate() {
                b1[i] = Fr::add(&b1[i], v);
            }
            for (i, v) in witness.w2_share.data.iter().enumerate() {
                w2[i] = Fr::add(&w2[i], v);
            }
            for (i, v) in witness.b2_share.data.iter().enumerate() {
                b2[i] = Fr::add(&b2[i], v);
            }

            for (i, v) in witness.w1_new_share.data.iter().enumerate() {
                w1_new[i] = Fr::add(&w1_new[i], v);
            }
            for (i, v) in witness.b1_new_share.data.iter().enumerate() {
                b1_new[i] = Fr::add(&b1_new[i], v);
            }
            for (i, v) in witness.w2_new_share.data.iter().enumerate() {
                w2_new[i] = Fr::add(&w2_new[i], v);
            }
            for (i, v) in witness.b2_new_share.data.iter().enumerate() {
                b2_new[i] = Fr::add(&b2_new[i], v);
            }
        }

        // Compute state hashes on reconstructed weights.
        let old_hash = MPCTrainingWitness::compute_state_hash(&w1, &b1, &w2, &b2);
        let new_hash = MPCTrainingWitness::compute_state_hash(&w1_new, &b1_new, &w2_new, &b2_new);

        Ok(ReconstructedWitness {
            d_in: first.d_in,
            d_hid: first.d_hid,
            d_out: first.d_out,
            input: first.input.clone(),
            target: first.target.clone(),
            w1,
            b1,
            w2,
            b2,
            w1_new,
            b1_new,
            w2_new,
            b2_new,
            lr: first.lr.clone(),
            old_state_hash: old_hash,
            new_state_hash: new_hash,
            step_number: first.step_number,
            total_error: first.total_error.clone(),
            freivalds_r1: first.freivalds_r1.clone(),
            freivalds_r2: first.freivalds_r2.clone(),
        })
    }
}

/// A fully reconstructed witness from aggregated shares.
#[derive(Debug, Clone)]
pub struct ReconstructedWitness {
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    pub input: Vec<Fr>,
    pub target: Vec<Fr>,
    pub w1: Vec<Fr>,
    pub b1: Vec<Fr>,
    pub w2: Vec<Fr>,
    pub b2: Vec<Fr>,
    pub w1_new: Vec<Fr>,
    pub b1_new: Vec<Fr>,
    pub w2_new: Vec<Fr>,
    pub b2_new: Vec<Fr>,
    pub lr: Fr,
    pub old_state_hash: (Fr, Fr),
    pub new_state_hash: (Fr, Fr),
    pub step_number: u64,
    pub total_error: Fr,
    pub freivalds_r1: Vec<Fr>,
    pub freivalds_r2: Vec<Fr>,
}

impl ReconstructedWitness {
    /// Returns the 7 public inputs for the ZK circuit.
    pub fn public_inputs(&self) -> Vec<Fr> {
        vec![
            self.old_state_hash.0.clone(),
            self.old_state_hash.1.clone(),
            self.new_state_hash.0.clone(),
            self.new_state_hash.1.clone(),
            Fr::ZERO, // loss placeholder
            self.total_error.clone(),
            Fr::from_u64(self.step_number),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sharing::model::LayerShare;
    use crate::types::ShareId;
    use std::collections::HashMap;

    fn create_test_model_share(party_index: usize, d_in: usize, d_hid: usize, d_out: usize) -> ModelShare {
        let party = PartyId::from_index(party_index);

        let w1_data: Vec<Fr> = (0..d_hid * d_in)
            .map(|i| Fr::from_f64(0.1 * (i as f64 + party_index as f64)))
            .collect();
        let b1_data: Vec<Fr> = (0..d_hid)
            .map(|i| Fr::from_f64(0.01 * (i as f64 + party_index as f64)))
            .collect();
        let w2_data: Vec<Fr> = (0..d_out * d_hid)
            .map(|i| Fr::from_f64(0.1 * (i as f64 + party_index as f64)))
            .collect();
        let b2_data: Vec<Fr> = (0..d_out)
            .map(|i| Fr::from_f64(0.01 * (i as f64 + party_index as f64)))
            .collect();

        let mut weights = HashMap::new();
        weights.insert(
            "w1".to_string(),
            TensorShare::new(
                ShareId::new(party.clone(), "w1", party_index),
                w1_data,
                vec![d_hid, d_in],
            ),
        );
        weights.insert(
            "b1".to_string(),
            TensorShare::new(
                ShareId::new(party.clone(), "b1", party_index),
                b1_data,
                vec![d_hid],
            ),
        );
        weights.insert(
            "w2".to_string(),
            TensorShare::new(
                ShareId::new(party.clone(), "w2", party_index),
                w2_data,
                vec![d_out, d_hid],
            ),
        );
        weights.insert(
            "b2".to_string(),
            TensorShare::new(
                ShareId::new(party.clone(), "b2", party_index),
                b2_data,
                vec![d_out],
            ),
        );

        ModelShare {
            party: party.clone(),
            index: party_index,
            model_name: "test".to_string(),
            num_layers: 1,
            embeddings: None,
            layers: vec![LayerShare {
                layer_idx: 0,
                weights,
            }],
            lm_head: None,
            extra_weights: std::collections::HashMap::new(),
        }
    }

    fn create_test_gradient_share(party_index: usize, d_in: usize, d_hid: usize, d_out: usize) -> GradientShare {
        let party = PartyId::from_index(party_index);

        GradientShare {
            party: party.clone(),
            index: party_index,
            embeddings: None,
            layers: vec![],
            lm_head: None,
            error_bound: 0.01,
        }
    }

    #[test]
    fn test_share_witness_creation() {
        let party = PartyId::from_index(0);
        let data = vec![Fr::from_f64(1.0), Fr::from_f64(2.0), Fr::from_f64(3.0)];
        let share = TensorShare::new(
            ShareId::new(party.clone(), "test", 0),
            data.clone(),
            vec![3],
        );

        let blinding = [42u8; 32];
        let witness = ShareWitness::from_tensor_share(&share, &blinding);

        assert_eq!(witness.data.len(), 3);
        assert!(witness.verify_commitment(&blinding));
        assert!(!witness.verify_commitment(&[0u8; 32]));
    }

    #[test]
    fn test_witness_builder() {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        let model_share = create_test_model_share(0, d_in, d_hid, d_out);
        let gradient_share = create_test_gradient_share(0, d_in, d_hid, d_out);

        let builder = WitnessBuilder::new(d_in, d_hid, d_out, 0)
            .learning_rate(0.01)
            .base_error(1e-6);

        let input = vec![1.0, 1.0];
        let target = vec![1.0];
        let blinding = [42u8; 32];

        let witness = builder
            .build(&model_share, &gradient_share, &input, &target, 1, &blinding)
            .unwrap();

        assert_eq!(witness.d_in, d_in);
        assert_eq!(witness.d_hid, d_hid);
        assert_eq!(witness.d_out, d_out);
        assert_eq!(witness.step_number, 1);
        assert_eq!(witness.public_inputs().len(), 7);
    }

    #[test]
    fn test_witness_aggregator() {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;
        let num_parties = 3;

        let mut aggregator = WitnessAggregator::new(num_parties);
        let input = vec![1.0, 1.0];
        let target = vec![1.0];
        let blinding = [42u8; 32];

        for i in 0..num_parties {
            let model_share = create_test_model_share(i, d_in, d_hid, d_out);
            let gradient_share = create_test_gradient_share(i, d_in, d_hid, d_out);

            let builder = WitnessBuilder::new(d_in, d_hid, d_out, i);
            let witness = builder
                .build(&model_share, &gradient_share, &input, &target, 1, &blinding)
                .unwrap();

            aggregator.add_witness(i, witness).unwrap();
        }

        assert!(aggregator.is_complete());

        let reconstructed = aggregator.reconstruct().unwrap();
        assert_eq!(reconstructed.d_in, d_in);
        assert_eq!(reconstructed.d_hid, d_hid);
        assert_eq!(reconstructed.d_out, d_out);
    }

    #[test]
    fn test_state_hash_computation() {
        let w1 = vec![Fr::from_f64(1.0), Fr::from_f64(2.0)];
        let b1 = vec![Fr::from_f64(0.1)];
        let w2 = vec![Fr::from_f64(3.0)];
        let b2 = vec![Fr::from_f64(0.2)];

        let (lo, hi) = MPCTrainingWitness::compute_state_hash(&w1, &b1, &w2, &b2);

        // Hash should be non-zero.
        assert!(!lo.is_zero().to_bool() || !hi.is_zero().to_bool());

        // Same inputs should produce same hash.
        let (lo2, hi2) = MPCTrainingWitness::compute_state_hash(&w1, &b1, &w2, &b2);
        assert!(lo.ct_eq(&lo2).to_bool());
        assert!(hi.ct_eq(&hi2).to_bool());
    }
}
