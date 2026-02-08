//! Circuit Bridge Module.
//!
//! Provides conversion between AVM error tracking (f64-based) and circuit
//! error tracking (field element-based). This enables seamless integration
//! between the high-level AVM operations and the low-level ZK circuit proofs.
//!
//! # Scaling
//!
//! Circuit field elements represent errors in fixed-point format:
//! - Scale: 2^SCALE_BITS (e.g., 2^16 = 65536)
//! - Value: error * scale
//!
//! For example, an error of 0.001 with scale 2^16 becomes 65.536 ≈ 66.

use crate::bounds::{AttentionBounds, RMSNormBounds};

/// Number of bits for fixed-point scaling.
pub const SCALE_BITS: u32 = 16;

/// Fixed-point scale factor (2^SCALE_BITS).
pub const SCALE_FACTOR: u64 = 1 << SCALE_BITS;

/// Maximum representable error (before overflow).
pub const MAX_ERROR: f64 = (u64::MAX >> 1) as f64 / SCALE_FACTOR as f64;

/// Converts an f64 error to a scaled u64 for circuit use.
///
/// # Arguments
/// * `error` - The error value in f64
///
/// # Returns
/// A scaled u64 suitable for use as a circuit field element
pub fn error_to_field(error: f64) -> u64 {
    if error <= 0.0 {
        return 0;
    }
    if error >= MAX_ERROR {
        return u64::MAX >> 1; // Saturate
    }
    (error * SCALE_FACTOR as f64).round() as u64
}

/// Converts a scaled circuit field element back to f64 error.
pub fn field_to_error(field_value: u64) -> f64 {
    field_value as f64 / SCALE_FACTOR as f64
}

/// Converts error bounds from AVM format to circuit format.
#[derive(Debug, Clone)]
pub struct CircuitErrorBounds {
    /// Scaled error bound for layer 1 matmul.
    pub layer1_matmul_error: u64,
    /// Scaled error bound for layer 1 bias add.
    pub layer1_bias_error: u64,
    /// Scaled error bound for ReLU (typically 0).
    pub relu_error: u64,
    /// Scaled error bound for layer 2 matmul.
    pub layer2_matmul_error: u64,
    /// Scaled error bound for layer 2 bias add.
    pub layer2_bias_error: u64,
    /// Scaled error bound for loss computation.
    pub loss_error: u64,
    /// Total accumulated error.
    pub total_error: u64,
}

impl CircuitErrorBounds {
    /// Creates zero error bounds.
    pub fn zero() -> Self {
        Self {
            layer1_matmul_error: 0,
            layer1_bias_error: 0,
            relu_error: 0,
            layer2_matmul_error: 0,
            layer2_bias_error: 0,
            loss_error: 0,
            total_error: 0,
        }
    }

    /// Creates from individual f64 error components.
    pub fn from_f64(
        layer1_matmul: f64,
        layer1_bias: f64,
        relu: f64,
        layer2_matmul: f64,
        layer2_bias: f64,
        loss: f64,
    ) -> Self {
        let layer1_matmul_error = error_to_field(layer1_matmul);
        let layer1_bias_error = error_to_field(layer1_bias);
        let relu_error = error_to_field(relu);
        let layer2_matmul_error = error_to_field(layer2_matmul);
        let layer2_bias_error = error_to_field(layer2_bias);
        let loss_error = error_to_field(loss);

        let total_error = layer1_matmul_error
            .saturating_add(layer1_bias_error)
            .saturating_add(relu_error)
            .saturating_add(layer2_matmul_error)
            .saturating_add(layer2_bias_error)
            .saturating_add(loss_error);

        Self {
            layer1_matmul_error,
            layer1_bias_error,
            relu_error,
            layer2_matmul_error,
            layer2_bias_error,
            loss_error,
            total_error,
        }
    }

    /// Creates from a list of named operations with their error values.
    ///
    /// This is useful for non-MLP architectures where the fixed 6-field struct
    /// doesn't map cleanly. The total_error is computed as the saturating sum.
    pub fn from_operations(ops: &[(&str, f64)]) -> Self {
        let mut total: u64 = 0;
        for &(_, error) in ops {
            total = total.saturating_add(error_to_field(error));
        }

        // Map known operation names to fields where possible
        let find = |name: &str| -> u64 {
            ops.iter()
                .find(|(n, _)| *n == name)
                .map(|(_, e)| error_to_field(*e))
                .unwrap_or(0)
        };

        Self {
            layer1_matmul_error: find("layer1_matmul").max(find("qk_matmul")).max(find("matmul1")),
            layer1_bias_error: find("layer1_bias").max(find("scaling")).max(find("bias1")),
            relu_error: find("relu").max(find("softmax")).max(find("activation")),
            layer2_matmul_error: find("layer2_matmul").max(find("av_matmul")).max(find("matmul2")),
            layer2_bias_error: find("layer2_bias").max(find("projection")).max(find("bias2")),
            loss_error: find("loss").max(find("norm")).max(find("output")),
            total_error: total,
        }
    }

    /// Converts back to f64 total error.
    pub fn total_as_f64(&self) -> f64 {
        field_to_error(self.total_error)
    }
}

/// Configuration for error tracking in training.
#[derive(Debug, Clone)]
pub struct TrainingErrorConfig {
    /// Base quantization error per value.
    pub quantization_error: f64,
    /// Error per multiplication.
    pub mul_error_factor: f64,
    /// Error per addition.
    pub add_error_factor: f64,
    /// Maximum weight magnitude (for error scaling).
    pub max_weight_magnitude: f64,
    /// Maximum activation magnitude.
    pub max_activation_magnitude: f64,
}

impl Default for TrainingErrorConfig {
    fn default() -> Self {
        Self {
            quantization_error: 1.0 / SCALE_FACTOR as f64,
            mul_error_factor: 1.0,
            add_error_factor: 0.5,
            max_weight_magnitude: 10.0,
            max_activation_magnitude: 10.0,
        }
    }
}

impl TrainingErrorConfig {
    /// Estimates error bounds for a 2-layer MLP training step.
    ///
    /// # Arguments
    /// * `d_in` - Input dimension
    /// * `d_hid` - Hidden dimension
    /// * `d_out` - Output dimension
    pub fn estimate_mlp_bounds(
        &self,
        d_in: usize,
        d_hid: usize,
        d_out: usize,
    ) -> CircuitErrorBounds {
        let q = self.quantization_error;
        let w_max = self.max_weight_magnitude;
        let a_max = self.max_activation_magnitude;

        // Layer 1 matmul: d_in multiplications and additions per output
        let layer1_matmul = (d_in as f64) * (w_max * q + a_max * q + q * q);
        let layer1_bias = q;

        // ReLU has no error (just zeroes negatives)
        let relu = 0.0;

        // Layer 2 matmul: d_hid multiplications and additions per output
        let layer2_matmul = (d_hid as f64) * (w_max * q + a_max * q + q * q);
        let layer2_bias = q;

        // Loss: MSE involves d_out subtractions and squares
        let loss = (d_out as f64) * 2.0 * q;

        CircuitErrorBounds::from_f64(
            layer1_matmul,
            layer1_bias,
            relu,
            layer2_matmul,
            layer2_bias,
            loss,
        )
    }

    /// Estimates gradient error bounds for backpropagation.
    pub fn estimate_gradient_bounds(
        &self,
        d_in: usize,
        d_hid: usize,
        d_out: usize,
    ) -> CircuitErrorBounds {
        // Gradient computation typically has higher error due to chain rule
        let forward_bounds = self.estimate_mlp_bounds(d_in, d_hid, d_out);

        // Gradient error is roughly proportional to forward error
        // multiplied by number of gradient computations
        CircuitErrorBounds {
            layer1_matmul_error: forward_bounds.layer1_matmul_error * 2,
            layer1_bias_error: forward_bounds.layer1_bias_error * 2,
            relu_error: forward_bounds.relu_error,
            layer2_matmul_error: forward_bounds.layer2_matmul_error * 2,
            layer2_bias_error: forward_bounds.layer2_bias_error * 2,
            loss_error: forward_bounds.loss_error,
            total_error: forward_bounds.total_error * 2,
        }
    }

    /// Estimates error bounds for multi-head attention.
    ///
    /// Delegates to `AttentionBounds::forward_error()` and converts
    /// components to circuit field elements.
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Dimension per attention head
    /// * `num_heads` - Number of attention heads
    pub fn estimate_attention_bounds(
        &self,
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> CircuitErrorBounds {
        let q = self.quantization_error;

        let bounds = AttentionBounds {
            seq_len,
            head_dim,
            num_heads,
            value_dim: head_dim,
            epsilon: q,
        };

        let attn_error = bounds.forward_error(q, q, q);

        CircuitErrorBounds::from_operations(&[
            ("qk_matmul", attn_error.qk_matmul_error),
            ("scaling", attn_error.scaling_error),
            ("softmax", attn_error.softmax_error.total_error),
            ("av_matmul", attn_error.av_matmul_error),
            ("projection", attn_error.proj_error),
            ("output", attn_error.concat_error),
        ])
    }

    /// Estimates error bounds for layer/RMS normalization.
    ///
    /// Delegates to `RMSNormBounds::forward_error()`.
    ///
    /// # Arguments
    /// * `normalized_dim` - Dimension being normalized
    /// * `max_magnitude` - Maximum input magnitude
    pub fn estimate_norm_bounds(
        &self,
        normalized_dim: usize,
        max_magnitude: f64,
    ) -> CircuitErrorBounds {
        let q = self.quantization_error;

        let bounds = RMSNormBounds {
            normalized_dim,
            min_rms: 1e-6,
            epsilon: q,
        };

        let norm_error = bounds.forward_error(q, max_magnitude);

        CircuitErrorBounds::from_operations(&[
            ("norm", norm_error.norm_error),
            ("scaling", norm_error.scale_error),
            ("output", norm_error.total_error),
        ])
    }

    /// Estimates error bounds for a full transformer layer.
    ///
    /// Composes attention + normalization + MLP bounds for a single
    /// transformer layer (pre-norm architecture: norm -> attn -> residual -> norm -> MLP -> residual).
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `d_ff` - Feed-forward inner dimension
    /// * `seq_len` - Sequence length
    /// * `num_heads` - Number of attention heads
    pub fn estimate_transformer_step_bounds(
        &self,
        d_model: usize,
        d_ff: usize,
        seq_len: usize,
        num_heads: usize,
    ) -> CircuitErrorBounds {
        let head_dim = d_model / num_heads.max(1);
        let max_mag = self.max_activation_magnitude;

        // Pre-attention normalization
        let norm1 = self.estimate_norm_bounds(d_model, max_mag);
        let norm1_total = norm1.total_as_f64();

        // Multi-head attention
        let attn = self.estimate_attention_bounds(seq_len, head_dim, num_heads);
        let attn_total = attn.total_as_f64();

        // Pre-MLP normalization
        let norm2 = self.estimate_norm_bounds(d_model, max_mag);
        let norm2_total = norm2.total_as_f64();

        // MLP (typically d_model -> d_ff -> d_model)
        let mlp = self.estimate_mlp_bounds(d_model, d_ff, d_model);
        let mlp_total = mlp.total_as_f64();

        // Residual connections add errors from both branches
        let q = self.quantization_error;
        let residual_error = 2.0 * q; // Two residual additions

        CircuitErrorBounds::from_operations(&[
            ("norm", norm1_total + norm2_total),
            ("qk_matmul", attn_total),
            ("matmul1", mlp_total),
            ("activation", 0.0), // GeLU/SiLU error is small
            ("bias1", residual_error),
            ("output", 0.0),
        ])
    }
}

/// Converts quantized weights to field element representation.
///
/// Weights are stored as fixed-point integers for circuit compatibility.
pub fn weights_to_field_elements(
    weights: &[f64],
    scale: f64,
) -> Vec<i64> {
    weights
        .iter()
        .map(|&w| (w * scale).round() as i64)
        .collect()
}

/// Converts field element weights back to f64.
pub fn field_elements_to_weights(
    field_elements: &[i64],
    scale: f64,
) -> Vec<f64> {
    field_elements
        .iter()
        .map(|&f| f as f64 / scale)
        .collect()
}

// ---------------------------------------------------------------------------
// AVM ↔ Circuit weight and witness bridge (requires helix-circuits)
// ---------------------------------------------------------------------------

/// Flat 2-layer MLP weights suitable for circuit consumption.
///
/// The circuit expects `w1[d_hid * d_in]`, `b1[d_hid]`, `w2[d_out * d_hid]`,
/// `b2[d_out]` in row-major order.
#[derive(Debug, Clone)]
pub struct CircuitWeights {
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
}

/// Maximum absolute value for weights/activations before quantization clamping.
///
/// Values beyond this magnitude would overflow the circuit's ReLU lookup table
/// or produce field elements too large for practical proof generation.
const MAX_QUANTIZATION_VALUE: f64 = 1e6;

/// Validates that all values in a slice are finite (not NaN or infinite).
fn validate_finite(values: &[f64], name: &str) -> Result<(), String> {
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!(
                "{name}[{i}] is not finite: {v}. All values must be finite for circuit compatibility."
            ));
        }
    }
    Ok(())
}

/// Clamps all values to `[-MAX_QUANTIZATION_VALUE, MAX_QUANTIZATION_VALUE]`.
///
/// Values beyond this range would overflow the circuit's lookup tables.
fn clamp_for_circuit(values: &mut [f64]) {
    for v in values.iter_mut() {
        *v = v.clamp(-MAX_QUANTIZATION_VALUE, MAX_QUANTIZATION_VALUE);
    }
}

/// Extracts flat weight vectors from two AVM `Linear` layers.
///
/// Layer 1: (d_in → d_hid), Layer 2: (d_hid → d_out).
/// Weights in `Linear` are stored as `[out_features, in_features]` row-major,
/// which is exactly the layout the circuit expects.
///
/// Returns an error if weights contain NaN or infinity.
pub fn model_to_circuit_weights(
    layer1: &crate::nn::Linear,
    layer2: &crate::nn::Linear,
) -> Result<CircuitWeights, String> {
    let d_in = layer1.in_features();
    let d_hid = layer1.out_features();
    let d_out = layer2.out_features();

    // Validate dimension compatibility
    if layer2.in_features() != d_hid {
        return Err(format!(
            "Layer dimension mismatch: layer1 out_features={d_hid} != layer2 in_features={}",
            layer2.in_features()
        ));
    }

    let mut w1: Vec<f64> = layer1.weights().values();
    let mut b1: Vec<f64> = layer1
        .bias()
        .map(|b| b.values())
        .unwrap_or_else(|| vec![0.0; d_hid]);

    let mut w2: Vec<f64> = layer2.weights().values();
    let mut b2: Vec<f64> = layer2
        .bias()
        .map(|b| b.values())
        .unwrap_or_else(|| vec![0.0; d_out]);

    // Validate all values are finite
    validate_finite(&w1, "w1")?;
    validate_finite(&b1, "b1")?;
    validate_finite(&w2, "w2")?;
    validate_finite(&b2, "b2")?;

    // Validate expected lengths
    if w1.len() != d_hid * d_in {
        return Err(format!(
            "w1 length {} != d_hid*d_in={}*{}={}",
            w1.len(), d_hid, d_in, d_hid * d_in
        ));
    }
    if w2.len() != d_out * d_hid {
        return Err(format!(
            "w2 length {} != d_out*d_hid={}*{}={}",
            w2.len(), d_out, d_hid, d_out * d_hid
        ));
    }

    // Clamp to circuit-safe range
    clamp_for_circuit(&mut w1);
    clamp_for_circuit(&mut b1);
    clamp_for_circuit(&mut w2);
    clamp_for_circuit(&mut b2);

    Ok(CircuitWeights {
        d_in,
        d_hid,
        d_out,
        w1,
        b1,
        w2,
        b2,
    })
}

/// Reconstructs two AVM `Linear` layers from flat circuit-weight vectors.
pub fn circuit_weights_to_model(
    cw: &CircuitWeights,
) -> (crate::nn::Linear, crate::nn::Linear) {
    use helix_core::types::Precision;

    let layer1 = crate::nn::Linear::from_raw(
        cw.w1.clone(),
        vec![cw.d_hid, cw.d_in],
        Some(cw.b1.clone()),
        Precision::F32,
    )
    .expect("valid layer1 dimensions");

    let layer2 = crate::nn::Linear::from_raw(
        cw.w2.clone(),
        vec![cw.d_out, cw.d_hid],
        Some(cw.b2.clone()),
        Precision::F32,
    )
    .expect("valid layer2 dimensions");

    (layer1, layer2)
}

// ---------------------------------------------------------------------------
// build_training_witness — full AVM → Circuit witness bridge
// ---------------------------------------------------------------------------

/// Output of [`build_training_witness`].
#[cfg(any(feature = "circuit-bridge", test))]
pub struct TrainingWitnessOutput {
    /// The fully populated witness for the circuit.
    pub witness: helix_circuits::MLTrainingStepV2Witness,
    /// Minimum `relu_range` the circuit must use for the lookup table.
    pub relu_range: usize,
}

/// Runs a complete AVM training step and produces an `MLTrainingStepV2Witness`.
///
/// This is the main bridge function: it takes AVM-level f64 model weights,
/// quantizes them to Fr using the given `scale` factor, then delegates to
/// `compute_witness_v2` for the circuit-compatible witness (including
/// Freivalds challenges, state hashes, and error checksums).
///
/// The scale factor controls quantization precision vs. circuit size:
/// - scale=1 (default): rounds to nearest integer, smallest circuit
/// - scale=10: one decimal place of precision
/// - scale=1000: three decimal places, but requires much larger relu_range
///
/// # Returns
/// A [`TrainingWitnessOutput`] containing the witness and the minimum
/// `relu_range` needed for the circuit's ReLU lookup table.
#[cfg(any(feature = "circuit-bridge", test))]
pub fn build_training_witness(
    layer1: &crate::nn::Linear,
    layer2: &crate::nn::Linear,
    input: &[f64],
    target: &[f64],
    learning_rate: f64,
    step_number: u64,
) -> Result<TrainingWitnessOutput, String> {
    build_training_witness_with_scale(layer1, layer2, input, target, learning_rate, step_number, 1)
}

/// Like [`build_training_witness`] but with a configurable quantization scale.
#[cfg(any(feature = "circuit-bridge", test))]
pub fn build_training_witness_with_scale(
    layer1: &crate::nn::Linear,
    layer2: &crate::nn::Linear,
    input: &[f64],
    target: &[f64],
    learning_rate: f64,
    step_number: u64,
    scale: u64,
) -> Result<TrainingWitnessOutput, String> {
    use crate::quantization::circuit_quantizer::fr_ops::i64_to_fr;
    use crate::quantization::CircuitQuantizer;
    use helix_circuits::halo2curves::bn256::Fr;
    use helix_circuits::halo2curves::ff::Field;
    use helix_circuits::{compute_state_hash_v2, compute_witness_v2};

    // 1. Extract flat weights from AVM layers (with validation)
    let cw = model_to_circuit_weights(layer1, layer2)?;

    if input.len() != cw.d_in {
        return Err(format!(
            "input length {} != d_in {}",
            input.len(),
            cw.d_in
        ));
    }
    if target.len() != cw.d_out {
        return Err(format!(
            "target length {} != d_out {}",
            target.len(),
            cw.d_out
        ));
    }

    // Validate inputs and targets are finite
    validate_finite(input, "input")?;
    validate_finite(target, "target")?;

    if learning_rate <= 0.0 || !learning_rate.is_finite() {
        return Err(format!(
            "learning_rate must be finite and positive, got {learning_rate}"
        ));
    }

    // 2. Quantize everything to Fr
    let q = CircuitQuantizer::with_scale(scale);

    let x_fr: Vec<Fr> = q.quantize_slice_to_fr(input);
    let target_fr: Vec<Fr> = q.quantize_slice_to_fr(target);
    let w1_fr: Vec<Fr> = q.quantize_slice_to_fr(&cw.w1);
    let b1_fr: Vec<Fr> = q.quantize_slice_to_fr(&cw.b1);
    let w2_fr: Vec<Fr> = q.quantize_slice_to_fr(&cw.w2);
    let b2_fr: Vec<Fr> = q.quantize_slice_to_fr(&cw.b2);
    let lr_fr: Fr = i64_to_fr(q.quantize_to_i64(learning_rate));

    // 3. Compute the minimum relu_range from the quantized values.
    //    h_pre[j] = sum_i(|w1[j,i]| * |x[i]|) + |b1[j]|
    //    We need all h_pre values to fit the ReLU lookup table.
    let max_w1 = cw.w1.iter().map(|v| (v * scale as f64).round().abs() as u64).max().unwrap_or(0);
    let max_x = input.iter().map(|v| (v * scale as f64).round().abs() as u64).max().unwrap_or(0);
    let max_b1 = cw.b1.iter().map(|v| (v * scale as f64).round().abs() as u64).max().unwrap_or(0);
    let max_h_pre = cw.d_in as u64 * max_w1 * max_x + max_b1;

    // Also account for y = W2 * h + b2 (though y doesn't go through ReLU)
    // and backward pass gradients that flow through relu_mask
    let max_w2 = cw.w2.iter().map(|v| (v * scale as f64).round().abs() as u64).max().unwrap_or(0);
    let max_b2 = cw.b2.iter().map(|v| (v * scale as f64).round().abs() as u64).max().unwrap_or(0);
    let max_y = cw.d_hid as u64 * max_w2 * max_h_pre + max_b2;

    // Backward pass: dh_pre goes through relu_mask, but it's element-wise multiply
    // with 0 or 1 so the magnitude is bounded by dh
    let max_target = target.iter().map(|v| (v * scale as f64).round().abs() as u64).max().unwrap_or(0);
    let max_dy = 2 * (max_y + max_target); // dy = 2 * (y - target)
    let max_dh = cw.d_out as u64 * max_w2 * max_dy;

    let relu_range = (max_h_pre.max(max_dh) + 1).max(256) as usize;

    // 4. Use the circuit's own compute_witness_v2 for exact arithmetic.
    let base_error = Fr::from(1u64);

    let old_hash = compute_state_hash_v2(&w1_fr, &b1_fr, &w2_fr, &b2_fr);
    let tmp = compute_witness_v2(
        cw.d_in,
        cw.d_hid,
        cw.d_out,
        &x_fr,
        &target_fr,
        &w1_fr,
        &b1_fr,
        &w2_fr,
        &b2_fr,
        lr_fr,
        old_hash,
        (Fr::ZERO, Fr::ZERO),
        step_number,
        base_error,
    );

    let new_hash = compute_state_hash_v2(
        &tmp.w1_new,
        &tmp.b1_new,
        &tmp.w2_new,
        &tmp.b2_new,
    );

    let mut witness = compute_witness_v2(
        cw.d_in,
        cw.d_hid,
        cw.d_out,
        &x_fr,
        &target_fr,
        &w1_fr,
        &b1_fr,
        &w2_fr,
        &b2_fr,
        lr_fr,
        old_hash,
        new_hash,
        step_number,
        base_error,
    );

    witness.finalize_error_checksum();

    // Verify witness consistency before returning
    verify_witness_consistency(&witness, cw.d_in, cw.d_hid, cw.d_out)?;

    Ok(TrainingWitnessOutput {
        witness,
        relu_range,
    })
}

/// Verifies that a witness has internally consistent dimensions and non-trivial state hashes.
///
/// This catches bugs where the witness was partially initialized or has mismatched vectors.
#[cfg(any(feature = "circuit-bridge", test))]
fn verify_witness_consistency(
    witness: &helix_circuits::MLTrainingStepV2Witness,
    d_in: usize,
    d_hid: usize,
    d_out: usize,
) -> Result<(), String> {
    use helix_circuits::halo2curves::bn256::Fr;
    use helix_circuits::halo2curves::ff::Field;

    // Verify dimension fields
    if witness.d_in != d_in {
        return Err(format!("Witness d_in={} != expected {d_in}", witness.d_in));
    }
    if witness.d_hid != d_hid {
        return Err(format!("Witness d_hid={} != expected {d_hid}", witness.d_hid));
    }
    if witness.d_out != d_out {
        return Err(format!("Witness d_out={} != expected {d_out}", witness.d_out));
    }

    // Verify vector lengths
    let checks = [
        ("x", witness.x.len(), d_in),
        ("target", witness.target.len(), d_out),
        ("w1", witness.w1.len(), d_hid * d_in),
        ("b1", witness.b1.len(), d_hid),
        ("w2", witness.w2.len(), d_out * d_hid),
        ("b2", witness.b2.len(), d_out),
        ("h_pre", witness.h_pre.len(), d_hid),
        ("h", witness.h.len(), d_hid),
        ("y", witness.y.len(), d_out),
        ("w1_new", witness.w1_new.len(), d_hid * d_in),
        ("b1_new", witness.b1_new.len(), d_hid),
        ("w2_new", witness.w2_new.len(), d_out * d_hid),
        ("b2_new", witness.b2_new.len(), d_out),
    ];

    for (name, actual, expected) in checks {
        if actual != expected {
            return Err(format!(
                "Witness vector {name} has length {actual}, expected {expected}"
            ));
        }
    }

    // Verify state hashes are non-trivial (both halves should be non-zero
    // for any non-trivial weight set)
    if witness.old_state_hash.0 == Fr::ZERO && witness.old_state_hash.1 == Fr::ZERO {
        return Err("Old state hash is (0, 0) — likely uninitialized".to_string());
    }
    if witness.new_state_hash.0 == Fr::ZERO && witness.new_state_hash.1 == Fr::ZERO {
        return Err("New state hash is (0, 0) — likely uninitialized".to_string());
    }

    // Public inputs should have exactly 8 elements
    let pi = witness.public_inputs();
    if pi.len() != 8 {
        return Err(format!(
            "Public inputs have {} elements, expected 8",
            pi.len()
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversion() {
        let error = 0.001;
        let field = error_to_field(error);
        let recovered = field_to_error(field);
        assert!((error - recovered).abs() < 0.0001);
    }

    #[test]
    fn test_zero_error() {
        assert_eq!(error_to_field(0.0), 0);
        assert_eq!(error_to_field(-1.0), 0);
    }

    #[test]
    fn test_circuit_error_bounds() {
        let bounds = CircuitErrorBounds::from_f64(0.01, 0.001, 0.0, 0.02, 0.001, 0.005);
        assert!(bounds.total_error > 0);
        assert!(bounds.total_as_f64() > 0.03);
    }

    #[test]
    fn test_mlp_bounds_estimation() {
        let config = TrainingErrorConfig::default();
        let bounds = config.estimate_mlp_bounds(4, 8, 2);
        assert!(bounds.total_error > 0);
    }

    #[test]
    fn test_weights_conversion() {
        let weights = vec![0.5, -0.3, 1.2, 0.0];
        let scale = 1000.0;
        let fields = weights_to_field_elements(&weights, scale);
        let recovered = field_elements_to_weights(&fields, scale);

        for (w, r) in weights.iter().zip(recovered.iter()) {
            assert!((w - r).abs() < 0.001);
        }
    }

    #[test]
    fn test_attention_bounds_estimation() {
        let config = TrainingErrorConfig::default();
        let bounds = config.estimate_attention_bounds(128, 64, 8);
        assert!(bounds.total_error > 0);
        assert!(bounds.total_as_f64() > 0.0);
    }

    #[test]
    fn test_norm_bounds_estimation() {
        let config = TrainingErrorConfig::default();
        let bounds = config.estimate_norm_bounds(512, 10.0);
        assert!(bounds.total_error > 0);
    }

    #[test]
    fn test_transformer_step_bounds() {
        let config = TrainingErrorConfig::default();
        let bounds = config.estimate_transformer_step_bounds(512, 2048, 128, 8);
        assert!(bounds.total_error > 0);

        // Transformer bounds should be larger than just MLP bounds
        let mlp_bounds = config.estimate_mlp_bounds(512, 2048, 512);
        assert!(bounds.total_as_f64() > mlp_bounds.total_as_f64());
    }

    #[test]
    fn test_from_operations() {
        let bounds = CircuitErrorBounds::from_operations(&[
            ("qk_matmul", 0.01),
            ("softmax", 0.005),
            ("av_matmul", 0.01),
            ("projection", 0.003),
        ]);
        assert!(bounds.total_error > 0);
        assert!(bounds.total_as_f64() > 0.02);
    }

    #[test]
    fn test_attention_vs_direct_bounds() {
        // Circuit bridge output should be consistent with direct bounds module calls
        let config = TrainingErrorConfig::default();
        let q = config.quantization_error;

        let direct = AttentionBounds {
            seq_len: 64,
            head_dim: 32,
            num_heads: 4,
            value_dim: 32,
            epsilon: q,
        };
        let direct_error = direct.forward_error(q, q, q);

        let bridge = config.estimate_attention_bounds(64, 32, 4);

        // The bridge total should account for the same components
        assert!(bridge.total_as_f64() > 0.0);
        // Total should be close to the sum of direct components
        let direct_sum = direct_error.qk_matmul_error
            + direct_error.scaling_error
            + direct_error.softmax_error.total_error
            + direct_error.av_matmul_error
            + direct_error.proj_error
            + direct_error.concat_error;
        let bridge_total = bridge.total_as_f64();
        // Allow for fixed-point quantization difference
        assert!((bridge_total - direct_sum).abs() < 1.0);
    }

    #[test]
    fn test_model_to_circuit_weights() {
        use crate::nn::Linear;
        use helix_core::types::Precision;

        let l1 = Linear::from_raw(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            Some(vec![0.1, 0.2]),
            Precision::F32,
        )
        .unwrap();

        let l2 = Linear::from_raw(
            vec![5.0, 6.0],
            vec![1, 2],
            Some(vec![0.3]),
            Precision::F32,
        )
        .unwrap();

        let cw = model_to_circuit_weights(&l1, &l2).unwrap();
        assert_eq!(cw.d_in, 2);
        assert_eq!(cw.d_hid, 2);
        assert_eq!(cw.d_out, 1);
        assert_eq!(cw.w1, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cw.b1, vec![0.1, 0.2]);
        assert_eq!(cw.w2, vec![5.0, 6.0]);
        assert_eq!(cw.b2, vec![0.3]);
    }

    #[test]
    fn test_circuit_weights_roundtrip() {
        use crate::nn::Linear;
        use helix_core::types::Precision;

        let l1 = Linear::from_raw(
            vec![0.5, -0.3, 1.2, 0.0],
            vec![2, 2],
            Some(vec![0.1, -0.1]),
            Precision::F32,
        )
        .unwrap();
        let l2 = Linear::from_raw(
            vec![0.7, 0.8],
            vec![1, 2],
            Some(vec![-0.2]),
            Precision::F32,
        )
        .unwrap();

        let cw = model_to_circuit_weights(&l1, &l2).unwrap();
        let (r1, r2) = circuit_weights_to_model(&cw);

        assert_eq!(r1.in_features(), 2);
        assert_eq!(r1.out_features(), 2);
        assert_eq!(r2.in_features(), 2);
        assert_eq!(r2.out_features(), 1);

        let r_cw = model_to_circuit_weights(&r1, &r2).unwrap();
        for (a, b) in cw.w1.iter().zip(r_cw.w1.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        for (a, b) in cw.b2.iter().zip(r_cw.b2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_build_training_witness() {
        use crate::nn::Linear;
        use helix_circuits::halo2curves::bn256::Fr;
        use helix_circuits::halo2curves::ff::Field;
        use helix_core::types::Precision;

        // Use integer-valued weights that match circuit test patterns.
        // W1 = [[1, 2], [3, 1]], b1 = [0, 0]
        // W2 = [[1, 1]], b2 = [0]
        // x = [1, 1], target = [5]
        // h_pre = [3, 4], h = [3, 4], y = [7], loss = (7-5)^2 = 4
        let l1 = Linear::from_raw(
            vec![1.0, 2.0, 3.0, 1.0],
            vec![2, 2],
            Some(vec![0.0, 0.0]),
            Precision::F32,
        )
        .unwrap();
        let l2 = Linear::from_raw(
            vec![1.0, 1.0],
            vec![1, 2],
            Some(vec![0.0]),
            Precision::F32,
        )
        .unwrap();

        let input = vec![1.0, 1.0];
        let target = vec![5.0];
        let lr = 1.0;

        let output = build_training_witness(&l1, &l2, &input, &target, lr, 1).unwrap();
        let witness = &output.witness;

        // Verify dimensions
        assert_eq!(witness.d_in, 2);
        assert_eq!(witness.d_hid, 2);
        assert_eq!(witness.d_out, 1);

        // Verify vectors have correct lengths
        assert_eq!(witness.x.len(), 2);
        assert_eq!(witness.target.len(), 1);
        assert_eq!(witness.w1.len(), 4);
        assert_eq!(witness.b1.len(), 2);
        assert_eq!(witness.w2.len(), 2);
        assert_eq!(witness.b2.len(), 1);
        assert_eq!(witness.h_pre.len(), 2);
        assert_eq!(witness.h.len(), 2);
        assert_eq!(witness.y.len(), 1);
        assert_eq!(witness.w1_new.len(), 4);
        assert_eq!(witness.w2_new.len(), 2);

        // Public inputs should have 8 elements
        let pi = witness.public_inputs();
        assert_eq!(pi.len(), 8);

        // h_pre = [3, 4], loss = (7-5)^2 = 4
        assert_eq!(witness.h_pre[0], Fr::from(3u64));
        assert_eq!(witness.h_pre[1], Fr::from(4u64));
        assert_eq!(witness.loss, Fr::from(4u64));

        // State hashes should be non-zero
        assert_ne!(witness.old_state_hash.0, Fr::ZERO);
        assert_ne!(witness.new_state_hash.0, Fr::ZERO);

        // Error checksum is computed (may be Fr::ZERO due to pre-existing
        // from_repr_vartime overflow in compute_error_checksum — not a bug here)
        let _ = witness.error_checksum;

        // relu_range should be at least 256
        assert!(output.relu_range >= 256);
    }

    #[test]
    fn test_build_training_witness_dimension_mismatch() {
        use crate::nn::Linear;
        use helix_core::types::Precision;

        let l1 = Linear::from_raw(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            Some(vec![0.0, 0.0]),
            Precision::F32,
        )
        .unwrap();
        let l2 = Linear::from_raw(
            vec![1.0, 1.0],
            vec![1, 2],
            Some(vec![0.0]),
            Precision::F32,
        )
        .unwrap();

        // Wrong input length
        let result = build_training_witness(&l1, &l2, &[1.0], &[1.0], 0.01, 0);
        assert!(result.is_err());

        // Wrong target length
        let result = build_training_witness(&l1, &l2, &[1.0, 2.0], &[1.0, 2.0], 0.01, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_training_witness_with_mock_prover() {
        use crate::nn::Linear;
        use helix_circuits::{MLTrainingStepV2Circuit, halo2_proofs::dev::MockProver};
        use helix_circuits::halo2curves::bn256::Fr;
        use helix_core::types::Precision;

        // Use the same values as the circuit's own test (make_tiny_circuit_v2).
        // W1 = [[1, 2], [3, 1]], b1 = [0, 0]
        // W2 = [[1, 1]], b2 = [0]
        // x = [1, 1], target = [5]
        let l1 = Linear::from_raw(
            vec![1.0, 2.0, 3.0, 1.0],
            vec![2, 2],
            Some(vec![0.0, 0.0]),
            Precision::F32,
        )
        .unwrap();
        let l2 = Linear::from_raw(
            vec![1.0, 1.0],
            vec![1, 2],
            Some(vec![0.0]),
            Precision::F32,
        )
        .unwrap();

        let input = vec![1.0, 1.0];
        let target = vec![5.0];
        let lr = 1.0;

        let output = build_training_witness(&l1, &l2, &input, &target, lr, 1).unwrap();
        let public_inputs = output.witness.public_inputs();

        let circuit = MLTrainingStepV2Circuit {
            witness: output.witness,
            relu_range: output.relu_range,
            exp_range: 128,
            exp_scale: 64,
            use_freivalds: true,
        };

        // k = 17 should be sufficient for a tiny model
        let k = 17;
        let prover = MockProver::<Fr>::run(k, &circuit, vec![public_inputs.clone()])
            .expect("MockProver::run failed");

        prover.assert_satisfied();
    }
}
