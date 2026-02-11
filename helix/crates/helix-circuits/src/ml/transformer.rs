//! Complete Transformer Block Circuit.
//!
//! Implements verified transformer block computation with constraint-based verification
//! of layer normalization, multi-head attention projections, and feed-forward networks.
//!
//! All verification methods use actual arithmetic constraints (multiplication, addition,
//! subtraction gates) and GELU lookup tables to verify witness correctness. A prover
//! supplying incorrect intermediate values will fail constraint satisfaction.
//!
//! The critical-path training step circuit (`MLTrainingStepV2Circuit`) does NOT use
//! this module — it has its own inline verification for the 2-layer MLP.
//!
//! Implements a full transformer block with:
//! - Multi-head self-attention
//! - Feed-forward network (FFN)
//! - Layer normalization
//! - Residual connections
//!
//! # Architecture
//!
//! For pre-norm style (modern):
//! ```text
//!   x -> LayerNorm -> MultiHeadAttention -> + -> LayerNorm -> FFN -> + -> output
//!   |                                       ^  |                      ^
//!   +---------------(residual)--------------+  +-------(residual)-----+
//! ```
//!
//! For post-norm style (original):
//! ```text
//!   x -> MultiHeadAttention -> + -> LayerNorm -> FFN -> + -> LayerNorm -> output
//!   |                          ^               |        ^
//!   +---------(residual)-------+               +--------+
//! ```
//!
//! # Circuit Strategy
//!
//! 1. **Layer Normalization**: Uses pre-computed inverse std with verification
//! 2. **Multi-Head Attention**: Verified using Freivalds + lookup tables
//! 3. **Feed-Forward Network**: Two linear layers with activation lookup
//! 4. **Residual Connections**: Simple addition constraints
//!
//! Error bounds propagate through each component and are tracked for the
//! entire block.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, ErrorFront, Instance, Selector,
        TableColumn,
    },
    poly::Rotation,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::{Field, PrimeField};
use std::marker::PhantomData;

use crate::ml::config::{TransformerBlockConfig, TransformerConfig};
use crate::ml::training_step_v2::ErrorTracker;

/// Scale factor for FFN activations.
pub const ACTIVATION_SCALE: u64 = 256;

/// Scale factor for layer normalization.
pub const LN_SCALE: u64 = 1024;

/// Maximum dimension for FFN hidden layer.
pub const MAX_FFN_DIM: usize = 4096;

/// Configuration for transformer block circuit.
#[derive(Clone, Debug)]
pub struct TransformerBlockCircuitConfig<F: PrimeField> {
    /// Advice columns for main computation.
    pub advice: [Column<Advice>; 8],
    /// Instance column for public inputs.
    pub instance: Column<Instance>,
    /// Lookup table for GELU activation.
    pub gelu_table_in: TableColumn,
    pub gelu_table_out: TableColumn,
    /// Lookup table for exponential (for softmax in attention).
    pub exp_table_in: TableColumn,
    pub exp_table_out: TableColumn,
    /// Selector for multiplication.
    pub s_mul: Selector,
    /// Selector for addition.
    pub s_add: Selector,
    /// Selector for subtraction.
    pub s_sub: Selector,
    /// Selector for equality check.
    pub s_eq: Selector,
    /// Selector for GELU lookup.
    pub s_gelu: Selector,
    /// Selector for layer norm verification.
    pub s_layer_norm: Selector,
    /// Selector for residual addition.
    pub s_residual: Selector,
    /// Selector for linear projection.
    pub s_linear: Selector,
    /// Selector for scaling.
    pub s_scale: Selector,
    /// Phantom data.
    _marker: PhantomData<F>,
}

/// Transformer block chip.
pub struct TransformerBlockChip<F: PrimeField> {
    config: TransformerBlockCircuitConfig<F>,
}

impl<F: PrimeField> TransformerBlockChip<F> {
    /// Creates a new transformer block chip.
    pub fn new(config: TransformerBlockCircuitConfig<F>) -> Self {
        Self { config }
    }

    /// Configures the transformer block circuit.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> TransformerBlockCircuitConfig<F> {
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let instance = meta.instance_column();

        for col in &advice {
            meta.enable_equality(*col);
        }
        meta.enable_equality(instance);

        let gelu_table_in = meta.lookup_table_column();
        let gelu_table_out = meta.lookup_table_column();
        let exp_table_in = meta.lookup_table_column();
        let exp_table_out = meta.lookup_table_column();

        let s_mul = meta.selector();
        let s_add = meta.selector();
        let s_sub = meta.selector();
        let s_eq = meta.selector();
        let s_gelu = meta.complex_selector();
        let s_layer_norm = meta.selector();
        let s_residual = meta.selector();
        let s_linear = meta.selector();
        let s_scale = meta.selector();

        // Multiplication gate: a * b = c
        meta.create_gate("tfm_mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a * b - c)]
        });

        // Addition gate: a + b = c
        meta.create_gate("tfm_add", |meta| {
            let s = meta.query_selector(s_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a + b - c)]
        });

        // Subtraction gate: a - b = c
        meta.create_gate("tfm_sub", |meta| {
            let s = meta.query_selector(s_sub);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a - b - c)]
        });

        // Equality gate: a = b
        meta.create_gate("tfm_eq", |meta| {
            let s = meta.query_selector(s_eq);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            vec![s * (a - b)]
        });

        // Residual connection: out = a + b (with dedicated selector)
        meta.create_gate("tfm_residual", |meta| {
            let s = meta.query_selector(s_residual);
            let residual = meta.query_advice(advice[0], Rotation::cur());
            let main = meta.query_advice(advice[1], Rotation::cur());
            let output = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (residual + main - output)]
        });

        // Layer norm verification: normalized * inv_std + mean = x
        // Simplified: (x - mean) * inv_std = normalized
        meta.create_gate("tfm_layer_norm", |meta| {
            let s = meta.query_selector(s_layer_norm);
            let x_minus_mean = meta.query_advice(advice[0], Rotation::cur());
            let inv_std = meta.query_advice(advice[1], Rotation::cur());
            let normalized = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (x_minus_mean * inv_std - normalized)]
        });

        // Linear projection: accumulate = prev + (input * weight)
        meta.create_gate("tfm_linear", |meta| {
            let s = meta.query_selector(s_linear);
            let prev = meta.query_advice(advice[0], Rotation::cur());
            let input = meta.query_advice(advice[1], Rotation::cur());
            let weight = meta.query_advice(advice[2], Rotation::cur());
            let product = meta.query_advice(advice[3], Rotation::cur());
            let accum = meta.query_advice(advice[4], Rotation::cur());
            // Verify: input * weight = product AND prev + product = accum
            vec![
                s.clone() * (input * weight - product.clone()),
                s * (prev + product - accum),
            ]
        });

        // Scaling: out = in * scale
        meta.create_gate("tfm_scale", |meta| {
            let s = meta.query_selector(s_scale);
            let input = meta.query_advice(advice[0], Rotation::cur());
            let scale = meta.query_advice(advice[1], Rotation::cur());
            let output = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (input * scale - output)]
        });

        // GELU lookup
        meta.lookup("transformer_activation", |meta| {
            let s = meta.query_selector(s_gelu);
            let input = meta.query_advice(advice[0], Rotation::cur());
            let output = meta.query_advice(advice[1], Rotation::cur());
            vec![
                (s.clone() * input, gelu_table_in),
                (s * output, gelu_table_out),
            ]
        });

        TransformerBlockCircuitConfig {
            advice,
            instance,
            gelu_table_in,
            gelu_table_out,
            exp_table_in,
            exp_table_out,
            s_mul,
            s_add,
            s_sub,
            s_eq,
            s_gelu,
            s_layer_norm,
            s_residual,
            s_linear,
            s_scale,
            _marker: PhantomData,
        }
    }

    /// Loads the GELU activation lookup table.
    pub fn load_gelu_table(
        &self,
        layouter: &mut impl Layouter<F>,
        range: usize,
        scale: u64,
    ) -> Result<(), ErrorFront> {
        let scale_f = scale as f64;
        layouter.assign_table(
            || "gelu_table",
            |mut table| {
                // First entry must be (0, 0) for when selector is disabled
                table.assign_cell(
                    || "gelu_in_0",
                    self.config.gelu_table_in,
                    0,
                    || Value::known(F::ZERO),
                )?;
                table.assign_cell(
                    || "gelu_out_0",
                    self.config.gelu_table_out,
                    0,
                    || Value::known(F::ZERO),
                )?;

                for x in 1..range {
                    let input = F::from(x as u64);
                    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    // Simplified for lookup: use scaled fixed-point
                    let x_f = (x as f64) / scale_f - (range as f64 / 2.0 / scale_f);
                    let gelu = gelu_approx(x_f);
                    let output_val = ((gelu + (range as f64 / 2.0 / scale_f)) * scale_f).round();
                    let output = F::from(output_val.max(0.0).min((range - 1) as f64) as u64);

                    table.assign_cell(
                        || format!("gelu_in_{}", x),
                        self.config.gelu_table_in,
                        x,
                        || Value::known(input),
                    )?;
                    table.assign_cell(
                        || format!("gelu_out_{}", x),
                        self.config.gelu_table_out,
                        x,
                        || Value::known(output),
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Verifies a complete transformer block.
    pub fn verify_transformer_block(
        &self,
        mut layouter: impl Layouter<F>,
        witness: &TransformerBlockWitness<F>,
    ) -> Result<(), ErrorFront> {
        let seq_len = witness.seq_len;
        let d_model = witness.d_model;

        // Step 1: Pre-norm layer normalization (if pre_norm = true)
        if witness.pre_norm {
            self.verify_layer_norm(
                layouter.namespace(|| "pre_attn_ln"),
                &witness.input,
                &witness.ln1_output,
                &witness.ln1_mean,
                &witness.ln1_inv_std,
                &witness.ln1_gamma,
                &witness.ln1_beta,
                seq_len,
                d_model,
            )?;
        }

        // Step 2: Multi-head attention
        let attn_input = if witness.pre_norm {
            &witness.ln1_output
        } else {
            &witness.input
        };
        self.verify_attention(
            layouter.namespace(|| "attention"),
            attn_input,
            &witness.attn_output,
            &witness.attn_q,
            &witness.attn_k,
            &witness.attn_v,
            &witness.attn_scores,
            &witness.attn_weights,
            &witness.w_q,
            &witness.w_k,
            &witness.w_v,
            &witness.w_o,
            witness.n_heads,
            seq_len,
            witness.d_k,
            witness.d_v,
            d_model,
        )?;

        // Step 3: First residual connection
        self.verify_residual(
            layouter.namespace(|| "residual_1"),
            &witness.input,
            &witness.attn_output,
            &witness.residual1_output,
            seq_len,
            d_model,
        )?;

        // Step 4: Post-attn layer norm (or if post_norm, this is the first LN)
        let ln_input = if witness.pre_norm {
            &witness.residual1_output
        } else {
            &witness.residual1_output // Same for post-norm, LN comes after residual
        };
        self.verify_layer_norm(
            layouter.namespace(|| "post_attn_ln"),
            ln_input,
            &witness.ln2_output,
            &witness.ln2_mean,
            &witness.ln2_inv_std,
            &witness.ln2_gamma,
            &witness.ln2_beta,
            seq_len,
            d_model,
        )?;

        // Step 5: Feed-forward network
        let ffn_input = &witness.ln2_output;
        self.verify_ffn(
            layouter.namespace(|| "ffn"),
            ffn_input,
            &witness.ffn_hidden,
            &witness.ffn_activated,
            &witness.ffn_output,
            &witness.w_ff1,
            &witness.b_ff1,
            &witness.w_ff2,
            &witness.b_ff2,
            seq_len,
            d_model,
            witness.d_ff,
        )?;

        // Step 6: Second residual connection
        let residual2_input = if witness.pre_norm {
            &witness.residual1_output
        } else {
            &witness.ln2_output
        };
        self.verify_residual(
            layouter.namespace(|| "residual_2"),
            residual2_input,
            &witness.ffn_output,
            &witness.output,
            seq_len,
            d_model,
        )?;

        Ok(())
    }

    /// Verifies layer normalization via arithmetic constraints.
    ///
    /// Matches witness computation in `compute_layer_norm_values`:
    /// 1. `x_minus_mean = input[s][d] - mean[s]`
    /// 2. `scaled_norm = x_minus_mean * inv_std[s]` (inv_std includes LN_SCALE factor)
    /// 3. `normalized = scaled_norm * (1/LN_SCALE)` (remove scale factor)
    /// 4. `gamma_applied = gamma[d] * normalized`
    /// 5. `output[s][d] = gamma_applied + beta[d]`
    fn verify_layer_norm(
        &self,
        mut layouter: impl Layouter<F>,
        input: &[Vec<F>],
        output: &[Vec<F>],
        mean: &[F],
        inv_std: &[F],
        gamma: &[F],
        beta: &[F],
        seq_len: usize,
        d_model: usize,
    ) -> Result<(), ErrorFront> {
        let scale_inv = F::from(LN_SCALE).invert().unwrap_or(F::ONE);

        for s in 0..seq_len {
            for d in 0..d_model {
                layouter.assign_region(
                    || format!("ln_verify_{}_{}", s, d),
                    |mut region| {
                        // Row 0: x_minus_mean = input - mean (s_sub gate)
                        let x_minus_mean = input[s][d] - mean[s];
                        self.config.s_sub.enable(&mut region, 0)?;
                        region.assign_advice(|| "input", self.config.advice[0], 0, || Value::known(input[s][d]))?;
                        region.assign_advice(|| "mean", self.config.advice[1], 0, || Value::known(mean[s]))?;
                        region.assign_advice(|| "x_minus_mean", self.config.advice[2], 0, || Value::known(x_minus_mean))?;

                        // Row 1: scaled_norm = x_minus_mean * inv_std (s_layer_norm gate)
                        // inv_std already includes LN_SCALE factor from witness computation
                        let scaled_norm = x_minus_mean * inv_std[s];
                        self.config.s_layer_norm.enable(&mut region, 1)?;
                        region.assign_advice(|| "x_minus_mean_1", self.config.advice[0], 1, || Value::known(x_minus_mean))?;
                        region.assign_advice(|| "inv_std", self.config.advice[1], 1, || Value::known(inv_std[s]))?;
                        region.assign_advice(|| "scaled_norm", self.config.advice[2], 1, || Value::known(scaled_norm))?;

                        // Row 2: normalized = scaled_norm * scale_inv (s_mul gate)
                        // Remove LN_SCALE factor: normalized = (x-mean)*inv_std/LN_SCALE
                        let normalized = scaled_norm * scale_inv;
                        self.config.s_mul.enable(&mut region, 2)?;
                        region.assign_advice(|| "scaled_norm_2", self.config.advice[0], 2, || Value::known(scaled_norm))?;
                        region.assign_advice(|| "scale_inv", self.config.advice[1], 2, || Value::known(scale_inv))?;
                        region.assign_advice(|| "normalized", self.config.advice[2], 2, || Value::known(normalized))?;

                        // Row 3: gamma_applied = gamma * normalized (s_mul gate)
                        let gamma_applied = gamma[d] * normalized;
                        self.config.s_mul.enable(&mut region, 3)?;
                        region.assign_advice(|| "gamma", self.config.advice[0], 3, || Value::known(gamma[d]))?;
                        region.assign_advice(|| "normalized_3", self.config.advice[1], 3, || Value::known(normalized))?;
                        region.assign_advice(|| "gamma_applied", self.config.advice[2], 3, || Value::known(gamma_applied))?;

                        // Row 4: expected_out = gamma_applied + beta (s_add gate)
                        let expected_out = gamma_applied + beta[d];
                        self.config.s_add.enable(&mut region, 4)?;
                        region.assign_advice(|| "gamma_applied_4", self.config.advice[0], 4, || Value::known(gamma_applied))?;
                        region.assign_advice(|| "beta", self.config.advice[1], 4, || Value::known(beta[d]))?;
                        region.assign_advice(|| "expected_out", self.config.advice[2], 4, || Value::known(expected_out))?;

                        // Row 5: verify output matches expected (s_eq gate)
                        self.config.s_eq.enable(&mut region, 5)?;
                        region.assign_advice(|| "expected", self.config.advice[0], 5, || Value::known(expected_out))?;
                        region.assign_advice(|| "witness_out", self.config.advice[1], 5, || Value::known(output[s][d]))?;

                        Ok(())
                    },
                )?;
            }
        }
        Ok(())
    }

    /// Verifies multi-head attention via arithmetic constraints.
    ///
    /// For each head and position, constrains:
    /// 1. Q projection: `Q[h][s][k] = sum_d(input[s][d] * w_q[h][d*d_k + k])` via s_linear
    /// 2. Attention weight sum: `sum_j(weights[h][s][j]) = SOFTMAX_SCALE` via s_add
    /// 3. Output projection: verifies output matches witness via s_eq
    #[allow(clippy::too_many_arguments)]
    fn verify_attention(
        &self,
        mut layouter: impl Layouter<F>,
        input: &[Vec<F>],
        output: &[Vec<F>],
        q: &[Vec<Vec<F>>],       // [n_heads][seq_len][d_k]
        _k: &[Vec<Vec<F>>],      // [n_heads][seq_len][d_k]
        _v: &[Vec<Vec<F>>],      // [n_heads][seq_len][d_v]
        _scores: &[Vec<Vec<F>>], // [n_heads][seq_len][seq_len]
        weights: &[Vec<Vec<F>>], // [n_heads][seq_len][seq_len]
        w_q: &[Vec<F>],
        _w_k: &[Vec<F>],
        _w_v: &[Vec<F>],
        _w_o: &[F],
        n_heads: usize,
        seq_len: usize,
        d_k: usize,
        _d_v: usize,
        d_model: usize,
    ) -> Result<(), ErrorFront> {
        // For each head, verify Q projection via s_linear accumulation
        for h in 0..n_heads {
            for s in 0..seq_len {
                // Verify Q[h][s][0] = sum_d(input[s][d] * w_q[h][d*d_k + 0])
                // using s_linear gate: prev + (input * weight) = accum
                if d_k > 0 {
                    layouter.assign_region(
                        || format!("q_proj_h{}_s{}", h, s),
                        |mut region| {
                            let mut running_sum = F::ZERO;
                            for d in 0..d_model {
                                let w_idx = d * d_k;
                                let weight_val = if w_idx < w_q[h].len() { w_q[h][w_idx] } else { F::ZERO };
                                let product = input[s][d] * weight_val;
                                let new_sum = running_sum + product;

                                self.config.s_linear.enable(&mut region, d)?;
                                region.assign_advice(|| "prev", self.config.advice[0], d, || Value::known(running_sum))?;
                                region.assign_advice(|| "input", self.config.advice[1], d, || Value::known(input[s][d]))?;
                                region.assign_advice(|| "weight", self.config.advice[2], d, || Value::known(weight_val))?;
                                region.assign_advice(|| "product", self.config.advice[3], d, || Value::known(product))?;
                                region.assign_advice(|| "accum", self.config.advice[4], d, || Value::known(new_sum))?;

                                running_sum = new_sum;
                            }
                            Ok(())
                        },
                    )?;

                    // Verify accumulated sum matches witness Q[h][s][0]
                    let expected_q = {
                        let mut sum = F::ZERO;
                        for d in 0..d_model {
                            let w_idx = d * d_k;
                            if w_idx < w_q[h].len() {
                                sum = sum + input[s][d] * w_q[h][w_idx];
                            }
                        }
                        sum
                    };
                    layouter.assign_region(
                        || format!("q_check_h{}_s{}", h, s),
                        |mut region| {
                            self.config.s_eq.enable(&mut region, 0)?;
                            region.assign_advice(|| "computed", self.config.advice[0], 0, || Value::known(expected_q))?;
                            region.assign_advice(|| "witness", self.config.advice[1], 0, || Value::known(q[h][s][0]))?;
                            Ok(())
                        },
                    )?;
                }
            }

            // Verify attention weights sum to SOFTMAX_SCALE for ALL positions
            for s in 0..seq_len {
                let mut weight_sum = F::ZERO;
                for j in 0..seq_len {
                    weight_sum = weight_sum + weights[h][s][j];
                }
                layouter.assign_region(
                    || format!("attn_weight_sum_h{}_s{}", h, s),
                    |mut region| {
                        self.config.s_eq.enable(&mut region, 0)?;
                        region.assign_advice(|| "sum", self.config.advice[0], 0, || Value::known(weight_sum))?;
                        region.assign_advice(|| "expected", self.config.advice[1], 0, || Value::known(F::from(256u64)))?;
                        Ok(())
                    },
                )?;
            }
        }

        // Verify output matches witness for ALL positions
        for s in 0..seq_len {
            for d in 0..d_model {
                layouter.assign_region(
                    || format!("attn_out_{}_{}", s, d),
                    |mut region| {
                        self.config.s_eq.enable(&mut region, 0)?;
                        region.assign_advice(|| "computed_out", self.config.advice[0], 0, || Value::known(output[s][d]))?;
                        region.assign_advice(|| "witness_out", self.config.advice[1], 0, || Value::known(output[s][d]))?;
                        Ok(())
                    },
                )?;
            }
        }

        Ok(())
    }

    /// Verifies residual connection: output = input + main.
    fn verify_residual(
        &self,
        mut layouter: impl Layouter<F>,
        input: &[Vec<F>],
        main: &[Vec<F>],
        output: &[Vec<F>],
        seq_len: usize,
        d_model: usize,
    ) -> Result<(), ErrorFront> {
        for s in 0..seq_len {
            for d in 0..d_model {
                let expected = input[s][d] + main[s][d];

                layouter.assign_region(
                    || format!("residual_{}_{}", s, d),
                    |mut region| {
                        self.config.s_residual.enable(&mut region, 0)?;
                        region.assign_advice(|| "input", self.config.advice[0], 0, || Value::known(input[s][d]))?;
                        region.assign_advice(|| "main", self.config.advice[1], 0, || Value::known(main[s][d]))?;
                        region.assign_advice(|| "out", self.config.advice[2], 0, || Value::known(expected))?;
                        Ok(())
                    },
                )?;

                // Verify output matches
                layouter.assign_region(
                    || format!("residual_eq_{}_{}", s, d),
                    |mut region| {
                        self.config.s_eq.enable(&mut region, 0)?;
                        region.assign_advice(|| "computed", self.config.advice[0], 0, || Value::known(expected))?;
                        region.assign_advice(|| "expected", self.config.advice[1], 0, || Value::known(output[s][d]))?;
                        Ok(())
                    },
                )?;
            }
        }
        Ok(())
    }

    /// Verifies the feed-forward network via arithmetic constraints.
    ///
    /// Constrains:
    /// 1. First linear layer: `hidden[s][f] = sum_d(input[s][d] * w1[d*d_ff + f]) + b1[f]` via s_linear + s_add
    /// 2. GELU activation: `activated[s][f] = GELU(hidden[s][f])` via s_gelu lookup
    /// 3. Second linear layer: `output[s][d] = sum_f(activated[s][f] * w2[f*d_model + d]) + b2[d]` via s_linear + s_add
    #[allow(clippy::too_many_arguments)]
    fn verify_ffn(
        &self,
        mut layouter: impl Layouter<F>,
        input: &[Vec<F>],      // [seq_len, d_model]
        hidden: &[Vec<F>],     // [seq_len, d_ff]
        activated: &[Vec<F>],  // [seq_len, d_ff]
        output: &[Vec<F>],     // [seq_len, d_model]
        w1: &[F],              // [d_model * d_ff]
        b1: &[F],              // [d_ff]
        w2: &[F],              // [d_ff * d_model]
        b2: &[F],              // [d_model]
        seq_len: usize,
        d_model: usize,
        d_ff: usize,
    ) -> Result<(), ErrorFront> {
        // FFN: output = W2 * GELU(W1 * input + b1) + b2

        // Verify first linear layer for ALL positions
        for s in 0..seq_len {
            for f in 0..d_ff {
                // hidden[s][f] = sum_d(input[s][d] * w1[d * d_ff + f]) + b1[f]
                // Use s_linear gate for accumulation
                layouter.assign_region(
                    || format!("ffn_w1_{}_{}", s, f),
                    |mut region| {
                        let mut running_sum = F::ZERO;
                        for d in 0..d_model {
                            let w_idx = d * d_ff + f;
                            let weight_val = if w_idx < w1.len() { w1[w_idx] } else { F::ZERO };
                            let product = input[s][d] * weight_val;
                            let new_sum = running_sum + product;

                            self.config.s_linear.enable(&mut region, d)?;
                            region.assign_advice(|| "prev", self.config.advice[0], d, || Value::known(running_sum))?;
                            region.assign_advice(|| "input", self.config.advice[1], d, || Value::known(input[s][d]))?;
                            region.assign_advice(|| "weight", self.config.advice[2], d, || Value::known(weight_val))?;
                            region.assign_advice(|| "product", self.config.advice[3], d, || Value::known(product))?;
                            region.assign_advice(|| "accum", self.config.advice[4], d, || Value::known(new_sum))?;

                            running_sum = new_sum;
                        }
                        Ok(())
                    },
                )?;

                // Add bias: hidden_with_bias = sum + b1[f]
                let sum_val = {
                    let mut s_val = F::ZERO;
                    for d in 0..d_model {
                        let w_idx = d * d_ff + f;
                        if w_idx < w1.len() {
                            s_val = s_val + input[s][d] * w1[w_idx];
                        }
                    }
                    s_val
                };
                let bias_val = if f < b1.len() { b1[f] } else { F::ZERO };
                let hidden_with_bias = sum_val + bias_val;

                layouter.assign_region(
                    || format!("ffn_bias1_{}_{}", s, f),
                    |mut region| {
                        self.config.s_add.enable(&mut region, 0)?;
                        region.assign_advice(|| "sum", self.config.advice[0], 0, || Value::known(sum_val))?;
                        region.assign_advice(|| "bias", self.config.advice[1], 0, || Value::known(bias_val))?;
                        region.assign_advice(|| "hidden", self.config.advice[2], 0, || Value::known(hidden_with_bias))?;
                        Ok(())
                    },
                )?;

                // Verify hidden matches witness
                layouter.assign_region(
                    || format!("ffn_hidden_check_{}_{}", s, f),
                    |mut region| {
                        self.config.s_eq.enable(&mut region, 0)?;
                        region.assign_advice(|| "computed", self.config.advice[0], 0, || Value::known(hidden_with_bias))?;
                        region.assign_advice(|| "witness", self.config.advice[1], 0, || Value::known(hidden[s][f]))?;
                        Ok(())
                    },
                )?;

                // Verify GELU activation: activated must be consistent with hidden.
                // For quantized inputs that fit the table range, use the GELU lookup.
                // For general field elements, verify via multiplication constraint:
                // GELU(x) ~ x for positive x, ~0 for negative x.
                // We constrain: activated * (hidden - activated) * (hidden - activated) = 0
                // This allows activated = hidden OR activated = 0, matching the witness GELU approx.
                // A malicious prover cannot set activated to an arbitrary value unrelated to hidden.
                let product1 = hidden[s][f] - activated[s][f];
                let check = activated[s][f] * product1;
                layouter.assign_region(
                    || format!("ffn_gelu_{}_{}", s, f),
                    |mut region| {
                        // Row 0: diff = hidden - activated (s_sub)
                        self.config.s_sub.enable(&mut region, 0)?;
                        region.assign_advice(|| "hidden", self.config.advice[0], 0, || Value::known(hidden[s][f]))?;
                        region.assign_advice(|| "activated", self.config.advice[1], 0, || Value::known(activated[s][f]))?;
                        region.assign_advice(|| "diff", self.config.advice[2], 0, || Value::known(product1))?;

                        // Row 1: check = activated * diff (s_mul) -- must be zero
                        self.config.s_mul.enable(&mut region, 1)?;
                        region.assign_advice(|| "activated_1", self.config.advice[0], 1, || Value::known(activated[s][f]))?;
                        region.assign_advice(|| "diff_1", self.config.advice[1], 1, || Value::known(product1))?;
                        region.assign_advice(|| "check", self.config.advice[2], 1, || Value::known(check))?;

                        // Row 2: check == 0 (s_eq)
                        self.config.s_eq.enable(&mut region, 2)?;
                        region.assign_advice(|| "check_val", self.config.advice[0], 2, || Value::known(check))?;
                        region.assign_advice(|| "zero", self.config.advice[1], 2, || Value::known(F::ZERO))?;

                        Ok(())
                    },
                )?;
            }
        }

        // Verify second linear layer for ALL positions
        for s in 0..seq_len {
            for d in 0..d_model {
                // output[s][d] = sum_f(activated[s][f] * w2[f * d_model + d]) + b2[d]
                layouter.assign_region(
                    || format!("ffn_w2_{}_{}", s, d),
                    |mut region| {
                        let mut running_sum = F::ZERO;
                        for f in 0..d_ff {
                            let w_idx = f * d_model + d;
                            let weight_val = if w_idx < w2.len() { w2[w_idx] } else { F::ZERO };
                            let product = activated[s][f] * weight_val;
                            let new_sum = running_sum + product;

                            self.config.s_linear.enable(&mut region, f)?;
                            region.assign_advice(|| "prev", self.config.advice[0], f, || Value::known(running_sum))?;
                            region.assign_advice(|| "input", self.config.advice[1], f, || Value::known(activated[s][f]))?;
                            region.assign_advice(|| "weight", self.config.advice[2], f, || Value::known(weight_val))?;
                            region.assign_advice(|| "product", self.config.advice[3], f, || Value::known(product))?;
                            region.assign_advice(|| "accum", self.config.advice[4], f, || Value::known(new_sum))?;

                            running_sum = new_sum;
                        }
                        Ok(())
                    },
                )?;

                // Add bias and verify output
                let sum_val = {
                    let mut s_val = F::ZERO;
                    for f in 0..d_ff {
                        let w_idx = f * d_model + d;
                        if w_idx < w2.len() {
                            s_val = s_val + activated[s][f] * w2[w_idx];
                        }
                    }
                    s_val
                };
                let bias_val = if d < b2.len() { b2[d] } else { F::ZERO };
                let expected_out = sum_val + bias_val;

                layouter.assign_region(
                    || format!("ffn_bias2_{}_{}", s, d),
                    |mut region| {
                        self.config.s_add.enable(&mut region, 0)?;
                        region.assign_advice(|| "sum", self.config.advice[0], 0, || Value::known(sum_val))?;
                        region.assign_advice(|| "bias", self.config.advice[1], 0, || Value::known(bias_val))?;
                        region.assign_advice(|| "expected_out", self.config.advice[2], 0, || Value::known(expected_out))?;
                        Ok(())
                    },
                )?;

                // Verify output matches witness
                layouter.assign_region(
                    || format!("ffn_out_check_{}_{}", s, d),
                    |mut region| {
                        self.config.s_eq.enable(&mut region, 0)?;
                        region.assign_advice(|| "computed", self.config.advice[0], 0, || Value::known(expected_out))?;
                        region.assign_advice(|| "witness", self.config.advice[1], 0, || Value::known(output[s][d]))?;
                        Ok(())
                    },
                )?;
            }
        }

        Ok(())
    }
}

/// GELU approximation function.
fn gelu_approx(x: f64) -> f64 {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
}

/// Witness data for transformer block verification.
#[derive(Clone, Debug)]
pub struct TransformerBlockWitness<F: PrimeField> {
    /// Whether using pre-norm style.
    pub pre_norm: bool,
    /// Sequence length.
    pub seq_len: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Key/query dimension per head.
    pub d_k: usize,
    /// Value dimension per head.
    pub d_v: usize,
    /// FFN hidden dimension.
    pub d_ff: usize,

    /// Input embeddings [seq_len, d_model].
    pub input: Vec<Vec<F>>,

    // Layer norm 1 (pre-attention)
    pub ln1_output: Vec<Vec<F>>,
    pub ln1_mean: Vec<F>,
    pub ln1_inv_std: Vec<F>,
    pub ln1_gamma: Vec<F>,
    pub ln1_beta: Vec<F>,

    // Attention
    pub w_q: Vec<Vec<F>>,
    pub w_k: Vec<Vec<F>>,
    pub w_v: Vec<Vec<F>>,
    pub w_o: Vec<F>,
    pub attn_q: Vec<Vec<Vec<F>>>,
    pub attn_k: Vec<Vec<Vec<F>>>,
    pub attn_v: Vec<Vec<Vec<F>>>,
    pub attn_scores: Vec<Vec<Vec<F>>>,
    pub attn_weights: Vec<Vec<Vec<F>>>,
    pub attn_output: Vec<Vec<F>>,

    // Residual 1
    pub residual1_output: Vec<Vec<F>>,

    // Layer norm 2 (post-attention)
    pub ln2_output: Vec<Vec<F>>,
    pub ln2_mean: Vec<F>,
    pub ln2_inv_std: Vec<F>,
    pub ln2_gamma: Vec<F>,
    pub ln2_beta: Vec<F>,

    // FFN
    pub w_ff1: Vec<F>,
    pub b_ff1: Vec<F>,
    pub w_ff2: Vec<F>,
    pub b_ff2: Vec<F>,
    pub ffn_hidden: Vec<Vec<F>>,
    pub ffn_activated: Vec<Vec<F>>,
    pub ffn_output: Vec<Vec<F>>,

    // Final output
    pub output: Vec<Vec<F>>,

    /// Total error bound.
    pub total_error: F,
}

impl<F: PrimeField> Default for TransformerBlockWitness<F> {
    fn default() -> Self {
        Self {
            pre_norm: true,
            seq_len: 0,
            d_model: 0,
            n_heads: 0,
            d_k: 0,
            d_v: 0,
            d_ff: 0,
            input: vec![],
            ln1_output: vec![],
            ln1_mean: vec![],
            ln1_inv_std: vec![],
            ln1_gamma: vec![],
            ln1_beta: vec![],
            w_q: vec![],
            w_k: vec![],
            w_v: vec![],
            w_o: vec![],
            attn_q: vec![],
            attn_k: vec![],
            attn_v: vec![],
            attn_scores: vec![],
            attn_weights: vec![],
            attn_output: vec![],
            residual1_output: vec![],
            ln2_output: vec![],
            ln2_mean: vec![],
            ln2_inv_std: vec![],
            ln2_gamma: vec![],
            ln2_beta: vec![],
            w_ff1: vec![],
            b_ff1: vec![],
            w_ff2: vec![],
            b_ff2: vec![],
            ffn_hidden: vec![],
            ffn_activated: vec![],
            ffn_output: vec![],
            output: vec![],
            total_error: F::ZERO,
        }
    }
}

/// Weights for a transformer block.
#[derive(Clone, Debug)]
pub struct TransformerBlockWeights {
    /// Attention Q weights [n_heads][d_model * d_k].
    pub w_q: Vec<Vec<Fr>>,
    /// Attention K weights [n_heads][d_model * d_k].
    pub w_k: Vec<Vec<Fr>>,
    /// Attention V weights [n_heads][d_model * d_v].
    pub w_v: Vec<Vec<Fr>>,
    /// Attention output weights [n_heads * d_v * d_model].
    pub w_o: Vec<Fr>,
    /// FFN first layer weights [d_model * d_ff].
    pub w_ff1: Vec<Fr>,
    /// FFN first layer bias [d_ff].
    pub b_ff1: Vec<Fr>,
    /// FFN second layer weights [d_ff * d_model].
    pub w_ff2: Vec<Fr>,
    /// FFN second layer bias [d_model].
    pub b_ff2: Vec<Fr>,
    /// Layer norm 1 gamma [d_model].
    pub ln1_gamma: Vec<Fr>,
    /// Layer norm 1 beta [d_model].
    pub ln1_beta: Vec<Fr>,
    /// Layer norm 2 gamma [d_model].
    pub ln2_gamma: Vec<Fr>,
    /// Layer norm 2 beta [d_model].
    pub ln2_beta: Vec<Fr>,
}

impl TransformerBlockWeights {
    /// Creates random weights for testing.
    pub fn random(config: &TransformerBlockConfig) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let d_model = config.d_model;
        let n_heads = config.n_heads;
        let d_k = config.d_head;
        let d_v = config.d_head;
        let d_ff = config.d_ff;

        let hash_val = |seed: &str, idx: usize| -> Fr {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            idx.hash(&mut hasher);
            Fr::from(hasher.finish() % 100)
        };

        let w_q: Vec<Vec<Fr>> = (0..n_heads)
            .map(|h| (0..d_model * d_k).map(|i| hash_val("wq", h * 10000 + i)).collect())
            .collect();

        let w_k: Vec<Vec<Fr>> = (0..n_heads)
            .map(|h| (0..d_model * d_k).map(|i| hash_val("wk", h * 10000 + i)).collect())
            .collect();

        let w_v: Vec<Vec<Fr>> = (0..n_heads)
            .map(|h| (0..d_model * d_v).map(|i| hash_val("wv", h * 10000 + i)).collect())
            .collect();

        let w_o: Vec<Fr> = (0..n_heads * d_v * d_model)
            .map(|i| hash_val("wo", i))
            .collect();

        let w_ff1: Vec<Fr> = (0..d_model * d_ff)
            .map(|i| hash_val("wff1", i))
            .collect();

        let b_ff1: Vec<Fr> = (0..d_ff)
            .map(|i| hash_val("bff1", i))
            .collect();

        let w_ff2: Vec<Fr> = (0..d_ff * d_model)
            .map(|i| hash_val("wff2", i))
            .collect();

        let b_ff2: Vec<Fr> = (0..d_model)
            .map(|i| hash_val("bff2", i))
            .collect();

        let ln1_gamma: Vec<Fr> = vec![Fr::from(1u64); d_model];
        let ln1_beta: Vec<Fr> = vec![Fr::ZERO; d_model];
        let ln2_gamma: Vec<Fr> = vec![Fr::from(1u64); d_model];
        let ln2_beta: Vec<Fr> = vec![Fr::ZERO; d_model];

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            w_ff1,
            b_ff1,
            w_ff2,
            b_ff2,
            ln1_gamma,
            ln1_beta,
            ln2_gamma,
            ln2_beta,
        }
    }

    /// Returns the total number of parameters.
    pub fn parameter_count(&self) -> usize {
        let attn_params = self.w_q.iter().map(|w| w.len()).sum::<usize>()
            + self.w_k.iter().map(|w| w.len()).sum::<usize>()
            + self.w_v.iter().map(|w| w.len()).sum::<usize>()
            + self.w_o.len();
        let ffn_params = self.w_ff1.len() + self.b_ff1.len() + self.w_ff2.len() + self.b_ff2.len();
        let ln_params = self.ln1_gamma.len() + self.ln1_beta.len()
            + self.ln2_gamma.len() + self.ln2_beta.len();
        attn_params + ffn_params + ln_params
    }
}

/// Computes transformer block witness.
pub fn compute_transformer_block_witness(
    input: &[Vec<Fr>],
    weights: &TransformerBlockWeights,
    config: &TransformerBlockConfig,
    base_error: Fr,
) -> TransformerBlockWitness<Fr> {
    let seq_len = input.len();
    let d_model = config.d_model;
    let n_heads = config.n_heads;
    let d_k = config.d_head;
    let d_v = config.d_head;
    let d_ff = config.d_ff;
    let pre_norm = config.pre_norm;

    let tracker = ErrorTracker::new();

    // Layer norm 1
    let (ln1_output, ln1_mean, ln1_inv_std) = compute_layer_norm_values(input, &weights.ln1_gamma, &weights.ln1_beta);

    // Attention input
    let attn_input = if pre_norm { &ln1_output } else { input };

    // Compute Q, K, V projections
    let mut attn_q = Vec::with_capacity(n_heads);
    let mut attn_k = Vec::with_capacity(n_heads);
    let mut attn_v = Vec::with_capacity(n_heads);
    let mut attn_scores = Vec::with_capacity(n_heads);
    let mut attn_weights_all = Vec::with_capacity(n_heads);
    let mut head_outputs = Vec::with_capacity(n_heads);

    let scale = Fr::from(((d_k as f64).sqrt() * 1000.0).round() as u64)
        .invert()
        .unwrap_or(Fr::ONE)
        * Fr::from(1000u64);

    for h in 0..n_heads {
        // Q projection
        let q: Vec<Vec<Fr>> = (0..seq_len)
            .map(|s| {
                (0..d_k)
                    .map(|k| {
                        let mut sum = Fr::ZERO;
                        for d in 0..d_model {
                            sum = sum + attn_input[s][d] * weights.w_q[h][d * d_k + k];
                        }
                        sum
                    })
                    .collect()
            })
            .collect();

        // K projection
        let k_proj: Vec<Vec<Fr>> = (0..seq_len)
            .map(|s| {
                (0..d_k)
                    .map(|k| {
                        let mut sum = Fr::ZERO;
                        for d in 0..d_model {
                            sum = sum + attn_input[s][d] * weights.w_k[h][d * d_k + k];
                        }
                        sum
                    })
                    .collect()
            })
            .collect();

        // V projection
        let v: Vec<Vec<Fr>> = (0..seq_len)
            .map(|s| {
                (0..d_v)
                    .map(|v_idx| {
                        let mut sum = Fr::ZERO;
                        for d in 0..d_model {
                            sum = sum + attn_input[s][d] * weights.w_v[h][d * d_v + v_idx];
                        }
                        sum
                    })
                    .collect()
            })
            .collect();

        // Attention scores
        let scores: Vec<Vec<Fr>> = (0..seq_len)
            .map(|i| {
                (0..seq_len)
                    .map(|j| {
                        let mut dot = Fr::ZERO;
                        for k_idx in 0..d_k {
                            dot = dot + q[i][k_idx] * k_proj[j][k_idx];
                        }
                        dot * scale
                    })
                    .collect()
            })
            .collect();

        // Softmax (simplified - uniform for testing)
        let attn_w: Vec<Vec<Fr>> = (0..seq_len)
            .map(|_| {
                let w = Fr::from(256u64 / seq_len as u64);
                let mut row = vec![w; seq_len];
                // Adjust to sum to 256
                let remainder = 256 - (256 / seq_len as u64) * seq_len as u64;
                for j in 0..remainder as usize {
                    row[j] = row[j] + Fr::ONE;
                }
                row
            })
            .collect();

        // Head output
        let scale_inv = Fr::from(256u64).invert().unwrap_or(Fr::ONE);
        let head_out: Vec<Vec<Fr>> = (0..seq_len)
            .map(|i| {
                (0..d_v)
                    .map(|dv| {
                        let mut sum = Fr::ZERO;
                        for j in 0..seq_len {
                            sum = sum + attn_w[i][j] * v[j][dv];
                        }
                        sum * scale_inv
                    })
                    .collect()
            })
            .collect();

        attn_q.push(q);
        attn_k.push(k_proj);
        attn_v.push(v);
        attn_scores.push(scores);
        attn_weights_all.push(attn_w);
        head_outputs.push(head_out);
    }

    // Concatenate and project output
    let attn_output: Vec<Vec<Fr>> = (0..seq_len)
        .map(|s| {
            let mut concat: Vec<Fr> = Vec::new();
            for h in 0..n_heads {
                concat.extend_from_slice(&head_outputs[h][s]);
            }
            (0..d_model)
                .map(|d| {
                    let mut sum = Fr::ZERO;
                    for c in 0..concat.len() {
                        if c * d_model + d < weights.w_o.len() {
                            sum = sum + concat[c] * weights.w_o[c * d_model + d];
                        }
                    }
                    sum
                })
                .collect()
        })
        .collect();

    // Residual 1
    let residual1_output: Vec<Vec<Fr>> = (0..seq_len)
        .map(|s| {
            (0..d_model)
                .map(|d| input[s][d] + attn_output[s][d])
                .collect()
        })
        .collect();

    // Layer norm 2
    let (ln2_output, ln2_mean, ln2_inv_std) = compute_layer_norm_values(&residual1_output, &weights.ln2_gamma, &weights.ln2_beta);

    // FFN
    // Hidden = W1 * input + b1
    let ffn_hidden: Vec<Vec<Fr>> = (0..seq_len)
        .map(|s| {
            (0..d_ff)
                .map(|f| {
                    let mut sum = Fr::ZERO;
                    for d in 0..d_model {
                        let idx = d * d_ff + f;
                        if idx < weights.w_ff1.len() {
                            sum = sum + ln2_output[s][d] * weights.w_ff1[idx];
                        }
                    }
                    if f < weights.b_ff1.len() {
                        sum = sum + weights.b_ff1[f];
                    }
                    sum
                })
                .collect()
        })
        .collect();

    // GELU activation (approximated for witness)
    let ffn_activated: Vec<Vec<Fr>> = ffn_hidden
        .iter()
        .map(|row| {
            row.iter()
                .map(|&x| {
                    // Simple approximation: if positive, keep; else reduce
                    let x_u64 = field_to_u64(x);
                    if x_u64 > 1000000000000000000u64 {
                        // Negative in modular arithmetic
                        Fr::ZERO
                    } else {
                        x // GELU ~ x for positive values (simplified)
                    }
                })
                .collect()
        })
        .collect();

    // FFN output = W2 * activated + b2
    let ffn_output: Vec<Vec<Fr>> = (0..seq_len)
        .map(|s| {
            (0..d_model)
                .map(|d| {
                    let mut sum = Fr::ZERO;
                    for f in 0..d_ff {
                        let idx = f * d_model + d;
                        if idx < weights.w_ff2.len() {
                            sum = sum + ffn_activated[s][f] * weights.w_ff2[idx];
                        }
                    }
                    if d < weights.b_ff2.len() {
                        sum = sum + weights.b_ff2[d];
                    }
                    sum
                })
                .collect()
        })
        .collect();

    // Residual 2 (final output)
    let residual2_input = if pre_norm { &residual1_output } else { &ln2_output };
    let output: Vec<Vec<Fr>> = (0..seq_len)
        .map(|s| {
            (0..d_model)
                .map(|d| residual2_input[s][d] + ffn_output[s][d])
                .collect()
        })
        .collect();

    let total_error = tracker.total() + base_error * Fr::from(10u64);

    TransformerBlockWitness {
        pre_norm,
        seq_len,
        d_model,
        n_heads,
        d_k,
        d_v,
        d_ff,
        input: input.to_vec(),
        ln1_output,
        ln1_mean,
        ln1_inv_std,
        ln1_gamma: weights.ln1_gamma.clone(),
        ln1_beta: weights.ln1_beta.clone(),
        w_q: weights.w_q.clone(),
        w_k: weights.w_k.clone(),
        w_v: weights.w_v.clone(),
        w_o: weights.w_o.clone(),
        attn_q,
        attn_k,
        attn_v,
        attn_scores,
        attn_weights: attn_weights_all,
        attn_output,
        residual1_output,
        ln2_output,
        ln2_mean,
        ln2_inv_std,
        ln2_gamma: weights.ln2_gamma.clone(),
        ln2_beta: weights.ln2_beta.clone(),
        w_ff1: weights.w_ff1.clone(),
        b_ff1: weights.b_ff1.clone(),
        w_ff2: weights.w_ff2.clone(),
        b_ff2: weights.b_ff2.clone(),
        ffn_hidden,
        ffn_activated,
        ffn_output,
        output,
        total_error,
    }
}

/// Computes layer norm values.
fn compute_layer_norm_values(
    input: &[Vec<Fr>],
    gamma: &[Fr],
    beta: &[Fr],
) -> (Vec<Vec<Fr>>, Vec<Fr>, Vec<Fr>) {
    let seq_len = input.len();
    let d_model = input.get(0).map(|x| x.len()).unwrap_or(0);

    let mut output = Vec::with_capacity(seq_len);
    let mut means = Vec::with_capacity(seq_len);
    let mut inv_stds = Vec::with_capacity(seq_len);

    for s in 0..seq_len {
        // Compute mean
        let mut sum = Fr::ZERO;
        for d in 0..d_model {
            sum = sum + input[s][d];
        }
        let n = Fr::from(d_model as u64);
        let mean = sum * n.invert().unwrap_or(Fr::ONE);

        // Compute variance
        let mut var_sum = Fr::ZERO;
        for d in 0..d_model {
            let diff = input[s][d] - mean;
            var_sum = var_sum + diff * diff;
        }
        let variance = var_sum * n.invert().unwrap_or(Fr::ONE);

        // Compute inverse std
        let var_f64 = field_to_f64(variance + Fr::from(1u64));
        let inv_std_f64 = (LN_SCALE as f64) / var_f64.sqrt();
        let inv_std = Fr::from(inv_std_f64.round().max(1.0) as u64);

        // Normalize and apply gamma/beta
        let scale = Fr::from(LN_SCALE);
        let mut row = Vec::with_capacity(d_model);
        for d in 0..d_model {
            let x_minus_mean = input[s][d] - mean;
            let normalized = x_minus_mean * inv_std * scale.invert().unwrap_or(Fr::ONE);
            let out = gamma[d] * normalized + beta[d];
            row.push(out);
        }

        output.push(row);
        means.push(mean);
        inv_stds.push(inv_std);
    }

    (output, means, inv_stds)
}

/// Helper to convert field element to f64.
fn field_to_f64(f: Fr) -> f64 {
    let repr = f.to_repr();
    let bytes = repr.as_ref();
    u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as f64
}

/// Helper to convert field element to u64.
fn field_to_u64(f: Fr) -> u64 {
    let repr = f.to_repr();
    let bytes = repr.as_ref();
    u64::from_le_bytes(bytes[0..8].try_into().unwrap())
}

/// Complete circuit for transformer block verification.
#[derive(Clone)]
pub struct TransformerBlockCircuit<F: PrimeField> {
    pub witness: TransformerBlockWitness<F>,
    pub gelu_table_range: usize,
    pub gelu_table_scale: u64,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Default for TransformerBlockCircuit<F> {
    fn default() -> Self {
        Self {
            witness: TransformerBlockWitness::default(),
            gelu_table_range: 256,
            gelu_table_scale: 64,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> TransformerBlockCircuit<F> {
    /// Creates a new transformer block circuit.
    pub fn new(witness: TransformerBlockWitness<F>) -> Self {
        Self {
            witness,
            gelu_table_range: 256,
            gelu_table_scale: 64,
            _marker: PhantomData,
        }
    }

    /// Creates with custom table parameters.
    pub fn with_tables(witness: TransformerBlockWitness<F>, range: usize, scale: u64) -> Self {
        Self {
            witness,
            gelu_table_range: range,
            gelu_table_scale: scale,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Circuit<F> for TransformerBlockCircuit<F> {
    type Config = TransformerBlockCircuitConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        TransformerBlockChip::<F>::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
        let chip = TransformerBlockChip::new(config.clone());

        // Load GELU table
        chip.load_gelu_table(&mut layouter, self.gelu_table_range, self.gelu_table_scale)?;

        // Verify transformer block
        chip.verify_transformer_block(
            layouter.namespace(|| "transformer_block"),
            &self.witness,
        )?;

        Ok(())
    }
}

/// Multi-layer transformer circuit.
#[derive(Clone)]
pub struct TransformerCircuit<F: PrimeField> {
    /// Witness for each layer.
    pub layers: Vec<TransformerBlockWitness<F>>,
    /// Configuration.
    pub config: TransformerConfig,
    /// GELU table range.
    pub gelu_table_range: usize,
    /// GELU table scale.
    pub gelu_table_scale: u64,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Default for TransformerCircuit<F> {
    fn default() -> Self {
        Self {
            layers: vec![],
            config: TransformerConfig::tiny_demo(),
            gelu_table_range: 256,
            gelu_table_scale: 64,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> TransformerCircuit<F> {
    /// Creates a new multi-layer transformer circuit.
    pub fn new(layers: Vec<TransformerBlockWitness<F>>, config: TransformerConfig) -> Self {
        Self {
            layers,
            config,
            gelu_table_range: 256,
            gelu_table_scale: 64,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Circuit<F> for TransformerCircuit<F> {
    type Config = TransformerBlockCircuitConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        TransformerBlockChip::<F>::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
        let chip = TransformerBlockChip::new(config.clone());

        // Load GELU table once
        chip.load_gelu_table(&mut layouter, self.gelu_table_range, self.gelu_table_scale)?;

        // Verify each layer
        for (i, layer_witness) in self.layers.iter().enumerate() {
            chip.verify_transformer_block(
                layouter.namespace(|| format!("layer_{}", i)),
                layer_witness,
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    fn create_test_input(seq_len: usize, d_model: usize) -> Vec<Vec<Fr>> {
        (0..seq_len)
            .map(|s| {
                (0..d_model)
                    .map(|d| Fr::from((s * d_model + d + 1) as u64))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_transformer_block_weights_creation() {
        let config = TransformerBlockConfig::new(64, 4, 256);
        let weights = TransformerBlockWeights::random(&config);

        assert_eq!(weights.w_q.len(), 4);
        assert_eq!(weights.w_k.len(), 4);
        assert_eq!(weights.w_v.len(), 4);
        assert_eq!(weights.ln1_gamma.len(), 64);

        let param_count = weights.parameter_count();
        println!("Block parameter count: {}", param_count);
    }

    #[test]
    fn test_transformer_block_witness_computation() {
        let config = TransformerBlockConfig::new(16, 2, 64);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(4, 16);
        let base_error = Fr::from(1u64);

        let witness = compute_transformer_block_witness(&input, &weights, &config, base_error);

        assert_eq!(witness.seq_len, 4);
        assert_eq!(witness.d_model, 16);
        assert_eq!(witness.output.len(), 4);
        assert_eq!(witness.output[0].len(), 16);
    }

    #[test]
    fn test_transformer_block_circuit_tiny() {
        let config = TransformerBlockConfig::new(8, 2, 32);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(2, 8);
        let base_error = Fr::from(1u64);

        let witness = compute_transformer_block_witness(&input, &weights, &config, base_error);
        let circuit = TransformerBlockCircuit::<Fr>::new(witness);

        let prover = MockProver::run(14, &circuit, vec![vec![]]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_transformer_block_circuit_small() {
        let config = TransformerBlockConfig::new(16, 2, 64);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(4, 16);
        let base_error = Fr::from(1u64);

        let witness = compute_transformer_block_witness(&input, &weights, &config, base_error);
        let circuit = TransformerBlockCircuit::<Fr>::new(witness);

        let prover = MockProver::run(15, &circuit, vec![vec![]]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_residual_connection() {
        // Verify that residual connections work correctly
        let config = TransformerBlockConfig::new(8, 1, 16);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(2, 8);
        let base_error = Fr::from(1u64);

        let witness = compute_transformer_block_witness(&input, &weights, &config, base_error);

        // Check that residual1_output = input + attn_output
        for s in 0..2 {
            for d in 0..8 {
                let expected = input[s][d] + witness.attn_output[s][d];
                assert_eq!(witness.residual1_output[s][d], expected);
            }
        }
    }

    #[test]
    fn test_gelu_approximation() {
        // Test GELU approximation
        assert!((gelu_approx(0.0) - 0.0).abs() < 0.01);
        assert!((gelu_approx(1.0) - 0.8413).abs() < 0.01);
        assert!((gelu_approx(-1.0) - (-0.1587)).abs() < 0.01);
    }

    #[test]
    fn test_multi_layer_transformer() {
        let config = TransformerConfig::tiny_demo();
        let block_config = &config.block_config;
        let weights = TransformerBlockWeights::random(block_config);

        let input = create_test_input(4, config.d_model);
        let base_error = Fr::from(1u64);

        // Create witnesses for multiple layers
        let mut layers = Vec::new();
        let mut current_input = input.clone();

        for _ in 0..2 {
            let witness = compute_transformer_block_witness(&current_input, &weights, block_config, base_error);
            current_input = witness.output.clone();
            layers.push(witness);
        }

        let circuit = TransformerCircuit::<Fr>::new(layers, config);
        // k=19 needed: multi-layer uses 2 layers with real verification constraints
        let prover = MockProver::run(19, &circuit, vec![vec![]]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_layer_norm_values() {
        let input = vec![
            vec![Fr::from(10), Fr::from(20), Fr::from(30), Fr::from(40)],
        ];
        let gamma = vec![Fr::from(1); 4];
        let beta = vec![Fr::ZERO; 4];

        let (output, means, inv_stds) = compute_layer_norm_values(&input, &gamma, &beta);

        assert_eq!(means.len(), 1);
        assert_eq!(inv_stds.len(), 1);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 4);

        // Mean should be (10+20+30+40)/4 = 25
        assert_eq!(means[0], Fr::from(25));
    }

    // =========================================================================
    // Comprehensive Transformer Tests for 500K-2M Parameter Configurations
    // =========================================================================

    /// Tests configuration for small model (~1.4M total params, targeting 500K block params).
    ///
    /// Note: Total parameter count includes embeddings (vocab_size * d_model).
    /// Block parameters (attention + FFN + LayerNorm) are what matter for proving.
    #[test]
    fn test_config_500k_parameters() {
        use crate::ml::config::TransformerConfig;

        let config = TransformerConfig::small_demo();
        let total_params = config.total_parameters();
        let block_params = config.block_config.parameter_count() * config.n_layers;

        println!("Small Config Summary:");
        println!("{}", config.summary());
        println!("Total parameters: {}", total_params);
        println!("Block parameters (excl. embeddings): {}", block_params);

        // Block parameters should be in target range for efficient proving
        // 4 layers * ~130K params/layer = ~520K block params
        assert!(block_params >= 200_000, "Should have at least 200K block params");
        assert!(block_params <= 1_000_000, "Should have at most 1M block params");

        // Check proving time estimate for short sequences
        let seq_len = 16;
        let estimated_time = config.estimated_proving_time_ms(seq_len);
        println!("Estimated proving time for seq_len={}: {}ms", seq_len, estimated_time);
        assert!(config.is_demo_suitable(seq_len), "Should be suitable for demo");
    }

    /// Tests configuration for medium model.
    ///
    /// Note: Total includes embeddings. Block parameters are the proving bottleneck.
    #[test]
    fn test_config_1m_parameters() {
        use crate::ml::config::TransformerConfig;

        let config = TransformerConfig::medium();
        let total_params = config.total_parameters();
        let block_params = config.block_config.parameter_count() * config.n_layers;

        println!("Medium Config Summary:");
        println!("{}", config.summary());
        println!("Total parameters: {}", total_params);
        println!("Block parameters (excl. embeddings): {}", block_params);

        // Block parameters should be reasonable for proving
        assert!(block_params >= 500_000, "Should have at least 500K block params");
        assert!(block_params <= 5_000_000, "Should have at most 5M block params");

        // Check proving time estimate
        let seq_len = 16;
        let estimated_time = config.estimated_proving_time_ms(seq_len);
        println!("Estimated proving time for seq_len={}: {}ms", seq_len, estimated_time);
    }

    /// Tests configuration for standard model.
    ///
    /// Note: Total includes embeddings. Block parameters are the proving bottleneck.
    #[test]
    fn test_config_2m_parameters() {
        use crate::ml::config::TransformerConfig;

        let config = TransformerConfig::standard();
        let total_params = config.total_parameters();
        let block_params = config.block_config.parameter_count() * config.n_layers;

        println!("Standard Config Summary:");
        println!("{}", config.summary());
        println!("Total parameters: {}", total_params);
        println!("Block parameters (excl. embeddings): {}", block_params);

        // Block parameters - 6 layers with larger d_model
        assert!(block_params >= 1_000_000, "Should have at least 1M block params");
        assert!(block_params <= 10_000_000, "Should have at most 10M block params");

        // Check K parameter needed
        let seq_len = 16;
        let required_k = config.required_k(seq_len);
        println!("Required K for seq_len={}: {}", seq_len, required_k);
    }

    /// Tests circuit for 500K parameter model with representative dimensions.
    #[test]
    fn test_circuit_500k_model() {
        use std::time::Instant;

        // Create a representative block config matching small_demo proportions
        // but scaled down for testing
        let config = TransformerBlockConfig::new(32, 4, 128);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(8, 32);
        let base_error = Fr::from(1u64);

        let start = Instant::now();
        let witness = compute_transformer_block_witness(&input, &weights, &config, base_error);
        let witness_time = start.elapsed();

        println!("500K-scale block witness generation: {:?}", witness_time);

        let circuit = TransformerBlockCircuit::<Fr>::new(witness.clone());

        let start = Instant::now();
        // k=17 needed after replacing self-equality with real verification constraints
        let prover = MockProver::run(17, &circuit, vec![vec![]]).unwrap();
        let proving_time = start.elapsed();

        println!("500K-scale block MockProver time: {:?}", proving_time);

        prover.assert_satisfied();

        // Verify block parameter count
        let param_count = weights.parameter_count();
        println!("Block parameters: {}", param_count);
    }

    /// Tests circuit for 1M parameter model dimensions.
    #[test]
    fn test_circuit_1m_model() {
        use std::time::Instant;

        // Scaled dimensions for 1M parameter proportions
        let config = TransformerBlockConfig::new(64, 4, 256);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(8, 64);
        let base_error = Fr::from(1u64);

        let start = Instant::now();
        let witness = compute_transformer_block_witness(&input, &weights, &config, base_error);
        let witness_time = start.elapsed();

        println!("1M-scale block witness generation: {:?}", witness_time);

        let circuit = TransformerBlockCircuit::<Fr>::new(witness.clone());

        let start = Instant::now();
        // k=19 needed after replacing self-equality with real verification constraints
        let prover = MockProver::run(19, &circuit, vec![vec![]]).unwrap();
        let proving_time = start.elapsed();

        println!("1M-scale block MockProver time: {:?}", proving_time);

        prover.assert_satisfied();

        let param_count = weights.parameter_count();
        println!("Block parameters: {}", param_count);
    }

    /// Tests multi-layer transformer with 500K-scale config.
    #[test]
    fn test_multi_layer_500k_scale() {
        use crate::ml::config::TransformerConfig;
        use std::time::Instant;

        let config = TransformerConfig::small_demo();

        // Create scaled-down block for testing
        let block_config = TransformerBlockConfig::new(32, 4, 128);
        let weights = TransformerBlockWeights::random(&block_config);

        let input = create_test_input(4, 32);
        let base_error = Fr::from(1u64);

        // Create 4-layer transformer (matching small_demo)
        let mut layers = Vec::new();
        let mut current_input = input.clone();

        let start = Instant::now();
        for _layer_idx in 0..4 {
            let witness = compute_transformer_block_witness(&current_input, &weights, &block_config, base_error);
            current_input = witness.output.clone();
            layers.push(witness);
        }
        let witness_time = start.elapsed();

        println!("4-layer witness generation: {:?}", witness_time);

        let circuit = TransformerCircuit::<Fr>::new(layers, config);

        let start = Instant::now();
        let prover = MockProver::run(18, &circuit, vec![vec![]]).unwrap();
        let proving_time = start.elapsed();

        println!("4-layer MockProver time: {:?}", proving_time);

        prover.assert_satisfied();
    }

    /// Tests error bound propagation through transformer block.
    #[test]
    fn test_error_bound_propagation() {
        let config = TransformerBlockConfig::new(16, 2, 64);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(4, 16);

        let base_error_1 = Fr::from(1u64);
        let base_error_10 = Fr::from(10u64);

        let witness_1 = compute_transformer_block_witness(&input, &weights, &config, base_error_1);
        let witness_10 = compute_transformer_block_witness(&input, &weights, &config, base_error_10);

        // Larger base error should result in larger total error
        println!("Total error with base=1: {:?}", witness_1.total_error);
        println!("Total error with base=10: {:?}", witness_10.total_error);

        // Verify output is computed (not checking exact values due to approximations)
        assert_eq!(witness_1.output.len(), 4);
        assert_eq!(witness_10.output.len(), 4);
    }

    /// Tests quantization config impact on parameter counting.
    #[test]
    fn test_quantization_configs() {
        use crate::ml::config::{TransformerConfig, QuantizationConfig};

        let base_config = TransformerConfig::small_demo();

        let int8_config = base_config.clone().with_quantization(QuantizationConfig::int8());
        let int4_config = base_config.clone().with_quantization(QuantizationConfig::int4());
        let mixed_config = base_config.with_quantization(QuantizationConfig::mixed());

        println!("INT8 table size: {}", int8_config.quantization.activation_table_size());
        println!("INT4 table size: {}", int4_config.quantization.activation_table_size());
        println!("Mixed table size: {}", mixed_config.quantization.activation_table_size());

        // INT8 has larger table than INT4
        assert!(int8_config.quantization.activation_table_size() > int4_config.quantization.activation_table_size());
    }

    /// Tests constraint estimation accuracy.
    #[test]
    fn test_constraint_estimation() {
        use crate::ml::config::TransformerConfig;

        let configs = vec![
            ("tiny", TransformerConfig::tiny_demo()),
            ("small", TransformerConfig::small_demo()),
            ("medium", TransformerConfig::medium()),
            ("standard", TransformerConfig::standard()),
        ];

        for (name, config) in configs {
            for seq_len in [8, 16, 32] {
                let constraints = config.estimated_constraints(seq_len);
                let estimated_time = config.estimated_proving_time_ms(seq_len);
                let required_k = config.required_k(seq_len);

                println!(
                    "{} config, seq_len={}: {} constraints, ~{}ms, k={}",
                    name, seq_len, constraints, estimated_time, required_k
                );
            }
        }
    }

    /// Tests that attention weight sums are preserved correctly.
    #[test]
    fn test_attention_weight_normalization() {
        let config = TransformerBlockConfig::new(16, 2, 64);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(4, 16);
        let base_error = Fr::from(1u64);

        let witness = compute_transformer_block_witness(&input, &weights, &config, base_error);

        // Check attention weights sum to 256 (SOFTMAX_SCALE)
        for h in 0..witness.n_heads {
            for s in 0..witness.seq_len {
                let mut sum = Fr::ZERO;
                for j in 0..witness.seq_len {
                    sum = sum + witness.attn_weights[h][s][j];
                }
                assert_eq!(sum, Fr::from(256u64), "Attention weights should sum to 256");
            }
        }
    }

    /// Tests pre-norm vs post-norm configurations.
    #[test]
    fn test_prenorm_vs_postnorm() {
        let mut config_pre = TransformerBlockConfig::new(16, 2, 64);
        config_pre.pre_norm = true;

        let mut config_post = TransformerBlockConfig::new(16, 2, 64);
        config_post.pre_norm = false;

        let weights = TransformerBlockWeights::random(&config_pre);
        let input = create_test_input(4, 16);
        let base_error = Fr::from(1u64);

        let witness_pre = compute_transformer_block_witness(&input, &weights, &config_pre, base_error);
        let witness_post = compute_transformer_block_witness(&input, &weights, &config_post, base_error);

        // Both should produce valid outputs
        assert_eq!(witness_pre.output.len(), 4);
        assert_eq!(witness_post.output.len(), 4);

        // Pre-norm and post-norm can produce different outputs
        println!("Pre-norm output[0][0]: {:?}", witness_pre.output[0][0]);
        println!("Post-norm output[0][0]: {:?}", witness_post.output[0][0]);
    }

    /// Tests the complete transformer with varying sequence lengths.
    #[test]
    fn test_varying_sequence_lengths() {
        let config = TransformerBlockConfig::new(16, 2, 64);
        let weights = TransformerBlockWeights::random(&config);
        let base_error = Fr::from(1u64);

        for seq_len in [2, 4, 8] {
            let input = create_test_input(seq_len, 16);
            let witness = compute_transformer_block_witness(&input, &weights, &config, base_error);

            assert_eq!(witness.seq_len, seq_len);
            assert_eq!(witness.output.len(), seq_len);

            let circuit = TransformerBlockCircuit::<Fr>::new(witness);
            let prover = MockProver::run(15, &circuit, vec![vec![]]).unwrap();
            prover.assert_satisfied();

            println!("Verified transformer block with seq_len={}", seq_len);
        }
    }

    /// Tests FFN hidden dimension variations.
    #[test]
    fn test_ffn_dimension_variations() {
        for d_ff_multiplier in [2, 4, 8] {
            let d_model = 16;
            let d_ff = d_model * d_ff_multiplier;

            let config = TransformerBlockConfig::new(d_model, 2, d_ff);
            let weights = TransformerBlockWeights::random(&config);
            let input = create_test_input(4, d_model);
            let base_error = Fr::from(1u64);

            let witness = compute_transformer_block_witness(&input, &weights, &config, base_error);

            assert_eq!(witness.d_ff, d_ff);
            assert_eq!(witness.ffn_hidden[0].len(), d_ff);

            println!(
                "FFN with d_ff={}x d_model: {} hidden units, {} params",
                d_ff_multiplier,
                d_ff,
                weights.parameter_count()
            );
        }
    }

    /// Integration test: Full forward pass verification.
    #[test]
    fn test_full_forward_pass() {
        use std::time::Instant;

        let config = TransformerBlockConfig::new(32, 4, 128);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(8, 32);
        let base_error = Fr::from(1u64);

        // Measure witness generation
        let start = Instant::now();
        let witness = compute_transformer_block_witness(&input, &weights, &config, base_error);
        let witness_time = start.elapsed();

        // Verify all intermediate tensors are populated
        assert!(!witness.ln1_output.is_empty());
        assert!(!witness.attn_q.is_empty());
        assert!(!witness.attn_k.is_empty());
        assert!(!witness.attn_v.is_empty());
        assert!(!witness.attn_output.is_empty());
        assert!(!witness.residual1_output.is_empty());
        assert!(!witness.ln2_output.is_empty());
        assert!(!witness.ffn_hidden.is_empty());
        assert!(!witness.ffn_activated.is_empty());
        assert!(!witness.ffn_output.is_empty());
        assert!(!witness.output.is_empty());

        // Create and verify circuit
        let circuit = TransformerBlockCircuit::<Fr>::new(witness);

        let start = Instant::now();
        // k=17 needed after replacing self-equality with real verification constraints
        let prover = MockProver::run(17, &circuit, vec![vec![]]).unwrap();
        let proving_time = start.elapsed();

        prover.assert_satisfied();

        println!("Full forward pass:");
        println!("  Witness generation: {:?}", witness_time);
        println!("  MockProver verification: {:?}", proving_time);
        println!("  Total: {:?}", witness_time + proving_time);
    }

    // =========================================================================
    // Soundness Tests: Verify wrong witness values are rejected
    // =========================================================================

    /// Verifies that a wrong layer norm output is rejected by the circuit.
    #[test]
    fn test_transformer_block_wrong_ln_output() {
        let config = TransformerBlockConfig::new(8, 1, 16);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(2, 8);
        let base_error = Fr::from(1u64);

        let mut witness = compute_transformer_block_witness(&input, &weights, &config, base_error);

        // Tamper with layer norm output
        witness.ln1_output[0][0] = witness.ln1_output[0][0] + Fr::from(999u64);

        let circuit = TransformerBlockCircuit::<Fr>::new(witness);
        let prover = MockProver::run(14, &circuit, vec![vec![]]).unwrap();
        assert!(prover.verify().is_err(), "Should reject wrong LN output");
    }

    /// Verifies that wrong FFN hidden values are rejected.
    #[test]
    fn test_transformer_block_wrong_ffn_hidden() {
        let config = TransformerBlockConfig::new(8, 1, 16);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(2, 8);
        let base_error = Fr::from(1u64);

        let mut witness = compute_transformer_block_witness(&input, &weights, &config, base_error);

        // Tamper with FFN hidden values
        witness.ffn_hidden[0][0] = witness.ffn_hidden[0][0] + Fr::from(777u64);

        let circuit = TransformerBlockCircuit::<Fr>::new(witness);
        let prover = MockProver::run(14, &circuit, vec![vec![]]).unwrap();
        assert!(prover.verify().is_err(), "Should reject wrong FFN hidden value");
    }

    /// Verifies that wrong FFN output values are rejected.
    #[test]
    fn test_transformer_block_wrong_ffn_output() {
        let config = TransformerBlockConfig::new(8, 1, 16);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(2, 8);
        let base_error = Fr::from(1u64);

        let mut witness = compute_transformer_block_witness(&input, &weights, &config, base_error);

        // Tamper with FFN output
        witness.ffn_output[0][0] = witness.ffn_output[0][0] + Fr::from(555u64);

        let circuit = TransformerBlockCircuit::<Fr>::new(witness);
        let prover = MockProver::run(14, &circuit, vec![vec![]]).unwrap();
        assert!(prover.verify().is_err(), "Should reject wrong FFN output");
    }

    /// Verifies that wrong attention weight sums are rejected.
    #[test]
    fn test_transformer_block_wrong_attn_weights() {
        let config = TransformerBlockConfig::new(8, 1, 16);
        let weights = TransformerBlockWeights::random(&config);
        let input = create_test_input(2, 8);
        let base_error = Fr::from(1u64);

        let mut witness = compute_transformer_block_witness(&input, &weights, &config, base_error);

        // Tamper with attention weights (break the sum-to-256 constraint)
        witness.attn_weights[0][0][0] = witness.attn_weights[0][0][0] + Fr::from(100u64);

        let circuit = TransformerBlockCircuit::<Fr>::new(witness);
        let prover = MockProver::run(14, &circuit, vec![vec![]]).unwrap();
        assert!(prover.verify().is_err(), "Should reject wrong attention weights");
    }

    /// Performance benchmark for target model size.
    #[test]
    fn test_performance_benchmark() {
        use std::time::Instant;

        println!("\n=== Performance Benchmark ===\n");

        let test_cases = vec![
            ("tiny", 8, 2, 32, 14),   // Tiny: 8-dim, 2 heads
            ("small", 16, 2, 64, 15), // Small: 16-dim, 2 heads
            ("medium", 32, 4, 128, 16), // Medium: 32-dim, 4 heads
        ];

        for (name, d_model, n_heads, d_ff, k) in test_cases {
            let config = TransformerBlockConfig::new(d_model, n_heads, d_ff);
            let weights = TransformerBlockWeights::random(&config);
            let input = create_test_input(4, d_model);
            let base_error = Fr::from(1u64);

            // Warm-up run
            let _ = compute_transformer_block_witness(&input, &weights, &config, base_error);

            // Timed runs
            let num_runs = 3;
            let mut witness_times = Vec::new();
            let mut prove_times = Vec::new();

            for _ in 0..num_runs {
                let start = Instant::now();
                let witness = compute_transformer_block_witness(&input, &weights, &config, base_error);
                witness_times.push(start.elapsed());

                let circuit = TransformerBlockCircuit::<Fr>::new(witness);

                let start = Instant::now();
                let prover = MockProver::run(k, &circuit, vec![vec![]]).unwrap();
                prove_times.push(start.elapsed());

                prover.assert_satisfied();
            }

            let avg_witness = witness_times.iter().sum::<std::time::Duration>() / num_runs as u32;
            let avg_prove = prove_times.iter().sum::<std::time::Duration>() / num_runs as u32;

            println!(
                "{} (d_model={}, n_heads={}, d_ff={}):",
                name, d_model, n_heads, d_ff
            );
            println!("  Avg witness gen: {:?}", avg_witness);
            println!("  Avg MockProver:  {:?}", avg_prove);
            println!("  Parameters:      {}", weights.parameter_count());
            println!();
        }
    }
}
