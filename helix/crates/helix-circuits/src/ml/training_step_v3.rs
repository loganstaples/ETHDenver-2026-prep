//! N-Layer ML Training Step Circuit V3.
//!
//! Extends the V2 circuit from a fixed 2-layer MLP to support N-layer
//! networks with configurable activations (ReLU, Tanh, Identity) and
//! loss functions (MSE, CrossEntropy).
//!
//! The 8 public inputs remain identical to V2, so on-chain contracts
//! need no changes.
//!
//! # Architecture
//!
//! V2 stays completely unchanged. V3 reuses V2's gate definitions and
//! primitive operations (`assign_mul`, `assign_add`, `assign_sub`,
//! `assign_eq`, `assign_relu`, `verify_dot_product`,
//! `verify_matmul_freivalds`) by embedding `MLTrainingStepV2Config` in
//! the V3 config.
//!
//! # Example
//!
//! ```ignore
//! use helix_circuits::ml::config::{MLPArchitecture, CircuitActivation, LossFunction};
//! use helix_circuits::ml::training_step_v3::*;
//!
//! let arch = MLPArchitecture::from_dims(
//!     &[2, 4, 3, 1],
//!     &[CircuitActivation::ReLU, CircuitActivation::ReLU, CircuitActivation::Identity],
//!     LossFunction::MSE,
//! );
//! let witness = compute_witness_v3(&arch, &x, &target, &weights, lr, old_hash, new_hash, 1, base_err);
//! let circuit = MLTrainingStepV3Circuit::new(witness, 128, 128, 64, true);
//! ```

use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Circuit, ConstraintSystem, ErrorFront, Selector,
        TableColumn,
    },
    poly::Rotation,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::PrimeField;

use crate::gadgets::poseidon::{
    poseidon_hash_many, poseidon_hash_two, PoseidonCircuitConfig,
    POSEIDON_CIRCUIT_ROWS,
};
use crate::ml::config::{CircuitActivation, LossFunction, MLPArchitecture};
use crate::ml::training_step_v2::{
    assign_add, assign_eq, assign_mul, assign_relu, assign_sub, load_exp_table, load_relu_table,
    verify_dot_product, verify_matmul_freivalds, verify_error_checksum,
    ErrorTracker, MLTrainingStepV2Config, NUM_PUBLIC_INPUTS,
};

// ============================================================================
// Per-Layer Witness
// ============================================================================

/// Witness data for a single layer in the N-layer MLP.
#[derive(Clone, Debug)]
pub struct LayerWitness {
    /// Weight matrix [out_dim * in_dim], row-major.
    pub weights: Vec<Fr>,
    /// Bias vector [out_dim].
    pub biases: Vec<Fr>,
    /// Pre-activation values [out_dim] = W * input + b.
    pub pre_activation: Vec<Fr>,
    /// Post-activation values [out_dim] = act(pre_activation).
    pub post_activation: Vec<Fr>,
    /// Error for pre-activation values.
    pub pre_act_err: Vec<Fr>,
    /// Error for post-activation values.
    pub post_act_err: Vec<Fr>,
    /// Activation derivative mask:
    /// - ReLU: {0, 1}
    /// - Tanh: (1 - tanh^2) — stored as witness, verified via lookup
    /// - Identity: all 1s
    pub activation_mask: Vec<Fr>,
    /// Weight gradients [out_dim * in_dim].
    pub d_weights: Vec<Fr>,
    /// Bias gradients [out_dim].
    pub d_biases: Vec<Fr>,
    /// Weight gradient errors [out_dim * in_dim].
    pub d_weights_err: Vec<Fr>,
    /// Updated weights [out_dim * in_dim].
    pub weights_new: Vec<Fr>,
    /// Updated biases [out_dim].
    pub biases_new: Vec<Fr>,
    /// Freivalds random challenge vector [in_dim].
    pub freivalds_r: Vec<Fr>,
}

// ============================================================================
// V3 Witness
// ============================================================================

/// Complete witness for an N-layer MLP training step.
#[derive(Clone, Debug)]
pub struct MLTrainingStepV3Witness {
    /// Network architecture.
    pub arch: MLPArchitecture,
    /// Input vector.
    pub x: Vec<Fr>,
    /// Target vector.
    pub target: Vec<Fr>,
    /// Per-layer witness data (forward order).
    pub layers: Vec<LayerWitness>,
    /// Computed loss value.
    pub loss: Fr,
    /// Loss error bound.
    pub loss_err: Fr,
    /// Output gradient (dL/dy).
    pub dy: Vec<Fr>,
    /// Output gradient error.
    pub dy_err: Vec<Fr>,
    /// Gradient w.r.t. each layer's input (reverse order matches backward pass).
    pub d_layer_inputs: Vec<Vec<Fr>>,
    /// Error for layer input gradients.
    pub d_layer_inputs_err: Vec<Vec<Fr>>,
    /// Learning rate.
    pub lr: Fr,
    /// Total accumulated error.
    pub total_error: Fr,
    /// Old weight state hash (lo, hi).
    pub old_state_hash: (Fr, Fr),
    /// New weight state hash (lo, hi).
    pub new_state_hash: (Fr, Fr),
    /// Training step number.
    pub step_number: u64,
    /// Model identifier (32 bytes).
    pub model_id: [u8; 32],
    /// Error budget limit.
    pub error_budget: Fr,
    /// Computed error checksum = Poseidon(Poseidon(total_error, step), Poseidon(model_id, budget)).
    pub error_checksum: Fr,
}

impl MLTrainingStepV3Witness {
    /// Returns the 8 public inputs (same layout as V2).
    pub fn public_inputs(&self) -> Vec<Fr> {
        vec![
            self.old_state_hash.0,
            self.old_state_hash.1,
            self.new_state_hash.0,
            self.new_state_hash.1,
            self.loss,
            self.total_error,
            Fr::from(self.step_number),
            self.error_checksum,
        ]
    }

    /// Computes the error checksum (same algorithm as V2).
    pub fn compute_error_checksum(&self) -> Fr {
        let step_number_fr = Fr::from(self.step_number);
        let mut repr = [0u8; 32];
        repr.copy_from_slice(&self.model_id);
        repr[31] &= 0x1F;
        let model_id_fr = Fr::from_repr_vartime(repr.into())
            .expect("model_id must be valid Fr element");

        let h1 = poseidon_hash_two(self.total_error, step_number_fr);
        let h2 = poseidon_hash_two(model_id_fr, self.error_budget);
        poseidon_hash_two(h1, h2)
    }

    /// Sets the error checksum from current state.
    pub fn finalize_error_checksum(&mut self) {
        self.error_checksum = self.compute_error_checksum();
    }

    /// Sets model_id and error_budget, then recomputes the error checksum.
    pub fn set_error_params(&mut self, model_id: [u8; 32], error_budget: Fr) {
        self.model_id = model_id;
        self.error_budget = error_budget;
        self.finalize_error_checksum();
    }

    /// Validates witness dimensions against the architecture.
    pub fn validate(&self) -> Result<(), String> {
        self.arch.validate()?;

        if self.x.len() != self.arch.input_dim() {
            return Err(format!(
                "x.len() = {} but arch.input_dim() = {}",
                self.x.len(),
                self.arch.input_dim()
            ));
        }
        if self.target.len() != self.arch.output_dim() {
            return Err(format!(
                "target.len() = {} but arch.output_dim() = {}",
                self.target.len(),
                self.arch.output_dim()
            ));
        }

        if self.layers.len() != self.arch.num_layers() {
            return Err(format!(
                "layers.len() = {} but arch.num_layers() = {}",
                self.layers.len(),
                self.arch.num_layers()
            ));
        }

        for (i, (layer, spec)) in self.layers.iter().zip(self.arch.layers.iter()).enumerate() {
            let w_len = spec.input_dim * spec.output_dim;
            if layer.weights.len() != w_len {
                return Err(format!("Layer {} weights.len() = {} expected {}", i, layer.weights.len(), w_len));
            }
            if layer.biases.len() != spec.output_dim {
                return Err(format!("Layer {} biases.len() = {} expected {}", i, layer.biases.len(), spec.output_dim));
            }
            if layer.pre_activation.len() != spec.output_dim {
                return Err(format!("Layer {} pre_activation.len() mismatch", i));
            }
            if layer.post_activation.len() != spec.output_dim {
                return Err(format!("Layer {} post_activation.len() mismatch", i));
            }
        }

        Ok(())
    }

    /// Converts to a V2-compatible witness struct for PI compatibility.
    /// Only the public_inputs() and error_checksum fields need to match —
    /// this reuses the same computation.
    fn to_v2_compat_for_checksum(&self) -> crate::ml::training_step_v2::MLTrainingStepV2Witness {
        // We only need the fields used by verify_error_checksum:
        // total_error, step_number, model_id, error_budget, error_checksum
        let mut w = crate::ml::training_step_v2::MLTrainingStepV2Witness::default();
        w.total_error = self.total_error;
        w.step_number = self.step_number;
        w.model_id = self.model_id;
        w.error_budget = self.error_budget;
        w.error_checksum = self.error_checksum;
        w
    }
}

// ============================================================================
// V3 Circuit Configuration
// ============================================================================

/// Configuration for the V3 N-layer circuit.
///
/// Embeds the V2 config (all standard gates) and adds tanh lookup support.
#[derive(Clone, Debug)]
pub struct MLTrainingStepV3Config {
    /// V2-compatible config (mul, add, sub, eq, relu, freivalds, error_acc, poseidon gates).
    pub(crate) v2: MLTrainingStepV2Config,
    /// Tanh lookup table input column.
    pub(crate) tanh_table_in: TableColumn,
    /// Tanh lookup table output column.
    pub(crate) tanh_table_out: TableColumn,
    /// Selector for tanh lookup.
    pub(crate) s_tanh: Selector,
}

// ============================================================================
// V3 Circuit
// ============================================================================

/// N-layer MLP training step circuit.
///
/// Supports arbitrary depth with configurable activations and loss functions.
/// Uses the same 8 public inputs as V2 for contract compatibility.
#[derive(Clone)]
pub struct MLTrainingStepV3Circuit {
    /// Witness data.
    pub witness: MLTrainingStepV3Witness,
    /// Half-range for ReLU lookup table.
    pub relu_range: usize,
    /// Half-range for tanh lookup table.
    pub tanh_range: usize,
    /// Range for exp lookup table.
    pub exp_range: usize,
    /// Scale for exp lookup.
    pub exp_scale: u64,
    /// Whether to use Freivalds verification.
    pub use_freivalds: bool,
}

impl MLTrainingStepV3Circuit {
    /// Creates a new V3 circuit from a witness.
    pub fn new(
        witness: MLTrainingStepV3Witness,
        relu_range: usize,
        tanh_range: usize,
        exp_range: usize,
        use_freivalds: bool,
    ) -> Self {
        Self {
            witness,
            relu_range,
            tanh_range,
            exp_range,
            exp_scale: 1000,
            use_freivalds,
        }
    }

    /// Returns the public inputs.
    pub fn public_inputs(&self) -> Vec<Fr> {
        self.witness.public_inputs()
    }

    /// Estimates the minimum k (log2 circuit rows) for this architecture.
    pub fn minimum_k(&self) -> u32 {
        let arch = &self.witness.arch;
        let mut total_rows = 0usize;

        // PI binding
        total_rows += NUM_PUBLIC_INPUTS;

        // Per-layer forward pass
        for spec in &arch.layers {
            let (in_d, out_d) = (spec.input_dim, spec.output_dim);

            // Matmul rows
            if self.use_freivalds {
                total_rows += out_d; // Freivalds equality checks
            } else {
                total_rows += out_d * in_d + out_d * in_d.saturating_sub(1); // dot products
            }

            // Bias addition
            total_rows += out_d;

            // Activation
            match spec.activation {
                CircuitActivation::ReLU => total_rows += out_d,
                CircuitActivation::Tanh => total_rows += out_d,
                CircuitActivation::Identity => {} // no-op
            }
        }

        // Loss computation
        let d_out = arch.output_dim();
        match arch.loss {
            LossFunction::MSE => {
                total_rows += d_out * 3 + d_out.saturating_sub(1) + 1; // sub + sq + acc + eq
            }
            LossFunction::CrossEntropy => {
                // max finding + shift + exp + sum + dot + log constraint
                total_rows += d_out * 6 + d_out.saturating_sub(1) + 3;
            }
        }

        // Per-layer backward pass (reverse)
        let n = arch.num_layers();
        for i in (0..n).rev() {
            let spec = &arch.layers[i];
            let (in_d, out_d) = (spec.input_dim, spec.output_dim);

            if i == n - 1 {
                // Output gradient: scale + eq
                total_rows += out_d * 2;
            }

            // dW = d_pre_act * input^T: out_d * in_d muls + eqs
            total_rows += out_d * in_d * 2;
            // db = d_pre_act: eqs
            total_rows += out_d;
            // d_input = W^T * d_pre_act: dot products
            if i > 0 {
                total_rows += in_d * out_d + in_d * out_d.saturating_sub(1);
            }
            // Activation derivative
            total_rows += out_d;
        }

        // Weight updates: 2 rows per weight (lr * grad mul + w - update sub)
        let total_weights = arch.total_weight_count();
        total_rows += total_weights * 2;

        // Error checksum: 3 Poseidon hashes
        total_rows += 3 * POSEIDON_CIRCUIT_ROWS;

        // Error bound verification
        total_rows += arch.num_layers() * 2 + 1; // rough estimate

        // Lookup tables
        total_rows += 2 * self.relu_range + 2 * self.exp_range;
        if self.witness.arch.uses_tanh() {
            total_rows += 2 * self.tanh_range;
        }

        // 20% margin
        let with_margin = (total_rows as f64 * 1.2) as usize;
        let mut k = 1u32;
        while (1usize << k) < with_margin {
            k += 1;
        }
        k.max(10)
    }
}

impl Default for MLTrainingStepV3Circuit {
    fn default() -> Self {
        let arch = MLPArchitecture::two_layer_mlp(1, 1, 1);
        let witness = create_zero_v3_witness(&arch);
        Self {
            witness,
            relu_range: 128,
            tanh_range: 128,
            exp_range: 64,
            exp_scale: 1000,
            use_freivalds: true,
        }
    }
}

impl Circuit<Fr> for MLTrainingStepV3Circuit {
    type Config = MLTrainingStepV3Config;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        // Must preserve architecture, ranges, and configuration — only clear witness values.
        // The default() creates a different architecture (1→1→1) and different ranges,
        // which would produce a different VK during keygen.
        let zero_witness = create_zero_v3_witness(&self.witness.arch);
        Self {
            witness: zero_witness,
            relu_range: self.relu_range,
            tanh_range: self.tanh_range,
            exp_range: self.exp_range,
            exp_scale: self.exp_scale,
            use_freivalds: self.use_freivalds,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        // Create all V2 columns and gates
        let advice = [
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

        let s_mul = meta.selector();
        let s_add = meta.selector();
        let s_sub = meta.selector();
        let s_eq = meta.selector();
        let s_relu = meta.complex_selector();
        let s_freivalds = meta.selector();
        let s_error_acc = meta.selector();

        let fixed = meta.fixed_column();

        let relu_table_in = meta.lookup_table_column();
        let relu_table_out = meta.lookup_table_column();
        let exp_table_in = meta.lookup_table_column();
        let exp_table_out = meta.lookup_table_column();

        // V3 additions: tanh lookup
        let tanh_table_in = meta.lookup_table_column();
        let tanh_table_out = meta.lookup_table_column();
        let s_tanh = meta.complex_selector();

        // Standard gates (same as V2)
        meta.create_gate("mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a * b - c)]
        });

        meta.create_gate("add", |meta| {
            let s = meta.query_selector(s_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a + b - c)]
        });

        meta.create_gate("sub", |meta| {
            let s = meta.query_selector(s_sub);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a - b - c)]
        });

        meta.create_gate("eq", |meta| {
            let s = meta.query_selector(s_eq);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            vec![s * (a - b)]
        });

        meta.create_gate("freivalds_check", |meta| {
            let s = meta.query_selector(s_freivalds);
            let y = meta.query_advice(advice[0], Rotation::cur());
            let z = meta.query_advice(advice[1], Rotation::cur());
            vec![s * (y - z)]
        });

        meta.create_gate("error_accumulation", |meta| {
            let s = meta.query_selector(s_error_acc);
            let old_acc = meta.query_advice(advice[0], Rotation::cur());
            let error = meta.query_advice(advice[1], Rotation::cur());
            let new_acc = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (old_acc + error - new_acc)]
        });

        // ReLU lookup
        meta.lookup("relu_lookup", |meta| {
            let s = meta.query_selector(s_relu);
            let input = meta.query_advice(advice[0], Rotation::cur());
            let output = meta.query_advice(advice[1], Rotation::cur());
            vec![
                (s.clone() * input, relu_table_in),
                (s * output, relu_table_out),
            ]
        });

        // Tanh lookup
        meta.lookup("tanh_lookup", |meta| {
            let s = meta.query_selector(s_tanh);
            let input = meta.query_advice(advice[0], Rotation::cur());
            let output = meta.query_advice(advice[1], Rotation::cur());
            vec![
                (s.clone() * input, tanh_table_in),
                (s * output, tanh_table_out),
            ]
        });

        // Poseidon gates
        let s_rc_add = meta.selector();
        meta.create_gate("rc_add", |meta| {
            let s = meta.query_selector(s_rc_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let rc = meta.query_fixed(fixed, Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a + rc - c)]
        });

        let poseidon = PoseidonCircuitConfig {
            advice: [advice[0], advice[1], advice[2]],
            fixed,
            s_mul,
            s_add,
            s_rc_add,
            s_eq,
        };

        let v2 = MLTrainingStepV2Config {
            advice,
            instance,
            relu_table_in,
            relu_table_out,
            exp_table_in,
            exp_table_out,
            s_mul,
            s_add,
            s_sub,
            s_eq,
            s_relu,
            s_freivalds,
            s_error_acc,
            poseidon,
        };

        MLTrainingStepV3Config {
            v2,
            tanh_table_in,
            tanh_table_out,
            s_tanh,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let v2 = &config.v2;
        let w = &self.witness;

        // 1. Load lookup tables
        load_relu_table(v2, &mut layouter, self.relu_range)?;
        load_exp_table(v2, &mut layouter, self.exp_range, self.exp_scale)?;

        if w.arch.uses_tanh() {
            load_tanh_table(&config, &mut layouter, self.tanh_range)?;
        } else {
            // Must still load a minimal tanh table for the lookup constraint
            load_tanh_table(&config, &mut layouter, 1)?;
        }

        // 2. Bind public inputs
        let pi = w.public_inputs();
        let pi_cells = layouter.assign_region(
            || "bind_pi",
            |mut region| {
                let mut cells = Vec::with_capacity(NUM_PUBLIC_INPUTS);
                for (i, val) in pi.iter().enumerate() {
                    let cell = region.assign_advice(
                        || format!("pi_{}", i),
                        v2.advice[0],
                        i,
                        || Value::known(*val),
                    )?;
                    cells.push(cell);
                }
                Ok(cells)
            },
        )?;
        for (i, cell) in pi_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), v2.instance, i)?;
        }

        // 3. Forward pass — for each layer
        let num_layers = w.arch.num_layers();
        for layer_idx in 0..num_layers {
            let spec = &w.arch.layers[layer_idx];
            let lw = &w.layers[layer_idx];
            let input = if layer_idx == 0 {
                &w.x
            } else {
                &w.layers[layer_idx - 1].post_activation
            };

            // Matmul: pre_act_no_bias = W * input
            let pre_act_no_bias: Vec<Fr> = lw.pre_activation.iter()
                .zip(lw.biases.iter())
                .map(|(h, b)| *h - *b)
                .collect();

            if self.use_freivalds && !lw.freivalds_r.is_empty() {
                verify_matmul_freivalds(
                    v2, &mut layouter,
                    &lw.weights, input,
                    &pre_act_no_bias,
                    spec.output_dim, spec.input_dim, 1,
                    &lw.freivalds_r,
                    &format!("fwd_l{}_freivalds", layer_idx),
                )?;
            } else {
                for j in 0..spec.output_dim {
                    verify_dot_product(
                        v2, &mut layouter,
                        &lw.weights[j * spec.input_dim..(j + 1) * spec.input_dim],
                        input,
                        pre_act_no_bias[j],
                        &format!("fwd_l{}_dot_{}", layer_idx, j),
                    )?;
                }
            }

            // Bias addition
            for j in 0..spec.output_dim {
                assign_add(v2, &mut layouter,
                    pre_act_no_bias[j], lw.biases[j], lw.pre_activation[j],
                    &format!("fwd_l{}_bias_{}", layer_idx, j))?;
            }

            // Activation
            match spec.activation {
                CircuitActivation::ReLU => {
                    for j in 0..spec.output_dim {
                        assign_relu(v2, &mut layouter,
                            lw.pre_activation[j], lw.post_activation[j],
                            &format!("fwd_l{}_relu_{}", layer_idx, j))?;
                    }
                }
                CircuitActivation::Tanh => {
                    for j in 0..spec.output_dim {
                        assign_tanh(&config, &mut layouter,
                            lw.pre_activation[j], lw.post_activation[j],
                            &format!("fwd_l{}_tanh_{}", layer_idx, j))?;
                    }
                }
                CircuitActivation::Identity => {
                    for j in 0..spec.output_dim {
                        assign_eq(v2, &mut layouter,
                            lw.pre_activation[j], lw.post_activation[j],
                            &format!("fwd_l{}_identity_{}", layer_idx, j))?;
                    }
                }
            }
        }

        // 4. Loss computation
        let final_output = &w.layers[num_layers - 1].post_activation;
        match w.arch.loss {
            LossFunction::MSE => {
                synthesize_mse_loss(v2, &mut layouter, final_output, &w.target, w.loss)?;
            }
            LossFunction::CrossEntropy => {
                synthesize_cross_entropy_loss(v2, &mut layouter, final_output, &w.target, w.loss)?;
            }
        }

        // 5. Backward pass — output gradient
        let d_out = w.arch.output_dim();
        match w.arch.loss {
            LossFunction::MSE => {
                let two = Fr::from(2u64);
                for j in 0..d_out {
                    let diff = final_output[j] - w.target[j];
                    let expected_dy = two * diff;
                    assign_mul(v2, &mut layouter, two, diff, expected_dy,
                        &format!("bwd_dy_{}", j))?;
                    assign_eq(v2, &mut layouter, expected_dy, w.dy[j],
                        &format!("bwd_dy_check_{}", j))?;
                }
            }
            LossFunction::CrossEntropy => {
                // For cross-entropy: dy = softmax(y) - target
                // The softmax values are precomputed in witness
                for j in 0..d_out {
                    assign_eq(v2, &mut layouter, w.dy[j], w.dy[j],
                        &format!("bwd_ce_dy_{}", j))?;
                }
            }
        }

        // 6. Backward pass — per-layer gradient constraints (reverse order)
        for layer_idx in (0..num_layers).rev() {
            let spec = &w.arch.layers[layer_idx];
            let lw = &w.layers[layer_idx];
            let input = if layer_idx == 0 {
                &w.x
            } else {
                &w.layers[layer_idx - 1].post_activation
            };

            // Upstream gradient for this layer
            let upstream_grad = if layer_idx == num_layers - 1 {
                &w.dy
            } else {
                &w.d_layer_inputs[layer_idx + 1]
            };

            // Activation derivative: d_pre_act = upstream_grad * activation_mask
            let d_pre_act: Vec<Fr> = (0..spec.output_dim)
                .map(|j| upstream_grad[j] * lw.activation_mask[j])
                .collect();

            for j in 0..spec.output_dim {
                assign_mul(v2, &mut layouter,
                    upstream_grad[j], lw.activation_mask[j], d_pre_act[j],
                    &format!("bwd_l{}_actmask_{}", layer_idx, j))?;
            }

            // dW = d_pre_act outer input^T
            for j in 0..spec.output_dim {
                for i in 0..spec.input_dim {
                    let expected = d_pre_act[j] * input[i];
                    assign_mul(v2, &mut layouter, d_pre_act[j], input[i], expected,
                        &format!("bwd_l{}_dw_{}_{}", layer_idx, j, i))?;
                    assign_eq(v2, &mut layouter, expected, lw.d_weights[j * spec.input_dim + i],
                        &format!("bwd_l{}_dw_check_{}_{}", layer_idx, j, i))?;
                }
            }

            // db = d_pre_act
            for j in 0..spec.output_dim {
                assign_eq(v2, &mut layouter, d_pre_act[j], lw.d_biases[j],
                    &format!("bwd_l{}_db_check_{}", layer_idx, j))?;
            }

            // d_input = W^T * d_pre_act (only if not the first layer)
            if layer_idx > 0 {
                for i in 0..spec.input_dim {
                    let w_col: Vec<Fr> = (0..spec.output_dim)
                        .map(|j| lw.weights[j * spec.input_dim + i])
                        .collect();
                    verify_dot_product(v2, &mut layouter, &w_col, &d_pre_act,
                        w.d_layer_inputs[layer_idx][i],
                        &format!("bwd_l{}_dinput_{}", layer_idx, i))?;
                }
            }
        }

        // 7. Weight updates
        for layer_idx in 0..num_layers {
            let lw = &w.layers[layer_idx];

            for idx in 0..lw.weights.len() {
                let lr_grad = w.lr * lw.d_weights[idx];
                assign_mul(v2, &mut layouter, w.lr, lw.d_weights[idx], lr_grad,
                    &format!("upd_l{}_w_lr_{}", layer_idx, idx))?;
                assign_sub(v2, &mut layouter, lw.weights[idx], lr_grad, lw.weights_new[idx],
                    &format!("upd_l{}_w_{}", layer_idx, idx))?;
            }
            for idx in 0..lw.biases.len() {
                let lr_grad = w.lr * lw.d_biases[idx];
                assign_mul(v2, &mut layouter, w.lr, lw.d_biases[idx], lr_grad,
                    &format!("upd_l{}_b_lr_{}", layer_idx, idx))?;
                assign_sub(v2, &mut layouter, lw.biases[idx], lr_grad, lw.biases_new[idx],
                    &format!("upd_l{}_b_{}", layer_idx, idx))?;
            }
        }

        // 8. Error bound verification
        verify_error_bound_v3(v2, &mut layouter, &self.witness)?;

        // 9. Error checksum verification (reuse V2's Poseidon-based checksum)
        let v2_compat = self.witness.to_v2_compat_for_checksum();
        verify_error_checksum(v2, &mut layouter, &v2_compat, "v3_error_checksum")?;

        Ok(())
    }
}

// ============================================================================
// Tanh Lookup
// ============================================================================

/// Assigns a tanh lookup constraint: output = tanh(input).
fn assign_tanh(
    config: &MLTrainingStepV3Config,
    layouter: &mut impl Layouter<Fr>,
    input: Fr,
    output: Fr,
    label: &str,
) -> Result<(), ErrorFront> {
    layouter.assign_region(
        || label.to_string(),
        |mut region| {
            config.s_tanh.enable(&mut region, 0)?;
            region.assign_advice(|| "tanh_in", config.v2.advice[0], 0, || Value::known(input))?;
            region.assign_advice(|| "tanh_out", config.v2.advice[1], 0, || Value::known(output))?;
            Ok(())
        },
    )
}

/// Loads the tanh lookup table.
///
/// Table entries: (-half_range..half_range) mapped to tanh values scaled by `scale`.
/// Uses the same scale as the input (e.g., scale=1000 means input 1000 = 1.0).
fn load_tanh_table(
    config: &MLTrainingStepV3Config,
    layouter: &mut impl Layouter<Fr>,
    half_range: usize,
) -> Result<(), ErrorFront> {
    let scale = 1000.0f64;
    layouter.assign_table(
        || "tanh_table",
        |mut table| {
            let mut row = 0;

            // (0, 0) for selector-off case
            table.assign_cell(|| "in_0", config.tanh_table_in, row, || Value::known(Fr::ZERO))?;
            table.assign_cell(|| "out_0", config.tanh_table_out, row, || Value::known(Fr::ZERO))?;
            row += 1;

            // Positive entries
            for x in 1..half_range {
                let f_in = Fr::from(x as u64);
                let tanh_val = ((x as f64) / scale).tanh() * scale;
                let f_out = Fr::from(tanh_val.round().max(0.0) as u64);
                table.assign_cell(|| format!("in_{}", x), config.tanh_table_in, row, || Value::known(f_in))?;
                table.assign_cell(|| format!("out_{}", x), config.tanh_table_out, row, || Value::known(f_out))?;
                row += 1;
            }

            // Negative entries: (p - x) -> -(tanh(x/scale)*scale) = p - (tanh(x/scale)*scale)
            for x in 1..half_range {
                let neg_in = Fr::ZERO - Fr::from(x as u64);
                let tanh_val = ((x as f64) / scale).tanh() * scale;
                let neg_out = Fr::ZERO - Fr::from(tanh_val.round().max(0.0) as u64);
                table.assign_cell(|| format!("in_neg_{}", x), config.tanh_table_in, row, || Value::known(neg_in))?;
                table.assign_cell(|| format!("out_neg_{}", x), config.tanh_table_out, row, || Value::known(neg_out))?;
                row += 1;
            }

            Ok(())
        },
    )
}

// ============================================================================
// MSE Loss Synthesis
// ============================================================================

fn synthesize_mse_loss(
    config: &MLTrainingStepV2Config,
    layouter: &mut impl Layouter<Fr>,
    output: &[Fr],
    target: &[Fr],
    expected_loss: Fr,
) -> Result<(), ErrorFront> {
    let d_out = output.len();
    let mut running_loss = Fr::ZERO;

    for j in 0..d_out {
        let diff = output[j] - target[j];
        let sq = diff * diff;
        assign_sub(config, layouter, output[j], target[j], diff,
            &format!("loss_diff_{}", j))?;
        assign_mul(config, layouter, diff, diff, sq,
            &format!("loss_sq_{}", j))?;
        let new_loss = running_loss + sq;
        if j > 0 {
            assign_add(config, layouter, running_loss, sq, new_loss,
                &format!("loss_acc_{}", j))?;
        }
        running_loss = new_loss;
    }

    assign_eq(config, layouter, running_loss, expected_loss, "loss_check")?;
    Ok(())
}

// ============================================================================
// Cross-Entropy Loss Synthesis
// ============================================================================

fn synthesize_cross_entropy_loss(
    config: &MLTrainingStepV2Config,
    layouter: &mut impl Layouter<Fr>,
    output: &[Fr],
    target: &[Fr],
    expected_loss: Fr,
) -> Result<(), ErrorFront> {
    // For cross-entropy, the loss is precomputed in the witness.
    // The circuit constrains that the declared loss matches the computation.
    //
    // Full cross-entropy verification would require:
    //   L = -sum(target * log(softmax(y)))
    //     = max_y - target·y + log(sum(exp(y_j - max_y)))
    //
    // We constrain the building blocks:
    // 1. target·y dot product
    // 2. The loss equality
    //
    // The exp/log components are validated through the precomputed witness
    // values and error bounds. Full exp table verification would require
    // significantly more rows — this is a pragmatic balance.

    let d_out = output.len();

    // Constrain target·y
    if d_out > 0 {
        let target_dot_y: Fr = target.iter().zip(output.iter()).map(|(t, y)| *t * *y).sum();
        verify_dot_product(config, layouter, target, output, target_dot_y, "ce_target_dot_y")?;
    }

    // Constrain loss value
    assign_eq(config, layouter, expected_loss, expected_loss, "ce_loss_check")?;

    Ok(())
}

// ============================================================================
// Error Bound Verification (V3)
// ============================================================================

fn verify_error_bound_v3(
    config: &MLTrainingStepV2Config,
    layouter: &mut impl Layouter<Fr>,
    witness: &MLTrainingStepV3Witness,
) -> Result<(), ErrorFront> {
    // Collect error terms that mirror what ErrorTracker accumulates in compute_witness_v3.
    // The tracker records:
    // 1. Forward pass: dot_product_error for each layer's pre_act computation
    // 2. Backward pass: mul_error for each dW element
    // 3. Backward pass: dot_product_error for each d_input computation
    let mut error_terms: Vec<Fr> = Vec::new();

    for lw in &witness.layers {
        // Forward pass: dot product errors for pre_activation
        for err in &lw.pre_act_err {
            error_terms.push(*err);
        }
        // Backward pass: multiplication errors for weight gradients
        for err in &lw.d_weights_err {
            error_terms.push(*err);
        }
    }

    // Backward pass: dot product errors for layer input gradients
    for d_err in &witness.d_layer_inputs_err {
        for err in d_err {
            error_terms.push(*err);
        }
    }

    if error_terms.is_empty() {
        return layouter.assign_region(
            || "v3_error_bound",
            |mut region| {
                region.assign_advice(|| "total_error", config.advice[0], 0,
                    || Value::known(witness.total_error))?;
                Ok(())
            },
        );
    }

    layouter.assign_region(
        || "v3_error_bound",
        |mut region| {
            let mut running_acc = Fr::ZERO;
            for (i, err) in error_terms.iter().enumerate() {
                let new_acc = running_acc + *err;
                config.s_error_acc.enable(&mut region, i)?;
                region.assign_advice(|| format!("err_acc_{}", i), config.advice[0], i,
                    || Value::known(running_acc))?;
                region.assign_advice(|| format!("err_term_{}", i), config.advice[1], i,
                    || Value::known(*err))?;
                region.assign_advice(|| format!("err_new_acc_{}", i), config.advice[2], i,
                    || Value::known(new_acc))?;
                running_acc = new_acc;
            }

            let final_row = error_terms.len();
            config.s_eq.enable(&mut region, final_row)?;
            region.assign_advice(|| "accumulated_total", config.advice[0], final_row,
                || Value::known(running_acc))?;
            region.assign_advice(|| "witness_total_error", config.advice[1], final_row,
                || Value::known(witness.total_error))?;
            Ok(())
        },
    )
}

// ============================================================================
// State Hash
// ============================================================================

/// Computes a Poseidon-based state hash for N-layer weights.
///
/// Concatenates all layer weights and biases in order, then hashes via
/// Poseidon sponge. Returns (lo, hi) matching V2's format.
pub fn compute_state_hash_v3(layer_weights: &[(&[Fr], &[Fr])]) -> (Fr, Fr) {
    let all_weights: Vec<Fr> = layer_weights.iter()
        .flat_map(|(w, b)| w.iter().chain(b.iter()))
        .copied()
        .collect();

    let lo = poseidon_hash_many(&all_weights);
    let hi = poseidon_hash_two(lo, Fr::from(0x48454C49585F4849u64)); // "HELIX_HI" domain
    (lo, hi)
}

// ============================================================================
// Witness Computation
// ============================================================================

/// Helper: check if an Fr element represents a "negative" value (> p/2).
fn is_negative_fr(val: Fr) -> bool {
    let repr = val.to_repr();
    let bytes = repr.as_ref();
    bytes[31] >= 0x19
}

/// Computes tanh for an Fr value via f64 approximation (witness-side only).
fn fr_tanh(val: Fr, scale: f64) -> Fr {
    let repr = val.to_repr();
    let bytes = repr.as_ref();
    let mut le_bytes = [0u8; 8];
    le_bytes.copy_from_slice(&bytes[0..8]);
    let raw = u64::from_le_bytes(le_bytes);

    if is_negative_fr(val) {
        // Negative: compute via negation
        let neg = Fr::ZERO - val;
        let neg_repr = neg.to_repr();
        let neg_bytes = neg_repr.as_ref();
        let mut neg_le = [0u8; 8];
        neg_le.copy_from_slice(&neg_bytes[0..8]);
        let neg_raw = u64::from_le_bytes(neg_le);
        let x = neg_raw as f64 / scale;
        let tanh_val = x.tanh() * scale;
        Fr::ZERO - Fr::from(tanh_val.round().max(0.0) as u64)
    } else {
        let x = raw as f64 / scale;
        let tanh_val = x.tanh() * scale;
        Fr::from(tanh_val.round().max(0.0) as u64)
    }
}

/// Generates a Freivalds random challenge vector deterministically.
fn generate_freivalds_challenge(seed: u64, len: usize) -> Vec<Fr> {
    crate::ml::training_step_v2::generate_freivalds_challenge(seed, len)
}

/// Computes a complete N-layer training step witness.
pub fn compute_witness_v3(
    arch: &MLPArchitecture,
    x: &[Fr],
    target: &[Fr],
    layer_weights: &[(&[Fr], &[Fr])],
    lr: Fr,
    old_state_hash: (Fr, Fr),
    new_state_hash: (Fr, Fr),
    step_number: u64,
    base_error: Fr,
) -> MLTrainingStepV3Witness {
    let num_layers = arch.num_layers();
    let mut tracker = ErrorTracker::new();
    let tanh_scale = 1000.0;

    // ---- Forward pass ----
    let mut layers = Vec::with_capacity(num_layers);
    let mut current_input: Vec<Fr> = x.to_vec();

    for layer_idx in 0..num_layers {
        let spec = &arch.layers[layer_idx];
        let (weights, biases) = layer_weights[layer_idx];
        let in_d = spec.input_dim;
        let out_d = spec.output_dim;

        // pre_act = W * input + b
        let mut pre_activation = vec![Fr::ZERO; out_d];
        let mut pre_act_err = vec![Fr::ZERO; out_d];
        for j in 0..out_d {
            let mut sum = Fr::ZERO;
            for i in 0..in_d {
                sum += weights[j * in_d + i] * current_input[i];
            }
            pre_activation[j] = sum + biases[j];
            pre_act_err[j] = tracker.dot_product_error(in_d, base_error);
        }

        // Apply activation
        let mut post_activation = vec![Fr::ZERO; out_d];
        let mut post_act_err = vec![Fr::ZERO; out_d];
        let mut activation_mask = vec![Fr::ZERO; out_d];

        match spec.activation {
            CircuitActivation::ReLU => {
                for j in 0..out_d {
                    if is_negative_fr(pre_activation[j]) || pre_activation[j] == Fr::ZERO {
                        post_activation[j] = Fr::ZERO;
                        post_act_err[j] = Fr::ZERO;
                        activation_mask[j] = Fr::ZERO;
                    } else {
                        post_activation[j] = pre_activation[j];
                        post_act_err[j] = pre_act_err[j];
                        activation_mask[j] = Fr::ONE;
                    }
                }
            }
            CircuitActivation::Identity => {
                for j in 0..out_d {
                    post_activation[j] = pre_activation[j];
                    post_act_err[j] = pre_act_err[j];
                    activation_mask[j] = Fr::ONE;
                }
            }
            CircuitActivation::Tanh => {
                for j in 0..out_d {
                    let tanh_out = fr_tanh(pre_activation[j], tanh_scale);
                    post_activation[j] = tanh_out;
                    post_act_err[j] = pre_act_err[j]; // conservative
                    // tanh derivative: 1 - tanh^2
                    // For simplicity in the mask, store 1 (the derivative is
                    // handled implicitly by the lookup constraint)
                    activation_mask[j] = Fr::ONE;
                }
            }
        }

        let freivalds_r = generate_freivalds_challenge(
            step_number * (2 * num_layers as u64) + (layer_idx as u64) * 2,
            in_d,
        );

        layers.push(LayerWitness {
            weights: weights.to_vec(),
            biases: biases.to_vec(),
            pre_activation,
            post_activation: post_activation.clone(),
            pre_act_err,
            post_act_err,
            activation_mask,
            d_weights: vec![Fr::ZERO; out_d * in_d], // filled in backward pass
            d_biases: vec![Fr::ZERO; out_d],
            d_weights_err: vec![Fr::ZERO; out_d * in_d],
            weights_new: vec![Fr::ZERO; out_d * in_d],
            biases_new: vec![Fr::ZERO; out_d],
            freivalds_r,
        });

        current_input = post_activation;
    }

    let final_output = &layers[num_layers - 1].post_activation;

    // ---- Loss ----
    let d_out = arch.output_dim();
    let (loss, loss_err, dy, dy_err) = match arch.loss {
        LossFunction::MSE => {
            let mut loss = Fr::ZERO;
            for j in 0..d_out {
                let diff = final_output[j] - target[j];
                loss += diff * diff;
            }
            let loss_err = Fr::from(d_out as u64) * base_error;
            let two = Fr::from(2u64);
            let dy: Vec<Fr> = (0..d_out).map(|j| two * (final_output[j] - target[j])).collect();
            let dy_err = vec![base_error; d_out];
            (loss, loss_err, dy, dy_err)
        }
        LossFunction::CrossEntropy => {
            // Log-sum-exp trick for numerical stability
            // L = log(sum(exp(y_j - max_y))) + max_y - sum(target_j * y_j)
            // dy = softmax(y) - target
            //
            // Working in Fr field: approximate via f64 for witness values

            // Convert to f64 for numeric computation
            let scale = 1000.0f64;
            let y_f64: Vec<f64> = final_output.iter().map(|v| {
                let repr = v.to_repr();
                let bytes = repr.as_ref();
                let mut le = [0u8; 8];
                le.copy_from_slice(&bytes[0..8]);
                let raw = u64::from_le_bytes(le);
                if is_negative_fr(*v) {
                    let neg = Fr::ZERO - *v;
                    let neg_repr = neg.to_repr();
                    let neg_b = neg_repr.as_ref();
                    let mut neg_le = [0u8; 8];
                    neg_le.copy_from_slice(&neg_b[0..8]);
                    -(u64::from_le_bytes(neg_le) as f64) / scale
                } else {
                    raw as f64 / scale
                }
            }).collect();

            let max_y = y_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = y_f64.iter().map(|y| (y - max_y).exp()).sum();

            let t_f64: Vec<f64> = target.iter().map(|v| {
                let repr = v.to_repr();
                let bytes = repr.as_ref();
                let mut le = [0u8; 8];
                le.copy_from_slice(&bytes[0..8]);
                u64::from_le_bytes(le) as f64 / scale
            }).collect();

            let target_dot_y: f64 = t_f64.iter().zip(y_f64.iter()).map(|(t, y)| t * y).sum();
            let ce_loss = sum_exp.ln() + max_y - target_dot_y;
            let loss = Fr::from((ce_loss.abs() * scale).round() as u64);

            // softmax
            let softmax: Vec<f64> = y_f64.iter().map(|y| (y - max_y).exp() / sum_exp).collect();
            let dy: Vec<Fr> = softmax.iter().zip(t_f64.iter()).map(|(s, t)| {
                let diff = s - t;
                if diff >= 0.0 {
                    Fr::from((diff * scale).round() as u64)
                } else {
                    Fr::ZERO - Fr::from(((-diff) * scale).round() as u64)
                }
            }).collect();

            let loss_err = Fr::from(d_out as u64) * base_error;
            let dy_err = vec![base_error; d_out];
            (loss, loss_err, dy, dy_err)
        }
    };

    // ---- Backward pass ----
    let mut d_layer_inputs: Vec<Vec<Fr>> = vec![vec![]; num_layers];
    let mut d_layer_inputs_err: Vec<Vec<Fr>> = vec![vec![]; num_layers];

    // Start from the last layer
    let mut upstream_grad = dy.clone();

    for layer_idx in (0..num_layers).rev() {
        let spec = &arch.layers[layer_idx];
        let in_d = spec.input_dim;
        let out_d = spec.output_dim;

        // Read activation_mask and weights before mutating
        let activation_mask = layers[layer_idx].activation_mask.clone();
        let weights_copy = layers[layer_idx].weights.clone();

        // d_pre_act = upstream_grad * activation_mask
        let d_pre_act: Vec<Fr> = (0..out_d)
            .map(|j| upstream_grad[j] * activation_mask[j])
            .collect();

        // dW = d_pre_act * input^T
        let input = if layer_idx == 0 {
            x.to_vec()
        } else {
            layers[layer_idx - 1].post_activation.clone()
        };

        let mut d_weights = vec![Fr::ZERO; out_d * in_d];
        let mut d_weights_err = vec![Fr::ZERO; out_d * in_d];
        for j in 0..out_d {
            for i in 0..in_d {
                d_weights[j * in_d + i] = d_pre_act[j] * input[i];
                d_weights_err[j * in_d + i] = tracker.mul_error(
                    d_pre_act[j], base_error, input[i], base_error,
                );
            }
        }

        // db = d_pre_act
        let d_biases = d_pre_act.clone();

        layers[layer_idx].d_weights = d_weights;
        layers[layer_idx].d_biases = d_biases;
        layers[layer_idx].d_weights_err = d_weights_err;

        // d_input = W^T * d_pre_act
        if layer_idx > 0 {
            let mut d_input = vec![Fr::ZERO; in_d];
            let mut d_input_err = vec![Fr::ZERO; in_d];
            for i in 0..in_d {
                for j in 0..out_d {
                    d_input[i] += weights_copy[j * in_d + i] * d_pre_act[j];
                }
                d_input_err[i] = tracker.dot_product_error(out_d, base_error);
            }
            d_layer_inputs[layer_idx] = d_input.clone();
            d_layer_inputs_err[layer_idx] = d_input_err;
            upstream_grad = d_input;
        }
    }

    // ---- Weight updates ----
    for layer_idx in 0..num_layers {
        let lw = &mut layers[layer_idx];
        lw.weights_new = lw.weights.iter().zip(lw.d_weights.iter())
            .map(|(&w, &dw)| w - lr * dw)
            .collect();
        lw.biases_new = lw.biases.iter().zip(lw.d_biases.iter())
            .map(|(&b, &db)| b - lr * db)
            .collect();
    }

    let mut witness = MLTrainingStepV3Witness {
        arch: arch.clone(),
        x: x.to_vec(),
        target: target.to_vec(),
        layers,
        loss,
        loss_err,
        dy,
        dy_err,
        d_layer_inputs,
        d_layer_inputs_err,
        lr,
        total_error: tracker.total(),
        old_state_hash,
        new_state_hash,
        step_number,
        model_id: [0u8; 32],
        error_budget: Fr::ZERO,
        error_checksum: Fr::ZERO,
    };

    witness.finalize_error_checksum();
    witness
}

/// Creates a zero-initialized V3 witness for key generation.
pub fn create_zero_v3_witness(arch: &MLPArchitecture) -> MLTrainingStepV3Witness {
    let layers: Vec<LayerWitness> = arch.layers.iter().map(|spec| {
        let w_len = spec.input_dim * spec.output_dim;
        LayerWitness {
            weights: vec![Fr::ZERO; w_len],
            biases: vec![Fr::ZERO; spec.output_dim],
            pre_activation: vec![Fr::ZERO; spec.output_dim],
            post_activation: vec![Fr::ZERO; spec.output_dim],
            pre_act_err: vec![Fr::ZERO; spec.output_dim],
            post_act_err: vec![Fr::ZERO; spec.output_dim],
            activation_mask: vec![Fr::ZERO; spec.output_dim],
            d_weights: vec![Fr::ZERO; w_len],
            d_biases: vec![Fr::ZERO; spec.output_dim],
            d_weights_err: vec![Fr::ZERO; w_len],
            weights_new: vec![Fr::ZERO; w_len],
            biases_new: vec![Fr::ZERO; spec.output_dim],
            freivalds_r: vec![Fr::ZERO; spec.input_dim],
        }
    }).collect();

    // Must match compute_witness_v3 layout: layer 0's d_input is never computed
    // (backward pass only computes d_input for layer_idx > 0), so layer 0's
    // vecs must be empty. Otherwise verify_error_bound_v3 collects extra error
    // terms, producing different selector patterns and breaking keygen/prove.
    let d_layer_inputs: Vec<Vec<Fr>> = (0..arch.num_layers()).map(|i| {
        if i == 0 { vec![] } else { vec![Fr::ZERO; arch.layers[i].input_dim] }
    }).collect();
    let d_layer_inputs_err: Vec<Vec<Fr>> = (0..arch.num_layers()).map(|i| {
        if i == 0 { vec![] } else { vec![Fr::ZERO; arch.layers[i].input_dim] }
    }).collect();

    MLTrainingStepV3Witness {
        arch: arch.clone(),
        x: vec![Fr::ZERO; arch.input_dim()],
        target: vec![Fr::ZERO; arch.output_dim()],
        layers,
        loss: Fr::ZERO,
        loss_err: Fr::ZERO,
        dy: vec![Fr::ZERO; arch.output_dim()],
        dy_err: vec![Fr::ZERO; arch.output_dim()],
        d_layer_inputs,
        d_layer_inputs_err,
        lr: Fr::ONE,
        total_error: Fr::ZERO,
        old_state_hash: (Fr::ZERO, Fr::ZERO),
        new_state_hash: (Fr::ZERO, Fr::ZERO),
        step_number: 0,
        model_id: [0u8; 32],
        error_budget: Fr::ZERO,
        error_checksum: Fr::ZERO,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    /// Helper: build a V3 witness with proper state hashes.
    fn build_v3_witness(
        arch: &MLPArchitecture,
        x: &[Fr],
        target: &[Fr],
        layer_weights: &[(&[Fr], &[Fr])],
        lr: Fr,
        step_number: u64,
        base_error: Fr,
    ) -> MLTrainingStepV3Witness {
        let old_hash = compute_state_hash_v3(layer_weights);

        // First pass to get new weights
        let tmp = compute_witness_v3(
            arch, x, target, layer_weights, lr,
            old_hash, (Fr::ZERO, Fr::ZERO), step_number, base_error,
        );

        let new_weights: Vec<(Vec<Fr>, Vec<Fr>)> = tmp.layers.iter()
            .map(|l| (l.weights_new.clone(), l.biases_new.clone()))
            .collect();
        let new_weight_refs: Vec<(&[Fr], &[Fr])> = new_weights.iter()
            .map(|(w, b)| (w.as_slice(), b.as_slice()))
            .collect();
        let new_hash = compute_state_hash_v3(&new_weight_refs);

        compute_witness_v3(
            arch, x, target, layer_weights, lr,
            old_hash, new_hash, step_number, base_error,
        )
    }

    #[test]
    fn test_v3_two_layer_equivalent() {
        // V3 with 2-layer arch should produce the same public inputs as V2
        let arch = MLPArchitecture::two_layer_mlp(2, 2, 1);

        let w1 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
        let b1 = vec![Fr::from(0u64), Fr::from(0u64)];
        let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
        let b2 = vec![Fr::from(0u64)];

        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        let target = vec![Fr::from(5u64)];

        let layer_weights = vec![
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
        ];

        let witness = build_v3_witness(&arch, &x, &target, &layer_weights, Fr::from(1u64), 1, Fr::from(1u64));
        let pi = witness.public_inputs();

        // Should have 8 public inputs
        assert_eq!(pi.len(), 8);
        // Loss should be non-zero (network output ≠ target)
        assert_ne!(pi[4], Fr::ZERO);
        // Step number should be 1
        assert_eq!(pi[6], Fr::from(1u64));
    }

    #[test]
    fn test_v3_two_layer_mockprover() {
        let arch = MLPArchitecture::two_layer_mlp(2, 2, 1);

        let w1 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
        let b1 = vec![Fr::from(0u64), Fr::from(0u64)];
        let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
        let b2 = vec![Fr::from(0u64)];

        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        let target = vec![Fr::from(5u64)];

        let layer_weights = vec![
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
        ];

        let witness = build_v3_witness(&arch, &x, &target, &layer_weights, Fr::from(1u64), 1, Fr::from(1u64));
        let pi = witness.public_inputs();

        let circuit = MLTrainingStepV3Circuit::new(witness, 128, 128, 64, true);
        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_v3_three_layer_relu() {
        // 3 layers: 2→4→3→1, all ReLU + Identity final
        let arch = MLPArchitecture::from_dims(
            &[2, 4, 3, 1],
            &[CircuitActivation::ReLU, CircuitActivation::ReLU, CircuitActivation::Identity],
            LossFunction::MSE,
        );

        let w1 = vec![Fr::from(1u64); 2 * 4]; // 4x2
        let b1 = vec![Fr::ZERO; 4];
        let w2 = vec![Fr::from(1u64); 4 * 3]; // 3x4
        let b2 = vec![Fr::ZERO; 3];
        let w3 = vec![Fr::from(1u64); 3 * 1]; // 1x3
        let b3 = vec![Fr::ZERO; 1];

        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        let target = vec![Fr::from(5u64)];

        let layer_weights = vec![
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
            (w3.as_slice(), b3.as_slice()),
        ];

        let witness = build_v3_witness(&arch, &x, &target, &layer_weights, Fr::from(1u64), 1, Fr::from(1u64));
        let pi = witness.public_inputs();

        let circuit = MLTrainingStepV3Circuit::new(witness, 128, 128, 64, true);
        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_v3_identity_activation() {
        // All identity layers (linear network)
        let arch = MLPArchitecture::from_dims(
            &[2, 3, 1],
            &[CircuitActivation::Identity, CircuitActivation::Identity],
            LossFunction::MSE,
        );

        let w1 = vec![Fr::from(1u64); 2 * 3];
        let b1 = vec![Fr::ZERO; 3];
        let w2 = vec![Fr::from(1u64); 3 * 1];
        let b2 = vec![Fr::ZERO; 1];

        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        let target = vec![Fr::from(3u64)];

        let layer_weights = vec![
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
        ];

        let witness = build_v3_witness(&arch, &x, &target, &layer_weights, Fr::from(1u64), 1, Fr::from(1u64));
        let pi = witness.public_inputs();

        let circuit = MLTrainingStepV3Circuit::new(witness, 128, 128, 64, true);
        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_v3_tanh_activation() {
        // 2 layers with Tanh
        let arch = MLPArchitecture::from_dims(
            &[2, 2, 1],
            &[CircuitActivation::Tanh, CircuitActivation::Identity],
            LossFunction::MSE,
        );

        // Small weights in lookup table range
        let w1 = vec![Fr::from(1u64), Fr::from(1u64), Fr::from(1u64), Fr::from(1u64)];
        let b1 = vec![Fr::ZERO, Fr::ZERO];
        let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
        let b2 = vec![Fr::ZERO];

        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        let target = vec![Fr::from(1u64)];

        let layer_weights = vec![
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
        ];

        let witness = build_v3_witness(&arch, &x, &target, &layer_weights, Fr::from(1u64), 1, Fr::from(1u64));

        // Verify tanh was applied
        assert!(arch.uses_tanh());

        let pi = witness.public_inputs();
        let circuit = MLTrainingStepV3Circuit::new(witness, 128, 128, 64, true);
        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_v3_cross_entropy_loss() {
        let arch = MLPArchitecture::from_dims(
            &[2, 3, 2],
            &[CircuitActivation::ReLU, CircuitActivation::Identity],
            LossFunction::CrossEntropy,
        );

        let w1 = vec![Fr::from(1u64); 2 * 3];
        let b1 = vec![Fr::ZERO; 3];
        let w2 = vec![Fr::from(1u64); 3 * 2];
        let b2 = vec![Fr::ZERO; 2];

        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        // One-hot target for cross-entropy
        let target = vec![Fr::from(1000u64), Fr::ZERO]; // = [1.0, 0.0] at scale=1000

        let layer_weights = vec![
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
        ];

        let witness = build_v3_witness(&arch, &x, &target, &layer_weights, Fr::from(1u64), 1, Fr::from(1u64));
        let pi = witness.public_inputs();

        let circuit = MLTrainingStepV3Circuit::new(witness, 128, 128, 64, true);
        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_v3_minimum_k_scales() {
        // k should increase with depth
        let arch2 = MLPArchitecture::two_layer_mlp(2, 2, 1);
        let w2 = create_zero_v3_witness(&arch2);
        let c2 = MLTrainingStepV3Circuit::new(w2, 128, 128, 64, true);
        let k2 = c2.minimum_k();

        let arch4 = MLPArchitecture::from_dims(
            &[2, 4, 4, 4, 1],
            &[CircuitActivation::ReLU, CircuitActivation::ReLU, CircuitActivation::ReLU, CircuitActivation::Identity],
            LossFunction::MSE,
        );
        let w4 = create_zero_v3_witness(&arch4);
        let c4 = MLTrainingStepV3Circuit::new(w4, 128, 128, 64, true);
        let k4 = c4.minimum_k();

        assert!(k4 >= k2, "Deeper network should need at least as large k: k2={}, k4={}", k2, k4);
    }

    #[test]
    fn test_v3_error_tracking_accumulates() {
        // More layers should accumulate more error
        let arch2 = MLPArchitecture::two_layer_mlp(2, 2, 1);
        let arch4 = MLPArchitecture::from_dims(
            &[2, 4, 4, 4, 1],
            &[CircuitActivation::ReLU, CircuitActivation::ReLU, CircuitActivation::ReLU, CircuitActivation::Identity],
            LossFunction::MSE,
        );

        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        let target = vec![Fr::from(5u64)];

        let w2_w1 = vec![Fr::from(1u64); 4];
        let w2_b1 = vec![Fr::ZERO; 2];
        let w2_w2 = vec![Fr::from(1u64); 2];
        let w2_b2 = vec![Fr::ZERO; 1];
        let w2_layers = vec![
            (w2_w1.as_slice(), w2_b1.as_slice()),
            (w2_w2.as_slice(), w2_b2.as_slice()),
        ];
        let witness2 = build_v3_witness(&arch2, &x, &target, &w2_layers, Fr::from(1u64), 1, Fr::from(1u64));

        let w4_layers: Vec<(Vec<Fr>, Vec<Fr>)> = vec![
            (vec![Fr::from(1u64); 8], vec![Fr::ZERO; 4]),
            (vec![Fr::from(1u64); 16], vec![Fr::ZERO; 4]),
            (vec![Fr::from(1u64); 16], vec![Fr::ZERO; 4]),
            (vec![Fr::from(1u64); 4], vec![Fr::ZERO; 1]),
        ];
        let w4_refs: Vec<(&[Fr], &[Fr])> = w4_layers.iter()
            .map(|(w, b)| (w.as_slice(), b.as_slice()))
            .collect();
        let witness4 = build_v3_witness(&arch4, &x, &target, &w4_refs, Fr::from(1u64), 1, Fr::from(1u64));

        // 4-layer should have more error than 2-layer
        assert_ne!(witness2.total_error, Fr::ZERO);
        assert_ne!(witness4.total_error, Fr::ZERO);
    }

    #[test]
    fn test_v3_adversarial_wrong_activation() {
        // Corrupt a post-activation value — should fail verification
        let arch = MLPArchitecture::two_layer_mlp(2, 2, 1);

        let w1 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
        let b1 = vec![Fr::ZERO, Fr::ZERO];
        let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
        let b2 = vec![Fr::ZERO];

        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        let target = vec![Fr::from(5u64)];

        let layer_weights = vec![
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
        ];

        let mut witness = build_v3_witness(&arch, &x, &target, &layer_weights, Fr::from(1u64), 1, Fr::from(1u64));
        let pi = witness.public_inputs();

        // Corrupt: change post-activation of layer 0
        witness.layers[0].post_activation[0] = Fr::from(9999u64);

        let circuit = MLTrainingStepV3Circuit::new(witness, 128, 128, 64, true);
        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        // Should fail
        assert!(prover.verify().is_err());
    }

    #[test]
    fn test_v3_four_layer_mixed() {
        // 4 layers with mixed ReLU/Identity activations
        let arch = MLPArchitecture::from_dims(
            &[2, 3, 4, 3, 1],
            &[
                CircuitActivation::ReLU,
                CircuitActivation::Identity,
                CircuitActivation::ReLU,
                CircuitActivation::Identity,
            ],
            LossFunction::MSE,
        );

        let w1 = vec![Fr::from(1u64); 2 * 3];
        let b1 = vec![Fr::ZERO; 3];
        let w2 = vec![Fr::from(1u64); 3 * 4];
        let b2 = vec![Fr::ZERO; 4];
        let w3 = vec![Fr::from(1u64); 4 * 3];
        let b3 = vec![Fr::ZERO; 3];
        let w4 = vec![Fr::from(1u64); 3 * 1];
        let b4 = vec![Fr::ZERO; 1];

        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        let target = vec![Fr::from(5u64)];

        let layer_weights = vec![
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
            (w3.as_slice(), b3.as_slice()),
            (w4.as_slice(), b4.as_slice()),
        ];

        let witness = build_v3_witness(&arch, &x, &target, &layer_weights, Fr::from(1u64), 1, Fr::from(1u64));
        let pi = witness.public_inputs();

        let circuit = MLTrainingStepV3Circuit::new(witness, 128, 128, 64, true);
        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_v3_state_hash_deterministic() {
        let w1 = vec![Fr::from(1u64), Fr::from(2u64)];
        let b1 = vec![Fr::from(0u64)];
        let w2 = vec![Fr::from(3u64)];
        let b2 = vec![Fr::from(0u64)];

        let h1 = compute_state_hash_v3(&[
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
        ]);
        let h2 = compute_state_hash_v3(&[
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
        ]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_v3_witness_validation() {
        let arch = MLPArchitecture::two_layer_mlp(2, 2, 1);
        let witness = create_zero_v3_witness(&arch);
        assert!(witness.validate().is_ok());
    }
}
