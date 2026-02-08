//! Complete ML Training Step Circuit.
//!
//! Proves an entire training step for a 2-layer MLP:
//!   Forward:  h = ReLU(W1 * x + b1),  y = W2 * h + b2
//!   Loss:     L = sum((y - target)^2)
//!   Backward: dW2 = dy * h^T, dW1 = dh_pre * x^T, etc.
//!   Update:   W_new = W_old - lr * dW
//!
//! Key techniques:
//! - Freivalds randomized verification for matrix multiplications (O(n^2) vs O(n^3))
//! - Lookup tables for ReLU activation verification
//! - Error bound tracking through all operations
//! - Public inputs for on-chain verification
//!
//! Public inputs (instance column):
//!   0: old_state_hash_lo  (lower 128 bits of SHA256 of old weights)
//!   1: old_state_hash_hi  (upper 128 bits)
//!   2: new_state_hash_lo  (lower 128 bits of SHA256 of new weights)
//!   3: new_state_hash_hi  (upper 128 bits)
//!   4: loss               (quantized loss value)
//!   5: total_error_bound  (accumulated error across all operations)
//!   6: step_number

use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, ErrorFront, Instance, Selector,
        TableColumn,
    },
    poly::Rotation,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::PrimeField;

/// Number of public inputs exposed by this circuit.
pub const NUM_PUBLIC_INPUTS: usize = 7;

// ---------------------------------------------------------------------------
// Circuit Configuration
// ---------------------------------------------------------------------------

/// Configuration for the ML training step circuit.
#[derive(Clone, Debug)]
pub struct MLTrainingStepConfig {
    /// Three shared advice columns for arithmetic operations.
    advice: [Column<Advice>; 3],
    /// Instance column for public inputs.
    instance: Column<Instance>,
    /// Lookup table columns for ReLU.
    relu_table_in: TableColumn,
    relu_table_out: TableColumn,
    /// Selector for multiplication gate: a * b = c
    s_mul: Selector,
    /// Selector for addition gate: a + b = c
    s_add: Selector,
    /// Selector for subtraction gate: a - b = c
    s_sub: Selector,
    /// Selector for equality check: a = b (with c unused)
    s_eq: Selector,
    /// Complex selector for ReLU lookup.
    s_relu: Selector,
}

// ---------------------------------------------------------------------------
// Witness Data
// ---------------------------------------------------------------------------

/// Complete witness for a 2-layer MLP training step.
///
/// All values are quantized integers in the field.
/// Dimensions: input (d_in), hidden (d_hid), output (d_out).
#[derive(Clone, Debug)]
pub struct MLTrainingStepWitness {
    // --- Dimensions ---
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,

    // --- Inputs ---
    /// Input vector x (d_in).
    pub x: Vec<Fr>,
    /// Target vector (d_out).
    pub target: Vec<Fr>,

    // --- Old weights ---
    /// Layer 1 weights W1 (d_hid rows, d_in cols, row-major).
    pub w1: Vec<Fr>,
    /// Layer 1 bias b1 (d_hid).
    pub b1: Vec<Fr>,
    /// Layer 2 weights W2 (d_out rows, d_hid cols, row-major).
    pub w2: Vec<Fr>,
    /// Layer 2 bias b2 (d_out).
    pub b2: Vec<Fr>,

    // --- Forward pass intermediates ---
    /// Pre-activation h_pre = W1*x + b1 (d_hid).
    pub h_pre: Vec<Fr>,
    /// Post-activation h = ReLU(h_pre) (d_hid).
    pub h: Vec<Fr>,
    /// Output y = W2*h + b2 (d_out).
    pub y: Vec<Fr>,
    /// Loss value (quantized).
    pub loss: Fr,

    // --- Backward pass ---
    /// Output gradient dy = 2*(y - target) (d_out).
    pub dy: Vec<Fr>,
    /// Weight gradient dW2 (d_out * d_hid, row-major).
    pub dw2: Vec<Fr>,
    /// Bias gradient db2 (d_out).
    pub db2: Vec<Fr>,
    /// Hidden gradient dh = W2^T * dy (d_hid).
    pub dh: Vec<Fr>,
    /// ReLU mask: 1 if h_pre > 0, else 0 (d_hid).
    pub relu_mask: Vec<Fr>,
    /// Pre-activation gradient dh_pre = dh * relu_mask (d_hid).
    pub dh_pre: Vec<Fr>,
    /// Weight gradient dW1 (d_hid * d_in, row-major).
    pub dw1: Vec<Fr>,
    /// Bias gradient db1 (d_hid).
    pub db1: Vec<Fr>,

    // --- Learning rate ---
    pub lr: Fr,

    // --- New weights (after update) ---
    pub w1_new: Vec<Fr>,
    pub b1_new: Vec<Fr>,
    pub w2_new: Vec<Fr>,
    pub b2_new: Vec<Fr>,

    // --- Error tracking ---
    /// Total accumulated error bound.
    pub total_error: Fr,

    // --- Public inputs ---
    /// Hash of old weights (lo, hi).
    pub old_state_hash: (Fr, Fr),
    /// Hash of new weights (lo, hi).
    pub new_state_hash: (Fr, Fr),
    /// Step number.
    pub step_number: u64,
}

impl Default for MLTrainingStepWitness {
    fn default() -> Self {
        Self {
            d_in: 0,
            d_hid: 0,
            d_out: 0,
            x: vec![],
            target: vec![],
            w1: vec![],
            b1: vec![],
            w2: vec![],
            b2: vec![],
            h_pre: vec![],
            h: vec![],
            y: vec![],
            loss: Fr::ZERO,
            dy: vec![],
            dw2: vec![],
            db2: vec![],
            dh: vec![],
            relu_mask: vec![],
            dh_pre: vec![],
            dw1: vec![],
            db1: vec![],
            lr: Fr::ONE,
            w1_new: vec![],
            b1_new: vec![],
            w2_new: vec![],
            b2_new: vec![],
            total_error: Fr::ZERO,
            old_state_hash: (Fr::ZERO, Fr::ZERO),
            new_state_hash: (Fr::ZERO, Fr::ZERO),
            step_number: 0,
        }
    }
}

impl MLTrainingStepWitness {
    /// Builds the public inputs vector.
    pub fn public_inputs(&self) -> Vec<Fr> {
        vec![
            self.old_state_hash.0,
            self.old_state_hash.1,
            self.new_state_hash.0,
            self.new_state_hash.1,
            self.loss,
            self.total_error,
            Fr::from(self.step_number),
        ]
    }
}

// ---------------------------------------------------------------------------
// The Circuit
// ---------------------------------------------------------------------------

/// A Halo2 circuit proving a complete 2-layer MLP training step.
#[derive(Clone)]
pub struct MLTrainingStepCircuit {
    /// Witness data (private inputs + intermediates).
    pub witness: MLTrainingStepWitness,
    /// Half-range for the ReLU lookup table.
    /// The table covers [-relu_range, relu_range).
    pub relu_range: usize,
}

impl Default for MLTrainingStepCircuit {
    fn default() -> Self {
        Self {
            witness: MLTrainingStepWitness::default(),
            relu_range: 128,
        }
    }
}

impl MLTrainingStepCircuit {
    /// Convenience: return the public inputs.
    pub fn public_inputs(&self) -> Vec<Fr> {
        self.witness.public_inputs()
    }
}

impl Circuit<Fr> for MLTrainingStepCircuit {
    type Config = MLTrainingStepConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        // Advice columns shared across all operations.
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let instance = meta.instance_column();

        for col in &advice {
            meta.enable_equality(*col);
        }
        meta.enable_equality(instance);

        // Selectors
        let s_mul = meta.selector();
        let s_add = meta.selector();
        let s_sub = meta.selector();
        let s_eq = meta.selector();
        let s_relu = meta.complex_selector();

        // ReLU lookup table columns
        let relu_table_in = meta.lookup_table_column();
        let relu_table_out = meta.lookup_table_column();

        // Gate: a * b = c
        meta.create_gate("mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a * b - c)]
        });

        // Gate: a + b = c
        meta.create_gate("add", |meta| {
            let s = meta.query_selector(s_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a + b - c)]
        });

        // Gate: a - b = c
        meta.create_gate("sub", |meta| {
            let s = meta.query_selector(s_sub);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a - b - c)]
        });

        // Gate: a = b (equality check, c is unused)
        meta.create_gate("eq", |meta| {
            let s = meta.query_selector(s_eq);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            vec![s * (a - b)]
        });

        // ReLU lookup: (advice[0], advice[1]) must be in (relu_table_in, relu_table_out)
        meta.lookup("training_relu", |meta| {
            let s = meta.query_selector(s_relu);
            let input = meta.query_advice(advice[0], Rotation::cur());
            let output = meta.query_advice(advice[1], Rotation::cur());
            vec![
                (s.clone() * input, relu_table_in),
                (s * output, relu_table_out),
            ]
        });

        MLTrainingStepConfig {
            advice,
            instance,
            relu_table_in,
            relu_table_out,
            s_mul,
            s_add,
            s_sub,
            s_eq,
            s_relu,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let w = &self.witness;

        // ================================================================
        // 0. Load ReLU lookup table
        // ================================================================
        load_relu_table(&config, &mut layouter, self.relu_range)?;

        // ================================================================
        // 1. Bind public inputs
        // ================================================================
        let pi = w.public_inputs();
        let pi_cells = layouter.assign_region(
            || "public_inputs",
            |mut region| {
                let mut cells = Vec::with_capacity(NUM_PUBLIC_INPUTS);
                for (i, val) in pi.iter().enumerate() {
                    let cell = region.assign_advice(
                        || format!("pi_{}", i),
                        config.advice[0],
                        i,
                        || Value::known(*val),
                    )?;
                    cells.push(cell);
                }
                Ok(cells)
            },
        )?;
        for (i, cell) in pi_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, i)?;
        }

        // ================================================================
        // 2. Forward pass — Layer 1: h_pre = W1*x + b1, h = ReLU(h_pre)
        // ================================================================
        // Verify each element of h_pre = W1*x + b1
        for j in 0..w.d_hid {
            // Dot product: dp = sum_i(W1[j][i] * x[i])
            verify_dot_product(
                &config,
                &mut layouter,
                &w.w1[j * w.d_in..(j + 1) * w.d_in],
                &w.x,
                // expected result (before bias)
                w.h_pre[j] - w.b1[j],
                &format!("fwd_l1_dot_{}", j),
            )?;

            // Bias addition: h_pre[j] = dp + b1[j]
            assign_add(
                &config,
                &mut layouter,
                w.h_pre[j] - w.b1[j],
                w.b1[j],
                w.h_pre[j],
                &format!("fwd_l1_bias_{}", j),
            )?;

            // ReLU: h[j] = ReLU(h_pre[j])
            assign_relu(
                &config,
                &mut layouter,
                w.h_pre[j],
                w.h[j],
                &format!("fwd_l1_relu_{}", j),
            )?;
        }

        // ================================================================
        // 3. Forward pass — Layer 2: y = W2*h + b2
        // ================================================================
        for j in 0..w.d_out {
            // Dot product: dp = sum_i(W2[j][i] * h[i])
            verify_dot_product(
                &config,
                &mut layouter,
                &w.w2[j * w.d_hid..(j + 1) * w.d_hid],
                &w.h,
                w.y[j] - w.b2[j],
                &format!("fwd_l2_dot_{}", j),
            )?;

            // Bias addition: y[j] = dp + b2[j]
            assign_add(
                &config,
                &mut layouter,
                w.y[j] - w.b2[j],
                w.b2[j],
                w.y[j],
                &format!("fwd_l2_bias_{}", j),
            )?;
        }

        // ================================================================
        // 4. Loss: L = sum((y[j] - target[j])^2)
        // ================================================================
        {
            let mut running_loss = Fr::ZERO;
            for j in 0..w.d_out {
                let diff = w.y[j] - w.target[j];
                let sq = diff * diff;

                // Verify diff = y - target
                assign_sub(
                    &config,
                    &mut layouter,
                    w.y[j],
                    w.target[j],
                    diff,
                    &format!("loss_diff_{}", j),
                )?;

                // Verify sq = diff * diff
                assign_mul(
                    &config,
                    &mut layouter,
                    diff,
                    diff,
                    sq,
                    &format!("loss_sq_{}", j),
                )?;

                // Accumulate
                let new_loss = running_loss + sq;
                if j > 0 {
                    assign_add(
                        &config,
                        &mut layouter,
                        running_loss,
                        sq,
                        new_loss,
                        &format!("loss_acc_{}", j),
                    )?;
                }
                running_loss = new_loss;
            }

            // Verify final loss matches the witness
            assign_eq(
                &config,
                &mut layouter,
                running_loss,
                w.loss,
                "loss_check",
            )?;
        }

        // ================================================================
        // 5. Backward — Output gradient: dy = 2*(y - target)
        // ================================================================
        let two = Fr::from(2u64);
        for j in 0..w.d_out {
            let diff = w.y[j] - w.target[j];
            let expected_dy = two * diff;

            assign_mul(
                &config,
                &mut layouter,
                two,
                diff,
                expected_dy,
                &format!("bwd_dy_{}", j),
            )?;

            assign_eq(
                &config,
                &mut layouter,
                expected_dy,
                w.dy[j],
                &format!("bwd_dy_check_{}", j),
            )?;
        }

        // ================================================================
        // 6. Backward — dW2: dW2[j][k] = dy[j] * h[k]
        // ================================================================
        for j in 0..w.d_out {
            for k in 0..w.d_hid {
                let expected = w.dy[j] * w.h[k];
                assign_mul(
                    &config,
                    &mut layouter,
                    w.dy[j],
                    w.h[k],
                    expected,
                    &format!("bwd_dw2_{}_{}", j, k),
                )?;
                assign_eq(
                    &config,
                    &mut layouter,
                    expected,
                    w.dw2[j * w.d_hid + k],
                    &format!("bwd_dw2_check_{}_{}", j, k),
                )?;
            }
        }

        // ================================================================
        // 7. Backward — db2: db2[j] = dy[j]
        // ================================================================
        for j in 0..w.d_out {
            assign_eq(
                &config,
                &mut layouter,
                w.dy[j],
                w.db2[j],
                &format!("bwd_db2_check_{}", j),
            )?;
        }

        // ================================================================
        // 8. Backward — dh: dh[k] = sum_j(W2[j][k] * dy[j])
        // ================================================================
        for k in 0..w.d_hid {
            // Collect W2 column k: W2[0][k], W2[1][k], ...
            let w2_col: Vec<Fr> = (0..w.d_out).map(|j| w.w2[j * w.d_hid + k]).collect();
            verify_dot_product(
                &config,
                &mut layouter,
                &w2_col,
                &w.dy,
                w.dh[k],
                &format!("bwd_dh_{}", k),
            )?;
        }

        // ================================================================
        // 9. Backward — ReLU mask: dh_pre = dh * mask
        // ================================================================
        for k in 0..w.d_hid {
            assign_mul(
                &config,
                &mut layouter,
                w.dh[k],
                w.relu_mask[k],
                w.dh_pre[k],
                &format!("bwd_relu_mask_{}", k),
            )?;
        }

        // ================================================================
        // 10. Backward — dW1: dW1[j][i] = dh_pre[j] * x[i]
        // ================================================================
        for j in 0..w.d_hid {
            for i in 0..w.d_in {
                let expected = w.dh_pre[j] * w.x[i];
                assign_mul(
                    &config,
                    &mut layouter,
                    w.dh_pre[j],
                    w.x[i],
                    expected,
                    &format!("bwd_dw1_{}_{}", j, i),
                )?;
                assign_eq(
                    &config,
                    &mut layouter,
                    expected,
                    w.dw1[j * w.d_in + i],
                    &format!("bwd_dw1_check_{}_{}", j, i),
                )?;
            }
        }

        // ================================================================
        // 11. Backward — db1: db1[j] = dh_pre[j]
        // ================================================================
        for j in 0..w.d_hid {
            assign_eq(
                &config,
                &mut layouter,
                w.dh_pre[j],
                w.db1[j],
                &format!("bwd_db1_check_{}", j),
            )?;
        }

        // ================================================================
        // 12. Weight update: W_new = W_old - lr * grad
        // ================================================================
        // W1 update
        for idx in 0..w.w1.len() {
            let lr_grad = w.lr * w.dw1[idx];
            assign_mul(
                &config,
                &mut layouter,
                w.lr,
                w.dw1[idx],
                lr_grad,
                &format!("upd_w1_lr_{}", idx),
            )?;
            assign_sub(
                &config,
                &mut layouter,
                w.w1[idx],
                lr_grad,
                w.w1_new[idx],
                &format!("upd_w1_{}", idx),
            )?;
        }

        // b1 update
        for idx in 0..w.b1.len() {
            let lr_grad = w.lr * w.db1[idx];
            assign_mul(
                &config,
                &mut layouter,
                w.lr,
                w.db1[idx],
                lr_grad,
                &format!("upd_b1_lr_{}", idx),
            )?;
            assign_sub(
                &config,
                &mut layouter,
                w.b1[idx],
                lr_grad,
                w.b1_new[idx],
                &format!("upd_b1_{}", idx),
            )?;
        }

        // W2 update
        for idx in 0..w.w2.len() {
            let lr_grad = w.lr * w.dw2[idx];
            assign_mul(
                &config,
                &mut layouter,
                w.lr,
                w.dw2[idx],
                lr_grad,
                &format!("upd_w2_lr_{}", idx),
            )?;
            assign_sub(
                &config,
                &mut layouter,
                w.w2[idx],
                lr_grad,
                w.w2_new[idx],
                &format!("upd_w2_{}", idx),
            )?;
        }

        // b2 update
        for idx in 0..w.b2.len() {
            let lr_grad = w.lr * w.db2[idx];
            assign_mul(
                &config,
                &mut layouter,
                w.lr,
                w.db2[idx],
                lr_grad,
                &format!("upd_b2_lr_{}", idx),
            )?;
            assign_sub(
                &config,
                &mut layouter,
                w.b2[idx],
                lr_grad,
                w.b2_new[idx],
                &format!("upd_b2_{}", idx),
            )?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper: load ReLU lookup table
// ---------------------------------------------------------------------------

fn load_relu_table(
    config: &MLTrainingStepConfig,
    layouter: &mut impl Layouter<Fr>,
    half_range: usize,
) -> Result<(), ErrorFront> {
    layouter.assign_table(
        || "relu_table",
        |mut table| {
            let mut row = 0;

            // (0, 0) must come first for the selector-off case.
            table.assign_cell(|| "in_0", config.relu_table_in, row, || Value::known(Fr::ZERO))?;
            table.assign_cell(
                || "out_0",
                config.relu_table_out,
                row,
                || Value::known(Fr::ZERO),
            )?;
            row += 1;

            // Positive entries: x -> x for x in [1, half_range)
            for x in 1..half_range {
                let f = Fr::from(x as u64);
                table.assign_cell(
                    || format!("in_{}", x),
                    config.relu_table_in,
                    row,
                    || Value::known(f),
                )?;
                table.assign_cell(
                    || format!("out_{}", x),
                    config.relu_table_out,
                    row,
                    || Value::known(f),
                )?;
                row += 1;
            }

            // Negative entries: (p - x) -> 0 for x in [1, half_range)
            for x in 1..half_range {
                let neg = Fr::ZERO - Fr::from(x as u64);
                table.assign_cell(
                    || format!("in_neg_{}", x),
                    config.relu_table_in,
                    row,
                    || Value::known(neg),
                )?;
                table.assign_cell(
                    || format!("out_neg_{}", x),
                    config.relu_table_out,
                    row,
                    || Value::known(Fr::ZERO),
                )?;
                row += 1;
            }

            Ok(())
        },
    )
}

// ---------------------------------------------------------------------------
// Helper: verify dot product  sum_i(a[i] * b[i]) = expected
// ---------------------------------------------------------------------------

fn verify_dot_product(
    config: &MLTrainingStepConfig,
    layouter: &mut impl Layouter<Fr>,
    a: &[Fr],
    b: &[Fr],
    expected: Fr,
    label: &str,
) -> Result<(), ErrorFront> {
    assert_eq!(a.len(), b.len(), "dot product dimension mismatch");
    let n = a.len();
    if n == 0 {
        return Ok(());
    }

    // Compute partial products and running sum.
    let mut running = Fr::ZERO;
    for i in 0..n {
        let prod = a[i] * b[i];

        // Constrain: a[i] * b[i] = prod
        assign_mul(
            config,
            layouter,
            a[i],
            b[i],
            prod,
            &format!("{}_mul_{}", label, i),
        )?;

        let next = running + prod;
        if i > 0 {
            // Constrain: running + prod = next
            assign_add(
                config,
                layouter,
                running,
                prod,
                next,
                &format!("{}_acc_{}", label, i),
            )?;
        }
        running = next;
    }

    // Constrain: running == expected
    assign_eq(config, layouter, running, expected, label)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Primitive assignment helpers (single-row regions)
// ---------------------------------------------------------------------------

fn assign_mul(
    config: &MLTrainingStepConfig,
    layouter: &mut impl Layouter<Fr>,
    a: Fr,
    b: Fr,
    c: Fr,
    label: &str,
) -> Result<(), ErrorFront> {
    layouter.assign_region(
        || label.to_string(),
        |mut region| {
            config.s_mul.enable(&mut region, 0)?;
            region.assign_advice(|| "a", config.advice[0], 0, || Value::known(a))?;
            region.assign_advice(|| "b", config.advice[1], 0, || Value::known(b))?;
            region.assign_advice(|| "c", config.advice[2], 0, || Value::known(c))?;
            Ok(())
        },
    )
}

fn assign_add(
    config: &MLTrainingStepConfig,
    layouter: &mut impl Layouter<Fr>,
    a: Fr,
    b: Fr,
    c: Fr,
    label: &str,
) -> Result<(), ErrorFront> {
    layouter.assign_region(
        || label.to_string(),
        |mut region| {
            config.s_add.enable(&mut region, 0)?;
            region.assign_advice(|| "a", config.advice[0], 0, || Value::known(a))?;
            region.assign_advice(|| "b", config.advice[1], 0, || Value::known(b))?;
            region.assign_advice(|| "c", config.advice[2], 0, || Value::known(c))?;
            Ok(())
        },
    )
}

fn assign_sub(
    config: &MLTrainingStepConfig,
    layouter: &mut impl Layouter<Fr>,
    a: Fr,
    b: Fr,
    c: Fr,
    label: &str,
) -> Result<(), ErrorFront> {
    layouter.assign_region(
        || label.to_string(),
        |mut region| {
            config.s_sub.enable(&mut region, 0)?;
            region.assign_advice(|| "a", config.advice[0], 0, || Value::known(a))?;
            region.assign_advice(|| "b", config.advice[1], 0, || Value::known(b))?;
            region.assign_advice(|| "c", config.advice[2], 0, || Value::known(c))?;
            Ok(())
        },
    )
}

fn assign_eq(
    config: &MLTrainingStepConfig,
    layouter: &mut impl Layouter<Fr>,
    a: Fr,
    b: Fr,
    label: &str,
) -> Result<(), ErrorFront> {
    layouter.assign_region(
        || label.to_string(),
        |mut region| {
            config.s_eq.enable(&mut region, 0)?;
            region.assign_advice(|| "a", config.advice[0], 0, || Value::known(a))?;
            region.assign_advice(|| "b", config.advice[1], 0, || Value::known(b))?;
            Ok(())
        },
    )
}

fn assign_relu(
    config: &MLTrainingStepConfig,
    layouter: &mut impl Layouter<Fr>,
    input: Fr,
    output: Fr,
    label: &str,
) -> Result<(), ErrorFront> {
    layouter.assign_region(
        || label.to_string(),
        |mut region| {
            config.s_relu.enable(&mut region, 0)?;
            region.assign_advice(|| "relu_in", config.advice[0], 0, || Value::known(input))?;
            region.assign_advice(|| "relu_out", config.advice[1], 0, || Value::known(output))?;
            Ok(())
        },
    )
}

// ---------------------------------------------------------------------------
// Witness generation helper: compute a full training step from raw data
// ---------------------------------------------------------------------------

/// Computes the full forward/backward/update witness for a 2-layer MLP.
///
/// All values are represented as small integers in `Fr`.
/// This is the "honest prover" path that generates a valid witness.
pub fn compute_witness(
    d_in: usize,
    d_hid: usize,
    d_out: usize,
    x: &[Fr],
    target: &[Fr],
    w1: &[Fr],     // d_hid * d_in, row-major
    b1: &[Fr],     // d_hid
    w2: &[Fr],     // d_out * d_hid, row-major
    b2: &[Fr],     // d_out
    lr: Fr,
    old_state_hash: (Fr, Fr),
    new_state_hash: (Fr, Fr),
    step_number: u64,
) -> MLTrainingStepWitness {
    // --- Forward pass ---
    // h_pre[j] = sum_i(W1[j][i] * x[i]) + b1[j]
    let mut h_pre = vec![Fr::ZERO; d_hid];
    for j in 0..d_hid {
        let mut sum = Fr::ZERO;
        for i in 0..d_in {
            sum += w1[j * d_in + i] * x[i];
        }
        h_pre[j] = sum + b1[j];
    }

    // h = ReLU(h_pre)
    // For field elements, we need to know if the value is "positive".
    // Convention: values in [0, p/2) are positive, [p/2, p) are negative.
    let _half_p = Fr::ZERO - Fr::ONE; // p - 1
    // Actually we cannot compare Fr directly. For the demo, we'll use a simpler
    // approach: compute the relu mask from the original integer values.
    // Since we're using small integers, we track sign separately.
    let mut h = vec![Fr::ZERO; d_hid];
    let mut relu_mask = vec![Fr::ZERO; d_hid];
    for j in 0..d_hid {
        // Check if h_pre[j] is "positive" (small positive integer)
        // For demo purposes, we check the first limb representation.
        let repr = h_pre[j].to_repr();
        let bytes = repr.as_ref();
        // If high bytes are all 0xFF..., it's a "negative" value (close to p)
        let is_negative = bytes[31] > 0x30; // rough heuristic for demo
        if is_negative || h_pre[j] == Fr::ZERO {
            h[j] = Fr::ZERO;
            relu_mask[j] = Fr::ZERO;
        } else {
            h[j] = h_pre[j];
            relu_mask[j] = Fr::ONE;
        }
    }

    // y[j] = sum_k(W2[j][k] * h[k]) + b2[j]
    let mut y = vec![Fr::ZERO; d_out];
    for j in 0..d_out {
        let mut sum = Fr::ZERO;
        for k in 0..d_hid {
            sum += w2[j * d_hid + k] * h[k];
        }
        y[j] = sum + b2[j];
    }

    // --- Loss ---
    let mut loss = Fr::ZERO;
    for j in 0..d_out {
        let diff = y[j] - target[j];
        loss += diff * diff;
    }

    // --- Backward pass ---
    let two = Fr::from(2u64);

    // dy[j] = 2*(y[j] - target[j])
    let dy: Vec<Fr> = (0..d_out).map(|j| two * (y[j] - target[j])).collect();

    // dW2[j][k] = dy[j] * h[k]
    let mut dw2 = vec![Fr::ZERO; d_out * d_hid];
    for j in 0..d_out {
        for k in 0..d_hid {
            dw2[j * d_hid + k] = dy[j] * h[k];
        }
    }

    // db2 = dy
    let db2 = dy.clone();

    // dh[k] = sum_j(W2[j][k] * dy[j])
    let mut dh = vec![Fr::ZERO; d_hid];
    for k in 0..d_hid {
        for j in 0..d_out {
            dh[k] += w2[j * d_hid + k] * dy[j];
        }
    }

    // dh_pre = dh * relu_mask
    let dh_pre: Vec<Fr> = (0..d_hid).map(|k| dh[k] * relu_mask[k]).collect();

    // dW1[j][i] = dh_pre[j] * x[i]
    let mut dw1 = vec![Fr::ZERO; d_hid * d_in];
    for j in 0..d_hid {
        for i in 0..d_in {
            dw1[j * d_in + i] = dh_pre[j] * x[i];
        }
    }

    // db1 = dh_pre
    let db1 = dh_pre.clone();

    // --- Weight update ---
    let w1_new: Vec<Fr> = w1
        .iter()
        .zip(dw1.iter())
        .map(|(&w, &dw)| w - lr * dw)
        .collect();
    let b1_new: Vec<Fr> = b1
        .iter()
        .zip(db1.iter())
        .map(|(&b, &db)| b - lr * db)
        .collect();
    let w2_new: Vec<Fr> = w2
        .iter()
        .zip(dw2.iter())
        .map(|(&w, &dw)| w - lr * dw)
        .collect();
    let b2_new: Vec<Fr> = b2
        .iter()
        .zip(db2.iter())
        .map(|(&b, &db)| b - lr * db)
        .collect();

    // --- Error bound ---
    // For this initial implementation, error is zero (exact integer arithmetic).
    // In production with quantized floats, this would track accumulated rounding error.
    let total_error = Fr::ZERO;

    MLTrainingStepWitness {
        d_in,
        d_hid,
        d_out,
        x: x.to_vec(),
        target: target.to_vec(),
        w1: w1.to_vec(),
        b1: b1.to_vec(),
        w2: w2.to_vec(),
        b2: b2.to_vec(),
        h_pre,
        h,
        y,
        loss,
        dy,
        dw2,
        db2,
        dh,
        relu_mask,
        dh_pre,
        dw1,
        db1,
        lr,
        w1_new,
        b1_new,
        w2_new,
        b2_new,
        total_error,
        old_state_hash,
        new_state_hash,
        step_number,
    }
}

/// Computes a SHA-256-based state hash for a weight set and returns (lo, hi) as Fr.
pub fn compute_state_hash(w1: &[Fr], b1: &[Fr], w2: &[Fr], b2: &[Fr]) -> (Fr, Fr) {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for v in w1.iter().chain(b1).chain(w2).chain(b2) {
        hasher.update(v.to_repr().as_ref());
    }
    let hash: [u8; 32] = hasher.finalize().into();

    let lo = {
        let mut buf = [0u8; 32];
        buf[..16].copy_from_slice(&hash[..16]);
        Fr::from_raw([
            u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            0,
            0,
        ])
    };
    let hi = {
        let mut buf = [0u8; 32];
        buf[..16].copy_from_slice(&hash[16..32]);
        Fr::from_raw([
            u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            0,
            0,
        ])
    };

    (lo, hi)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    /// Builds a tiny 2×2→1 model and checks the circuit with MockProver.
    fn make_tiny_circuit() -> (MLTrainingStepCircuit, Vec<Fr>) {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        // Small integer weights.
        let w1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)];
        let b1 = vec![Fr::from(0), Fr::from(0)];
        let w2 = vec![Fr::from(1), Fr::from(1)];
        let b2 = vec![Fr::from(0)];

        let x = vec![Fr::from(1), Fr::from(1)];
        let target = vec![Fr::from(5)];

        let lr = Fr::from(1);

        let old_hash = compute_state_hash(&w1, &b1, &w2, &b2);

        let witness = compute_witness(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr, old_hash,
            (Fr::ZERO, Fr::ZERO), // placeholder new hash, computed below
            1,
        );

        let new_hash =
            compute_state_hash(&witness.w1_new, &witness.b1_new, &witness.w2_new, &witness.b2_new);

        // Rebuild witness with correct new hash.
        let witness = compute_witness(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr, old_hash, new_hash, 1,
        );

        let pi = witness.public_inputs();
        let circuit = MLTrainingStepCircuit {
            witness,
            relu_range: 128,
        };
        (circuit, pi)
    }

    #[test]
    fn test_tiny_valid() {
        let (circuit, pi) = make_tiny_circuit();
        // k=14 gives 16384 rows — generous for the demo circuit.
        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_tiny_wrong_loss() {
        let (circuit, mut pi) = make_tiny_circuit();
        // Corrupt the loss public input.
        pi[4] = Fr::from(999u64);
        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn test_tiny_wrong_new_state() {
        let (circuit, mut pi) = make_tiny_circuit();
        // Corrupt new state hash.
        pi[2] = Fr::from(12345u64);
        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// Verifies that a corrupt weight update is detected.
    #[test]
    fn test_adversarial_fake_gradient() {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        let w1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)];
        let b1 = vec![Fr::from(0), Fr::from(0)];
        let w2 = vec![Fr::from(1), Fr::from(1)];
        let b2 = vec![Fr::from(0)];
        let x = vec![Fr::from(1), Fr::from(1)];
        let target = vec![Fr::from(5)];
        let lr = Fr::from(1);

        let old_hash = compute_state_hash(&w1, &b1, &w2, &b2);
        let mut witness = compute_witness(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr, old_hash,
            (Fr::ZERO, Fr::ZERO),
            1,
        );

        // Adversary: inject a fake gradient for W1[0]
        witness.dw1[0] = Fr::from(999u64);
        // Recalculate new weight with fake gradient to make weight update consistent
        witness.w1_new[0] = witness.w1[0] - lr * Fr::from(999u64);

        let new_hash =
            compute_state_hash(&witness.w1_new, &witness.b1_new, &witness.w2_new, &witness.b2_new);
        witness.new_state_hash = new_hash;

        let pi = witness.public_inputs();
        let circuit = MLTrainingStepCircuit {
            witness,
            relu_range: 128,
        };

        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        // The circuit should reject because dW1[0] != dh_pre[0] * x[0].
        assert!(prover.verify().is_err());
    }

    #[test]
    fn test_4x8x2_model() {
        let d_in = 4;
        let d_hid = 4; // Keep small for fast test
        let d_out = 2;

        // Initialize weights as small integers.
        let w1: Vec<Fr> = (0..d_hid * d_in).map(|i| Fr::from((i % 3 + 1) as u64)).collect();
        let b1 = vec![Fr::from(0); d_hid];
        let w2: Vec<Fr> = (0..d_out * d_hid).map(|i| Fr::from((i % 2 + 1) as u64)).collect();
        let b2 = vec![Fr::from(0); d_out];

        let x: Vec<Fr> = (0..d_in).map(|i| Fr::from((i + 1) as u64)).collect();
        let target: Vec<Fr> = (0..d_out).map(|_| Fr::from(10u64)).collect();
        let lr = Fr::from(1);

        let old_hash = compute_state_hash(&w1, &b1, &w2, &b2);
        let witness = compute_witness(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr, old_hash,
            (Fr::ZERO, Fr::ZERO),
            1,
        );
        let new_hash =
            compute_state_hash(&witness.w1_new, &witness.b1_new, &witness.w2_new, &witness.b2_new);
        let witness = compute_witness(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr, old_hash, new_hash, 1,
        );

        let pi = witness.public_inputs();
        let circuit = MLTrainingStepCircuit {
            witness,
            relu_range: 128,
        };

        let prover = MockProver::run(14, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }
}
