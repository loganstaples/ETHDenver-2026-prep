//! Positional Encoding Circuit.
//!
//! Implements positional encoding verification for transformer models.
//! Positional encodings provide sequence position information to the model,
//! since attention is inherently position-agnostic.
//!
//! # Encoding Types
//!
//! 1. **Sinusoidal**: Fixed encodings using sine/cosine functions
//!    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
//!    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
//!
//! 2. **Learned**: Trainable position embeddings (like token embeddings)
//!
//! 3. **Rotary (RoPE)**: Applies rotation to Q and K based on position
//!
//! # Circuit Strategy
//!
//! For sinusoidal encodings, we use lookup tables for sin/cos approximations.
//! For learned encodings, we use the same approach as token embeddings.
//! The circuit verifies that position encodings are correctly added to embeddings.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, Expression, Fixed, Instance, Selector,
        TableColumn,
    },
    poly::Rotation,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::{Field, PrimeField};
use std::f64::consts::PI;
use std::marker::PhantomData;

/// Scale factor for fixed-point sin/cos values.
pub const TRIG_SCALE: u64 = 1024;

/// Maximum sequence length supported.
pub const MAX_SEQ_LEN: usize = 2048;

/// Base for positional encoding frequency computation.
pub const POSITION_BASE: f64 = 10000.0;

/// Configuration for positional encoding circuit.
#[derive(Clone, Debug)]
pub struct PositionalConfig<F: PrimeField> {
    /// Advice columns for computation.
    pub advice: [Column<Advice>; 4],
    /// Instance column for public inputs.
    pub instance: Column<Instance>,
    /// Lookup table for sin values.
    pub sin_table_in: TableColumn,
    pub sin_table_out: TableColumn,
    /// Lookup table for cos values.
    pub cos_table_in: TableColumn,
    pub cos_table_out: TableColumn,
    /// Selector for addition.
    pub s_add: Selector,
    /// Selector for sin lookup.
    pub s_sin: Selector,
    /// Selector for cos lookup.
    pub s_cos: Selector,
    /// Selector for multiplication.
    pub s_mul: Selector,
    /// Phantom data.
    _marker: PhantomData<F>,
}

/// Positional encoding chip.
pub struct PositionalChip<F: PrimeField> {
    config: PositionalConfig<F>,
}

impl<F: PrimeField> PositionalChip<F> {
    /// Creates a new positional encoding chip.
    pub fn new(config: PositionalConfig<F>) -> Self {
        Self { config }
    }

    /// Configures the positional encoding circuit.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> PositionalConfig<F> {
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

        let sin_table_in = meta.lookup_table_column();
        let sin_table_out = meta.lookup_table_column();
        let cos_table_in = meta.lookup_table_column();
        let cos_table_out = meta.lookup_table_column();

        let s_add = meta.selector();
        let s_sin = meta.complex_selector();
        let s_cos = meta.complex_selector();
        let s_mul = meta.selector();

        // Addition gate: a + b = c
        meta.create_gate("pos_add", |meta| {
            let s = meta.query_selector(s_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a + b - c)]
        });

        // Multiplication gate: a * b = c
        meta.create_gate("pos_mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a * b - c)]
        });

        // Sin lookup
        meta.lookup(|meta| {
            let s = meta.query_selector(s_sin);
            let input = meta.query_advice(advice[0], Rotation::cur());
            let output = meta.query_advice(advice[1], Rotation::cur());
            vec![
                (s.clone() * input, sin_table_in),
                (s * output, sin_table_out),
            ]
        });

        // Cos lookup
        meta.lookup(|meta| {
            let s = meta.query_selector(s_cos);
            let input = meta.query_advice(advice[0], Rotation::cur());
            let output = meta.query_advice(advice[1], Rotation::cur());
            vec![
                (s.clone() * input, cos_table_in),
                (s * output, cos_table_out),
            ]
        });

        PositionalConfig {
            advice,
            instance,
            sin_table_in,
            sin_table_out,
            cos_table_in,
            cos_table_out,
            s_add,
            s_sin,
            s_cos,
            s_mul,
            _marker: PhantomData,
        }
    }

    /// Loads the sin/cos lookup tables.
    pub fn load_trig_tables(
        &self,
        layouter: &mut impl Layouter<F>,
        table_size: usize,
    ) -> Result<(), Error> {
        let scale = TRIG_SCALE as f64;

        // Load sin table
        layouter.assign_table(
            || "sin_table",
            |mut table| {
                // First row: (0, 0) for disabled selector rows
                table.assign_cell(
                    || "sin_in_zero",
                    self.config.sin_table_in,
                    0,
                    || Value::known(F::ZERO),
                )?;
                table.assign_cell(
                    || "sin_out_zero",
                    self.config.sin_table_out,
                    0,
                    || Value::known(F::ZERO),
                )?;

                // Remaining rows: actual sin values (offset by 1)
                for i in 0..table_size {
                    let input = F::from(i as u64);
                    // Map i to angle: i * 2π / table_size
                    let angle = (i as f64) * 2.0 * PI / (table_size as f64);
                    let sin_val = angle.sin() * scale + scale; // Shift to positive
                    let output = F::from(sin_val.round().max(0.0) as u64);

                    table.assign_cell(
                        || format!("sin_in_{}", i),
                        self.config.sin_table_in,
                        i + 1,
                        || Value::known(input),
                    )?;
                    table.assign_cell(
                        || format!("sin_out_{}", i),
                        self.config.sin_table_out,
                        i + 1,
                        || Value::known(output),
                    )?;
                }
                Ok(())
            },
        )?;

        // Load cos table
        layouter.assign_table(
            || "cos_table",
            |mut table| {
                // First row: (0, 0) for disabled selector rows
                table.assign_cell(
                    || "cos_in_zero",
                    self.config.cos_table_in,
                    0,
                    || Value::known(F::ZERO),
                )?;
                table.assign_cell(
                    || "cos_out_zero",
                    self.config.cos_table_out,
                    0,
                    || Value::known(F::ZERO),
                )?;

                // Remaining rows: actual cos values (offset by 1)
                for i in 0..table_size {
                    let input = F::from(i as u64);
                    let angle = (i as f64) * 2.0 * PI / (table_size as f64);
                    let cos_val = angle.cos() * scale + scale; // Shift to positive
                    let output = F::from(cos_val.round().max(0.0) as u64);

                    table.assign_cell(
                        || format!("cos_in_{}", i),
                        self.config.cos_table_in,
                        i + 1,
                        || Value::known(input),
                    )?;
                    table.assign_cell(
                        || format!("cos_out_{}", i),
                        self.config.cos_table_out,
                        i + 1,
                        || Value::known(output),
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Verifies positional encoding addition.
    pub fn verify_positional_encoding(
        &self,
        mut layouter: impl Layouter<F>,
        witness: &PositionalWitness<F>,
    ) -> Result<(), Error> {
        let seq_len = witness.positions.len();
        let d_model = witness.pos_encodings.get(0).map(|e| e.len()).unwrap_or(0);

        for pos in 0..seq_len {
            for dim in 0..d_model {
                // Verify: output[pos][dim] = input[pos][dim] + pos_encoding[pos][dim]
                let input_val = witness.input_embeddings[pos][dim];
                let pos_val = witness.pos_encodings[pos][dim];
                let output_val = witness.output_embeddings[pos][dim];

                layouter.assign_region(
                    || format!("add_pos_{}_{}", pos, dim),
                    |mut region| {
                        self.config.s_add.enable(&mut region, 0)?;
                        region.assign_advice(|| "input", self.config.advice[0], 0, || Value::known(input_val))?;
                        region.assign_advice(|| "pos", self.config.advice[1], 0, || Value::known(pos_val))?;
                        region.assign_advice(|| "output", self.config.advice[2], 0, || Value::known(output_val))?;
                        Ok(())
                    },
                )?;

                // If using sinusoidal, verify the sin/cos lookup
                if witness.use_sinusoidal && dim < witness.sin_values.len() {
                    let (sin_in, sin_out) = witness.sin_values[pos * d_model + dim];
                    let (cos_in, cos_out) = witness.cos_values[pos * d_model + dim];

                    // Verify sin lookup
                    if dim % 2 == 0 {
                        layouter.assign_region(
                            || format!("sin_lookup_{}_{}", pos, dim),
                            |mut region| {
                                self.config.s_sin.enable(&mut region, 0)?;
                                region.assign_advice(|| "sin_in", self.config.advice[0], 0, || Value::known(sin_in))?;
                                region.assign_advice(|| "sin_out", self.config.advice[1], 0, || Value::known(sin_out))?;
                                Ok(())
                            },
                        )?;
                    } else {
                        // Verify cos lookup
                        layouter.assign_region(
                            || format!("cos_lookup_{}_{}", pos, dim),
                            |mut region| {
                                self.config.s_cos.enable(&mut region, 0)?;
                                region.assign_advice(|| "cos_in", self.config.advice[0], 0, || Value::known(cos_in))?;
                                region.assign_advice(|| "cos_out", self.config.advice[1], 0, || Value::known(cos_out))?;
                                Ok(())
                            },
                        )?;
                    }
                }
            }
        }

        Ok(())
    }
}

/// Witness data for positional encoding verification.
#[derive(Clone, Debug)]
pub struct PositionalWitness<F: PrimeField> {
    /// Sequence positions.
    pub positions: Vec<usize>,
    /// Input embeddings [seq_len, d_model].
    pub input_embeddings: Vec<Vec<F>>,
    /// Positional encodings [seq_len, d_model].
    pub pos_encodings: Vec<Vec<F>>,
    /// Output embeddings (input + pos) [seq_len, d_model].
    pub output_embeddings: Vec<Vec<F>>,
    /// Whether using sinusoidal encoding.
    pub use_sinusoidal: bool,
    /// Sin lookup values (input, output) for each position/dim.
    pub sin_values: Vec<(F, F)>,
    /// Cos lookup values (input, output) for each position/dim.
    pub cos_values: Vec<(F, F)>,
    /// Error bound.
    pub error_bound: F,
    /// Model dimension.
    pub d_model: usize,
    /// Lookup table size used.
    pub table_size: usize,
}

impl<F: PrimeField> Default for PositionalWitness<F> {
    fn default() -> Self {
        Self {
            positions: vec![],
            input_embeddings: vec![],
            pos_encodings: vec![],
            output_embeddings: vec![],
            use_sinusoidal: true,
            sin_values: vec![],
            cos_values: vec![],
            error_bound: F::ZERO,
            d_model: 0,
            table_size: 256,
        }
    }
}

/// Precomputed sinusoidal positional encodings.
#[derive(Clone, Debug)]
pub struct SinusoidalEncoding {
    /// Encoding matrix [max_seq_len, d_model].
    pub encodings: Vec<Vec<Fr>>,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Scale factor used.
    pub scale: u64,
}

impl SinusoidalEncoding {
    /// Creates new sinusoidal encodings.
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        let scale = TRIG_SCALE;
        let scale_f64 = scale as f64;
        let mut encodings = Vec::with_capacity(max_seq_len);

        for pos in 0..max_seq_len {
            let mut encoding = Vec::with_capacity(d_model);
            for i in 0..d_model {
                let div_term = (POSITION_BASE).powf((2.0 * (i / 2) as f64) / (d_model as f64));
                let angle = (pos as f64) / div_term;

                let val = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };

                // Scale and shift to positive range
                let scaled = (val * scale_f64 + scale_f64).round().max(0.0) as u64;
                encoding.push(Fr::from(scaled));
            }
            encodings.push(encoding);
        }

        Self {
            encodings,
            max_seq_len,
            d_model,
            scale,
        }
    }

    /// Gets encoding for a specific position.
    pub fn get(&self, pos: usize) -> Option<&[Fr]> {
        self.encodings.get(pos).map(|e| e.as_slice())
    }

    /// Gets encodings for a range of positions.
    pub fn get_range(&self, start: usize, end: usize) -> Vec<Vec<Fr>> {
        self.encodings[start.min(self.max_seq_len)..end.min(self.max_seq_len)].to_vec()
    }
}

/// Learned positional embeddings.
#[derive(Clone, Debug)]
pub struct LearnedPositionalEmbedding {
    /// Embedding matrix [max_seq_len, d_model].
    pub embeddings: Vec<Vec<Fr>>,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Model dimension.
    pub d_model: usize,
}

impl LearnedPositionalEmbedding {
    /// Creates new learned embeddings (initialized randomly for testing).
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let embeddings: Vec<Vec<Fr>> = (0..max_seq_len)
            .map(|pos| {
                (0..d_model)
                    .map(|dim| {
                        let mut hasher = DefaultHasher::new();
                        pos.hash(&mut hasher);
                        dim.hash(&mut hasher);
                        "pos_embed".hash(&mut hasher);
                        Fr::from(hasher.finish() % 1000) // Small values for testing
                    })
                    .collect()
            })
            .collect();

        Self {
            embeddings,
            max_seq_len,
            d_model,
        }
    }

    /// Gets embedding for a specific position.
    pub fn get(&self, pos: usize) -> Option<&[Fr]> {
        self.embeddings.get(pos).map(|e| e.as_slice())
    }
}

/// Computes sinusoidal positional encoding witness.
pub fn compute_sinusoidal_witness(
    input_embeddings: &[Vec<Fr>],
    positions: &[usize],
    d_model: usize,
    table_size: usize,
    base_error: Fr,
) -> PositionalWitness<Fr> {
    let seq_len = positions.len();
    let encoding = SinusoidalEncoding::new(seq_len.max(*positions.iter().max().unwrap_or(&0) + 1), d_model);

    let mut pos_encodings = Vec::with_capacity(seq_len);
    let mut output_embeddings = Vec::with_capacity(seq_len);
    let mut sin_values = Vec::new();
    let mut cos_values = Vec::new();

    let scale = TRIG_SCALE as f64;

    for (idx, &pos) in positions.iter().enumerate() {
        let pos_enc = encoding.get(pos).unwrap_or(&[]).to_vec();
        let input = &input_embeddings[idx];

        // Compute output = input + pos_encoding
        let output: Vec<Fr> = input
            .iter()
            .zip(pos_enc.iter())
            .map(|(&i, &p)| i + p)
            .collect();

        // Compute sin/cos lookup values for verification
        // IMPORTANT: The lookup values must match exactly what's in the lookup table
        for dim in 0..d_model {
            let div_term = (POSITION_BASE).powf((2.0 * (dim / 2) as f64) / (d_model as f64));
            let angle = (pos as f64) / div_term;

            // Map angle to table index
            let angle_normalized = ((angle / (2.0 * PI)) % 1.0).abs();
            let table_idx = ((angle_normalized * table_size as f64).round() as usize) % table_size;

            // Use the TABLE's angle (quantized) to compute sin/cos, not the original angle
            // This ensures the (input, output) pair matches what's in the lookup table
            let table_angle = (table_idx as f64) * 2.0 * PI / (table_size as f64);

            let sin_in = Fr::from(table_idx as u64);
            let sin_out = Fr::from((table_angle.sin() * scale + scale).round().max(0.0) as u64);
            let cos_in = Fr::from(table_idx as u64);
            let cos_out = Fr::from((table_angle.cos() * scale + scale).round().max(0.0) as u64);

            sin_values.push((sin_in, sin_out));
            cos_values.push((cos_in, cos_out));
        }

        pos_encodings.push(pos_enc);
        output_embeddings.push(output);
    }

    PositionalWitness {
        positions: positions.to_vec(),
        input_embeddings: input_embeddings.to_vec(),
        pos_encodings,
        output_embeddings,
        use_sinusoidal: true,
        sin_values,
        cos_values,
        error_bound: base_error * Fr::from(seq_len as u64),
        d_model,
        table_size,
    }
}

/// Computes learned positional encoding witness.
pub fn compute_learned_witness(
    input_embeddings: &[Vec<Fr>],
    positions: &[usize],
    learned_embeddings: &LearnedPositionalEmbedding,
    base_error: Fr,
) -> PositionalWitness<Fr> {
    let seq_len = positions.len();
    let d_model = learned_embeddings.d_model;

    let mut pos_encodings = Vec::with_capacity(seq_len);
    let mut output_embeddings = Vec::with_capacity(seq_len);

    for (idx, &pos) in positions.iter().enumerate() {
        let pos_enc = learned_embeddings.get(pos).unwrap_or(&[]).to_vec();
        let input = &input_embeddings[idx];

        let output: Vec<Fr> = input
            .iter()
            .zip(pos_enc.iter())
            .map(|(&i, &p)| i + p)
            .collect();

        pos_encodings.push(pos_enc);
        output_embeddings.push(output);
    }

    PositionalWitness {
        positions: positions.to_vec(),
        input_embeddings: input_embeddings.to_vec(),
        pos_encodings,
        output_embeddings,
        use_sinusoidal: false,
        sin_values: vec![],
        cos_values: vec![],
        error_bound: base_error * Fr::from(seq_len as u64),
        d_model,
        table_size: 0,
    }
}

/// Complete circuit for positional encoding verification.
#[derive(Clone)]
pub struct PositionalCircuit<F: PrimeField> {
    pub witness: PositionalWitness<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Default for PositionalCircuit<F> {
    fn default() -> Self {
        Self {
            witness: PositionalWitness::default(),
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> PositionalCircuit<F> {
    /// Creates a new positional encoding circuit.
    pub fn new(witness: PositionalWitness<F>) -> Self {
        Self {
            witness,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Circuit<F> for PositionalCircuit<F> {
    type Config = PositionalConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        PositionalChip::<F>::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = PositionalChip::new(config.clone());

        // Load trig tables if using sinusoidal
        if self.witness.use_sinusoidal {
            chip.load_trig_tables(&mut layouter, self.witness.table_size.max(1))?;
        }

        // Verify positional encoding
        chip.verify_positional_encoding(layouter.namespace(|| "pos_encoding"), &self.witness)?;

        Ok(())
    }
}

/// Rotary Position Embeddings (RoPE).
///
/// Applies rotation to query and key vectors based on position.
/// This is used in modern transformers like LLaMA and GPT-NeoX.
#[derive(Clone, Debug)]
pub struct RotaryPositionalEncoding {
    /// Cosine components [max_seq_len, d_head/2].
    pub cos_cache: Vec<Vec<Fr>>,
    /// Sine components [max_seq_len, d_head/2].
    pub sin_cache: Vec<Vec<Fr>>,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Head dimension (must be even).
    pub d_head: usize,
    /// Base frequency.
    pub base: f64,
}

impl RotaryPositionalEncoding {
    /// Creates new RoPE encodings.
    pub fn new(max_seq_len: usize, d_head: usize, base: f64) -> Self {
        assert!(d_head % 2 == 0, "d_head must be even for RoPE");

        let scale = TRIG_SCALE as f64;
        let half_dim = d_head / 2;

        let mut cos_cache = Vec::with_capacity(max_seq_len);
        let mut sin_cache = Vec::with_capacity(max_seq_len);

        // Compute inverse frequencies
        let inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / base.powf((2.0 * i as f64) / d_head as f64))
            .collect();

        for pos in 0..max_seq_len {
            let pos_f64 = pos as f64;
            let mut cos_row = Vec::with_capacity(half_dim);
            let mut sin_row = Vec::with_capacity(half_dim);

            for &freq in &inv_freq {
                let angle = pos_f64 * freq;
                cos_row.push(Fr::from((angle.cos() * scale + scale).round().max(0.0) as u64));
                sin_row.push(Fr::from((angle.sin() * scale + scale).round().max(0.0) as u64));
            }

            cos_cache.push(cos_row);
            sin_cache.push(sin_row);
        }

        Self {
            cos_cache,
            sin_cache,
            max_seq_len,
            d_head,
            base,
        }
    }

    /// Applies rotary embedding to a vector.
    ///
    /// For a vector [x0, x1, x2, x3, ...], applies rotation:
    /// [x0 * cos - x1 * sin, x0 * sin + x1 * cos, x2 * cos - x3 * sin, ...]
    pub fn apply(&self, vec: &[Fr], pos: usize) -> Vec<Fr> {
        let half_dim = self.d_head / 2;
        let cos_row = &self.cos_cache[pos.min(self.max_seq_len - 1)];
        let sin_row = &self.sin_cache[pos.min(self.max_seq_len - 1)];

        let mut result = vec![Fr::ZERO; self.d_head];

        for i in 0..half_dim {
            let x0 = vec.get(2 * i).cloned().unwrap_or(Fr::ZERO);
            let x1 = vec.get(2 * i + 1).cloned().unwrap_or(Fr::ZERO);

            // Apply rotation (simplified - actual implementation needs proper scaling)
            result[2 * i] = x0 * cos_row[i] - x1 * sin_row[i];
            result[2 * i + 1] = x0 * sin_row[i] + x1 * cos_row[i];
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    #[test]
    fn test_sinusoidal_encoding() {
        let encoding = SinusoidalEncoding::new(100, 64);
        assert_eq!(encoding.max_seq_len, 100);
        assert_eq!(encoding.d_model, 64);

        let enc = encoding.get(0);
        assert!(enc.is_some());
        assert_eq!(enc.unwrap().len(), 64);
    }

    #[test]
    fn test_learned_embedding() {
        let embedding = LearnedPositionalEmbedding::new(100, 64);
        let emb = embedding.get(50);
        assert!(emb.is_some());
        assert_eq!(emb.unwrap().len(), 64);
    }

    #[test]
    fn test_sinusoidal_witness() {
        let d_model = 16;
        let input_embeddings: Vec<Vec<Fr>> = vec![
            vec![Fr::from(1); d_model],
            vec![Fr::from(2); d_model],
        ];
        let positions = vec![0, 1];
        let base_error = Fr::from(1);

        let witness = compute_sinusoidal_witness(&input_embeddings, &positions, d_model, 256, base_error);

        assert_eq!(witness.positions.len(), 2);
        assert_eq!(witness.pos_encodings.len(), 2);
        assert_eq!(witness.output_embeddings.len(), 2);
    }

    #[test]
    fn test_positional_circuit() {
        let d_model = 8;
        let input_embeddings: Vec<Vec<Fr>> = vec![
            vec![Fr::from(1); d_model],
            vec![Fr::from(2); d_model],
        ];
        let positions = vec![0, 1];
        let base_error = Fr::from(1);

        let witness = compute_sinusoidal_witness(&input_embeddings, &positions, d_model, 256, base_error);
        let circuit = PositionalCircuit::<Fr>::new(witness);

        // Instance column is configured but not used - pass empty vec for it
        let prover = MockProver::run(12, &circuit, vec![vec![]]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_learned_witness() {
        let d_model = 16;
        let learned = LearnedPositionalEmbedding::new(100, d_model);
        let input_embeddings: Vec<Vec<Fr>> = vec![
            vec![Fr::from(1); d_model],
            vec![Fr::from(2); d_model],
        ];
        let positions = vec![0, 1];
        let base_error = Fr::from(1);

        let witness = compute_learned_witness(&input_embeddings, &positions, &learned, base_error);

        assert!(!witness.use_sinusoidal);
        assert_eq!(witness.output_embeddings.len(), 2);
    }

    #[test]
    fn test_rope_encoding() {
        let rope = RotaryPositionalEncoding::new(100, 64, 10000.0);
        assert_eq!(rope.max_seq_len, 100);
        assert_eq!(rope.d_head, 64);

        let vec = vec![Fr::from(1); 64];
        let rotated = rope.apply(&vec, 5);
        assert_eq!(rotated.len(), 64);
    }

    #[test]
    fn test_encoding_consistency() {
        let encoding = SinusoidalEncoding::new(10, 8);

        // Position 0 encoding should be consistent
        let enc0 = encoding.get(0).unwrap();
        let enc0_again = encoding.get(0).unwrap();
        assert_eq!(enc0, enc0_again);

        // Different positions should have different encodings
        let enc1 = encoding.get(1).unwrap();
        assert_ne!(enc0, enc1);
    }
}
