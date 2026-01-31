//! GKR-Compatible Layered Circuit Representation.
//!
//! This module provides a layered circuit representation suitable for the GKR
//! interactive proof protocol. In GKR, circuits are organized into layers where:
//!
//! - Layer 0 is the output layer
//! - Layer d is the input layer
//! - Each gate in layer i has inputs from layer i+1
//!
//! # Circuit Structure
//!
//! ```text
//! Layer 0 (Output):    [g0]  [g1]  [g2]  [g3]
//!                        ↑     ↑     ↑     ↑
//!                        └─────┴─────┴─────┘
//! Layer 1:             [g0]  [g1]  [g2]  [g3]  [g4]  [g5]  [g6]  [g7]
//!                        ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑
//!                        └─────┴─────┴─────┴─────┴─────┴─────┴─────┘
//! Layer 2 (Input):     [x0]  [x1]  [x2]  [x3]  [x4]  [x5]  [x6]  [x7] ...
//! ```
//!
//! # Wiring Functions
//!
//! GKR uses wiring functions to describe connectivity:
//! - `W_add(g, a, b)` = 1 if gate g is an ADD gate with inputs a and b
//! - `W_mul(g, a, b)` = 1 if gate g is a MUL gate with inputs a and b

use halo2curves::ff::PrimeField;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Type of gate in the layered circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateType {
    /// Addition gate: output = left + right
    Add,
    /// Multiplication gate: output = left * right
    Mul,
    /// Constant gate: output = constant
    Const,
    /// Input gate: output = external input
    Input,
    /// Copy gate: output = left (unary)
    Copy,
    /// Negation gate: output = -left
    Neg,
}

/// A single gate in the layered circuit.
#[derive(Clone, Debug)]
pub struct Gate<F: PrimeField> {
    /// Type of this gate.
    pub gate_type: GateType,
    /// Index of this gate within its layer.
    pub index: usize,
    /// Index of the left input gate (in the next layer).
    pub left_input: usize,
    /// Index of the right input gate (in the next layer).
    pub right_input: usize,
    /// Constant value (for Const gates).
    pub constant: Option<F>,
    /// Optional label for debugging.
    pub label: Option<String>,
}

impl<F: PrimeField> Gate<F> {
    /// Creates a new add gate.
    pub fn add(index: usize, left: usize, right: usize) -> Self {
        Self {
            gate_type: GateType::Add,
            index,
            left_input: left,
            right_input: right,
            constant: None,
            label: None,
        }
    }

    /// Creates a new multiply gate.
    pub fn mul(index: usize, left: usize, right: usize) -> Self {
        Self {
            gate_type: GateType::Mul,
            index,
            left_input: left,
            right_input: right,
            constant: None,
            label: None,
        }
    }

    /// Creates a new constant gate.
    pub fn constant(index: usize, value: F) -> Self {
        Self {
            gate_type: GateType::Const,
            index,
            left_input: 0,
            right_input: 0,
            constant: Some(value),
            label: None,
        }
    }

    /// Creates a new input gate.
    pub fn input(index: usize) -> Self {
        Self {
            gate_type: GateType::Input,
            index,
            left_input: 0,
            right_input: 0,
            constant: None,
            label: None,
        }
    }

    /// Creates a copy gate.
    pub fn copy(index: usize, source: usize) -> Self {
        Self {
            gate_type: GateType::Copy,
            index,
            left_input: source,
            right_input: 0,
            constant: None,
            label: None,
        }
    }

    /// Creates a negation gate.
    pub fn neg(index: usize, source: usize) -> Self {
        Self {
            gate_type: GateType::Neg,
            index,
            left_input: source,
            right_input: 0,
            constant: None,
            label: None,
        }
    }

    /// Adds a label to this gate.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    /// Evaluates this gate given input values.
    pub fn evaluate(&self, left: F, right: F) -> F {
        match self.gate_type {
            GateType::Add => left + right,
            GateType::Mul => left * right,
            GateType::Const => self.constant.unwrap_or(F::ZERO),
            GateType::Input => F::ZERO, // Filled from external input
            GateType::Copy => left,
            GateType::Neg => F::ZERO - left,
        }
    }
}

/// A single layer in the layered circuit.
#[derive(Clone, Debug)]
pub struct Layer<F: PrimeField> {
    /// Gates in this layer.
    pub gates: Vec<Gate<F>>,
    /// Layer index (0 = output layer).
    pub layer_index: usize,
    /// Number of bits needed to address gates in this layer.
    pub log_size: usize,
}

impl<F: PrimeField> Layer<F> {
    /// Creates a new layer.
    pub fn new(layer_index: usize) -> Self {
        Self {
            gates: Vec::new(),
            layer_index,
            log_size: 0,
        }
    }

    /// Adds a gate to this layer.
    pub fn add_gate(&mut self, gate: Gate<F>) -> usize {
        let idx = self.gates.len();
        self.gates.push(Gate { index: idx, ..gate });
        self.update_log_size();
        idx
    }

    /// Updates the log_size based on gate count.
    fn update_log_size(&mut self) {
        self.log_size = (self.gates.len() as f64).log2().ceil() as usize;
    }

    /// Returns the number of gates in this layer.
    pub fn size(&self) -> usize {
        self.gates.len()
    }

    /// Returns whether this layer is empty.
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Pads the layer to a power of 2 with identity gates.
    pub fn pad_to_power_of_two(&mut self) {
        let target_size = 1 << self.log_size;
        while self.gates.len() < target_size {
            let idx = self.gates.len();
            self.gates.push(Gate::constant(idx, F::ZERO));
        }
    }

    /// Evaluates all gates in this layer given the next layer's values.
    pub fn evaluate(&self, next_layer_values: &[F]) -> Vec<F> {
        self.gates
            .iter()
            .map(|gate| {
                let left = if gate.left_input < next_layer_values.len() {
                    next_layer_values[gate.left_input]
                } else {
                    F::ZERO
                };
                let right = if gate.right_input < next_layer_values.len() {
                    next_layer_values[gate.right_input]
                } else {
                    F::ZERO
                };
                gate.evaluate(left, right)
            })
            .collect()
    }
}

/// Statistics for a single layer.
#[derive(Clone, Debug, Default)]
pub struct LayerStats {
    /// Number of add gates.
    pub num_add: usize,
    /// Number of mul gates.
    pub num_mul: usize,
    /// Number of const gates.
    pub num_const: usize,
    /// Number of input gates.
    pub num_input: usize,
    /// Number of copy gates.
    pub num_copy: usize,
    /// Number of neg gates.
    pub num_neg: usize,
}

impl LayerStats {
    /// Computes stats from a layer.
    pub fn from_layer<F: PrimeField>(layer: &Layer<F>) -> Self {
        let mut stats = Self::default();
        for gate in &layer.gates {
            match gate.gate_type {
                GateType::Add => stats.num_add += 1,
                GateType::Mul => stats.num_mul += 1,
                GateType::Const => stats.num_const += 1,
                GateType::Input => stats.num_input += 1,
                GateType::Copy => stats.num_copy += 1,
                GateType::Neg => stats.num_neg += 1,
            }
        }
        stats
    }

    /// Returns total gate count.
    pub fn total(&self) -> usize {
        self.num_add + self.num_mul + self.num_const + self.num_input + self.num_copy + self.num_neg
    }
}

/// Metadata for the entire circuit.
#[derive(Clone, Debug)]
pub struct CircuitMetadata {
    /// Total depth (number of layers).
    pub depth: usize,
    /// Total number of gates.
    pub total_gates: usize,
    /// Maximum layer width.
    pub max_width: usize,
    /// Per-layer statistics.
    pub layer_stats: Vec<LayerStats>,
    /// Description of the computation.
    pub description: String,
}

/// Wiring function representation for GKR.
///
/// For each gate g at position `pos` in layer i, the wiring function tells us:
/// - Is it an ADD gate? If so, what are the input positions?
/// - Is it a MUL gate? If so, what are the input positions?
#[derive(Clone, Debug)]
pub struct WiringFunction<F: PrimeField> {
    /// Mapping from gate index to (gate_type, left_input, right_input).
    pub wiring: HashMap<usize, (GateType, usize, usize)>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> WiringFunction<F> {
    /// Creates a wiring function from a layer.
    pub fn from_layer(layer: &Layer<F>) -> Self {
        let mut wiring = HashMap::new();
        for gate in &layer.gates {
            wiring.insert(gate.index, (gate.gate_type, gate.left_input, gate.right_input));
        }
        Self {
            wiring,
            _marker: PhantomData,
        }
    }

    /// Checks if gate at position is an ADD gate with given inputs.
    pub fn is_add(&self, gate_idx: usize, left: usize, right: usize) -> bool {
        self.wiring.get(&gate_idx)
            .map(|(t, l, r)| *t == GateType::Add && *l == left && *r == right)
            .unwrap_or(false)
    }

    /// Checks if gate at position is a MUL gate with given inputs.
    pub fn is_mul(&self, gate_idx: usize, left: usize, right: usize) -> bool {
        self.wiring.get(&gate_idx)
            .map(|(t, l, r)| *t == GateType::Mul && *l == left && *r == right)
            .unwrap_or(false)
    }

    /// Gets the wiring for a gate.
    pub fn get(&self, gate_idx: usize) -> Option<(GateType, usize, usize)> {
        self.wiring.get(&gate_idx).copied()
    }
}

/// The complete layered circuit.
#[derive(Clone, Debug)]
pub struct LayeredCircuit<F: PrimeField> {
    /// Layers from output (0) to input (depth-1).
    pub layers: Vec<Layer<F>>,
    /// Number of public inputs.
    pub num_inputs: usize,
    /// Number of public outputs.
    pub num_outputs: usize,
    /// Wiring functions for each layer.
    pub wiring: Vec<WiringFunction<F>>,
    /// Circuit metadata.
    pub metadata: CircuitMetadata,
}

impl<F: PrimeField> LayeredCircuit<F> {
    /// Creates an empty layered circuit.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            num_inputs: 0,
            num_outputs: 0,
            wiring: Vec::new(),
            metadata: CircuitMetadata {
                depth: 0,
                total_gates: 0,
                max_width: 0,
                layer_stats: Vec::new(),
                description: String::new(),
            },
        }
    }

    /// Returns the circuit depth.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Returns the total number of gates.
    pub fn total_gates(&self) -> usize {
        self.layers.iter().map(|l| l.size()).sum()
    }

    /// Adds a new layer (at the input end).
    pub fn add_layer(&mut self) -> usize {
        let layer_idx = self.layers.len();
        self.layers.push(Layer::new(layer_idx));
        layer_idx
    }

    /// Gets a mutable reference to a layer.
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut Layer<F>> {
        self.layers.get_mut(idx)
    }

    /// Gets a reference to a layer.
    pub fn layer(&self, idx: usize) -> Option<&Layer<F>> {
        self.layers.get(idx)
    }

    /// Finalizes the circuit, computing wiring functions and metadata.
    pub fn finalize(&mut self) {
        // Compute wiring functions
        self.wiring = self.layers.iter().map(WiringFunction::from_layer).collect();

        // Compute metadata
        let layer_stats: Vec<LayerStats> = self.layers.iter().map(LayerStats::from_layer).collect();
        let total_gates = layer_stats.iter().map(|s| s.total()).sum();
        let max_width = self.layers.iter().map(|l| l.size()).max().unwrap_or(0);

        self.metadata = CircuitMetadata {
            depth: self.layers.len(),
            total_gates,
            max_width,
            layer_stats,
            description: String::new(),
        };

        // Pad layers to powers of 2
        for layer in &mut self.layers {
            layer.pad_to_power_of_two();
        }
    }

    /// Evaluates the circuit on given inputs.
    pub fn evaluate(&self, inputs: &[F]) -> Vec<F> {
        if self.layers.is_empty() {
            return vec![];
        }

        // Start from the input layer (last layer)
        let mut current_values = inputs.to_vec();

        // Pad to layer size if needed
        let input_layer = &self.layers[self.layers.len() - 1];
        while current_values.len() < input_layer.size() {
            current_values.push(F::ZERO);
        }

        // Evaluate layer by layer from input to output
        for layer in self.layers.iter().rev().skip(1) {
            current_values = layer.evaluate(&current_values);
        }

        current_values
    }

    /// Gets layer values for all intermediate layers (for witness generation).
    pub fn get_all_layer_values(&self, inputs: &[F]) -> Vec<Vec<F>> {
        if self.layers.is_empty() {
            return vec![];
        }

        let mut all_values = Vec::with_capacity(self.layers.len());

        // Start from the input layer
        let mut current_values = inputs.to_vec();
        let input_layer = &self.layers[self.layers.len() - 1];
        while current_values.len() < input_layer.size() {
            current_values.push(F::ZERO);
        }

        // Store input layer values
        all_values.push(current_values.clone());

        // Evaluate and store each layer
        for layer in self.layers.iter().rev().skip(1) {
            current_values = layer.evaluate(&current_values);
            all_values.push(current_values.clone());
        }

        // Reverse so index 0 is output layer
        all_values.reverse();
        all_values
    }
}

impl<F: PrimeField> Default for LayeredCircuit<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing layered circuits.
pub struct LayeredCircuitBuilder<F: PrimeField> {
    circuit: LayeredCircuit<F>,
    current_layer: usize,
}

impl<F: PrimeField> LayeredCircuitBuilder<F> {
    /// Creates a new builder.
    pub fn new() -> Self {
        let mut circuit = LayeredCircuit::new();
        circuit.add_layer();
        Self {
            circuit,
            current_layer: 0,
        }
    }

    /// Adds inputs to the circuit.
    pub fn add_inputs(&mut self, count: usize) -> Vec<usize> {
        let layer = self.circuit.layer_mut(self.current_layer).unwrap();
        let start = layer.size();
        for _ in 0..count {
            layer.add_gate(Gate::input(0));
        }
        self.circuit.num_inputs = count;
        (start..start + count).collect()
    }

    /// Adds a new computation layer.
    pub fn new_layer(&mut self) {
        self.circuit.add_layer();
        self.current_layer = self.circuit.layers.len() - 1;
    }

    /// Adds an add gate to the current layer.
    pub fn add(&mut self, left: usize, right: usize) -> usize {
        let layer = self.circuit.layer_mut(self.current_layer).unwrap();
        layer.add_gate(Gate::add(0, left, right))
    }

    /// Adds a mul gate to the current layer.
    pub fn mul(&mut self, left: usize, right: usize) -> usize {
        let layer = self.circuit.layer_mut(self.current_layer).unwrap();
        layer.add_gate(Gate::mul(0, left, right))
    }

    /// Adds a constant gate.
    pub fn constant(&mut self, value: F) -> usize {
        let layer = self.circuit.layer_mut(self.current_layer).unwrap();
        layer.add_gate(Gate::constant(0, value))
    }

    /// Adds a copy gate.
    pub fn copy(&mut self, source: usize) -> usize {
        let layer = self.circuit.layer_mut(self.current_layer).unwrap();
        layer.add_gate(Gate::copy(0, source))
    }

    /// Adds a negation gate.
    pub fn neg(&mut self, source: usize) -> usize {
        let layer = self.circuit.layer_mut(self.current_layer).unwrap();
        layer.add_gate(Gate::neg(0, source))
    }

    /// Sets the number of outputs.
    pub fn set_outputs(&mut self, count: usize) {
        self.circuit.num_outputs = count;
    }

    /// Builds the circuit.
    pub fn build(mut self) -> LayeredCircuit<F> {
        self.circuit.finalize();
        self.circuit
    }

    /// Creates a circuit for a dot product.
    pub fn dot_product(n: usize) -> LayeredCircuit<F> {
        let mut builder = Self::new();

        // Input layer: a0, a1, ..., an-1, b0, b1, ..., bn-1
        let inputs = builder.add_inputs(2 * n);

        // Multiplication layer: a0*b0, a1*b1, ..., an-1*bn-1
        builder.new_layer();
        let products: Vec<usize> = (0..n)
            .map(|i| builder.mul(inputs[i], inputs[n + i]))
            .collect();

        // Addition tree
        let mut current = products;
        while current.len() > 1 {
            builder.new_layer();
            let mut next = Vec::new();
            for chunk in current.chunks(2) {
                if chunk.len() == 2 {
                    next.push(builder.add(chunk[0], chunk[1]));
                } else {
                    next.push(builder.copy(chunk[0]));
                }
            }
            current = next;
        }

        builder.set_outputs(1);
        builder.build()
    }

    /// Creates a circuit for matrix multiplication C = A * B.
    ///
    /// A is m x k, B is k x n, C is m x n.
    pub fn matmul(m: usize, k: usize, n: usize) -> LayeredCircuit<F> {
        let mut builder = Self::new();

        // Inputs: A (m*k values) then B (k*n values)
        let inputs = builder.add_inputs(m * k + k * n);

        // For each output cell C[i][j], compute dot product of row i of A and column j of B
        let mut output_gates = Vec::with_capacity(m * n);

        for i in 0..m {
            for j in 0..n {
                // Multiplication layer for this output
                builder.new_layer();

                let products: Vec<usize> = (0..k)
                    .map(|kk| {
                        let a_idx = i * k + kk; // A[i][kk]
                        let b_idx = m * k + kk * n + j; // B[kk][j]
                        builder.mul(inputs[a_idx], inputs[b_idx])
                    })
                    .collect();

                // Sum the products
                let mut current = products;
                while current.len() > 1 {
                    builder.new_layer();
                    let mut next = Vec::new();
                    for chunk in current.chunks(2) {
                        if chunk.len() == 2 {
                            next.push(builder.add(chunk[0], chunk[1]));
                        } else {
                            next.push(builder.copy(chunk[0]));
                        }
                    }
                    current = next;
                }

                output_gates.push(current[0]);
            }
        }

        builder.set_outputs(m * n);
        builder.build()
    }
}

impl<F: PrimeField> Default for LayeredCircuitBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::arithmetic::Field;
    use halo2curves::bn256::Fr;

    #[test]
    fn test_gate_evaluation() {
        let add_gate = Gate::<Fr>::add(0, 0, 1);
        assert_eq!(add_gate.evaluate(Fr::from(3), Fr::from(5)), Fr::from(8));

        let mul_gate = Gate::<Fr>::mul(0, 0, 1);
        assert_eq!(mul_gate.evaluate(Fr::from(3), Fr::from(5)), Fr::from(15));

        let const_gate = Gate::<Fr>::constant(0, Fr::from(42));
        assert_eq!(const_gate.evaluate(Fr::ZERO, Fr::ZERO), Fr::from(42));
    }

    #[test]
    fn test_layer_evaluation() {
        let mut layer = Layer::<Fr>::new(0);

        // Add gates: out[0] = in[0] + in[1], out[1] = in[2] * in[3]
        layer.add_gate(Gate::add(0, 0, 1));
        layer.add_gate(Gate::mul(0, 2, 3));

        let inputs = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
        let outputs = layer.evaluate(&inputs);

        assert_eq!(outputs[0], Fr::from(3)); // 1 + 2
        assert_eq!(outputs[1], Fr::from(12)); // 3 * 4
    }

    #[test]
    fn test_dot_product_circuit() {
        let circuit = LayeredCircuitBuilder::<Fr>::dot_product(4);

        // Inputs: a = [1, 2, 3, 4], b = [5, 6, 7, 8]
        // Dot product: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let inputs: Vec<Fr> = vec![1, 2, 3, 4, 5, 6, 7, 8]
            .into_iter()
            .map(Fr::from)
            .collect();

        let outputs = circuit.evaluate(&inputs);
        assert_eq!(outputs[0], Fr::from(70u64));
    }

    #[test]
    fn test_circuit_metadata() {
        let circuit = LayeredCircuitBuilder::<Fr>::dot_product(4);

        assert!(circuit.depth() >= 2); // At least mul layer and add layers
        assert!(circuit.total_gates() >= 4 + 3); // 4 muls + at least 3 adds
        assert_eq!(circuit.num_inputs, 8);
        assert_eq!(circuit.num_outputs, 1);
    }

    #[test]
    fn test_wiring_function() {
        let mut layer = Layer::<Fr>::new(0);
        layer.add_gate(Gate::add(0, 0, 1));
        layer.add_gate(Gate::mul(0, 2, 3));

        let wiring = WiringFunction::from_layer(&layer);

        assert!(wiring.is_add(0, 0, 1));
        assert!(!wiring.is_mul(0, 0, 1));
        assert!(wiring.is_mul(1, 2, 3));
    }

    #[test]
    fn test_get_all_layer_values() {
        let mut builder = LayeredCircuitBuilder::<Fr>::new();
        let inputs = builder.add_inputs(2);
        builder.new_layer();
        let _sum = builder.add(inputs[0], inputs[1]);
        let circuit = builder.build();

        let input_vals = vec![Fr::from(3), Fr::from(5)];
        let all_values = circuit.get_all_layer_values(&input_vals);

        // Should have 2 layers: output and input
        assert!(all_values.len() >= 1);

        // Output layer should contain the sum
        assert!(all_values[0].contains(&Fr::from(8)));
    }
}
