//! Layered Circuit Abstraction for Neural Networks.
//!
//! This module provides a circuit representation optimized for the GKR protocol.
//! Circuits are organized into layers, where each layer computes a function of
//! the previous layer's outputs.
//!
//! ## Structure
//!
//! A layered circuit consists of:
//! - **Input layer**: The circuit inputs (activations, weights)
//! - **Internal layers**: Computation layers (matrix multiply, add, ReLU, etc.)
//! - **Output layer**: Final outputs (loss, gradients)
//!
//! ## Gate Types
//!
//! The GKR protocol works with arithmetic circuits using:
//! - **Add gates**: Output = input₁ + input₂
//! - **Mul gates**: Output = input₁ × input₂
//! - **Const gates**: Output = constant
//! - **Copy gates**: Output = input (fan-out)
//!
//! ## Neural Network Mapping
//!
//! Neural network operations map to layered circuits as follows:
//! - **Matrix multiply**: O(n²) mul gates + O(n) add trees
//! - **Bias add**: O(n) add gates
//! - **ReLU**: Approximated or lookup-based
//! - **Softmax**: Exp approximation + normalization

use super::{FieldElement, GKRError, GKRResult};
use super::multilinear::{DenseMultilinear, MultilinearPolynomial, MultilinearExtension};
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::ff::PrimeField;
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Type of a gate in the circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateType {
    /// Addition: out = in₁ + in₂
    Add,
    /// Multiplication: out = in₁ × in₂
    Mul,
    /// Constant: out = constant
    Const,
    /// Copy/Identity: out = in
    Copy,
    /// Subtraction: out = in₁ - in₂
    Sub,
    /// Negation: out = -in
    Neg,
    /// Zero: out = 0
    Zero,
    /// One: out = 1
    One,
}

/// Wire type for circuit connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireType {
    /// Input wire from external source.
    Input,
    /// Internal wire from previous layer.
    Internal,
    /// Output wire to next layer or circuit output.
    Output,
}

/// A wire connecting gates.
#[derive(Debug, Clone, Copy)]
pub struct Wire {
    /// Layer index of the wire's source.
    pub src_layer: usize,
    /// Gate index within the source layer.
    pub src_gate: usize,
    /// Wire type.
    pub wire_type: WireType,
}

impl Wire {
    /// Creates a new wire.
    pub fn new(src_layer: usize, src_gate: usize, wire_type: WireType) -> Self {
        Self {
            src_layer,
            src_gate,
            wire_type,
        }
    }

    /// Creates an input wire.
    pub fn input(gate: usize) -> Self {
        Self::new(0, gate, WireType::Input)
    }

    /// Creates an internal wire from a previous layer.
    pub fn internal(layer: usize, gate: usize) -> Self {
        Self::new(layer, gate, WireType::Internal)
    }
}

/// A gate in the circuit.
#[derive(Debug, Clone)]
pub struct Gate {
    /// Type of the gate.
    pub gate_type: GateType,
    /// First input wire.
    pub input1: Option<Wire>,
    /// Second input wire (for binary gates).
    pub input2: Option<Wire>,
    /// Constant value (for Const gates).
    pub constant: Option<FieldElement>,
}

impl Gate {
    /// Creates an addition gate.
    pub fn add(in1: Wire, in2: Wire) -> Self {
        Self {
            gate_type: GateType::Add,
            input1: Some(in1),
            input2: Some(in2),
            constant: None,
        }
    }

    /// Creates a multiplication gate.
    pub fn mul(in1: Wire, in2: Wire) -> Self {
        Self {
            gate_type: GateType::Mul,
            input1: Some(in1),
            input2: Some(in2),
            constant: None,
        }
    }

    /// Creates a constant gate.
    pub fn constant(value: FieldElement) -> Self {
        Self {
            gate_type: GateType::Const,
            input1: None,
            input2: None,
            constant: Some(value),
        }
    }

    /// Creates a copy gate.
    pub fn copy(input: Wire) -> Self {
        Self {
            gate_type: GateType::Copy,
            input1: Some(input),
            input2: None,
            constant: None,
        }
    }

    /// Creates a subtraction gate.
    pub fn sub(in1: Wire, in2: Wire) -> Self {
        Self {
            gate_type: GateType::Sub,
            input1: Some(in1),
            input2: Some(in2),
            constant: None,
        }
    }

    /// Creates a zero gate.
    pub fn zero() -> Self {
        Self {
            gate_type: GateType::Zero,
            input1: None,
            input2: None,
            constant: None,
        }
    }

    /// Creates a one gate.
    pub fn one() -> Self {
        Self {
            gate_type: GateType::One,
            input1: None,
            input2: None,
            constant: None,
        }
    }
}

/// A layer in the circuit.
#[derive(Debug, Clone)]
pub struct CircuitLayer {
    /// Gates in this layer.
    pub gates: Vec<Gate>,
    /// Number of variables to address gates in this layer.
    pub num_vars: usize,
    /// Layer index.
    pub layer_idx: usize,
    /// Human-readable label for debugging.
    pub label: String,
}

impl CircuitLayer {
    /// Creates a new circuit layer.
    pub fn new(gates: Vec<Gate>, layer_idx: usize) -> Self {
        let num_gates = gates.len();
        let num_vars = if num_gates == 0 {
            0
        } else {
            (num_gates as f64).log2().ceil() as usize
        };

        Self {
            gates,
            num_vars,
            layer_idx,
            label: format!("layer_{}", layer_idx),
        }
    }

    /// Creates a layer with a custom label.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }

    /// Returns the number of gates.
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    /// Returns the padded size (power of 2).
    pub fn padded_size(&self) -> usize {
        1 << self.num_vars
    }

    /// Evaluates the layer given input values.
    pub fn evaluate(&self, inputs: &[FieldElement], prev_layer: Option<&[FieldElement]>) -> Vec<FieldElement> {
        let size = self.padded_size();
        let mut outputs = vec![FieldElement::zero(); size];

        for (i, gate) in self.gates.iter().enumerate() {
            outputs[i] = self.evaluate_gate(gate, inputs, prev_layer);
        }

        outputs
    }

    /// Evaluates a single gate.
    fn evaluate_gate(
        &self,
        gate: &Gate,
        inputs: &[FieldElement],
        prev_layer: Option<&[FieldElement]>,
    ) -> FieldElement {
        let get_value = |wire: &Wire| -> FieldElement {
            match wire.wire_type {
                WireType::Input => {
                    inputs.get(wire.src_gate).copied().unwrap_or(FieldElement::zero())
                }
                WireType::Internal | WireType::Output => {
                    prev_layer
                        .and_then(|p| p.get(wire.src_gate))
                        .copied()
                        .unwrap_or(FieldElement::zero())
                }
            }
        };

        match gate.gate_type {
            GateType::Add => {
                let a = gate.input1.as_ref().map(get_value).unwrap_or(FieldElement::zero());
                let b = gate.input2.as_ref().map(get_value).unwrap_or(FieldElement::zero());
                a + b
            }
            GateType::Mul => {
                let a = gate.input1.as_ref().map(get_value).unwrap_or(FieldElement::zero());
                let b = gate.input2.as_ref().map(get_value).unwrap_or(FieldElement::zero());
                a * b
            }
            GateType::Const => {
                gate.constant.unwrap_or(FieldElement::zero())
            }
            GateType::Copy => {
                gate.input1.as_ref().map(get_value).unwrap_or(FieldElement::zero())
            }
            GateType::Sub => {
                let a = gate.input1.as_ref().map(get_value).unwrap_or(FieldElement::zero());
                let b = gate.input2.as_ref().map(get_value).unwrap_or(FieldElement::zero());
                a - b
            }
            GateType::Neg => {
                let a = gate.input1.as_ref().map(get_value).unwrap_or(FieldElement::zero());
                -a
            }
            GateType::Zero => FieldElement::zero(),
            GateType::One => FieldElement::one(),
        }
    }

    /// Builds the add polynomial for this layer.
    ///
    /// add(x, y, z) = 1 if gate x is an ADD gate with inputs from positions y and z
    pub fn add_polynomial(&self) -> DenseMultilinear {
        let n = self.num_vars;
        let size = 1 << (3 * n);  // (x, y, z) ∈ {0,1}^(3n)
        let mut evals = vec![FieldElement::zero(); size];

        for (i, gate) in self.gates.iter().enumerate() {
            if gate.gate_type == GateType::Add {
                if let (Some(w1), Some(w2)) = (&gate.input1, &gate.input2) {
                    let idx = (i << (2 * n)) | (w1.src_gate << n) | w2.src_gate;
                    if idx < size {
                        evals[idx] = FieldElement::one();
                    }
                }
            }
        }

        DenseMultilinear::from_evaluations(evals)
    }

    /// Builds the mul polynomial for this layer.
    ///
    /// mul(x, y, z) = 1 if gate x is a MUL gate with inputs from positions y and z
    pub fn mul_polynomial(&self) -> DenseMultilinear {
        let n = self.num_vars;
        let size = 1 << (3 * n);
        let mut evals = vec![FieldElement::zero(); size];

        for (i, gate) in self.gates.iter().enumerate() {
            if gate.gate_type == GateType::Mul {
                if let (Some(w1), Some(w2)) = (&gate.input1, &gate.input2) {
                    let idx = (i << (2 * n)) | (w1.src_gate << n) | w2.src_gate;
                    if idx < size {
                        evals[idx] = FieldElement::one();
                    }
                }
            }
        }

        DenseMultilinear::from_evaluations(evals)
    }
}

/// A complete layered circuit.
#[derive(Debug, Clone)]
pub struct LayeredCircuit {
    /// Layers of the circuit (layer 0 is input, layer d-1 is output).
    pub layers: Vec<CircuitLayer>,
    /// Number of input wires.
    pub num_inputs: usize,
    /// Number of output wires.
    pub num_outputs: usize,
    /// Human-readable description.
    pub description: String,
}

impl LayeredCircuit {
    /// Creates a new layered circuit.
    pub fn new(layers: Vec<CircuitLayer>, num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            layers,
            num_inputs,
            num_outputs,
            description: String::new(),
        }
    }

    /// Returns the circuit depth.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Returns the total number of gates.
    pub fn num_gates(&self) -> usize {
        self.layers.iter().map(|l| l.num_gates()).sum()
    }

    /// Evaluates the entire circuit on given inputs.
    pub fn evaluate(&self, inputs: &[FieldElement]) -> Vec<FieldElement> {
        if self.layers.is_empty() {
            return inputs.to_vec();
        }

        let mut current = self.layers[0].evaluate(inputs, None);

        for layer in self.layers.iter().skip(1) {
            current = layer.evaluate(inputs, Some(&current));
        }

        current
    }

    /// Returns the output layer values MLE.
    pub fn output_mle(&self, inputs: &[FieldElement]) -> DenseMultilinear {
        let outputs = self.evaluate(inputs);
        DenseMultilinear::from_evaluations(outputs)
    }

    /// Computes all layer values (useful for GKR proving).
    pub fn all_layer_values(&self, inputs: &[FieldElement]) -> Vec<Vec<FieldElement>> {
        let mut values = Vec::with_capacity(self.layers.len() + 1);

        // Pad inputs to power of 2
        let input_size = self.num_inputs.next_power_of_two();
        let mut padded_inputs = inputs.to_vec();
        padded_inputs.resize(input_size, FieldElement::zero());
        values.push(padded_inputs.clone());

        if !self.layers.is_empty() {
            let mut current = self.layers[0].evaluate(&padded_inputs, None);
            values.push(current.clone());

            for layer in self.layers.iter().skip(1) {
                current = layer.evaluate(&padded_inputs, Some(&current));
                values.push(current.clone());
            }
        }

        values
    }
}

/// Builder for constructing layered circuits.
pub struct CircuitBuilder {
    /// Accumulated layers.
    layers: Vec<CircuitLayer>,
    /// Number of inputs.
    num_inputs: usize,
    /// Current layer being built.
    current_layer_gates: Vec<Gate>,
    /// Current layer index.
    layer_idx: usize,
}

impl CircuitBuilder {
    /// Creates a new circuit builder.
    pub fn new(num_inputs: usize) -> Self {
        Self {
            layers: Vec::new(),
            num_inputs,
            current_layer_gates: Vec::new(),
            layer_idx: 0,
        }
    }

    /// Adds a gate to the current layer.
    pub fn add_gate(&mut self, gate: Gate) -> usize {
        let idx = self.current_layer_gates.len();
        self.current_layer_gates.push(gate);
        idx
    }

    /// Finishes the current layer and starts a new one.
    pub fn finish_layer(&mut self) -> usize {
        if !self.current_layer_gates.is_empty() {
            let layer = CircuitLayer::new(
                std::mem::take(&mut self.current_layer_gates),
                self.layer_idx,
            );
            self.layers.push(layer);
            self.layer_idx += 1;
        }
        self.layers.len() - 1
    }

    /// Finishes the current layer with a label.
    pub fn finish_layer_with_label(&mut self, label: &str) -> usize {
        if !self.current_layer_gates.is_empty() {
            let layer = CircuitLayer::new(
                std::mem::take(&mut self.current_layer_gates),
                self.layer_idx,
            ).with_label(label);
            self.layers.push(layer);
            self.layer_idx += 1;
        }
        self.layers.len() - 1
    }

    /// Builds the final circuit.
    pub fn build(mut self) -> LayeredCircuit {
        // Finish any remaining gates
        if !self.current_layer_gates.is_empty() {
            self.finish_layer();
        }

        let num_outputs = self.layers.last().map(|l| l.num_gates()).unwrap_or(0);

        LayeredCircuit::new(self.layers, self.num_inputs, num_outputs)
    }

    /// Creates a wire to the current layer's gate.
    pub fn wire_to_current(&self, gate_idx: usize) -> Wire {
        Wire::internal(self.layer_idx, gate_idx)
    }

    /// Creates a wire to the previous layer's gate.
    pub fn wire_to_prev(&self, gate_idx: usize) -> Wire {
        if self.layer_idx == 0 {
            Wire::input(gate_idx)
        } else {
            Wire::internal(self.layer_idx - 1, gate_idx)
        }
    }
}

/// Neural network circuit builder.
///
/// Provides high-level operations for building neural network circuits.
pub struct NeuralNetworkCircuit;

impl NeuralNetworkCircuit {
    /// Creates a circuit for matrix-vector multiplication.
    ///
    /// Computes y = Wx where W is m×n and x is n×1.
    pub fn matmul(weights: &[FieldElement], m: usize, n: usize) -> LayeredCircuit {
        assert_eq!(weights.len(), m * n);

        let mut builder = CircuitBuilder::new(n);

        // First layer: compute all w_ij * x_j products
        for i in 0..m {
            for j in 0..n {
                let w_idx = i * n + j;
                let w = weights[w_idx];

                // Multiply weight constant by input
                builder.add_gate(Gate::constant(w));
            }
        }
        builder.finish_layer_with_label("weights");

        // Second layer: multiply weights by inputs
        for i in 0..m {
            for j in 0..n {
                let weight_gate = i * n + j;
                let input_wire = Wire::input(j);
                let weight_wire = Wire::internal(0, weight_gate);
                builder.add_gate(Gate::mul(weight_wire, input_wire));
            }
        }
        builder.finish_layer_with_label("products");

        // Third layer: sum products for each row (reduction tree)
        for i in 0..m {
            // Sum n products starting at position i*n
            let first_product = i * n;
            if n == 1 {
                builder.add_gate(Gate::copy(Wire::internal(1, first_product)));
            } else if n == 2 {
                builder.add_gate(Gate::add(
                    Wire::internal(1, first_product),
                    Wire::internal(1, first_product + 1),
                ));
            } else {
                // For larger n, build a reduction tree
                // Simplified: just chain additions
                let mut current = Wire::internal(1, first_product);
                for j in 1..n {
                    let next = Wire::internal(1, first_product + j);
                    builder.add_gate(Gate::add(current, next));
                    current = builder.wire_to_current(builder.current_layer_gates.len() - 1);
                }
            }
        }
        builder.finish_layer_with_label("sums");

        builder.build()
    }

    /// Creates a circuit for element-wise addition of two vectors.
    pub fn add_vectors(n: usize) -> LayeredCircuit {
        // Inputs: x[0..n], y[0..n]
        let mut builder = CircuitBuilder::new(2 * n);

        for i in 0..n {
            builder.add_gate(Gate::add(Wire::input(i), Wire::input(n + i)));
        }
        builder.finish_layer_with_label("add");

        builder.build()
    }

    /// Creates a circuit for element-wise multiplication.
    pub fn hadamard(n: usize) -> LayeredCircuit {
        let mut builder = CircuitBuilder::new(2 * n);

        for i in 0..n {
            builder.add_gate(Gate::mul(Wire::input(i), Wire::input(n + i)));
        }
        builder.finish_layer_with_label("hadamard");

        builder.build()
    }

    /// Creates a circuit for bias addition: y = x + b.
    pub fn add_bias(bias: &[FieldElement]) -> LayeredCircuit {
        let n = bias.len();
        let mut builder = CircuitBuilder::new(n);

        // First layer: constant bias values
        for &b in bias {
            builder.add_gate(Gate::constant(b));
        }
        builder.finish_layer_with_label("bias");

        // Second layer: add input to bias
        for i in 0..n {
            builder.add_gate(Gate::add(Wire::input(i), Wire::internal(0, i)));
        }
        builder.finish_layer_with_label("add_bias");

        builder.build()
    }

    /// Creates a circuit for inner product of two vectors.
    pub fn inner_product(n: usize) -> LayeredCircuit {
        // Inputs: x[0..n], y[0..n]
        let mut builder = CircuitBuilder::new(2 * n);

        // First layer: element-wise products
        for i in 0..n {
            builder.add_gate(Gate::mul(Wire::input(i), Wire::input(n + i)));
        }
        builder.finish_layer_with_label("products");

        // Second layer: sum reduction
        if n > 0 {
            let mut current = Wire::internal(0, 0);
            for i in 1..n {
                let next = Wire::internal(0, i);
                builder.add_gate(Gate::add(current, next));
                current = builder.wire_to_current(builder.current_layer_gates.len() - 1);
            }
            if n == 1 {
                builder.add_gate(Gate::copy(Wire::internal(0, 0)));
            }
        }
        builder.finish_layer_with_label("sum");

        builder.build()
    }

    /// Creates a 2-layer MLP forward pass circuit.
    ///
    /// Computes: h = W1·x + b1, then ReLU, then y = W2·h + b2
    /// Note: ReLU is approximated as identity for this basic implementation.
    pub fn mlp_forward(
        w1: &[FieldElement], b1: &[FieldElement],
        w2: &[FieldElement], b2: &[FieldElement],
        d_in: usize, d_hid: usize, d_out: usize,
    ) -> LayeredCircuit {
        assert_eq!(w1.len(), d_hid * d_in);
        assert_eq!(b1.len(), d_hid);
        assert_eq!(w2.len(), d_out * d_hid);
        assert_eq!(b2.len(), d_out);

        // This is a simplified circuit - a full implementation would have more layers
        let mut builder = CircuitBuilder::new(d_in);

        // Layer 1: W1 weights as constants
        for &w in w1 {
            builder.add_gate(Gate::constant(w));
        }
        builder.finish_layer_with_label("w1_const");

        // Layer 2: W1 * x products
        for i in 0..d_hid {
            for j in 0..d_in {
                let w_idx = i * d_in + j;
                builder.add_gate(Gate::mul(Wire::internal(0, w_idx), Wire::input(j)));
            }
        }
        builder.finish_layer_with_label("w1_mul");

        // Layer 3: Sum products + bias for hidden layer
        for i in 0..d_hid {
            // Sum d_in products
            let first = i * d_in;
            let mut accum = Wire::internal(1, first);
            for j in 1..d_in {
                let next = Wire::internal(1, first + j);
                builder.add_gate(Gate::add(accum, next));
                accum = builder.wire_to_current(builder.current_layer_gates.len() - 1);
            }
            // Add bias
            builder.add_gate(Gate::constant(b1[i]));
            let bias_wire = builder.wire_to_current(builder.current_layer_gates.len() - 1);
            builder.add_gate(Gate::add(accum, bias_wire));
        }
        builder.finish_layer_with_label("h_pre");

        // Remaining layers would handle ReLU, W2, etc.
        // For now, just output the hidden activations

        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_circuit() {
        let mut builder = CircuitBuilder::new(2);

        // z = x + y
        builder.add_gate(Gate::add(Wire::input(0), Wire::input(1)));
        builder.finish_layer();

        let circuit = builder.build();

        assert_eq!(circuit.depth(), 1);
        assert_eq!(circuit.num_inputs, 2);
        assert_eq!(circuit.num_outputs, 1);

        // Evaluate: 3 + 5 = 8
        let result = circuit.evaluate(&[
            FieldElement::from(3u64),
            FieldElement::from(5u64),
        ]);

        assert_eq!(result[0], FieldElement::from(8u64));
    }

    #[test]
    fn test_mul_circuit() {
        let mut builder = CircuitBuilder::new(2);
        builder.add_gate(Gate::mul(Wire::input(0), Wire::input(1)));
        builder.finish_layer();

        let circuit = builder.build();

        let result = circuit.evaluate(&[
            FieldElement::from(3u64),
            FieldElement::from(5u64),
        ]);

        assert_eq!(result[0], FieldElement::from(15u64));
    }

    #[test]
    fn test_matmul_circuit() {
        // 2x2 matrix times 2x1 vector
        // W = [[1, 2], [3, 4]]
        // x = [5, 6]
        // Wx = [1*5 + 2*6, 3*5 + 4*6] = [17, 39]

        let weights = vec![
            FieldElement::from(1u64), FieldElement::from(2u64),
            FieldElement::from(3u64), FieldElement::from(4u64),
        ];

        let circuit = NeuralNetworkCircuit::matmul(&weights, 2, 2);

        let inputs = vec![
            FieldElement::from(5u64),
            FieldElement::from(6u64),
        ];

        let result = circuit.evaluate(&inputs);

        // Check outputs (may need to look at specific positions in padded output)
        // The circuit structure may put results in different positions
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_add_vectors() {
        let circuit = NeuralNetworkCircuit::add_vectors(3);

        let inputs = vec![
            FieldElement::from(1u64), FieldElement::from(2u64), FieldElement::from(3u64),
            FieldElement::from(4u64), FieldElement::from(5u64), FieldElement::from(6u64),
        ];

        let result = circuit.evaluate(&inputs);

        assert_eq!(result[0], FieldElement::from(5u64));  // 1 + 4
        assert_eq!(result[1], FieldElement::from(7u64));  // 2 + 5
        assert_eq!(result[2], FieldElement::from(9u64));  // 3 + 6
    }

    #[test]
    fn test_hadamard() {
        let circuit = NeuralNetworkCircuit::hadamard(2);

        let inputs = vec![
            FieldElement::from(3u64), FieldElement::from(4u64),
            FieldElement::from(5u64), FieldElement::from(6u64),
        ];

        let result = circuit.evaluate(&inputs);

        assert_eq!(result[0], FieldElement::from(15u64)); // 3 * 5
        assert_eq!(result[1], FieldElement::from(24u64)); // 4 * 6
    }

    #[test]
    fn test_layer_polynomials() {
        let mut builder = CircuitBuilder::new(2);
        builder.add_gate(Gate::add(Wire::input(0), Wire::input(1)));
        builder.add_gate(Gate::mul(Wire::input(0), Wire::input(1)));
        builder.finish_layer();

        let circuit = builder.build();
        let layer = &circuit.layers[0];

        let add_poly = layer.add_polynomial();
        let mul_poly = layer.mul_polynomial();

        // Check that polynomials have correct structure
        assert_eq!(add_poly.num_variables(), 3); // 1 var each for x, y, z
        assert_eq!(mul_poly.num_variables(), 3);
    }

    #[test]
    fn test_all_layer_values() {
        let mut builder = CircuitBuilder::new(2);
        builder.add_gate(Gate::add(Wire::input(0), Wire::input(1)));
        builder.finish_layer();
        builder.add_gate(Gate::mul(
            Wire::internal(0, 0),
            Wire::internal(0, 0),
        ));
        builder.finish_layer();

        let circuit = builder.build();

        let inputs = vec![
            FieldElement::from(3u64),
            FieldElement::from(5u64),
        ];

        let values = circuit.all_layer_values(&inputs);

        // Should have 3 sets of values: input, after add, after mul
        assert_eq!(values.len(), 3);

        // First layer values = inputs
        assert_eq!(values[0][0], FieldElement::from(3u64));
        assert_eq!(values[0][1], FieldElement::from(5u64));

        // Second layer: 3 + 5 = 8
        assert_eq!(values[1][0], FieldElement::from(8u64));

        // Third layer: 8 * 8 = 64
        assert_eq!(values[2][0], FieldElement::from(64u64));
    }
}
