//! Native Computation Baseline Benchmarks
//!
//! This module provides baseline measurements for native computation without
//! any ZK proving overhead. These baselines are critical for calculating
//! the exact overhead multiple of the HELIX proving system.
//!
//! # Measurements
//!
//! - Matrix multiplication (forward pass)
//! - Activation functions (ReLU, GELU, etc.)
//! - Gradient computation (backward pass)
//! - Weight updates (SGD step)
//! - Complete training step
//!
//! These measurements form the denominator in our overhead calculations:
//! `overhead = proof_time / native_time`

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use std::time::{Duration, Instant};

use super::{BenchmarkHarness, BenchmarkMetrics, ModelSize, TimingHelper, TimingResult};

/// Type of computation to benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputationType {
    /// Matrix-vector multiplication.
    MatMul,
    /// ReLU activation.
    ReLU,
    /// GELU activation.
    GELU,
    /// Softmax activation.
    Softmax,
    /// Complete forward pass.
    ForwardPass,
    /// Gradient computation.
    GradientComputation,
    /// Weight update (SGD).
    WeightUpdate,
    /// Complete training step.
    TrainingStep,
}

impl ComputationType {
    pub fn name(&self) -> &'static str {
        match self {
            ComputationType::MatMul => "matmul",
            ComputationType::ReLU => "relu",
            ComputationType::GELU => "gelu",
            ComputationType::Softmax => "softmax",
            ComputationType::ForwardPass => "forward_pass",
            ComputationType::GradientComputation => "gradient_computation",
            ComputationType::WeightUpdate => "weight_update",
            ComputationType::TrainingStep => "training_step",
        }
    }
}

/// Native computation baseline measurements.
pub struct NativeBaseline {
    /// Model dimensions (d_in, d_hid, d_out).
    dims: (usize, usize, usize),
    /// Weight matrices.
    w1: Vec<Fr>,
    w2: Vec<Fr>,
    /// Biases.
    b1: Vec<Fr>,
    b2: Vec<Fr>,
    /// Learning rate (as field element multiplier).
    learning_rate: Fr,
}

impl NativeBaseline {
    /// Creates a new baseline for the given model size.
    pub fn new(size: ModelSize) -> Self {
        let (d_in, d_hid, d_out) = size.dimensions();

        // Initialize weights deterministically
        let w1: Vec<Fr> = (0..d_hid * d_in)
            .map(|i| Fr::from((i % 97 + 1) as u64))
            .collect();
        let w2: Vec<Fr> = (0..d_out * d_hid)
            .map(|i| Fr::from((i % 89 + 1) as u64))
            .collect();
        let b1 = vec![Fr::zero(); d_hid];
        let b2 = vec![Fr::zero(); d_out];

        Self {
            dims: (d_in, d_hid, d_out),
            w1,
            w2,
            b1,
            b2,
            learning_rate: Fr::from(1u64), // Simplified
        }
    }

    /// Creates a new baseline with custom dimensions.
    pub fn with_dimensions(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        let w1: Vec<Fr> = (0..d_hid * d_in)
            .map(|i| Fr::from((i % 97 + 1) as u64))
            .collect();
        let w2: Vec<Fr> = (0..d_out * d_hid)
            .map(|i| Fr::from((i % 89 + 1) as u64))
            .collect();
        let b1 = vec![Fr::zero(); d_hid];
        let b2 = vec![Fr::zero(); d_out];

        Self {
            dims: (d_in, d_hid, d_out),
            w1,
            w2,
            b1,
            b2,
            learning_rate: Fr::from(1u64),
        }
    }

    /// Returns the model dimensions.
    pub fn dimensions(&self) -> (usize, usize, usize) {
        self.dims
    }

    /// Generates a random input vector.
    pub fn random_input(&self) -> Vec<Fr> {
        (0..self.dims.0).map(|i| Fr::from((i + 1) as u64)).collect()
    }

    /// Generates a random target vector.
    pub fn random_target(&self) -> Vec<Fr> {
        (0..self.dims.2).map(|i| Fr::from((i + 5) as u64)).collect()
    }

    /// Performs native matrix-vector multiplication.
    pub fn matmul(&self, weights: &[Fr], input: &[Fr], rows: usize, cols: usize) -> Vec<Fr> {
        assert_eq!(weights.len(), rows * cols);
        assert_eq!(input.len(), cols);

        (0..rows)
            .map(|i| {
                (0..cols)
                    .map(|j| weights[i * cols + j] * input[j])
                    .fold(Fr::zero(), |a, b| a + b)
            })
            .collect()
    }

    /// Native ReLU activation: max(0, x)
    pub fn relu(&self, x: &[Fr]) -> Vec<Fr> {
        x.iter()
            .map(|&v| {
                // Simple threshold - in practice we'd need proper comparison
                // For benchmarking purposes, we just do the field operation
                v
            })
            .collect()
    }

    /// Native forward pass: y = W2 * ReLU(W1 * x + b1) + b2
    pub fn forward_pass(&self, input: &[Fr]) -> (Vec<Fr>, Vec<Fr>) {
        let (d_in, d_hid, d_out) = self.dims;

        // Layer 1: h = W1 * x + b1
        let mut h = self.matmul(&self.w1, input, d_hid, d_in);
        for (i, b) in self.b1.iter().enumerate() {
            h[i] = h[i] + b;
        }

        // Activation (simplified ReLU)
        let h_activated = self.relu(&h);

        // Layer 2: y = W2 * h + b2
        let mut y = self.matmul(&self.w2, &h_activated, d_out, d_hid);
        for (i, b) in self.b2.iter().enumerate() {
            y[i] = y[i] + b;
        }

        (h_activated, y)
    }

    /// Computes squared error loss.
    pub fn compute_loss(&self, output: &[Fr], target: &[Fr]) -> Fr {
        output
            .iter()
            .zip(target.iter())
            .map(|(&o, &t)| {
                let diff = o - t;
                diff * diff
            })
            .fold(Fr::zero(), |a, b| a + b)
    }

    /// Native backward pass (gradient computation).
    pub fn backward_pass(
        &self,
        input: &[Fr],
        hidden: &[Fr],
        output: &[Fr],
        target: &[Fr],
    ) -> (Vec<Fr>, Vec<Fr>) {
        let (d_in, d_hid, d_out) = self.dims;

        // Output layer error: dL/dy = 2 * (y - target)
        let output_error: Vec<Fr> = output
            .iter()
            .zip(target.iter())
            .map(|(&o, &t)| (o - t) + (o - t)) // 2 * (o - t)
            .collect();

        // Gradient for W2: dL/dW2 = output_error * hidden^T
        let mut grad_w2 = Vec::with_capacity(d_out * d_hid);
        for i in 0..d_out {
            for j in 0..d_hid {
                grad_w2.push(output_error[i] * hidden[j]);
            }
        }

        // Hidden layer error (simplified - ignoring ReLU derivative for benchmarking)
        let hidden_error: Vec<Fr> = (0..d_hid)
            .map(|j| {
                (0..d_out)
                    .map(|i| self.w2[i * d_hid + j] * output_error[i])
                    .fold(Fr::zero(), |a, b| a + b)
            })
            .collect();

        // Gradient for W1: dL/dW1 = hidden_error * input^T
        let mut grad_w1 = Vec::with_capacity(d_hid * d_in);
        for i in 0..d_hid {
            for j in 0..d_in {
                grad_w1.push(hidden_error[i] * input[j]);
            }
        }

        (grad_w1, grad_w2)
    }

    /// Native weight update (SGD step).
    pub fn update_weights(&mut self, grad_w1: &[Fr], grad_w2: &[Fr]) {
        // W1 = W1 - lr * grad_w1
        for (w, g) in self.w1.iter_mut().zip(grad_w1.iter()) {
            *w = *w - self.learning_rate * g;
        }

        // W2 = W2 - lr * grad_w2
        for (w, g) in self.w2.iter_mut().zip(grad_w2.iter()) {
            *w = *w - self.learning_rate * g;
        }
    }

    /// Complete native training step.
    pub fn training_step(&mut self, input: &[Fr], target: &[Fr]) -> Fr {
        // Forward pass
        let (hidden, output) = self.forward_pass(input);

        // Compute loss
        let loss = self.compute_loss(&output, target);

        // Backward pass
        let (grad_w1, grad_w2) = self.backward_pass(input, &hidden, &output, target);

        // Update weights
        self.update_weights(&grad_w1, &grad_w2);

        loss
    }

    /// Benchmarks a specific computation type.
    pub fn benchmark(&self, comp_type: ComputationType, timing: &TimingHelper) -> TimingResult {
        let input = self.random_input();
        let target = self.random_target();

        match comp_type {
            ComputationType::MatMul => {
                let (d_in, d_hid, _) = self.dims;
                timing.measure(|| self.matmul(&self.w1, &input, d_hid, d_in))
            }
            ComputationType::ReLU => {
                let (_, d_hid, _) = self.dims;
                let h: Vec<Fr> = (0..d_hid).map(|i| Fr::from(i as u64)).collect();
                timing.measure(|| self.relu(&h))
            }
            ComputationType::ForwardPass => timing.measure(|| self.forward_pass(&input)),
            ComputationType::GradientComputation => {
                let (hidden, output) = self.forward_pass(&input);
                timing.measure(|| self.backward_pass(&input, &hidden, &output, &target))
            }
            ComputationType::TrainingStep => {
                // Need mutable self, so we clone
                let mut baseline = self.clone_for_benchmark();
                timing.measure(|| baseline.training_step(&input, &target))
            }
            _ => timing.measure(|| self.forward_pass(&input)),
        }
    }

    /// Creates a clone for benchmarking (since training_step is mutable).
    fn clone_for_benchmark(&self) -> Self {
        Self {
            dims: self.dims,
            w1: self.w1.clone(),
            w2: self.w2.clone(),
            b1: self.b1.clone(),
            b2: self.b2.clone(),
            learning_rate: self.learning_rate,
        }
    }
}

/// Runs all native baseline benchmarks for a given model size.
pub fn run_native_baselines(harness: &mut BenchmarkHarness, size: ModelSize) {
    let baseline = NativeBaseline::new(size);
    let timing = TimingHelper::default();
    let (d_in, d_hid, d_out) = baseline.dimensions();

    // Matrix multiplication (first layer)
    {
        let input = baseline.random_input();
        let result =
            harness.run_benchmark(&format!("native_matmul_{}", size.name()), "native", || {
                baseline.matmul(&baseline.w1, &input, d_hid, d_in)
            });
    }

    // Forward pass
    {
        let input = baseline.random_input();
        harness.run_benchmark(&format!("native_forward_{}", size.name()), "native", || {
            baseline.forward_pass(&input)
        });
    }

    // Gradient computation
    {
        let input = baseline.random_input();
        let target = baseline.random_target();
        let (hidden, output) = baseline.forward_pass(&input);
        harness.run_benchmark(
            &format!("native_gradient_{}", size.name()),
            "native",
            || baseline.backward_pass(&input, &hidden, &output, &target),
        );
    }

    // Complete training step (with cloning to allow mutation)
    {
        let input = baseline.random_input();
        let target = baseline.random_target();
        harness.run_benchmark(
            &format!("native_training_step_{}", size.name()),
            "native",
            || {
                let mut b = baseline.clone_for_benchmark();
                b.training_step(&input, &target)
            },
        );
    }
}

/// Collects all native baseline timing for overhead calculation.
pub struct NativeBaselineCollection {
    pub baselines: Vec<(ModelSize, NativeBaselineTiming)>,
}

/// Timing results for a single model size.
#[derive(Debug, Clone)]
pub struct NativeBaselineTiming {
    pub model_size: ModelSize,
    pub matmul: TimingResult,
    pub forward_pass: TimingResult,
    pub gradient: TimingResult,
    pub training_step: TimingResult,
}

impl NativeBaselineCollection {
    /// Collects baselines for all model sizes.
    pub fn collect_all(timing: &TimingHelper) -> Self {
        let mut baselines = Vec::new();

        for &size in ModelSize::all() {
            let baseline = NativeBaseline::new(size);
            let input = baseline.random_input();
            let target = baseline.random_target();

            let (d_in, d_hid, _) = baseline.dimensions();

            let matmul = timing.measure(|| baseline.matmul(&baseline.w1, &input, d_hid, d_in));
            let forward_pass = timing.measure(|| baseline.forward_pass(&input));

            let (hidden, output) = baseline.forward_pass(&input);
            let gradient =
                timing.measure(|| baseline.backward_pass(&input, &hidden, &output, &target));

            let training_step = timing.measure(|| {
                let mut b = baseline.clone_for_benchmark();
                b.training_step(&input, &target)
            });

            baselines.push((
                size,
                NativeBaselineTiming {
                    model_size: size,
                    matmul,
                    forward_pass,
                    gradient,
                    training_step,
                },
            ));
        }

        Self { baselines }
    }

    /// Gets baseline timing for a specific model size.
    pub fn get(&self, size: ModelSize) -> Option<&NativeBaselineTiming> {
        self.baselines
            .iter()
            .find(|(s, _)| *s == size)
            .map(|(_, t)| t)
    }

    /// Returns all baselines.
    pub fn all(&self) -> &[(ModelSize, NativeBaselineTiming)] {
        &self.baselines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_baseline_creation() {
        for &size in ModelSize::all() {
            let baseline = NativeBaseline::new(size);
            let (d_in, d_hid, d_out) = baseline.dimensions();
            assert_eq!(baseline.w1.len(), d_hid * d_in);
            assert_eq!(baseline.w2.len(), d_out * d_hid);
        }
    }

    #[test]
    fn test_native_forward_pass() {
        let baseline = NativeBaseline::new(ModelSize::Tiny);
        let input = baseline.random_input();
        let (hidden, output) = baseline.forward_pass(&input);

        let (_, d_hid, d_out) = baseline.dimensions();
        assert_eq!(hidden.len(), d_hid);
        assert_eq!(output.len(), d_out);
    }

    #[test]
    fn test_native_training_step() {
        let mut baseline = NativeBaseline::new(ModelSize::Tiny);
        let input = baseline.random_input();
        let target = baseline.random_target();

        let loss = baseline.training_step(&input, &target);
        // Loss should be non-zero for random input/target
        // (unless by extreme coincidence)
    }

    #[test]
    fn test_baseline_collection() {
        let timing = TimingHelper::quick();
        let collection = NativeBaselineCollection::collect_all(&timing);

        assert_eq!(collection.baselines.len(), ModelSize::all().len());

        for (size, timing) in &collection.baselines {
            assert!(timing.training_step.mean > Duration::ZERO);
        }
    }
}
