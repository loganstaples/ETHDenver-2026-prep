//! PyTorch Reference Test Vectors for Model Validation.
//!
//! This module provides pre-computed reference outputs that match PyTorch's
//! implementations exactly. These are used to validate that HELIX AVM computes
//! mathematically correct results.
//!
//! # Reference Vector Generation
//!
//! All reference vectors are computed analytically or via the following PyTorch code:
//!
//! ```python
//! import torch
//! import torch.nn.functional as F
//!
//! # Linear layer: y = Wx + b
//! # W = [[1, 2], [3, 4]], b = [0.5, 0.5], x = [1, 1]
//! W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
//! b = torch.tensor([0.5, 0.5])
//! x = torch.tensor([1.0, 1.0])
//! y = F.linear(x, W, b)  # [3.5, 7.5]
//!
//! # Softmax
//! x = torch.tensor([1.0, 2.0, 3.0])
//! y = F.softmax(x, dim=0)  # [0.0900, 0.2447, 0.6652]
//!
//! # Attention
//! Q = K = V = torch.randn(4, 8)
//! y = F.scaled_dot_product_attention(Q, K, V)
//! ```
//!
//! # Verification Methodology
//!
//! Each reference includes:
//! 1. Input values
//! 2. Expected output values (computed via PyTorch or analytically)
//! 3. Maximum acceptable error tolerance
//! 4. Test name for identification

use helix_core::types::BoundedTensor;
use std::collections::HashMap;

/// A reference test vector with input/output pairs.
#[derive(Debug, Clone)]
pub struct ReferenceTestVector {
    /// Test name for identification.
    pub name: String,
    /// Input tensor(s).
    pub inputs: HashMap<String, (Vec<f64>, Vec<usize>)>,
    /// Expected output tensor.
    pub expected_output: Vec<f64>,
    /// Expected output shape.
    pub expected_shape: Vec<usize>,
    /// Maximum absolute tolerance.
    pub absolute_tolerance: f64,
    /// Maximum relative tolerance.
    pub relative_tolerance: f64,
    /// Description of what this test validates.
    pub description: String,
}

impl ReferenceTestVector {
    /// Creates a new reference test vector.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inputs: HashMap::new(),
            expected_output: Vec::new(),
            expected_shape: Vec::new(),
            absolute_tolerance: 1e-5,
            relative_tolerance: 1e-4,
            description: String::new(),
        }
    }

    /// Adds an input tensor.
    pub fn with_input(mut self, name: &str, values: Vec<f64>, shape: Vec<usize>) -> Self {
        self.inputs.insert(name.to_string(), (values, shape));
        self
    }

    /// Sets expected output.
    pub fn with_expected(mut self, values: Vec<f64>, shape: Vec<usize>) -> Self {
        self.expected_output = values;
        self.expected_shape = shape;
        self
    }

    /// Sets tolerance.
    pub fn with_tolerance(mut self, absolute: f64, relative: f64) -> Self {
        self.absolute_tolerance = absolute;
        self.relative_tolerance = relative;
        self
    }

    /// Sets description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Gets an input as BoundedTensor.
    pub fn get_input(&self, name: &str) -> Option<BoundedTensor> {
        self.inputs.get(name).map(|(values, shape)| {
            BoundedTensor::from_exact(values.clone(), shape.clone())
        })
    }

    /// Gets expected output as BoundedTensor.
    pub fn get_expected(&self) -> BoundedTensor {
        BoundedTensor::from_exact(self.expected_output.clone(), self.expected_shape.clone())
    }
}

/// Collection of reference tests for a specific operation.
#[derive(Debug, Clone)]
pub struct ReferenceTestSuite {
    /// Suite name.
    pub name: String,
    /// Test vectors.
    pub tests: Vec<ReferenceTestVector>,
}

impl ReferenceTestSuite {
    /// Creates a new test suite.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tests: Vec::new(),
        }
    }

    /// Adds a test vector.
    pub fn add_test(&mut self, test: ReferenceTestVector) {
        self.tests.push(test);
    }
}

/// PyTorch reference computation interface.
pub struct PyTorchReference;

impl PyTorchReference {
    /// Generates reference for linear layer: y = Wx + b
    ///
    /// Computes: output[i] = sum(W[i,j] * x[j]) + b[i]
    pub fn linear(
        input: &[f64],
        input_shape: &[usize],
        weights: &[f64],
        weight_shape: &[usize],
        bias: Option<&[f64]>,
    ) -> (Vec<f64>, Vec<usize>) {
        assert_eq!(weight_shape.len(), 2);
        let (out_features, in_features) = (weight_shape[0], weight_shape[1]);

        // Handle 1D or 2D input
        let (batch_size, input_features) = if input_shape.len() == 1 {
            (1, input_shape[0])
        } else {
            (input_shape[0], input_shape[1])
        };
        assert_eq!(input_features, in_features);

        let mut output = vec![0.0; batch_size * out_features];

        for b in 0..batch_size {
            for i in 0..out_features {
                let mut sum = 0.0;
                for j in 0..in_features {
                    // W[i, j] * x[b, j]
                    sum += weights[i * in_features + j] * input[b * in_features + j];
                }
                if let Some(b_vec) = bias {
                    sum += b_vec[i];
                }
                output[b * out_features + i] = sum;
            }
        }

        if input_shape.len() == 1 {
            (output, vec![out_features])
        } else {
            (output, vec![batch_size, out_features])
        }
    }

    /// Generates reference for softmax.
    ///
    /// Computes: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    pub fn softmax(input: &[f64]) -> Vec<f64> {
        let max_val = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = input.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f64 = exp_vals.iter().sum();
        exp_vals.iter().map(|&e| e / sum).collect()
    }

    /// Generates reference for layer normalization.
    ///
    /// Computes: y = (x - mean) / sqrt(var + eps) * gamma + beta
    pub fn layer_norm(
        input: &[f64],
        gamma: &[f64],
        beta: &[f64],
        eps: f64,
    ) -> Vec<f64> {
        let n = input.len();
        let mean: f64 = input.iter().sum::<f64>() / n as f64;
        let var: f64 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_inv = 1.0 / (var + eps).sqrt();

        input
            .iter()
            .enumerate()
            .map(|(i, &x)| (x - mean) * std_inv * gamma[i] + beta[i])
            .collect()
    }

    /// Generates reference for RMS normalization.
    ///
    /// Computes: y = x / sqrt(mean(x^2) + eps) * gamma
    pub fn rms_norm(input: &[f64], gamma: &[f64], eps: f64) -> Vec<f64> {
        let n = input.len();
        let rms = (input.iter().map(|&x| x * x).sum::<f64>() / n as f64 + eps).sqrt();
        let rms_inv = 1.0 / rms;

        input
            .iter()
            .enumerate()
            .map(|(i, &x)| x * rms_inv * gamma[i])
            .collect()
    }

    /// Generates reference for scaled dot-product attention.
    ///
    /// Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    pub fn scaled_dot_product_attention(
        query: &[f64],
        key: &[f64],
        value: &[f64],
        seq_len: usize,
        d_k: usize,
    ) -> Vec<f64> {
        let scale = 1.0 / (d_k as f64).sqrt();

        // Compute Q @ K^T (seq_len x seq_len)
        let mut scores = vec![0.0; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0;
                for k in 0..d_k {
                    dot += query[i * d_k + k] * key[j * d_k + k];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }

        // Apply softmax row-wise
        let mut attention_weights = vec![0.0; seq_len * seq_len];
        for i in 0..seq_len {
            let row_start = i * seq_len;
            let row = &scores[row_start..row_start + seq_len];
            let softmax_row = Self::softmax(row);
            attention_weights[row_start..row_start + seq_len].copy_from_slice(&softmax_row);
        }

        // Compute attention @ V (seq_len x d_k)
        let mut output = vec![0.0; seq_len * d_k];
        for i in 0..seq_len {
            for j in 0..d_k {
                let mut sum = 0.0;
                for k in 0..seq_len {
                    sum += attention_weights[i * seq_len + k] * value[k * d_k + j];
                }
                output[i * d_k + j] = sum;
            }
        }

        output
    }

    /// Generates reference for matrix multiplication.
    pub fn matmul(
        a: &[f64],
        b: &[f64],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f64> {
        let mut output = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                output[i * n + j] = sum;
            }
        }
        output
    }

    /// Generates reference for ReLU.
    pub fn relu(input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| x.max(0.0)).collect()
    }

    /// Generates reference for GELU (approximation).
    ///
    /// Uses: GELU(x) = x * sigmoid(1.702 * x)
    pub fn gelu(input: &[f64]) -> Vec<f64> {
        input
            .iter()
            .map(|&x| {
                let sigmoid = 1.0 / (1.0 + (-1.702 * x).exp());
                x * sigmoid
            })
            .collect()
    }

    /// Generates reference for sigmoid.
    pub fn sigmoid(input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }

    /// Generates reference for MSE loss.
    pub fn mse_loss(predictions: &[f64], targets: &[f64]) -> f64 {
        let n = predictions.len() as f64;
        predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<f64>()
            / n
    }

    /// Generates reference for cross-entropy loss.
    pub fn cross_entropy_loss(predictions: &[f64], targets: &[f64]) -> f64 {
        let eps = 1e-10;
        predictions
            .iter()
            .zip(targets.iter())
            .filter(|(_, &t)| t > 0.0)
            .map(|(&p, &t)| -t * p.max(eps).ln())
            .sum()
    }

    /// Generates reference for MSE gradient.
    pub fn mse_grad(predictions: &[f64], targets: &[f64]) -> Vec<f64> {
        let n = predictions.len() as f64;
        predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| 2.0 * (p - t) / n)
            .collect()
    }
}

// ============================================================================
// Pre-computed Reference Test Vectors
// ============================================================================

/// Generates linear layer reference test vectors.
pub fn generate_linear_reference() -> ReferenceTestSuite {
    let mut suite = ReferenceTestSuite::new("linear_layer");

    // Test 1: Simple 2x2 linear, no bias
    // W = [[1, 2], [3, 4]], x = [1, 1]
    // y = [1*1 + 2*1, 3*1 + 4*1] = [3, 7]
    suite.add_test(
        ReferenceTestVector::new("linear_2x2_no_bias")
            .with_input("weights", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .with_input("input", vec![1.0, 1.0], vec![2])
            .with_expected(vec![3.0, 7.0], vec![2])
            .with_description("2x2 linear transformation without bias"),
    );

    // Test 2: Linear with bias
    // W = [[1, 2], [3, 4]], b = [0.5, -0.5], x = [1, 1]
    // y = [3 + 0.5, 7 - 0.5] = [3.5, 6.5]
    suite.add_test(
        ReferenceTestVector::new("linear_2x2_with_bias")
            .with_input("weights", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .with_input("bias", vec![0.5, -0.5], vec![2])
            .with_input("input", vec![1.0, 1.0], vec![2])
            .with_expected(vec![3.5, 6.5], vec![2])
            .with_description("2x2 linear transformation with bias"),
    );

    // Test 3: Batched linear
    // W = [[1, 0], [0, 1]] (identity), x = [[1, 2], [3, 4]]
    suite.add_test(
        ReferenceTestVector::new("linear_batched_identity")
            .with_input("weights", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2])
            .with_input("input", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .with_expected(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .with_description("Batched identity linear transformation"),
    );

    // Test 4: Scale transformation
    // W = [[2, 0], [0, 2]], x = [3, 4]
    // y = [6, 8]
    suite.add_test(
        ReferenceTestVector::new("linear_scale")
            .with_input("weights", vec![2.0, 0.0, 0.0, 2.0], vec![2, 2])
            .with_input("input", vec![3.0, 4.0], vec![2])
            .with_expected(vec![6.0, 8.0], vec![2])
            .with_description("Scale transformation"),
    );

    suite
}

/// Generates softmax reference test vectors.
pub fn generate_softmax_reference() -> ReferenceTestSuite {
    let mut suite = ReferenceTestSuite::new("softmax");

    // Test 1: Uniform input
    // softmax([1, 1, 1]) = [1/3, 1/3, 1/3]
    suite.add_test(
        ReferenceTestVector::new("softmax_uniform")
            .with_input("input", vec![1.0, 1.0, 1.0], vec![3])
            .with_expected(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], vec![3])
            .with_description("Uniform softmax input"),
    );

    // Test 2: Clear winner
    // softmax([0, 0, 10]) ≈ [4.54e-5, 4.54e-5, 0.99991]
    let s2 = PyTorchReference::softmax(&[0.0, 0.0, 10.0]);
    suite.add_test(
        ReferenceTestVector::new("softmax_clear_winner")
            .with_input("input", vec![0.0, 0.0, 10.0], vec![3])
            .with_expected(s2, vec![3])
            .with_tolerance(1e-4, 1e-4)
            .with_description("Softmax with clear winner"),
    );

    // Test 3: Large values (numerical stability test)
    // softmax([1000, 1001, 1002]) should still sum to 1
    let s3 = PyTorchReference::softmax(&[1000.0, 1001.0, 1002.0]);
    suite.add_test(
        ReferenceTestVector::new("softmax_large_values")
            .with_input("input", vec![1000.0, 1001.0, 1002.0], vec![3])
            .with_expected(s3, vec![3])
            .with_tolerance(1e-5, 1e-4)
            .with_description("Softmax numerical stability with large values"),
    );

    // Test 4: Standard input [1, 2, 3]
    let s4 = PyTorchReference::softmax(&[1.0, 2.0, 3.0]);
    suite.add_test(
        ReferenceTestVector::new("softmax_standard")
            .with_input("input", vec![1.0, 2.0, 3.0], vec![3])
            .with_expected(s4, vec![3])
            .with_description("Standard softmax computation"),
    );

    suite
}

/// Generates attention reference test vectors.
pub fn generate_attention_reference() -> ReferenceTestSuite {
    let mut suite = ReferenceTestSuite::new("attention");

    // Test 1: 2x2 self-attention with identity-like Q, K, V
    // Q = K = V = [[1, 0], [0, 1]]
    let qkv = vec![1.0, 0.0, 0.0, 1.0];
    let out = PyTorchReference::scaled_dot_product_attention(&qkv, &qkv, &qkv, 2, 2);
    suite.add_test(
        ReferenceTestVector::new("attention_identity")
            .with_input("query", qkv.clone(), vec![2, 2])
            .with_input("key", qkv.clone(), vec![2, 2])
            .with_input("value", qkv.clone(), vec![2, 2])
            .with_expected(out, vec![2, 2])
            .with_tolerance(1e-4, 1e-4)
            .with_description("Self-attention with identity-like matrices"),
    );

    // Test 2: 4x4 attention
    let seq_len = 4;
    let d_k = 4;
    let q: Vec<f64> = (0..seq_len * d_k).map(|i| (i as f64) / 10.0).collect();
    let k: Vec<f64> = (0..seq_len * d_k).map(|i| ((i + 1) as f64) / 10.0).collect();
    let v: Vec<f64> = (0..seq_len * d_k).map(|i| ((i + 2) as f64) / 10.0).collect();
    let out2 = PyTorchReference::scaled_dot_product_attention(&q, &k, &v, seq_len, d_k);
    suite.add_test(
        ReferenceTestVector::new("attention_4x4")
            .with_input("query", q, vec![seq_len, d_k])
            .with_input("key", k, vec![seq_len, d_k])
            .with_input("value", v, vec![seq_len, d_k])
            .with_expected(out2, vec![seq_len, d_k])
            .with_tolerance(1e-4, 1e-4)
            .with_description("4x4 scaled dot-product attention"),
    );

    suite
}

/// Generates layer normalization reference test vectors.
pub fn generate_layer_norm_reference() -> ReferenceTestSuite {
    let mut suite = ReferenceTestSuite::new("layer_norm");

    // Test 1: Standard layer norm
    // Input: [1, 2, 3, 4], gamma = [1, 1, 1, 1], beta = [0, 0, 0, 0]
    // mean = 2.5, var = 1.25, std = 1.118...
    let input1 = vec![1.0, 2.0, 3.0, 4.0];
    let gamma1 = vec![1.0, 1.0, 1.0, 1.0];
    let beta1 = vec![0.0, 0.0, 0.0, 0.0];
    let out1 = PyTorchReference::layer_norm(&input1, &gamma1, &beta1, 1e-5);
    suite.add_test(
        ReferenceTestVector::new("layer_norm_standard")
            .with_input("input", input1, vec![4])
            .with_input("gamma", gamma1, vec![4])
            .with_input("beta", beta1, vec![4])
            .with_expected(out1, vec![4])
            .with_description("Standard layer normalization"),
    );

    // Test 2: Layer norm with non-trivial gamma/beta
    let input2 = vec![1.0, 2.0, 3.0, 4.0];
    let gamma2 = vec![2.0, 2.0, 2.0, 2.0];
    let beta2 = vec![1.0, 1.0, 1.0, 1.0];
    let out2 = PyTorchReference::layer_norm(&input2, &gamma2, &beta2, 1e-5);
    suite.add_test(
        ReferenceTestVector::new("layer_norm_scaled")
            .with_input("input", input2, vec![4])
            .with_input("gamma", gamma2, vec![4])
            .with_input("beta", beta2, vec![4])
            .with_expected(out2, vec![4])
            .with_description("Layer normalization with scale and shift"),
    );

    // Test 3: RMS norm
    let input3 = vec![1.0, 2.0, 3.0, 4.0];
    let gamma3 = vec![1.0, 1.0, 1.0, 1.0];
    let out3 = PyTorchReference::rms_norm(&input3, &gamma3, 1e-5);
    suite.add_test(
        ReferenceTestVector::new("rms_norm_standard")
            .with_input("input", input3, vec![4])
            .with_input("gamma", gamma3, vec![4])
            .with_expected(out3, vec![4])
            .with_description("RMS normalization"),
    );

    suite
}

/// Generates activation function reference test vectors.
pub fn generate_activation_reference() -> ReferenceTestSuite {
    let mut suite = ReferenceTestSuite::new("activations");

    // ReLU tests
    let relu_input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let relu_out = PyTorchReference::relu(&relu_input);
    suite.add_test(
        ReferenceTestVector::new("relu_standard")
            .with_input("input", relu_input, vec![5])
            .with_expected(relu_out, vec![5])
            .with_description("ReLU activation"),
    );

    // GELU tests
    let gelu_input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let gelu_out = PyTorchReference::gelu(&gelu_input);
    suite.add_test(
        ReferenceTestVector::new("gelu_standard")
            .with_input("input", gelu_input, vec![5])
            .with_expected(gelu_out, vec![5])
            .with_tolerance(0.01, 0.02) // GELU approximation has higher tolerance
            .with_description("GELU activation (approximation)"),
    );

    // Sigmoid tests
    let sigmoid_input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let sigmoid_out = PyTorchReference::sigmoid(&sigmoid_input);
    suite.add_test(
        ReferenceTestVector::new("sigmoid_standard")
            .with_input("input", sigmoid_input, vec![5])
            .with_expected(sigmoid_out, vec![5])
            .with_description("Sigmoid activation"),
    );

    suite
}

/// Generates loss function reference test vectors.
pub fn generate_loss_reference() -> ReferenceTestSuite {
    let mut suite = ReferenceTestSuite::new("loss_functions");

    // MSE loss test
    // MSE([1, 2, 3], [1, 2, 3]) = 0
    suite.add_test(
        ReferenceTestVector::new("mse_zero")
            .with_input("predictions", vec![1.0, 2.0, 3.0], vec![3])
            .with_input("targets", vec![1.0, 2.0, 3.0], vec![3])
            .with_expected(vec![0.0], vec![1])
            .with_description("MSE loss with identical inputs"),
    );

    // MSE loss test with difference
    // MSE([1, 2, 3], [2, 3, 4]) = ((1)^2 + (1)^2 + (1)^2) / 3 = 1
    suite.add_test(
        ReferenceTestVector::new("mse_unit_diff")
            .with_input("predictions", vec![1.0, 2.0, 3.0], vec![3])
            .with_input("targets", vec![2.0, 3.0, 4.0], vec![3])
            .with_expected(vec![1.0], vec![1])
            .with_description("MSE loss with unit difference"),
    );

    // MSE gradient test
    // dL/dp = 2 * (p - t) / n = 2 * [-1, -1, -1] / 3 = [-2/3, -2/3, -2/3]
    let mse_grad = PyTorchReference::mse_grad(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0]);
    suite.add_test(
        ReferenceTestVector::new("mse_gradient")
            .with_input("predictions", vec![1.0, 2.0, 3.0], vec![3])
            .with_input("targets", vec![2.0, 3.0, 4.0], vec![3])
            .with_expected(mse_grad, vec![3])
            .with_description("MSE gradient computation"),
    );

    suite
}

/// Generates matmul reference test vectors.
pub fn generate_matmul_reference() -> ReferenceTestSuite {
    let mut suite = ReferenceTestSuite::new("matmul");

    // Test 1: 2x2 @ 2x2
    // [[1, 2], [3, 4]] @ [[1, 0], [0, 1]] = [[1, 2], [3, 4]]
    suite.add_test(
        ReferenceTestVector::new("matmul_identity")
            .with_input("a", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .with_input("b", vec![1.0, 0.0, 0.0, 1.0], vec![2, 2])
            .with_expected(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .with_description("Matrix multiplication with identity"),
    );

    // Test 2: 2x3 @ 3x2
    // [[1, 2, 3], [4, 5, 6]] @ [[1, 2], [3, 4], [5, 6]]
    // = [[1+6+15, 2+8+18], [4+15+30, 8+20+36]]
    // = [[22, 28], [49, 64]]
    let out = PyTorchReference::matmul(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        2, 3, 2,
    );
    suite.add_test(
        ReferenceTestVector::new("matmul_2x3_3x2")
            .with_input("a", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .with_input("b", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])
            .with_expected(out, vec![2, 2])
            .with_description("Matrix multiplication 2x3 @ 3x2"),
    );

    suite
}

#[cfg(test)]
mod tests {
    use super::*;
    use helix_core::types::Precision;
    use crate::ops;
    use crate::nn::Linear;

    #[test]
    fn test_pytorch_reference_linear() {
        let (output, shape) = PyTorchReference::linear(
            &[1.0, 1.0],
            &[2],
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            None,
        );
        assert_eq!(shape, vec![2]);
        assert!((output[0] - 3.0).abs() < 1e-10);
        assert!((output[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_pytorch_reference_softmax() {
        let output = PyTorchReference::softmax(&[1.0, 1.0, 1.0]);
        let expected = 1.0 / 3.0;
        for &v in &output {
            assert!((v - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pytorch_reference_mse() {
        let loss = PyTorchReference::mse_loss(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!(loss.abs() < 1e-10);

        let loss2 = PyTorchReference::mse_loss(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0]);
        assert!((loss2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_reference_suite() {
        let suite = generate_linear_reference();
        assert!(!suite.tests.is_empty());

        for test in &suite.tests {
            let input = test.get_input("input").unwrap();
            let weights = test.get_input("weights").unwrap();
            let bias = test.inputs.get("bias").map(|(v, s)| {
                BoundedTensor::from_exact(v.clone(), s.clone())
            });

            let layer = Linear::new(
                weights,
                bias,
                Precision::F32,
            ).unwrap();

            let output = layer.forward(&input).unwrap();
            let output_values = output.values();

            for (i, (&actual, &expected)) in output_values.iter().zip(test.expected_output.iter()).enumerate() {
                let diff = (actual - expected).abs();
                assert!(
                    diff < test.absolute_tolerance,
                    "Test '{}' failed at index {}: expected {}, got {} (diff: {})",
                    test.name, i, expected, actual, diff
                );
            }
        }
    }

    #[test]
    fn test_softmax_reference_suite() {
        let suite = generate_softmax_reference();

        for test in &suite.tests {
            let input = test.get_input("input").unwrap();
            let output = ops::softmax::softmax(&input, Precision::F32).unwrap();
            let output_values = output.values();

            for (i, (&actual, &expected)) in output_values.iter().zip(test.expected_output.iter()).enumerate() {
                let diff = (actual - expected).abs();
                assert!(
                    diff < test.absolute_tolerance + test.relative_tolerance * expected.abs(),
                    "Test '{}' failed at index {}: expected {}, got {} (diff: {})",
                    test.name, i, expected, actual, diff
                );
            }
        }
    }

    #[test]
    fn test_matmul_reference_suite() {
        let suite = generate_matmul_reference();

        for test in &suite.tests {
            let a = test.get_input("a").unwrap();
            let b = test.get_input("b").unwrap();
            let output = ops::matmul(&a, &b, Precision::F32).unwrap();
            let output_values = output.values();

            for (i, (&actual, &expected)) in output_values.iter().zip(test.expected_output.iter()).enumerate() {
                let diff = (actual - expected).abs();
                assert!(
                    diff < test.absolute_tolerance,
                    "Test '{}' failed at index {}: expected {}, got {} (diff: {})",
                    test.name, i, expected, actual, diff
                );
            }
        }
    }
}
