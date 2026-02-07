//! Standardized model architectures for benchmarking.
//!
//! This module provides reference implementations of common neural network
//! architectures used for consistent benchmarking across different systems.

use crate::types::{BoundedTensor, Precision};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Standard model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name.
    pub name: String,
    /// Model architecture type.
    pub architecture: Architecture,
    /// Model size category.
    pub size: ModelSize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of layers.
    pub num_layers: usize,
    /// Number of attention heads (for transformers).
    pub num_heads: usize,
    /// Sequence length.
    pub seq_length: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Intermediate dimension (for FFN).
    pub intermediate_dim: usize,
    /// Dropout rate.
    pub dropout: f64,
    /// Precision for computation.
    pub precision: Precision,
}

/// Model architecture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    /// Multi-layer perceptron.
    MLP,
    /// Convolutional neural network.
    CNN,
    /// Transformer encoder.
    TransformerEncoder,
    /// Transformer decoder.
    TransformerDecoder,
    /// GPT-style decoder-only transformer.
    GPT,
    /// BERT-style encoder-only transformer.
    BERT,
    /// Vision Transformer.
    ViT,
}

/// Model size category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSize {
    /// Tiny: ~500K parameters.
    Tiny,
    /// Small: ~2M parameters.
    Small,
    /// Medium: ~10M parameters.
    Medium,
    /// Large: ~50M parameters.
    Large,
    /// XLarge: ~100M+ parameters.
    XLarge,
    /// Custom size.
    Custom(usize),
}

impl ModelSize {
    /// Returns approximate parameter count.
    pub fn param_count(&self) -> usize {
        match self {
            ModelSize::Tiny => 500_000,
            ModelSize::Small => 2_000_000,
            ModelSize::Medium => 10_000_000,
            ModelSize::Large => 50_000_000,
            ModelSize::XLarge => 100_000_000,
            ModelSize::Custom(n) => *n,
        }
    }
}

impl ModelConfig {
    /// Creates a tiny transformer configuration.
    pub fn tiny_transformer() -> Self {
        Self {
            name: "tiny_transformer".to_string(),
            architecture: Architecture::GPT,
            size: ModelSize::Tiny,
            hidden_dim: 128,
            num_layers: 4,
            num_heads: 4,
            seq_length: 64,
            vocab_size: 1000,
            intermediate_dim: 512,
            dropout: 0.0,
            precision: Precision::F32,
        }
    }

    /// Creates a small transformer configuration (demo size).
    pub fn small_transformer() -> Self {
        Self {
            name: "small_transformer".to_string(),
            architecture: Architecture::GPT,
            size: ModelSize::Small,
            hidden_dim: 256,
            num_layers: 6,
            num_heads: 8,
            seq_length: 128,
            vocab_size: 5000,
            intermediate_dim: 1024,
            dropout: 0.0,
            precision: Precision::F32,
        }
    }

    /// Creates a medium transformer configuration.
    pub fn medium_transformer() -> Self {
        Self {
            name: "medium_transformer".to_string(),
            architecture: Architecture::GPT,
            size: ModelSize::Medium,
            hidden_dim: 512,
            num_layers: 8,
            num_heads: 8,
            seq_length: 256,
            vocab_size: 10000,
            intermediate_dim: 2048,
            dropout: 0.0,
            precision: Precision::F32,
        }
    }

    /// Creates a tiny MLP configuration.
    pub fn tiny_mlp() -> Self {
        Self {
            name: "tiny_mlp".to_string(),
            architecture: Architecture::MLP,
            size: ModelSize::Tiny,
            hidden_dim: 256,
            num_layers: 3,
            num_heads: 0,
            seq_length: 1,
            vocab_size: 0,
            intermediate_dim: 512,
            dropout: 0.0,
            precision: Precision::F32,
        }
    }

    /// Estimates parameter count for this configuration.
    pub fn estimated_params(&self) -> usize {
        match self.architecture {
            Architecture::MLP => {
                // Input layer + hidden layers + output layer
                let input_layer = self.hidden_dim * self.hidden_dim;
                let hidden_layers = (self.num_layers - 2) * self.hidden_dim * self.hidden_dim;
                let output_layer = self.hidden_dim * self.hidden_dim;
                input_layer + hidden_layers + output_layer
            }
            Architecture::GPT | Architecture::BERT | Architecture::TransformerEncoder | Architecture::TransformerDecoder => {
                // Embedding: vocab_size * hidden_dim
                let embedding = self.vocab_size * self.hidden_dim;

                // Per layer:
                // - Attention: 4 * hidden_dim * hidden_dim (Q, K, V, O projections)
                // - FFN: 2 * hidden_dim * intermediate_dim
                // - LayerNorm: 2 * hidden_dim
                let attention_per_layer = 4 * self.hidden_dim * self.hidden_dim;
                let ffn_per_layer = 2 * self.hidden_dim * self.intermediate_dim;
                let layernorm_per_layer = 4 * self.hidden_dim; // 2 layernorms
                let per_layer = attention_per_layer + ffn_per_layer + layernorm_per_layer;

                // Output: hidden_dim * vocab_size (often shared with embedding)
                let output = self.hidden_dim * self.vocab_size;

                embedding + self.num_layers * per_layer + output
            }
            Architecture::CNN | Architecture::ViT => {
                // Simplified estimate
                self.num_layers * self.hidden_dim * self.hidden_dim * 9
            }
        }
    }

    /// Estimates FLOPs per forward pass.
    pub fn estimated_flops(&self) -> usize {
        let batch_size = 1;
        let seq_len = self.seq_length;
        let d = self.hidden_dim;
        let ff = self.intermediate_dim;

        match self.architecture {
            Architecture::GPT | Architecture::BERT | Architecture::TransformerEncoder | Architecture::TransformerDecoder => {
                // Per layer:
                // - QKV projection: 3 * 2 * seq_len * d * d
                // - Attention scores: 2 * seq_len * seq_len * d
                // - Attention output: 2 * seq_len * d * d
                // - FFN: 2 * 2 * seq_len * d * ff
                let attention = 6 * seq_len * d * d + 2 * seq_len * seq_len * d;
                let ffn = 4 * seq_len * d * ff;
                let per_layer = attention + ffn;

                self.num_layers * per_layer * batch_size
            }
            Architecture::MLP => {
                // Simple matrix multiplies
                self.num_layers * 2 * d * d * batch_size
            }
            Architecture::CNN | Architecture::ViT => {
                // Simplified
                self.num_layers * seq_len * d * d * 9 * batch_size
            }
        }
    }
}

/// Benchmark result for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBenchmarkResult {
    /// Model configuration.
    pub config: ModelConfig,
    /// Forward pass time.
    pub forward_time: Duration,
    /// Backward pass time.
    pub backward_time: Duration,
    /// Total step time.
    pub step_time: Duration,
    /// Proof generation time (if applicable).
    pub proof_time: Option<Duration>,
    /// Memory usage (bytes).
    pub memory_bytes: usize,
    /// Throughput (samples/sec).
    pub throughput: f64,
    /// Error statistics.
    pub error_stats: ErrorStats,
    /// Overhead ratio (proof_time / compute_time).
    pub overhead_ratio: Option<f64>,
    /// FLOPs achieved.
    pub flops: usize,
    /// TFLOPs/s.
    pub tflops_per_sec: f64,
}

/// Error statistics for a benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStats {
    /// Mean error.
    pub mean_error: f64,
    /// Max error.
    pub max_error: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// 95th percentile error.
    pub p95_error: f64,
    /// 99th percentile error.
    pub p99_error: f64,
}

impl Default for ErrorStats {
    fn default() -> Self {
        Self {
            mean_error: 0.0,
            max_error: 0.0,
            std_dev: 0.0,
            p95_error: 0.0,
            p99_error: 0.0,
        }
    }
}

/// Standard benchmark suite.
#[derive(Debug, Clone)]
pub struct StandardBenchmarkSuite {
    /// Models to benchmark.
    models: Vec<ModelConfig>,
    /// Number of warmup iterations.
    warmup_iters: usize,
    /// Number of benchmark iterations.
    bench_iters: usize,
    /// Results.
    results: Vec<ModelBenchmarkResult>,
}

impl StandardBenchmarkSuite {
    /// Creates a new benchmark suite.
    pub fn new() -> Self {
        Self {
            models: vec![
                ModelConfig::tiny_transformer(),
                ModelConfig::small_transformer(),
                ModelConfig::tiny_mlp(),
            ],
            warmup_iters: 5,
            bench_iters: 20,
            results: Vec::new(),
        }
    }

    /// Creates a minimal suite for quick testing.
    pub fn minimal() -> Self {
        Self {
            models: vec![ModelConfig::tiny_transformer()],
            warmup_iters: 2,
            bench_iters: 5,
            results: Vec::new(),
        }
    }

    /// Creates a comprehensive suite.
    pub fn comprehensive() -> Self {
        Self {
            models: vec![
                ModelConfig::tiny_transformer(),
                ModelConfig::small_transformer(),
                ModelConfig::medium_transformer(),
                ModelConfig::tiny_mlp(),
            ],
            warmup_iters: 10,
            bench_iters: 50,
            results: Vec::new(),
        }
    }

    /// Adds a custom model configuration.
    pub fn add_model(&mut self, config: ModelConfig) {
        self.models.push(config);
    }

    /// Runs the benchmark for a single model.
    pub fn benchmark_model(&mut self, config: &ModelConfig) -> ModelBenchmarkResult {
        // Create simulated computation
        let _d = config.hidden_dim;
        let _seq_len = config.seq_length;

        // Warmup
        for _ in 0..self.warmup_iters {
            let _ = self.simulate_forward(config);
        }

        // Benchmark forward
        let mut forward_times = Vec::with_capacity(self.bench_iters);
        let mut backward_times = Vec::with_capacity(self.bench_iters);
        let mut errors = Vec::with_capacity(self.bench_iters);

        for _ in 0..self.bench_iters {
            let start = Instant::now();
            let (_, error) = self.simulate_forward(config);
            forward_times.push(start.elapsed());
            errors.push(error);

            let start = Instant::now();
            self.simulate_backward(config);
            backward_times.push(start.elapsed());
        }

        // Compute statistics
        let avg_forward = forward_times.iter().sum::<Duration>() / self.bench_iters as u32;
        let avg_backward = backward_times.iter().sum::<Duration>() / self.bench_iters as u32;
        let step_time = avg_forward + avg_backward;

        let error_stats = Self::compute_error_stats(&errors);

        let flops = config.estimated_flops();
        let tflops_per_sec = if step_time.as_secs_f64() > 0.0 {
            flops as f64 / step_time.as_secs_f64() / 1e12
        } else {
            0.0
        };

        let memory_bytes = self.estimate_memory(config);
        let throughput = if step_time.as_secs_f64() > 0.0 {
            1.0 / step_time.as_secs_f64()
        } else {
            0.0
        };

        ModelBenchmarkResult {
            config: config.clone(),
            forward_time: avg_forward,
            backward_time: avg_backward,
            step_time,
            proof_time: None,
            memory_bytes,
            throughput,
            error_stats,
            overhead_ratio: None,
            flops,
            tflops_per_sec,
        }
    }

    /// Simulates a forward pass.
    fn simulate_forward(&self, config: &ModelConfig) -> (BoundedTensor, f64) {
        let d = config.hidden_dim;
        let seq_len = config.seq_length;

        // Simulate computation with error tracking
        let precision_error = config.precision.max_relative_error();
        let mut total_error = 0.0;

        match config.architecture {
            Architecture::GPT | Architecture::BERT | Architecture::TransformerEncoder | Architecture::TransformerDecoder => {
                // Simulate transformer layers
                for _ in 0..config.num_layers {
                    // Attention: O(seq_len^2 * d) operations
                    let attention_error = precision_error * (seq_len as f64).sqrt() * (d as f64).sqrt();

                    // FFN: O(seq_len * d * ff) operations
                    let ffn_error = precision_error * (d as f64).sqrt() * (config.intermediate_dim as f64).sqrt();

                    // Layer norm reduces error somewhat
                    total_error += (attention_error + ffn_error) * 0.5;
                }
            }
            Architecture::MLP => {
                for _ in 0..config.num_layers {
                    total_error += precision_error * (d as f64).sqrt();
                }
            }
            _ => {
                total_error = precision_error * config.num_layers as f64;
            }
        }

        let output = BoundedTensor::from_approximate(
            vec![0.0; seq_len * d],
            vec![seq_len, d],
            total_error,
        );

        (output, total_error)
    }

    /// Simulates a backward pass.
    fn simulate_backward(&self, config: &ModelConfig) {
        // Similar to forward but typically ~2x operations for gradients
        let _ = self.simulate_forward(config);
    }

    /// Estimates memory usage.
    fn estimate_memory(&self, config: &ModelConfig) -> usize {
        let params = config.estimated_params();
        let bytes_per_param = config.precision.bytes_per_element();

        // Parameters + gradients + optimizer states (~4x for Adam)
        params * bytes_per_param * 4
    }

    /// Computes error statistics.
    fn compute_error_stats(errors: &[f64]) -> ErrorStats {
        if errors.is_empty() {
            return ErrorStats::default();
        }

        let n = errors.len();
        let mean_error = errors.iter().sum::<f64>() / n as f64;
        let max_error = errors.iter().cloned().fold(0.0, f64::max);
        let variance = errors.iter().map(|e| (e - mean_error).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        let mut sorted = errors.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p95_idx = (0.95 * n as f64) as usize;
        let p99_idx = (0.99 * n as f64) as usize;

        ErrorStats {
            mean_error,
            max_error,
            std_dev,
            p95_error: sorted[p95_idx.min(n - 1)],
            p99_error: sorted[p99_idx.min(n - 1)],
        }
    }

    /// Runs the full benchmark suite.
    pub fn run(&mut self) -> Vec<ModelBenchmarkResult> {
        self.results.clear();

        for config in self.models.clone() {
            let result = self.benchmark_model(&config);
            self.results.push(result);
        }

        self.results.clone()
    }

    /// Runs benchmarks with proof generation.
    pub fn run_with_proofs<F>(&mut self, proof_generator: F) -> Vec<ModelBenchmarkResult>
    where
        F: Fn(&ModelConfig) -> Duration,
    {
        let mut results = self.run();

        for (i, config) in self.models.iter().enumerate() {
            let proof_time = proof_generator(config);
            results[i].proof_time = Some(proof_time);

            if results[i].step_time.as_nanos() > 0 {
                results[i].overhead_ratio = Some(
                    proof_time.as_secs_f64() / results[i].step_time.as_secs_f64()
                );
            }
        }

        self.results = results.clone();
        results
    }

    /// Generates a benchmark report.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Standard Model Benchmark Report\n\n");
        report.push_str(&format!(
            "Warmup: {} iters | Benchmark: {} iters\n\n",
            self.warmup_iters, self.bench_iters
        ));

        report.push_str("## Results Summary\n\n");
        report.push_str("| Model | Params | Forward | Backward | Step | Throughput | TFLOPs/s | Max Error |\n");
        report.push_str("|-------|--------|---------|----------|------|------------|----------|----------|\n");

        for result in &self.results {
            report.push_str(&format!(
                "| {} | {:.2}M | {:?} | {:?} | {:?} | {:.2}/s | {:.3} | {:.2e} |\n",
                result.config.name,
                result.config.estimated_params() as f64 / 1e6,
                result.forward_time,
                result.backward_time,
                result.step_time,
                result.throughput,
                result.tflops_per_sec,
                result.error_stats.max_error,
            ));
        }

        if self.results.iter().any(|r| r.proof_time.is_some()) {
            report.push_str("\n## Proof Generation Overhead\n\n");
            report.push_str("| Model | Proof Time | Overhead Ratio | Target (30x) |\n");
            report.push_str("|-------|------------|----------------|---------------|\n");

            for result in &self.results {
                if let (Some(proof_time), Some(overhead)) = (result.proof_time, result.overhead_ratio) {
                    let status = if overhead <= 30.0 { "PASS" } else { "FAIL" };
                    report.push_str(&format!(
                        "| {} | {:?} | {:.1}x | {} |\n",
                        result.config.name, proof_time, overhead, status
                    ));
                }
            }
        }

        report.push_str("\n## Error Statistics\n\n");
        report.push_str("| Model | Mean | Std Dev | P95 | P99 | Max |\n");
        report.push_str("|-------|------|---------|-----|-----|-----|\n");

        for result in &self.results {
            let stats = &result.error_stats;
            report.push_str(&format!(
                "| {} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} |\n",
                result.config.name,
                stats.mean_error,
                stats.std_dev,
                stats.p95_error,
                stats.p99_error,
                stats.max_error,
            ));
        }

        report
    }

    /// Exports results to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.results).unwrap_or_else(|_| "[]".to_string())
    }

    /// Returns the results.
    pub fn results(&self) -> &[ModelBenchmarkResult] {
        &self.results
    }
}

impl Default for StandardBenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Comparison result between two benchmark runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    /// Model name.
    pub model_name: String,
    /// Baseline step time.
    pub baseline_step_time: Duration,
    /// Current step time.
    pub current_step_time: Duration,
    /// Speedup (baseline / current).
    pub speedup: f64,
    /// Baseline error.
    pub baseline_error: f64,
    /// Current error.
    pub current_error: f64,
    /// Error improvement (baseline - current).
    pub error_improvement: f64,
}

/// Compares two benchmark result sets.
pub fn compare_results(
    baseline: &[ModelBenchmarkResult],
    current: &[ModelBenchmarkResult],
) -> Vec<BenchmarkComparison> {
    let mut comparisons = Vec::new();

    for base_result in baseline {
        if let Some(curr_result) = current.iter().find(|r| r.config.name == base_result.config.name) {
            let speedup = base_result.step_time.as_secs_f64() / curr_result.step_time.as_secs_f64();
            let error_improvement = base_result.error_stats.max_error - curr_result.error_stats.max_error;

            comparisons.push(BenchmarkComparison {
                model_name: base_result.config.name.clone(),
                baseline_step_time: base_result.step_time,
                current_step_time: curr_result.step_time,
                speedup,
                baseline_error: base_result.error_stats.max_error,
                current_error: curr_result.error_stats.max_error,
                error_improvement,
            });
        }
    }

    comparisons
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config() {
        let config = ModelConfig::tiny_transformer();
        assert_eq!(config.hidden_dim, 128);
        assert_eq!(config.num_layers, 4);
        assert!(config.estimated_params() > 0);
        assert!(config.estimated_flops() > 0);
    }

    #[test]
    fn test_param_estimation() {
        let tiny = ModelConfig::tiny_transformer();
        let small = ModelConfig::small_transformer();

        // Small should have more params than tiny
        assert!(small.estimated_params() > tiny.estimated_params());
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = StandardBenchmarkSuite::minimal();
        let results = suite.run();

        assert_eq!(results.len(), 1);
        assert!(results[0].step_time > Duration::ZERO);
        assert!(results[0].throughput > 0.0);
    }

    #[test]
    fn test_benchmark_report() {
        let mut suite = StandardBenchmarkSuite::minimal();
        suite.run();

        let report = suite.generate_report();
        assert!(report.contains("tiny_transformer"));
        assert!(report.contains("Forward"));
    }

    #[test]
    fn test_error_stats() {
        let errors = vec![0.001, 0.002, 0.0015, 0.003, 0.0025];
        let stats = StandardBenchmarkSuite::compute_error_stats(&errors);

        assert!(stats.mean_error > 0.0);
        assert!(stats.max_error >= stats.p99_error);
        assert!(stats.p99_error >= stats.p95_error);
    }

    #[test]
    fn test_with_proofs() {
        let mut suite = StandardBenchmarkSuite::minimal();

        let results = suite.run_with_proofs(|config| {
            // Simulate proof time proportional to params
            Duration::from_micros((config.estimated_params() / 1000) as u64)
        });

        assert!(results[0].proof_time.is_some());
        assert!(results[0].overhead_ratio.is_some());
    }

    #[test]
    fn test_comparison() {
        let mut suite1 = StandardBenchmarkSuite::minimal();
        let mut suite2 = StandardBenchmarkSuite::minimal();

        let baseline = suite1.run();
        let current = suite2.run();

        let comparisons = compare_results(&baseline, &current);
        assert_eq!(comparisons.len(), 1);
    }
}
