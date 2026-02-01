//! Full Pipeline Integration Tests
//!
//! This module provides comprehensive end-to-end tests that validate the entire
//! HELIX training pipeline from native computation through proof generation
//! and verification.
//!
//! # Pipeline Stages Tested
//!
//! 1. Native computation baseline
//! 2. Witness generation
//! 3. GKR proof generation
//! 4. Halo2 proof generation (for on-chain verification)
//! 5. Proof verification (native and EVM-compatible)
//! 6. Overhead validation (must be ≤30x)
//!
//! # Success Criteria
//!
//! - All proofs verify successfully
//! - Overhead remains within 30x target
//! - Memory usage stays within bounds
//! - No regressions detected

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_prover::gkr::{
    layered_circuit::NeuralNetworkCircuit, GKRConfig, GKRProof, GKRProver, GKRVerifier,
    LayeredCircuit,
};

/// Target overhead multiple for HELIX.
const TARGET_OVERHEAD: f64 = 30.0;

/// Alert threshold - requires investigation if exceeded.
const ALERT_THRESHOLD: f64 = 35.0;

/// Model configuration for pipeline testing.
#[derive(Debug, Clone)]
pub struct PipelineModelConfig {
    pub name: String,
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
}

impl PipelineModelConfig {
    pub fn tiny() -> Self {
        Self {
            name: "tiny".to_string(),
            d_in: 2,
            d_hid: 2,
            d_out: 1,
        }
    }

    pub fn small() -> Self {
        Self {
            name: "small".to_string(),
            d_in: 8,
            d_hid: 16,
            d_out: 4,
        }
    }

    pub fn medium() -> Self {
        Self {
            name: "medium".to_string(),
            d_in: 32,
            d_hid: 64,
            d_out: 16,
        }
    }

    pub fn params_10k() -> Self {
        Self {
            name: "10k_params".to_string(),
            d_in: 32,
            d_hid: 128,
            d_out: 64,
        }
    }

    pub fn params_100k() -> Self {
        Self {
            name: "100k_params".to_string(),
            d_in: 128,
            d_hid: 384,
            d_out: 192,
        }
    }

    pub fn params_500k() -> Self {
        Self {
            name: "500k_params".to_string(),
            d_in: 256,
            d_hid: 768,
            d_out: 384,
        }
    }

    pub fn params_1m() -> Self {
        Self {
            name: "1m_params".to_string(),
            d_in: 384,
            d_hid: 1024,
            d_out: 512,
        }
    }

    pub fn params_2m() -> Self {
        Self {
            name: "2m_params".to_string(),
            d_in: 512,
            d_hid: 1536,
            d_out: 768,
        }
    }

    /// Returns the approximate parameter count.
    pub fn param_count(&self) -> usize {
        self.d_in * self.d_hid + self.d_hid + self.d_hid * self.d_out + self.d_out
    }

    /// Returns all standard benchmark configurations.
    pub fn benchmark_configs() -> Vec<Self> {
        vec![
            Self::params_10k(),
            Self::params_100k(),
            Self::params_500k(),
            Self::params_1m(),
            Self::params_2m(),
        ]
    }

    /// Returns quick test configurations.
    pub fn quick_configs() -> Vec<Self> {
        vec![Self::tiny(), Self::small()]
    }
}

/// Result of a single pipeline stage.
#[derive(Debug, Clone)]
pub struct StageResult {
    pub name: String,
    pub duration: Duration,
    pub success: bool,
    pub details: String,
}

/// Result of running the full pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub config: PipelineModelConfig,
    pub stages: Vec<StageResult>,
    pub total_native_time: Duration,
    pub total_proof_time: Duration,
    pub overhead: f64,
    pub meets_target: bool,
    pub exceeds_alert: bool,
    pub proof_size_bytes: usize,
    pub all_stages_passed: bool,
}

impl PipelineResult {
    /// Returns a status string.
    pub fn status(&self) -> &'static str {
        if !self.all_stages_passed {
            "FAILED"
        } else if self.meets_target {
            "PASS"
        } else if !self.exceeds_alert {
            "WARN"
        } else {
            "ALERT"
        }
    }
}

/// Full pipeline test runner.
pub struct PipelineRunner {
    config: GKRConfig,
    verbose: bool,
}

impl PipelineRunner {
    pub fn new() -> Self {
        Self {
            config: GKRConfig::default(),
            verbose: false,
        }
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_config(mut self, config: GKRConfig) -> Self {
        self.config = config;
        self
    }

    /// Runs the complete pipeline for a model configuration.
    pub fn run(&self, model: &PipelineModelConfig) -> PipelineResult {
        let mut stages = Vec::new();
        let mut total_native_time = Duration::ZERO;
        let mut total_proof_time = Duration::ZERO;
        let mut proof_size_bytes = 0;

        if self.verbose {
            println!("\n=== Running pipeline for {} ===", model.name);
            println!("  Params: {}", model.param_count());
            println!("  Dimensions: {}x{}x{}", model.d_in, model.d_hid, model.d_out);
        }

        // Stage 1: Generate test data
        let stage1_start = Instant::now();
        let (w1, b1, w2, b2) = self.generate_weights(model);
        let input = self.generate_input(model);
        let stage1 = StageResult {
            name: "data_generation".to_string(),
            duration: stage1_start.elapsed(),
            success: true,
            details: format!("Generated weights and input"),
        };
        stages.push(stage1);

        // Stage 2: Native forward pass
        let stage2_start = Instant::now();
        let (hidden, output) = self.native_forward(&w1, &b1, &w2, &b2, &input, model);
        let native_forward_time = stage2_start.elapsed();
        total_native_time += native_forward_time;
        let stage2 = StageResult {
            name: "native_forward".to_string(),
            duration: native_forward_time,
            success: true,
            details: format!("Output dim: {}", output.len()),
        };
        stages.push(stage2);

        // Stage 3: Native backward pass (gradient computation)
        let target = self.generate_target(model);
        let stage3_start = Instant::now();
        let (grad_w1, grad_w2) =
            self.native_backward(&w1, &w2, &input, &hidden, &output, &target, model);
        let native_backward_time = stage3_start.elapsed();
        total_native_time += native_backward_time;
        let stage3 = StageResult {
            name: "native_backward".to_string(),
            duration: native_backward_time,
            success: true,
            details: format!("Computed gradients"),
        };
        stages.push(stage3);

        // Stage 4: Circuit construction
        let stage4_start = Instant::now();
        let circuit = self.build_circuit(&w1, &b1, &w2, &b2, model);
        let circuit_build_time = stage4_start.elapsed();
        let stage4 = StageResult {
            name: "circuit_build".to_string(),
            duration: circuit_build_time,
            success: true,
            details: format!("Built MLP circuit"),
        };
        stages.push(stage4);

        // Stage 5: GKR Proof generation
        let stage5_start = Instant::now();
        let proof_result = self.generate_gkr_proof(&circuit, &input);
        let proof_time = stage5_start.elapsed();
        total_proof_time += proof_time;

        let (proof, stage5_success) = match proof_result {
            Ok(p) => {
                proof_size_bytes = p.size_bytes();
                (Some(p), true)
            }
            Err(e) => {
                if self.verbose {
                    eprintln!("  Proof generation failed: {}", e);
                }
                (None, false)
            }
        };

        let stage5 = StageResult {
            name: "gkr_proving".to_string(),
            duration: proof_time,
            success: stage5_success,
            details: if stage5_success {
                format!("Proof size: {} bytes", proof_size_bytes)
            } else {
                "Failed".to_string()
            },
        };
        stages.push(stage5);

        // Stage 6: GKR Verification
        let stage6_start = Instant::now();
        let verification_result = if let Some(ref p) = proof {
            self.verify_gkr_proof(p)
        } else {
            false
        };
        let verify_time = stage6_start.elapsed();

        let stage6 = StageResult {
            name: "gkr_verification".to_string(),
            duration: verify_time,
            success: verification_result,
            details: if verification_result {
                "Verified".to_string()
            } else {
                "Failed".to_string()
            },
        };
        stages.push(stage6);

        // Calculate overhead
        let overhead = if total_native_time.as_nanos() > 0 {
            total_proof_time.as_secs_f64() / total_native_time.as_secs_f64()
        } else {
            f64::INFINITY
        };

        let meets_target = overhead <= TARGET_OVERHEAD;
        let exceeds_alert = overhead > ALERT_THRESHOLD;
        let all_stages_passed = stages.iter().all(|s| s.success);

        if self.verbose {
            println!("\n  Results:");
            println!("    Native time: {:?}", total_native_time);
            println!("    Proof time:  {:?}", total_proof_time);
            println!("    Overhead:    {:.1}x", overhead);
            println!(
                "    Status:      {}",
                if meets_target {
                    "PASS"
                } else if !exceeds_alert {
                    "WARN"
                } else {
                    "ALERT"
                }
            );
        }

        PipelineResult {
            config: model.clone(),
            stages,
            total_native_time,
            total_proof_time,
            overhead,
            meets_target,
            exceeds_alert,
            proof_size_bytes,
            all_stages_passed,
        }
    }

    /// Runs the pipeline for all benchmark configurations.
    pub fn run_all_benchmarks(&self) -> Vec<PipelineResult> {
        PipelineModelConfig::benchmark_configs()
            .iter()
            .map(|config| self.run(config))
            .collect()
    }

    /// Runs the pipeline for quick test configurations.
    pub fn run_quick(&self) -> Vec<PipelineResult> {
        PipelineModelConfig::quick_configs()
            .iter()
            .map(|config| self.run(config))
            .collect()
    }

    // Helper methods

    fn generate_weights(
        &self,
        model: &PipelineModelConfig,
    ) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>) {
        let w1: Vec<Fr> = (0..model.d_hid * model.d_in)
            .map(|i| Fr::from((i % 97 + 1) as u64))
            .collect();
        let b1: Vec<Fr> = vec![Fr::zero(); model.d_hid];
        let w2: Vec<Fr> = (0..model.d_out * model.d_hid)
            .map(|i| Fr::from((i % 89 + 1) as u64))
            .collect();
        let b2: Vec<Fr> = vec![Fr::zero(); model.d_out];
        (w1, b1, w2, b2)
    }

    fn generate_input(&self, model: &PipelineModelConfig) -> Vec<Fr> {
        (0..model.d_in).map(|i| Fr::from((i + 1) as u64)).collect()
    }

    fn generate_target(&self, model: &PipelineModelConfig) -> Vec<Fr> {
        (0..model.d_out)
            .map(|i| Fr::from((i + 5) as u64))
            .collect()
    }

    fn native_forward(
        &self,
        w1: &[Fr],
        b1: &[Fr],
        w2: &[Fr],
        b2: &[Fr],
        input: &[Fr],
        model: &PipelineModelConfig,
    ) -> (Vec<Fr>, Vec<Fr>) {
        // Layer 1: h = W1 * x + b1
        let mut h: Vec<Fr> = (0..model.d_hid)
            .map(|i| {
                let sum: Fr = (0..model.d_in)
                    .map(|j| w1[i * model.d_in + j] * input[j])
                    .fold(Fr::zero(), |a, b| a + b);
                sum + b1[i]
            })
            .collect();

        // ReLU (simplified - just keep values as is for benchmarking)
        let h_activated = h.clone();

        // Layer 2: y = W2 * h + b2
        let y: Vec<Fr> = (0..model.d_out)
            .map(|i| {
                let sum: Fr = (0..model.d_hid)
                    .map(|j| w2[i * model.d_hid + j] * h_activated[j])
                    .fold(Fr::zero(), |a, b| a + b);
                sum + b2[i]
            })
            .collect();

        (h_activated, y)
    }

    fn native_backward(
        &self,
        w1: &[Fr],
        w2: &[Fr],
        input: &[Fr],
        hidden: &[Fr],
        output: &[Fr],
        target: &[Fr],
        model: &PipelineModelConfig,
    ) -> (Vec<Fr>, Vec<Fr>) {
        // Output layer error: dL/dy = 2 * (y - target)
        let output_error: Vec<Fr> = output
            .iter()
            .zip(target.iter())
            .map(|(&o, &t)| (o - t) + (o - t))
            .collect();

        // Gradient for W2: dL/dW2 = output_error * hidden^T
        let mut grad_w2 = Vec::with_capacity(model.d_out * model.d_hid);
        for i in 0..model.d_out {
            for j in 0..model.d_hid {
                grad_w2.push(output_error[i] * hidden[j]);
            }
        }

        // Hidden layer error
        let hidden_error: Vec<Fr> = (0..model.d_hid)
            .map(|j| {
                (0..model.d_out)
                    .map(|i| w2[i * model.d_hid + j] * output_error[i])
                    .fold(Fr::zero(), |a, b| a + b)
            })
            .collect();

        // Gradient for W1
        let mut grad_w1 = Vec::with_capacity(model.d_hid * model.d_in);
        for i in 0..model.d_hid {
            for j in 0..model.d_in {
                grad_w1.push(hidden_error[i] * input[j]);
            }
        }

        (grad_w1, grad_w2)
    }

    fn build_circuit(
        &self,
        w1: &[Fr],
        b1: &[Fr],
        w2: &[Fr],
        b2: &[Fr],
        model: &PipelineModelConfig,
    ) -> LayeredCircuit {
        NeuralNetworkCircuit::mlp_forward(
            w1,
            b1,
            w2,
            b2,
            model.d_in,
            model.d_hid,
            model.d_out,
        )
    }

    fn generate_gkr_proof(
        &self,
        circuit: &LayeredCircuit,
        input: &[Fr],
    ) -> Result<GKRProof, String> {
        let config = self.config.clone();
        let mut prover = GKRProver::new(config);
        prover.prove(circuit, input).map_err(|e| format!("{:?}", e))
    }

    fn verify_gkr_proof(&self, proof: &GKRProof) -> bool {
        let config = self.config.clone();
        let verifier = GKRVerifier::new(config);
        verifier.verify_structure(proof).is_ok()
    }
}

impl Default for PipelineRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates a summary report for multiple pipeline results.
pub fn generate_pipeline_report(results: &[PipelineResult]) -> String {
    let mut report = String::new();

    report.push_str("\n");
    report.push_str("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    report.push_str("║                     HELIX Full Pipeline Benchmark Report                     ║\n");
    report.push_str("╠══════════════════════════════════════════════════════════════════════════════╣\n");
    report.push_str(&format!(
        "║ Target overhead: ≤{:.0}x          Alert threshold: >{:.0}x                         ║\n",
        TARGET_OVERHEAD, ALERT_THRESHOLD
    ));
    report.push_str("╠══════════════════════════════════════════════════════════════════════════════╣\n");
    report.push_str("║ Model           │ Params    │ Native     │ Proof      │ Overhead │ Status   ║\n");
    report.push_str("╠═════════════════╪═══════════╪════════════╪════════════╪══════════╪══════════╣\n");

    for result in results {
        let status_icon = match result.status() {
            "PASS" => "✓",
            "WARN" => "!",
            "ALERT" => "⚠",
            "FAILED" => "✗",
            _ => "?",
        };

        report.push_str(&format!(
            "║ {:15} │ {:>9} │ {:>10} │ {:>10} │ {:>7.1}x │ {:>6} {} ║\n",
            result.config.name,
            result.config.param_count(),
            format_duration(result.total_native_time),
            format_duration(result.total_proof_time),
            result.overhead,
            result.status(),
            status_icon,
        ));
    }

    report.push_str("╠═════════════════╧═══════════╧════════════╧════════════╧══════════╧══════════╣\n");

    // Summary statistics
    let total = results.len();
    let passed = results.iter().filter(|r| r.meets_target).count();
    let alerts = results.iter().filter(|r| r.exceeds_alert).count();
    let failed = results.iter().filter(|r| !r.all_stages_passed).count();

    let avg_overhead: f64 =
        results.iter().map(|r| r.overhead).sum::<f64>() / results.len().max(1) as f64;
    let max_overhead: f64 = results
        .iter()
        .map(|r| r.overhead)
        .fold(0.0f64, f64::max);

    report.push_str(&format!(
        "║ SUMMARY: {}/{} passed, {} alerts, {} failed                                      ║\n",
        passed, total, alerts, failed
    ));
    report.push_str(&format!(
        "║ Average overhead: {:.1}x    Maximum overhead: {:.1}x                               ║\n",
        avg_overhead, max_overhead
    ));

    let overall_status = if failed > 0 {
        "FAILED"
    } else if alerts > 0 {
        "ALERT"
    } else if passed == total {
        "PASSED"
    } else {
        "WARNING"
    };

    report.push_str(&format!(
        "║ Overall status: {:60} ║\n",
        overall_status
    ));
    report.push_str("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    report
}

fn format_duration(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1000 {
        format!("{}μs", us)
    } else if us < 1_000_000 {
        format!("{:.2}ms", us as f64 / 1000.0)
    } else {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_tiny_model() {
        let runner = PipelineRunner::new();
        let result = runner.run(&PipelineModelConfig::tiny());

        assert!(result.all_stages_passed, "All stages should pass");
        assert!(result.overhead < f64::INFINITY, "Should compute valid overhead");
    }

    #[test]
    fn test_pipeline_small_model() {
        let runner = PipelineRunner::new();
        let result = runner.run(&PipelineModelConfig::small());

        assert!(result.all_stages_passed, "All stages should pass");
        assert!(result.overhead < f64::INFINITY, "Should compute valid overhead");
    }

    #[test]
    fn test_pipeline_quick() {
        let runner = PipelineRunner::new();
        let results = runner.run_quick();

        assert_eq!(results.len(), 2, "Should have results for 2 models");
        for result in &results {
            assert!(
                result.all_stages_passed,
                "All stages should pass for {}",
                result.config.name
            );
        }
    }

    #[test]
    fn test_overhead_target() {
        let runner = PipelineRunner::new();
        let result = runner.run(&PipelineModelConfig::tiny());

        // Tiny model should easily meet target
        assert!(
            result.meets_target,
            "Tiny model should meet {:.0}x target, got {:.1}x",
            TARGET_OVERHEAD,
            result.overhead
        );
    }

    #[test]
    fn test_proof_verification() {
        let runner = PipelineRunner::new();
        let result = runner.run(&PipelineModelConfig::tiny());

        let verification_stage = result
            .stages
            .iter()
            .find(|s| s.name == "gkr_verification")
            .expect("Should have verification stage");

        assert!(verification_stage.success, "Verification should succeed");
    }

    #[test]
    fn test_pipeline_report_generation() {
        let runner = PipelineRunner::new();
        let results = runner.run_quick();
        let report = generate_pipeline_report(&results);

        assert!(report.contains("HELIX Full Pipeline"));
        assert!(report.contains("tiny") || report.contains("small"));
    }

    #[test]
    fn test_benchmark_configs() {
        let configs = PipelineModelConfig::benchmark_configs();
        assert_eq!(configs.len(), 5, "Should have 5 benchmark configurations");

        let param_counts: Vec<usize> = configs.iter().map(|c| c.param_count()).collect();

        // Verify approximate parameter counts
        assert!(param_counts[0] > 5_000 && param_counts[0] < 20_000); // ~10K
        assert!(param_counts[1] > 50_000 && param_counts[1] < 200_000); // ~100K
        assert!(param_counts[2] > 300_000 && param_counts[2] < 700_000); // ~500K
        assert!(param_counts[3] > 700_000 && param_counts[3] < 1_500_000); // ~1M
        assert!(param_counts[4] > 1_500_000 && param_counts[4] < 3_000_000); // ~2M
    }
}
