//! Inference orchestration: distributed MPC forward pass + on-chain attestation.
//!
//! The owner orchestrates inference by:
//! 1. Secret-sharing model weights across N worker tasks
//! 2. Each worker runs `distributed_forward_pass` with only its share
//! 3. Workers communicate via transport (LocalTransport or TcpTransport)
//! 4. Results are signed for on-chain attestation
//! 5. Multi-party attestation is submitted to HelixCoordinatorV4

use std::time::Instant;

use serde::{Deserialize, Serialize};
use rand::Rng;
use sha2::{Digest, Sha256};

use helix_mpc::distributed_inference::{
    WorkerWeightShares, DistributedInferenceResult, distributed_forward_pass,
};
use helix_mpc::field::Fr;
use helix_mpc::session::transport::LocalTransport;
use helix_mpc::types::PartyId;

use crate::mpc_inference::ModelWeights;

/// Configuration for an inference orchestration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Number of MPC parties (workers).
    pub num_parties: usize,
    /// Optional job ID for on-chain attestation.
    pub job_id: Option<u64>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            num_parties: 3,
            job_id: None,
        }
    }
}

/// Result of the full inference orchestration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOrchestrationResult {
    /// Predicted class (0-9 for MNIST).
    pub prediction: usize,
    /// Confidence (probability of predicted class).
    pub confidence: f64,
    /// Full output probabilities after softmax.
    pub probabilities: Vec<f64>,
    /// Number of MPC parties that participated.
    pub num_parties: usize,
    /// SHA-256 hash of the input (for attestation).
    pub input_hash: [u8; 32],
    /// SHA-256 hash of the output (for attestation).
    pub output_hash: [u8; 32],
    /// Signatures from each worker (for on-chain submission).
    pub worker_signatures: Vec<Vec<u8>>,
    /// Timing breakdown.
    pub timing: InferenceOrcTiming,
}

/// Timing breakdown for inference orchestration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOrcTiming {
    /// Time to secret-share weights across parties.
    pub share_generation_ms: u64,
    /// Time for the distributed forward pass (including communication).
    pub forward_pass_ms: u64,
    /// Time for worker signature collection.
    pub signing_ms: u64,
    /// Total end-to-end time.
    pub total_ms: u64,
}

/// Runs a fully distributed MPC inference with on-chain attestation signing.
///
/// Each worker is a separate tokio task with only its additive share of the
/// model weights. Workers communicate via LocalTransport channels — the
/// protocol is identical to TCP-distributed workers.
pub async fn run_inference_orchestration(
    weights: &ModelWeights,
    pixels: &[f64],
    config: &InferenceConfig,
) -> Result<InferenceOrchestrationResult, String> {
    let total_start = Instant::now();
    let num_parties = config.num_parties;

    if num_parties < 2 {
        return Err("Need at least 2 parties for MPC inference".into());
    }

    // ── Phase 1: Secret-share weights ───────────────────────────────────
    let share_start = Instant::now();

    let w1_shares = share_vector(&weights.w1, num_parties);
    let b1_shares = share_vector(&weights.b1, num_parties);
    let w2_shares = share_vector(&weights.w2, num_parties);
    let b2_shares = share_vector(&weights.b2, num_parties);

    let share_generation_ms = share_start.elapsed().as_millis() as u64;

    // Derive model dimensions from weights
    let hidden_features = weights.b1.len();
    let in_features = weights.w1.len() / hidden_features;
    let out_features = weights.b2.len();

    // ── Phase 2: Create transport mesh and run distributed forward pass ─
    let forward_start = Instant::now();

    let parties: Vec<PartyId> = (0..num_parties)
        .map(|i| PartyId::from_index(i))
        .collect();
    let transports = LocalTransport::create_mesh(&parties);

    // Spawn a task for each worker — each only gets its own shares
    let mut handles = Vec::with_capacity(num_parties);
    for (i, transport) in transports.into_iter().enumerate() {
        let shares = WorkerWeightShares {
            w1: w1_shares[i].clone(),
            b1: b1_shares[i].clone(),
            w2: w2_shares[i].clone(),
            b2: b2_shares[i].clone(),
        };
        let inp = pixels.to_vec();
        let in_f = in_features;
        let hid_f = hidden_features;
        let out_f = out_features;

        handles.push(tokio::spawn(async move {
            distributed_forward_pass(
                &transport,
                &shares,
                &inp,
                in_f,
                hid_f,
                out_f,
            )
            .await
        }));
    }

    // Collect results from all workers
    let mut results: Vec<DistributedInferenceResult> = Vec::with_capacity(num_parties);
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle
            .await
            .map_err(|e| format!("Worker {} task panicked: {}", i, e))?
            .map_err(|e| format!("Worker {} inference failed: {}", i, e))?;
        results.push(result);
    }

    let forward_pass_ms = forward_start.elapsed().as_millis() as u64;

    // Verify all workers agree
    let prediction = results[0].prediction;
    let confidence = results[0].confidence;
    let probabilities = results[0].probabilities.clone();
    let output_hash = results[0].output_hash;
    let input_hash = results[0].input_hash;

    for (i, r) in results.iter().enumerate().skip(1) {
        if r.prediction != prediction {
            return Err(format!(
                "Worker {} disagrees on prediction: {} vs {}",
                i, r.prediction, prediction
            ));
        }
    }

    // ── Phase 3: Generate attestation signatures ────────────────────────
    let sign_start = Instant::now();

    let mut worker_signatures = Vec::with_capacity(num_parties);
    let job_id = config.job_id.unwrap_or(0);

    for i in 0..num_parties {
        // Off-chain signature: SHA-256 of domain tag + fields
        // In chain mode, this would be ECDSA via worker wallets
        let mut data = Vec::with_capacity(80);
        data.extend_from_slice(b"HELIX_INFERENCE");
        data.extend_from_slice(&job_id.to_le_bytes());
        data.extend_from_slice(&(prediction as u64).to_le_bytes());
        data.extend_from_slice(&input_hash);
        data.extend_from_slice(&output_hash);
        data.extend_from_slice(&(i as u32).to_le_bytes()); // party index
        let sig = sha256_hash(&data);
        worker_signatures.push(sig.to_vec());
    }

    let signing_ms = sign_start.elapsed().as_millis() as u64;
    let total_ms = total_start.elapsed().as_millis() as u64;

    Ok(InferenceOrchestrationResult {
        prediction,
        confidence,
        probabilities,
        num_parties,
        input_hash,
        output_hash,
        worker_signatures,
        timing: InferenceOrcTiming {
            share_generation_ms,
            forward_pass_ms,
            signing_ms,
            total_ms,
        },
    })
}

// ============================================================================
// Helpers
// ============================================================================

/// Secret-share a vector of f64 values into Fr field element shares.
fn share_vector(values: &[f64], num_parties: usize) -> Vec<Vec<Fr>> {
    let mut rng = rand::thread_rng();
    let dim = values.len();
    let mut shares = vec![vec![Fr::ZERO; dim]; num_parties];
    for d in 0..dim {
        let target = Fr::from_f64(values[d]);
        let mut sum = Fr::ZERO;
        for i in 0..num_parties - 1 {
            let r = Fr::from_f64(rng.gen_range(-100.0..100.0));
            shares[i][d] = r;
            sum = &sum + &r;
        }
        shares[num_parties - 1][d] = &target - &sum;
    }
    shares
}

fn sha256_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_inference_orchestration_basic() {
        // Tiny model: 4 inputs → 3 hidden → 2 outputs
        let weights = ModelWeights {
            w1: (0..12).map(|i| 0.1 * (i as f64 + 1.0)).collect(),
            b1: vec![0.01; 3],
            w2: (0..6).map(|i| 0.05 * (i as f64 + 1.0)).collect(),
            b2: vec![0.02; 2],
        };

        let pixels = vec![0.5, 0.3, 0.8, 0.1];
        let config = InferenceConfig {
            num_parties: 3,
            job_id: Some(1),
        };

        let result = run_inference_orchestration(&weights, &pixels, &config)
            .await
            .expect("Inference should succeed");

        // Basic sanity checks
        assert!(result.prediction < 2, "Prediction should be 0 or 1");
        assert!(result.confidence > 0.0 && result.confidence <= 1.0);
        assert_eq!(result.probabilities.len(), 2);
        assert!((result.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert_eq!(result.num_parties, 3);
        assert_eq!(result.worker_signatures.len(), 3);
        // Timing may be 0ms for tiny models (sub-millisecond)
        assert!(result.timing.total_ms < 5000, "Should complete quickly");
    }

    #[tokio::test]
    async fn test_inference_orchestration_deterministic() {
        // Same weights + input should give same prediction
        let weights = ModelWeights {
            w1: vec![0.1; 12],
            b1: vec![0.01; 3],
            w2: vec![0.05; 6],
            b2: vec![0.02; 2],
        };
        let pixels = vec![1.0, 0.0, 1.0, 0.0];
        let config = InferenceConfig::default();

        let r1 = run_inference_orchestration(&weights, &pixels, &config).await.unwrap();
        let r2 = run_inference_orchestration(&weights, &pixels, &config).await.unwrap();

        assert_eq!(r1.prediction, r2.prediction);
        // Output hashes may differ due to floating point precision in different runs
        // but prediction should be deterministic
    }
}
