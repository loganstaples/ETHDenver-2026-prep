//! MPC-based inference: forward pass through a neural network using secret-shared weights.
//!
//! The model owner provides the full weights (e.g. loaded from 0G Storage).
//! Weights are split into additive secret shares across N simulated parties.
//! Each party computes their share of the forward pass using the MPC primitives:
//!
//! - Layer 1: `SecureLinear::forward_public_input` (input is public, no communication)
//! - ReLU: reconstruct-reshare (fast, reveals activations but not weights)
//! - Layer 2: `SecureLinear::forward_shared` (shared input × shared weights, Beaver triples)
//! - Output: reconstruct shares → softmax → classification

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use helix_mpc::beaver::dealer::TrustedDealer;
use helix_mpc::beaver::pool::BeaverPool;
use helix_mpc::field::Fr;
use helix_mpc::nn::linear::SecureLinear;

/// Model weights in the standard MNIST 2-layer format.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelWeights {
    pub w1: Vec<f64>, // [hidden × input] = [128 × 784] = 100352 elements
    pub b1: Vec<f64>, // [hidden] = [128]
    pub w2: Vec<f64>, // [output × hidden] = [10 × 128] = 1280 elements
    pub b2: Vec<f64>, // [output] = [10]
}

/// Result of MPC inference.
#[derive(Debug, Clone, Serialize)]
pub struct MPCInferenceResult {
    /// Predicted digit (0-9)
    pub prediction: usize,
    /// Confidence (probability of predicted digit)
    pub confidence: f64,
    /// Full probability distribution over digits 0-9
    pub probabilities: Vec<f64>,
    /// Number of MPC parties used
    pub num_parties: usize,
    /// Time breakdown in milliseconds
    pub timing: InferenceTiming,
}

#[derive(Debug, Clone, Serialize)]
pub struct InferenceTiming {
    pub share_generation_ms: u64,
    pub layer1_ms: u64,
    pub relu_ms: u64,
    pub layer2_ms: u64,
    pub reconstruction_ms: u64,
    pub total_ms: u64,
}

/// Run MPC inference on an input vector (e.g. 784 pixels for MNIST).
///
/// `weights`: Full model weights (from 0G Storage or training session).
/// `pixels`: Input vector (784 grayscale pixel values in [0, 1] for MNIST).
/// `num_parties`: Number of simulated MPC parties (e.g. 3).
pub fn run_mpc_inference(
    weights: &ModelWeights,
    pixels: &[f64],
    num_parties: usize,
) -> Result<MPCInferenceResult, String> {
    let hidden_features = weights.b1.len();
    let out_features = weights.b2.len();
    let in_features = if hidden_features > 0 {
        weights.w1.len() / hidden_features
    } else {
        return Err("Hidden layer size is 0".into());
    };

    if pixels.len() != in_features {
        return Err(format!(
            "Expected {} pixels (matching model input), got {}",
            in_features,
            pixels.len()
        ));
    }
    if num_parties < 2 {
        return Err("Need at least 2 MPC parties".into());
    }

    let total_start = Instant::now();
    let mut rng = rand::thread_rng();

    // =========================================================================
    // Phase 1: Secret-share the weights across N parties
    // =========================================================================
    let share_start = Instant::now();

    let w1_shares = share_vector(&weights.w1, num_parties, &mut rng);
    let b1_shares = share_vector(&weights.b1, num_parties, &mut rng);
    let w2_shares = share_vector(&weights.w2, num_parties, &mut rng);
    let b2_shares = share_vector(&weights.b2, num_parties, &mut rng);

    // Convert public input to Fr
    let x_public: Vec<Fr> = pixels.iter().map(|&p| Fr::from_f64(p)).collect();

    let share_ms = share_start.elapsed().as_millis() as u64;

    // =========================================================================
    // Phase 2: Layer 1 — public input × shared weights (no communication)
    // =========================================================================
    let l1_start = Instant::now();

    let hidden_shares = SecureLinear::forward_public_input(
        &x_public,
        &w1_shares,
        Some(b1_shares.as_slice()),
        1,               // batch_size
        in_features,
        hidden_features,
    );

    let l1_ms = l1_start.elapsed().as_millis() as u64;

    // =========================================================================
    // Phase 3: ReLU activation (reconstruct-reshare)
    //
    // Reveals hidden activations to parties, but keeps weights secret.
    // For inference where the input is already public, this is acceptable
    // and much faster than garbled-circuit ReLU.
    // =========================================================================
    let relu_start = Instant::now();

    let relu_shares = reconstruct_reshare_relu(&hidden_shares, &mut rng);

    let relu_ms = relu_start.elapsed().as_millis() as u64;

    // =========================================================================
    // Phase 4: Layer 2 — shared input × shared weights (needs Beaver triples)
    // =========================================================================
    let l2_start = Instant::now();

    let mut dealer = TrustedDealer::new();

    // Generate matrix Beaver triples for the layer 2 matmul
    let matrix_triples = dealer.generate_matrix_triple(1, hidden_features, out_features, num_parties);
    let mut pools: Vec<BeaverPool> = (0..num_parties)
        .map(|i| {
            let mut pool = BeaverPool::new(i, num_parties, 0);
            pool.fill_matrix(1, hidden_features, out_features, vec![matrix_triples[i].clone()]);
            pool
        })
        .collect();

    let output_shares = SecureLinear::forward_shared(
        &relu_shares,
        &w2_shares,
        Some(b2_shares.as_slice()),
        1,               // batch_size
        hidden_features,
        out_features,
        &mut pools,
    )
    .map_err(|e| format!("Layer 2 failed: {e}"))?;

    let l2_ms = l2_start.elapsed().as_millis() as u64;

    // =========================================================================
    // Phase 5: Reconstruct output and apply softmax
    // =========================================================================
    let recon_start = Instant::now();

    let logits = reconstruct_shares(&output_shares, out_features);
    let probabilities = softmax(&logits);
    let prediction = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let confidence = probabilities[prediction];

    let recon_ms = recon_start.elapsed().as_millis() as u64;
    let total_ms = total_start.elapsed().as_millis() as u64;

    Ok(MPCInferenceResult {
        prediction,
        confidence,
        probabilities,
        num_parties,
        timing: InferenceTiming {
            share_generation_ms: share_ms,
            layer1_ms: l1_ms,
            relu_ms,
            layer2_ms: l2_ms,
            reconstruction_ms: recon_ms,
            total_ms,
        },
    })
}

// =============================================================================
// Helper functions
// =============================================================================

/// Split a vector of f64 values into additive Fr shares across N parties.
fn share_vector(values: &[f64], num_parties: usize, rng: &mut impl Rng) -> Vec<Vec<Fr>> {
    let dim = values.len();
    let mut shares: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; num_parties];

    for d in 0..dim {
        let target = Fr::from_f64(values[d]);
        let mut sum = Fr::ZERO;
        for i in 0..num_parties - 1 {
            let r = Fr::from_f64(rng.gen_range(-100.0..100.0));
            shares[i][d] = r;
            sum = Fr::add(&sum, &r);
        }
        shares[num_parties - 1][d] = Fr::sub(&target, &sum);
    }

    shares
}

/// Reconstruct-and-reshare ReLU (reveals activations to parties, not weights).
fn reconstruct_reshare_relu(shares: &[Vec<Fr>], rng: &mut impl Rng) -> Vec<Vec<Fr>> {
    let num_parties = shares.len();
    let dim = shares[0].len();

    // Reconstruct: sum all party shares
    let mut values = vec![Fr::ZERO; dim];
    for s in shares {
        for (i, v) in s.iter().enumerate() {
            values[i] = Fr::add(&values[i], v);
        }
    }

    // Apply ReLU and re-share
    let activated: Vec<f64> = values
        .iter()
        .map(|v| {
            let x = v.to_f64();
            if x > 0.0 { x } else { 0.0 }
        })
        .collect();

    let mut new_shares: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; num_parties];
    for d in 0..dim {
        let target = Fr::from_f64(activated[d]);
        let mut sum = Fr::ZERO;
        for i in 0..num_parties - 1 {
            let r = Fr::from_f64(rng.gen_range(-100.0..100.0));
            new_shares[i][d] = r;
            sum = Fr::add(&sum, &r);
        }
        new_shares[num_parties - 1][d] = Fr::sub(&target, &sum);
    }

    new_shares
}

/// Reconstruct output by summing shares from all parties.
fn reconstruct_shares(shares: &[Vec<Fr>], dim: usize) -> Vec<f64> {
    let mut result = vec![Fr::ZERO; dim];
    for s in shares {
        for (i, v) in s.iter().enumerate() {
            if i < dim {
                result[i] = Fr::add(&result[i], v);
            }
        }
    }
    result.iter().map(|v| v.to_f64()).collect()
}

/// Softmax over logits.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_logits: Vec<f64> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum_exp: f64 = exp_logits.iter().sum();
    exp_logits.iter().map(|e| e / sum_exp).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_mpc_inference_basic() {
        // Create tiny random weights for a 4→3→2 network
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let weights = ModelWeights {
            w1: (0..12).map(|_| rng.gen_range(-0.5..0.5)).collect(), // 3×4
            b1: vec![0.0; 3],
            w2: (0..6).map(|_| rng.gen_range(-0.5..0.5)).collect(), // 2×3
            b2: vec![0.0; 2],
        };
        let pixels: Vec<f64> = (0..4).map(|_| rng.gen_range(0.0..1.0)).collect();

        let result = run_mpc_inference(&weights, &pixels, 3).unwrap();
        assert!(result.prediction < 2);
        assert!(result.confidence > 0.0 && result.confidence <= 1.0);
        assert_eq!(result.probabilities.len(), 2);
        assert_eq!(result.num_parties, 3);
    }
}
