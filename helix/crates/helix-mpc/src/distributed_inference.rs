//! Distributed MPC inference forward pass.
//!
//! Each worker runs this function with only their share of the model weights.
//! Communication happens over the `MPCTransport` (TCP mesh between workers).
//!
//! # Protocol (2-layer MNIST: 784→32→10)
//!
//! 1. **Layer 1** (local): `hidden_share_i = x_pub @ W1_i^T + b1_i`
//!    - No communication needed (input is public, weights are shared).
//! 2. **ReLU all-reduce**: Workers broadcast hidden shares, reconstruct h,
//!    apply ReLU. Reveals activations (acceptable: input is public).
//! 3. **Layer 2** (local): `output_share_i = relu_pub @ W2_i^T + b2_i`
//!    - No communication needed (relu output is now public).
//! 4. **Output all-reduce**: Workers broadcast output shares, reconstruct logits.
//! 5. **Softmax**: Applied locally on reconstructed logits.
//!
//! Total network rounds: 2 (one per all-reduce). Weight shares never leave workers.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::nn::linear::SecureLinear;
use crate::protocols::arithmetic::SecureArithmetic;
use crate::session::transport::MPCTransport;

/// Weight shares held by a single worker (their additive share of each parameter).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerWeightShares {
    /// Share of Layer 1 weights, flattened [hidden × input].
    pub w1: Vec<Fr>,
    /// Share of Layer 1 biases [hidden].
    pub b1: Vec<Fr>,
    /// Share of Layer 2 weights, flattened [output × hidden].
    pub w2: Vec<Fr>,
    /// Share of Layer 2 biases [output].
    pub b2: Vec<Fr>,
}

/// Result of a distributed MPC inference run.
#[derive(Debug, Clone)]
pub struct DistributedInferenceResult {
    /// Predicted class (0-9 for MNIST).
    pub prediction: usize,
    /// Confidence (probability of predicted class).
    pub confidence: f64,
    /// Full output probabilities after softmax.
    pub probabilities: Vec<f64>,
    /// SHA-256 hash of the output probabilities (for on-chain attestation).
    pub output_hash: [u8; 32],
    /// SHA-256 hash of the input pixels (for on-chain attestation).
    pub input_hash: [u8; 32],
}

/// Runs the distributed MPC inference forward pass on this worker's share.
///
/// Each worker calls this function concurrently. The `transport` handles
/// inter-worker communication for the all-reduce steps.
pub async fn distributed_forward_pass<T: MPCTransport>(
    transport: &T,
    shares: &WorkerWeightShares,
    input_pixels: &[f64],
    in_features: usize,
    hidden_features: usize,
    out_features: usize,
) -> MPCResult<DistributedInferenceResult> {
    // Convert public input to field elements
    let x_public: Vec<Fr> = input_pixels.iter().map(|&v| Fr::from_f64(v)).collect();

    // ── Phase 1: Layer 1 (local, no communication) ──────────────────────
    //
    // hidden_share_i = x_pub @ W1_i^T + b1_i
    // Wrap this party's shares in a 1-element slice for forward_public_input.
    let w1_wrapped = [shares.w1.clone()];
    let b1_wrapped = [shares.b1.clone()];
    let hidden_shares = SecureLinear::forward_public_input(
        &x_public,
        &w1_wrapped,
        Some(&b1_wrapped),
        1, // batch_size
        in_features,
        hidden_features,
    );
    let my_hidden_share = &hidden_shares[0]; // This party's share

    // ── Phase 2: ReLU via all-reduce ────────────────────────────────────
    //
    // Broadcast hidden shares, reconstruct full hidden values, apply ReLU.
    let hidden_full = all_reduce(transport, my_hidden_share).await?;
    let relu_values: Vec<Fr> = hidden_full
        .into_iter()
        .map(|v| {
            let f = v.to_f64();
            Fr::from_f64(if f > 0.0 { f } else { 0.0 })
        })
        .collect();

    // ── Phase 3: Layer 2 (local, no communication) ──────────────────────
    //
    // Since relu_values are now public (reconstructed), treat as public input.
    // output_share_i = relu_pub @ W2_i^T + b2_i
    let w2_wrapped = [shares.w2.clone()];
    let b2_wrapped = [shares.b2.clone()];
    let output_shares = SecureLinear::forward_public_input(
        &relu_values,
        &w2_wrapped,
        Some(&b2_wrapped),
        1, // batch_size
        hidden_features,
        out_features,
    );
    let my_output_share = &output_shares[0];

    // ── Phase 4: Reconstruct output via all-reduce ──────────────────────
    let logits = all_reduce(transport, my_output_share).await?;

    // ── Phase 5: Softmax and result ─────────────────────────────────────
    let logit_f64: Vec<f64> = logits.iter().map(|v| v.to_f64()).collect();
    let probabilities = softmax(&logit_f64);
    let prediction = probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let confidence = probabilities[prediction];

    // Compute attestation hashes
    let input_hash = sha256_f64(input_pixels);
    let output_hash = sha256_f64(&probabilities);

    Ok(DistributedInferenceResult {
        prediction,
        confidence,
        probabilities,
        output_hash,
        input_hash,
    })
}

// ============================================================================
// All-reduce: broadcast + sum reconstruction
// ============================================================================

/// Broadcasts this party's share to all peers and sums all shares to reconstruct.
async fn all_reduce<T: MPCTransport>(
    transport: &T,
    my_share: &[Fr],
) -> MPCResult<Vec<Fr>> {
    let msg = SecureArithmetic::serialize_share_batch(my_share);
    transport.broadcast(&msg).await?;

    // Start with own share
    let mut sum: Vec<Fr> = my_share.to_vec();

    // Receive and accumulate from all peers
    for peer in transport.peers() {
        let data = transport.recv(&peer).await?;
        let peer_share = SecureArithmetic::deserialize_share_batch(&data)?;
        if peer_share.len() != sum.len() {
            return Err(MPCError::ProtocolError(format!(
                "share size mismatch: expected {}, got {} from {:?}",
                sum.len(),
                peer_share.len(),
                peer,
            )));
        }
        for (s, p) in sum.iter_mut().zip(peer_share.iter()) {
            *s = &*s + p;
        }
    }

    Ok(sum)
}

// ============================================================================
// Helpers
// ============================================================================

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

fn sha256_f64(data: &[f64]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for &v in data {
        hasher.update(v.to_le_bytes());
    }
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
    use crate::session::transport::LocalTransport;
    use crate::types::PartyId;
    use rand::Rng;

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

    /// Test: 3-party distributed inference on a tiny model.
    #[tokio::test]
    async fn test_distributed_inference_3_party() {
        let num_parties = 3;
        let in_features = 4; // Tiny model for testing
        let hidden_features = 3;
        let out_features = 2;

        // Create model weights (plaintext)
        let w1: Vec<f64> = (0..hidden_features * in_features)
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        let b1 = vec![0.01; hidden_features];
        let w2: Vec<f64> = (0..out_features * hidden_features)
            .map(|i| 0.05 * (i as f64 + 1.0))
            .collect();
        let b2 = vec![0.02; out_features];

        // Secret-share the weights
        let w1_shares = share_vector(&w1, num_parties);
        let b1_shares = share_vector(&b1, num_parties);
        let w2_shares = share_vector(&w2, num_parties);
        let b2_shares = share_vector(&b2, num_parties);

        // Create transport mesh (each party gets an owned transport)
        let parties: Vec<PartyId> = (0..num_parties)
            .map(|i| PartyId::from_index(i))
            .collect();
        let transports = LocalTransport::create_mesh(&parties);

        // Input (public)
        let input = vec![0.5, 0.3, 0.8, 0.1];

        // Run inference on all parties concurrently
        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let shares = WorkerWeightShares {
                w1: w1_shares[i].clone(),
                b1: b1_shares[i].clone(),
                w2: w2_shares[i].clone(),
                b2: b2_shares[i].clone(),
            };
            let inp = input.clone();
            handles.push(tokio::spawn(async move {
                distributed_forward_pass(
                    &transport,
                    &shares,
                    &inp,
                    in_features,
                    hidden_features,
                    out_features,
                )
                .await
            }));
        }

        // Collect results — all parties should agree
        let mut results = Vec::new();
        for h in handles {
            let r = h.await.unwrap().unwrap();
            results.push(r);
        }

        // All parties must agree on prediction
        let prediction = results[0].prediction;
        for r in &results {
            assert_eq!(r.prediction, prediction, "All parties must agree on prediction");
            assert_eq!(r.output_hash, results[0].output_hash, "Output hashes must match");
            assert_eq!(r.input_hash, results[0].input_hash, "Input hashes must match");
        }

        // Verify probabilities sum to ~1
        let prob_sum: f64 = results[0].probabilities.iter().sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1, got {}",
            prob_sum
        );

        // Verify confidence is reasonable
        assert!(results[0].confidence > 0.0 && results[0].confidence <= 1.0);
    }
}
