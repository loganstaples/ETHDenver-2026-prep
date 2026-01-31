//! Secure comparison protocols for secret-shared values.
//!
//! This module implements protocols for comparing secret-shared values
//! without revealing them. Key operations:
//! - Less than: [x] < [y] → [b] where b ∈ {0, 1}
//! - Sign: sign([x]) → [b] where b = 1 if x ≥ 0, else 0
//! - ReLU: max(0, [x]) → [max(0, x)]
//!
//! # Approaches
//!
//! 1. **Bit Decomposition**: Convert shares to bit representation and compare bitwise
//! 2. **Garbled Circuits**: Use garbled circuits for comparison
//! 3. **Polynomial Approximation**: Use smooth approximations (less accurate but fast)
//!
//! For the demo, we implement a hybrid approach using bit decomposition
//! for smaller values and polynomial approximation for speed.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

use crate::beaver::pool::BeaverPool;
use crate::beaver::triple::BeaverTriple;
use crate::error::{MPCError, MPCResult};
use crate::protocols::arithmetic::SecureArithmetic;
use crate::types::PartyId;

/// Configuration for comparison protocols.
#[derive(Debug, Clone)]
pub struct ComparisonConfig {
    /// Bit length for integer comparison (e.g., 32 for i32)
    pub bit_length: usize,
    /// Use garbled circuits (more secure, slower)
    pub use_garbled_circuits: bool,
    /// Scaling factor for fixed-point representation
    pub scale: f64,
    /// Number of iterations for iterative protocols
    pub iterations: usize,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            bit_length: 32,
            use_garbled_circuits: false,
            scale: 1000.0, // 3 decimal places precision
            iterations: 10,
        }
    }
}

/// Secure comparison protocol implementation.
pub struct SecureComparison {
    config: ComparisonConfig,
}

impl SecureComparison {
    pub fn new(config: ComparisonConfig) -> Self {
        Self { config }
    }

    /// Computes [x < 0] using bit decomposition simulation.
    ///
    /// Returns shares of 1 if x < 0, shares of 0 otherwise.
    pub fn sign_bit(
        &self,
        x_shares: &[f64],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<f64>> {
        let num_parties = x_shares.len();

        // Reconstruct x to determine sign (simulation mode)
        // In production, this would use bit decomposition + carry propagation
        let x: f64 = x_shares.iter().sum();

        // Result: 1 if x >= 0, 0 if x < 0
        let sign = if x >= 0.0 { 1.0 } else { 0.0 };

        // Re-share the result
        self.reshare_bit(sign, num_parties)
    }

    /// Computes [x < y] securely.
    ///
    /// Returns shares of 1 if x < y, shares of 0 otherwise.
    pub fn less_than(
        &self,
        x_shares: &[f64],
        y_shares: &[f64],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<f64>> {
        let num_parties = x_shares.len();

        // Compute [x - y]
        let diff_shares: Vec<f64> = x_shares
            .iter()
            .zip(y_shares.iter())
            .map(|(x, y)| x - y)
            .collect();

        // [x < y] iff [x - y < 0], i.e., sign bit is 1
        let sign_shares = self.sign_bit(&diff_shares, pools)?;

        // Flip: 1 - sign gives us "is negative"
        let result: Vec<f64> = sign_shares
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if i == 0 {
                    1.0 - s  // Party 0 computes 1 - s
                } else {
                    -s       // Others negate
                }
            })
            .collect();

        Ok(result)
    }

    /// Computes secure ReLU: max(0, [x]) → [max(0, x)]
    ///
    /// This is the core primitive for neural network training.
    pub fn relu(
        &self,
        x_shares: &[f64],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<f64>> {
        let num_parties = x_shares.len();

        // Compute sign bit: b = 1 if x >= 0, 0 otherwise
        let sign_shares = self.sign_bit(x_shares, pools)?;

        // ReLU = x * b
        // We need to multiply shares, which requires Beaver triples
        let triples: Vec<BeaverTriple> = pools
            .iter_mut()
            .map(|p| p.take_scalar())
            .collect::<MPCResult<Vec<_>>>()?;

        // Beaver multiplication of x and sign
        let result = SecureArithmetic::simulate_multiply(x_shares, &sign_shares, &triples);

        Ok(result)
    }

    /// Computes secure ReLU for a vector of values.
    pub fn relu_vector(
        &self,
        x_shares: &[Vec<f64>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<f64>>> {
        let num_parties = x_shares.len();
        let dim = x_shares[0].len();

        let mut result: Vec<Vec<f64>> = vec![vec![0.0; dim]; num_parties];

        for d in 0..dim {
            let elem_shares: Vec<f64> = x_shares.iter().map(|s| s[d]).collect();
            let relu_shares = self.relu(&elem_shares, pools)?;

            for i in 0..num_parties {
                result[i][d] = relu_shares[i];
            }
        }

        Ok(result)
    }

    /// Computes secure Leaky ReLU: max(αx, x) for negative x.
    pub fn leaky_relu(
        &self,
        x_shares: &[f64],
        alpha: f64,
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<f64>> {
        let num_parties = x_shares.len();

        // sign = 1 if x >= 0, 0 otherwise
        let sign_shares = self.sign_bit(x_shares, pools)?;

        // leaky_relu = x * sign + alpha * x * (1 - sign)
        //            = x * sign + alpha * x - alpha * x * sign
        //            = x * (sign + alpha - alpha * sign)
        //            = x * (alpha + sign * (1 - alpha))

        // First compute sign * (1 - alpha)
        let one_minus_alpha = 1.0 - alpha;
        let scaled_sign: Vec<f64> = sign_shares
            .iter()
            .map(|s| s * one_minus_alpha)
            .collect();

        // Add alpha to get: alpha + sign * (1 - alpha)
        let multiplier: Vec<f64> = scaled_sign
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if i == 0 {
                    alpha + s
                } else {
                    *s
                }
            })
            .collect();

        // Multiply x by this
        let triples: Vec<BeaverTriple> = pools
            .iter_mut()
            .map(|p| p.take_scalar())
            .collect::<MPCResult<Vec<_>>>()?;

        let result = SecureArithmetic::simulate_multiply(x_shares, &multiplier, &triples);

        Ok(result)
    }

    /// Polynomial approximation of sign function.
    /// sign(x) ≈ x / (|x| + ε) where we approximate |x| with sqrt(x²)
    pub fn sign_polynomial(
        &self,
        x_shares: &[f64],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<f64>> {
        let num_parties = x_shares.len();
        let eps = 0.01;

        // Compute x²
        let triples: Vec<BeaverTriple> = pools
            .iter_mut()
            .map(|p| p.take_scalar())
            .collect::<MPCResult<Vec<_>>>()?;

        let x_sq = SecureArithmetic::simulate_multiply(x_shares, x_shares, &triples);

        // Reconstruct to compute sqrt (simulation - real would use Newton iteration)
        let x_sq_val: f64 = x_sq.iter().sum();
        let abs_x = x_sq_val.abs().sqrt() + eps;

        // Compute x / (|x| + ε)
        let result: Vec<f64> = x_shares
            .iter()
            .enumerate()
            .map(|(i, xi)| {
                if i == 0 {
                    xi / abs_x
                } else {
                    *xi / abs_x
                }
            })
            .collect();

        Ok(result)
    }

    /// Re-shares a single bit value among parties.
    fn reshare_bit(&self, bit: f64, num_parties: usize) -> MPCResult<Vec<f64>> {
        let mut rng = ChaCha20Rng::from_entropy();
        let mut shares = Vec::with_capacity(num_parties);
        let mut sum = 0.0;

        for _ in 0..num_parties - 1 {
            let r: f64 = rng.gen_range(-100.0..100.0);
            shares.push(r);
            sum += r;
        }
        shares.push(bit - sum);

        Ok(shares)
    }
}

/// Garbled circuit-based comparison (simplified simulation).
///
/// In a real implementation, this would:
/// 1. Generate garbled circuit for comparison
/// 2. Use OT to transfer garbled inputs
/// 3. Evaluate circuit to get encrypted output
/// 4. Decode output
pub struct GarbledComparison {
    /// Circuit seed for reproducibility
    seed: [u8; 32],
}

impl GarbledComparison {
    pub fn new(seed: [u8; 32]) -> Self {
        Self { seed }
    }

    /// Creates a garbled circuit for less-than comparison.
    /// Returns (garbler_tables, input_labels)
    pub fn garble_less_than(
        &self,
        bit_length: usize,
    ) -> (Vec<GarbledGate>, Vec<InputLabel>) {
        let mut rng = ChaCha20Rng::from_seed(self.seed);

        let num_gates = bit_length * 3; // AND, XOR, etc.
        let mut gates = Vec::with_capacity(num_gates);
        let mut labels = Vec::with_capacity(bit_length * 2);

        // Generate input labels for both parties
        for _ in 0..(bit_length * 2) {
            let mut label0 = [0u8; 16];
            let mut label1 = [0u8; 16];
            rng.fill(&mut label0);
            rng.fill(&mut label1);

            labels.push(InputLabel {
                zero: label0,
                one: label1,
            });
        }

        // Generate garbled gates (simplified)
        for i in 0..num_gates {
            let mut table = [[0u8; 16]; 4];
            for row in &mut table {
                rng.fill(row);
            }
            gates.push(GarbledGate {
                id: i,
                table,
            });
        }

        (gates, labels)
    }

    /// Evaluates a garbled circuit given input labels.
    pub fn evaluate(
        &self,
        gates: &[GarbledGate],
        input_labels: &[[u8; 16]],
    ) -> [u8; 16] {
        // Simplified: XOR all input labels
        let mut result = [0u8; 16];
        for label in input_labels {
            for (i, b) in label.iter().enumerate() {
                result[i] ^= b;
            }
        }
        result
    }

    /// Decodes the output label to a bit.
    pub fn decode(&self, output_label: &[u8; 16], true_label: &[u8; 16]) -> bool {
        output_label == true_label
    }
}

/// A garbled gate with its truth table.
#[derive(Debug, Clone)]
pub struct GarbledGate {
    pub id: usize,
    pub table: [[u8; 16]; 4],
}

/// Input wire labels for garbled circuits.
#[derive(Debug, Clone)]
pub struct InputLabel {
    pub zero: [u8; 16],
    pub one: [u8; 16],
}

/// Bit decomposition for secure comparison.
pub struct BitDecomposition {
    /// Number of bits
    bit_length: usize,
}

impl BitDecomposition {
    pub fn new(bit_length: usize) -> Self {
        Self { bit_length }
    }

    /// Decomposes a shared value into shared bits.
    ///
    /// Each bit is represented as shares summing to 0 or 1.
    pub fn decompose(
        &self,
        value_shares: &[f64],
        scale: f64,
    ) -> MPCResult<Vec<Vec<f64>>> {
        let num_parties = value_shares.len();

        // Reconstruct value (simulation mode)
        let value: f64 = value_shares.iter().sum();
        let scaled = (value * scale).round() as i64;

        // Extract bits
        let mut bit_shares: Vec<Vec<f64>> = Vec::with_capacity(self.bit_length);
        let mut rng = ChaCha20Rng::from_entropy();

        for b in 0..self.bit_length {
            let bit = ((scaled >> b) & 1) as f64;

            // Share this bit
            let mut shares = Vec::with_capacity(num_parties);
            let mut sum = 0.0;
            for _ in 0..num_parties - 1 {
                let r: f64 = rng.gen_range(-10.0..10.0);
                shares.push(r);
                sum += r;
            }
            shares.push(bit - sum);
            bit_shares.push(shares);
        }

        Ok(bit_shares)
    }

    /// Compares two bit-decomposed values.
    /// Returns shares of 1 if first < second.
    pub fn compare_bits(
        &self,
        a_bits: &[Vec<f64>],
        b_bits: &[Vec<f64>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<f64>> {
        let num_parties = a_bits[0].len();

        // Simulate comparison by reconstructing (demo mode)
        let a_val = self.reconstruct_from_bits(a_bits);
        let b_val = self.reconstruct_from_bits(b_bits);

        let result = if a_val < b_val { 1.0 } else { 0.0 };

        // Re-share
        let mut rng = ChaCha20Rng::from_entropy();
        let mut shares = Vec::with_capacity(num_parties);
        let mut sum = 0.0;

        for _ in 0..num_parties - 1 {
            let r: f64 = rng.gen_range(-10.0..10.0);
            shares.push(r);
            sum += r;
        }
        shares.push(result - sum);

        Ok(shares)
    }

    /// Reconstructs a value from bit shares (simulation).
    fn reconstruct_from_bits(&self, bit_shares: &[Vec<f64>]) -> i64 {
        let mut value = 0i64;
        for (b, shares) in bit_shares.iter().enumerate() {
            let bit: f64 = shares.iter().sum();
            if bit.round() as i64 == 1 {
                value |= 1 << b;
            }
        }
        value
    }
}

/// DReLU (Derivative of ReLU) for backpropagation.
pub fn drelu(
    x_shares: &[f64],
    comparison: &SecureComparison,
    pools: &mut [BeaverPool],
) -> MPCResult<Vec<f64>> {
    // DReLU = 1 if x >= 0, 0 otherwise = sign_bit(x)
    comparison.sign_bit(x_shares, pools)
}

/// Secure ReLU with gradient for training.
pub struct SecureReLUWithGradient {
    comparison: SecureComparison,
}

impl SecureReLUWithGradient {
    pub fn new(config: ComparisonConfig) -> Self {
        Self {
            comparison: SecureComparison::new(config),
        }
    }

    /// Forward pass: computes ReLU and caches sign for backward.
    pub fn forward(
        &self,
        x_shares: &[f64],
        pools: &mut [BeaverPool],
    ) -> MPCResult<(Vec<f64>, Vec<f64>)> {
        // Compute sign
        let sign_shares = self.comparison.sign_bit(x_shares, pools)?;

        // Compute ReLU = x * sign
        let triples: Vec<BeaverTriple> = pools
            .iter_mut()
            .map(|p| p.take_scalar())
            .collect::<MPCResult<Vec<_>>>()?;

        let relu_shares = SecureArithmetic::simulate_multiply(x_shares, &sign_shares, &triples);

        // Return (output, cached_sign_for_backward)
        Ok((relu_shares, sign_shares))
    }

    /// Backward pass: computes gradient using cached sign.
    pub fn backward(
        &self,
        grad_output_shares: &[f64],
        cached_sign_shares: &[f64],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<f64>> {
        // Gradient = grad_output * sign
        let triples: Vec<BeaverTriple> = pools
            .iter_mut()
            .map(|p| p.take_scalar())
            .collect::<MPCResult<Vec<_>>>()?;

        let grad_input = SecureArithmetic::simulate_multiply(
            grad_output_shares,
            cached_sign_shares,
            &triples,
        );

        Ok(grad_input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::dealer::TrustedDealer;

    fn create_pools(num_parties: usize, num_triples: usize) -> Vec<BeaverPool> {
        let mut dealer = TrustedDealer::with_seed(42);
        let per_party = dealer.generate_scalar_triples(num_triples, num_parties);

        (0..num_parties)
            .map(|i| {
                let mut p = BeaverPool::new(i, num_parties, 64);
                p.fill_scalar(per_party[i].clone());
                p
            })
            .collect()
    }

    fn split_value(value: f64, n: usize) -> Vec<f64> {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut shares = Vec::with_capacity(n);
        let mut sum = 0.0;
        for _ in 0..n - 1 {
            let r: f64 = rng.gen_range(-100.0..100.0);
            shares.push(r);
            sum += r;
        }
        shares.push(value - sum);
        shares
    }

    fn reconstruct(shares: &[f64]) -> f64 {
        shares.iter().sum()
    }

    #[test]
    fn test_sign_bit_positive() {
        let mut pools = create_pools(3, 100);
        let comp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(5.0, 3);
        let sign_shares = comp.sign_bit(&x_shares, &mut pools).unwrap();

        let sign = reconstruct(&sign_shares);
        assert!((sign - 1.0).abs() < 0.01, "Sign of positive should be 1, got {}", sign);
    }

    #[test]
    fn test_sign_bit_negative() {
        let mut pools = create_pools(3, 100);
        let comp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(-5.0, 3);
        let sign_shares = comp.sign_bit(&x_shares, &mut pools).unwrap();

        let sign = reconstruct(&sign_shares);
        assert!((sign - 0.0).abs() < 0.01, "Sign of negative should be 0, got {}", sign);
    }

    #[test]
    fn test_relu_positive() {
        let mut pools = create_pools(3, 100);
        let comp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(5.0, 3);
        let relu_shares = comp.relu(&x_shares, &mut pools).unwrap();

        let result = reconstruct(&relu_shares);
        assert!((result - 5.0).abs() < 0.01, "ReLU(5) should be 5, got {}", result);
    }

    #[test]
    fn test_relu_negative() {
        let mut pools = create_pools(3, 100);
        let comp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(-5.0, 3);
        let relu_shares = comp.relu(&x_shares, &mut pools).unwrap();

        let result = reconstruct(&relu_shares);
        assert!(result.abs() < 0.01, "ReLU(-5) should be 0, got {}", result);
    }

    #[test]
    fn test_leaky_relu() {
        let mut pools = create_pools(3, 100);
        let comp = SecureComparison::new(ComparisonConfig::default());

        // Positive input
        let x_shares = split_value(5.0, 3);
        let result_shares = comp.leaky_relu(&x_shares, 0.01, &mut pools).unwrap();
        let result = reconstruct(&result_shares);
        assert!((result - 5.0).abs() < 0.1, "LeakyReLU(5) should be ~5, got {}", result);

        // Negative input
        let x_shares = split_value(-5.0, 3);
        let result_shares = comp.leaky_relu(&x_shares, 0.01, &mut pools).unwrap();
        let result = reconstruct(&result_shares);
        assert!((result - (-0.05)).abs() < 0.1, "LeakyReLU(-5) should be ~-0.05, got {}", result);
    }

    #[test]
    fn test_less_than() {
        let mut pools = create_pools(3, 100);
        let comp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(3.0, 3);
        let y_shares = split_value(5.0, 3);

        let lt_shares = comp.less_than(&x_shares, &y_shares, &mut pools).unwrap();
        let lt = reconstruct(&lt_shares);

        // 3 < 5 should be true (1)
        assert!((lt - 1.0).abs() < 0.1, "3 < 5 should be 1, got {}", lt);

        // 5 < 3 should be false (0)
        let lt_shares = comp.less_than(&y_shares, &x_shares, &mut pools).unwrap();
        let lt = reconstruct(&lt_shares);
        assert!((lt - 0.0).abs() < 0.1, "5 < 3 should be 0, got {}", lt);
    }

    #[test]
    fn test_bit_decomposition() {
        let decomp = BitDecomposition::new(8);

        let value_shares = split_value(5.0, 3);
        let bit_shares = decomp.decompose(&value_shares, 1.0).unwrap();

        // 5 = 101 in binary
        assert_eq!(bit_shares.len(), 8);

        let reconstructed = decomp.reconstruct_from_bits(&bit_shares);
        assert_eq!(reconstructed, 5, "Reconstructed value should be 5, got {}", reconstructed);
    }

    #[test]
    fn test_relu_with_gradient() {
        let mut pools = create_pools(3, 200);
        let relu = SecureReLUWithGradient::new(ComparisonConfig::default());

        // Forward pass
        let x_shares = split_value(3.0, 3);
        let (output_shares, sign_shares) = relu.forward(&x_shares, &mut pools).unwrap();

        let output = reconstruct(&output_shares);
        assert!((output - 3.0).abs() < 0.1, "ReLU forward failed");

        // Backward pass
        let grad_output_shares = split_value(1.0, 3); // upstream gradient = 1
        let grad_input_shares = relu.backward(&grad_output_shares, &sign_shares, &mut pools).unwrap();

        let grad_input = reconstruct(&grad_input_shares);
        // For positive x, gradient should pass through
        assert!((grad_input - 1.0).abs() < 0.1, "ReLU backward failed: {}", grad_input);
    }

    #[test]
    fn test_garbled_comparison_creation() {
        let gc = GarbledComparison::new([42u8; 32]);
        let (gates, labels) = gc.garble_less_than(8);

        assert!(!gates.is_empty());
        assert_eq!(labels.len(), 16); // 8 bits * 2 parties
    }
}
