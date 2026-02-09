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
use crate::error::MPCResult;
use crate::field::Fr;
use crate::protocols::arithmetic::SecureArithmetic;

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
#[allow(dead_code)]
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
        x_shares: &[Fr],
        _pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let num_parties = x_shares.len();

        // Reconstruct x to determine sign (simulation mode)
        // In production, this would use bit decomposition + carry propagation
        let mut x = Fr::ZERO;
        for share in x_shares {
            x = Fr::add(&x, share);
        }
        let x_f64 = x.to_f64();

        // Result: 1 if x >= 0, 0 if x < 0
        let sign = if x_f64 >= 0.0 {
            Fr::from_f64(1.0)
        } else {
            Fr::ZERO
        };

        // Re-share the result
        self.reshare_bit(&sign, num_parties)
    }

    /// Computes [x < y] securely.
    ///
    /// Returns shares of 1 if x < y, shares of 0 otherwise.
    pub fn less_than(
        &self,
        x_shares: &[Fr],
        y_shares: &[Fr],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let _num_parties = x_shares.len();

        // Compute [x - y]
        let diff_shares: Vec<Fr> = x_shares
            .iter()
            .zip(y_shares.iter())
            .map(|(x, y)| Fr::sub(x, y))
            .collect();

        // [x < y] iff [x - y < 0], i.e., sign bit is 1
        let sign_shares = self.sign_bit(&diff_shares, pools)?;

        // Flip: 1 - sign gives us "is negative"
        let one = Fr::from_f64(1.0);
        let result: Vec<Fr> = sign_shares
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if i == 0 {
                    Fr::sub(&one, s) // Party 0 computes 1 - s
                } else {
                    Fr::neg(s) // Others negate
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
        x_shares: &[Fr],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let _num_parties = x_shares.len();

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
        x_shares: &[Vec<Fr>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<Fr>>> {
        let num_parties = x_shares.len();
        let dim = x_shares[0].len();

        let mut result: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; num_parties];

        for d in 0..dim {
            let elem_shares: Vec<Fr> = x_shares.iter().map(|s| s[d].clone()).collect();
            let relu_shares = self.relu(&elem_shares, pools)?;

            for i in 0..num_parties {
                result[i][d] = relu_shares[i].clone();
            }
        }

        Ok(result)
    }

    /// Computes secure Leaky ReLU: max(αx, x) for negative x.
    pub fn leaky_relu(
        &self,
        x_shares: &[Fr],
        alpha: f64,
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let _num_parties = x_shares.len();

        // sign = 1 if x >= 0, 0 otherwise
        let sign_shares = self.sign_bit(x_shares, pools)?;

        // leaky_relu = x * sign + alpha * x * (1 - sign)
        //            = x * sign + alpha * x - alpha * x * sign
        //            = x * (sign + alpha - alpha * sign)
        //            = x * (alpha + sign * (1 - alpha))

        // First compute sign * (1 - alpha)
        let one_minus_alpha = Fr::from_f64(1.0 - alpha);
        let alpha_fr = Fr::from_f64(alpha);

        let scaled_sign: Vec<Fr> = sign_shares
            .iter()
            .map(|s| s.fixed_mul(&one_minus_alpha))
            .collect();

        // Add alpha to get: alpha + sign * (1 - alpha)
        let multiplier: Vec<Fr> = scaled_sign
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if i == 0 {
                    Fr::add(&alpha_fr, s)
                } else {
                    s.clone()
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
        x_shares: &[Fr],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let _num_parties = x_shares.len();
        let eps = 0.01;

        // Compute x²
        let triples: Vec<BeaverTriple> = pools
            .iter_mut()
            .map(|p| p.take_scalar())
            .collect::<MPCResult<Vec<_>>>()?;

        let x_sq = SecureArithmetic::simulate_multiply(x_shares, x_shares, &triples);

        // Reconstruct to compute sqrt (simulation - real would use Newton iteration)
        let mut x_sq_val = Fr::ZERO;
        for share in &x_sq {
            x_sq_val = Fr::add(&x_sq_val, share);
        }
        let x_sq_f64 = x_sq_val.to_f64();
        let abs_x = x_sq_f64.abs().sqrt() + eps;
        let inv_abs_x = Fr::from_f64(1.0 / abs_x);

        // Compute x / (|x| + ε)
        let result: Vec<Fr> = x_shares
            .iter()
            .map(|xi| xi.fixed_mul(&inv_abs_x))
            .collect();

        Ok(result)
    }

    /// Re-shares a single bit value among parties.
    fn reshare_bit(&self, bit: &Fr, num_parties: usize) -> MPCResult<Vec<Fr>> {
        let mut rng = ChaCha20Rng::from_entropy();
        let mut shares = Vec::with_capacity(num_parties);
        let mut sum = Fr::ZERO;

        for _ in 0..num_parties - 1 {
            let r = Fr::random(&mut rng);
            shares.push(r.clone());
            sum = Fr::add(&sum, &r);
        }
        shares.push(Fr::sub(bit, &sum));

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
    pub fn garble_less_than(&self, bit_length: usize) -> GarbledCircuit {
        let mut rng = ChaCha20Rng::from_seed(self.seed);

        // Generate random wire labels for each bit
        let mut input_labels_a = Vec::with_capacity(bit_length);
        let mut input_labels_b = Vec::with_capacity(bit_length);

        for _ in 0..bit_length {
            let label0: [u8; 16] = rng.gen();
            let label1: [u8; 16] = rng.gen();
            input_labels_a.push((label0, label1));

            let label0: [u8; 16] = rng.gen();
            let label1: [u8; 16] = rng.gen();
            input_labels_b.push((label0, label1));
        }

        // Generate output labels
        let output_false: [u8; 16] = rng.gen();
        let output_true: [u8; 16] = rng.gen();

        // Generate garbled gates (simplified - just stores gate info)
        let mut gates = Vec::new();
        for i in 0..bit_length {
            gates.push(GarbledGate {
                input_wires: (i, bit_length + i),
                output_wire: 2 * bit_length + i,
                garbled_table: vec![rng.gen(), rng.gen(), rng.gen(), rng.gen()],
            });
        }

        GarbledCircuit {
            gates,
            input_labels_a,
            input_labels_b,
            output_labels: (output_false, output_true),
        }
    }

    /// Evaluates a garbled circuit given input labels.
    pub fn evaluate(
        &self,
        _circuit: &GarbledCircuit,
        input_labels_a: &[[u8; 16]],
        input_labels_b: &[[u8; 16]],
    ) -> [u8; 16] {
        // Simplified evaluation - just returns a deterministic result
        // Real implementation would evaluate through the circuit
        let mut hasher = Sha256::new();
        for label in input_labels_a {
            hasher.update(label);
        }
        for label in input_labels_b {
            hasher.update(label);
        }
        let result = hasher.finalize();
        let mut output = [0u8; 16];
        output.copy_from_slice(&result[..16]);
        output
    }

    /// Securely computes x < y using garbled circuits (simulation).
    pub fn secure_less_than(
        &self,
        x_shares: &[Fr],
        y_shares: &[Fr],
        _pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let num_parties = x_shares.len();

        // Reconstruct x and y (simulation mode)
        let mut x = Fr::ZERO;
        let mut y = Fr::ZERO;
        for (xs, ys) in x_shares.iter().zip(y_shares) {
            x = Fr::add(&x, xs);
            y = Fr::add(&y, ys);
        }

        let x_f64 = x.to_f64();
        let y_f64 = y.to_f64();

        // Compute result
        let result = if x_f64 < y_f64 {
            Fr::from_f64(1.0)
        } else {
            Fr::ZERO
        };

        // Re-share
        let mut rng = ChaCha20Rng::from_entropy();
        let mut shares = Vec::with_capacity(num_parties);
        let mut sum = Fr::ZERO;

        for _ in 0..num_parties - 1 {
            let r = Fr::random(&mut rng);
            shares.push(r.clone());
            sum = Fr::add(&sum, &r);
        }
        shares.push(Fr::sub(&result, &sum));

        Ok(shares)
    }
}

/// A garbled circuit for secure computation.
#[derive(Debug, Clone)]
pub struct GarbledCircuit {
    pub gates: Vec<GarbledGate>,
    pub input_labels_a: Vec<([u8; 16], [u8; 16])>,
    pub input_labels_b: Vec<([u8; 16], [u8; 16])>,
    pub output_labels: ([u8; 16], [u8; 16]),
}

/// A single garbled gate.
#[derive(Debug, Clone)]
pub struct GarbledGate {
    pub input_wires: (usize, usize),
    pub output_wire: usize,
    pub garbled_table: Vec<[u8; 16]>,
}

/// Bit decomposition for secure comparison.
///
/// Converts field elements to bit representation for bitwise comparison protocols.
pub struct BitDecomposition {
    /// Number of bits to decompose
    pub bit_length: usize,
}

impl BitDecomposition {
    pub fn new(bit_length: usize) -> Self {
        Self { bit_length }
    }

    /// Decomposes a value into bit shares (simulation).
    ///
    /// In production, this would use secure bit decomposition protocol.
    pub fn decompose(&self, x_shares: &[Fr], _pools: &mut [BeaverPool]) -> MPCResult<Vec<Vec<Fr>>> {
        let num_parties = x_shares.len();

        // Reconstruct x (simulation mode)
        let mut x = Fr::ZERO;
        for share in x_shares {
            x = Fr::add(&x, share);
        }

        // Convert to integer bits
        let x_int = x.to_f64() as i64;
        let bits: Vec<bool> = (0..self.bit_length)
            .map(|i| ((x_int >> i) & 1) == 1)
            .collect();

        // Share each bit
        let mut rng = ChaCha20Rng::from_entropy();
        let mut result = Vec::with_capacity(self.bit_length);

        for bit in bits {
            let bit_val = if bit { Fr::from_f64(1.0) } else { Fr::ZERO };
            let mut shares = Vec::with_capacity(num_parties);
            let mut sum = Fr::ZERO;

            for _ in 0..num_parties - 1 {
                let r = Fr::random(&mut rng);
                shares.push(r.clone());
                sum = Fr::add(&sum, &r);
            }
            shares.push(Fr::sub(&bit_val, &sum));
            result.push(shares);
        }

        Ok(result)
    }

    /// Recomposes bits back to a value.
    pub fn recompose(&self, bit_shares: &[Vec<Fr>]) -> Vec<Fr> {
        let num_parties = bit_shares[0].len();
        let mut result = vec![Fr::ZERO; num_parties];

        for (i, bit_sh) in bit_shares.iter().enumerate() {
            let scale = Fr::from_f64((1u64 << i) as f64);
            for (j, bit) in bit_sh.iter().enumerate() {
                // Use fixed_mul for proper fixed-point arithmetic
                result[j] = Fr::add(&result[j], &bit.fixed_mul(&scale));
            }
        }

        result
    }
}

/// Secure ReLU with gradient computation for backpropagation.
///
/// Computes both forward pass (ReLU) and maintains information for backward pass.
pub struct SecureReLUWithGradient {
    /// Inner comparison module
    comparison: SecureComparison,
}

impl SecureReLUWithGradient {
    pub fn new(config: ComparisonConfig) -> Self {
        Self {
            comparison: SecureComparison::new(config),
        }
    }

    /// Computes ReLU forward pass, returning both output and mask for gradient.
    ///
    /// Returns (relu_output, relu_mask) where mask is 1 where input >= 0.
    pub fn forward(
        &self,
        x_shares: &[Fr],
        pools: &mut [BeaverPool],
    ) -> MPCResult<(Vec<Fr>, Vec<Fr>)> {
        // Compute sign mask (1 if x >= 0, 0 otherwise)
        let mask_shares = self.comparison.sign_bit(x_shares, pools)?;

        // Compute ReLU output
        let relu_shares = self.comparison.relu(x_shares, pools)?;

        Ok((relu_shares, mask_shares))
    }

    /// Computes backward pass for ReLU gradient.
    ///
    /// ReLU gradient: d_output * mask
    pub fn backward(
        &self,
        grad_output: &[Fr],
        mask_shares: &[Fr],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        // Gradient = grad_output * mask
        let triples: Vec<BeaverTriple> = pools
            .iter_mut()
            .map(|p| p.take_scalar())
            .collect::<MPCResult<Vec<_>>>()?;

        let grad_input = SecureArithmetic::simulate_multiply(grad_output, mask_shares, &triples);

        Ok(grad_input)
    }

    /// Computes vectorized ReLU with gradients.
    pub fn forward_vector(
        &self,
        x_shares: &[Vec<Fr>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<(Vec<Vec<Fr>>, Vec<Vec<Fr>>)> {
        let num_parties = x_shares.len();
        let dim = x_shares[0].len();

        let mut relu_result = vec![vec![Fr::ZERO; dim]; num_parties];
        let mut mask_result = vec![vec![Fr::ZERO; dim]; num_parties];

        for d in 0..dim {
            let elem_shares: Vec<Fr> = x_shares.iter().map(|s| s[d].clone()).collect();
            let (relu_sh, mask_sh) = self.forward(&elem_shares, pools)?;

            for i in 0..num_parties {
                relu_result[i][d] = relu_sh[i].clone();
                mask_result[i][d] = mask_sh[i].clone();
            }
        }

        Ok((relu_result, mask_result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::dealer::TrustedDealer;

    fn create_pools(dealer: &mut TrustedDealer, num_parties: usize, count: usize) -> Vec<BeaverPool> {
        let per_party = dealer.generate_scalar_triples(count, num_parties);
        (0..num_parties)
            .map(|i| {
                let mut pool = BeaverPool::new(i, num_parties, 64);
                pool.fill_scalar(per_party[i].clone());
                pool
            })
            .collect()
    }

    fn split_value(value: f64, n: usize, seed: u64) -> Vec<Fr> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut shares = Vec::with_capacity(n);
        let mut sum = Fr::ZERO;

        for _ in 0..n - 1 {
            let r = Fr::from_f64(rng.gen_range(-100.0..100.0));
            shares.push(r.clone());
            sum = Fr::add(&sum, &r);
        }
        shares.push(Fr::sub(&Fr::from_f64(value), &sum));
        shares
    }

    fn reconstruct(shares: &[Fr]) -> f64 {
        let mut sum = Fr::ZERO;
        for s in shares {
            sum = Fr::add(&sum, s);
        }
        sum.to_f64()
    }

    #[test]
    fn test_sign_bit_positive() {
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 10);

        let cmp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(5.0, 3, 42);
        let sign_shares = cmp.sign_bit(&x_shares, &mut pools).unwrap();
        let sign = reconstruct(&sign_shares);

        assert!((sign - 1.0).abs() < 0.01, "Expected 1.0, got {}", sign);
    }

    #[test]
    fn test_sign_bit_negative() {
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 10);

        let cmp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(-5.0, 3, 42);
        let sign_shares = cmp.sign_bit(&x_shares, &mut pools).unwrap();
        let sign = reconstruct(&sign_shares);

        assert!((sign - 0.0).abs() < 0.01, "Expected 0.0, got {}", sign);
    }

    #[test]
    fn test_less_than() {
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 20);

        let cmp = SecureComparison::new(ComparisonConfig::default());

        // Test 3 < 5 (should be 1)
        let x_shares = split_value(3.0, 3, 42);
        let y_shares = split_value(5.0, 3, 99);
        let lt_shares = cmp.less_than(&x_shares, &y_shares, &mut pools).unwrap();
        let lt = reconstruct(&lt_shares);
        assert!((lt - 1.0).abs() < 0.01, "Expected 1.0, got {}", lt);
    }

    #[test]
    fn test_relu_positive() {
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 20);

        let cmp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(5.0, 3, 42);
        let relu_shares = cmp.relu(&x_shares, &mut pools).unwrap();
        let relu = reconstruct(&relu_shares);

        assert!((relu - 5.0).abs() < 0.1, "Expected 5.0, got {}", relu);
    }

    #[test]
    fn test_relu_negative() {
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 20);

        let cmp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(-5.0, 3, 42);
        let relu_shares = cmp.relu(&x_shares, &mut pools).unwrap();
        let relu = reconstruct(&relu_shares);

        assert!(relu.abs() < 0.1, "Expected 0.0, got {}", relu);
    }
}
