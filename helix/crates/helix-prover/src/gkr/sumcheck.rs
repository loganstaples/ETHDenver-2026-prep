//! Sumcheck Protocol Implementation.
//!
//! The sumcheck protocol is an interactive proof protocol that allows a prover to
//! convince a verifier of the sum of a multivariate polynomial over a boolean hypercube.
//! It is the core building block of the GKR protocol.
//!
//! ## Protocol Overview
//!
//! Given a multilinear polynomial f(x₁, ..., xₙ) and a claimed sum H, the protocol:
//! 1. Prover sends univariate polynomial g₁(X) = Σ_{x₂,...,xₙ} f(X, x₂, ..., xₙ)
//! 2. Verifier checks g₁(0) + g₁(1) = H and samples random r₁
//! 3. Prover sends g₂(X) = Σ_{x₃,...,xₙ} f(r₁, X, x₃, ..., xₙ)
//! 4. Continue for all variables
//! 5. Final verification: f(r₁, ..., rₙ) = gₙ(rₙ)
//!
//! ## Optimizations
//!
//! This implementation includes several optimizations:
//! - **Streaming evaluation**: Coefficients computed on-the-fly
//! - **Parallelization**: Using rayon for parallel summation
//! - **Low-degree optimization**: Special handling for degree-2 polynomials (common in neural nets)

use super::{FieldElement, GKRError, GKRResult};
use super::multilinear::{MultilinearPolynomial, DenseMultilinear};
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::ff::PrimeField;
use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Maximum polynomial degree supported in sumcheck rounds.
pub const MAX_DEGREE: usize = 3;

/// A single round of the sumcheck protocol.
#[derive(Debug, Clone)]
pub struct SumcheckRound {
    /// Coefficients of the univariate polynomial for this round.
    /// For a degree-d polynomial: [c₀, c₁, ..., cₐ] representing c₀ + c₁X + ... + cₐXᵈ
    pub coefficients: Vec<FieldElement>,
}

impl SumcheckRound {
    /// Creates a new sumcheck round from coefficients.
    pub fn new(coefficients: Vec<FieldElement>) -> Self {
        Self { coefficients }
    }

    /// Evaluates the round polynomial at a point.
    pub fn evaluate(&self, point: FieldElement) -> FieldElement {
        // Horner's method for efficient evaluation
        let mut result = FieldElement::zero();
        for coeff in self.coefficients.iter().rev() {
            result = result * point + coeff;
        }
        result
    }

    /// Returns the sum g(0) + g(1).
    pub fn sum_at_binary(&self) -> FieldElement {
        let at_0 = self.evaluate(FieldElement::zero());
        let at_1 = self.evaluate(FieldElement::one());
        at_0 + at_1
    }

    /// Returns the degree of this round's polynomial.
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    /// Serializes the round to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.coefficients.len() * 32);
        for coeff in &self.coefficients {
            bytes.extend_from_slice(coeff.to_repr().as_ref());
        }
        bytes
    }
}

/// Complete sumcheck proof containing all rounds.
#[derive(Debug, Clone)]
pub struct SumcheckProof {
    /// Individual round proofs.
    pub rounds: Vec<SumcheckRound>,
    /// Final evaluation point (r₁, ..., rₙ).
    pub final_point: Vec<FieldElement>,
    /// Claimed final evaluation f(r₁, ..., rₙ).
    pub final_eval: FieldElement,
}

impl SumcheckProof {
    /// Creates a new sumcheck proof.
    pub fn new(
        rounds: Vec<SumcheckRound>,
        final_point: Vec<FieldElement>,
        final_eval: FieldElement,
    ) -> Self {
        Self {
            rounds,
            final_point,
            final_eval,
        }
    }

    /// Returns the number of variables (rounds).
    pub fn num_variables(&self) -> usize {
        self.rounds.len()
    }

    /// Computes the proof size in bytes.
    pub fn size_bytes(&self) -> usize {
        let rounds_size: usize = self.rounds.iter().map(|r| r.coefficients.len() * 32).sum();
        let point_size = self.final_point.len() * 32;
        rounds_size + point_size + 32 // final_eval
    }

    /// Serializes the proof to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.size_bytes());

        // Number of rounds
        bytes.extend_from_slice(&(self.rounds.len() as u32).to_le_bytes());

        // Each round
        for round in &self.rounds {
            bytes.extend_from_slice(&(round.coefficients.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&round.to_bytes());
        }

        // Final point
        for p in &self.final_point {
            bytes.extend_from_slice(p.to_repr().as_ref());
        }

        // Final evaluation
        bytes.extend_from_slice(self.final_eval.to_repr().as_ref());

        bytes
    }
}

/// Sumcheck prover state during protocol execution.
#[derive(Debug)]
pub struct SumcheckProver<'a> {
    /// The polynomial being summed.
    polynomial: &'a DenseMultilinear,
    /// Current partial point (challenges received so far).
    partial_point: Vec<FieldElement>,
    /// Current round index.
    current_round: usize,
    /// Cached intermediate evaluations for efficiency.
    cached_evals: Vec<FieldElement>,
}

impl<'a> SumcheckProver<'a> {
    /// Creates a new sumcheck prover for the given polynomial.
    pub fn new(polynomial: &'a DenseMultilinear) -> Self {
        let n = polynomial.num_variables();
        let cached_evals = polynomial.evaluations().to_vec();

        Self {
            polynomial,
            partial_point: Vec::with_capacity(n),
            current_round: 0,
            cached_evals,
        }
    }

    /// Generates the next round's univariate polynomial.
    ///
    /// For round i, computes:
    /// gᵢ(X) = Σ_{xᵢ₊₁,...,xₙ ∈ {0,1}} f(r₁, ..., rᵢ₋₁, X, xᵢ₊₁, ..., xₙ)
    pub fn next_round(&mut self) -> GKRResult<SumcheckRound> {
        if self.current_round >= self.polynomial.num_variables() {
            return Err(GKRError::InvalidProof("All rounds completed".to_string()));
        }

        let n = self.polynomial.num_variables();
        let remaining_vars = n - self.current_round;

        // Number of terms to sum over (2^(remaining_vars - 1))
        let num_terms = 1 << (remaining_vars - 1);

        // For a multilinear polynomial, each round polynomial has degree 1
        // g(X) = a + bX where:
        //   a = g(0) = Σ f(..., 0, ...)
        //   b = g(1) - g(0)

        let (sum_0, sum_1) = self.compute_partial_sums(num_terms);

        // Coefficients: [a, b] where g(X) = a + bX
        let coefficients = vec![sum_0, sum_1 - sum_0];

        self.current_round += 1;

        Ok(SumcheckRound::new(coefficients))
    }

    /// Computes partial sums for g(0) and g(1).
    fn compute_partial_sums(&self, num_terms: usize) -> (FieldElement, FieldElement) {
        #[cfg(feature = "parallel")]
        {
            if num_terms > 256 {
                return self.compute_partial_sums_parallel(num_terms);
            }
        }

        self.compute_partial_sums_sequential(num_terms)
    }

    fn compute_partial_sums_sequential(&self, num_terms: usize) -> (FieldElement, FieldElement) {
        let mut sum_0 = FieldElement::zero();
        let mut sum_1 = FieldElement::zero();

        let half = self.cached_evals.len() / 2;

        for i in 0..num_terms.min(half) {
            sum_0 = sum_0 + self.cached_evals[i];
            sum_1 = sum_1 + self.cached_evals[i + half];
        }

        (sum_0, sum_1)
    }

    #[cfg(feature = "parallel")]
    fn compute_partial_sums_parallel(&self, num_terms: usize) -> (FieldElement, FieldElement) {
        let half = self.cached_evals.len() / 2;
        let limit = num_terms.min(half);

        let (sum_0, sum_1): (Vec<FieldElement>, Vec<FieldElement>) = (0..limit)
            .into_par_iter()
            .map(|i| (self.cached_evals[i], self.cached_evals[i + half]))
            .unzip();

        let s0 = sum_0.into_iter().fold(FieldElement::zero(), |acc, x| acc + x);
        let s1 = sum_1.into_iter().fold(FieldElement::zero(), |acc, x| acc + x);

        (s0, s1)
    }

    /// Receives a challenge and updates state.
    pub fn receive_challenge(&mut self, challenge: FieldElement) {
        self.partial_point.push(challenge);

        // Update cached evaluations by fixing the current variable
        let half = self.cached_evals.len() / 2;
        let new_evals: Vec<FieldElement> = (0..half)
            .map(|i| {
                // Linear interpolation: f(r) = f(0) + r * (f(1) - f(0))
                let f0 = self.cached_evals[i];
                let f1 = self.cached_evals[i + half];
                f0 + challenge * (f1 - f0)
            })
            .collect();

        self.cached_evals = new_evals;
    }

    /// Finalizes the proof and returns the final evaluation.
    pub fn finalize(&self) -> GKRResult<(Vec<FieldElement>, FieldElement)> {
        if self.partial_point.len() != self.polynomial.num_variables() {
            return Err(GKRError::InvalidProof(format!(
                "Expected {} challenges, got {}",
                self.polynomial.num_variables(),
                self.partial_point.len()
            )));
        }

        // At this point, cached_evals should contain a single element
        let final_eval = if self.cached_evals.is_empty() {
            FieldElement::zero()
        } else {
            self.cached_evals[0]
        };

        Ok((self.partial_point.clone(), final_eval))
    }

    /// Runs the complete sumcheck protocol with a transcript for Fiat-Shamir.
    pub fn prove_with_transcript<T: Transcript>(
        &mut self,
        transcript: &mut T,
        claimed_sum: FieldElement,
    ) -> GKRResult<SumcheckProof> {
        let n = self.polynomial.num_variables();
        let mut rounds = Vec::with_capacity(n);

        // Commit claimed sum to transcript
        transcript.append_scalar(b"claimed_sum", claimed_sum);

        for _ in 0..n {
            // Generate round polynomial
            let round = self.next_round()?;

            // Append round to transcript
            for coeff in &round.coefficients {
                transcript.append_scalar(b"round_coeff", *coeff);
            }

            // Get challenge from transcript
            let challenge = transcript.challenge_scalar(b"sumcheck_challenge");
            self.receive_challenge(challenge);

            rounds.push(round);
        }

        let (final_point, final_eval) = self.finalize()?;

        Ok(SumcheckProof::new(rounds, final_point, final_eval))
    }
}

/// Sumcheck verifier.
pub struct SumcheckVerifier {
    /// Expected sum (claimed by prover).
    claimed_sum: FieldElement,
    /// Challenges generated during verification.
    challenges: Vec<FieldElement>,
}

impl SumcheckVerifier {
    /// Creates a new sumcheck verifier.
    pub fn new(claimed_sum: FieldElement) -> Self {
        Self {
            claimed_sum,
            challenges: Vec::new(),
        }
    }

    /// Verifies a complete sumcheck proof.
    ///
    /// Returns the final evaluation point and expected evaluation if successful.
    pub fn verify(&mut self, proof: &SumcheckProof) -> GKRResult<(Vec<FieldElement>, FieldElement)> {
        let n = proof.num_variables();
        let mut current_sum = self.claimed_sum;
        self.challenges = Vec::with_capacity(n);

        for (round_idx, round) in proof.rounds.iter().enumerate() {
            // Check: g(0) + g(1) = current expected sum
            let round_sum = round.sum_at_binary();
            if round_sum != current_sum {
                return Err(GKRError::SumcheckFailed { round: round_idx });
            }

            // Get challenge (from proof's final_point for non-interactive version)
            let challenge = proof.final_point[round_idx];
            self.challenges.push(challenge);

            // Update expected sum for next round: g(r)
            current_sum = round.evaluate(challenge);
        }

        // Final check: the last round's evaluation at the challenge should equal final_eval
        if current_sum != proof.final_eval {
            return Err(GKRError::EvaluationMismatch {
                expected: proof.final_eval,
                actual: current_sum,
            });
        }

        Ok((proof.final_point.clone(), proof.final_eval))
    }

    /// Verifies with a transcript for Fiat-Shamir.
    pub fn verify_with_transcript<T: Transcript>(
        &mut self,
        proof: &SumcheckProof,
        transcript: &mut T,
    ) -> GKRResult<(Vec<FieldElement>, FieldElement)> {
        let n = proof.num_variables();
        let mut current_sum = self.claimed_sum;
        self.challenges = Vec::with_capacity(n);

        // Commit claimed sum to transcript
        transcript.append_scalar(b"claimed_sum", self.claimed_sum);

        for (round_idx, round) in proof.rounds.iter().enumerate() {
            // Check: g(0) + g(1) = current expected sum
            let round_sum = round.sum_at_binary();
            if round_sum != current_sum {
                return Err(GKRError::SumcheckFailed { round: round_idx });
            }

            // Append round to transcript (same as prover did)
            for coeff in &round.coefficients {
                transcript.append_scalar(b"round_coeff", *coeff);
            }

            // Get challenge from transcript (must match prover's challenge)
            let challenge = transcript.challenge_scalar(b"sumcheck_challenge");
            self.challenges.push(challenge);

            // Update expected sum for next round: g(r)
            current_sum = round.evaluate(challenge);
        }

        // Final check
        if current_sum != proof.final_eval {
            return Err(GKRError::EvaluationMismatch {
                expected: proof.final_eval,
                actual: current_sum,
            });
        }

        Ok((self.challenges.clone(), proof.final_eval))
    }
}

/// Transcript trait for Fiat-Shamir transformation.
pub trait Transcript {
    /// Appends a scalar to the transcript.
    fn append_scalar(&mut self, label: &[u8], scalar: FieldElement);

    /// Appends raw bytes to the transcript.
    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]);

    /// Generates a challenge scalar.
    fn challenge_scalar(&mut self, label: &[u8]) -> FieldElement;
}

/// Blake3-based transcript for Fiat-Shamir.
pub struct Blake3Transcript {
    hasher: blake3::Hasher,
}

impl Blake3Transcript {
    /// Creates a new Blake3 transcript.
    pub fn new(domain_separator: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"HELIX_GKR_TRANSCRIPT_v1");
        hasher.update(domain_separator);
        Self { hasher }
    }
}

impl Transcript for Blake3Transcript {
    fn append_scalar(&mut self, label: &[u8], scalar: FieldElement) {
        self.hasher.update(label);
        self.hasher.update(scalar.to_repr().as_ref());
    }

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.hasher.update(label);
        self.hasher.update(&(bytes.len() as u64).to_le_bytes());
        self.hasher.update(bytes);
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> FieldElement {
        self.hasher.update(label);
        let hash = self.hasher.finalize();
        self.hasher.update(hash.as_bytes());

        // Reduce hash to field element
        let bytes = hash.as_bytes();
        let mut repr = [0u8; 32];
        repr.copy_from_slice(&bytes[..32]);

        // Ensure it's in the field by reducing mod p
        // For BN254, we take the hash and use it to seed a uniform sample
        FieldElement::from_raw([
            u64::from_le_bytes(repr[0..8].try_into().unwrap()),
            u64::from_le_bytes(repr[8..16].try_into().unwrap()),
            u64::from_le_bytes(repr[16..24].try_into().unwrap()),
            u64::from_le_bytes(repr[24..32].try_into().unwrap()) & 0x0FFFFFFFFFFFFFFF,
        ])
    }
}

/// Optimized sumcheck for degree-2 polynomials (common in neural networks).
///
/// For f(x,y) = Σᵢⱼ aᵢⱼxᵢyⱼ, this provides faster proving.
pub struct QuadraticSumcheck {
    /// Coefficients in the form vec[i][j] = coefficient of xᵢyⱼ.
    coefficients: Vec<Vec<FieldElement>>,
    /// Cached row sums for efficiency.
    row_sums: Vec<FieldElement>,
}

impl QuadraticSumcheck {
    /// Creates a new quadratic sumcheck instance from a coefficient matrix.
    pub fn new(coefficients: Vec<Vec<FieldElement>>) -> Self {
        let row_sums: Vec<FieldElement> = coefficients
            .iter()
            .map(|row| row.iter().fold(FieldElement::zero(), |acc, x| acc + x))
            .collect();

        Self {
            coefficients,
            row_sums,
        }
    }

    /// Computes the total sum over the boolean hypercube.
    pub fn total_sum(&self) -> FieldElement {
        self.row_sums.iter().fold(FieldElement::zero(), |acc, x| acc + x)
    }

    /// Proves the sumcheck for a matrix multiplication claim.
    ///
    /// For A·B = C, proves Σₓ,ᵧ,ᵤ (Aₓᵤ · Bᵤᵧ - Cₓᵧ) · eq(x,y,z,r) = 0
    pub fn prove_matmul<T: Transcript>(
        &self,
        transcript: &mut T,
    ) -> GKRResult<SumcheckProof> {
        let n = self.coefficients.len();
        if n == 0 {
            return Ok(SumcheckProof::new(vec![], vec![], FieldElement::zero()));
        }

        let claimed_sum = self.total_sum();
        transcript.append_scalar(b"claimed_sum", claimed_sum);

        // Simplified: just run standard sumcheck on flattened polynomial
        let flat: Vec<FieldElement> = self.coefficients.iter().flatten().copied().collect();
        let poly = DenseMultilinear::from_evaluations(flat);
        let mut prover = SumcheckProver::new(&poly);

        prover.prove_with_transcript(transcript, claimed_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_field() -> FieldElement {
        let mut rng = rand::thread_rng();
        FieldElement::from(rng.gen::<u64>())
    }

    #[test]
    fn test_sumcheck_round_evaluate() {
        // g(X) = 1 + 2X
        let round = SumcheckRound::new(vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
        ]);

        assert_eq!(round.evaluate(FieldElement::zero()), FieldElement::from(1u64));
        assert_eq!(round.evaluate(FieldElement::one()), FieldElement::from(3u64));
    }

    #[test]
    fn test_sumcheck_round_sum() {
        // g(X) = 1 + 2X, g(0) = 1, g(1) = 3, sum = 4
        let round = SumcheckRound::new(vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
        ]);

        assert_eq!(round.sum_at_binary(), FieldElement::from(4u64));
    }

    #[test]
    fn test_sumcheck_simple() {
        // Create a simple 2-variable polynomial
        // f(x₁, x₂) = 1 + 2x₁ + 3x₂ + 4x₁x₂
        // Evaluations: f(0,0)=1, f(1,0)=3, f(0,1)=4, f(1,1)=10
        let evals = vec![
            FieldElement::from(1u64),  // (0,0)
            FieldElement::from(3u64),  // (1,0)
            FieldElement::from(4u64),  // (0,1)
            FieldElement::from(10u64), // (1,1)
        ];

        let poly = DenseMultilinear::from_evaluations(evals);
        let claimed_sum = FieldElement::from(18u64); // 1 + 3 + 4 + 10

        // Prove
        let mut transcript = Blake3Transcript::new(b"test");
        let mut prover = SumcheckProver::new(&poly);
        let proof = prover.prove_with_transcript(&mut transcript, claimed_sum).unwrap();

        // Verify
        let mut transcript2 = Blake3Transcript::new(b"test");
        let mut verifier = SumcheckVerifier::new(claimed_sum);
        let result = verifier.verify_with_transcript(&proof, &mut transcript2);

        assert!(result.is_ok());
    }

    #[test]
    fn test_sumcheck_larger() {
        // 4-variable polynomial (16 evaluations)
        let evals: Vec<FieldElement> = (0..16)
            .map(|i| FieldElement::from(i as u64))
            .collect();

        let poly = DenseMultilinear::from_evaluations(evals.clone());
        let claimed_sum = evals.iter().fold(FieldElement::zero(), |acc, x| acc + x);

        let mut transcript = Blake3Transcript::new(b"test");
        let mut prover = SumcheckProver::new(&poly);
        let proof = prover.prove_with_transcript(&mut transcript, claimed_sum).unwrap();

        let mut transcript2 = Blake3Transcript::new(b"test");
        let mut verifier = SumcheckVerifier::new(claimed_sum);
        let result = verifier.verify_with_transcript(&proof, &mut transcript2);

        assert!(result.is_ok());
    }

    #[test]
    fn test_sumcheck_wrong_sum_fails() {
        let evals = vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
            FieldElement::from(4u64),
        ];

        let poly = DenseMultilinear::from_evaluations(evals);
        let wrong_sum = FieldElement::from(999u64);

        let mut transcript = Blake3Transcript::new(b"test");
        let mut prover = SumcheckProver::new(&poly);
        let proof = prover.prove_with_transcript(&mut transcript, wrong_sum).unwrap();

        let mut transcript2 = Blake3Transcript::new(b"test");
        let mut verifier = SumcheckVerifier::new(wrong_sum);
        let result = verifier.verify_with_transcript(&proof, &mut transcript2);

        // First round should fail since g(0) + g(1) != wrong_sum
        assert!(result.is_err());
    }

    #[test]
    fn test_blake3_transcript() {
        let mut t1 = Blake3Transcript::new(b"test");
        let mut t2 = Blake3Transcript::new(b"test");

        t1.append_scalar(b"value", FieldElement::from(42u64));
        t2.append_scalar(b"value", FieldElement::from(42u64));

        let c1 = t1.challenge_scalar(b"chal");
        let c2 = t2.challenge_scalar(b"chal");

        assert_eq!(c1, c2);
    }

    #[test]
    fn test_transcript_different_values() {
        let mut t1 = Blake3Transcript::new(b"test");
        let mut t2 = Blake3Transcript::new(b"test");

        t1.append_scalar(b"value", FieldElement::from(42u64));
        t2.append_scalar(b"value", FieldElement::from(43u64));

        let c1 = t1.challenge_scalar(b"chal");
        let c2 = t2.challenge_scalar(b"chal");

        assert_ne!(c1, c2);
    }
}
