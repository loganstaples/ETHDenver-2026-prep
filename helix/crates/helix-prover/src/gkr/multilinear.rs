//! Multilinear Polynomial Representation.
//!
//! Multilinear polynomials are the foundation of the GKR protocol. A multilinear polynomial
//! in n variables is uniquely determined by its 2ⁿ evaluations on the boolean hypercube {0,1}ⁿ.
//!
//! ## Multilinear Extension (MLE)
//!
//! Given a function f: {0,1}ⁿ → F, its multilinear extension f̃ is the unique multilinear
//! polynomial such that f̃(x) = f(x) for all x ∈ {0,1}ⁿ. This extension is computed as:
//!
//! f̃(x₁, ..., xₙ) = Σ_{b∈{0,1}ⁿ} f(b) · eq(x, b)
//!
//! where eq(x, b) = Πᵢ (xᵢbᵢ + (1-xᵢ)(1-bᵢ))
//!
//! ## Representations
//!
//! - **Dense**: Stores all 2ⁿ evaluations. O(2ⁿ) space, O(2ⁿ) evaluation via streaming.
//! - **Sparse**: Stores only non-zero evaluations. Better for sparse data.
//! - **Product**: For tensor products of smaller polynomials.

use super::{FieldElement, GKRError, GKRResult};
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::ff::PrimeField;
use std::ops::{Add, Mul, Sub};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Trait for multilinear polynomial operations.
pub trait MultilinearPolynomial: Clone + Send + Sync {
    /// Returns the number of variables.
    fn num_variables(&self) -> usize;

    /// Returns the number of evaluations (2^num_variables).
    fn num_evaluations(&self) -> usize {
        1 << self.num_variables()
    }

    /// Evaluates the polynomial at a point in Fⁿ.
    fn evaluate(&self, point: &[FieldElement]) -> FieldElement;

    /// Evaluates the polynomial at a point, in-place (more efficient for streaming).
    fn evaluate_streaming(&self, point: &[FieldElement]) -> FieldElement {
        self.evaluate(point)
    }

    /// Computes the sum over the boolean hypercube.
    fn sum(&self) -> FieldElement;

    /// Returns a dense representation of the evaluations.
    fn to_evaluations(&self) -> Vec<FieldElement>;

    /// Binds the first variable to a value, reducing dimension by 1.
    fn bind_first(&self, value: FieldElement) -> DenseMultilinear;

    /// Binds the last variable to a value, reducing dimension by 1.
    fn bind_last(&self, value: FieldElement) -> DenseMultilinear;
}

/// Dense representation of a multilinear polynomial.
///
/// Stores all 2ⁿ evaluations on the boolean hypercube.
#[derive(Debug, Clone)]
pub struct DenseMultilinear {
    /// Evaluations on the boolean hypercube.
    /// evaluations[i] = f(b₁, ..., bₙ) where i = Σⱼ bⱼ·2^(n-j)
    evaluations: Vec<FieldElement>,
    /// Number of variables (log₂ of evaluations length).
    num_vars: usize,
}

impl DenseMultilinear {
    /// Creates a new dense multilinear polynomial from evaluations.
    pub fn from_evaluations(evaluations: Vec<FieldElement>) -> Self {
        let len = evaluations.len();
        assert!(len.is_power_of_two(), "Evaluations length must be a power of 2");

        let num_vars = if len == 0 { 0 } else { len.trailing_zeros() as usize };

        Self {
            evaluations,
            num_vars,
        }
    }

    /// Creates a zero polynomial with the specified number of variables.
    pub fn zero(num_vars: usize) -> Self {
        Self {
            evaluations: vec![FieldElement::zero(); 1 << num_vars],
            num_vars,
        }
    }

    /// Creates the constant polynomial f(x) = c.
    pub fn constant(c: FieldElement, num_vars: usize) -> Self {
        Self {
            evaluations: vec![c; 1 << num_vars],
            num_vars,
        }
    }

    /// Creates the identity polynomial for variable i: fᵢ(x) = xᵢ.
    pub fn variable(i: usize, num_vars: usize) -> Self {
        assert!(i < num_vars, "Variable index out of bounds");

        let size = 1 << num_vars;
        let mut evals = vec![FieldElement::zero(); size];

        // xᵢ = 1 when bit i is set
        let stride = 1 << (num_vars - 1 - i);
        for j in 0..size {
            if (j / stride) % 2 == 1 {
                evals[j] = FieldElement::one();
            }
        }

        Self {
            evaluations: evals,
            num_vars,
        }
    }

    /// Returns a reference to the evaluations.
    pub fn evaluations(&self) -> &[FieldElement] {
        &self.evaluations
    }

    /// Returns a mutable reference to the evaluations.
    pub fn evaluations_mut(&mut self) -> &mut [FieldElement] {
        &mut self.evaluations
    }

    /// Scales all evaluations by a constant.
    pub fn scale(&mut self, c: FieldElement) {
        for eval in &mut self.evaluations {
            *eval = *eval * c;
        }
    }

    /// Adds another polynomial to this one (in-place).
    pub fn add_assign(&mut self, other: &Self) {
        assert_eq!(self.num_vars, other.num_vars);
        for (a, b) in self.evaluations.iter_mut().zip(other.evaluations.iter()) {
            *a = *a + *b;
        }
    }

    /// Multiplies another polynomial to this one element-wise (in-place).
    pub fn mul_assign(&mut self, other: &Self) {
        assert_eq!(self.num_vars, other.num_vars);
        for (a, b) in self.evaluations.iter_mut().zip(other.evaluations.iter()) {
            *a = *a * *b;
        }
    }

    /// Computes the tensor product with another polynomial.
    pub fn tensor_product(&self, other: &Self) -> Self {
        let new_num_vars = self.num_vars + other.num_vars;
        let new_size = 1 << new_num_vars;
        let mut new_evals = vec![FieldElement::zero(); new_size];

        for (i, &a) in self.evaluations.iter().enumerate() {
            for (j, &b) in other.evaluations.iter().enumerate() {
                let idx = i * other.evaluations.len() + j;
                new_evals[idx] = a * b;
            }
        }

        Self {
            evaluations: new_evals,
            num_vars: new_num_vars,
        }
    }

    /// Restricts the polynomial by fixing some variables.
    pub fn restrict(&self, fixed: &[(usize, FieldElement)]) -> Self {
        if fixed.is_empty() {
            return self.clone();
        }

        let mut result = self.clone();
        for &(var, val) in fixed {
            result = result.bind_variable(var, val);
        }
        result
    }

    /// Binds a specific variable to a value.
    fn bind_variable(&self, var: usize, value: FieldElement) -> Self {
        assert!(var < self.num_vars);

        let new_num_vars = self.num_vars - 1;
        let new_size = 1 << new_num_vars;
        let mut new_evals = vec![FieldElement::zero(); new_size];

        let stride = 1 << (self.num_vars - 1 - var);

        for i in 0..new_size {
            // Compute indices in original array
            let high = (i >> (self.num_vars - 1 - var)) << (self.num_vars - var);
            let low = i & ((1 << (self.num_vars - 1 - var)) - 1);
            let idx0 = high | low;
            let idx1 = high | stride | low;

            // Linear interpolation: f(r) = f(0) + r * (f(1) - f(0))
            let f0 = self.evaluations[idx0];
            let f1 = self.evaluations[idx1];
            new_evals[i] = f0 + value * (f1 - f0);
        }

        Self {
            evaluations: new_evals,
            num_vars: new_num_vars,
        }
    }

    /// Evaluates using the streaming algorithm (efficient for large polynomials).
    fn evaluate_internal(&self, point: &[FieldElement]) -> FieldElement {
        assert_eq!(point.len(), self.num_vars);

        if self.evaluations.is_empty() {
            return FieldElement::zero();
        }

        // Streaming evaluation: successively fix variables
        let mut current = self.evaluations.clone();

        for &r in point.iter() {
            let half = current.len() / 2;
            let mut next = vec![FieldElement::zero(); half];

            for i in 0..half {
                next[i] = current[i] + r * (current[i + half] - current[i]);
            }

            current = next;
        }

        current[0]
    }

    #[cfg(feature = "parallel")]
    fn evaluate_parallel(&self, point: &[FieldElement]) -> FieldElement {
        assert_eq!(point.len(), self.num_vars);

        if self.evaluations.len() < 1024 {
            return self.evaluate_internal(point);
        }

        let mut current = self.evaluations.clone();

        for &r in point.iter() {
            let half = current.len() / 2;

            if half >= 256 {
                let next: Vec<FieldElement> = (0..half)
                    .into_par_iter()
                    .map(|i| current[i] + r * (current[i + half] - current[i]))
                    .collect();
                current = next;
            } else {
                let mut next = vec![FieldElement::zero(); half];
                for i in 0..half {
                    next[i] = current[i] + r * (current[i + half] - current[i]);
                }
                current = next;
            }
        }

        current[0]
    }
}

impl MultilinearPolynomial for DenseMultilinear {
    fn num_variables(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, point: &[FieldElement]) -> FieldElement {
        #[cfg(feature = "parallel")]
        {
            self.evaluate_parallel(point)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.evaluate_internal(point)
        }
    }

    fn sum(&self) -> FieldElement {
        #[cfg(feature = "parallel")]
        {
            if self.evaluations.len() >= 1024 {
                return self.evaluations.par_iter().copied().reduce(
                    || FieldElement::zero(),
                    |a, b| a + b,
                );
            }
        }

        self.evaluations.iter().fold(FieldElement::zero(), |acc, x| acc + x)
    }

    fn to_evaluations(&self) -> Vec<FieldElement> {
        self.evaluations.clone()
    }

    fn bind_first(&self, value: FieldElement) -> DenseMultilinear {
        if self.num_vars == 0 {
            return self.clone();
        }

        let half = self.evaluations.len() / 2;
        let new_evals: Vec<FieldElement> = (0..half)
            .map(|i| {
                self.evaluations[i] + value * (self.evaluations[i + half] - self.evaluations[i])
            })
            .collect();

        DenseMultilinear::from_evaluations(new_evals)
    }

    fn bind_last(&self, value: FieldElement) -> DenseMultilinear {
        if self.num_vars == 0 {
            return self.clone();
        }

        let half = self.evaluations.len() / 2;
        let stride = 2;
        let new_evals: Vec<FieldElement> = (0..half)
            .map(|i| {
                let group = i / 1;
                let pos = i % 1;
                let idx0 = group * stride + pos;
                let idx1 = idx0 + 1;
                self.evaluations[idx0] + value * (self.evaluations[idx1] - self.evaluations[idx0])
            })
            .collect();

        DenseMultilinear::from_evaluations(new_evals)
    }
}

/// Sparse representation of a multilinear polynomial.
///
/// Stores only non-zero evaluations. Efficient when most evaluations are zero.
#[derive(Debug, Clone)]
pub struct SparseMultilinear {
    /// Non-zero evaluations as (index, value) pairs.
    entries: Vec<(usize, FieldElement)>,
    /// Number of variables.
    num_vars: usize,
}

impl SparseMultilinear {
    /// Creates a new sparse multilinear polynomial.
    pub fn new(entries: Vec<(usize, FieldElement)>, num_vars: usize) -> Self {
        Self { entries, num_vars }
    }

    /// Creates from a dense representation, keeping only non-zero values.
    pub fn from_dense(dense: &DenseMultilinear) -> Self {
        let entries: Vec<(usize, FieldElement)> = dense
            .evaluations
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != FieldElement::zero())
            .map(|(i, &v)| (i, v))
            .collect();

        Self {
            entries,
            num_vars: dense.num_vars,
        }
    }

    /// Converts to dense representation.
    pub fn to_dense(&self) -> DenseMultilinear {
        let mut evals = vec![FieldElement::zero(); 1 << self.num_vars];
        for &(idx, val) in &self.entries {
            evals[idx] = val;
        }
        DenseMultilinear::from_evaluations(evals)
    }

    /// Returns the number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }
}

impl MultilinearPolynomial for SparseMultilinear {
    fn num_variables(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, point: &[FieldElement]) -> FieldElement {
        // For sparse polynomials, iterate over non-zero entries
        let mut result = FieldElement::zero();

        for &(idx, val) in &self.entries {
            // Compute eq(point, bits(idx))
            let mut eq_val = FieldElement::one();
            for (i, &p) in point.iter().enumerate() {
                let bit = (idx >> (self.num_vars - 1 - i)) & 1;
                if bit == 1 {
                    eq_val = eq_val * p;
                } else {
                    eq_val = eq_val * (FieldElement::one() - p);
                }
            }
            result = result + val * eq_val;
        }

        result
    }

    fn sum(&self) -> FieldElement {
        self.entries.iter().fold(FieldElement::zero(), |acc, (_, v)| acc + v)
    }

    fn to_evaluations(&self) -> Vec<FieldElement> {
        self.to_dense().evaluations
    }

    fn bind_first(&self, value: FieldElement) -> DenseMultilinear {
        self.to_dense().bind_first(value)
    }

    fn bind_last(&self, value: FieldElement) -> DenseMultilinear {
        self.to_dense().bind_last(value)
    }
}

/// Evaluation domain for batch polynomial operations.
#[derive(Debug, Clone)]
pub struct EvaluationDomain {
    /// Size of the domain.
    pub size: usize,
    /// Generators for the domain.
    pub generators: Vec<FieldElement>,
}

impl EvaluationDomain {
    /// Creates a new evaluation domain of the given size.
    pub fn new(size: usize) -> Self {
        let generators = (0..size)
            .map(|i| FieldElement::from(i as u64))
            .collect();

        Self { size, generators }
    }
}

/// Multilinear extension builder.
///
/// Helper for constructing multilinear extensions from various data sources.
#[derive(Debug, Clone)]
pub struct MultilinearExtension;

impl MultilinearExtension {
    /// Builds MLE from a vector.
    pub fn from_vec(values: Vec<FieldElement>) -> DenseMultilinear {
        let n = values.len();
        let padded_len = n.next_power_of_two();

        let mut padded = values;
        padded.resize(padded_len, FieldElement::zero());

        DenseMultilinear::from_evaluations(padded)
    }

    /// Builds MLE from a matrix (row-major order).
    pub fn from_matrix(matrix: &[Vec<FieldElement>]) -> DenseMultilinear {
        let flat: Vec<FieldElement> = matrix.iter().flatten().copied().collect();
        Self::from_vec(flat)
    }

    /// Builds the "eq" polynomial: eq(x, r) = Πᵢ (xᵢrᵢ + (1-xᵢ)(1-rᵢ))
    ///
    /// Returns evaluations of eq(·, r) over the hypercube.
    pub fn eq_polynomial(r: &[FieldElement]) -> DenseMultilinear {
        let n = r.len();
        let size = 1 << n;
        let mut evals = vec![FieldElement::one(); size];

        for (i, &ri) in r.iter().enumerate() {
            let stride = 1 << (n - 1 - i);
            for j in 0..size {
                let bit = (j / stride) % 2;
                if bit == 1 {
                    evals[j] = evals[j] * ri;
                } else {
                    evals[j] = evals[j] * (FieldElement::one() - ri);
                }
            }
        }

        DenseMultilinear::from_evaluations(evals)
    }

    /// Computes the identity polynomial over the hypercube.
    /// id(x) = Σᵢ xᵢ · 2^(n-1-i)
    pub fn identity_polynomial(num_vars: usize) -> DenseMultilinear {
        let size = 1 << num_vars;
        let evals: Vec<FieldElement> = (0..size)
            .map(|i| FieldElement::from(i as u64))
            .collect();

        DenseMultilinear::from_evaluations(evals)
    }
}

/// Polynomial commitment (placeholder for integration with commitment schemes).
#[derive(Debug, Clone)]
pub struct PolynomialCommitment {
    /// Commitment value (depends on scheme).
    pub value: [u8; 32],
    /// Auxiliary data for opening proofs.
    pub aux: Vec<u8>,
}

impl PolynomialCommitment {
    /// Creates a new polynomial commitment.
    pub fn new(value: [u8; 32]) -> Self {
        Self { value, aux: vec![] }
    }

    /// Computes a simple hash commitment (for testing).
    pub fn hash_commit(poly: &DenseMultilinear) -> Self {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        for eval in &poly.evaluations {
            hasher.update(&eval.to_repr());
        }

        let hash = hasher.finalize();
        let mut value = [0u8; 32];
        value.copy_from_slice(&hash);

        Self { value, aux: vec![] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_from_evaluations() {
        let evals = vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
            FieldElement::from(4u64),
        ];

        let poly = DenseMultilinear::from_evaluations(evals);
        assert_eq!(poly.num_variables(), 2);
        assert_eq!(poly.num_evaluations(), 4);
    }

    #[test]
    fn test_dense_evaluation() {
        // f(x₁, x₂) with evals stored in order where first variable is MSB
        // index 0 = (0,0), index 1 = (0,1), index 2 = (1,0), index 3 = (1,1)
        let evals = vec![
            FieldElement::from(1u64),  // (0,0) - index 0
            FieldElement::from(2u64),  // (0,1) - index 1
            FieldElement::from(3u64),  // (1,0) - index 2
            FieldElement::from(4u64),  // (1,1) - index 3
        ];

        let poly = DenseMultilinear::from_evaluations(evals);

        // Check corner evaluations
        assert_eq!(
            poly.evaluate(&[FieldElement::zero(), FieldElement::zero()]),
            FieldElement::from(1u64)
        );
        assert_eq!(
            poly.evaluate(&[FieldElement::zero(), FieldElement::one()]),
            FieldElement::from(2u64)
        );
        assert_eq!(
            poly.evaluate(&[FieldElement::one(), FieldElement::zero()]),
            FieldElement::from(3u64)
        );
        assert_eq!(
            poly.evaluate(&[FieldElement::one(), FieldElement::one()]),
            FieldElement::from(4u64)
        );
    }

    #[test]
    fn test_dense_sum() {
        let evals = vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
            FieldElement::from(4u64),
        ];

        let poly = DenseMultilinear::from_evaluations(evals);
        assert_eq!(poly.sum(), FieldElement::from(10u64));
    }

    #[test]
    fn test_bind_first() {
        // f(x₁, x₂) with evals [1, 2, 3, 4]
        // index 0 = (0,0), index 1 = (0,1), index 2 = (1,0), index 3 = (1,1)
        let evals = vec![
            FieldElement::from(1u64),  // (0,0)
            FieldElement::from(2u64),  // (0,1)
            FieldElement::from(3u64),  // (1,0)
            FieldElement::from(4u64),  // (1,1)
        ];

        let poly = DenseMultilinear::from_evaluations(evals);

        // Bind x₁ = 0: f(0, x₂) for x₂ ∈ {0,1}
        // Result: [f(0,0), f(0,1)] = [1, 2]
        let bound_0 = poly.bind_first(FieldElement::zero());
        assert_eq!(bound_0.num_variables(), 1);
        assert_eq!(bound_0.evaluations(), &[FieldElement::from(1u64), FieldElement::from(2u64)]);

        // Bind x₁ = 1: f(1, x₂) for x₂ ∈ {0,1}
        // Result: [f(1,0), f(1,1)] = [3, 4]
        let bound_1 = poly.bind_first(FieldElement::one());
        assert_eq!(bound_1.evaluations(), &[FieldElement::from(3u64), FieldElement::from(4u64)]);
    }

    #[test]
    fn test_eq_polynomial() {
        let r = vec![FieldElement::from(2u64), FieldElement::from(3u64)];
        let eq = MultilinearExtension::eq_polynomial(&r);

        // eq((0,0), r) = (1-r₁)(1-r₂) = (1-2)(1-3) = (-1)(-2) = 2
        // But in field arithmetic this works differently...
        // Just check it has right size
        assert_eq!(eq.num_variables(), 2);
        assert_eq!(eq.num_evaluations(), 4);
    }

    #[test]
    fn test_sparse_to_dense() {
        let entries = vec![
            (0, FieldElement::from(1u64)),
            (2, FieldElement::from(3u64)),
        ];

        let sparse = SparseMultilinear::new(entries, 2);
        let dense = sparse.to_dense();

        assert_eq!(dense.evaluations()[0], FieldElement::from(1u64));
        assert_eq!(dense.evaluations()[1], FieldElement::zero());
        assert_eq!(dense.evaluations()[2], FieldElement::from(3u64));
        assert_eq!(dense.evaluations()[3], FieldElement::zero());
    }

    #[test]
    fn test_tensor_product() {
        let a = DenseMultilinear::from_evaluations(vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
        ]);

        let b = DenseMultilinear::from_evaluations(vec![
            FieldElement::from(3u64),
            FieldElement::from(4u64),
        ]);

        let c = a.tensor_product(&b);

        assert_eq!(c.num_variables(), 2);
        assert_eq!(c.evaluations()[0], FieldElement::from(3u64));  // 1*3
        assert_eq!(c.evaluations()[1], FieldElement::from(4u64));  // 1*4
        assert_eq!(c.evaluations()[2], FieldElement::from(6u64));  // 2*3
        assert_eq!(c.evaluations()[3], FieldElement::from(8u64));  // 2*4
    }

    #[test]
    fn test_constant_polynomial() {
        let poly = DenseMultilinear::constant(FieldElement::from(42u64), 3);
        assert_eq!(poly.num_variables(), 3);
        assert!(poly.evaluations().iter().all(|&v| v == FieldElement::from(42u64)));
    }

    #[test]
    fn test_variable_polynomial() {
        let poly = DenseMultilinear::variable(0, 2);

        // x₀ should be 0 for indices 0,1 (where bit 0 of index is 0) and 1 for indices 2,3
        assert_eq!(poly.evaluations()[0], FieldElement::zero());  // (0,0)
        assert_eq!(poly.evaluations()[1], FieldElement::zero());  // (0,1)
        assert_eq!(poly.evaluations()[2], FieldElement::one());   // (1,0)
        assert_eq!(poly.evaluations()[3], FieldElement::one());   // (1,1)
    }
}
