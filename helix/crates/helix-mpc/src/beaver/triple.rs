//! Beaver triple data structures.
//!
//! Beaver triples are precomputed random values (a, b, c) where c = a * b,
//! used to enable secure multiplication on secret-shared values without
//! revealing the underlying values.

use crate::field::Fr;
use serde::{Deserialize, Serialize};

/// A scalar Beaver triple: (a, b, c) where c = a * b.
/// Each value here is one party's share of the triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeaverTriple {
    /// Share of first random value.
    pub a: Fr,
    /// Share of second random value.
    pub b: Fr,
    /// Share of product (a * b).
    pub c: Fr,
}

impl BeaverTriple {
    pub fn new(a: Fr, b: Fr, c: Fr) -> Self {
        Self { a, b, c }
    }

    /// Creates a triple from f64 values (for compatibility during migration).
    pub fn from_f64(a: f64, b: f64, c: f64) -> Self {
        Self {
            a: Fr::from_f64(a),
            b: Fr::from_f64(b),
            c: Fr::from_f64(c),
        }
    }
}

/// A vector Beaver triple for element-wise multiplication of vectors.
/// (a, b, c) where c[i] = a[i] * b[i] for all i.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorBeaverTriple {
    /// Shares of first random vector.
    pub a: Vec<Fr>,
    /// Shares of second random vector.
    pub b: Vec<Fr>,
    /// Shares of element-wise product.
    pub c: Vec<Fr>,
    /// Dimension of the vectors.
    pub dim: usize,
}

impl VectorBeaverTriple {
    pub fn new(a: Vec<Fr>, b: Vec<Fr>, c: Vec<Fr>) -> Self {
        let dim = a.len();
        assert_eq!(b.len(), dim);
        assert_eq!(c.len(), dim);
        Self { a, b, c, dim }
    }

    /// Creates from f64 vectors (for compatibility).
    pub fn from_f64(a: Vec<f64>, b: Vec<f64>, c: Vec<f64>) -> Self {
        let dim = a.len();
        Self {
            a: a.into_iter().map(Fr::from_f64).collect(),
            b: b.into_iter().map(Fr::from_f64).collect(),
            c: c.into_iter().map(Fr::from_f64).collect(),
            dim,
        }
    }
}

/// A matrix Beaver triple for secure matrix multiplication.
/// (A, B, C) where C = A @ B.
///
/// A is [m x k], B is [k x n], C is [m x n].
/// All stored in row-major flattened form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixBeaverTriple {
    /// Share of first random matrix [m x k], flattened.
    pub a: Vec<Fr>,
    /// Share of second random matrix [k x n], flattened.
    pub b: Vec<Fr>,
    /// Share of matrix product [m x n], flattened.
    pub c: Vec<Fr>,
    /// Rows of A / rows of C.
    pub m: usize,
    /// Columns of A / rows of B (inner dimension).
    pub k: usize,
    /// Columns of B / columns of C.
    pub n: usize,
}

impl MatrixBeaverTriple {
    pub fn new(a: Vec<Fr>, b: Vec<Fr>, c: Vec<Fr>, m: usize, k: usize, n: usize) -> Self {
        assert_eq!(a.len(), m * k, "A must be [{}x{}]", m, k);
        assert_eq!(b.len(), k * n, "B must be [{}x{}]", k, n);
        assert_eq!(c.len(), m * n, "C must be [{}x{}]", m, n);
        Self { a, b, c, m, k, n }
    }

    /// Creates from f64 vectors (for compatibility).
    pub fn from_f64(
        a: Vec<f64>,
        b: Vec<f64>,
        c: Vec<f64>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self {
        Self {
            a: a.into_iter().map(Fr::from_f64).collect(),
            b: b.into_iter().map(Fr::from_f64).collect(),
            c: c.into_iter().map(Fr::from_f64).collect(),
            m,
            k,
            n,
        }
    }
}
