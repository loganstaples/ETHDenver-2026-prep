//! Beaver triple data structures.

use serde::{Deserialize, Serialize};

/// A scalar Beaver triple: (a, b, c) where c = a * b.
/// Each value here is one party's share of the triple.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeaverTriple {
    /// Share of first random value.
    pub a: f64,
    /// Share of second random value.
    pub b: f64,
    /// Share of product (a * b).
    pub c: f64,
}

impl BeaverTriple {
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self { a, b, c }
    }
}

/// A vector Beaver triple for element-wise multiplication of vectors.
/// (a, b, c) where c[i] = a[i] * b[i] for all i.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorBeaverTriple {
    /// Shares of first random vector.
    pub a: Vec<f64>,
    /// Shares of second random vector.
    pub b: Vec<f64>,
    /// Shares of element-wise product.
    pub c: Vec<f64>,
    /// Dimension of the vectors.
    pub dim: usize,
}

impl VectorBeaverTriple {
    pub fn new(a: Vec<f64>, b: Vec<f64>, c: Vec<f64>) -> Self {
        let dim = a.len();
        assert_eq!(b.len(), dim);
        assert_eq!(c.len(), dim);
        Self { a, b, c, dim }
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
    pub a: Vec<f64>,
    /// Share of second random matrix [k x n], flattened.
    pub b: Vec<f64>,
    /// Share of matrix product [m x n], flattened.
    pub c: Vec<f64>,
    /// Rows of A / rows of C.
    pub m: usize,
    /// Columns of A / rows of B (inner dimension).
    pub k: usize,
    /// Columns of B / columns of C.
    pub n: usize,
}

impl MatrixBeaverTriple {
    pub fn new(a: Vec<f64>, b: Vec<f64>, c: Vec<f64>, m: usize, k: usize, n: usize) -> Self {
        assert_eq!(a.len(), m * k, "A must be [{}x{}]", m, k);
        assert_eq!(b.len(), k * n, "B must be [{}x{}]", k, n);
        assert_eq!(c.len(), m * n, "C must be [{}x{}]", m, n);
        Self { a, b, c, m, k, n }
    }
}
