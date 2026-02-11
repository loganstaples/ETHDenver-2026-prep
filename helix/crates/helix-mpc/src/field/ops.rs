//! Extended field operations for MPC protocols.
//!
//! This module provides higher-level operations on field elements that are
//! commonly needed in MPC protocols, including batch operations, vector
//! operations, and matrix operations.

use super::unified::Fr;
use rand::RngCore;
use zeroize::Zeroize;

// ========== Batch Operations ==========

/// Adds multiple field elements.
#[inline]
pub fn sum(elements: &[Fr]) -> Fr {
    elements.iter().fold(Fr::ZERO, |acc, x| &acc + x)
}

/// Computes the inner product of two vectors: sum(a[i] * b[i]).
#[inline]
pub fn inner_product(a: &[Fr], b: &[Fr]) -> Fr {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    a.iter()
        .zip(b.iter())
        .fold(Fr::ZERO, |acc, (x, y)| &acc + &(x * y))
}

/// Computes the inner product using fixed-point multiplication.
/// Both inputs should be from_f64 encoded vectors.
#[inline]
pub fn inner_product_fixed(a: &[Fr], b: &[Fr]) -> Fr {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    a.iter()
        .zip(b.iter())
        .fold(Fr::ZERO, |acc, (x, y)| &acc + &x.fixed_mul(y))
}

/// Scales a vector by a scalar: result[i] = scalar * v[i].
#[inline]
pub fn scale_vec(v: &[Fr], scalar: &Fr) -> Vec<Fr> {
    v.iter().map(|x| x * scalar).collect()
}

/// Adds two vectors element-wise.
#[inline]
pub fn add_vec(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Subtracts two vectors element-wise.
#[inline]
pub fn sub_vec(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Multiplies two vectors element-wise (Hadamard product).
#[inline]
pub fn mul_vec(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Multiplies two vectors element-wise using fixed-point multiplication.
/// Both inputs should be from_f64 encoded vectors.
#[inline]
pub fn mul_vec_fixed(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    a.iter().zip(b.iter()).map(|(x, y)| x.fixed_mul(y)).collect()
}

/// Negates all elements in a vector.
#[inline]
pub fn neg_vec(v: &[Fr]) -> Vec<Fr> {
    v.iter().map(|x| x.neg()).collect()
}

// ========== Matrix Operations ==========

/// Matrix multiplication: C = A @ B.
/// A is [m x k], B is [k x n], result C is [m x n].
/// All matrices in row-major flattened form.
#[inline]
pub fn matmul(a: &[Fr], b: &[Fr], m: usize, k: usize, n: usize) -> Vec<Fr> {
    assert_eq!(a.len(), m * k, "A must be [{}x{}]", m, k);
    assert_eq!(b.len(), k * n, "B must be [{}x{}]", k, n);

    let mut c = vec![Fr::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = Fr::ZERO;
            for l in 0..k {
                sum = &sum + &(&a[i * k + l] * &b[l * n + j]);
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Matrix multiplication using fixed-point arithmetic.
/// Both inputs should be from_f64 encoded matrices.
#[inline]
pub fn matmul_fixed(a: &[Fr], b: &[Fr], m: usize, k: usize, n: usize) -> Vec<Fr> {
    assert_eq!(a.len(), m * k, "A must be [{}x{}]", m, k);
    assert_eq!(b.len(), k * n, "B must be [{}x{}]", k, n);

    let mut c = vec![Fr::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = Fr::ZERO;
            for l in 0..k {
                sum = &sum + &a[i * k + l].fixed_mul(&b[l * n + j]);
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Matrix-vector multiplication: y = A @ x.
/// A is [m x n], x is [n], result y is [m].
#[inline]
pub fn matvec(a: &[Fr], x: &[Fr], m: usize, n: usize) -> Vec<Fr> {
    matmul(a, x, m, n, 1)
}

/// Matrix transpose.
/// A is [rows x cols], result is [cols x rows].
#[inline]
pub fn transpose(a: &[Fr], rows: usize, cols: usize) -> Vec<Fr> {
    assert_eq!(a.len(), rows * cols, "Matrix must be [{}x{}]", rows, cols);

    let mut result = vec![Fr::ZERO; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = a[i * cols + j].clone();
        }
    }
    result
}

/// Outer product: C = u @ v^T.
/// u is [m], v is [n], result C is [m x n].
#[inline]
pub fn outer_product(u: &[Fr], v: &[Fr]) -> Vec<Fr> {
    let m = u.len();
    let n = v.len();
    let mut c = vec![Fr::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = &u[i] * &v[j];
        }
    }
    c
}

// ========== Random Generation ==========

/// Generates a random field element vector.
#[inline]
pub fn random_vec<R: RngCore>(rng: &mut R, len: usize) -> Vec<Fr> {
    (0..len).map(|_| Fr::random(rng)).collect()
}

/// Generates a random field element matrix.
#[inline]
pub fn random_matrix<R: RngCore>(rng: &mut R, rows: usize, cols: usize) -> Vec<Fr> {
    random_vec(rng, rows * cols)
}

// ========== Conversion Helpers ==========

/// Converts a slice of f64 values to field elements.
#[inline]
pub fn from_f64_vec(values: &[f64]) -> Vec<Fr> {
    values.iter().map(|v| Fr::from_f64(*v)).collect()
}

/// Converts field elements back to f64 values.
#[inline]
pub fn to_f64_vec(elements: &[Fr]) -> Vec<f64> {
    elements.iter().map(|e| e.to_f64()).collect()
}

/// Converts a slice of u64 values to field elements.
#[inline]
pub fn from_u64_vec(values: &[u64]) -> Vec<Fr> {
    values.iter().map(|v| Fr::from_u64(*v)).collect()
}

// ========== Zeroization ==========

/// Zeroizes a vector of field elements securely.
#[inline]
pub fn zeroize_vec(v: &mut Vec<Fr>) {
    for elem in v.iter_mut() {
        elem.zeroize();
    }
}

/// A vector that zeroizes on drop.
#[derive(Clone)]
pub struct SecureVec {
    data: Vec<Fr>,
}

impl SecureVec {
    /// Creates a new empty secure vector.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Creates a secure vector with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Creates a secure vector from a Vec<Fr>.
    pub fn from_vec(data: Vec<Fr>) -> Self {
        Self { data }
    }

    /// Creates a zeroed secure vector of the given length.
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![Fr::ZERO; len],
        }
    }

    /// Returns the length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a slice.
    pub fn as_slice(&self) -> &[Fr] {
        &self.data
    }

    /// Returns a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [Fr] {
        &mut self.data
    }

    /// Pushes an element.
    pub fn push(&mut self, elem: Fr) {
        self.data.push(elem);
    }

    /// Pops an element.
    pub fn pop(&mut self) -> Option<Fr> {
        self.data.pop()
    }

    /// Converts to a regular Vec (consuming self).
    pub fn into_vec(mut self) -> Vec<Fr> {
        std::mem::take(&mut self.data)
    }
}

impl Default for SecureVec {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SecureVec {
    fn drop(&mut self) {
        zeroize_vec(&mut self.data);
    }
}

impl std::ops::Index<usize> for SecureVec {
    type Output = Fr;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl std::ops::IndexMut<usize> for SecureVec {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl std::ops::Deref for SecureVec {
    type Target = [Fr];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for SecureVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

// ========== Polynomial Operations ==========

/// Evaluates a polynomial at a point.
/// coeffs[i] is the coefficient of x^i.
#[inline]
pub fn poly_eval(coeffs: &[Fr], x: &Fr) -> Fr {
    // Horner's method
    let mut result = Fr::ZERO;
    for coeff in coeffs.iter().rev() {
        result = &(&result * x) + coeff;
    }
    result
}

/// Interpolates a polynomial through points using Lagrange interpolation.
/// Returns the coefficients of the polynomial.
#[inline]
pub fn lagrange_interpolate(points: &[(Fr, Fr)]) -> Vec<Fr> {
    let n = points.len();
    if n == 0 {
        return vec![];
    }

    let mut result = vec![Fr::ZERO; n];

    for i in 0..n {
        let (xi, yi) = &points[i];

        // Compute the Lagrange basis polynomial L_i
        // L_i(x) = product over j != i of (x - xj) / (xi - xj)

        // First compute the denominator: product over j != i of (xi - xj)
        let mut denom = Fr::ONE;
        for (j, (xj, _)) in points.iter().enumerate() {
            if i != j {
                denom = &denom * &(xi - xj);
            }
        }
        let denom_inv = denom.inverse().expect("Distinct x values required");

        // Compute the numerator polynomial coefficients
        // Start with polynomial = 1
        let mut basis = vec![Fr::ONE];

        for (j, (xj, _)) in points.iter().enumerate() {
            if i != j {
                // Multiply by (x - xj)
                let mut new_basis = vec![Fr::ZERO; basis.len() + 1];
                for (k, coeff) in basis.iter().enumerate() {
                    new_basis[k + 1] = &new_basis[k + 1] + coeff; // x term
                    new_basis[k] = &new_basis[k] - &(coeff * xj); // constant term
                }
                basis = new_basis;
            }
        }

        // Scale by yi / denom and add to result
        let scale = yi * &denom_inv;
        for (k, coeff) in basis.iter().enumerate() {
            if k < result.len() {
                result[k] = &result[k] + &(coeff * &scale);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_sum() {
        let v = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let s = sum(&v);
        assert_eq!(s.to_u64(), Some(6));
    }

    #[test]
    fn test_inner_product() {
        let a = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let b = vec![Fr::from_u64(4), Fr::from_u64(5), Fr::from_u64(6)];
        let ip = inner_product(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(ip.to_u64(), Some(32));
    }

    #[test]
    fn test_scale_vec() {
        let v = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let scalar = Fr::from_u64(10);
        let scaled = scale_vec(&v, &scalar);
        assert_eq!(scaled[0].to_u64(), Some(10));
        assert_eq!(scaled[1].to_u64(), Some(20));
        assert_eq!(scaled[2].to_u64(), Some(30));
    }

    #[test]
    fn test_add_vec() {
        let a = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let b = vec![Fr::from_u64(3), Fr::from_u64(4)];
        let c = add_vec(&a, &b);
        assert_eq!(c[0].to_u64(), Some(4));
        assert_eq!(c[1].to_u64(), Some(6));
    }

    #[test]
    fn test_sub_vec() {
        let a = vec![Fr::from_u64(5), Fr::from_u64(7)];
        let b = vec![Fr::from_u64(2), Fr::from_u64(3)];
        let c = sub_vec(&a, &b);
        assert_eq!(c[0].to_u64(), Some(3));
        assert_eq!(c[1].to_u64(), Some(4));
    }

    #[test]
    fn test_matmul() {
        // [1 2; 3 4] @ [5 6; 7 8] = [19 22; 43 50]
        let a = vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ];
        let b = vec![
            Fr::from_u64(5),
            Fr::from_u64(6),
            Fr::from_u64(7),
            Fr::from_u64(8),
        ];
        let c = matmul(&a, &b, 2, 2, 2);
        assert_eq!(c[0].to_u64(), Some(19));
        assert_eq!(c[1].to_u64(), Some(22));
        assert_eq!(c[2].to_u64(), Some(43));
        assert_eq!(c[3].to_u64(), Some(50));
    }

    #[test]
    fn test_transpose() {
        let a = vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
            Fr::from_u64(5),
            Fr::from_u64(6),
        ];
        let t = transpose(&a, 2, 3);
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        assert_eq!(t[0].to_u64(), Some(1));
        assert_eq!(t[1].to_u64(), Some(4));
        assert_eq!(t[2].to_u64(), Some(2));
        assert_eq!(t[3].to_u64(), Some(5));
        assert_eq!(t[4].to_u64(), Some(3));
        assert_eq!(t[5].to_u64(), Some(6));
    }

    #[test]
    fn test_outer_product() {
        let u = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let v = vec![Fr::from_u64(3), Fr::from_u64(4), Fr::from_u64(5)];
        let c = outer_product(&u, &v);
        // [[3,4,5],[6,8,10]]
        assert_eq!(c[0].to_u64(), Some(3));
        assert_eq!(c[1].to_u64(), Some(4));
        assert_eq!(c[2].to_u64(), Some(5));
        assert_eq!(c[3].to_u64(), Some(6));
        assert_eq!(c[4].to_u64(), Some(8));
        assert_eq!(c[5].to_u64(), Some(10));
    }

    #[test]
    fn test_from_f64_vec() {
        let v = from_f64_vec(&[1.0, 2.5, -3.0]);
        let back = to_f64_vec(&v);
        assert!((back[0] - 1.0).abs() < 1e-10);
        assert!((back[1] - 2.5).abs() < 1e-10);
        assert!((back[2] - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_poly_eval() {
        // p(x) = 2 + 3x + x^2
        let coeffs = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(1)];
        let x = Fr::from_u64(5);
        let y = poly_eval(&coeffs, &x);
        // p(5) = 2 + 15 + 25 = 42
        assert_eq!(y.to_u64(), Some(42));
    }

    #[test]
    fn test_lagrange_interpolate() {
        // Points: (1, 1), (2, 4), (3, 9) - lies on p(x) = x^2
        let points = vec![
            (Fr::from_u64(1), Fr::from_u64(1)),
            (Fr::from_u64(2), Fr::from_u64(4)),
            (Fr::from_u64(3), Fr::from_u64(9)),
        ];
        let coeffs = lagrange_interpolate(&points);

        // Verify interpolation
        for (x, y) in &points {
            let evaluated = poly_eval(&coeffs, x);
            assert!(evaluated.ct_eq(y).to_bool(), "Interpolation failed at x={:?}", x);
        }
    }

    #[test]
    fn test_secure_vec_zeroize() {
        let v = SecureVec::from_vec(vec![Fr::from_u64(42), Fr::from_u64(99)]);
        assert_eq!(v.len(), 2);
        // When dropped, should zeroize
    }

    #[test]
    fn test_random_vec() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let v = random_vec(&mut rng, 10);
        assert_eq!(v.len(), 10);
        // Should all be different (with overwhelming probability)
        for i in 0..v.len() {
            for j in i + 1..v.len() {
                assert!(!v[i].ct_eq(&v[j]).to_bool());
            }
        }
    }
}
