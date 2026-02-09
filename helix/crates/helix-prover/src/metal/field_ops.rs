//! Metal-Accelerated Field Operations.
//!
//! This module provides GPU-accelerated field arithmetic for the BN254 scalar field.
//! Operations are batched for efficiency.

use super::{MetalDevice, MetalError, MetalResult, MetalConfig, MetalStats};
use crate::gkr::{FieldElement, multilinear::DenseMultilinear, multilinear::MultilinearPolynomial};
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::ff::PrimeField;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Type of field operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOpType {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Inversion.
    Inv,
    /// Negation.
    Neg,
    /// Square.
    Square,
}

/// A batch field operation to execute on GPU.
#[derive(Debug, Clone)]
pub struct BatchFieldOperation {
    /// Type of operation.
    pub op_type: FieldOpType,
    /// First operand (for all operations).
    pub operands_a: Vec<FieldElement>,
    /// Second operand (for binary operations).
    pub operands_b: Option<Vec<FieldElement>>,
}

impl BatchFieldOperation {
    /// Creates a batch addition.
    pub fn add(a: Vec<FieldElement>, b: Vec<FieldElement>) -> Self {
        assert_eq!(a.len(), b.len());
        Self {
            op_type: FieldOpType::Add,
            operands_a: a,
            operands_b: Some(b),
        }
    }

    /// Creates a batch multiplication.
    pub fn mul(a: Vec<FieldElement>, b: Vec<FieldElement>) -> Self {
        assert_eq!(a.len(), b.len());
        Self {
            op_type: FieldOpType::Mul,
            operands_a: a,
            operands_b: Some(b),
        }
    }

    /// Creates a batch subtraction.
    pub fn sub(a: Vec<FieldElement>, b: Vec<FieldElement>) -> Self {
        assert_eq!(a.len(), b.len());
        Self {
            op_type: FieldOpType::Sub,
            operands_a: a,
            operands_b: Some(b),
        }
    }

    /// Creates a batch inversion.
    pub fn inv(a: Vec<FieldElement>) -> Self {
        Self {
            op_type: FieldOpType::Inv,
            operands_a: a,
            operands_b: None,
        }
    }

    /// Returns the batch size.
    pub fn len(&self) -> usize {
        self.operands_a.len()
    }

    /// Returns whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.operands_a.is_empty()
    }
}

/// Metal field operations accelerator.
///
/// Provides GPU-accelerated batch field arithmetic. Falls back to CPU
/// when Metal is not available or for small batches.
pub struct MetalFieldOps {
    /// Device (if Metal is available).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    device: Option<MetalDevice>,
    /// Minimum batch size to use GPU.
    min_gpu_batch_size: usize,
    /// Statistics.
    stats: MetalStats,
}

impl MetalFieldOps {
    /// Creates a new field operations accelerator.
    pub fn new() -> Self {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let device = MetalDevice::new().ok();

        Self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            device,
            min_gpu_batch_size: 1024, // GPU overhead not worth it below this
            stats: MetalStats::default(),
        }
    }

    /// Creates with a custom minimum batch size.
    pub fn with_min_batch_size(min_batch_size: usize) -> Self {
        let mut ops = Self::new();
        ops.min_gpu_batch_size = min_batch_size;
        ops
    }

    /// Checks if GPU acceleration is available.
    pub fn is_gpu_available(&self) -> bool {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            self.device.is_some()
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            false
        }
    }

    /// Executes a batch operation.
    pub fn execute(&mut self, op: &BatchFieldOperation) -> MetalResult<Vec<FieldElement>> {
        if op.len() < self.min_gpu_batch_size || !self.is_gpu_available() {
            return Ok(self.execute_cpu(op));
        }

        self.execute_gpu(op)
    }

    /// Executes on CPU (always available).
    fn execute_cpu(&self, op: &BatchFieldOperation) -> Vec<FieldElement> {
        #[cfg(feature = "parallel")]
        {
            if op.len() >= 1024 {
                return self.execute_cpu_parallel(op);
            }
        }

        self.execute_cpu_sequential(op)
    }

    fn execute_cpu_sequential(&self, op: &BatchFieldOperation) -> Vec<FieldElement> {
        match op.op_type {
            FieldOpType::Add => {
                let b = op.operands_b.as_ref().expect("invariant: binary op requires operands_b");
                op.operands_a.iter()
                    .zip(b.iter())
                    .map(|(&a, &b)| a + b)
                    .collect()
            }
            FieldOpType::Sub => {
                let b = op.operands_b.as_ref().expect("invariant: binary op requires operands_b");
                op.operands_a.iter()
                    .zip(b.iter())
                    .map(|(&a, &b)| a - b)
                    .collect()
            }
            FieldOpType::Mul => {
                let b = op.operands_b.as_ref().expect("invariant: binary op requires operands_b");
                op.operands_a.iter()
                    .zip(b.iter())
                    .map(|(&a, &b)| a * b)
                    .collect()
            }
            FieldOpType::Inv => {
                // Batch inversion using Montgomery's trick
                self.batch_invert(&op.operands_a)
            }
            FieldOpType::Neg => {
                op.operands_a.iter().map(|&a| -a).collect()
            }
            FieldOpType::Square => {
                op.operands_a.iter().map(|&a| a.square()).collect()
            }
        }
    }

    #[cfg(feature = "parallel")]
    fn execute_cpu_parallel(&self, op: &BatchFieldOperation) -> Vec<FieldElement> {
        match op.op_type {
            FieldOpType::Add => {
                let b = op.operands_b.as_ref().expect("invariant: binary op requires operands_b");
                op.operands_a.par_iter()
                    .zip(b.par_iter())
                    .map(|(&a, &b)| a + b)
                    .collect()
            }
            FieldOpType::Sub => {
                let b = op.operands_b.as_ref().expect("invariant: binary op requires operands_b");
                op.operands_a.par_iter()
                    .zip(b.par_iter())
                    .map(|(&a, &b)| a - b)
                    .collect()
            }
            FieldOpType::Mul => {
                let b = op.operands_b.as_ref().expect("invariant: binary op requires operands_b");
                op.operands_a.par_iter()
                    .zip(b.par_iter())
                    .map(|(&a, &b)| a * b)
                    .collect()
            }
            FieldOpType::Inv => {
                // Batch inversion is sequential due to Montgomery's trick
                self.batch_invert(&op.operands_a)
            }
            FieldOpType::Neg => {
                op.operands_a.par_iter().map(|&a| -a).collect()
            }
            FieldOpType::Square => {
                op.operands_a.par_iter().map(|&a| a.square()).collect()
            }
        }
    }

    /// Batch inversion using Montgomery's trick.
    ///
    /// Computes [a₁⁻¹, a₂⁻¹, ..., aₙ⁻¹] using only one field inversion
    /// (plus O(n) multiplications).
    fn batch_invert(&self, elements: &[FieldElement]) -> Vec<FieldElement> {
        if elements.is_empty() {
            return vec![];
        }

        let n = elements.len();
        let mut results = vec![FieldElement::zero(); n];

        // Compute partial products: p[i] = a₀ * a₁ * ... * aᵢ
        let mut partial_products = Vec::with_capacity(n);
        let mut acc = FieldElement::one();

        for &a in elements {
            acc = acc * a;
            partial_products.push(acc);
        }

        // Invert the total product
        let mut inv_acc = acc.invert().unwrap_or(FieldElement::zero());

        // Compute inverses in reverse order
        for i in (0..n).rev() {
            if i == 0 {
                results[i] = inv_acc;
            } else {
                results[i] = partial_products[i - 1] * inv_acc;
                inv_acc = inv_acc * elements[i];
            }
        }

        results
    }

    /// Executes on GPU using Metal shaders.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn execute_gpu(&mut self, op: &BatchFieldOperation) -> MetalResult<Vec<FieldElement>> {
        use metal_rs::{MTLSize, MTLResourceOptions};

        let device = match &self.device {
            Some(d) => d,
            None => return Ok(self.execute_cpu(op)),
        };

        let n = op.len();
        let start_time = std::time::Instant::now();

        // Convert FieldElements to raw bytes
        let a_bytes: Vec<[u64; 4]> = op.operands_a.iter()
            .map(|f| field_element_to_limbs(f))
            .collect();

        // Create input buffer A
        let a_buffer = device.metal_device().new_buffer_with_data(
            a_bytes.as_ptr() as *const _,
            (n * 32) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create output buffer
        let c_buffer = device.metal_device().new_buffer(
            (n * 32) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Compile the appropriate kernel
        let shader_source = include_str!("shaders.metal");

        let kernel_name = match op.op_type {
            FieldOpType::Add => "field_add",
            FieldOpType::Sub => "field_sub",
            FieldOpType::Mul => "field_mul",
            FieldOpType::Neg => "field_neg",
            FieldOpType::Square => "field_square",
            FieldOpType::Inv => {
                // Batch inversion is special - use CPU (Montgomery's trick is better)
                return Ok(self.batch_invert(&op.operands_a));
            }
        };

        let pipeline = device.compile_shader(shader_source, kernel_name)?;

        // Create command buffer and encoder
        let command_buffer = device.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&a_buffer), 0);

        // Handle binary operations (need second operand)
        match op.op_type {
            FieldOpType::Add | FieldOpType::Sub | FieldOpType::Mul => {
                if let Some(ref b_operands) = op.operands_b {
                    let b_bytes: Vec<[u64; 4]> = b_operands.iter()
                        .map(|f| field_element_to_limbs(f))
                        .collect();
                    let b_buffer = device.metal_device().new_buffer_with_data(
                        b_bytes.as_ptr() as *const _,
                        (n * 32) as u64,
                        MTLResourceOptions::StorageModeShared,
                    );
                    encoder.set_buffer(1, Some(&b_buffer), 0);
                    encoder.set_buffer(2, Some(&c_buffer), 0);
                }
            }
            FieldOpType::Neg | FieldOpType::Square => {
                encoder.set_buffer(1, Some(&c_buffer), 0);
            }
            FieldOpType::Inv => unreachable!(),
        }

        // Dispatch threads
        let threadgroup_size = 256.min(n);
        let grid_size = MTLSize::new(n as u64, 1, 1);
        let tg_size = MTLSize::new(threadgroup_size as u64, 1, 1);

        encoder.dispatch_threads(grid_size, tg_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results back
        let result_ptr = c_buffer.contents() as *const [u64; 4];
        let mut results = Vec::with_capacity(n);
        unsafe {
            for i in 0..n {
                let limbs = *result_ptr.add(i);
                results.push(limbs_to_field_element(&limbs));
            }
        }

        // Update stats
        self.stats.gpu_time_us += start_time.elapsed().as_micros() as u64;
        self.stats.num_dispatches += 1;
        self.stats.bytes_transferred += n * 64; // Input + output

        Ok(results)
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn execute_gpu(&mut self, op: &BatchFieldOperation) -> MetalResult<Vec<FieldElement>> {
        Ok(self.execute_cpu(op))
    }

    /// Returns statistics.
    pub fn stats(&self) -> &MetalStats {
        &self.stats
    }

    /// Resets statistics.
    pub fn reset_stats(&mut self) {
        self.stats = MetalStats::default();
    }
}

impl Default for MetalFieldOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Metal-accelerated polynomial operations.
pub struct MetalPolynomialOps {
    /// Field operations.
    field_ops: MetalFieldOps,
}

impl MetalPolynomialOps {
    /// Creates a new polynomial operations accelerator.
    pub fn new() -> Self {
        Self {
            field_ops: MetalFieldOps::new(),
        }
    }

    /// Evaluates a multilinear polynomial at a point.
    ///
    /// Uses GPU acceleration for large polynomials.
    pub fn evaluate(&mut self, poly: &DenseMultilinear, point: &[FieldElement]) -> MetalResult<FieldElement> {
        let n = poly.num_variables();
        if n == 0 || poly.evaluations().is_empty() {
            return Ok(FieldElement::zero());
        }

        // For small polynomials, use CPU
        if poly.num_evaluations() < 1024 {
            return Ok(poly.evaluate(point));
        }

        // GPU-accelerated streaming evaluation
        let mut current = poly.evaluations().to_vec();

        for &r in point.iter() {
            let half = current.len() / 2;

            // Compute: new[i] = current[i] + r * (current[i + half] - current[i])
            // = current[i] * (1 - r) + current[i + half] * r

            let (first_half, second_half) = current.split_at(half);

            // Compute differences: d[i] = current[i + half] - current[i]
            let diffs = BatchFieldOperation::sub(
                second_half.to_vec(),
                first_half.to_vec(),
            );
            let diff_results = self.field_ops.execute(&diffs)?;

            // Multiply by r
            let r_vec = vec![r; half];
            let scaled = BatchFieldOperation::mul(diff_results, r_vec);
            let scaled_results = self.field_ops.execute(&scaled)?;

            // Add to first half
            let sums = BatchFieldOperation::add(first_half.to_vec(), scaled_results);
            current = self.field_ops.execute(&sums)?;
        }

        Ok(current.first().copied().unwrap_or(FieldElement::zero()))
    }

    /// Computes the tensor product of two polynomials.
    pub fn tensor_product(
        &mut self,
        a: &DenseMultilinear,
        b: &DenseMultilinear,
    ) -> MetalResult<DenseMultilinear> {
        let a_evals = a.evaluations();
        let b_evals = b.evaluations();

        let new_size = a_evals.len() * b_evals.len();

        if new_size < 4096 {
            // CPU fallback for small tensors
            return Ok(a.tensor_product(b));
        }

        // GPU-accelerated tensor product
        let mut result = Vec::with_capacity(new_size);

        for &a_val in a_evals {
            // Multiply all of b by a_val
            let a_vec = vec![a_val; b_evals.len()];
            let products = BatchFieldOperation::mul(a_vec, b_evals.to_vec());
            let prod_results = self.field_ops.execute(&products)?;
            result.extend(prod_results);
        }

        Ok(DenseMultilinear::from_evaluations(result))
    }

    /// Computes the sum of polynomial evaluations.
    pub fn sum(&mut self, poly: &DenseMultilinear) -> MetalResult<FieldElement> {
        let evals = poly.evaluations();

        if evals.len() < 1024 {
            return Ok(evals.iter().fold(FieldElement::zero(), |a, b| a + b));
        }

        // GPU-accelerated reduction
        let mut current = evals.to_vec();

        while current.len() > 1 {
            let half = current.len() / 2;
            let (first, second) = current.split_at(half);

            // Handle odd length
            let second_vec = if second.len() < first.len() {
                let mut s = second.to_vec();
                s.push(FieldElement::zero());
                s
            } else {
                second.to_vec()
            };

            let sums = BatchFieldOperation::add(first.to_vec(), second_vec);
            current = self.field_ops.execute(&sums)?;
        }

        Ok(current.first().copied().unwrap_or(FieldElement::zero()))
    }
}

impl Default for MetalPolynomialOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Metal-accelerated sumcheck operations.
pub struct MetalSumcheckAccelerator {
    /// Polynomial operations.
    poly_ops: MetalPolynomialOps,
}

impl MetalSumcheckAccelerator {
    /// Creates a new sumcheck accelerator.
    pub fn new() -> Self {
        Self {
            poly_ops: MetalPolynomialOps::new(),
        }
    }

    /// Computes partial sums for a sumcheck round.
    ///
    /// Given evaluations [f(0, x₂, ..., xₙ), f(1, x₂, ..., xₙ), ...],
    /// computes (Σ f(0, ...), Σ f(1, ...)).
    pub fn compute_partial_sums(
        &mut self,
        evaluations: &[FieldElement],
    ) -> MetalResult<(FieldElement, FieldElement)> {
        if evaluations.len() < 2 {
            let val = evaluations.first().copied().unwrap_or(FieldElement::zero());
            return Ok((val, FieldElement::zero()));
        }

        let half = evaluations.len() / 2;
        let first_half = &evaluations[..half];
        let second_half = &evaluations[half..];

        // Use polynomial sum for GPU acceleration
        let first_poly = DenseMultilinear::from_evaluations(first_half.to_vec());
        let second_poly = DenseMultilinear::from_evaluations(second_half.to_vec());

        let sum_0 = self.poly_ops.sum(&first_poly)?;
        let sum_1 = self.poly_ops.sum(&second_poly)?;

        Ok((sum_0, sum_1))
    }

    /// Computes the next round's evaluations after receiving a challenge.
    ///
    /// Given evaluations and challenge r, computes:
    /// new[i] = old[i] + r * (old[i + half] - old[i])
    pub fn compute_next_round(
        &mut self,
        evaluations: &[FieldElement],
        challenge: FieldElement,
    ) -> MetalResult<Vec<FieldElement>> {
        if evaluations.len() < 2 {
            return Ok(evaluations.to_vec());
        }

        let half = evaluations.len() / 2;
        let first_half = &evaluations[..half];
        let second_half = &evaluations[half..];

        // Compute: new[i] = first[i] + r * (second[i] - first[i])

        // diff = second - first
        let diffs = BatchFieldOperation::sub(second_half.to_vec(), first_half.to_vec());
        let diff_results = self.poly_ops.field_ops.execute(&diffs)?;

        // scaled = r * diff
        let r_vec = vec![challenge; half];
        let scaled = BatchFieldOperation::mul(diff_results, r_vec);
        let scaled_results = self.poly_ops.field_ops.execute(&scaled)?;

        // result = first + scaled
        let sums = BatchFieldOperation::add(first_half.to_vec(), scaled_results);
        self.poly_ops.field_ops.execute(&sums)
    }
}

impl Default for MetalSumcheckAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions for Field Element Conversion
// ============================================================================

/// Convert a FieldElement to raw u64 limbs (Montgomery form).
fn field_element_to_limbs(f: &FieldElement) -> [u64; 4] {
    use helix_circuits::halo2curves::ff::PrimeField;
    let repr = f.to_repr();
    let bytes = repr.as_ref();

    // Convert from little-endian bytes to u64 limbs
    let mut limbs = [0u64; 4];
    for i in 0..4 {
        let offset = i * 8;
        limbs[i] = u64::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
            bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7],
        ]);
    }
    limbs
}

/// Convert raw u64 limbs back to FieldElement.
fn limbs_to_field_element(limbs: &[u64; 4]) -> FieldElement {
    use helix_circuits::halo2curves::ff::PrimeField;

    // Convert limbs to little-endian bytes
    let mut bytes = [0u8; 32];
    for i in 0..4 {
        let le_bytes = limbs[i].to_le_bytes();
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&le_bytes);
    }

    // Try to convert - this may fail for invalid values, so use from_repr
    FieldElement::from_repr_vartime(bytes.into()).unwrap_or(FieldElement::zero())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_add() {
        let a = vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
        ];
        let b = vec![
            FieldElement::from(4u64),
            FieldElement::from(5u64),
            FieldElement::from(6u64),
        ];

        let op = BatchFieldOperation::add(a, b);
        let mut ops = MetalFieldOps::new();
        let result = ops.execute(&op).unwrap();

        assert_eq!(result[0], FieldElement::from(5u64));
        assert_eq!(result[1], FieldElement::from(7u64));
        assert_eq!(result[2], FieldElement::from(9u64));
    }

    #[test]
    fn test_batch_mul() {
        let a = vec![
            FieldElement::from(2u64),
            FieldElement::from(3u64),
        ];
        let b = vec![
            FieldElement::from(4u64),
            FieldElement::from(5u64),
        ];

        let op = BatchFieldOperation::mul(a, b);
        let mut ops = MetalFieldOps::new();
        let result = ops.execute(&op).unwrap();

        assert_eq!(result[0], FieldElement::from(8u64));
        assert_eq!(result[1], FieldElement::from(15u64));
    }

    #[test]
    fn test_batch_invert() {
        let elements = vec![
            FieldElement::from(2u64),
            FieldElement::from(3u64),
            FieldElement::from(5u64),
        ];

        let ops = MetalFieldOps::new();
        let inverses = ops.batch_invert(&elements);

        // Verify: a * a^-1 = 1
        for (&a, &inv) in elements.iter().zip(inverses.iter()) {
            assert_eq!(a * inv, FieldElement::one());
        }
    }

    #[test]
    fn test_polynomial_evaluate() {
        let evals = vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
            FieldElement::from(4u64),
        ];

        let poly = DenseMultilinear::from_evaluations(evals);
        let mut ops = MetalPolynomialOps::new();

        // Evaluate at (0, 0)
        let result = ops.evaluate(&poly, &[FieldElement::zero(), FieldElement::zero()]).unwrap();
        assert_eq!(result, FieldElement::from(1u64));

        // Evaluate at (1, 1)
        let result = ops.evaluate(&poly, &[FieldElement::one(), FieldElement::one()]).unwrap();
        assert_eq!(result, FieldElement::from(4u64));
    }

    #[test]
    fn test_polynomial_sum() {
        let evals = vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
            FieldElement::from(4u64),
        ];

        let poly = DenseMultilinear::from_evaluations(evals);
        let mut ops = MetalPolynomialOps::new();

        let sum = ops.sum(&poly).unwrap();
        assert_eq!(sum, FieldElement::from(10u64));
    }

    #[test]
    fn test_sumcheck_partial_sums() {
        let evals = vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
            FieldElement::from(4u64),
        ];

        let mut accel = MetalSumcheckAccelerator::new();
        let (sum_0, sum_1) = accel.compute_partial_sums(&evals).unwrap();

        // First half sum: 1 + 2 = 3
        // Second half sum: 3 + 4 = 7
        assert_eq!(sum_0, FieldElement::from(3u64));
        assert_eq!(sum_1, FieldElement::from(7u64));
    }

    #[test]
    fn test_sumcheck_next_round() {
        let evals = vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
            FieldElement::from(4u64),
        ];

        let mut accel = MetalSumcheckAccelerator::new();
        let r = FieldElement::from(2u64);
        let next = accel.compute_next_round(&evals, r).unwrap();

        // new[0] = 1 + 2*(3-1) = 1 + 4 = 5
        // new[1] = 2 + 2*(4-2) = 2 + 4 = 6
        assert_eq!(next[0], FieldElement::from(5u64));
        assert_eq!(next[1], FieldElement::from(6u64));
    }
}
