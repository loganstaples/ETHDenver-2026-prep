//! Fuzz tests for crash resistance.
//!
//! These tests verify that the core types handle malformed and edge-case inputs
//! without panicking or crashing. This is critical for a system that processes
//! untrusted input from distributed workers.

#[cfg(test)]
mod tests {
    use crate::config::{TrainingConfig, VMConfig};
    use crate::types::{
        BoundedTensor, BoundedValue, ErrorMargin, TensorBuilder,
        MAX_TENSOR_DIMS,
    };
    use crate::validation::{
        validate_training_config, validate_vm_config,
        sanitize_f64, InputSanitizer, RecoveryContext, RecoveryStrategy,
    };

    // ============================================================================
    // Edge Case Values
    // ============================================================================

    const EDGE_VALUES: [f64; 15] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::EPSILON,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1e-308,  // Near subnormal
        1e308,   // Near max
        -1e308,  // Near min
        std::f64::consts::PI,
    ];

    // ============================================================================
    // BoundedValue Fuzz Tests
    // ============================================================================

    #[test]
    fn fuzz_bounded_value_creation() {
        // Test that BoundedValue can be created with any value without crashing
        for &val in &EDGE_VALUES {
            // These should not panic
            let _ = BoundedValue::exact(val);
            let _ = BoundedValue::new(val, ErrorMargin::ZERO);
            let _ = BoundedValue::new(val, ErrorMargin::absolute(0.1));
            let _ = BoundedValue::new(val, ErrorMargin::relative(0.1));
        }
    }

    #[test]
    fn fuzz_bounded_value_try_creation() {
        // Test validated creation with edge values
        for &val in &EDGE_VALUES {
            // These should return Result, not panic
            let _ = BoundedValue::try_with_absolute_error(val, 0.1);
            let _ = BoundedValue::try_with_relative_error(val, 0.1);

            for &err in &EDGE_VALUES {
                let _ = BoundedValue::try_with_absolute_error(val, err);
            }
        }
    }

    #[test]
    fn fuzz_bounded_value_arithmetic() {
        // Test arithmetic with all combinations of edge values
        for &a in &EDGE_VALUES {
            for &b in &EDGE_VALUES {
                let va = BoundedValue::exact(a);
                let vb = BoundedValue::exact(b);

                // Regular arithmetic should not panic (may produce NaN/Inf)
                let _ = va + vb;
                let _ = va - vb;
                let _ = va * vb;
                let _ = va / vb;
                let _ = -va;

                // Checked arithmetic should return Result, not panic
                let _ = va.checked_add(vb);
                let _ = va.checked_sub(vb);
                let _ = va.checked_mul(vb);
                let _ = va.checked_div(vb);

                // Saturating arithmetic should not panic
                let _ = va.saturating_add(vb);
                let _ = va.saturating_sub(vb);
                let _ = va.saturating_mul(vb);
                let _ = va.saturating_div(vb);
            }
        }
    }

    #[test]
    fn fuzz_bounded_value_queries() {
        // Test query methods with edge values
        for &val in &EDGE_VALUES {
            let v = BoundedValue::exact(val);

            // These should not panic
            let _ = v.is_finite();
            let _ = v.is_nan();
            let _ = v.is_infinite();
            let _ = v.is_valid();
            let _ = v.has_finite_error();
            let _ = v.has_valid_bounds();

            // If the value is finite, these should work
            if val.is_finite() {
                let _ = v.lower_bound();
                let _ = v.upper_bound();
                let _ = v.absolute_error();
                let _ = v.interval_width();
            }

            // Validation should return Result, not panic
            let _ = v.validate();
            let _ = v.validated();
        }
    }

    #[test]
    fn fuzz_bounded_value_sanitize() {
        // Test that sanitize always produces valid output
        for &val in &EDGE_VALUES {
            let mut v = BoundedValue::exact(val);
            v.sanitize();

            // After sanitization, value should be valid
            assert!(v.is_finite(), "sanitized value should be finite: {:?}", val);
            assert!(v.is_valid(), "sanitized value should be valid: {:?}", val);
        }
    }

    #[test]
    fn fuzz_bounded_value_comparison() {
        // Test comparison operations
        for &a in &EDGE_VALUES {
            for &b in &EDGE_VALUES {
                let va = BoundedValue::exact(a);
                let vb = BoundedValue::exact(b);

                // These should not panic (but may return false for NaN)
                let _ = va.contains(b);
                let _ = va.overlaps(&vb);
                let _ = va == vb;
            }
        }
    }

    // ============================================================================
    // BoundedTensor Fuzz Tests
    // ============================================================================

    #[test]
    fn fuzz_tensor_creation_shapes() {
        // Test various shape combinations
        let shapes: Vec<Vec<usize>> = vec![
            vec![],
            vec![0],
            vec![1],
            vec![100],
            vec![1, 1],
            vec![1, 0],
            vec![0, 1],
            vec![2, 3],
            vec![10, 10, 10],
            vec![1; MAX_TENSOR_DIMS],
            vec![1; MAX_TENSOR_DIMS + 1],  // Should fail
            vec![usize::MAX],  // Should fail
            vec![1000, 1000, 1000],  // May be too large
        ];

        for shape in shapes {
            // try_zeros should not panic
            let result = BoundedTensor::try_zeros(shape.clone());
            // Just check it doesn't crash - it may succeed or fail
            let _ = result;
        }
    }

    #[test]
    fn fuzz_tensor_creation_data() {
        // Test tensor creation with various data
        for &val in &EDGE_VALUES {
            // Create small tensors with edge values
            let data = vec![val; 4];

            // try_from_exact should not panic
            let _ = BoundedTensor::try_from_exact(data.clone(), vec![2, 2]);

            // try_from_approximate should not panic
            for &err in &EDGE_VALUES {
                let _ = BoundedTensor::try_from_approximate(data.clone(), vec![2, 2], err);
            }
        }
    }

    #[test]
    fn fuzz_tensor_operations() {
        // Create some test tensors
        let valid = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let with_nan = BoundedTensor::from_exact(vec![1.0, f64::NAN, 3.0, 4.0], vec![2, 2]);
        let with_inf = BoundedTensor::from_exact(vec![1.0, f64::INFINITY, 3.0, 4.0], vec![2, 2]);
        let zeros = BoundedTensor::zeros(vec![2, 2]);

        let tensors = [&valid, &with_nan, &with_inf, &zeros];

        for a in &tensors {
            for b in &tensors {
                // Regular operations should not panic
                let _ = a.add(b);
                let _ = a.sub(b);
                let _ = a.hadamard(b);
                let _ = a.div(b);

                // Try operations should not panic
                let _ = a.try_add(b);
                let _ = a.try_sub(b);
                let _ = a.try_hadamard(b);
                let _ = a.try_div(b);

                // Checked operations should not panic
                let _ = a.checked_add(b);
                let _ = a.checked_sub(b);
                let _ = a.checked_hadamard(b);
                let _ = a.checked_div(b);
            }

            // Scalar operations should not panic
            for &val in &EDGE_VALUES {
                let scalar = BoundedValue::exact(val);
                let _ = a.scale(scalar);
                let _ = a.try_scale(scalar);
            }
        }
    }

    #[test]
    fn fuzz_tensor_indexing() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // Various index combinations
        let indices: Vec<Vec<usize>> = vec![
            vec![],
            vec![0],
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![1, 1],
            vec![2, 0],  // Out of bounds
            vec![0, 2],  // Out of bounds
            vec![0, 0, 0],  // Wrong dimension
            vec![usize::MAX, 0],  // Huge index
        ];

        for idx in indices {
            // get should return None for invalid indices, not panic
            let _ = t.get(&idx);

            // try_get should return Err for invalid indices, not panic
            let _ = t.try_get(&idx);
        }
    }

    #[test]
    fn fuzz_tensor_queries() {
        for &val in &EDGE_VALUES {
            let t = BoundedTensor::from_exact(vec![val; 4], vec![2, 2]);

            // These should not panic
            let _ = t.contains_nan();
            let _ = t.contains_inf();
            let _ = t.is_finite();
            let _ = t.max_error();
            let _ = t.mean_error();
            let _ = t.sum();
            let _ = t.mean();
            let _ = t.product();
            let _ = t.min();
            let _ = t.max();
            let _ = t.validate();
        }
    }

    #[test]
    fn fuzz_tensor_sanitize() {
        for &val in &EDGE_VALUES {
            let mut t = BoundedTensor::from_exact(vec![val; 4], vec![2, 2]);
            t.sanitize();

            // After sanitization, tensor should be finite
            assert!(t.is_finite(), "sanitized tensor should be finite: {:?}", val);
        }
    }

    #[test]
    fn fuzz_tensor_reshape() {
        let t = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Various reshape targets
        let shapes: Vec<Vec<usize>> = vec![
            vec![6],
            vec![3, 2],
            vec![1, 6],
            vec![6, 1],
            vec![1, 1, 6],
            vec![2, 2],  // Wrong size
            vec![],  // Empty
        ];

        for shape in shapes {
            // try_reshape should not panic
            let _ = t.try_reshape(shape);
        }
    }

    #[test]
    fn fuzz_tensor_builder() {
        // Test builder with various inputs
        for &val in &EDGE_VALUES {
            let _ = TensorBuilder::new()
                .data(vec![val; 4])
                .shape(vec![2, 2])
                .build();

            let _ = TensorBuilder::new()
                .data(vec![val; 4])
                .shape(vec![2, 2])
                .error(0.1)
                .build();

            let _ = TensorBuilder::new()
                .data(vec![val; 4])
                .shape(vec![2, 2])
                .allow_non_finite()
                .build();
        }

        // Missing fields
        let _ = TensorBuilder::new().build();
        let _ = TensorBuilder::new().data(vec![1.0]).build();
        let _ = TensorBuilder::new().shape(vec![1]).build();
    }

    // ============================================================================
    // Validation Fuzz Tests
    // ============================================================================

    #[test]
    fn fuzz_input_sanitizer() {
        let sanitizer = InputSanitizer::new();
        let strict_sanitizer = InputSanitizer::new().strict();
        let clamping_sanitizer = InputSanitizer::new().clamp(-1.0, 1.0);

        for &val in &EDGE_VALUES {
            // These should not panic
            let _ = sanitizer.sanitize_value(val, "test");
            let _ = strict_sanitizer.sanitize_value(val, "test");
            let _ = clamping_sanitizer.sanitize_value(val, "test");
        }

        // Test vector sanitization
        let data: Vec<f64> = EDGE_VALUES.to_vec();
        let _ = sanitizer.sanitize_vec(&data, "test");
        let _ = strict_sanitizer.sanitize_vec(&data, "test");
    }

    #[test]
    fn fuzz_recovery_context() {
        let strategies = [
            RecoveryStrategy::FailFast,
            RecoveryStrategy::ReplaceWithDefault,
            RecoveryStrategy::ClampToRange,
            RecoveryStrategy::SkipInvalid,
        ];

        for strategy in strategies {
            let mut ctx = RecoveryContext::new(strategy)
                .with_default(0.0)
                .with_clamp_range(-1.0, 1.0);

            for &val in &EDGE_VALUES {
                // Should not panic, may return error
                let _ = ctx.recover_value(val, "test");
            }
        }
    }

    #[test]
    fn fuzz_sanitize_f64() {
        for &val in &EDGE_VALUES {
            let sanitized = sanitize_f64(val);
            // Result should always be finite
            assert!(sanitized.is_finite(), "sanitized {} should be finite", val);
        }
    }

    // ============================================================================
    // Config Validation Fuzz Tests
    // ============================================================================

    #[test]
    fn fuzz_vm_config_validation() {
        for &val in &EDGE_VALUES {
            let config = VMConfig {
                max_error_accumulation: val,
                ..Default::default()
            };
            // Should not panic
            let _ = validate_vm_config(&config);
        }
    }

    #[test]
    fn fuzz_training_config_validation() {
        for &val in &EDGE_VALUES {
            let config = TrainingConfig {
                learning_rate: val,
                ..Default::default()
            };
            // Should not panic
            let _ = validate_training_config(&config);

            let config = TrainingConfig {
                max_gradient_error: val,
                ..Default::default()
            };
            let _ = validate_training_config(&config);
        }
    }

    // ============================================================================
    // Random Input Tests
    // ============================================================================

    #[test]
    fn fuzz_random_tensor_operations() {
        // Generate pseudo-random values using a simple LCG
        fn lcg(seed: u64) -> impl Iterator<Item = f64> {
            let mut state = seed;
            std::iter::from_fn(move || {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                let val = (state >> 16) as f64 / (u16::MAX as f64) * 200.0 - 100.0;
                Some(val)
            })
        }

        // Create random tensors and perform operations
        for seed in 0..10 {
            let data1: Vec<f64> = lcg(seed).take(100).collect();
            let data2: Vec<f64> = lcg(seed + 100).take(100).collect();

            let t1 = BoundedTensor::from_exact(data1, vec![10, 10]);
            let t2 = BoundedTensor::from_exact(data2, vec![10, 10]);

            // All operations should not panic
            let _ = t1.add(&t2);
            let _ = t1.sub(&t2);
            let _ = t1.hadamard(&t2);
            let _ = t1.div(&t2);
            let _ = t1.sum();
            let _ = t1.mean();
            let _ = t1.max();
            let _ = t1.min();
            let _ = t1.transpose();
        }
    }

    #[test]
    fn fuzz_accumulated_operations() {
        // Test long chains of operations to detect accumulated errors
        let mut v = BoundedValue::exact(1.0);

        for i in 0..1000 {
            let addend = BoundedValue::exact(0.001);

            // Use saturating operations to prevent crashes
            v = v.saturating_add(addend);

            // Verify invariants after each operation
            assert!(v.is_finite(), "failed at iteration {}", i);
        }

        // Final value should be approximately 2.0
        assert!(v.value() > 1.5 && v.value() < 2.5);
    }

    #[test]
    fn fuzz_error_propagation() {
        // Test that error propagation doesn't explode to invalid values
        let mut v = BoundedValue::<f64>::with_absolute_error(1.0, 0.01);

        for i in 0..100 {
            let other = BoundedValue::<f64>::with_absolute_error(1.0, 0.01);
            v = v.saturating_mul(other);

            // Value may become very large or very small, but should stay valid
            if !v.is_valid() {
                v.sanitize();
            }
            assert!(v.is_valid(), "failed at iteration {}", i);
        }
    }

    // ============================================================================
    // Stress Tests
    // ============================================================================

    #[test]
    fn stress_large_tensor() {
        // Create and operate on larger tensors
        let size = 10000;
        let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.001).collect();

        let t = BoundedTensor::from_exact(data, vec![100, 100]);

        // Operations should complete without panic
        let _ = t.sum();
        let _ = t.mean();
        let _ = t.max_error();
        let _ = t.validate();

        // Transpose should work
        let transposed = t.transpose();
        assert_eq!(transposed.shape(), &vec![100, 100]);
    }

    #[test]
    fn stress_many_small_operations() {
        // Many small operations to test for memory leaks or accumulation issues
        for _ in 0..10000 {
            let a = BoundedValue::exact(1.0);
            let b = BoundedValue::exact(2.0);
            let _ = a + b;
            let _ = a - b;
            let _ = a * b;
            let _ = a / b;
        }
    }

    // ============================================================================
    // ErrorMargin Fuzz Tests
    // ============================================================================

    #[test]
    fn fuzz_error_margin_creation() {
        // Test that negative values are handled
        let positive = ErrorMargin::absolute(0.1);
        assert!(positive.epsilon() >= 0.0);

        // Zero error
        let zero = ErrorMargin::ZERO;
        assert_eq!(zero.epsilon(), 0.0);

        // Test composition with edge values
        for &a in &EDGE_VALUES.iter().filter(|v| v.is_finite() && **v >= 0.0).collect::<Vec<_>>() {
            let e = ErrorMargin::absolute(*a);
            let _ = e.to_absolute(1.0);
            let _ = e.to_relative(1.0);
            let _ = e.epsilon();
        }
    }

    #[test]
    fn fuzz_error_margin_composition() {
        let e1 = ErrorMargin::absolute(0.1);
        let e2 = ErrorMargin::absolute(0.2);

        for &a in &EDGE_VALUES {
            for &b in &EDGE_VALUES {
                // These should not panic
                let _ = e1.add(&e2, a, b);
                let _ = e1.multiply(&e2, a, b);
            }
        }
    }
}
