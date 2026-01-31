//! Metal Shader Source Code for GPU-accelerated Field Operations.
//!
//! This module contains the Metal compute shader source code that implements
//! BN254 field arithmetic on Apple GPUs.

/// Metal shader source code for field operations.
pub const FIELD_OPS_SOURCE: &str = include_str!("shaders.metal");

/// Get the source code for a specific kernel.
pub fn get_kernel_source(kernel_name: &str) -> Option<&'static str> {
    // All kernels are in the same source file
    match kernel_name {
        "field_add" | "field_sub" | "field_neg" | "field_mul" | "field_square"
        | "parallel_sum" | "poly_eval_step" | "matmul" => Some(FIELD_OPS_SOURCE),
        _ => None,
    }
}

/// List of available kernel names.
pub const KERNEL_NAMES: &[&str] = &[
    "field_add",
    "field_sub",
    "field_neg",
    "field_mul",
    "field_square",
    "parallel_sum",
    "poly_eval_step",
    "matmul",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_source_not_empty() {
        assert!(!FIELD_OPS_SOURCE.is_empty());
    }

    #[test]
    fn test_shader_contains_kernels() {
        for kernel in KERNEL_NAMES {
            assert!(
                FIELD_OPS_SOURCE.contains(&format!("kernel void {}", kernel)),
                "Missing kernel: {}",
                kernel
            );
        }
    }

    #[test]
    fn test_get_kernel_source() {
        assert!(get_kernel_source("field_add").is_some());
        assert!(get_kernel_source("invalid_kernel").is_none());
    }
}
