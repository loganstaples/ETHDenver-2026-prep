//! Constants for the HELIX project.

/// Default error bounds.
pub mod error_bounds {
    /// Default maximum relative error for F32 operations.
    pub const F32_MAX_RELATIVE_ERROR: f64 = 1.19e-7;

    /// Default maximum relative error for F16 operations.
    pub const F16_MAX_RELATIVE_ERROR: f64 = 9.77e-4;

    /// Default maximum relative error for INT8 quantization.
    pub const INT8_MAX_RELATIVE_ERROR: f64 = 1.0 / 256.0;

    /// Default maximum error accumulation threshold (for F32 precision).
    pub const MAX_ERROR_ACCUMULATION: f64 = 0.01;

    /// Maximum error accumulation threshold for BF16 precision.
    /// BF16 single-step error (~1.23e-2) exceeds the F32 budget, so a more
    /// permissive threshold is needed.
    pub const BF16_MAX_ERROR_ACCUMULATION: f64 = 0.05;

    /// Maximum error accumulation threshold for INT8 precision.
    /// INT8 quantization noise (~3.9e-3 per element) compounds through matmuls,
    /// requiring ~10x the F32 budget.
    pub const INT8_MAX_ERROR_ACCUMULATION: f64 = 0.10;

    /// Default error margin for gradient computations.
    pub const GRADIENT_ERROR_MARGIN: f64 = 0.001;
}

/// Numerical constants.
pub mod numeric {
    /// Machine epsilon for f64.
    pub const F64_EPSILON: f64 = 2.220446049250313e-16;

    /// Machine epsilon for f32.
    pub const F32_EPSILON: f32 = 1.1920929e-7;

    /// Minimum positive normal f64.
    pub const F64_MIN_POSITIVE: f64 = 2.2250738585072014e-308;

    /// Small value to prevent division by zero.
    pub const EPSILON: f64 = 1e-15;

    /// Pi.
    pub const PI: f64 = std::f64::consts::PI;

    /// Euler's number.
    pub const E: f64 = std::f64::consts::E;
}

/// Limits and thresholds.
pub mod limits {
    /// Maximum tensor dimensions.
    pub const MAX_TENSOR_DIMS: usize = 8;

    /// Maximum tensor size (number of elements).
    pub const MAX_TENSOR_SIZE: usize = 1_000_000_000;

    /// Maximum matrix dimension for single operation.
    pub const MAX_MATRIX_DIM: usize = 65536;

    /// Default chunk size for large matrix operations.
    pub const DEFAULT_CHUNK_SIZE: usize = 1024;

    /// Maximum proof size in bytes.
    pub const MAX_PROOF_SIZE: usize = 10 * 1024 * 1024; // 10MB

    /// Maximum number of operations in a single execution trace.
    pub const MAX_TRACE_OPERATIONS: usize = 10_000_000;
}

/// ZK circuit constants.
pub mod circuit {
    /// Field modulus for BN254 (commonly used in Ethereum).
    pub const BN254_MODULUS: &str =
        "21888242871839275222246405745257275088548364400416034343698204186575808495617";

    /// Number of bits in field element.
    pub const FIELD_BITS: usize = 254;

    /// Lookup table size for activations.
    pub const ACTIVATION_LOOKUP_SIZE: usize = 1024;

    /// Range check bit decomposition size.
    pub const RANGE_CHECK_BITS: usize = 16;
}

/// Network and protocol constants.
pub mod network {
    /// Protocol version string.
    pub const PROTOCOL_VERSION: &str = "helix/1.0.0";

    /// Default port for P2P communication.
    pub const DEFAULT_P2P_PORT: u16 = 9000;

    /// Default port for RPC server.
    pub const DEFAULT_RPC_PORT: u16 = 8545;

    /// Maximum message size in bytes.
    pub const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024; // 100MB

    /// Heartbeat interval in seconds.
    pub const HEARTBEAT_INTERVAL: u64 = 30;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_bounds() {
        assert!(error_bounds::F32_MAX_RELATIVE_ERROR < error_bounds::F16_MAX_RELATIVE_ERROR);
    }

    #[test]
    fn test_numeric_constants() {
        assert!(numeric::F64_EPSILON < 1e-15);
        assert!(numeric::EPSILON > 0.0);
    }
}
