//! Arithmetic module - approximate operations with error tracking.

pub mod error_propagation;
pub mod fixed_approx;
pub mod float_approx;
pub mod int_approx;

pub use error_propagation::*;
pub use fixed_approx::FixedPoint;
pub use float_approx::*;
pub use int_approx::{QuantParams, dequantize_i8, quantize_i8, quantize_with_error};
