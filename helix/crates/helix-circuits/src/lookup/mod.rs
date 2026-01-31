//! Lookup Table Module for HELIX Circuits.
//!
//! This module provides production-grade plookup-style lookup table infrastructure
//! for efficient verification of non-linear operations in neural networks.
//!
//! # Architecture
//!
//! The lookup system is organized hierarchically:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        Lookup Table Infrastructure                       │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                    PlookupTable (Core)                           │   │
//! │  │  - Multi-column lookup support                                   │   │
//! │  │  - Batch lookup optimization                                     │   │
//! │  │  - Preprocessing for constant tables                             │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────────┐   │
//! │  │   ReLU    │  │   GELU    │  │  Sigmoid  │  │     Softmax       │   │
//! │  │  Lookup   │  │  Lookup   │  │  Lookup   │  │     Lookup        │   │
//! │  └───────────┘  └───────────┘  └───────────┘  └───────────────────┘   │
//! │                                                                         │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                 Quantized Activation Tables                      │   │
//! │  │  - INT8 range (256 entries)                                      │   │
//! │  │  - INT4 range (16 entries)                                       │   │
//! │  │  - Error-bounded approximations                                  │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! Lookup-based activations provide 10x+ speedup over arithmetic circuits:
//! - ReLU: 1 lookup vs 2+ comparison constraints
//! - GELU: 1 lookup vs 50+ polynomial approximation constraints
//! - Sigmoid: 1 lookup vs 20+ exp approximation constraints

pub mod table;
pub mod relu;
pub mod gelu;
pub mod softmax;

pub use table::{
    PlookupTable, PlookupTableConfig, PlookupChip,
    MultiColumnLookup, BatchLookupOptimizer,
    LookupTableBuilder, PrecomputedTable,
    LookupStats, LookupError,
};

pub use relu::{
    ReLULookup, ReLULookupConfig, ReLUChip,
    LeakyReLULookup, ReLU6Lookup, PReLULookup,
    relu_table_entries, leaky_relu_table_entries,
};

pub use gelu::{
    GELULookup, GELULookupConfig, GELUChip,
    FastGELULookup, GELUDerivativeLookup,
    gelu_table_entries, gelu_derivative_entries,
};

pub use softmax::{
    SigmoidLookup, SigmoidLookupConfig, SigmoidChip,
    TanhLookup, SoftmaxExpLookup,
    sigmoid_table_entries, tanh_table_entries,
    softmax_exp_entries,
};

/// Default scale factor for quantized lookup tables.
pub const DEFAULT_LOOKUP_SCALE: u64 = 256;

/// Maximum lookup table size for INT8 operations.
pub const INT8_TABLE_SIZE: usize = 256;

/// Maximum lookup table size for INT4 operations.
pub const INT4_TABLE_SIZE: usize = 16;

/// Lookup table size for extended precision (INT12).
pub const INT12_TABLE_SIZE: usize = 4096;
