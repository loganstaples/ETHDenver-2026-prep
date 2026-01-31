//! GKR Compatibility Module for HELIX.
//!
//! This module provides infrastructure for expressing computations in a format
//! compatible with the GKR (Goldwasser-Kalai-Rothblum) interactive proof protocol.
//!
//! # Background
//!
//! GKR is an interactive proof protocol for verifying layered arithmetic circuits.
//! It uses the sumcheck protocol to verify computations layer by layer, achieving
//! O(d log n) prover time for depth-d circuits with n gates per layer.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    GKR Compatibility Layer                               │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                 Layered Circuit Representation                   │   │
//! │  │  - Gates organized into layers                                   │   │
//! │  │  - Add/Mul gates with layer-local indices                        │   │
//! │  │  - Wiring functions W_add and W_mul                              │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                 Halo2 ↔ GKR Conversion                           │   │
//! │  │  - Convert Halo2 constraint system to layered circuit           │   │
//! │  │  - Convert GKR layered circuit to Halo2 constraints             │   │
//! │  │  - Handle custom gates and lookups                               │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │                 GKR Proof Verification Circuit                   │   │
//! │  │  - Verify GKR sumcheck proofs in Halo2                           │   │
//! │  │  - Enable on-chain aggregation of GKR proofs                     │   │
//! │  │  - SNARK wrapper for GKR proofs                                  │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Use Cases
//!
//! 1. **Hybrid Proving**: Use GKR for dense matrix operations (cheaper), then
//!    wrap in Halo2 for on-chain verification.
//!
//! 2. **On-Chain Aggregation**: Verify multiple GKR proofs within a single
//!    SNARK, reducing on-chain costs.
//!
//! 3. **Neural Network Layers**: Express linear layers as GKR circuits for
//!    efficient sumcheck-based verification.

pub mod layered;
pub mod conversion;

pub use layered::{
    LayeredCircuit, Layer, Gate, GateType,
    WiringFunction, LayeredCircuitBuilder,
    LayerStats, CircuitMetadata,
};

pub use conversion::{
    Halo2ToGkr, GkrToHalo2, ConversionConfig,
    GkrProofVerifierCircuit, GkrProofVerifierConfig,
    GkrProof, GkrProofWitness, SumcheckRound,
    convert_halo2_to_layered, convert_layered_to_halo2,
};

/// Maximum supported circuit depth for GKR.
pub const MAX_GKR_DEPTH: usize = 64;

/// Maximum gates per layer for efficient sumcheck.
pub const MAX_GATES_PER_LAYER: usize = 1 << 20;

/// Minimum gates per layer (for batching efficiency).
pub const MIN_GATES_PER_LAYER: usize = 64;
