//! Formal verification stubs for MPC protocols.
//!
//! This module provides a framework for specifying and checking protocol properties
//! that can be formally verified. The stubs define:
//!
//! - **Protocol invariants**: Properties that must hold throughout execution
//! - **Security properties**: Confidentiality, integrity, and correctness guarantees
//! - **Pre/post conditions**: Contracts for protocol operations
//! - **Proof obligations**: Theorems to be verified by external tools
//!
//! # Architecture
//!
//! The verification framework consists of:
//! - `properties`: Declarative specification of protocol properties
//! - `invariants`: Runtime invariant checking (debug builds)
//! - `proofs`: Stub types for formal proof obligations
//! - `models`: Abstract models for verification tools
//!
//! # Integration with Formal Tools
//!
//! These stubs are designed to integrate with:
//! - **Kani**: Rust model checker for bounded verification
//! - **Prusti**: Rust verification tool based on Viper
//! - **Creusot**: Deductive verification for Rust
//! - **Custom SMT encoding**: Z3/CVC5 integration

pub mod invariants;
pub mod models;
pub mod properties;
pub mod proofs;

pub use invariants::{
    InvariantChecker, InvariantViolation, ProtocolInvariant, RuntimeInvariantChecker,
};
pub use models::{
    AbstractChannel, AbstractParty, AbstractShare, IdealFunctionality, ProtocolModel,
    SimulatorModel, ViewIndistinguishability,
};
pub use properties::{
    AdversaryModel, CorrectnessProperty, IntegrityProperty, PrivacyProperty,
    PropertyVerifier, ProtocolProperty, SecurityLevel, SecurityProperty,
};
pub use proofs::{
    ProofContext, ProofObligation, ProofStatus, ProofWitness, TheoremStatement,
    VerificationResult,
};
