pub mod evm;
pub mod native;

pub use native::{NativeVerifier, SerializedProof, VerifierError, ProofMetadata};
pub use evm::{SolidityGenerator, generate_verifier_contract};
