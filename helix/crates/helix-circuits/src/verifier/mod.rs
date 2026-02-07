pub mod evm;
pub mod format_spec;
pub mod native;
pub mod transcript;

pub use native::{NativeVerifier, SerializedProof, VerifierError, ProofMetadata};
pub use evm::{
    SolidityGenerator, VkData, KzgSrs, generate_verifier_contract,
    EvmProof, EvmProofBuilder, EvmPublicInputsArray,
    create_test_proof, create_test_public_inputs,
};
pub use format_spec::{
    MIN_PROOF_SIZE, NUM_ADVICE_COMMITS, G1_POINT_SIZE, SCALAR_SIZE, NUM_PUBLIC_INPUTS,
    fr_to_evm_bytes, fq_to_evm_bytes, g1_to_evm_bytes,
    evm_bytes_to_fr, evm_bytes_to_fq, evm_bytes_to_g1,
    validate_proof_format, validate_public_inputs,
    serialize_proof_for_evm,
    compute_hash_pair, verify_commitment,
    ProofFormatError, ProofStructure, ProofSection, EvmPublicInputs,
};
pub use transcript::{Keccak256Write, Keccak256Read};
