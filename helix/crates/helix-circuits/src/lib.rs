pub mod approximate;
pub mod benchmark;
pub mod commitment;
pub mod gadgets;
pub mod ivc;
pub mod ml;
pub mod params;
pub mod verifier;

pub use halo2_proofs;
pub use halo2curves;

// Re-export key types for convenience
pub use benchmark::{BenchmarkResult, BenchmarkSuite, OverheadAnalysis};
pub use ivc::{IVCAccumulator, IVCChain, IVCStepCircuit, IVCStepWitness};
pub use ml::training_step_v2::{
    MLTrainingStepV2Circuit, MLTrainingStepV2Witness, compute_witness_v2, compute_state_hash_v2,
    ErrorTracker,
};
pub use ml::softmax::{SoftmaxChip, SoftmaxCircuit, SoftmaxWitness, compute_softmax};

#[cfg(test)]
mod tests;
