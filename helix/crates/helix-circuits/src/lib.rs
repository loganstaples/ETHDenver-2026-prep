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

// Re-export params module types
pub use params::{
    // Setup and SRS
    HelixSRS, SRSCache, SRSMetadata, ParameterProfile,
    PowersOfTauCeremony, FinalizedCeremony, CeremonyContribution,
    SetupError, SRSGenerationBenchmark,
    global_cache, init_global_cache,
    MAX_K, MIN_K, SRS_FORMAT_VERSION,

    // Keys
    HelixProvingKey, HelixVerificationKey, KeyBundle, KeyCache,
    ProvingKeyMetadata, VerificationKeyMetadata,
    KeygenBenchmark, CircuitConfig, CommitmentScheme, KeyError,
    KEY_FORMAT_VERSION, PK_MAGIC, VK_MAGIC,

    // Optimization
    CircuitAnalysis, CircuitOptimizer, OptimizationPass, OptimizationResult,
    OptimizationSummary, ProofSizeEstimator, ProofSizeEstimate,
    ModelConfig, CircuitBenchmarkSuite, BenchmarkReport,
};

// Re-export approximate module types
pub use approximate::{
    QuantFormat, QuantParams, QuantizedValue, QuantErrorTracker,
    QuantizationChip, QuantizationConfig, QuantizedMatMulCircuit,
    QuantizedActivationTable, estimate_layer_error,
    INT4_RANGE, INT8_RANGE, DEFAULT_SCALE,
};

#[cfg(test)]
mod tests;
