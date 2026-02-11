pub mod approximate;
pub mod benchmark;
pub mod cache;
pub mod commitment;
pub mod gadgets;
pub mod ivc;
pub mod lookup;
pub mod ml;
pub mod params;
pub mod quantization;
pub mod verifier;

pub use halo2_proofs;
pub use halo2curves;

// Re-export key types for convenience
pub use benchmark::{BenchmarkResult, BenchmarkSuite, OverheadAnalysis};
pub use ivc::{
    IVCAccumulator, IVCChain, IVCStepCircuit, IVCStepWitness,
    IVCFoldingCircuit, IVCFoldingWitness, IVCMultiStepCircuit, IVCMultiStepWitness,
    IVCProofStep, FoldResult, fold_accumulators, generate_folding_challenge,
    IVC_PUBLIC_INPUTS, FOLDING_PUBLIC_INPUTS, MAX_MULTI_STEPS,
};
pub use ml::training_step_v2::{
    MLTrainingStepV2Circuit, MLTrainingStepV2Witness, compute_witness_v2, compute_state_hash_v2,
    ErrorTracker, ToEvmProof, ToEvmPublicInputs,
};
pub use ml::batch::{MLBatchCircuit, BatchProofResult, MAX_BATCH_SIZE};
pub use ml::proof_aggregation::{
    SHPLONKAggregationCircuit, SHPLONKAggregationWitness, SHPLONKAggConfig,
    AggregationStepWitness, step_witness_from_public_inputs,
    MAX_AGGREGATION_BATCH, AGGREGATION_NUM_PUBLIC_INPUTS,
};

// Re-export EVM format types for proof-to-contract compatibility
pub use verifier::{
    // Core EVM proof types
    EvmProof, EvmProofBuilder, EvmPublicInputsArray,
    // Format specification constants
    MIN_PROOF_SIZE, NUM_ADVICE_COMMITS, G1_POINT_SIZE, SCALAR_SIZE,
    NUM_PUBLIC_INPUTS as EVM_NUM_PUBLIC_INPUTS,
    // Serialization functions
    fr_to_evm_bytes, fq_to_evm_bytes, g1_to_evm_bytes,
    evm_bytes_to_fr, evm_bytes_to_fq, evm_bytes_to_g1,
    // Validation
    validate_proof_format, validate_public_inputs, compute_hash_pair, verify_commitment,
    // Errors and structures
    ProofFormatError, ProofStructure, ProofSection, EvmPublicInputs,
    // Test utilities
    create_test_proof, create_test_public_inputs,
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
    HelixProvingKey, HelixVerificationKey, KeyBundle, RealKeyBundle, KeyCache,
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

// Re-export lookup module types
pub use lookup::{
    // Core lookup infrastructure
    PlookupTable, PlookupTableConfig, PlookupChip,
    MultiColumnLookup, BatchLookupOptimizer,
    LookupTableBuilder, PrecomputedTable,
    LookupStats, LookupError,
    DEFAULT_LOOKUP_SCALE, INT8_TABLE_SIZE, INT4_TABLE_SIZE,

    // ReLU lookups
    ReLULookup, ReLULookupConfig, ReLUChip,
    LeakyReLULookup, ReLU6Lookup, PReLULookup,
    relu_table_entries, leaky_relu_table_entries,

    // GELU lookups
    GELULookup, GELULookupConfig, GELUChip,
    FastGELULookup, GELUDerivativeLookup,
    gelu_table_entries, gelu_derivative_entries,

    // Sigmoid/Softmax lookups
    SigmoidLookup, SigmoidLookupConfig, SigmoidChip,
    TanhLookup, SoftmaxExpLookup,
    sigmoid_table_entries, tanh_table_entries, softmax_exp_entries,
};

// Re-export quantization module types
pub use quantization::{
    // INT8
    Int8QuantCircuit, Int8QuantConfig, Int8QuantChip,
    Int8SymmetricParams, Int8AsymmetricParams,
    Int8MatMulCircuit, Int8DotProductCircuit,
    PerChannelInt8Params, Int8QuantWitness,

    // INT4
    Int4QuantCircuit, Int4QuantConfig, Int4QuantChip,
    Int4PackedCircuit, Int4MixedPrecisionCircuit,
    Int4WeightParams, pack_int4_values, unpack_int4_values,

    // Calibration
    CalibrationCircuit, CalibrationConfig, CalibrationChip,
    MinMaxCalibration, HistogramCalibration, EntropyCalibration,
    DynamicRangeVerifier, CalibrationWitness,
    verify_calibration_bounds, compute_optimal_scale,

    // Constants
    INT8_MAX, INT8_MIN, INT4_MAX, INT4_MIN,
    DEFAULT_INT8_SCALE, DEFAULT_INT4_SCALE,
};


// Re-export cache module types (KeyCache aliased to avoid conflict with params::KeyCache)
pub use cache::{
    CircuitCache, CircuitCacheConfig, CircuitCacheBuilder,
    StructureCache, CachedStructure, StructureKey,
    CacheStats, CacheConfig, EvictionPolicy,
    WitnessCache, TableCache, KeyCache as WitnessKeyCache,
    global_cache as circuit_cache,
};

#[cfg(test)]
mod tests;
