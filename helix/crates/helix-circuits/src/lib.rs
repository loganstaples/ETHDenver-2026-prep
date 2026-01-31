pub mod approximate;
pub mod benchmark;
pub mod commitment;
pub mod gadgets;
pub mod gkr_compat;
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

// Re-export GKR compatibility module types
pub use gkr_compat::{
    // Layered circuit representation
    LayeredCircuit, Layer, Gate, GateType,
    WiringFunction, LayeredCircuitBuilder,
    LayerStats, CircuitMetadata,

    // Conversion
    Halo2ToGkr, GkrToHalo2, ConversionConfig,
    GkrProofVerifierCircuit, GkrProofVerifierConfig,
    GkrProof, GkrProofWitness, SumcheckRound,
    convert_halo2_to_layered, convert_layered_to_halo2,

    // Constants
    MAX_GKR_DEPTH, MAX_GATES_PER_LAYER, MIN_GATES_PER_LAYER,
};

#[cfg(test)]
mod tests;
