pub mod aggregation;
pub mod attention;
pub mod config;
pub mod embedding;
pub mod gradient;
pub mod layer_norm;
pub mod linear_layer;
pub mod positional;
pub mod softmax;
pub mod batch;
pub mod proof_aggregation;
pub mod training_step_v2;
pub mod training_step_v3;
pub mod transformer;

#[cfg(test)]
mod adversarial;

pub use config::{
    TransformerConfig, TransformerConfigBuilder, TransformerBlockConfig,
    QuantizationConfig, QuantizationPrecision,
    ActivationType, PositionalEncodingType, AttentionType,
    CircuitActivation, LossFunction, LayerSpec, MLPArchitecture,
};
