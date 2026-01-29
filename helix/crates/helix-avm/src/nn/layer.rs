//! Generic layer trait for neural network components.

use helix_core::types::BoundedTensor;
use std::fmt::Debug;

/// Trait for neural network layers that can perform forward passes.
pub trait Layer: Debug + Clone {
    /// Error type for forward pass.
    type Error: std::error::Error;

    /// Performs forward pass on input.
    fn forward(&self, input: &BoundedTensor) -> Result<BoundedTensor, Self::Error>;

    /// Returns the input dimension (if applicable).
    fn input_dim(&self) -> Option<usize> {
        None
    }

    /// Returns the output dimension (if applicable).
    fn output_dim(&self) -> Option<usize> {
        None
    }
}

/// A sequence of layers applied in order.
#[derive(Debug, Clone)]
pub struct Sequential<L: Layer> {
    layers: Vec<L>,
}

impl<L: Layer> Sequential<L> {
    /// Creates an empty sequential container.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Adds a layer to the sequence.
    pub fn add(mut self, layer: L) -> Self {
        self.layers.push(layer);
        self
    }

    /// Returns the number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Performs forward pass through all layers.
    pub fn forward(&self, input: &BoundedTensor) -> Result<BoundedTensor, L::Error> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

impl<L: Layer> Default for Sequential<L> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple identity layer for testing
    #[derive(Debug, Clone)]
    struct IdentityLayer;

    impl Layer for IdentityLayer {
        type Error = std::convert::Infallible;

        fn forward(&self, input: &BoundedTensor) -> Result<BoundedTensor, Self::Error> {
            Ok(input.clone())
        }
    }

    #[test]
    fn test_sequential() {
        let seq = Sequential::new()
            .add(IdentityLayer)
            .add(IdentityLayer);

        assert_eq!(seq.len(), 2);

        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        let output = seq.forward(&input).unwrap();
        assert_eq!(input.values(), output.values());
    }
}
