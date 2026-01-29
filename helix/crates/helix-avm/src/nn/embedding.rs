//! Embedding layer implementation with error bound tracking.
//!
//! Implements a lookup table for token embeddings.

use helix_core::types::{BoundedTensor, BoundedValue};
use thiserror::Error;

/// Errors from embedding operations.
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Token index {0} out of range (vocab size: {1})")]
    IndexOutOfRange(usize, usize),

    #[error("Empty embedding table")]
    EmptyTable,

    #[error("Inconsistent embedding dimension")]
    InconsistentDimension,
}

/// An embedding layer for token lookup.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Embedding table (vocab_size x embedding_dim).
    table: BoundedTensor,
    /// Vocabulary size.
    vocab_size: usize,
    /// Embedding dimension.
    embedding_dim: usize,
}

impl Embedding {
    /// Creates a new embedding layer from a table.
    ///
    /// Table shape should be (vocab_size, embedding_dim).
    pub fn new(table: BoundedTensor) -> Result<Self, EmbeddingError> {
        if !table.is_matrix() {
            return Err(EmbeddingError::InconsistentDimension);
        }

        let vocab_size = table.shape()[0];
        let embedding_dim = table.shape()[1];

        if vocab_size == 0 || embedding_dim == 0 {
            return Err(EmbeddingError::EmptyTable);
        }

        Ok(Self {
            table,
            vocab_size,
            embedding_dim,
        })
    }

    /// Creates an embedding layer with zero-initialized embeddings.
    pub fn zeros(vocab_size: usize, embedding_dim: usize) -> Self {
        Self {
            table: BoundedTensor::zeros(vec![vocab_size, embedding_dim]),
            vocab_size,
            embedding_dim,
        }
    }

    /// Creates an embedding layer from raw f64 values.
    pub fn from_raw(
        data: Vec<f64>,
        vocab_size: usize,
        embedding_dim: usize,
    ) -> Result<Self, EmbeddingError> {
        if data.len() != vocab_size * embedding_dim {
            return Err(EmbeddingError::InconsistentDimension);
        }

        let table = BoundedTensor::from_exact(data, vec![vocab_size, embedding_dim]);
        Self::new(table)
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Returns the embedding table.
    pub fn table(&self) -> &BoundedTensor {
        &self.table
    }

    /// Looks up a single token embedding.
    pub fn lookup(&self, token_id: usize) -> Result<BoundedTensor, EmbeddingError> {
        if token_id >= self.vocab_size {
            return Err(EmbeddingError::IndexOutOfRange(token_id, self.vocab_size));
        }

        let mut embedding = Vec::with_capacity(self.embedding_dim);
        for j in 0..self.embedding_dim {
            embedding.push(*self.table.get(&[token_id, j]).unwrap());
        }

        Ok(BoundedTensor::new(embedding, vec![self.embedding_dim]))
    }

    /// Forward pass: looks up embeddings for a sequence of token IDs.
    ///
    /// Input: tensor of token IDs (1D: sequence_length, or 2D: batch_size x sequence_length)
    /// Output: embedded sequence (2D: sequence_length x embedding_dim, or 3D: batch x seq x dim)
    pub fn forward(&self, token_ids: &[usize]) -> Result<BoundedTensor, EmbeddingError> {
        let seq_len = token_ids.len();
        let mut result = Vec::with_capacity(seq_len * self.embedding_dim);

        for &token_id in token_ids {
            if token_id >= self.vocab_size {
                return Err(EmbeddingError::IndexOutOfRange(token_id, self.vocab_size));
            }

            for j in 0..self.embedding_dim {
                result.push(*self.table.get(&[token_id, j]).unwrap());
            }
        }

        Ok(BoundedTensor::new(result, vec![seq_len, self.embedding_dim]))
    }

    /// Batched forward pass.
    ///
    /// Input: 2D array of token IDs [batch_size][sequence_length]
    /// Output: 3D tensor (batch_size x sequence_length x embedding_dim)
    pub fn forward_batch(
        &self,
        batch_token_ids: &[Vec<usize>],
    ) -> Result<BoundedTensor, EmbeddingError> {
        if batch_token_ids.is_empty() {
            return Ok(BoundedTensor::zeros(vec![0, 0, self.embedding_dim]));
        }

        let batch_size = batch_token_ids.len();
        let seq_len = batch_token_ids[0].len();

        // Verify uniform sequence lengths
        for tokens in batch_token_ids {
            if tokens.len() != seq_len {
                // Padding should be handled externally
                return Err(EmbeddingError::InconsistentDimension);
            }
        }

        let mut result = Vec::with_capacity(batch_size * seq_len * self.embedding_dim);

        for tokens in batch_token_ids {
            for &token_id in tokens {
                if token_id >= self.vocab_size {
                    return Err(EmbeddingError::IndexOutOfRange(token_id, self.vocab_size));
                }

                for j in 0..self.embedding_dim {
                    result.push(*self.table.get(&[token_id, j]).unwrap());
                }
            }
        }

        Ok(BoundedTensor::new(
            result,
            vec![batch_size, seq_len, self.embedding_dim],
        ))
    }
}

/// Positional encoding for transformer models.
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    /// Positional encoding matrix (max_seq_len x embedding_dim).
    encodings: BoundedTensor,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Embedding dimension.
    embedding_dim: usize,
}

impl PositionalEncoding {
    /// Creates sinusoidal positional encodings.
    pub fn sinusoidal(max_seq_len: usize, embedding_dim: usize) -> Self {
        let mut data = Vec::with_capacity(max_seq_len * embedding_dim);

        for pos in 0..max_seq_len {
            for i in 0..embedding_dim {
                let angle = (pos as f64) / f64::powf(10000.0, (2 * (i / 2)) as f64 / embedding_dim as f64);
                let value = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
                data.push(BoundedValue::exact(value));
            }
        }

        Self {
            encodings: BoundedTensor::new(data, vec![max_seq_len, embedding_dim]),
            max_seq_len,
            embedding_dim,
        }
    }

    /// Returns the positional encoding for a given position.
    pub fn get(&self, position: usize) -> Option<BoundedTensor> {
        if position >= self.max_seq_len {
            return None;
        }

        let mut encoding = Vec::with_capacity(self.embedding_dim);
        for j in 0..self.embedding_dim {
            encoding.push(*self.encodings.get(&[position, j])?);
        }

        Some(BoundedTensor::new(encoding, vec![self.embedding_dim]))
    }

    /// Adds positional encoding to an embedded sequence.
    ///
    /// Input: (sequence_length x embedding_dim)
    /// Output: input + positional_encoding
    pub fn forward(&self, input: &BoundedTensor) -> Result<BoundedTensor, EmbeddingError> {
        if !input.is_matrix() {
            return Err(EmbeddingError::InconsistentDimension);
        }

        let seq_len = input.shape()[0];
        let dim = input.shape()[1];

        if dim != self.embedding_dim {
            return Err(EmbeddingError::InconsistentDimension);
        }

        if seq_len > self.max_seq_len {
            return Err(EmbeddingError::IndexOutOfRange(seq_len, self.max_seq_len));
        }

        let mut result = Vec::with_capacity(seq_len * dim);

        for i in 0..seq_len {
            for j in 0..dim {
                let input_val = input.get(&[i, j]).unwrap();
                let pos_val = self.encodings.get(&[i, j]).unwrap();
                result.push(*input_val + *pos_val);
            }
        }

        Ok(BoundedTensor::new(result, vec![seq_len, dim]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_creation() {
        let embed = Embedding::zeros(100, 64);
        assert_eq!(embed.vocab_size(), 100);
        assert_eq!(embed.embedding_dim(), 64);
    }

    #[test]
    fn test_embedding_lookup() {
        // Simple 3x2 embedding table
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let embed = Embedding::from_raw(data, 3, 2).unwrap();

        let e0 = embed.lookup(0).unwrap();
        assert_eq!(e0.values(), vec![1.0, 2.0]);

        let e2 = embed.lookup(2).unwrap();
        assert_eq!(e2.values(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_embedding_forward() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let embed = Embedding::from_raw(data, 3, 2).unwrap();

        let output = embed.forward(&[0, 2, 1]).unwrap();
        assert_eq!(output.shape(), &vec![3, 2]);

        let values = output.values();
        // Token 0: [1, 2], Token 2: [5, 6], Token 1: [3, 4]
        assert_eq!(values, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0]);
    }

    #[test]
    fn test_embedding_out_of_range() {
        let embed = Embedding::zeros(10, 4);
        assert!(embed.lookup(10).is_err());
        assert!(embed.forward(&[5, 10, 3]).is_err());
    }

    #[test]
    fn test_positional_encoding() {
        let pe = PositionalEncoding::sinusoidal(100, 64);

        // Position 0 should have sin(0) = 0 for even dims
        let enc0 = pe.get(0).unwrap();
        assert!(enc0.values()[0].abs() < 1e-10); // sin(0) = 0
    }

    #[test]
    fn test_positional_encoding_forward() {
        let pe = PositionalEncoding::sinusoidal(100, 4);
        let input = BoundedTensor::zeros(vec![10, 4]);

        let output = pe.forward(&input).unwrap();
        assert_eq!(output.shape(), &vec![10, 4]);
    }
}
