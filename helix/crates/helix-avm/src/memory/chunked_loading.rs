//! Chunked Weight Loading for Large Models.
//!
//! Enables loading and processing model weights in chunks to bound peak memory usage.
//! This is essential for models that don't fit entirely in memory.

use helix_core::types::{BoundedTensor, BoundedValue};
use std::collections::VecDeque;
use thiserror::Error;

/// Errors during chunked loading.
#[derive(Error, Debug)]
pub enum ChunkError {
    #[error("Invalid chunk configuration: {0}")]
    InvalidConfig(String),

    #[error("Chunk index {index} out of bounds (total: {total})")]
    IndexOutOfBounds { index: usize, total: usize },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty tensor cannot be chunked")]
    EmptyTensor,
}

/// Configuration for chunking tensors.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum elements per chunk.
    pub max_elements: usize,
    /// Maximum bytes per chunk.
    pub max_bytes: usize,
    /// Dimension to chunk along (0 = rows, 1 = columns for matrices).
    pub chunk_dim: usize,
    /// Overlap between chunks (for convolutions, etc.).
    pub overlap: usize,
    /// Whether to pad the last chunk to full size.
    pub pad_last: bool,
}

impl ChunkConfig {
    /// Creates a new chunk configuration.
    pub fn new(max_elements: usize) -> Self {
        Self {
            max_elements,
            max_bytes: max_elements * 8, // Assuming f64
            chunk_dim: 0,
            overlap: 0,
            pad_last: false,
        }
    }

    /// Creates configuration for a specific memory limit.
    pub fn for_memory_limit(bytes: usize, bytes_per_element: usize) -> Self {
        let max_elements = bytes / bytes_per_element;
        Self {
            max_elements,
            max_bytes: bytes,
            chunk_dim: 0,
            overlap: 0,
            pad_last: false,
        }
    }

    /// Creates configuration for matrix row chunking.
    pub fn for_matrix_rows(rows_per_chunk: usize, cols: usize) -> Self {
        Self {
            max_elements: rows_per_chunk * cols,
            max_bytes: rows_per_chunk * cols * 8,
            chunk_dim: 0,
            overlap: 0,
            pad_last: false,
        }
    }

    /// Sets the dimension to chunk along.
    pub fn with_chunk_dim(mut self, dim: usize) -> Self {
        self.chunk_dim = dim;
        self
    }

    /// Sets the overlap between chunks.
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Enables padding of the last chunk.
    pub fn with_padding(mut self) -> Self {
        self.pad_last = true;
        self
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        // Default: 1M elements per chunk (~8MB for f64)
        Self::new(1_000_000)
    }
}

/// A single weight chunk.
#[derive(Debug, Clone)]
pub struct WeightChunk {
    /// The tensor data for this chunk.
    pub data: BoundedTensor,
    /// Start index in the original tensor along the chunk dimension.
    pub start_idx: usize,
    /// End index (exclusive) in the original tensor.
    pub end_idx: usize,
    /// Chunk index (0-based).
    pub chunk_idx: usize,
    /// Total number of chunks.
    pub total_chunks: usize,
    /// Original shape of the full tensor.
    pub original_shape: Vec<usize>,
    /// Whether this is the last chunk (may be smaller).
    pub is_last: bool,
}

impl WeightChunk {
    /// Returns the number of elements in this chunk.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether this chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    /// Returns the shape of this chunk.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Returns the size along the chunk dimension.
    pub fn chunk_size(&self) -> usize {
        self.end_idx - self.start_idx
    }
}

/// A tensor that can be accessed in chunks.
#[derive(Debug, Clone)]
pub struct ChunkedTensor {
    /// Full tensor data (lazily loaded chunks may be None).
    data: Option<BoundedTensor>,
    /// Pre-computed chunks (if chunked).
    chunks: Vec<WeightChunk>,
    /// Original shape.
    shape: Vec<usize>,
    /// Chunk configuration.
    config: ChunkConfig,
    /// Total number of elements.
    total_elements: usize,
}

impl ChunkedTensor {
    /// Creates a new chunked tensor from a full tensor.
    pub fn from_tensor(tensor: BoundedTensor, config: ChunkConfig) -> Result<Self, ChunkError> {
        if tensor.len() == 0 {
            return Err(ChunkError::EmptyTensor);
        }

        let shape = tensor.shape().to_vec();
        let total_elements = tensor.len();

        // Compute chunks
        let chunks = Self::compute_chunks(&tensor, &config)?;

        Ok(Self {
            data: Some(tensor),
            chunks,
            shape,
            config,
            total_elements,
        })
    }

    /// Creates a chunked tensor without loading all data (lazy loading).
    pub fn lazy(shape: Vec<usize>, config: ChunkConfig) -> Self {
        let total_elements: usize = shape.iter().product();

        Self {
            data: None,
            chunks: Vec::new(),
            shape,
            config,
            total_elements,
        }
    }

    /// Computes chunk boundaries.
    fn compute_chunks(tensor: &BoundedTensor, config: &ChunkConfig) -> Result<Vec<WeightChunk>, ChunkError> {
        let shape = tensor.shape();

        if config.chunk_dim >= shape.len() {
            return Err(ChunkError::InvalidConfig(format!(
                "Chunk dimension {} exceeds tensor dimensions {}",
                config.chunk_dim,
                shape.len()
            )));
        }

        let dim_size = shape[config.chunk_dim];

        // Calculate chunk size along the target dimension
        let elements_per_slice: usize = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != config.chunk_dim)
            .map(|(_, &s)| s)
            .product::<usize>()
            .max(1);

        let slices_per_chunk = config.max_elements / elements_per_slice.max(1);
        let chunk_size = slices_per_chunk.max(1);

        // Calculate number of chunks
        let effective_chunk_size = chunk_size.saturating_sub(config.overlap);
        let num_chunks = if effective_chunk_size > 0 {
            (dim_size + effective_chunk_size - 1) / effective_chunk_size
        } else {
            1
        };

        let mut chunks = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let start_idx = if i == 0 {
                0
            } else {
                i * effective_chunk_size
            };

            let end_idx = (start_idx + chunk_size).min(dim_size);
            let is_last = i == num_chunks - 1;

            // Extract chunk data
            let chunk_data = Self::extract_chunk(tensor, config.chunk_dim, start_idx, end_idx)?;

            chunks.push(WeightChunk {
                data: chunk_data,
                start_idx,
                end_idx,
                chunk_idx: i,
                total_chunks: num_chunks,
                original_shape: shape.to_vec(),
                is_last,
            });
        }

        Ok(chunks)
    }

    /// Extracts a chunk of data from a tensor along a dimension.
    fn extract_chunk(
        tensor: &BoundedTensor,
        dim: usize,
        start: usize,
        end: usize,
    ) -> Result<BoundedTensor, ChunkError> {
        let shape = tensor.shape();
        let chunk_size = end - start;

        // Calculate new shape
        let mut new_shape = shape.to_vec();
        new_shape[dim] = chunk_size;

        // Calculate strides
        let suffix_size: usize = shape[dim + 1..].iter().product::<usize>().max(1);
        let prefix_size: usize = shape[..dim].iter().product::<usize>().max(1);

        let mut data = Vec::with_capacity(prefix_size * chunk_size * suffix_size);
        let tensor_data = tensor.data();

        for p in 0..prefix_size {
            for s in start..end {
                for q in 0..suffix_size {
                    let idx = p * shape[dim] * suffix_size + s * suffix_size + q;
                    data.push(tensor_data[idx]);
                }
            }
        }

        Ok(BoundedTensor::new(data, new_shape))
    }

    /// Returns the total number of chunks.
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Returns a specific chunk.
    pub fn chunk(&self, idx: usize) -> Result<&WeightChunk, ChunkError> {
        self.chunks.get(idx).ok_or(ChunkError::IndexOutOfBounds {
            index: idx,
            total: self.chunks.len(),
        })
    }

    /// Returns an iterator over chunks.
    pub fn iter_chunks(&self) -> impl Iterator<Item = &WeightChunk> {
        self.chunks.iter()
    }

    /// Returns the original shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the total number of elements.
    pub fn total_elements(&self) -> usize {
        self.total_elements
    }

    /// Returns the chunk configuration.
    pub fn config(&self) -> &ChunkConfig {
        &self.config
    }

    /// Reconstructs the full tensor from chunks.
    pub fn to_tensor(&self) -> Result<BoundedTensor, ChunkError> {
        if let Some(ref data) = self.data {
            return Ok(data.clone());
        }

        if self.chunks.is_empty() {
            return Err(ChunkError::EmptyTensor);
        }

        // Reconstruct from chunks
        let total_size: usize = self.shape.iter().product();
        let mut data: Vec<BoundedValue<f64>> = Vec::with_capacity(total_size);

        let dim = self.config.chunk_dim;
        let suffix_size: usize = self.shape[dim + 1..].iter().product::<usize>().max(1);
        let prefix_size: usize = self.shape[..dim].iter().product::<usize>().max(1);

        // Pre-allocate with zeros
        data.resize(total_size, BoundedValue::exact(0.0));

        for chunk in &self.chunks {
            let chunk_data = chunk.data.data();
            let chunk_size = chunk.end_idx - chunk.start_idx;

            for p in 0..prefix_size {
                for (local_s, global_s) in (chunk.start_idx..chunk.end_idx).enumerate() {
                    for q in 0..suffix_size {
                        let src_idx = p * chunk_size * suffix_size + local_s * suffix_size + q;
                        let dst_idx = p * self.shape[dim] * suffix_size + global_s * suffix_size + q;
                        data[dst_idx] = chunk_data[src_idx];
                    }
                }
            }
        }

        Ok(BoundedTensor::new(data, self.shape.clone()))
    }

    /// Returns memory usage of currently loaded chunks.
    pub fn memory_usage(&self) -> usize {
        self.chunks.iter().map(|c| c.data.len() * 8).sum()
    }
}

/// Iterator for processing chunks.
pub struct ChunkIterator<'a> {
    chunks: &'a [WeightChunk],
    current: usize,
}

impl<'a> ChunkIterator<'a> {
    /// Creates a new chunk iterator.
    pub fn new(chunks: &'a [WeightChunk]) -> Self {
        Self { chunks, current: 0 }
    }
}

impl<'a> Iterator for ChunkIterator<'a> {
    type Item = &'a WeightChunk;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.chunks.len() {
            let chunk = &self.chunks[self.current];
            self.current += 1;
            Some(chunk)
        } else {
            None
        }
    }
}

/// Loads weights in chunks for large models.
#[derive(Debug)]
pub struct ChunkedWeightLoader {
    /// Chunk configuration.
    config: ChunkConfig,
    /// Buffer for loaded chunks.
    loaded_chunks: VecDeque<WeightChunk>,
    /// Maximum chunks to keep in memory.
    max_cached_chunks: usize,
    /// Statistics.
    stats: LoaderStats,
}

/// Statistics for the weight loader.
#[derive(Debug, Clone, Default)]
pub struct LoaderStats {
    /// Number of chunks loaded.
    pub chunks_loaded: usize,
    /// Number of chunks evicted.
    pub chunks_evicted: usize,
    /// Total bytes loaded.
    pub bytes_loaded: usize,
    /// Cache hits.
    pub cache_hits: usize,
    /// Cache misses.
    pub cache_misses: usize,
}

impl ChunkedWeightLoader {
    /// Creates a new weight loader.
    pub fn new(config: ChunkConfig) -> Self {
        Self {
            config,
            loaded_chunks: VecDeque::new(),
            max_cached_chunks: 4,
            stats: LoaderStats::default(),
        }
    }

    /// Creates a loader with a specific cache size.
    pub fn with_cache_size(config: ChunkConfig, max_chunks: usize) -> Self {
        Self {
            config,
            loaded_chunks: VecDeque::new(),
            max_cached_chunks: max_chunks,
            stats: LoaderStats::default(),
        }
    }

    /// Returns the chunk configuration.
    pub fn config(&self) -> &ChunkConfig {
        &self.config
    }

    /// Loads a tensor and returns a chunked representation.
    pub fn load(&mut self, tensor: &BoundedTensor) -> Result<ChunkedTensor, ChunkError> {
        let chunked = ChunkedTensor::from_tensor(tensor.clone(), self.config.clone())?;
        self.stats.chunks_loaded += chunked.num_chunks();
        self.stats.bytes_loaded += tensor.len() * 8;
        Ok(chunked)
    }

    /// Loads a specific chunk from a chunked tensor.
    pub fn load_chunk(&mut self, tensor: &ChunkedTensor, idx: usize) -> Result<WeightChunk, ChunkError> {
        // Check cache
        for cached in &self.loaded_chunks {
            if cached.chunk_idx == idx {
                self.stats.cache_hits += 1;
                return Ok(cached.clone());
            }
        }

        self.stats.cache_misses += 1;

        // Load chunk
        let chunk = tensor.chunk(idx)?.clone();

        // Add to cache
        if self.loaded_chunks.len() >= self.max_cached_chunks {
            self.loaded_chunks.pop_front();
            self.stats.chunks_evicted += 1;
        }
        self.loaded_chunks.push_back(chunk.clone());

        Ok(chunk)
    }

    /// Clears the cache.
    pub fn clear_cache(&mut self) {
        self.loaded_chunks.clear();
    }

    /// Returns loader statistics.
    pub fn stats(&self) -> &LoaderStats {
        &self.stats
    }

    /// Resets statistics.
    pub fn reset_stats(&mut self) {
        self.stats = LoaderStats::default();
    }

    /// Processes a tensor chunk by chunk with a callback.
    pub fn process_chunked<F, R>(&mut self, tensor: &BoundedTensor, mut f: F) -> Result<Vec<R>, ChunkError>
    where
        F: FnMut(&WeightChunk) -> R,
    {
        let chunked = ChunkedTensor::from_tensor(tensor.clone(), self.config.clone())?;
        let results: Vec<R> = chunked.iter_chunks().map(|chunk| f(chunk)).collect();
        Ok(results)
    }

    /// Applies a transformation to each chunk and reconstructs the tensor.
    pub fn transform_chunked<F>(
        &mut self,
        tensor: &BoundedTensor,
        mut f: F,
    ) -> Result<BoundedTensor, ChunkError>
    where
        F: FnMut(&BoundedTensor) -> BoundedTensor,
    {
        let chunked = ChunkedTensor::from_tensor(tensor.clone(), self.config.clone())?;

        let transformed_chunks: Vec<WeightChunk> = chunked
            .iter_chunks()
            .map(|chunk| {
                let transformed_data = f(&chunk.data);
                WeightChunk {
                    data: transformed_data,
                    start_idx: chunk.start_idx,
                    end_idx: chunk.end_idx,
                    chunk_idx: chunk.chunk_idx,
                    total_chunks: chunk.total_chunks,
                    original_shape: chunk.original_shape.clone(),
                    is_last: chunk.is_last,
                }
            })
            .collect();

        // Reconstruct
        let mut reconstructed = ChunkedTensor::lazy(chunked.shape().to_vec(), self.config.clone());
        reconstructed.chunks = transformed_chunks;
        reconstructed.to_tensor()
    }
}

/// Helper function to chunk a matrix for row-parallel processing.
pub fn chunk_matrix_rows(tensor: &BoundedTensor, rows_per_chunk: usize) -> Result<Vec<WeightChunk>, ChunkError> {
    if tensor.ndim() != 2 {
        return Err(ChunkError::InvalidConfig(
            "chunk_matrix_rows requires a 2D tensor".to_string(),
        ));
    }

    let cols = tensor.shape()[1];
    let config = ChunkConfig::for_matrix_rows(rows_per_chunk, cols);
    let chunked = ChunkedTensor::from_tensor(tensor.clone(), config)?;

    Ok(chunked.chunks)
}

/// Helper function to chunk a matrix for column-parallel processing.
pub fn chunk_matrix_cols(tensor: &BoundedTensor, cols_per_chunk: usize) -> Result<Vec<WeightChunk>, ChunkError> {
    if tensor.ndim() != 2 {
        return Err(ChunkError::InvalidConfig(
            "chunk_matrix_cols requires a 2D tensor".to_string(),
        ));
    }

    let rows = tensor.shape()[0];
    let config = ChunkConfig::new(rows * cols_per_chunk).with_chunk_dim(1);
    let chunked = ChunkedTensor::from_tensor(tensor.clone(), config)?;

    Ok(chunked.chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_config() {
        let config = ChunkConfig::new(1000);
        assert_eq!(config.max_elements, 1000);
        assert_eq!(config.chunk_dim, 0);
    }

    #[test]
    fn test_chunked_tensor_creation() {
        let tensor = BoundedTensor::from_exact(vec![1.0; 100], vec![10, 10]);
        let config = ChunkConfig::new(30);

        let chunked = ChunkedTensor::from_tensor(tensor, config).unwrap();
        assert!(chunked.num_chunks() > 1);
    }

    #[test]
    fn test_chunk_reconstruction() {
        let original = BoundedTensor::from_exact((0..100).map(|i| i as f64).collect(), vec![10, 10]);
        let config = ChunkConfig::new(30);

        let chunked = ChunkedTensor::from_tensor(original.clone(), config).unwrap();
        let reconstructed = chunked.to_tensor().unwrap();

        assert_eq!(original.shape(), reconstructed.shape());
        for (a, b) in original.values().iter().zip(reconstructed.values().iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_weight_loader() {
        let tensor = BoundedTensor::from_exact(vec![1.0; 100], vec![10, 10]);
        let config = ChunkConfig::new(30);

        let mut loader = ChunkedWeightLoader::new(config);
        let chunked = loader.load(&tensor).unwrap();

        assert!(loader.stats().chunks_loaded > 0);
        assert!(chunked.num_chunks() > 1);
    }

    #[test]
    fn test_chunk_matrix_rows() {
        let tensor = BoundedTensor::from_exact((0..100).map(|i| i as f64).collect(), vec![10, 10]);

        let chunks = chunk_matrix_rows(&tensor, 3).unwrap();
        assert!(chunks.len() >= 3);

        // Verify chunk shapes
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.shape()[1], 10); // All chunks should have 10 columns
            if !chunk.is_last {
                assert!(chunk.shape()[0] <= 3); // Non-last chunks should have at most 3 rows
            }
        }
    }

    #[test]
    fn test_loader_cache() {
        let tensor = BoundedTensor::from_exact(vec![1.0; 100], vec![10, 10]);
        let config = ChunkConfig::new(20);

        let mut loader = ChunkedWeightLoader::with_cache_size(config, 2);
        let chunked = loader.load(&tensor).unwrap();

        // Load same chunk twice - second should be cache hit
        let _ = loader.load_chunk(&chunked, 0).unwrap();
        let _ = loader.load_chunk(&chunked, 0).unwrap();

        assert_eq!(loader.stats().cache_hits, 1);
        assert_eq!(loader.stats().cache_misses, 1);
    }
}
