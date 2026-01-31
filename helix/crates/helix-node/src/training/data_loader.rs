//! Data Loading and Batching for Training.
//!
//! Provides streaming data loading with bounded memory usage,
//! tokenization pipeline, and batch construction for text training.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::io::{BufRead, BufReader};
use std::fs::File;
use serde::{Deserialize, Serialize};
use helix_core::types::BoundedTensor;

/// Token ID type.
pub type TokenId = u32;

/// A batch of training data.
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// Input token IDs [batch_size, seq_len].
    pub input_ids: Vec<Vec<TokenId>>,
    /// Attention mask [batch_size, seq_len].
    pub attention_mask: Vec<Vec<u8>>,
    /// Labels for next-token prediction [batch_size, seq_len].
    pub labels: Vec<Vec<TokenId>>,
    /// Batch index within the epoch.
    pub batch_idx: usize,
    /// Epoch number.
    pub epoch: usize,
}

impl TrainingBatch {
    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.input_ids.len()
    }

    /// Returns the sequence length.
    pub fn seq_len(&self) -> usize {
        self.input_ids.first().map_or(0, |ids| ids.len())
    }

    /// Converts input IDs to a BoundedTensor.
    pub fn input_tensor(&self) -> BoundedTensor {
        let batch_size = self.batch_size();
        let seq_len = self.seq_len();
        
        let data: Vec<f64> = self.input_ids
            .iter()
            .flat_map(|seq| seq.iter().map(|&id| id as f64))
            .collect();
        
        BoundedTensor::from_exact(data, vec![batch_size, seq_len])
    }

    /// Converts labels to a BoundedTensor.
    pub fn labels_tensor(&self) -> BoundedTensor {
        let batch_size = self.batch_size();
        let seq_len = self.seq_len();
        
        let data: Vec<f64> = self.labels
            .iter()
            .flat_map(|seq| seq.iter().map(|&id| id as f64))
            .collect();
        
        BoundedTensor::from_exact(data, vec![batch_size, seq_len])
    }
}

/// Configuration for data loading.
#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    /// Batch size.
    pub batch_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Number of batches to prefetch.
    pub prefetch_count: usize,
    /// Whether to shuffle data.
    pub shuffle: bool,
    /// Random seed for shuffling.
    pub seed: u64,
    /// Whether to drop the last incomplete batch.
    pub drop_last: bool,
    /// Padding token ID.
    pub pad_token_id: TokenId,
    /// End of sequence token ID.
    pub eos_token_id: TokenId,
    /// Start of sequence token ID.
    pub bos_token_id: TokenId,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            max_seq_len: 512,
            prefetch_count: 4,
            shuffle: true,
            seed: 42,
            drop_last: true,
            pad_token_id: 0,
            eos_token_id: 2,
            bos_token_id: 1,
        }
    }
}

/// Simple BPE-like tokenizer (simplified for demo).
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    /// Vocabulary mapping (token string -> ID).
    vocab: std::collections::HashMap<String, TokenId>,
    /// Reverse vocabulary (ID -> token string).
    reverse_vocab: std::collections::HashMap<TokenId, String>,
    /// Special tokens.
    special_tokens: SpecialTokens,
    /// Vocabulary size.
    vocab_size: usize,
}

/// Special tokens configuration.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub pad: (String, TokenId),
    pub bos: (String, TokenId),
    pub eos: (String, TokenId),
    pub unk: (String, TokenId),
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            pad: ("<pad>".to_string(), 0),
            bos: ("<bos>".to_string(), 1),
            eos: ("<eos>".to_string(), 2),
            unk: ("<unk>".to_string(), 3),
        }
    }
}

impl SimpleTokenizer {
    /// Creates a tokenizer with a basic character-level vocabulary.
    pub fn basic(vocab_size: usize) -> Self {
        let special = SpecialTokens::default();
        let mut vocab = std::collections::HashMap::new();
        let mut reverse_vocab = std::collections::HashMap::new();

        // Add special tokens
        vocab.insert(special.pad.0.clone(), special.pad.1);
        vocab.insert(special.bos.0.clone(), special.bos.1);
        vocab.insert(special.eos.0.clone(), special.eos.1);
        vocab.insert(special.unk.0.clone(), special.unk.1);

        reverse_vocab.insert(special.pad.1, special.pad.0.clone());
        reverse_vocab.insert(special.bos.1, special.bos.0.clone());
        reverse_vocab.insert(special.eos.1, special.eos.0.clone());
        reverse_vocab.insert(special.unk.1, special.unk.0.clone());

        // Add printable ASCII characters
        let mut next_id = 4;
        for c in 32u8..127u8 {
            if next_id >= vocab_size as TokenId {
                break;
            }
            let s = (c as char).to_string();
            vocab.insert(s.clone(), next_id);
            reverse_vocab.insert(next_id, s);
            next_id += 1;
        }

        // Add common byte sequences
        let common_sequences = ["th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", 
                               "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
                               "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
                               "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea",
                               "ra", "ce", "li", "ch", "ll", "be", "ma", "si", "om", "ur"];

        for seq in &common_sequences {
            if (next_id as usize) >= vocab_size {
                break;
            }
            if !vocab.contains_key(*seq) {
                vocab.insert(seq.to_string(), next_id);
                reverse_vocab.insert(next_id, seq.to_string());
                next_id += 1;
            }
        }

        Self {
            vocab,
            reverse_vocab,
            special_tokens: special,
            vocab_size: next_id as usize,
        }
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Encodes text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        let mut tokens = vec![self.special_tokens.bos.1];
        
        // Simple character-level encoding with basic merging
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            // Try to match 2-char sequences first
            if i + 1 < chars.len() {
                let two_char: String = chars[i..i+2].iter().collect();
                if let Some(&id) = self.vocab.get(&two_char) {
                    tokens.push(id);
                    i += 2;
                    continue;
                }
            }
            
            // Fall back to single character
            let c = chars[i].to_string();
            let id = self.vocab.get(&c)
                .copied()
                .unwrap_or(self.special_tokens.unk.1);
            tokens.push(id);
            i += 1;
        }
        
        tokens.push(self.special_tokens.eos.1);
        tokens
    }

    /// Decodes token IDs to text.
    pub fn decode(&self, tokens: &[TokenId]) -> String {
        tokens
            .iter()
            .filter_map(|id| self.reverse_vocab.get(id))
            .filter(|s| !s.starts_with('<'))
            .cloned()
            .collect()
    }

    /// Pads a sequence to the target length.
    pub fn pad(&self, tokens: &[TokenId], target_len: usize) -> Vec<TokenId> {
        if tokens.len() >= target_len {
            tokens[..target_len].to_vec()
        } else {
            let mut padded = tokens.to_vec();
            padded.resize(target_len, self.special_tokens.pad.1);
            padded
        }
    }

    /// Creates an attention mask for a sequence.
    pub fn attention_mask(&self, tokens: &[TokenId]) -> Vec<u8> {
        tokens
            .iter()
            .map(|&id| if id == self.special_tokens.pad.1 { 0 } else { 1 })
            .collect()
    }
}

/// Data shard for distributed training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataShard {
    /// Shard index.
    pub shard_idx: usize,
    /// Total number of shards.
    pub num_shards: usize,
    /// File path or range for this shard.
    pub source: ShardSource,
}

/// Source of shard data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardSource {
    /// File path.
    File(PathBuf),
    /// Range within a file (start_line, end_line).
    FileRange { path: PathBuf, start_line: usize, end_line: usize },
    /// IPFS CID with range.
    Ipfs { cid: String, offset: usize, length: usize },
    /// In-memory data (for testing).
    InMemory(Vec<String>),
}

/// Streaming data loader with bounded memory.
pub struct DataLoader {
    /// Configuration.
    config: DataLoaderConfig,
    /// Tokenizer.
    tokenizer: SimpleTokenizer,
    /// Data shard.
    shard: DataShard,
    /// Prefetched batches.
    prefetch_queue: VecDeque<TrainingBatch>,
    /// Current epoch.
    current_epoch: usize,
    /// Current batch index within epoch.
    current_batch_idx: usize,
    /// Line buffer for streaming.
    line_buffer: VecDeque<String>,
    /// Whether we've exhausted the data source.
    exhausted: bool,
}

impl DataLoader {
    /// Creates a new data loader.
    pub fn new(
        config: DataLoaderConfig,
        tokenizer: SimpleTokenizer,
        shard: DataShard,
    ) -> Self {
        Self {
            config,
            tokenizer,
            shard,
            prefetch_queue: VecDeque::new(),
            current_epoch: 0,
            current_batch_idx: 0,
            line_buffer: VecDeque::new(),
            exhausted: false,
        }
    }

    /// Creates a data loader from a text file.
    pub fn from_file(
        path: impl AsRef<Path>,
        config: DataLoaderConfig,
        tokenizer: SimpleTokenizer,
        shard_idx: usize,
        num_shards: usize,
    ) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        // Count lines to determine shard boundaries
        let file = File::open(&path)?;
        let line_count = BufReader::new(file).lines().count();
        
        let lines_per_shard = line_count / num_shards;
        let start_line = shard_idx * lines_per_shard;
        let end_line = if shard_idx == num_shards - 1 {
            line_count
        } else {
            (shard_idx + 1) * lines_per_shard
        };

        let shard = DataShard {
            shard_idx,
            num_shards,
            source: ShardSource::FileRange {
                path,
                start_line,
                end_line,
            },
        };

        Ok(Self::new(config, tokenizer, shard))
    }

    /// Creates a data loader from in-memory data (for testing).
    pub fn from_memory(
        data: Vec<String>,
        config: DataLoaderConfig,
        tokenizer: SimpleTokenizer,
    ) -> Self {
        let shard = DataShard {
            shard_idx: 0,
            num_shards: 1,
            source: ShardSource::InMemory(data),
        };
        Self::new(config, tokenizer, shard)
    }

    /// Returns the next batch, or None if exhausted.
    pub fn next_batch(&mut self) -> Option<TrainingBatch> {
        // Try to get from prefetch queue
        if let Some(batch) = self.prefetch_queue.pop_front() {
            // Refill prefetch queue
            self.prefetch_batches();
            return Some(batch);
        }

        // Try to build a new batch
        self.build_batch()
    }

    /// Resets the loader for a new epoch.
    pub fn reset_epoch(&mut self) {
        self.current_epoch += 1;
        self.current_batch_idx = 0;
        self.exhausted = false;
        self.line_buffer.clear();
        self.prefetch_queue.clear();
    }

    /// Returns the current epoch.
    pub fn epoch(&self) -> usize {
        self.current_epoch
    }

    /// Returns the number of batches produced so far in this epoch.
    pub fn batch_count(&self) -> usize {
        self.current_batch_idx
    }

    /// Prefetches batches into the queue.
    fn prefetch_batches(&mut self) {
        while self.prefetch_queue.len() < self.config.prefetch_count {
            match self.build_batch() {
                Some(batch) => self.prefetch_queue.push_back(batch),
                None => break,
            }
        }
    }

    /// Builds a single batch.
    fn build_batch(&mut self) -> Option<TrainingBatch> {
        if self.exhausted {
            return None;
        }

        // Ensure we have enough lines
        self.fill_line_buffer();

        let mut input_ids = Vec::with_capacity(self.config.batch_size);
        let mut attention_mask = Vec::with_capacity(self.config.batch_size);
        let mut labels = Vec::with_capacity(self.config.batch_size);

        while input_ids.len() < self.config.batch_size {
            match self.line_buffer.pop_front() {
                Some(line) => {
                    if line.trim().is_empty() {
                        continue;
                    }

                    let tokens = self.tokenizer.encode(&line);
                    
                    // Truncate if too long
                    let tokens = if tokens.len() > self.config.max_seq_len {
                        tokens[..self.config.max_seq_len].to_vec()
                    } else {
                        tokens
                    };

                    // Only include sequences with enough tokens
                    if tokens.len() >= 4 {
                        let padded = self.tokenizer.pad(&tokens, self.config.max_seq_len);
                        let mask = self.tokenizer.attention_mask(&padded);
                        
                        // Labels are shifted input IDs
                        let mut label = padded[1..].to_vec();
                        label.push(self.config.pad_token_id);

                        input_ids.push(padded);
                        attention_mask.push(mask);
                        labels.push(label);
                    }
                }
                None => {
                    // Try to get more lines
                    self.fill_line_buffer();
                    if self.line_buffer.is_empty() {
                        self.exhausted = true;
                        break;
                    }
                }
            }
        }

        // Check if we have enough for a batch
        if input_ids.is_empty() {
            return None;
        }

        if input_ids.len() < self.config.batch_size && self.config.drop_last {
            return None;
        }

        let batch = TrainingBatch {
            input_ids,
            attention_mask,
            labels,
            batch_idx: self.current_batch_idx,
            epoch: self.current_epoch,
        };

        self.current_batch_idx += 1;
        Some(batch)
    }

    /// Fills the line buffer from the data source.
    fn fill_line_buffer(&mut self) {
        // Target: have at least 2x batch_size lines ready
        let target_lines = self.config.batch_size * 2;
        
        if self.line_buffer.len() >= target_lines {
            return;
        }

        let lines_to_fetch = target_lines - self.line_buffer.len();
        
        match &self.shard.source {
            ShardSource::InMemory(data) => {
                // For in-memory, we just cycle through
                for line in data.iter().take(lines_to_fetch) {
                    self.line_buffer.push_back(line.clone());
                }
            }
            ShardSource::File(path) => {
                if let Ok(file) = File::open(path) {
                    let reader = BufReader::new(file);
                    for line in reader.lines().take(lines_to_fetch).flatten() {
                        self.line_buffer.push_back(line);
                    }
                }
            }
            ShardSource::FileRange { path, start_line, end_line } => {
                if let Ok(file) = File::open(path) {
                    let reader = BufReader::new(file);
                    for line in reader.lines()
                        .skip(*start_line)
                        .take(end_line - start_line)
                        .take(lines_to_fetch)
                        .flatten()
                    {
                        self.line_buffer.push_back(line);
                    }
                }
            }
            ShardSource::Ipfs { cid: _, offset: _, length: _ } => {
                // IPFS loading would be async - for now, mark as exhausted
                self.exhausted = true;
            }
        }
    }
}

/// Iterator implementation for DataLoader.
impl Iterator for DataLoader {
    type Item = TrainingBatch;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer() {
        let tokenizer = SimpleTokenizer::basic(256);
        
        let text = "Hello world";
        let tokens = tokenizer.encode(text);
        
        // Should have BOS, content, EOS
        assert!(tokens.len() >= 3);
        assert_eq!(tokens[0], 1); // BOS
        assert_eq!(*tokens.last().unwrap(), 2); // EOS

        let decoded = tokenizer.decode(&tokens);
        assert!(decoded.contains("llo")); // Some content preserved
    }

    #[test]
    fn test_data_loader_memory() {
        let data = vec![
            "Hello world, this is a test.".to_string(),
            "Another line of text for training.".to_string(),
            "The quick brown fox jumps.".to_string(),
            "More training data here.".to_string(),
        ];

        let config = DataLoaderConfig {
            batch_size: 2,
            max_seq_len: 64,
            drop_last: false,
            ..Default::default()
        };

        let tokenizer = SimpleTokenizer::basic(256);
        let mut loader = DataLoader::from_memory(data, config, tokenizer);

        let batch = loader.next_batch().unwrap();
        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.seq_len(), 64);
    }

    #[test]
    fn test_training_batch_tensor() {
        let batch = TrainingBatch {
            input_ids: vec![vec![1, 5, 6, 7, 2], vec![1, 8, 9, 10, 2]],
            attention_mask: vec![vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1]],
            labels: vec![vec![5, 6, 7, 2, 0], vec![8, 9, 10, 2, 0]],
            batch_idx: 0,
            epoch: 0,
        };

        let input_tensor = batch.input_tensor();
        assert_eq!(input_tensor.shape(), &vec![2, 5]);

        let labels_tensor = batch.labels_tensor();
        assert_eq!(labels_tensor.shape(), &vec![2, 5]);
    }
}
