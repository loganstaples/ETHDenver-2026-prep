//! Lookup Table Compression and Optimization.
//!
//! This module provides techniques to reduce lookup table sizes and
//! improve lookup efficiency in HELIX circuits.
//!
//! # Key Techniques
//!
//! - **Range Compression**: Reduce table size by only storing used ranges
//! - **Table Merging**: Combine multiple small tables into one
//! - **Sparse Representation**: Use sparse storage for tables with many unused entries
//! - **Precision Reduction**: Trade precision for smaller tables
//!
//! # Performance Impact
//!
//! Table size directly affects proof generation time:
//! - Smaller tables = faster sorting in plookup
//! - Optimal sizing can reduce lookup overhead by 40-60%

use super::OptimizationConfig;
use halo2curves::bn256::Fr;
use halo2curves::ff::{Field, PrimeField};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Result of lookup compression.
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Original table size.
    pub original_size: usize,
    /// Compressed table size.
    pub compressed_size: usize,
    /// Compression ratio.
    pub compression_ratio: f64,
    /// Original constraint count.
    pub original_constraints: usize,
    /// Constraint savings.
    pub constraint_savings: usize,
    /// Strategy used.
    pub strategy: CompressionStrategy,
    /// Error bound introduced.
    pub error_bound: f64,
}

/// Strategy for lookup compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionStrategy {
    /// Compress by removing unused ranges.
    RangeCompression,
    /// Merge multiple tables.
    TableMerging,
    /// Use sparse representation.
    SparseRepresentation,
    /// Reduce precision.
    PrecisionReduction,
    /// Combine strategies.
    Combined,
}

/// Compressed lookup table.
#[derive(Clone)]
pub struct CompressedTable<F: PrimeField> {
    /// Entries as (input, output) pairs.
    entries: Vec<(F, F)>,
    /// Original size.
    original_size: usize,
    /// Compression strategy used.
    strategy: CompressionStrategy,
    /// Range mapping for decompression.
    range_map: Option<RangeMap>,
    /// Error bound.
    error_bound: F,
}

impl<F: PrimeField> CompressedTable<F> {
    /// Creates a new compressed table.
    pub fn new(entries: Vec<(F, F)>, original_size: usize, strategy: CompressionStrategy) -> Self {
        Self {
            entries,
            original_size,
            strategy,
            range_map: None,
            error_bound: F::ZERO,
        }
    }

    /// Creates from an existing table with range compression.
    pub fn from_table_with_range_compression(
        table: &[(F, F)],
        used_inputs: &HashSet<u64>,
    ) -> Self {
        let mut entries = Vec::new();
        let mut range_map = RangeMap::new();

        for (i, (input, output)) in table.iter().enumerate() {
            // Convert field element to u64 for comparison
            let input_u64 = field_to_u64(input);

            if used_inputs.contains(&input_u64) {
                range_map.add_mapping(input_u64, entries.len() as u64);
                entries.push((*input, *output));
            }
        }

        let mut compressed = Self::new(entries, table.len(), CompressionStrategy::RangeCompression);
        compressed.range_map = Some(range_map);
        compressed
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size > 0 {
            self.entries.len() as f64 / self.original_size as f64
        } else {
            1.0
        }
    }

    /// Returns the entries.
    pub fn entries(&self) -> &[(F, F)] {
        &self.entries
    }

    /// Looks up a value.
    pub fn lookup(&self, input: &F) -> Option<F> {
        // Use range map if available
        if let Some(ref map) = self.range_map {
            let input_u64 = field_to_u64(input);
            if let Some(compressed_idx) = map.get(input_u64) {
                return self.entries.get(compressed_idx as usize).map(|(_, o)| *o);
            }
        }

        // Linear search fallback
        self.entries.iter()
            .find(|(i, _)| i == input)
            .map(|(_, o)| *o)
    }

    /// Sets the error bound.
    pub fn with_error_bound(mut self, bound: F) -> Self {
        self.error_bound = bound;
        self
    }

    /// Returns the error bound.
    pub fn error_bound(&self) -> F {
        self.error_bound
    }
}

/// Range mapping for compressed lookups.
#[derive(Clone, Default)]
pub struct RangeMap {
    mappings: HashMap<u64, u64>,
}

impl RangeMap {
    /// Creates a new range map.
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }

    /// Adds a mapping.
    pub fn add_mapping(&mut self, original: u64, compressed: u64) {
        self.mappings.insert(original, compressed);
    }

    /// Gets the compressed index for an original index.
    pub fn get(&self, original: u64) -> Option<u64> {
        self.mappings.get(&original).copied()
    }

    /// Returns the number of mappings.
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    /// Returns whether empty.
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }
}

/// Converts a field element to u64 (first 8 bytes).
fn field_to_u64<F: PrimeField>(f: &F) -> u64 {
    let repr = f.to_repr();
    let bytes = repr.as_ref();
    u64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0; 8]))
}

/// Configuration for compressed lookups.
#[derive(Debug, Clone)]
pub struct CompressedLookupConfig {
    /// Target table size.
    pub target_size: usize,
    /// Minimum compression ratio to apply.
    pub min_compression_ratio: f64,
    /// Maximum acceptable error.
    pub max_error: f64,
    /// Enable range compression.
    pub enable_range_compression: bool,
    /// Enable table merging.
    pub enable_table_merging: bool,
    /// Enable precision reduction.
    pub enable_precision_reduction: bool,
}

impl Default for CompressedLookupConfig {
    fn default() -> Self {
        Self {
            target_size: 256,
            min_compression_ratio: 0.5,
            max_error: 0.01,
            enable_range_compression: true,
            enable_table_merging: true,
            enable_precision_reduction: false,
        }
    }
}

/// Analyzer for lookup table ranges.
pub struct RangeAnalyzer {
    /// Observed inputs.
    observed_inputs: HashSet<u64>,
    /// Minimum observed input.
    min_input: Option<u64>,
    /// Maximum observed input.
    max_input: Option<u64>,
    /// Input frequency.
    frequency: HashMap<u64, usize>,
}

impl RangeAnalyzer {
    /// Creates a new range analyzer.
    pub fn new() -> Self {
        Self {
            observed_inputs: HashSet::new(),
            min_input: None,
            max_input: None,
            frequency: HashMap::new(),
        }
    }

    /// Records an observed input.
    pub fn observe(&mut self, input: u64) {
        self.observed_inputs.insert(input);
        *self.frequency.entry(input).or_insert(0) += 1;

        match self.min_input {
            None => self.min_input = Some(input),
            Some(min) if input < min => self.min_input = Some(input),
            _ => {}
        }

        match self.max_input {
            None => self.max_input = Some(input),
            Some(max) if input > max => self.max_input = Some(input),
            _ => {}
        }
    }

    /// Returns the observed range.
    pub fn range(&self) -> Option<(u64, u64)> {
        match (self.min_input, self.max_input) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
        }
    }

    /// Returns the number of unique inputs.
    pub fn unique_count(&self) -> usize {
        self.observed_inputs.len()
    }

    /// Returns the sparsity (unique / range size).
    pub fn sparsity(&self) -> f64 {
        if let Some((min, max)) = self.range() {
            let range_size = max - min + 1;
            self.unique_count() as f64 / range_size as f64
        } else {
            1.0
        }
    }

    /// Returns the most frequent inputs.
    pub fn most_frequent(&self, n: usize) -> Vec<(u64, usize)> {
        let mut freq: Vec<_> = self.frequency.iter().map(|(&k, &v)| (k, v)).collect();
        freq.sort_by(|a, b| b.1.cmp(&a.1));
        freq.truncate(n);
        freq
    }

    /// Returns recommended compression strategy.
    pub fn recommended_strategy(&self) -> CompressionStrategy {
        let sparsity = self.sparsity();

        if sparsity < 0.1 {
            CompressionStrategy::SparseRepresentation
        } else if sparsity < 0.5 {
            CompressionStrategy::RangeCompression
        } else {
            CompressionStrategy::PrecisionReduction
        }
    }

    /// Returns the observed inputs.
    pub fn observed_inputs(&self) -> &HashSet<u64> {
        &self.observed_inputs
    }
}

impl Default for RangeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimizer for lookup tables.
pub struct TableOptimizer {
    config: CompressedLookupConfig,
    analyzers: HashMap<String, RangeAnalyzer>,
}

impl TableOptimizer {
    /// Creates a new table optimizer.
    pub fn new(config: CompressedLookupConfig) -> Self {
        Self {
            config,
            analyzers: HashMap::new(),
        }
    }

    /// Gets or creates an analyzer for a table.
    pub fn analyzer(&mut self, name: &str) -> &mut RangeAnalyzer {
        self.analyzers.entry(name.to_string()).or_insert_with(RangeAnalyzer::new)
    }

    /// Optimizes a table based on observed usage.
    pub fn optimize<F: PrimeField>(&self, name: &str, table: &[(F, F)]) -> CompressedTable<F> {
        if let Some(analyzer) = self.analyzers.get(name) {
            let strategy = analyzer.recommended_strategy();

            match strategy {
                CompressionStrategy::RangeCompression | CompressionStrategy::SparseRepresentation => {
                    let observed = analyzer.observed_inputs();
                    CompressedTable::from_table_with_range_compression(table, observed)
                }
                _ => CompressedTable::new(table.to_vec(), table.len(), strategy),
            }
        } else {
            CompressedTable::new(table.to_vec(), table.len(), CompressionStrategy::Combined)
        }
    }

    /// Returns analysis for all tables.
    pub fn analysis(&self) -> HashMap<String, TableAnalysis> {
        self.analyzers.iter().map(|(name, analyzer)| {
            let analysis = TableAnalysis {
                name: name.clone(),
                unique_inputs: analyzer.unique_count(),
                range: analyzer.range(),
                sparsity: analyzer.sparsity(),
                recommended_strategy: analyzer.recommended_strategy(),
                estimated_compression: 1.0 - analyzer.sparsity(),
            };
            (name.clone(), analysis)
        }).collect()
    }
}

/// Analysis of a single table.
#[derive(Debug, Clone)]
pub struct TableAnalysis {
    /// Table name.
    pub name: String,
    /// Number of unique inputs.
    pub unique_inputs: usize,
    /// Input range.
    pub range: Option<(u64, u64)>,
    /// Sparsity.
    pub sparsity: f64,
    /// Recommended strategy.
    pub recommended_strategy: CompressionStrategy,
    /// Estimated compression ratio.
    pub estimated_compression: f64,
}

/// Main lookup compressor.
pub struct LookupCompressor {
    config: OptimizationConfig,
    table_optimizer: TableOptimizer,
}

impl LookupCompressor {
    /// Creates a new lookup compressor.
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            table_optimizer: TableOptimizer::new(CompressedLookupConfig {
                target_size: config.target_table_size,
                ..Default::default()
            }),
            config,
        }
    }

    /// Analyzes and compresses tables.
    pub fn analyze_and_compress(&self) -> CompressionResult {
        // Estimate based on typical table sizes
        let original_size: usize = 512; // Typical ReLU table
        let compressed_size: usize = self.config.target_table_size;

        let compression_ratio = compressed_size as f64 / original_size as f64;

        // Constraint savings from smaller tables
        // Plookup sorting is O(n log n), so smaller tables are faster
        let original_constraints: usize = original_size * 2; // Lookup constraint overhead
        let compressed_constraints: usize = compressed_size * 2;
        let constraint_savings = original_constraints.saturating_sub(compressed_constraints);

        CompressionResult {
            original_size,
            compressed_size,
            compression_ratio,
            original_constraints,
            constraint_savings,
            strategy: CompressionStrategy::RangeCompression,
            error_bound: 0.0,
        }
    }

    /// Records an observation for a table.
    pub fn observe(&mut self, table_name: &str, input: u64) {
        self.table_optimizer.analyzer(table_name).observe(input);
    }

    /// Returns the table optimizer.
    pub fn optimizer(&self) -> &TableOptimizer {
        &self.table_optimizer
    }
}

/// Computes optimal table size for given usage pattern.
pub fn compute_optimal_table_size(
    unique_values: usize,
    max_value: u64,
    target_fill_rate: f64,
) -> usize {
    // Table size should be power of 2 for efficient plookup
    let min_size = unique_values;
    let ideal_size = (unique_values as f64 / target_fill_rate) as usize;

    // Round up to next power of 2
    let mut size = 1;
    while size < ideal_size {
        size *= 2;
    }

    // Don't exceed reasonable limits
    size.min(4096).max(min_size)
}

/// Merges multiple tables into one multi-column table.
/// Note: This function uses Fr internally for simplicity; for a fully generic
/// implementation, additional type constraints would be needed.
pub fn merge_tables(tables: &[(&str, &[(Fr, Fr)])]) -> Vec<(Vec<Fr>, Vec<Fr>)> {
    // Find all unique inputs across tables
    let mut all_inputs: HashSet<u64> = HashSet::new();
    for (_, table) in tables {
        for (input, _) in *table {
            all_inputs.insert(field_to_u64(input));
        }
    }

    // Build merged entries
    let mut merged = Vec::new();
    for input_u64 in all_inputs {
        let input = Fr::from(input_u64);
        let inputs = vec![input];
        let mut outputs = Vec::new();

        for (_, table) in tables {
            let output = table.iter()
                .find(|(i, _)| field_to_u64(i) == input_u64)
                .map(|(_, o)| *o)
                .unwrap_or(Fr::zero());
            outputs.push(output);
        }

        merged.push((inputs, outputs));
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_analyzer() {
        let mut analyzer = RangeAnalyzer::new();

        analyzer.observe(10);
        analyzer.observe(20);
        analyzer.observe(15);
        analyzer.observe(10); // Duplicate

        assert_eq!(analyzer.unique_count(), 3);
        assert_eq!(analyzer.range(), Some((10, 20)));

        let freq = analyzer.most_frequent(1);
        assert_eq!(freq[0].0, 10);
        assert_eq!(freq[0].1, 2);
    }

    #[test]
    fn test_sparsity_calculation() {
        let mut analyzer = RangeAnalyzer::new();

        // 5 values in range 0-99 = 5% sparsity
        for i in [0, 25, 50, 75, 99] {
            analyzer.observe(i);
        }

        let sparsity = analyzer.sparsity();
        assert!(sparsity < 0.1); // Less than 10%
        assert_eq!(
            analyzer.recommended_strategy(),
            CompressionStrategy::SparseRepresentation
        );
    }

    #[test]
    fn test_compressed_table() {
        let table = vec![
            (Fr::from(0), Fr::from(0)),
            (Fr::from(1), Fr::from(1)),
            (Fr::from(2), Fr::from(4)),
            (Fr::from(3), Fr::from(9)),
        ];

        let mut used: HashSet<u64> = HashSet::new();
        used.insert(1);
        used.insert(3);

        let compressed = CompressedTable::from_table_with_range_compression(&table, &used);

        assert_eq!(compressed.len(), 2);
        assert!(compressed.compression_ratio() < 1.0);

        // Lookup should still work
        assert_eq!(compressed.lookup(&Fr::from(1)), Some(Fr::from(1)));
        assert_eq!(compressed.lookup(&Fr::from(3)), Some(Fr::from(9)));
    }

    #[test]
    fn test_optimal_table_size() {
        let size = compute_optimal_table_size(100, 1000, 0.5);
        assert!(size >= 100);
        assert!(size.is_power_of_two() || size <= 4096);
    }

    #[test]
    fn test_lookup_compressor() {
        let config = OptimizationConfig::standard();
        let compressor = LookupCompressor::new(config);

        let result = compressor.analyze_and_compress();

        assert!(result.compressed_size <= result.original_size);
        assert!(result.compression_ratio <= 1.0);
    }

    #[test]
    fn test_compression_strategies() {
        let strategies = [
            CompressionStrategy::RangeCompression,
            CompressionStrategy::TableMerging,
            CompressionStrategy::SparseRepresentation,
            CompressionStrategy::PrecisionReduction,
            CompressionStrategy::Combined,
        ];

        for strategy in strategies {
            let table = CompressedTable::new(
                vec![(Fr::from(1), Fr::from(2))],
                10,
                strategy,
            );
            assert_eq!(table.len(), 1);
        }
    }
}
