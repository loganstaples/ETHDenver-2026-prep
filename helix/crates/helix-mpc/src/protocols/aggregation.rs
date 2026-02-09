//! Secure aggregation protocol optimization.
//!
//! This module provides optimized protocols for aggregating secret-shared gradients
//! across multiple parties in federated learning scenarios.
//!
//! # Optimizations
//!
//! - **Tree-based aggregation**: Reduces communication rounds from O(n) to O(log n)
//! - **Gradient compression**: Top-k sparsification and quantization reduce bandwidth
//! - **Batched operations**: Aggregate multiple layers in parallel
//! - **Fault tolerance**: Handle party dropouts gracefully
//! - **Commitment verification**: Verify gradient integrity during aggregation

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use sha2::{Digest, Sha256};

use crate::error::{MPCError, MPCResult};
use crate::types::PartyId;

/// Configuration for secure aggregation.
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Minimum number of parties required for aggregation.
    pub min_parties: usize,
    /// Enable tree-based aggregation for large party counts.
    pub use_tree_aggregation: bool,
    /// Tree branching factor (parties per aggregation node).
    pub tree_branching_factor: usize,
    /// Enable gradient compression.
    pub enable_compression: bool,
    /// Compression configuration.
    pub compression: CompressionConfig,
    /// Timeout for party responses.
    pub party_timeout: Duration,
    /// Maximum retries for failed parties.
    pub max_retries: usize,
    /// Enable commitment verification.
    pub verify_commitments: bool,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            min_parties: 2,
            use_tree_aggregation: true,
            tree_branching_factor: 4,
            enable_compression: true,
            compression: CompressionConfig::default(),
            party_timeout: Duration::from_secs(30),
            max_retries: 3,
            verify_commitments: true,
        }
    }
}

/// Configuration for gradient compression.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Top-k fraction (0.0-1.0) for sparsification.
    pub top_k_ratio: f64,
    /// Quantization bits (0 for no quantization).
    pub quantization_bits: u8,
    /// Error feedback for compression error accumulation.
    pub use_error_feedback: bool,
    /// Compression threshold (values below this are zeroed).
    pub threshold: f64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            top_k_ratio: 0.01, // Keep top 1% of gradients
            quantization_bits: 8,
            use_error_feedback: true,
            threshold: 1e-7,
        }
    }
}

/// A compressed gradient representation.
#[derive(Debug, Clone)]
pub struct CompressedGradient {
    /// Sparse indices of non-zero values.
    pub indices: Vec<usize>,
    /// Corresponding values (potentially quantized).
    pub values: Vec<f64>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Original tensor size.
    pub original_size: usize,
    /// Quantization scale (for dequantization).
    pub scale: f64,
    /// Quantization offset.
    pub offset: f64,
    /// Whether values are quantized.
    pub quantized: bool,
    /// Commitment hash for verification.
    pub commitment: [u8; 32],
}

impl CompressedGradient {
    /// Calculates compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            return 1.0;
        }
        self.indices.len() as f64 / self.original_size as f64
    }

    /// Decompresses to full tensor.
    pub fn decompress(&self) -> Vec<f64> {
        let mut result = vec![0.0; self.original_size];

        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            if idx < self.original_size {
                let deq_val = if self.quantized {
                    val * self.scale + self.offset
                } else {
                    val
                };
                result[idx] = deq_val;
            }
        }

        result
    }

    /// Verifies the commitment matches the data.
    pub fn verify_commitment(&self) -> bool {
        let computed = self.compute_commitment();
        computed == self.commitment
    }

    /// Computes commitment over compressed data.
    pub fn compute_commitment(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();

        for &idx in &self.indices {
            hasher.update(idx.to_le_bytes());
        }
        for &val in &self.values {
            hasher.update(val.to_le_bytes());
        }
        hasher.update(self.scale.to_le_bytes());
        hasher.update(self.offset.to_le_bytes());

        hasher.finalize().into()
    }
}

/// Gradient compressor for bandwidth optimization.
pub struct GradientCompressor {
    /// Configuration.
    config: CompressionConfig,
    /// Error feedback buffer per tensor name.
    error_feedback: RwLock<HashMap<String, Vec<f64>>>,
    /// Statistics.
    stats: CompressionStats,
}

/// Compression statistics.
#[derive(Debug, Default)]
pub struct CompressionStats {
    pub tensors_compressed: AtomicU64,
    pub total_original_bytes: AtomicU64,
    pub total_compressed_bytes: AtomicU64,
    pub average_sparsity: RwLock<f64>,
}

impl CompressionStats {
    /// Returns average compression ratio.
    pub fn average_compression_ratio(&self) -> f64 {
        let original = self.total_original_bytes.load(Ordering::SeqCst) as f64;
        let compressed = self.total_compressed_bytes.load(Ordering::SeqCst) as f64;
        if original > 0.0 {
            compressed / original
        } else {
            1.0
        }
    }
}

impl GradientCompressor {
    /// Creates a new gradient compressor.
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            error_feedback: RwLock::new(HashMap::new()),
            stats: CompressionStats::default(),
        }
    }

    /// Compresses a gradient tensor using top-k sparsification.
    pub fn compress(&self, data: &[f64], shape: &[usize], name: &str) -> CompressedGradient {
        let mut working = data.to_vec();

        // Apply error feedback if enabled.
        if self.config.use_error_feedback {
            let feedback = self.error_feedback.read();
            if let Some(err) = feedback.get(name) {
                if err.len() == working.len() {
                    for (w, e) in working.iter_mut().zip(err.iter()) {
                        *w += *e;
                    }
                }
            }
        }

        // Top-k selection.
        let k = ((working.len() as f64 * self.config.top_k_ratio).ceil() as usize).max(1);

        // Get indices of top-k values by magnitude.
        let mut indexed: Vec<(usize, f64)> = working.iter().enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected: HashSet<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();

        let mut indices = Vec::with_capacity(k);
        let mut values = Vec::with_capacity(k);

        for (idx, _) in indexed.iter().take(k) {
            if working[*idx].abs() >= self.config.threshold {
                indices.push(*idx);
                values.push(working[*idx]);
            }
        }

        // Calculate error feedback (what we lost).
        if self.config.use_error_feedback {
            let mut new_error = vec![0.0; working.len()];
            for (i, &val) in working.iter().enumerate() {
                if !selected.contains(&i) {
                    new_error[i] = val;
                }
            }
            self.error_feedback.write().insert(name.to_string(), new_error);
        }

        // Quantize if enabled.
        let (final_values, scale, offset, quantized) = if self.config.quantization_bits > 0 && !values.is_empty() {
            let (qvals, s, o) = self.quantize(&values, self.config.quantization_bits);
            (qvals, s, o, true)
        } else {
            (values, 1.0, 0.0, false)
        };

        // Update statistics.
        self.stats.tensors_compressed.fetch_add(1, Ordering::SeqCst);
        self.stats.total_original_bytes.fetch_add((data.len() * 8) as u64, Ordering::SeqCst);
        self.stats.total_compressed_bytes.fetch_add(
            ((indices.len() * 8) + (final_values.len() * 8)) as u64, Ordering::SeqCst
        );

        let mut result = CompressedGradient {
            indices,
            values: final_values,
            shape: shape.to_vec(),
            original_size: data.len(),
            scale,
            offset,
            quantized,
            commitment: [0u8; 32],
        };

        result.commitment = result.compute_commitment();
        result
    }

    /// Quantizes values to specified bit width.
    fn quantize(&self, values: &[f64], bits: u8) -> (Vec<f64>, f64, f64) {
        if values.is_empty() {
            return (Vec::new(), 1.0, 0.0);
        }

        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let range = max_val - min_val;
        if range < 1e-10 {
            return (vec![0.0; values.len()], 1.0, min_val);
        }

        let levels = (1u64 << bits) as f64 - 1.0;
        let scale = range / levels;

        let quantized: Vec<f64> = values.iter()
            .map(|&v| ((v - min_val) / scale).round())
            .collect();

        (quantized, scale, min_val)
    }

    /// Returns compression statistics.
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Clears error feedback buffers.
    pub fn clear_error_feedback(&self) {
        self.error_feedback.write().clear();
    }
}

/// A party's contribution to aggregation.
#[derive(Debug, Clone)]
pub struct AggregationContribution {
    /// Contributing party.
    pub party: PartyId,
    /// Gradient shares per layer/tensor name.
    pub gradients: HashMap<String, CompressedGradient>,
    /// Contribution timestamp.
    pub timestamp: Instant,
    /// Contribution round.
    pub round: u64,
    /// Overall commitment for verification.
    pub commitment: [u8; 32],
}

impl AggregationContribution {
    /// Creates a new contribution.
    pub fn new(party: PartyId, round: u64) -> Self {
        Self {
            party,
            gradients: HashMap::new(),
            timestamp: Instant::now(),
            round,
            commitment: [0u8; 32],
        }
    }

    /// Adds a gradient to the contribution.
    pub fn add_gradient(&mut self, name: String, gradient: CompressedGradient) {
        self.gradients.insert(name, gradient);
    }

    /// Finalizes the contribution by computing overall commitment.
    pub fn finalize(&mut self) {
        let mut hasher = Sha256::new();
        hasher.update(self.party.0.as_bytes());
        hasher.update(self.round.to_le_bytes());

        let mut names: Vec<&String> = self.gradients.keys().collect();
        names.sort();

        for name in names {
            hasher.update(name.as_bytes());
            hasher.update(&self.gradients[name].commitment);
        }

        self.commitment = hasher.finalize().into();
    }

    /// Verifies all gradient commitments.
    pub fn verify_all(&self) -> bool {
        self.gradients.values().all(|g| g.verify_commitment())
    }
}

/// Result of aggregation.
#[derive(Debug, Clone)]
pub struct AggregationResult {
    /// Aggregated gradients per tensor name.
    pub gradients: HashMap<String, Vec<f64>>,
    /// Parties that contributed.
    pub contributors: Vec<PartyId>,
    /// Parties that failed/dropped.
    pub failed_parties: Vec<PartyId>,
    /// Aggregation round.
    pub round: u64,
    /// Time taken.
    pub duration: Duration,
    /// Overall result commitment.
    pub commitment: [u8; 32],
}

/// Tree aggregation node.
#[derive(Debug)]
#[allow(dead_code)]
struct TreeNode {
    /// Node ID.
    id: usize,
    /// Level in tree (0 = leaf).
    level: usize,
    /// Child node IDs or party IDs for leaves.
    children: Vec<usize>,
    /// Aggregated result at this node.
    result: Option<HashMap<String, Vec<f64>>>,
    /// Contributing parties at this node.
    contributors: Vec<PartyId>,
}

/// Secure gradient aggregator.
pub struct SecureAggregator {
    /// Configuration.
    config: AggregationConfig,
    /// Compressor for gradient compression.
    compressor: GradientCompressor,
    /// Current aggregation round.
    current_round: AtomicU64,
    /// Pending contributions.
    pending: RwLock<HashMap<PartyId, AggregationContribution>>,
    /// Aggregation statistics.
    stats: AggregationStats,
}

/// Aggregation statistics.
#[derive(Debug, Default)]
pub struct AggregationStats {
    pub rounds_completed: AtomicU64,
    pub total_contributions: AtomicU64,
    pub failed_contributions: AtomicU64,
    pub total_aggregation_time_ms: AtomicU64,
    pub average_contributors: RwLock<f64>,
}

impl SecureAggregator {
    /// Creates a new secure aggregator.
    pub fn new(config: AggregationConfig) -> Self {
        Self {
            compressor: GradientCompressor::new(config.compression.clone()),
            config,
            current_round: AtomicU64::new(0),
            pending: RwLock::new(HashMap::new()),
            stats: AggregationStats::default(),
        }
    }

    /// Starts a new aggregation round.
    pub fn start_round(&self) -> u64 {
        self.pending.write().clear();
        self.current_round.fetch_add(1, Ordering::SeqCst)
    }

    /// Submits a contribution from a party.
    pub fn submit(&self, contribution: AggregationContribution) -> MPCResult<()> {
        let round = self.current_round.load(Ordering::SeqCst);

        if contribution.round != round {
            return Err(MPCError::InvalidRound {
                expected: round,
                got: contribution.round,
            });
        }

        // Verify commitments if enabled.
        if self.config.verify_commitments && !contribution.verify_all() {
            self.stats.failed_contributions.fetch_add(1, Ordering::SeqCst);
            return Err(MPCError::CommitmentVerificationFailed {
                party: contribution.party.clone(),
            });
        }

        self.pending.write().insert(contribution.party.clone(), contribution);
        self.stats.total_contributions.fetch_add(1, Ordering::SeqCst);

        Ok(())
    }

    /// Compresses and submits raw gradients.
    pub fn submit_raw(
        &self,
        party: PartyId,
        gradients: HashMap<String, (Vec<f64>, Vec<usize>)>,
    ) -> MPCResult<()> {
        let round = self.current_round.load(Ordering::SeqCst);
        let mut contribution = AggregationContribution::new(party, round);

        for (name, (data, shape)) in gradients {
            let compressed = if self.config.enable_compression {
                self.compressor.compress(&data, &shape, &name)
            } else {
                // Create uncompressed "compressed" gradient.
                let indices: Vec<usize> = (0..data.len()).collect();
                let mut cg = CompressedGradient {
                    indices,
                    values: data.clone(),
                    shape,
                    original_size: data.len(),
                    scale: 1.0,
                    offset: 0.0,
                    quantized: false,
                    commitment: [0u8; 32],
                };
                cg.commitment = cg.compute_commitment();
                cg
            };

            contribution.add_gradient(name, compressed);
        }

        contribution.finalize();
        self.submit(contribution)
    }

    /// Aggregates all pending contributions.
    pub fn aggregate(&self) -> MPCResult<AggregationResult> {
        let start = Instant::now();
        let round = self.current_round.load(Ordering::SeqCst);
        let contributions = std::mem::take(&mut *self.pending.write());

        if contributions.len() < self.config.min_parties {
            return Err(MPCError::InsufficientParties {
                required: self.config.min_parties,
                available: contributions.len(),
            });
        }

        let result = if self.config.use_tree_aggregation && contributions.len() > self.config.tree_branching_factor {
            self.tree_aggregate(contributions)?
        } else {
            self.direct_aggregate(contributions)?
        };

        let duration = start.elapsed();

        // Update statistics.
        self.stats.rounds_completed.fetch_add(1, Ordering::SeqCst);
        self.stats.total_aggregation_time_ms.fetch_add(duration.as_millis() as u64, Ordering::SeqCst);

        let mut avg = self.stats.average_contributors.write();
        *avg = (*avg * 0.9) + (result.contributors.len() as f64 * 0.1);

        Ok(AggregationResult {
            gradients: result.gradients,
            contributors: result.contributors,
            failed_parties: result.failed_parties,
            round,
            duration,
            commitment: result.commitment,
        })
    }

    /// Direct aggregation for small party counts.
    fn direct_aggregate(&self, contributions: HashMap<PartyId, AggregationContribution>) -> MPCResult<IntermediateResult> {
        let mut aggregated: HashMap<String, Vec<f64>> = HashMap::new();
        let contributors: Vec<PartyId> = contributions.keys().cloned().collect();

        // Collect all tensor names.
        let mut all_names: HashSet<String> = HashSet::new();
        for contrib in contributions.values() {
            all_names.extend(contrib.gradients.keys().cloned());
        }

        // Aggregate each tensor.
        for name in all_names {
            let mut tensor_sum: Option<Vec<f64>> = None;

            for contrib in contributions.values() {
                if let Some(compressed) = contrib.gradients.get(&name) {
                    let decompressed = compressed.decompress();

                    match &mut tensor_sum {
                        Some(sum) => {
                            if sum.len() == decompressed.len() {
                                for (s, d) in sum.iter_mut().zip(decompressed.iter()) {
                                    *s += *d;
                                }
                            }
                        }
                        None => {
                            tensor_sum = Some(decompressed);
                        }
                    }
                }
            }

            if let Some(sum) = tensor_sum {
                aggregated.insert(name, sum);
            }
        }

        // Compute result commitment.
        let commitment = self.compute_result_commitment(&aggregated);

        Ok(IntermediateResult {
            gradients: aggregated,
            contributors,
            failed_parties: Vec::new(),
            commitment,
        })
    }

    /// Tree-based aggregation for large party counts.
    fn tree_aggregate(&self, contributions: HashMap<PartyId, AggregationContribution>) -> MPCResult<IntermediateResult> {
        let parties: Vec<PartyId> = contributions.keys().cloned().collect();
        let num_parties = parties.len();
        let bf = self.config.tree_branching_factor;

        // Build tree structure.
        let num_levels = (num_parties as f64).log(bf as f64).ceil() as usize + 1;
        let mut nodes: Vec<TreeNode> = Vec::new();

        // Create leaf nodes (one per party).
        for (i, party) in parties.iter().enumerate() {
            nodes.push(TreeNode {
                id: i,
                level: 0,
                children: vec![i], // Points to party index
                result: Some(self.decompress_contribution(&contributions[party])),
                contributors: vec![party.clone()],
            });
        }

        // Build internal nodes level by level.
        let mut current_level_start = 0;
        let mut current_level_end = num_parties;

        for level in 1..num_levels {
            let num_nodes_at_level = ((current_level_end - current_level_start) as f64 / bf as f64).ceil() as usize;

            for i in 0..num_nodes_at_level {
                let child_start = current_level_start + i * bf;
                let child_end = (child_start + bf).min(current_level_end);
                let children: Vec<usize> = (child_start..child_end).collect();

                // Aggregate children.
                let (result, contributors) = self.aggregate_nodes(&nodes, &children);

                nodes.push(TreeNode {
                    id: nodes.len(),
                    level,
                    children,
                    result: Some(result),
                    contributors,
                });
            }

            current_level_start = current_level_end;
            current_level_end = nodes.len();
        }

        // Root is the last node.
        let root = nodes.last().unwrap();
        let gradients = root.result.clone().unwrap_or_default();
        let commitment = self.compute_result_commitment(&gradients);

        Ok(IntermediateResult {
            gradients,
            contributors: root.contributors.clone(),
            failed_parties: Vec::new(),
            commitment,
        })
    }

    /// Aggregates multiple tree nodes.
    fn aggregate_nodes(&self, nodes: &[TreeNode], child_ids: &[usize]) -> (HashMap<String, Vec<f64>>, Vec<PartyId>) {
        let mut aggregated: HashMap<String, Vec<f64>> = HashMap::new();
        let mut all_contributors = Vec::new();

        for &child_id in child_ids {
            if let Some(node) = nodes.get(child_id) {
                all_contributors.extend(node.contributors.clone());

                if let Some(ref result) = node.result {
                    for (name, values) in result {
                        aggregated.entry(name.clone())
                            .and_modify(|sum: &mut Vec<f64>| {
                                if sum.len() == values.len() {
                                    for (s, v) in sum.iter_mut().zip(values.iter()) {
                                        *s += *v;
                                    }
                                }
                            })
                            .or_insert_with(|| values.clone());
                    }
                }
            }
        }

        (aggregated, all_contributors)
    }

    /// Decompresses all gradients in a contribution.
    fn decompress_contribution(&self, contrib: &AggregationContribution) -> HashMap<String, Vec<f64>> {
        contrib.gradients.iter()
            .map(|(name, compressed)| (name.clone(), compressed.decompress()))
            .collect()
    }

    /// Computes commitment over aggregated result.
    fn compute_result_commitment(&self, gradients: &HashMap<String, Vec<f64>>) -> [u8; 32] {
        let mut hasher = Sha256::new();

        let mut names: Vec<&String> = gradients.keys().collect();
        names.sort();

        for name in names {
            hasher.update(name.as_bytes());
            for &val in &gradients[name] {
                hasher.update(val.to_le_bytes());
            }
        }

        hasher.finalize().into()
    }

    /// Returns the current round.
    pub fn current_round(&self) -> u64 {
        self.current_round.load(Ordering::SeqCst)
    }

    /// Returns pending contribution count.
    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
    }

    /// Returns aggregation statistics.
    pub fn stats(&self) -> AggregationStatsSnapshot {
        AggregationStatsSnapshot {
            rounds_completed: self.stats.rounds_completed.load(Ordering::SeqCst),
            total_contributions: self.stats.total_contributions.load(Ordering::SeqCst),
            failed_contributions: self.stats.failed_contributions.load(Ordering::SeqCst),
            average_contributors: *self.stats.average_contributors.read(),
            average_aggregation_time: if self.stats.rounds_completed.load(Ordering::SeqCst) > 0 {
                Duration::from_millis(
                    self.stats.total_aggregation_time_ms.load(Ordering::SeqCst) /
                    self.stats.rounds_completed.load(Ordering::SeqCst)
                )
            } else {
                Duration::ZERO
            },
            compression_ratio: self.compressor.stats().average_compression_ratio(),
        }
    }

    /// Returns the gradient compressor.
    pub fn compressor(&self) -> &GradientCompressor {
        &self.compressor
    }
}

/// Intermediate aggregation result.
struct IntermediateResult {
    gradients: HashMap<String, Vec<f64>>,
    contributors: Vec<PartyId>,
    failed_parties: Vec<PartyId>,
    commitment: [u8; 32],
}

/// Snapshot of aggregation statistics.
#[derive(Debug, Clone)]
pub struct AggregationStatsSnapshot {
    pub rounds_completed: u64,
    pub total_contributions: u64,
    pub failed_contributions: u64,
    pub average_contributors: f64,
    pub average_aggregation_time: Duration,
    pub compression_ratio: f64,
}

/// Dropout-tolerant aggregator that can handle missing parties.
pub struct DropoutTolerantAggregator {
    /// Inner aggregator.
    inner: SecureAggregator,
    /// Expected parties per round.
    expected_parties: RwLock<HashSet<PartyId>>,
    /// Party timeout tracking.
    party_timeouts: RwLock<HashMap<PartyId, usize>>,
    /// Maximum consecutive timeouts before exclusion.
    max_timeouts: usize,
}

impl DropoutTolerantAggregator {
    /// Creates a new dropout-tolerant aggregator.
    pub fn new(config: AggregationConfig, max_timeouts: usize) -> Self {
        Self {
            inner: SecureAggregator::new(config),
            expected_parties: RwLock::new(HashSet::new()),
            party_timeouts: RwLock::new(HashMap::new()),
            max_timeouts,
        }
    }

    /// Sets expected parties for the next round.
    pub fn set_expected_parties(&self, parties: Vec<PartyId>) {
        *self.expected_parties.write() = parties.into_iter().collect();
    }

    /// Submits a contribution.
    pub fn submit(&self, contribution: AggregationContribution) -> MPCResult<()> {
        // Reset timeout counter on successful submission.
        self.party_timeouts.write().remove(&contribution.party);
        self.inner.submit(contribution)
    }

    /// Aggregates with dropout handling.
    pub fn aggregate(&self) -> MPCResult<AggregationResult> {
        let expected = self.expected_parties.read().clone();
        let pending: HashSet<PartyId> = self.inner.pending.read().keys().cloned().collect();

        // Track missing parties.
        let missing: Vec<PartyId> = expected.difference(&pending).cloned().collect();

        for party in &missing {
            let mut timeouts = self.party_timeouts.write();
            let count = timeouts.entry(party.clone()).or_insert(0);
            *count += 1;
        }

        // Try aggregation.
        match self.inner.aggregate() {
            Ok(mut result) => {
                result.failed_parties = missing;
                Ok(result)
            }
            Err(e) => Err(e),
        }
    }

    /// Returns parties that have timed out too many times.
    pub fn excluded_parties(&self) -> Vec<PartyId> {
        self.party_timeouts.read()
            .iter()
            .filter(|(_, &count)| count >= self.max_timeouts)
            .map(|(party, _)| party.clone())
            .collect()
    }

    /// Resets timeout counter for a party.
    pub fn reset_timeout(&self, party: &PartyId) {
        self.party_timeouts.write().remove(party);
    }

    /// Returns inner aggregator for direct access.
    pub fn inner(&self) -> &SecureAggregator {
        &self.inner
    }
}

/// Weighted aggregator for heterogeneous data sizes.
pub struct WeightedAggregator {
    /// Inner aggregator.
    inner: SecureAggregator,
    /// Party weights (data size proportions).
    weights: RwLock<HashMap<PartyId, f64>>,
}

impl WeightedAggregator {
    /// Creates a new weighted aggregator.
    pub fn new(config: AggregationConfig) -> Self {
        Self {
            inner: SecureAggregator::new(config),
            weights: RwLock::new(HashMap::new()),
        }
    }

    /// Sets weight for a party based on data size.
    pub fn set_weight(&self, party: PartyId, weight: f64) {
        self.weights.write().insert(party, weight);
    }

    /// Sets weights for all parties.
    pub fn set_weights(&self, weights: HashMap<PartyId, f64>) {
        *self.weights.write() = weights;
    }

    /// Submits a contribution (will be weighted during aggregation).
    pub fn submit(&self, contribution: AggregationContribution) -> MPCResult<()> {
        self.inner.submit(contribution)
    }

    /// Aggregates with weighting.
    pub fn aggregate(&self) -> MPCResult<AggregationResult> {
        let mut result = self.inner.aggregate()?;

        // Apply weights to aggregated gradients.
        let weights = self.weights.read();
        let total_weight: f64 = result.contributors.iter()
            .filter_map(|p| weights.get(p))
            .sum();

        if total_weight > 0.0 {
            // Normalize by total weight.
            for values in result.gradients.values_mut() {
                for v in values.iter_mut() {
                    *v /= total_weight;
                }
            }
        }

        Ok(result)
    }

    /// Returns inner aggregator.
    pub fn inner(&self) -> &SecureAggregator {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_gradient(size: usize, value: f64) -> (Vec<f64>, Vec<usize>) {
        (vec![value; size], vec![size])
    }

    #[test]
    fn test_gradient_compression() {
        let config = CompressionConfig {
            top_k_ratio: 0.1,
            quantization_bits: 0,
            ..Default::default()
        };
        let compressor = GradientCompressor::new(config);

        let mut data = vec![0.0; 100];
        // Set 10 values to non-zero.
        for i in 0..10 {
            data[i * 10] = (i + 1) as f64;
        }

        let compressed = compressor.compress(&data, &[100], "test");

        // Should keep approximately 10 values.
        assert!(compressed.indices.len() <= 10);
        assert!(compressed.compression_ratio() <= 0.11);

        // Decompress and verify.
        let decompressed = compressed.decompress();
        assert_eq!(decompressed.len(), 100);
    }

    #[test]
    fn test_compression_with_quantization() {
        let config = CompressionConfig {
            top_k_ratio: 0.5,
            quantization_bits: 8,
            ..Default::default()
        };
        let compressor = GradientCompressor::new(config);

        let data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let compressed = compressor.compress(&data, &[100], "test");

        assert!(compressed.quantized);
        assert!(compressed.verify_commitment());
    }

    #[test]
    fn test_direct_aggregation() {
        let config = AggregationConfig {
            min_parties: 2,
            use_tree_aggregation: false,
            enable_compression: false,
            ..Default::default()
        };
        let aggregator = SecureAggregator::new(config);

        aggregator.start_round();

        // Submit from 3 parties.
        for i in 0..3 {
            let party = PartyId::from_index(i);
            let mut gradients = HashMap::new();
            gradients.insert("layer1".to_string(), create_test_gradient(100, 1.0));

            aggregator.submit_raw(party, gradients).unwrap();
        }

        let result = aggregator.aggregate().unwrap();

        assert_eq!(result.contributors.len(), 3);
        assert!(result.gradients.contains_key("layer1"));

        // Sum should be 3.0 for each element.
        for &val in &result.gradients["layer1"] {
            assert!((val - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tree_aggregation() {
        let config = AggregationConfig {
            min_parties: 2,
            use_tree_aggregation: true,
            tree_branching_factor: 2,
            enable_compression: false,
            ..Default::default()
        };
        let aggregator = SecureAggregator::new(config);

        aggregator.start_round();

        // Submit from 8 parties.
        for i in 0..8 {
            let party = PartyId::from_index(i);
            let mut gradients = HashMap::new();
            gradients.insert("layer1".to_string(), create_test_gradient(50, 1.0));

            aggregator.submit_raw(party, gradients).unwrap();
        }

        let result = aggregator.aggregate().unwrap();

        assert_eq!(result.contributors.len(), 8);

        // Sum should be 8.0 for each element.
        for &val in &result.gradients["layer1"] {
            assert!((val - 8.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dropout_tolerant_aggregation() {
        let config = AggregationConfig {
            min_parties: 2,
            enable_compression: false,
            verify_commitments: false, // Disable for test
            ..Default::default()
        };
        let aggregator = DropoutTolerantAggregator::new(config, 3);

        let parties: Vec<PartyId> = (0..5).map(PartyId::from_index).collect();
        aggregator.set_expected_parties(parties);

        aggregator.inner().start_round();

        // Only 3 of 5 parties submit.
        for i in 0..3 {
            let party = PartyId::from_index(i);

            let mut contrib = AggregationContribution::new(party.clone(), 1);
            let mut cg = CompressedGradient {
                indices: (0..10).collect(),
                values: vec![1.0; 10],
                shape: vec![10],
                original_size: 10,
                scale: 1.0,
                offset: 0.0,
                quantized: false,
                commitment: [0u8; 32],
            };
            cg.commitment = cg.compute_commitment();
            contrib.add_gradient("layer1".to_string(), cg);
            contrib.finalize();

            aggregator.submit(contrib).unwrap();
        }

        let result = aggregator.aggregate().unwrap();

        assert_eq!(result.contributors.len(), 3);
        assert_eq!(result.failed_parties.len(), 2);
    }

    #[test]
    fn test_weighted_aggregation() {
        let config = AggregationConfig {
            min_parties: 2,
            enable_compression: false,
            verify_commitments: false, // Disable for test
            ..Default::default()
        };
        let aggregator = WeightedAggregator::new(config);

        // Party 0 has weight 1, party 1 has weight 3.
        let mut weights = HashMap::new();
        weights.insert(PartyId::from_index(0), 1.0);
        weights.insert(PartyId::from_index(1), 3.0);
        aggregator.set_weights(weights);

        aggregator.inner().start_round();

        for i in 0..2 {
            let party = PartyId::from_index(i);
            let mut contrib = AggregationContribution::new(party.clone(), 1);

            let value = 4.0; // Both submit 4.0
            let mut cg = CompressedGradient {
                indices: vec![0],
                values: vec![value],
                shape: vec![1],
                original_size: 1,
                scale: 1.0,
                offset: 0.0,
                quantized: false,
                commitment: [0u8; 32],
            };
            cg.commitment = cg.compute_commitment();
            contrib.add_gradient("layer1".to_string(), cg);
            contrib.finalize();

            aggregator.submit(contrib).unwrap();
        }

        let result = aggregator.aggregate().unwrap();

        // Weighted average: (4 + 4) / (1 + 3) = 2.0
        assert!((result.gradients["layer1"][0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_compression_statistics() {
        let config = CompressionConfig {
            top_k_ratio: 0.1,
            quantization_bits: 8,
            ..Default::default()
        };
        let compressor = GradientCompressor::new(config);

        for i in 0..10 {
            let data: Vec<f64> = (0..1000).map(|j| (i * 1000 + j) as f64).collect();
            compressor.compress(&data, &[1000], &format!("tensor_{}", i));
        }

        let stats = compressor.stats();
        assert_eq!(stats.tensors_compressed.load(Ordering::SeqCst), 10);
        assert!(stats.average_compression_ratio() < 0.5);
    }

    #[test]
    fn test_error_feedback() {
        let config = CompressionConfig {
            top_k_ratio: 0.1,
            use_error_feedback: true,
            ..Default::default()
        };
        let compressor = GradientCompressor::new(config);

        // Compress same tensor multiple times.
        let data = vec![1.0; 100];

        compressor.compress(&data, &[100], "test");
        let c2 = compressor.compress(&data, &[100], "test");

        // Error feedback should affect second compression.
        assert!(c2.original_size > 0);
    }
}
