//! Gradient Aggregation.
//!
//! Byzantine-fault-tolerant gradient aggregation algorithms including
//! FedAvg, Krum, Median, and Trimmed Mean.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use super::model::{ModelGradient, WeightData, LayerGradient};
use super::round::GradientSubmission;

/// Aggregation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Federated Averaging with stake-weighted mean.
    FedAvg,
    /// Krum: Byzantine-resilient selection.
    Krum { num_byzantine: usize },
    /// Multi-Krum: Select multiple gradients.
    MultiKrum { num_byzantine: usize, num_select: usize },
    /// Coordinate-wise median.
    Median,
    /// Trimmed mean (excludes outliers).
    TrimmedMean { trim_fraction: f64 },
    /// Geometric median approximation.
    GeometricMedian { max_iterations: usize },
}

impl Default for AggregationStrategy {
    fn default() -> Self {
        Self::FedAvg
    }
}

/// Configuration for gradient aggregation.
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Aggregation strategy.
    pub strategy: AggregationStrategy,
    /// Minimum number of gradients required.
    pub min_gradients: usize,
    /// Maximum gradient norm (for clipping).
    pub max_gradient_norm: f64,
    /// Whether to normalize by stake.
    pub stake_weighted: bool,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            strategy: AggregationStrategy::FedAvg,
            min_gradients: 2,
            max_gradient_norm: 10.0,
            stake_weighted: true,
        }
    }
}

/// A gradient with associated metadata for aggregation.
#[derive(Debug, Clone)]
pub struct WeightedGradient {
    /// Participant ID.
    pub participant_id: String,
    /// Stake weight.
    pub stake: u64,
    /// The gradient data.
    pub gradient: ModelGradient,
    /// Claimed error bound.
    pub error_bound: f64,
    /// Whether this gradient passed validation.
    pub is_valid: bool,
}

/// Result of gradient aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedGradient {
    /// The aggregated gradient.
    pub gradient: ModelGradient,
    /// Combined error bound.
    pub error_bound: f64,
    /// Total stake that contributed.
    pub total_stake: u64,
    /// Number of gradients included.
    pub num_included: usize,
    /// IDs of excluded participants.
    pub excluded: Vec<String>,
    /// Strategy used for aggregation.
    pub strategy: AggregationStrategy,
}

/// Gradient aggregator.
#[derive(Debug)]
pub struct GradientAggregator {
    /// Configuration.
    config: AggregationConfig,
    /// Collected gradients.
    gradients: Vec<WeightedGradient>,
}

impl GradientAggregator {
    /// Creates a new aggregator.
    pub fn new(config: AggregationConfig) -> Self {
        Self {
            config,
            gradients: Vec::new(),
        }
    }

    /// Adds a gradient to the aggregator.
    pub fn add_gradient(&mut self, gradient: WeightedGradient) {
        self.gradients.push(gradient);
    }

    /// Returns the number of collected gradients.
    pub fn gradient_count(&self) -> usize {
        self.gradients.len()
    }

    /// Clears all collected gradients.
    pub fn clear(&mut self) {
        self.gradients.clear();
    }

    /// Performs aggregation.
    pub fn aggregate(&self) -> Result<AggregatedGradient, AggregationError> {
        if self.gradients.len() < self.config.min_gradients {
            return Err(AggregationError::InsufficientGradients {
                required: self.config.min_gradients,
                received: self.gradients.len(),
            });
        }

        // Filter valid gradients
        let valid_gradients: Vec<_> = self.gradients.iter()
            .filter(|g| g.is_valid)
            .collect();

        if valid_gradients.is_empty() {
            return Err(AggregationError::NoValidGradients);
        }

        match self.config.strategy {
            AggregationStrategy::FedAvg => self.fedavg(&valid_gradients),
            AggregationStrategy::Krum { num_byzantine } => self.krum(&valid_gradients, num_byzantine, 1),
            AggregationStrategy::MultiKrum { num_byzantine, num_select } => self.krum(&valid_gradients, num_byzantine, num_select),
            AggregationStrategy::Median => self.median(&valid_gradients),
            AggregationStrategy::TrimmedMean { trim_fraction } => self.trimmed_mean(&valid_gradients, trim_fraction),
            AggregationStrategy::GeometricMedian { max_iterations } => self.geometric_median(&valid_gradients, max_iterations),
        }
    }

    /// Federated Averaging: stake-weighted mean of all gradients.
    fn fedavg(&self, gradients: &[&WeightedGradient]) -> Result<AggregatedGradient, AggregationError> {
        let total_stake: u64 = gradients.iter().map(|g| g.stake).sum();
        
        if total_stake == 0 {
            return Err(AggregationError::ZeroTotalStake);
        }

        // Use first gradient as template
        let template = &gradients[0].gradient;
        let mut aggregated = ModelGradient::zeros_like_from_gradient(template);
        let mut combined_error = 0.0;

        // Aggregate embeddings
        if let Some(ref mut agg_embed) = aggregated.embeddings {
            for grad in gradients {
                if let Some(ref embed) = grad.gradient.embeddings {
                    let weight = if self.config.stake_weighted {
                        grad.stake as f64 / total_stake as f64
                    } else {
                        1.0 / gradients.len() as f64
                    };
                    
                    for (i, val) in embed.data.iter().enumerate() {
                        agg_embed.data[i] += weight as f32 * val;
                    }
                }
            }
        }

        // Aggregate layer gradients
        for (layer_idx, layer_grad) in aggregated.layers.iter_mut().enumerate() {
            for (name, weight_grad) in layer_grad.gradients.iter_mut() {
                for grad in gradients {
                    if layer_idx < grad.gradient.layers.len() {
                        if let Some(src) = grad.gradient.layers[layer_idx].gradients.get(name) {
                            let weight = if self.config.stake_weighted {
                                grad.stake as f64 / total_stake as f64
                            } else {
                                1.0 / gradients.len() as f64
                            };
                            
                            for (i, val) in src.data.iter().enumerate() {
                                weight_grad.data[i] += weight as f32 * val;
                            }
                        }
                    }
                }
            }
        }

        // Aggregate LM head
        if let Some(ref mut agg_lm) = aggregated.lm_head {
            for grad in gradients {
                if let Some(ref lm) = grad.gradient.lm_head {
                    let weight = if self.config.stake_weighted {
                        grad.stake as f64 / total_stake as f64
                    } else {
                        1.0 / gradients.len() as f64
                    };
                    
                    for (i, val) in lm.data.iter().enumerate() {
                        agg_lm.data[i] += weight as f32 * val;
                    }
                }
            }
        }

        // Compute combined error bound (weighted average of individual bounds)
        for grad in gradients {
            let weight = if self.config.stake_weighted {
                grad.stake as f64 / total_stake as f64
            } else {
                1.0 / gradients.len() as f64
            };
            combined_error += weight * grad.error_bound;
        }

        aggregated.error_bound = combined_error;

        Ok(AggregatedGradient {
            gradient: aggregated,
            error_bound: combined_error,
            total_stake,
            num_included: gradients.len(),
            excluded: vec![],
            strategy: self.config.strategy,
        })
    }

    /// Krum: Select gradient(s) with minimum sum of squared distances to neighbors.
    fn krum(&self, gradients: &[&WeightedGradient], num_byzantine: usize, num_select: usize) -> Result<AggregatedGradient, AggregationError> {
        let n = gradients.len();
        
        if n <= 2 * num_byzantine + 2 {
            return Err(AggregationError::InsufficientForByzantine {
                num_gradients: n,
                num_byzantine,
            });
        }

        // Compute pairwise distances
        let mut distances: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = gradient_distance(&gradients[i].gradient, &gradients[j].gradient);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        // For each gradient, compute sum of n - num_byzantine - 2 closest distances
        let num_closest = n - num_byzantine - 2;
        let mut scores: Vec<(usize, f64)> = (0..n).map(|i| {
            let mut dists: Vec<f64> = distances[i].clone();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let score: f64 = dists[1..=num_closest].iter().sum(); // Skip self (0)
            (i, score)
        }).collect();

        // Sort by score (lower is better)
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top num_select gradients
        let selected_indices: Vec<usize> = scores.iter().take(num_select).map(|(i, _)| *i).collect();
        let selected: Vec<&WeightedGradient> = selected_indices.iter()
            .map(|&i| gradients[i])
            .collect();

        // Excluded participants
        let excluded: Vec<String> = gradients.iter()
            .enumerate()
            .filter(|(i, _)| !selected_indices.contains(i))
            .map(|(_, g)| g.participant_id.clone())
            .collect();

        // Average the selected gradients
        let mut result = self.fedavg(&selected)?;
        result.excluded = excluded;
        result.strategy = self.config.strategy;

        Ok(result)
    }

    /// Coordinate-wise median.
    fn median(&self, gradients: &[&WeightedGradient]) -> Result<AggregatedGradient, AggregationError> {
        let template = &gradients[0].gradient;
        let mut aggregated = ModelGradient::zeros_like_from_gradient(template);

        // Median of embeddings
        if let Some(ref mut agg_embed) = aggregated.embeddings {
            let n = agg_embed.data.len();
            for i in 0..n {
                let mut values: Vec<f32> = gradients.iter()
                    .filter_map(|g| g.gradient.embeddings.as_ref())
                    .map(|e| e.data[i])
                    .collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                agg_embed.data[i] = median_of_sorted(&values);
            }
        }

        // Median of layer gradients
        for (layer_idx, layer_grad) in aggregated.layers.iter_mut().enumerate() {
            for (name, weight_grad) in layer_grad.gradients.iter_mut() {
                let n = weight_grad.data.len();
                for i in 0..n {
                    let mut values: Vec<f32> = gradients.iter()
                        .filter_map(|g| g.gradient.layers.get(layer_idx))
                        .filter_map(|l| l.gradients.get(name))
                        .map(|w| w.data[i])
                        .collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    weight_grad.data[i] = median_of_sorted(&values);
                }
            }
        }

        // Median of LM head
        if let Some(ref mut agg_lm) = aggregated.lm_head {
            let n = agg_lm.data.len();
            for i in 0..n {
                let mut values: Vec<f32> = gradients.iter()
                    .filter_map(|g| g.gradient.lm_head.as_ref())
                    .map(|l| l.data[i])
                    .collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                agg_lm.data[i] = median_of_sorted(&values);
            }
        }

        let total_stake: u64 = gradients.iter().map(|g| g.stake).sum();
        let avg_error: f64 = gradients.iter().map(|g| g.error_bound).sum::<f64>() / gradients.len() as f64;
        aggregated.error_bound = avg_error;

        Ok(AggregatedGradient {
            gradient: aggregated,
            error_bound: avg_error,
            total_stake,
            num_included: gradients.len(),
            excluded: vec![],
            strategy: self.config.strategy,
        })
    }

    /// Trimmed mean: exclude top and bottom fraction, then average.
    fn trimmed_mean(&self, gradients: &[&WeightedGradient], trim_fraction: f64) -> Result<AggregatedGradient, AggregationError> {
        let n = gradients.len();
        let trim_count = ((n as f64 * trim_fraction).floor() as usize).min(n / 2 - 1);
        
        if n <= 2 * trim_count {
            return Err(AggregationError::InsufficientForTrimming {
                num_gradients: n,
                trim_count,
            });
        }

        let template = &gradients[0].gradient;
        let mut aggregated = ModelGradient::zeros_like_from_gradient(template);
        let remaining = n - 2 * trim_count;

        // Trimmed mean of embeddings
        if let Some(ref mut agg_embed) = aggregated.embeddings {
            let dim = agg_embed.data.len();
            for i in 0..dim {
                let mut values: Vec<f32> = gradients.iter()
                    .filter_map(|g| g.gradient.embeddings.as_ref())
                    .map(|e| e.data[i])
                    .collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let trimmed = &values[trim_count..n - trim_count];
                agg_embed.data[i] = trimmed.iter().sum::<f32>() / remaining as f32;
            }
        }

        // Trimmed mean of layer gradients
        for (layer_idx, layer_grad) in aggregated.layers.iter_mut().enumerate() {
            for (name, weight_grad) in layer_grad.gradients.iter_mut() {
                let dim = weight_grad.data.len();
                for i in 0..dim {
                    let mut values: Vec<f32> = gradients.iter()
                        .filter_map(|g| g.gradient.layers.get(layer_idx))
                        .filter_map(|l| l.gradients.get(name))
                        .map(|w| w.data[i])
                        .collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    if values.len() > 2 * trim_count {
                        let trimmed = &values[trim_count..values.len() - trim_count];
                        weight_grad.data[i] = trimmed.iter().sum::<f32>() / trimmed.len() as f32;
                    }
                }
            }
        }

        // Trimmed mean of LM head
        if let Some(ref mut agg_lm) = aggregated.lm_head {
            let dim = agg_lm.data.len();
            for i in 0..dim {
                let mut values: Vec<f32> = gradients.iter()
                    .filter_map(|g| g.gradient.lm_head.as_ref())
                    .map(|l| l.data[i])
                    .collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if values.len() > 2 * trim_count {
                    let trimmed = &values[trim_count..values.len() - trim_count];
                    agg_lm.data[i] = trimmed.iter().sum::<f32>() / trimmed.len() as f32;
                }
            }
        }

        let total_stake: u64 = gradients.iter().map(|g| g.stake).sum();
        let avg_error: f64 = gradients.iter().map(|g| g.error_bound).sum::<f64>() / gradients.len() as f64;
        aggregated.error_bound = avg_error;

        Ok(AggregatedGradient {
            gradient: aggregated,
            error_bound: avg_error,
            total_stake,
            num_included: gradients.len(),
            excluded: vec![],
            strategy: self.config.strategy,
        })
    }

    /// Geometric median approximation using Weiszfeld's algorithm.
    fn geometric_median(&self, gradients: &[&WeightedGradient], max_iterations: usize) -> Result<AggregatedGradient, AggregationError> {
        // Start with arithmetic mean
        let mut result = self.fedavg(gradients)?;
        
        // Iteratively refine (simplified version)
        for _ in 0..max_iterations {
            // Compute distance from current estimate to each gradient
            let mut total_weight = 0.0;
            let mut new_estimate = ModelGradient::zeros_like_from_gradient(&result.gradient);
            
            for grad in gradients {
                let dist = gradient_distance(&result.gradient, &grad.gradient);
                if dist > 1e-10 {
                    let weight = 1.0 / dist;
                    total_weight += weight;
                    
                    // Add weighted contribution
                    add_weighted_gradient(&mut new_estimate, &grad.gradient, weight as f32);
                }
            }
            
            if total_weight > 1e-10 {
                scale_gradient(&mut new_estimate, 1.0 / total_weight as f32);
                result.gradient = new_estimate;
            }
        }

        result.strategy = self.config.strategy;
        Ok(result)
    }
}

/// Computes L2 distance between two gradients.
fn gradient_distance(a: &ModelGradient, b: &ModelGradient) -> f64 {
    let mut sum_sq = 0.0;
    
    if let (Some(ea), Some(eb)) = (&a.embeddings, &b.embeddings) {
        for (va, vb) in ea.data.iter().zip(eb.data.iter()) {
            let diff = (*va - *vb) as f64;
            sum_sq += diff * diff;
        }
    }
    
    for (la, lb) in a.layers.iter().zip(b.layers.iter()) {
        for (name, wa) in &la.gradients {
            if let Some(wb) = lb.gradients.get(name) {
                for (va, vb) in wa.data.iter().zip(wb.data.iter()) {
                    let diff = (*va - *vb) as f64;
                    sum_sq += diff * diff;
                }
            }
        }
    }
    
    if let (Some(la), Some(lb)) = (&a.lm_head, &b.lm_head) {
        for (va, vb) in la.data.iter().zip(lb.data.iter()) {
            let diff = (*va - *vb) as f64;
            sum_sq += diff * diff;
        }
    }
    
    sum_sq.sqrt()
}

/// Adds weighted gradient to accumulator.
fn add_weighted_gradient(acc: &mut ModelGradient, grad: &ModelGradient, weight: f32) {
    if let (Some(ref mut ae), Some(ref ge)) = (&mut acc.embeddings, &grad.embeddings) {
        for (a, g) in ae.data.iter_mut().zip(ge.data.iter()) {
            *a += weight * g;
        }
    }
    
    for (al, gl) in acc.layers.iter_mut().zip(grad.layers.iter()) {
        for (name, aw) in al.gradients.iter_mut() {
            if let Some(gw) = gl.gradients.get(name) {
                for (a, g) in aw.data.iter_mut().zip(gw.data.iter()) {
                    *a += weight * g;
                }
            }
        }
    }
    
    if let (Some(ref mut al), Some(ref gl)) = (&mut acc.lm_head, &grad.lm_head) {
        for (a, g) in al.data.iter_mut().zip(gl.data.iter()) {
            *a += weight * g;
        }
    }
}

/// Scales a gradient by a factor.
fn scale_gradient(grad: &mut ModelGradient, factor: f32) {
    if let Some(ref mut e) = grad.embeddings {
        for v in e.data.iter_mut() {
            *v *= factor;
        }
    }
    
    for layer in grad.layers.iter_mut() {
        for (_, weight) in layer.gradients.iter_mut() {
            for v in weight.data.iter_mut() {
                *v *= factor;
            }
        }
    }
    
    if let Some(ref mut l) = grad.lm_head {
        for v in l.data.iter_mut() {
            *v *= factor;
        }
    }
}

/// Computes median of a sorted slice.
fn median_of_sorted(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

// Add helper method to ModelGradient
impl ModelGradient {
    /// Creates a zero gradient from another gradient as template.
    pub fn zeros_like_from_gradient(template: &ModelGradient) -> Self {
        let embeddings = template.embeddings.as_ref().map(|e| WeightData {
            shape: e.shape.clone(),
            data: vec![0.0; e.data.len()],
            error_bound: 0.0,
        });

        let layers: Vec<LayerGradient> = template.layers.iter().map(|layer| {
            let gradients = layer.gradients.iter().map(|(name, weight)| {
                (name.clone(), WeightData {
                    shape: weight.shape.clone(),
                    data: vec![0.0; weight.data.len()],
                    error_bound: 0.0,
                })
            }).collect();
            LayerGradient {
                layer_idx: layer.layer_idx,
                gradients,
            }
        }).collect();

        let lm_head = template.lm_head.as_ref().map(|l| WeightData {
            shape: l.shape.clone(),
            data: vec![0.0; l.data.len()],
            error_bound: 0.0,
        });

        Self {
            embeddings,
            layers,
            lm_head,
            error_bound: 0.0,
        }
    }
}

/// Aggregation errors.
#[derive(Debug)]
pub enum AggregationError {
    /// Not enough gradients to aggregate.
    InsufficientGradients {
        required: usize,
        received: usize,
    },
    /// No valid gradients available.
    NoValidGradients,
    /// Total stake is zero.
    ZeroTotalStake,
    /// Not enough gradients for Byzantine tolerance.
    InsufficientForByzantine {
        num_gradients: usize,
        num_byzantine: usize,
    },
    /// Not enough gradients for trimming.
    InsufficientForTrimming {
        num_gradients: usize,
        trim_count: usize,
    },
}

impl std::fmt::Display for AggregationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientGradients { required, received } => {
                write!(f, "Insufficient gradients: required {}, received {}", required, received)
            }
            Self::NoValidGradients => write!(f, "No valid gradients available"),
            Self::ZeroTotalStake => write!(f, "Total stake is zero"),
            Self::InsufficientForByzantine { num_gradients, num_byzantine } => {
                write!(f, "Insufficient gradients ({}) for {} Byzantine nodes", num_gradients, num_byzantine)
            }
            Self::InsufficientForTrimming { num_gradients, trim_count } => {
                write!(f, "Insufficient gradients ({}) for trim count {}", num_gradients, trim_count)
            }
        }
    }
}

impl std::error::Error for AggregationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_gradient(offset: f32) -> ModelGradient {
        ModelGradient {
            embeddings: Some(WeightData {
                shape: vec![4],
                data: vec![1.0 + offset, 2.0 + offset, 3.0 + offset, 4.0 + offset],
                error_bound: 0.01,
            }),
            layers: vec![],
            lm_head: None,
            error_bound: 0.01,
        }
    }

    #[test]
    fn test_fedavg_aggregation() {
        let config = AggregationConfig::default();
        let mut aggregator = GradientAggregator::new(config);

        aggregator.add_gradient(WeightedGradient {
            participant_id: "node1".to_string(),
            stake: 100,
            gradient: create_test_gradient(0.0),
            error_bound: 0.01,
            is_valid: true,
        });

        aggregator.add_gradient(WeightedGradient {
            participant_id: "node2".to_string(),
            stake: 100,
            gradient: create_test_gradient(1.0),
            error_bound: 0.01,
            is_valid: true,
        });

        let result = aggregator.aggregate().unwrap();
        assert_eq!(result.num_included, 2);
        
        // Average of [1,2,3,4] and [2,3,4,5] = [1.5, 2.5, 3.5, 4.5]
        let embed = result.gradient.embeddings.unwrap();
        assert!((embed.data[0] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_median_aggregation() {
        let config = AggregationConfig {
            strategy: AggregationStrategy::Median,
            min_gradients: 2,
            ..Default::default()
        };
        let mut aggregator = GradientAggregator::new(config);

        for i in 0..5 {
            aggregator.add_gradient(WeightedGradient {
                participant_id: format!("node{}", i),
                stake: 100,
                gradient: create_test_gradient(i as f32),
                error_bound: 0.01,
                is_valid: true,
            });
        }

        let result = aggregator.aggregate().unwrap();
        
        // Median offset should be 2.0
        let embed = result.gradient.embeddings.unwrap();
        assert!((embed.data[0] - 3.0).abs() < 0.01); // 1+2=3 is median
    }
}
