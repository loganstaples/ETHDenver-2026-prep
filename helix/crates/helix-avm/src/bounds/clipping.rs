//! Gradient Clipping Error Bounds.
//!
//! Analyzes error introduced by gradient clipping operations,
//! which are essential for stable training.

/// Configuration for gradient clipping analysis.
#[derive(Debug, Clone)]
pub struct ClippingBounds {
    /// Clipping threshold.
    pub clip_threshold: f64,
    /// Expected gradient magnitude.
    pub expected_grad_magnitude: f64,
    /// Gradient dimension.
    pub grad_dim: usize,
    /// Machine epsilon.
    pub epsilon: f64,
}

impl Default for ClippingBounds {
    fn default() -> Self {
        Self {
            clip_threshold: 1.0,
            expected_grad_magnitude: 0.5,
            grad_dim: 1000,
            epsilon: f64::EPSILON,
        }
    }
}

/// Error analysis for gradient clipping.
#[derive(Debug, Clone)]
pub struct GradientClipError {
    /// Error from norm computation.
    pub norm_error: f64,
    /// Error from scaling.
    pub scale_error: f64,
    /// Effective gradient loss (due to clipping).
    pub gradient_loss: f64,
    /// Total error.
    pub total_error: f64,
    /// Probability of clipping occurring.
    pub clip_probability: f64,
}

impl ClippingBounds {
    /// Creates new clipping bounds.
    pub fn new(clip_threshold: f64, grad_dim: usize) -> Self {
        Self {
            clip_threshold,
            expected_grad_magnitude: clip_threshold / 2.0,
            grad_dim,
            epsilon: f64::EPSILON,
        }
    }

    /// Computes error bounds for gradient clipping by value.
    ///
    /// clip_by_value(g, -c, c) = clamp(g, -c, c)
    pub fn value_clipping_error(&self, grad_error: f64) -> ValueClipError {
        // Error in clamping: only affects values near boundary
        // If |g| is close to c, error in g can push it across boundary
        let boundary_error = grad_error; // Can cause incorrect clipping
        
        // Information loss when clipping
        let clipped_magnitude = self.expected_grad_magnitude - self.clip_threshold;
        let info_loss = if clipped_magnitude > 0.0 {
            clipped_magnitude / self.expected_grad_magnitude
        } else {
            0.0
        };
        
        ValueClipError {
            boundary_error,
            info_loss,
            total_error: boundary_error,
        }
    }

    /// Computes error bounds for gradient clipping by global norm.
    ///
    /// If ||g|| > c: g' = g * c / ||g||
    pub fn norm_clipping_error(&self, grad_error: f64, actual_norm: f64) -> GradientClipError {
        let d = self.grad_dim as f64;
        
        // 1. Norm computation error
        // ||g|| = √(Σg_i²), error propagates through sum and sqrt
        let sum_error = d * self.epsilon + d * 2.0 * actual_norm * grad_error;
        let norm_error = sum_error / (2.0 * actual_norm.max(1e-10));
        
        // 2. Scaling error (when clipping occurs)
        let scale_error = if actual_norm > self.clip_threshold {
            let scale = self.clip_threshold / actual_norm;
            let scale_derivative = -self.clip_threshold / (actual_norm * actual_norm);
            
            grad_error * scale + actual_norm * norm_error * scale_derivative.abs()
        } else {
            0.0
        };
        
        // 3. Gradient information loss
        let gradient_loss = if actual_norm > self.clip_threshold {
            (actual_norm - self.clip_threshold) / actual_norm
        } else {
            0.0
        };
        
        // Clip probability (assuming Gaussian gradients)
        let clip_probability = self.estimate_clip_probability(actual_norm);
        
        let total_error = norm_error + scale_error;
        
        GradientClipError {
            norm_error,
            scale_error,
            gradient_loss,
            total_error,
            clip_probability,
        }
    }

    /// Estimates probability of clipping based on expected gradient statistics.
    fn estimate_clip_probability(&self, expected_norm: f64) -> f64 {
        // Simplified: assume norm is roughly constant
        if expected_norm > self.clip_threshold {
            1.0
        } else if expected_norm > self.clip_threshold * 0.8 {
            0.5
        } else {
            0.1
        }
    }

    /// Analyzes adaptive gradient clipping (AGC).
    ///
    /// AGC clips gradient relative to parameter norm:
    /// If ||g|| / ||w|| > c: g' = g * c * ||w|| / ||g||
    pub fn adaptive_clipping_error(
        &self,
        grad_error: f64,
        param_error: f64,
        grad_norm: f64,
        param_norm: f64,
    ) -> AdaptiveClipError {
        let ratio = grad_norm / param_norm.max(1e-10);
        
        // Error in ratio computation
        let ratio_error = (grad_error * param_norm + grad_norm * param_error) 
            / (param_norm * param_norm).max(1e-10);
        
        // Threshold comparison error
        let threshold_error = ratio_error;
        
        // Scaling error when clipping
        let scale_error = if ratio > self.clip_threshold {
            let scale = self.clip_threshold * param_norm / grad_norm;
            grad_error * scale + param_error * self.clip_threshold / grad_norm
        } else {
            0.0
        };
        
        AdaptiveClipError {
            ratio_error,
            threshold_error,
            scale_error,
            total_error: threshold_error + scale_error,
        }
    }

    /// Computes error for layer-wise clipping.
    pub fn layerwise_clipping_error(
        &self,
        num_layers: usize,
        per_layer_grad_error: f64,
    ) -> LayerwiseClipError {
        // Each layer clipped independently
        let per_layer_error = self.norm_clipping_error(
            per_layer_grad_error,
            self.expected_grad_magnitude,
        );
        
        // Total error across layers
        let total_error = per_layer_error.total_error * num_layers as f64;
        
        // Variance in clipping behavior across layers
        let variance = per_layer_error.total_error.powi(2) * num_layers as f64;
        
        LayerwiseClipError {
            num_layers,
            per_layer_error: per_layer_error.total_error,
            total_error,
            variance,
        }
    }
}

/// Error from value clipping.
#[derive(Debug, Clone)]
pub struct ValueClipError {
    /// Error at clipping boundary.
    pub boundary_error: f64,
    /// Information loss ratio.
    pub info_loss: f64,
    /// Total error.
    pub total_error: f64,
}

/// Error from adaptive gradient clipping.
#[derive(Debug, Clone)]
pub struct AdaptiveClipError {
    /// Error in gradient/parameter ratio.
    pub ratio_error: f64,
    /// Error in threshold comparison.
    pub threshold_error: f64,
    /// Error from scaling.
    pub scale_error: f64,
    /// Total error.
    pub total_error: f64,
}

/// Error from layer-wise clipping.
#[derive(Debug, Clone)]
pub struct LayerwiseClipError {
    /// Number of layers.
    pub num_layers: usize,
    /// Error per layer.
    pub per_layer_error: f64,
    /// Total error across layers.
    pub total_error: f64,
    /// Variance in clipping.
    pub variance: f64,
}

/// Compares different clipping strategies.
pub fn compare_clipping_strategies(
    grad_norm: f64,
    param_norm: f64,
    clip_threshold: f64,
    grad_error: f64,
) -> ClippingComparison {
    let bounds = ClippingBounds::new(clip_threshold, 1000);
    
    let global_error = bounds.norm_clipping_error(grad_error, grad_norm);
    let adaptive_error = bounds.adaptive_clipping_error(
        grad_error, 
        grad_error, // Assume similar param error
        grad_norm,
        param_norm,
    );
    
    let recommendation = if adaptive_error.total_error < global_error.total_error * 0.8 {
        ClippingStrategy::Adaptive
    } else if global_error.gradient_loss > 0.5 {
        ClippingStrategy::ValueClipping
    } else {
        ClippingStrategy::GlobalNorm
    };
    
    ClippingComparison {
        global_norm_error: global_error.total_error,
        adaptive_error: adaptive_error.total_error,
        gradient_loss: global_error.gradient_loss,
        recommendation,
    }
}

/// Comparison of clipping strategies.
#[derive(Debug, Clone)]
pub struct ClippingComparison {
    /// Error from global norm clipping.
    pub global_norm_error: f64,
    /// Error from adaptive clipping.
    pub adaptive_error: f64,
    /// Gradient information loss.
    pub gradient_loss: f64,
    /// Recommended strategy.
    pub recommendation: ClippingStrategy,
}

/// Clipping strategy options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClippingStrategy {
    GlobalNorm,
    Adaptive,
    ValueClipping,
    NoClipping,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_clipping_error() {
        let bounds = ClippingBounds::new(1.0, 1000);
        
        // Within threshold
        let no_clip = bounds.norm_clipping_error(1e-7, 0.5);
        assert_eq!(no_clip.gradient_loss, 0.0);
        
        // Exceeds threshold
        let clipped = bounds.norm_clipping_error(1e-7, 2.0);
        assert!(clipped.gradient_loss > 0.0);
    }

    #[test]
    fn test_value_clipping() {
        let bounds = ClippingBounds::new(1.0, 1000);
        let error = bounds.value_clipping_error(1e-7);
        
        assert!(error.total_error > 0.0);
    }

    #[test]
    fn test_adaptive_clipping() {
        let bounds = ClippingBounds::new(0.1, 1000);
        
        let error = bounds.adaptive_clipping_error(1e-7, 1e-7, 1.0, 10.0);
        assert!(error.total_error > 0.0);
    }

    #[test]
    fn test_compare_strategies() {
        let comparison = compare_clipping_strategies(2.0, 10.0, 1.0, 1e-7);
        
        assert!(comparison.global_norm_error > 0.0);
        assert!(comparison.gradient_loss > 0.0);
    }
}
