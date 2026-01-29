//! Residual Connection Error Bounds.
//!
//! Analyzes error propagation through skip connections,
//! which are critical for deep network training.

/// Error bounds for residual connections.
#[derive(Debug, Clone)]
pub struct ResidualBounds {
    /// Expected residual magnitude (relative to skip).
    pub residual_scale: f64,
    /// Machine epsilon.
    pub epsilon: f64,
}

impl Default for ResidualBounds {
    fn default() -> Self {
        Self {
            residual_scale: 1.0,
            epsilon: f64::EPSILON,
        }
    }
}

/// Error in skip connection computation.
#[derive(Debug, Clone)]
pub struct SkipConnectionError {
    /// Error from addition.
    pub addition_error: f64,
    /// Error propagated from residual branch.
    pub residual_error: f64,
    /// Error propagated from skip branch.
    pub skip_error: f64,
    /// Total error.
    pub total_error: f64,
    /// Error amplification factor.
    pub amplification: f64,
}

impl ResidualBounds {
    /// Creates new residual bounds.
    pub fn new(residual_scale: f64) -> Self {
        Self {
            residual_scale,
            epsilon: f64::EPSILON,
        }
    }

    /// Computes error bounds for residual addition.
    ///
    /// y = x + F(x)  (pre-norm style)
    /// or
    /// y = x + F(LayerNorm(x))  (post-norm style)
    pub fn forward_error(
        &self,
        skip_error: f64,
        residual_error: f64,
        skip_magnitude: f64,
        residual_magnitude: f64,
    ) -> SkipConnectionError {
        // Addition error: O(ε * max(|a|, |b|))
        let max_mag = skip_magnitude.max(residual_magnitude);
        let addition_error = self.epsilon * max_mag;
        
        // Total error is sum of component errors plus addition error
        let total_error = skip_error + residual_error + addition_error;
        
        // Error amplification: how much does error grow through residual?
        // In residual networks, error from early layers adds to later layers
        let combined_magnitude = skip_magnitude + residual_magnitude;
        let amplification = if skip_magnitude > 0.0 {
            combined_magnitude / skip_magnitude
        } else {
            1.0
        };
        
        SkipConnectionError {
            addition_error,
            residual_error,
            skip_error,
            total_error,
            amplification,
        }
    }

    /// Computes backward error propagation.
    ///
    /// For y = x + F(x):
    /// ∂L/∂x = ∂L/∂y + ∂L/∂y * ∂F/∂x
    pub fn backward_error(
        &self,
        output_grad_error: f64,
        residual_grad_error: f64,
    ) -> ResidualGradError {
        // Skip path: gradient passes through unchanged
        let skip_grad_error = output_grad_error;
        
        // Residual path: gradient propagates through F
        let residual_path_error = residual_grad_error;
        
        // Combined (addition of gradients)
        let total_error = skip_grad_error + residual_path_error + self.epsilon;
        
        ResidualGradError {
            skip_grad_error,
            residual_path_error,
            total_error,
        }
    }

    /// Analyzes error accumulation through multiple residual blocks.
    pub fn multi_block_error(&self, num_blocks: usize, per_block_error: f64) -> MultiBlockError {
        // In residual networks, errors accumulate additively
        // Total error ≈ O(num_blocks * per_block_error)
        
        // But gradient signal can also strengthen
        // which helps training but doesn't reduce error
        
        let accumulated_error = num_blocks as f64 * per_block_error;
        
        // Error growth rate
        let growth_rate = if num_blocks > 1 {
            accumulated_error / per_block_error
        } else {
            1.0
        };
        
        // Stability assessment
        let stability = if growth_rate < num_blocks as f64 * 0.5 {
            ResidualStability::Stable
        } else if growth_rate < num_blocks as f64 * 2.0 {
            ResidualStability::Marginal
        } else {
            ResidualStability::Unstable
        };
        
        MultiBlockError {
            num_blocks,
            per_block_error,
            accumulated_error,
            growth_rate,
            stability,
        }
    }
}

/// Gradient error for residual backward pass.
#[derive(Debug, Clone)]
pub struct ResidualGradError {
    /// Error in skip gradient.
    pub skip_grad_error: f64,
    /// Error from residual branch gradient.
    pub residual_path_error: f64,
    /// Total gradient error.
    pub total_error: f64,
}

/// Error analysis for multiple residual blocks.
#[derive(Debug, Clone)]
pub struct MultiBlockError {
    /// Number of blocks.
    pub num_blocks: usize,
    /// Error per block.
    pub per_block_error: f64,
    /// Total accumulated error.
    pub accumulated_error: f64,
    /// Error growth rate.
    pub growth_rate: f64,
    /// Stability assessment.
    pub stability: ResidualStability,
}

/// Residual network stability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResidualStability {
    Stable,
    Marginal,
    Unstable,
}

/// Pre-norm vs post-norm residual error comparison.
#[derive(Debug, Clone)]
pub struct NormPlacementComparison {
    /// Pre-norm error (norm before residual).
    pub pre_norm_error: f64,
    /// Post-norm error (norm after residual).
    pub post_norm_error: f64,
    /// Recommendation.
    pub recommendation: NormPlacement,
}

/// Norm placement options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormPlacement {
    PreNorm,
    PostNorm,
}

/// Compares pre-norm vs post-norm error characteristics.
pub fn compare_norm_placement(
    residual_error: f64,
    norm_error: f64,
    num_blocks: usize,
) -> NormPlacementComparison {
    // Pre-norm: y = x + F(Norm(x))
    // Norm error absorbed by F, residual adds to normalized
    let pre_norm_per_block = residual_error + norm_error;
    let pre_norm_total = pre_norm_per_block * num_blocks as f64;
    
    // Post-norm: y = Norm(x + F(x))
    // Norm applied after addition, can amplify accumulated error
    let post_norm_per_block = residual_error;
    let post_norm_accumulated = post_norm_per_block * num_blocks as f64;
    let post_norm_total = post_norm_accumulated + norm_error * post_norm_accumulated.log2().max(1.0);
    
    let recommendation = if pre_norm_total < post_norm_total {
        NormPlacement::PreNorm
    } else {
        NormPlacement::PostNorm
    };
    
    NormPlacementComparison {
        pre_norm_error: pre_norm_total,
        post_norm_error: post_norm_total,
        recommendation,
    }
}

/// Dense residual (DenseNet-style) error analysis.
pub fn dense_residual_error(num_layers: usize, per_layer_error: f64) -> f64 {
    // In DenseNet, each layer receives features from all previous layers
    // Error accumulation: O(n² * per_layer_error)
    let n = num_layers as f64;
    n * (n + 1.0) / 2.0 * per_layer_error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_bounds() {
        let bounds = ResidualBounds::default();
        let error = bounds.forward_error(1e-7, 1e-6, 1.0, 0.1);
        
        assert!(error.total_error > 0.0);
        assert!(error.total_error >= error.skip_error);
    }

    #[test]
    fn test_multi_block_error() {
        let bounds = ResidualBounds::default();
        let analysis = bounds.multi_block_error(24, 1e-6);
        
        assert_eq!(analysis.num_blocks, 24);
        assert!(analysis.accumulated_error > analysis.per_block_error);
    }

    #[test]
    fn test_norm_placement() {
        let comparison = compare_norm_placement(1e-6, 1e-7, 12);
        
        // Pre-norm is generally more stable for deep networks
        assert!(matches!(comparison.recommendation, NormPlacement::PreNorm | NormPlacement::PostNorm));
    }

    #[test]
    fn test_dense_residual() {
        let error_12 = dense_residual_error(12, 1e-6);
        let error_6 = dense_residual_error(6, 1e-6);
        
        // More layers = more error accumulation (quadratic)
        assert!(error_12 > error_6 * 2.0);
    }
}
